"""
VTA (Ventral Tegmental Area) — 全局学习调节模块。

类脑映射：VTA 多巴胺能神经元编码奖励预测误差（RPE），调节全脑学习强度。
在本模型中，VTA 位于 MPFC 输出之后、loss 计算之前，根据预测误差、
规则—神经分支冲突和 CA3 原型不确定性，产生三条反馈路径：

1. sample_weight → 调节主任务 BCE（经由 main_loss.backward() 更新主干）
2. gate_bias_offset → 延迟调节下一 batch 的 MPFC fusion gate
3. ca3_learning_signal → 调节当前 batch 的 CA3 contrastive loss

设计原则：
- 所有反馈路径必须 detach，避免循环梯度
- 推理阶段 label=None，prediction_error=0，不更新状态
- use_vta=False 时退化为恒等映射
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class VTAInput:
    """VTA 前向输入（所有张量已存在于 forward_batch 中）。"""
    final_logit: torch.Tensor       # [batch, 1]  MPFC 融合后 logit
    final_prob: torch.Tensor        # [batch, 1]  sigmoid(final_logit)
    neural_score: torch.Tensor      # [batch, 1]  纯神经分支风险概率
    rule_score: torch.Tensor        # [batch, 1]  规则分支风险概率
    fusion_weight: torch.Tensor     # [batch, 1]  MPFC 融合权重
    ca3_entropy: torch.Tensor       # [batch,]    CA3 原型分配熵（未归一化）
    label: Optional[torch.Tensor]   # [batch,]    训练时传入，推理时 None


@dataclass
class VTAOutput:
    # Surprise 组件
    surprise: torch.Tensor           # [batch,]    完整 surprise S_i
    prediction_error: torch.Tensor   # [batch,]    代价敏感预测误差 E_i
    conflict: torch.Tensor           # [batch,]    规则—神经分支冲突 C_i
    entropy: torch.Tensor            # [batch,]    归一化 CA3 熵 H_i

    # 反馈路径一：主任务重加权
    sample_weight: torch.Tensor      # [batch,]    用于 BCE 逐样本加权

    # 反馈路径二：MPFC gate 延迟调节
    dopamine_state: torch.Tensor     # scalar       EMA 累计 surprise 强度
    gate_direction_state: torch.Tensor  # scalar    EMA 累计分支误差差
    gate_bias_offset_next: torch.Tensor  # scalar   下一 batch 的 gate 偏移量

    # 反馈路径三：CA3 学习调节
    ca3_learning_signal: torch.Tensor  # scalar     CA3 contrastive 调节系数

    # 推理辅助
    uncertainty: torch.Tensor        # [batch,]    推理代理不确定性
    review_score: torch.Tensor       # [batch,]    人工复核建议分数


# ─────────────────────────────────────────────────────────────────────────────
# VTA 模块
# ─────────────────────────────────────────────────────────────────────────────


class VTAModule(nn.Module):
    """全局学习调节模块（VTA / 多巴胺 RPE 信号）。

    Parameters
    ----------
    use_vta:
        False 时所有 forward 返回 identity 值，不影响训练。
    num_prototypes:
        用于 CA3 熵归一化（除以 ln(num_prototypes)）。
    lambda_fn:
        Surprise 公式中 FN（漏报）代价权重。
    lambda_fp:
        Surprise 公式中 FP（误报）代价权重。
    prediction_weight:
        Surprise 三成分中预测误差权重 w_pred。
    conflict_weight:
        Surprise 三成分中分支冲突权重 w_conflict。
    entropy_weight:
        Surprise 三成分中 CA3 熵权重 w_entropy。
    reweight_strength:
        sample_weight = 1 + strength * surprise。
    reweight_max:
        sample_weight 上限。
    ema_decay:
        EMA 状态更新系数 ρ。
    state_temperature:
        dopamine_state → tanh(·/T) 的温度参数。
    gate_strength:
        最终 gate_bias_offset 的缩放系数。
    gate_bias_max:
        gate_bias_offset 绝对值上限。
    ca3_strength:
        CA3 learning signal 中 κ 系数。
    ca3_scale_min:
        CA3 learning signal 下限。
    ca3_scale_max:
        CA3 learning signal 上限。
    """
    def __init__(
        self,
        *,
        use_vta: bool = False,
        num_prototypes: int = 16,
        lambda_fn: float = 5.0,
        lambda_fp: float = 1.0,
        prediction_weight: float = 1.0,
        conflict_weight: float = 0.3,
        entropy_weight: float = 0.3,
        reweight_strength: float = 1.0,
        reweight_max: float = 8.0,
        ema_decay: float = 0.9,
        state_temperature: float = 1.0,
        gate_strength: float = 0.1,
        gate_bias_max: float = 0.5,
        ca3_strength: float = 0.5,
        ca3_scale_min: float = 0.5,
        ca3_scale_max: float = 2.0,
    ):
        super().__init__()
        self.use_vta = bool(use_vta)
        self._ln_prototypes = math.log(max(num_prototypes, 2))
        self.lambda_fn = float(lambda_fn)
        self.lambda_fp = float(lambda_fp)
        self.w_pred = float(prediction_weight)
        self.w_conflict = float(conflict_weight)
        self.w_entropy = float(entropy_weight)
        self.reweight_strength = float(reweight_strength)
        self.reweight_max = float(reweight_max)
        self.ema_decay = float(ema_decay)
        self.state_temperature = float(state_temperature)
        self.gate_strength = float(gate_strength)
        self.gate_bias_max = float(gate_bias_max)
        self.ca3_strength = float(ca3_strength)
        self.ca3_scale_min = float(ca3_scale_min)
        self.ca3_scale_max = float(ca3_scale_max)

        # 持久化状态（随 checkpoint 保存）
        self.register_buffer("_dopamine_state", torch.tensor(0.0))
        self.register_buffer("_gate_direction_state", torch.tensor(0.0))

    # ── 公开接口 ────────────────────────────────────────────────────────

    def forward(
        self,
        inp: VTAInput,
        mode: str = "train",
    ) -> VTAOutput:
        """计算 VTA 信号。

        Parameters
        ----------
        inp:
            包含 MPFC 输出、CA3 熵、标签的输入结构体。
        mode:
            "train" — 使用 label 计算完整 surprise，更新状态。
            "eval"  — 无 label，计算代理信号，不更新状态。

        Returns
        -------
        VTAOutput 包含三条反馈路径和辅助信号。
        """
        if not self.use_vta:
            return self._identity(inp)

        is_train = mode == "train"
        p = inp.final_prob.detach().squeeze(-1)           # [batch]
        p_n = inp.neural_score.detach().squeeze(-1)        # [batch]
        p_r = inp.rule_score.detach().squeeze(-1)          # [batch]

        # ── 归一化 CA3 熵 → [0, 1] ──────────────────────────────────
        h = inp.ca3_entropy.detach() / self._ln_prototypes  # [batch]
        h = h.clamp(0.0, 1.0)

        # ── Surprise 三成分 ──────────────────────────────────────────
        if is_train and inp.label is not None:
            y = inp.label.detach().float()                   # [batch]
            # E_i: 代价敏感预测误差（基于 final_prob）
            e = self.lambda_fn * y * (1.0 - p) + self.lambda_fp * (1.0 - y) * p
            # C_i: 规则—神经分支冲突
            c = (p_r - p_n).abs()
            # 分支级误差（用于 gate direction）
            e_n = self.lambda_fn * y * (1.0 - p_n) + self.lambda_fp * (1.0 - y) * p_n
            e_r = self.lambda_fn * y * (1.0 - p_r) + self.lambda_fp * (1.0 - y) * p_r
            gate_delta = (e_n - e_r).mean()  # >0 → 神经更差 → 增加规则权重
        else:
            e = torch.zeros_like(p)
            c = (p_r - p_n).abs()
            e_n = torch.zeros_like(p)
            e_r = torch.zeros_like(p)
            gate_delta = torch.tensor(0.0, device=p.device)

        surprise = self.w_pred * e + self.w_conflict * c + self.w_entropy * h  # [batch]

        # ── 反馈路径一：主任务逐样本重加权 ────────────────────────────
        sample_weight = torch.clamp(
            1.0 + self.reweight_strength * surprise.detach(),
            min=1.0,
            max=self.reweight_max,
        )

        # ── 反馈路径三：CA3 学习调节（仅使用 E_i） ──────────────────
        # M_i = (1 - H_i) * (1 + kappa * E_i)
        ca3_signal_per_sample = (1.0 - h) * (1.0 + self.ca3_strength * e.detach())
        ca3_signal = ca3_signal_per_sample.mean().clamp(
            self.ca3_scale_min, self.ca3_scale_max,
        )

        # ── 更新延迟状态（仅 train 模式） ────────────────────────────
        if is_train and inp.label is not None:
            with torch.no_grad():
                batch_surprise = surprise.detach().mean()
                self._dopamine_state.mul_(self.ema_decay).add_(
                    (1.0 - self.ema_decay) * batch_surprise)
                self._gate_direction_state.mul_(self.ema_decay).add_(
                    (1.0 - self.ema_decay) * gate_delta.detach())

        # ── 反馈路径二：下一 batch 的 MPFC gate 偏移量 ──────────────
        dopamine_tanh = torch.tanh(
            self._dopamine_state / self.state_temperature
        )
        direction_tanh = torch.tanh(self._gate_direction_state)
        gate_bias_next = (
            self.gate_strength * dopamine_tanh * direction_tanh
        ).clamp(-self.gate_bias_max, self.gate_bias_max).detach()

        # ── 推理辅助 ──────────────────────────────────────────────────
        boundary_proximity = (p - 0.5).abs()  # 越小越靠近边界
        uncertainty = (
            0.4 * h + 0.3 * c + 0.3 * (1.0 - 2.0 * boundary_proximity)
        )
        review_score = torch.sigmoid(uncertainty - 0.5)

        return VTAOutput(
            surprise=surprise,
            prediction_error=e,
            conflict=c,
            entropy=h,
            sample_weight=sample_weight,
            dopamine_state=self._dopamine_state.detach(),
            gate_direction_state=self._gate_direction_state.detach(),
            gate_bias_offset_next=gate_bias_next,
            ca3_learning_signal=ca3_signal.detach(),
            uncertainty=uncertainty,
            review_score=review_score,
        )

    # ── 恒等退化 ───────────────────────────────────────────────────────

    def _identity(self, inp: VTAInput) -> VTAOutput:
        """use_vta=False 时返回 identity 值，完全不影响训练。"""
        b = inp.final_prob.shape[0]
        dev = inp.final_prob.device
        zeros_b = torch.zeros(b, device=dev)
        return VTAOutput(
            surprise=zeros_b,
            prediction_error=zeros_b,
            conflict=zeros_b,
            entropy=zeros_b,
            sample_weight=torch.ones(b, device=dev),
            dopamine_state=torch.tensor(0.0, device=dev),
            gate_direction_state=torch.tensor(0.0, device=dev),
            gate_bias_offset_next=torch.tensor(0.0, device=dev),
            ca3_learning_signal=torch.tensor(1.0, device=dev),
            uncertainty=zeros_b,
            review_score=torch.zeros(b, device=dev),
        )

    # ── 状态访问器 ─────────────────────────────────────────────────────

    def get_gate_bias_offset(self) -> torch.Tensor:
        """返回供 MPFC forward 使用的 gate_bias_offset（标量）。"""
        if not self.use_vta:
            return torch.tensor(0.0, device=self._dopamine_state.device)
        dopamine_tanh = torch.tanh(
            self._dopamine_state / self.state_temperature
        )
        direction_tanh = torch.tanh(self._gate_direction_state)
        return (
            self.gate_strength * dopamine_tanh * direction_tanh
        ).clamp(-self.gate_bias_max, self.gate_bias_max).detach()

    @torch.no_grad()
    def reset_state(self) -> None:
        """重置所有 VTA 状态（用于训练重启）。"""
        self._dopamine_state.zero_()
        self._gate_direction_state.zero_()
