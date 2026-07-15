"""
RuleEngine — 确定性规则执行器。

职责：
1. 加载编译后的 RuleBank（RuleBank 对象 + encoding_map）
2. 批量评估所有 active 规则
3. 聚合 rule_score / rule_confidence
4. 输出可直接送入 MPFCDecisionFusion 的评分向量

纯 PyTorch 运算，支持 GPU batch 执行。
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch

from methods.modules.rule_bank import RuleBank

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    MAX = "max"
    NOISY_OR = "noisy_or"
    WEIGHTED = "weighted"


class RuleEngine:
    """确定性规则执行器。

    Parameters
    ----------
    rulebank:
        已加载的 RuleBank 实例（仅 active_rules 被使用）。
    encoding_map:
        分类字段的字符串→编码 ID 映射。用于 Type / Target 匹配。
        {field_name: {str_value: int_id}}
    aggregation:
        评分聚合策略。
    device:
        PyTorch 设备。
    """

    def __init__(
        self,
        rulebank: RuleBank,
        encoding_map: Optional[Dict[str, Dict[str, int]]] = None,
        aggregation: Union[str, AggregationStrategy] = AggregationStrategy.NOISY_OR,
        device: torch.device = torch.device("cpu"),
    ):
        self._rules = rulebank.active_rules()
        self._encoding_map = encoding_map or {}
        self._aggregation = (AggregationStrategy(aggregation)
                             if isinstance(aggregation, str) else aggregation)
        self.device = device

        if not self._rules:
            logger.warning("RuleEngine initialised with zero active rules")
            return

        # 编译规则 → (field, operator, value, field_str?) 元组列表
        self._compiled: List[dict] = []
        for r in self._rules:
            for c in r.clauses:
                field = str(c.get("field", ""))
                op = str(c.get("operator", ""))
                raw_val = c.get("value")
                # 如果 value_ref 存在且 value 未设置，编译时写入 0（运行时用已解析值覆盖）
                val = raw_val if raw_val is not None else 0.0
                if field in ("Type", "Target") and isinstance(val, str):
                    self._compiled.append({
                        "rule_idx": len(self._compiled),
                        "rule_id": r.rule_id,
                        "risk_score": r.risk_score,
                        "base_confidence": r.base_confidence,
                        "field": field,
                        "operator": op,
                        "value_str": val,      # 原始字符串，运行时查 encoding_map
                        "value_num": None,     # 运行时查询后填充
                        "is_string": True,
                    })
                else:
                    self._compiled.append({
                        "rule_idx": len(self._compiled),
                        "rule_id": r.rule_id,
                        "risk_score": r.risk_score,
                        "base_confidence": r.base_confidence,
                        "field": field,
                        "operator": op,
                        "value_str": None,
                        "value_num": float(val) if val is not None else 0.0,
                        "is_string": False,
                    })

        # 预测值解析：将 string 条件预编码
        self._resolve_string_values()

        # 将编译后的规则按 rule_id 分组，便于聚合
        self._rule_boundaries: Dict[str, List[int]] = {}
        for idx, c in enumerate(self._compiled):
            self._rule_boundaries.setdefault(c["rule_id"], []).append(idx)

        self._risk_scores = torch.tensor(
            [r.risk_score for r in self._rules],
            dtype=torch.float32, device=self.device,
        )
        self._base_confidences = torch.tensor(
            [r.base_confidence for r in self._rules],
            dtype=torch.float32, device=self.device,
        )

    # ── 公开接口 ────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        batch: Dict[str, torch.Tensor],
        _cpu: bool = True,  # avoid NVRTC JIT on new GPU archs (PyTorch 1.12 compat)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._rules:
            b = self._batch_size(batch)
            return (torch.zeros(b, 1, device=self.device),
                    torch.zeros(b, 1, device=self.device))

        # Work on CPU to avoid NVRTC JIT compilation issues with torch.prod
        # on newer GPU architectures (Ada Lovelace / Hopper) under PyTorch 1.12.
        batch = {k: v.cpu() for k, v in batch.items()} if _cpu else batch
        n = self._batch_size(batch)
        hit_mat = torch.ones(n, len(self._rules), dtype=torch.bool)

        for ridx in range(len(self._rules)):
            clause_indices = self._rule_boundaries.get(
                self._rules[ridx].rule_id, []
            )
            if not clause_indices:
                hit_mat[:, ridx] = False
                continue
            op = self._rules[ridx].conditions_operator
            clause_results = []
            for ci in clause_indices:
                clause_results.append(
                    self._eval_clause(ci, batch, n)
                )
            if op == "AND":
                combined = torch.stack(clause_results, dim=1).all(dim=1)
            else:
                combined = torch.stack(clause_results, dim=1).any(dim=1)
            hit_mat[:, ridx] = combined

        rule_score = self._aggregate_scores(hit_mat)
        rule_confidence = self._aggregate_confidence(hit_mat)
        if _cpu and self.device.type != "cpu":
            rule_score = rule_score.to(self.device)
            rule_confidence = rule_confidence.to(self.device)
        return rule_score, rule_confidence

    # ── 内部方法 ────────────────────────────────────────────────────────

    def _eval_clause(
        self,
        ci: int,
        batch: Dict[str, torch.Tensor],
        n: int,
    ) -> torch.Tensor:
        c = self._compiled[ci]
        field = c["field"]
        if field not in batch:
            return torch.zeros(n, dtype=torch.bool, device=self.device)
        x = batch[field].view(-1)

        if c["is_string"]:
            val = c.get("value_num")
            if val is None:
                return torch.zeros(n, dtype=torch.bool, device=self.device)
        else:
            val = c["value_num"]

        op = c["operator"]
        if op == ">":
            return x > val
        elif op == "<":
            return x < val
        elif op == ">=":
            return x >= val
        elif op == "<=":
            return x <= val
        elif op == "==":
            return x == val
        elif op == "between":
            if isinstance(val, (list, tuple)) and len(val) == 2:
                return (x >= val[0]) & (x <= val[1])
            return x == val  # fallback
        elif op == "in":
            if not isinstance(val, (list, tuple)):
                val = [val]
            result = torch.zeros(n, dtype=torch.bool, device=self.device)
            for v in val:
                result = result | (x == float(v) if not c["is_string"] else (x == v))
            return result
        elif op == "not_in":
            if not isinstance(val, (list, tuple)):
                val = [val]
            result = torch.ones(n, dtype=torch.bool, device=self.device)
            for v in val:
                result = result & (x != (float(v) if not c["is_string"] else v))
            return result
        else:
            logger.warning("Unknown operator %s, returning False", op)
            return torch.zeros(n, dtype=torch.bool, device=self.device)

    def _aggregate_scores(self, hit_mat: torch.Tensor) -> torch.Tensor:
        """hit_mat: [batch, n_rules] bool → [batch, 1]"""
        hit_float = hit_mat.float() * self._risk_scores.unsqueeze(0)  # [b, n]
        if self._aggregation == AggregationStrategy.MAX:
            val, _ = hit_float.max(dim=1, keepdim=True)
            return val
        elif self._aggregation == AggregationStrategy.NOISY_OR:
            # 1 - prod(1 - hit_float)
            safe = torch.clamp(1.0 - hit_float, min=0.0, max=1.0)
            return 1.0 - safe.prod(dim=1, keepdim=True)
        elif self._aggregation == AggregationStrategy.WEIGHTED:
            weights = self._base_confidences.unsqueeze(0)  # [1, n]
            numer = (hit_float * weights).sum(dim=1, keepdim=True)
            denom = (hit_mat.float() * weights).sum(dim=1, keepdim=True).clamp_min(1e-8)
            return numer / denom
        else:
            return torch.zeros(hit_mat.size(0), 1, device=self.device)

    def _aggregate_confidence(self, hit_mat: torch.Tensor) -> torch.Tensor:
        """hit_mat: [batch, n_rules] bool → [batch, 1]"""
        n_rules = max(len(self._rules), 1)
        hit_count = hit_mat.float().sum(dim=1, keepdim=True)       # [b, 1]
        avg_conf = (hit_mat.float() * self._base_confidences.unsqueeze(0)
                    ).sum(dim=1, keepdim=True) / hit_count.clamp_min(1)
        hit_frac = hit_count / n_rules
        return avg_conf * (1.0 + hit_frac) / 2.0

    # ── 辅助 ────────────────────────────────────────────────────────────

    def _resolve_string_values(self) -> None:
        for c in self._compiled:
            if not c["is_string"] or c["value_str"] is None:
                continue
            field = c["field"]
            emap = self._encoding_map.get(field, {})
            raw = c["value_str"]
            if raw in emap:
                c["value_num"] = emap[raw]
            else:
                logger.warning(
                    "Rule %s: string '%s' not in encoding_map[%s]",
                    c["rule_id"], raw, field,
                )
                c["value_num"] = None  # 无法匹配

    def _batch_size(self, batch: Dict[str, torch.Tensor]) -> int:
        for v in batch.values():
            return v.size(0)
        return 0
