import math
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CA3Output(NamedTuple):
    enhanced_embedding: torch.Tensor
    memory_context: torch.Tensor
    similarity: torch.Tensor
    assignment_weights: torch.Tensor
    top_proto_id: torch.Tensor
    top_proto_sim: torch.Tensor
    assignment_entropy: torch.Tensor
    gate_probs: torch.Tensor


class CA3PrototypeMemory(nn.Module):
    """Deployable risk-pattern memory; no labels or group IDs enter forward()."""

    def __init__(self, embedding_dim=64, num_prototypes=16, temperature=0.2,
                 top_k=3, fusion="gated_residual",
                 gate_bias_init=-2.0,
                 gate_bias_final=-1.0,
                 anneal_epochs=10,
                 dead_epoch_threshold=3,
                 entropy_gate_beta=1.0,
                 contrastive_temperature=0.1,
                 diversity_margin=0.5):
        super().__init__()
        if num_prototypes < 1:
            raise ValueError("num_prototypes must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if fusion != "gated_residual":
            raise ValueError("CA3 v1 supports only gated_residual fusion")
        self.embedding_dim = int(embedding_dim)
        self.num_prototypes = int(num_prototypes)
        self.temperature = float(temperature)
        self.top_k = min(max(int(top_k), 1), self.num_prototypes)
        self.anneal_epochs = int(anneal_epochs)
        self.dead_epoch_threshold = int(dead_epoch_threshold)
        self.entropy_gate_beta = float(entropy_gate_beta)
        self.contrastive_temperature = float(contrastive_temperature)
        self.diversity_margin = float(diversity_margin)

        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embedding_dim) * 0.02)
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.memory_proj = nn.Linear(embedding_dim, embedding_dim)
        self.gate_mlp = nn.Linear(embedding_dim * 2, 1)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # C2: 退火状态
        self.register_buffer("_anneal_progress", torch.tensor(0.0))
        self.register_buffer("_gate_bias_init", torch.tensor(float(gate_bias_init)))
        self.register_buffer("_gate_bias_final", torch.tensor(float(gate_bias_final)))
        nn.init.constant_(self.gate_mlp.bias, gate_bias_init)

        # A1: 死原型跟踪
        self.register_buffer("_usage_this_epoch",
                             torch.zeros(num_prototypes, dtype=torch.long))
        self.register_buffer("_dead_epoch_count",
                             torch.zeros(num_prototypes, dtype=torch.long))

        nn.init.eye_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.eye_(self.memory_proj.weight)
        nn.init.zeros_(self.memory_proj.bias)
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("init_prototypes", torch.zeros(num_prototypes, embedding_dim))

    # ── C2: 融合退火 ──────────────────────────────────────────────

    @torch.no_grad()
    def set_anneal_progress(self, progress: float) -> None:
        """线性插值 gate_mlp.bias 从 _gate_bias_init 到 _gate_bias_final。"""
        p = max(0.0, min(1.0, float(progress)))
        bias = (self._gate_bias_init.item()
                + p * (self._gate_bias_final.item() - self._gate_bias_init.item()))
        self.gate_mlp.bias.fill_(bias)

    # ── A1: 死原型检测与刷新 ──────────────────────────────────────

    @torch.no_grad()
    def on_epoch_end(self) -> None:
        """epoch 结束时更新死亡计数并重置使用统计。"""
        if self.training:
            dead = (self._usage_this_epoch == 0)
            self._dead_epoch_count = torch.where(
                dead,
                self._dead_epoch_count + 1,
                torch.zeros_like(self._dead_epoch_count))
        self._usage_this_epoch.zero_()

    @torch.no_grad()
    def refresh_dead_prototypes(self, seed_embeddings: torch.Tensor) -> int:
        """替换连续 dead_epoch_threshold 个 epoch 未使用的原型。返回刷新数。"""
        dead_mask = self._dead_epoch_count >= self.dead_epoch_threshold
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        if len(dead_indices) == 0:
            return 0
        n_dead = len(dead_indices)
        n_seed = len(seed_embeddings)
        indices = torch.randint(0, n_seed, (n_dead,), device=seed_embeddings.device)
        choices = F.normalize(seed_embeddings[indices], dim=-1)
        self.prototypes[dead_indices] = choices
        self.init_prototypes[dead_indices] = choices
        self._dead_epoch_count[dead_indices] = 0
        return n_dead

    # ── B1: 对比原型学习 ──────────────────────────────────────────

    def contrastive_loss(self, embedding: torch.Tensor) -> torch.Tensor:
        """InfoNCE 损失：拉近 embedding≡最近原型，推远其他原型。"""
        query = F.normalize(embedding, dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)
        logits = query @ proto.t() / self.contrastive_temperature
        with torch.no_grad():
            targets = logits.softmax(dim=-1).argmax(dim=-1)
        return F.cross_entropy(logits, targets)

    # ── 初始化 ────────────────────────────────────────────────────

    @torch.no_grad()
    def initialize_prototypes(self, centers):
        centers = torch.as_tensor(centers, dtype=self.prototypes.dtype,
                                  device=self.prototypes.device)
        if centers.shape != self.prototypes.shape:
            raise ValueError(
                f"Expected prototype centers {tuple(self.prototypes.shape)}, got {tuple(centers.shape)}")
        centers = F.normalize(centers, dim=-1)
        self.prototypes.copy_(centers)
        self.init_prototypes.copy_(centers)
        self.initialized.fill_(True)

    # ── 前向传播 ──────────────────────────────────────────────────

    def forward(self, embedding, enabled=True):
        n = embedding.shape[0]
        if not enabled:
            zeros_h = torch.zeros_like(embedding)
            zeros_m = embedding.new_zeros((n, self.num_prototypes))
            zeros_n = embedding.new_zeros(n)
            return CA3Output(
                embedding, zeros_h, zeros_m, zeros_m,
                torch.zeros(n, dtype=torch.long, device=embedding.device),
                zeros_n, zeros_n, zeros_h.new_zeros((n, 1)),
            )
        if not bool(self.initialized.item()):
            raise RuntimeError("CA3 cannot be enabled before prototype initialization")

        query = F.normalize(self.query_proj(embedding), dim=-1)
        prototype_norm = F.normalize(self.prototypes, dim=-1)
        similarity = query @ prototype_norm.t()
        assignment = F.softmax(similarity / self.temperature, dim=-1)
        context = assignment @ self.prototypes
        entropy = -(assignment * assignment.clamp_min(1e-12).log()).sum(dim=-1)
        top_values, top_indices = similarity.topk(self.top_k, dim=-1)

        # C3: 置信度门控 — 熵越高（分配模糊）→ gate 越低
        gate_raw = self.gate_mlp(torch.cat([embedding, context], dim=-1))
        if self.entropy_gate_beta != 0.0:
            entropy_norm = entropy / math.log(self.num_prototypes)
            gate = torch.sigmoid(gate_raw - self.entropy_gate_beta * entropy_norm.unsqueeze(-1))
        else:
            gate = torch.sigmoid(gate_raw)

        enhanced = self.layer_norm(embedding + gate * self.memory_proj(context))

        # A1: 训练模式下跟踪原型使用
        if self.training:
            self._usage_this_epoch.scatter_add_(
                0, top_indices[:, 0],
                torch.ones_like(top_indices[:, 0], dtype=torch.long))

        return CA3Output(
            enhanced, context, similarity, assignment,
            top_indices[:, 0], top_values[:, 0], entropy, gate,
        )

    # ── 多样性损失（A2: margin-based） ─────────────────────────────

    def diversity_loss(self, margin: Optional[float] = None) -> torch.Tensor:
        m = margin if margin is not None else self.diversity_margin
        normalized = F.normalize(self.prototypes, dim=-1)
        cosine = normalized @ normalized.t()
        eye = torch.eye(self.num_prototypes, device=cosine.device, dtype=cosine.dtype)
        losses = (cosine - eye - m).clamp_min(0).pow(2)
        return losses.sum() / max(self.num_prototypes * (self.num_prototypes - 1), 1)
