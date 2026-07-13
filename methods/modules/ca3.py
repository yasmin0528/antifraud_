from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CA3Output(NamedTuple):
    enhanced_embedding: torch.Tensor
    memory_context: torch.Tensor
    similarity: torch.Tensor
    assignment_weights: torch.Tensor
    proto_risk_logit: torch.Tensor
    proto_score: torch.Tensor
    top_proto_id: torch.Tensor
    top_proto_sim: torch.Tensor
    assignment_entropy: torch.Tensor
    gate: torch.Tensor


class CA3PrototypeMemory(nn.Module):
    """Deployable risk-pattern memory; no labels or group IDs enter forward()."""

    def __init__(self, embedding_dim=64, num_prototypes=16, temperature=0.2,
                 top_k=3, fusion="gated_residual"):
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
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embedding_dim) * 0.02)
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.memory_proj = nn.Linear(embedding_dim, embedding_dim)
        self.gate_mlp = nn.Linear(embedding_dim * 2, 1)
        self.risk_head = nn.Linear(3, 1)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        nn.init.eye_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.eye_(self.memory_proj.weight)
        nn.init.zeros_(self.memory_proj.bias)
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("init_prototypes", torch.zeros(num_prototypes, embedding_dim))

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

    def forward(self, embedding, enabled=True):
        n = embedding.shape[0]
        if not enabled:
            zeros_h = torch.zeros_like(embedding)
            zeros_m = embedding.new_zeros((n, self.num_prototypes))
            zeros_1 = embedding.new_zeros((n, 1))
            zeros_n = embedding.new_zeros(n)
            return CA3Output(
                embedding, zeros_h, zeros_m, zeros_m, zeros_1,
                torch.sigmoid(zeros_1), torch.zeros(n, dtype=torch.long, device=embedding.device),
                zeros_n, zeros_n, zeros_1,
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
        risk_features = torch.stack(
            [top_values[:, 0], top_values.mean(dim=-1), entropy], dim=-1)
        risk_logit = self.risk_head(risk_features)
        gate = torch.sigmoid(self.gate_mlp(torch.cat([embedding, context], dim=-1)))
        enhanced = self.layer_norm(embedding + gate * self.memory_proj(context))
        return CA3Output(
            enhanced, context, similarity, assignment, risk_logit,
            torch.sigmoid(risk_logit), top_indices[:, 0], top_values[:, 0], entropy, gate,
        )

    def diversity_loss(self):
        normalized = F.normalize(self.prototypes, dim=-1)
        cosine = normalized @ normalized.t()
        off_diagonal = cosine - torch.eye(
            self.num_prototypes, device=cosine.device, dtype=cosine.dtype)
        return off_diagonal.pow(2).sum() / max(self.num_prototypes * (self.num_prototypes - 1), 1)
