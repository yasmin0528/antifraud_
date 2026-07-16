from typing import NamedTuple, Optional

import torch
import torch.nn as nn


class MPFCOutput(NamedTuple):
    final_logit: torch.Tensor
    score_mid: torch.Tensor
    neural_score: torch.Tensor
    rule_score: torch.Tensor
    rule_confidence: torch.Tensor
    fusion_weight: torch.Tensor


def binary_logits_to_risk_logit(class_logits: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(class_logits):
        raise TypeError("class_logits must be a floating-point tensor")
    if class_logits.ndim != 2 or class_logits.shape[1] != 2:
        raise ValueError("class_logits must have shape [batch_size, 2]")
    if not torch.isfinite(class_logits).all():
        raise ValueError("class_logits must contain only finite values")
    return class_logits[:, 1:2] - class_logits[:, 0:1]


class MPFCDecisionFusion(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1,
                 gate_bias_init: float = -2.0, eps: float = 1e-6,
                 confidence_constrained: bool = True):
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if not 0.0 < eps < 0.5:
            raise ValueError("eps must satisfy 0 < eps < 0.5")

        self.input_dim = int(input_dim)
        self.eps = float(eps)
        self.confidence_constrained = bool(confidence_constrained)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.input_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.constant_(self.gate_mlp[-1].bias, gate_bias_init)

    def forward(self, rgtan_hidden: torch.Tensor, neural_logit: torch.Tensor,
                rule_score: torch.Tensor, rule_confidence: torch.Tensor,
                gate_bias_offset: Optional[torch.Tensor] = None) -> MPFCOutput:
        tensors = {
            "rgtan_hidden": rgtan_hidden,
            "neural_logit": neural_logit,
            "rule_score": rule_score,
            "rule_confidence": rule_confidence,
        }
        batch_size = rgtan_hidden.shape[0] if rgtan_hidden.ndim >= 1 else None
        for name, tensor in tensors.items():
            if not torch.is_floating_point(tensor):
                raise TypeError(f"{name} must be a floating-point tensor")
            if tensor.ndim != 2:
                raise ValueError(f"{name} must be a two-dimensional tensor")
            if tensor.shape[0] != batch_size:
                raise ValueError("all MPFC inputs must have the same batch dimension")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must contain only finite values")

        if rgtan_hidden.shape[1] != self.input_dim:
            raise ValueError(
                f"rgtan_hidden must have feature dimension {self.input_dim}, "
                f"got {rgtan_hidden.shape[1]}"
            )
        for name, tensor in (("neural_logit", neural_logit),
                             ("rule_score", rule_score),
                             ("rule_confidence", rule_confidence)):
            if tensor.shape[1] != 1:
                raise ValueError(f"{name} must have shape [batch_size, 1]")
        if rule_score.lt(0).any() or rule_score.gt(1).any():
            raise ValueError("rule_score must be within [0, 1]")
        if rule_confidence.lt(0).any() or rule_confidence.gt(1).any():
            raise ValueError("rule_confidence must be within [0, 1]")

        gate_input = torch.cat([rgtan_hidden, rule_score, rule_confidence], dim=-1)
        gate_logit = self.gate_mlp(gate_input)
        if gate_bias_offset is not None:
            gate_logit = gate_logit + gate_bias_offset
        gate = torch.sigmoid(gate_logit)
        fusion_weight = rule_confidence * gate if self.confidence_constrained else gate
        rule_logit = torch.logit(rule_score.clamp(min=self.eps, max=1.0 - self.eps))
        final_logit = neural_logit + fusion_weight * (rule_logit - neural_logit)

        return MPFCOutput(
            final_logit=final_logit,
            score_mid=torch.sigmoid(final_logit),
            neural_score=torch.sigmoid(neural_logit),
            rule_score=rule_score,
            rule_confidence=rule_confidence,
            fusion_weight=fusion_weight,
        )
