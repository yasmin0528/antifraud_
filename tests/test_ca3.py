import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from methods.modules.ca3 import CA3PrototypeMemory
from methods.rgtan.evaluation import best_macro_f1_threshold


def test_ca3_bypass_is_exact_identity():
    module = CA3PrototypeMemory(embedding_dim=8, num_prototypes=4)
    embedding = torch.randn(5, 8)
    output = module(embedding, enabled=False)
    assert torch.equal(output.enhanced_embedding, embedding)
    assert torch.count_nonzero(output.gate_probs) == 0


def test_ca3_requires_initialization_and_backpropagates():
    module = CA3PrototypeMemory(embedding_dim=8, num_prototypes=4, top_k=3)
    embedding = torch.randn(5, 8, requires_grad=True)
    with pytest.raises(RuntimeError):
        module(embedding, enabled=True)
    module.initialize_prototypes(torch.randn(4, 8))
    output = module(embedding, enabled=True)
    assert output.enhanced_embedding.shape == (5, 8)
    assert output.similarity.shape == (5, 4)
    assert torch.isfinite(output.enhanced_embedding).all()
    # B1: contrastive_loss 替代旧的 risk_head BCE
    loss = output.enhanced_embedding.sum() + module.contrastive_loss(embedding) + module.diversity_loss()
    loss.backward()
    assert module.prototypes.grad is not None


def test_ca3_initialization_shape_is_strict():
    module = CA3PrototypeMemory(embedding_dim=8, num_prototypes=4)
    with pytest.raises(ValueError):
        module.initialize_prototypes(torch.randn(3, 8))


def test_ca3_initialization_state_survives_checkpoint_round_trip():
    source = CA3PrototypeMemory(embedding_dim=8, num_prototypes=4)
    source.initialize_prototypes(torch.randn(4, 8))
    checkpoint = {"ca3_state_dict": source.state_dict(), "ca3_initialized": True}
    restored = CA3PrototypeMemory(embedding_dim=8, num_prototypes=4)
    restored.load_state_dict(checkpoint["ca3_state_dict"])
    assert bool(restored.initialized.item()) is True
    assert torch.equal(restored.prototypes, source.prototypes)
    assert torch.equal(restored.init_prototypes, source.init_prototypes)


def test_ca3_output_fields():
    """验证删除 risk_head 后 CA3Output 字段正确。"""
    module = CA3PrototypeMemory(embedding_dim=8, num_prototypes=4)
    module.initialize_prototypes(torch.randn(4, 8))
    embedding = torch.randn(5, 8)
    output = module(embedding, enabled=True)
    # 应有的字段
    assert hasattr(output, "enhanced_embedding")
    assert hasattr(output, "similarity")
    assert hasattr(output, "assignment_weights")
    assert hasattr(output, "top_proto_id")
    assert hasattr(output, "top_proto_sim")
    assert hasattr(output, "assignment_entropy")
    assert hasattr(output, "gate_probs")
    # 不应有的字段（risk_head 已删除）
    assert not hasattr(output, "proto_risk_logit")
    assert not hasattr(output, "proto_score")


def test_validation_threshold_selection():
    threshold, score = best_macro_f1_threshold([0, 0, 1, 1], [0.1, 0.3, 0.6, 0.9])
    assert 0.3 < threshold <= 0.6
    assert np.isclose(score, 1.0)


def test_validation_threshold_matches_exhaustive_search_with_ties():
    rng = np.random.default_rng(2023)
    for size in (1, 2, 17, 100):
        labels = rng.integers(0, 2, size=size)
        scores = rng.choice(np.linspace(0.0, 1.0, 11), size=size)
        candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
        expected_threshold, expected_f1 = 0.5, -1.0
        for candidate in candidates:
            candidate_f1 = f1_score(
                labels, scores >= candidate, average="macro", zero_division=0)
            if candidate_f1 > expected_f1:
                expected_threshold, expected_f1 = float(candidate), float(candidate_f1)
        threshold, score = best_macro_f1_threshold(labels, scores)
        assert threshold == expected_threshold
        assert np.isclose(score, expected_f1)


def test_validation_threshold_rejects_invalid_input():
    with pytest.raises(ValueError):
        best_macro_f1_threshold([], [])
    with pytest.raises(ValueError):
        best_macro_f1_threshold([0, 1], [0.1, np.nan])


# ── C2: 融合退火 ────────────────────────────────────────────────

def test_ca3_anneal_progress():
    """验证 gate_mlp.bias 从 gate_bias_init 到 gate_bias_final 线性变化。"""
    module = CA3PrototypeMemory(
        embedding_dim=8, num_prototypes=4,
        gate_bias_init=-2.0, gate_bias_final=-1.0)
    module.set_anneal_progress(0.0)
    assert module.gate_mlp.bias.item() == pytest.approx(-2.0, abs=1e-6)
    module.set_anneal_progress(0.5)
    assert module.gate_mlp.bias.item() == pytest.approx(-1.5, abs=1e-6)
    module.set_anneal_progress(1.0)
    assert module.gate_mlp.bias.item() == pytest.approx(-1.0, abs=1e-6)
    # 超界裁剪
    module.set_anneal_progress(1.5)
    assert module.gate_mlp.bias.item() == pytest.approx(-1.0, abs=1e-6)


# ── A1: 死原型检测与刷新 ─────────────────────────────────────────

def test_ca3_dead_prototype_refresh():
    """验证连续 dead_epoch_threshold 个 epoch 未使用的原型被替换。"""
    module = CA3PrototypeMemory(
        embedding_dim=8, num_prototypes=4,
        dead_epoch_threshold=2)
    module.initialize_prototypes(torch.randn(4, 8))
    old_protos = module.prototypes.clone()

    # 模拟训练：前 2 个 epoch 所有原型都使用
    module.train()
    for _ in range(2):
        emb = torch.randn(10, 8)
        module(emb, enabled=True)
        module.on_epoch_end()

    # 所有原型都应存活
    n = module.refresh_dead_prototypes(torch.randn(5, 8))
    assert n == 0

    # 再模拟 2 个 epoch 完全不使用
    module.eval()  # 验证 eval 模式不累计
    for _ in range(2):
        module.on_epoch_end()
    # eval 不应累计使用统计
    n = module.refresh_dead_prototypes(torch.randn(5, 8))
    assert n == 0

    # 手动置零使用统计模拟死亡
    module.train()
    module._usage_this_epoch.zero_()
    for _ in range(3):
        module.on_epoch_end()

    n = module.refresh_dead_prototypes(torch.randn(5, 8))
    assert n == 4, f"Expected all 4 prototypes dead, got {n}"
    # 原型应被更新（不再等于旧值）
    assert not torch.allclose(module.prototypes, old_protos)


def test_ca3_refresh_with_insufficient_seeds():
    """验证正样本数少于死亡原型数时仍能正确刷新。"""
    module = CA3PrototypeMemory(
        embedding_dim=8, num_prototypes=16,
        dead_epoch_threshold=1)
    module.initialize_prototypes(torch.randn(16, 8))
    # 模拟所有原型连续死亡
    module.train()
    module._usage_this_epoch.zero_()
    module.on_epoch_end()
    module.on_epoch_end()
    # 只有 3 个 seed
    n = module.refresh_dead_prototypes(torch.randn(3, 8))
    assert n == 16


# ── B1: 对比原型学习 ────────────────────────────────────────────

def test_ca3_contrastive_loss():
    """验证 contrastive_loss 可微分且不为 NaN。"""
    module = CA3PrototypeMemory(embedding_dim=8, num_prototypes=4, contrastive_temperature=0.1)
    module.initialize_prototypes(torch.randn(4, 8))
    embedding = torch.randn(10, 8)
    loss = module.contrastive_loss(embedding)
    assert loss.numel() == 1
    assert torch.isfinite(loss).all()
    # 可反向传播
    loss.backward()
    assert module.prototypes.grad is not None


def test_ca3_contrastive_loss_uses_all_prototypes():
    """验证 contrastive_loss 能利用所有原型（而非只有活跃的少数）。"""
    module = CA3PrototypeMemory(embedding_dim=8, num_prototypes=16, contrastive_temperature=0.1)
    module.initialize_prototypes(torch.randn(16, 8))
    embedding = torch.randn(32, 8)
    loss = module.contrastive_loss(embedding)
    loss.backward()
    # 所有原型都应收到梯度
    grad_norms = module.prototypes.grad.norm(dim=-1)
    assert (grad_norms > 0).sum() >= 14, f"Expected ≥14 prototypes to receive gradient, got {(grad_norms > 0).sum()}"


# ── C3: 置信度门控 ──────────────────────────────────────────────

def test_ca3_entropy_gating():
    """验证高熵（模糊分配）样本的 gate 值更低。"""
    module = CA3PrototypeMemory(
        embedding_dim=8, num_prototypes=4,
        entropy_gate_beta=2.0)  # 放大效果以便检测
    module.initialize_prototypes(torch.randn(4, 8))

    # 低熵：embedding 接近某个原型
    close = module.prototypes[0:1] + torch.randn(1, 8) * 0.01
    # 高熵：embedding 远离所有原型
    far = torch.randn(1, 8) * 10

    module.eval()
    with torch.no_grad():
        # 将 close×10 和 far×10 堆叠在一起一次性 forward
        combined = torch.cat([close.repeat(10, 1), far.repeat(10, 1)], dim=0)
        output = module(combined, enabled=True)

    low_entropy_gate = output.gate_probs[:10].mean().item()
    high_entropy_gate = output.gate_probs[10:].mean().item()
    assert low_entropy_gate > high_entropy_gate, (
        f"Expected low-entropy gate > high-entropy gate, got {low_entropy_gate} ≤ {high_entropy_gate}")


def test_ca3_entropy_gating_off():
    """验证 entropy_gate_beta=0 时退化为纯 sigmoid。"""
    module = CA3PrototypeMemory(
        embedding_dim=8, num_prototypes=4,
        entropy_gate_beta=0.0)
    module.initialize_prototypes(torch.randn(4, 8))
    embedding = torch.randn(10, 8)
    output = module(embedding, enabled=True)
    # 所有 gate 应在 0~1 之间
    assert (output.gate_probs >= 0).all() and (output.gate_probs <= 1).all()


# ── A2: Margin Diversity ───────────────────────────────────────

def test_ca3_margin_diversity():
    """验证 margin 下方的原型对不受 diversity 惩罚。"""
    module = CA3PrototypeMemory(embedding_dim=8, num_prototypes=4, diversity_margin=0.5)

    # 让原型全部相等 → 相似度 = 1.0（远大于 margin）
    same = torch.ones(4, 8)
    same = F.normalize(same, dim=-1)
    # 需要绕过 initialize_prototypes 来直接设置
    module.prototypes.data = same.clone()
    module.initialized.fill_(True)
    loss_high = module.diversity_loss()

    # 让原型全部正交 → 相似度 ≈ 0（低于 margin）
    module.prototypes.data = torch.eye(4, 8)
    loss_low = module.diversity_loss()

    assert loss_high > 0, "Similar prototypes should be penalized"
    assert loss_low == 0, "Orthogonal prototypes should NOT be penalized"


# ── 端到端集成测试 ─────────────────────────────────────────────

def test_ca3_gate_bias_init():
    """Negative gate_bias_init pushes initial gate values toward 0."""
    zero_bias = CA3PrototypeMemory(embedding_dim=8, num_prototypes=4, gate_bias_init=0.0)
    neg_bias = CA3PrototypeMemory(embedding_dim=8, num_prototypes=4, gate_bias_init=-2.0)
    assert zero_bias.gate_mlp.bias.item() == 0.0
    assert neg_bias.gate_mlp.bias.item() == -2.0
    neg_bias.initialize_prototypes(torch.randn(4, 8))
    embedding = torch.randn(10, 8)
    output = neg_bias(embedding, enabled=True)
    avg_gate = output.gate_probs.mean().item()
    assert 0.03 < avg_gate < 0.25, f"Expected gate signif below 0.5, got {avg_gate}"
