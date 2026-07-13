import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score

from methods.modules.ca3 import CA3PrototypeMemory
from methods.rgtan.evaluation import best_macro_f1_threshold


def test_ca3_bypass_is_exact_identity():
    module = CA3PrototypeMemory(embedding_dim=8, num_prototypes=4)
    embedding = torch.randn(5, 8)
    output = module(embedding, enabled=False)
    assert torch.equal(output.enhanced_embedding, embedding)
    assert torch.count_nonzero(output.gate) == 0


def test_ca3_requires_initialization_and_backpropagates():
    module = CA3PrototypeMemory(embedding_dim=8, num_prototypes=4, top_k=3)
    embedding = torch.randn(5, 8, requires_grad=True)
    with pytest.raises(RuntimeError):
        module(embedding, enabled=True)
    module.initialize_prototypes(torch.randn(4, 8))
    output = module(embedding, enabled=True)
    assert output.enhanced_embedding.shape == (5, 8)
    assert output.similarity.shape == (5, 4)
    assert output.proto_risk_logit.shape == (5, 1)
    assert torch.isfinite(output.enhanced_embedding).all()
    loss = output.enhanced_embedding.sum() + output.proto_risk_logit.sum() + module.diversity_loss()
    loss.backward()
    assert module.prototypes.grad is not None


def test_ca3_initialization_shape_is_strict():
    module = CA3PrototypeMemory(embedding_dim=8, num_prototypes=4)
    with pytest.raises(ValueError):
        module.initialize_prototypes(torch.randn(3, 8))


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
