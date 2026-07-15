import pytest
import torch

from methods.modules.mpfc import (
    MPFCDecisionFusion,
    binary_logits_to_risk_logit,
)


def _inputs(batch_size=5, input_dim=8):
    return (
        torch.randn(batch_size, input_dim),
        torch.randn(batch_size, 1),
        torch.rand(batch_size, 1),
        torch.rand(batch_size, 1),
    )


def test_binary_logits_to_risk_logit_matches_softmax_probability():
    class_logits = torch.randn(16, 2)
    risk_logit = binary_logits_to_risk_logit(class_logits)
    assert risk_logit.shape == (16, 1)
    assert torch.allclose(
        torch.sigmoid(risk_logit),
        torch.softmax(class_logits, dim=-1)[:, 1:2],
    )


def test_mpfc_output_contract_and_gradients():
    module = MPFCDecisionFusion(input_dim=8, hidden_dim=4, dropout=0.0)
    hidden, neural_logit, rule_score, rule_confidence = _inputs()
    hidden.requires_grad_()
    neural_logit.requires_grad_()

    output = module(hidden, neural_logit, rule_score, rule_confidence)

    for field in output:
        assert field.shape == (5, 1)
        assert field.ndim == 2
        assert torch.is_floating_point(field)
        assert torch.isfinite(field).all()
    assert (output.score_mid >= 0).all() and (output.score_mid <= 1).all()
    assert (output.neural_score >= 0).all() and (output.neural_score <= 1).all()
    assert (output.fusion_weight <= rule_confidence).all()

    output.final_logit.sum().backward()
    assert hidden.grad is not None
    assert neural_logit.grad is not None
    assert module.gate_mlp[-1].weight.grad is not None


def test_mpfc_zero_confidence_is_exact_neural_bypass():
    module = MPFCDecisionFusion(input_dim=8, hidden_dim=4, dropout=0.0)
    hidden, neural_logit, rule_score, _ = _inputs()
    confidence = torch.zeros(5, 1)

    output = module(hidden, neural_logit, rule_score, confidence)

    assert torch.count_nonzero(output.fusion_weight) == 0
    assert torch.equal(output.final_logit, neural_logit)
    assert torch.equal(output.score_mid, torch.sigmoid(neural_logit))


def test_mpfc_default_gate_bias():
    module = MPFCDecisionFusion(input_dim=8, gate_bias_init=-2.0)
    assert module.gate_mlp[-1].bias.item() == pytest.approx(-2.0)


@pytest.mark.parametrize(
    "input_index,replacement,error_type",
    [
        (0, torch.randn(5, 7), ValueError),
        (1, torch.randn(5), ValueError),
        (2, torch.full((5, 1), 1.1), ValueError),
        (3, torch.full((5, 1), -0.1), ValueError),
        (1, torch.ones(5, 1, dtype=torch.long), TypeError),
        (0, torch.full((5, 8), float("nan")), ValueError),
    ],
)
def test_mpfc_rejects_invalid_inputs(input_index, replacement, error_type):
    module = MPFCDecisionFusion(input_dim=8)
    inputs = list(_inputs())
    inputs[input_index] = replacement
    with pytest.raises(error_type):
        module(*inputs)


def test_mpfc_checkpoint_round_trip():
    source = MPFCDecisionFusion(input_dim=8, hidden_dim=4, dropout=0.0)
    restored = MPFCDecisionFusion(input_dim=8, hidden_dim=4, dropout=0.0)
    restored.load_state_dict(source.state_dict())
    inputs = _inputs()
    source.eval()
    restored.eval()
    with torch.no_grad():
        expected = source(*inputs)
        actual = restored(*inputs)
    for expected_field, actual_field in zip(expected, actual):
        assert torch.equal(expected_field, actual_field)
