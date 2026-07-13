import pandas as pd
import torch

from feature_engineering.ca1_cache import build_aml_ca1_cache
from methods.modules.ca1 import CA1Encoder
from methods.rgtan.rgtan_main import _seed_local_indices
from methods.rgtan.rgtan_lpa import load_lpa_subtensor


def test_cache_uses_only_prior_sender_history(tmp_path):
    source = tmp_path / "AML_gtan_processed.csv"
    cache_path = tmp_path / "aml_ca1_k2.pt"
    pd.DataFrame({
        "TX_ID": [2, 1, 3], "Source": ["a", "a", "b"],
        "Type": ["x", "x", "y"], "Time": [2.0, 1.0, 1.0],
        "AmountNorm": [20.0, 10.0, 30.0], "LogAmount": [2.0, 1.0, 3.0],
        "TimeDiff": [1.0, 0.0, 0.0],
    }).to_csv(source, index=False)
    cache = build_aml_ca1_cache(str(source), str(cache_path), k=2)
    assert cache["sample_ids"] == ["2", "1", "3"]
    assert cache["padding_mask_true_is_pad"] is True
    assert cache["sequence_len"].tolist() == [1, 0, 0]
    assert cache["sequence"][0, -1, 0].item() == 10.0
    assert cache["padding_mask"][0].tolist() == [True, False]


def test_ca1_all_padding_is_finite_zero_embedding():
    encoder = CA1Encoder(input_dim=4, hidden_dim=8, dropout=0.0)
    sequence = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    embedding, logit, score = encoder(sequence, torch.tensor([0, 2]), mask)
    assert embedding.shape == (2, 8)
    assert logit.shape == score.shape == (2, 1)
    assert torch.equal(embedding[0], torch.zeros_like(embedding[0]))
    assert torch.isfinite(embedding).all() and torch.isfinite(logit).all()
    (embedding.sum() + logit.sum()).backward()


def test_seed_mapping_does_not_assume_prefix_order():
    input_nodes = torch.tensor([8, 3, 5, 1])
    seeds = torch.tensor([1, 8])
    assert _seed_local_indices(input_nodes, seeds).tolist() == [3, 0]


def test_label_propagation_uses_train_whitelist_and_masks_seed():
    node_feat = torch.zeros(4, 2)
    labels = torch.tensor([0, 1, 1, 0])
    known = torch.tensor([0, 2, 1, 2])
    input_nodes = torch.tensor([2, 0, 3, 1])
    seeds = torch.tensor([0])
    result = load_lpa_subtensor(
        node_feat, {}, {}, {}, labels, seeds, input_nodes, "cpu", [], known)
    propagated = result[-1]
    assert propagated.tolist() == [1, 2, 2, 2]
