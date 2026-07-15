import dgl
import torch
import torch.nn as nn

from methods.rgtan.rgtan_model import RGTAN


def test_rgtan_optionally_returns_classification_input_hidden_state():
    graph = dgl.graph((torch.arange(3), torch.arange(3)))
    model = RGTAN(
        in_feats=4,
        hidden_dim=3,
        n_layers=1,
        n_classes=2,
        heads=[2],
        activation=nn.PReLU(),
        post_proc=False,
        n2v_feat=False,
        drop=[0.0, 0.0],
        cat_features=[],
        neigh_features=[],
    )
    model.eval()
    features = torch.randn(3, 4)
    labels = torch.full((3,), 2, dtype=torch.long)

    with torch.no_grad():
        legacy_logits = model([graph], features, labels)
        class_logits, hidden = model([graph], features, labels, return_hidden=True)

    assert isinstance(legacy_logits, torch.Tensor)
    assert legacy_logits.shape == class_logits.shape == (3, 2)
    assert torch.equal(legacy_logits, class_logits)
    assert model.classification_input_dim == 6
    assert hidden.shape == (3, model.classification_input_dim)
