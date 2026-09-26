"""Phase 0/1 acceptance: model checkpoint round-trip, and native class posteriors.

Checkpointing was disabled because `save_model` read `model.encoder.neighbor_loader`, which does
not exist on this version's TGNEncoder (`neighbor_loader` is a local of the batching helper). These
tests pin the guarded behaviour and the posterior contract the MIA attacks depend on.

Run:  pytest tests/test_checkpoint_and_posterior.py -v
"""

import os

import pytest
import torch
import torch.nn as nn

from pidsmaker.utils.data_utils import load_model, save_model


class _FakeTGN(nn.Module):
    """Stands in for TGNEncoder: has `memory`, deliberately has NO `neighbor_loader`."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 3)
        self.memory = nn.Parameter(torch.randn(5, 4), requires_grad=False)


class _FakeModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(3, 2)


class _Ns:
    """Attribute bag, so the fake cfg reads like the real dotted config."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _Cfg():
    """Minimal duck-typed cfg exposing only what save_model/load_model read."""
    return _Ns(
        training=_Ns(encoder=_Ns(tgn=_Ns(use_memory=False))),
        batching=_Ns(edge_features=["edge_type"]),
    )


def test_save_model_does_not_crash_without_neighbor_loader(tmp_path, monkeypatch):
    """The regression that disabled checkpointing: encoder has no `neighbor_loader` attribute."""
    import pidsmaker.utils.data_utils as du

    monkeypatch.setattr(du, "TGNEncoder", _FakeTGN)  # make isinstance() take the TGN branch
    model = _FakeModel(_FakeTGN())
    assert not hasattr(model.encoder, "neighbor_loader")

    save_model(model, str(tmp_path / "ckpt"), _Cfg())

    assert os.path.exists(tmp_path / "ckpt" / "state_dict.pkl")
    assert not os.path.exists(tmp_path / "ckpt" / "neighbor_loader.pkl"), (
        "must not write a file for an attribute the encoder does not have"
    )


def test_save_load_round_trip_restores_every_tensor(tmp_path, monkeypatch):
    import pidsmaker.utils.data_utils as du

    monkeypatch.setattr(du, "TGNEncoder", _FakeTGN)
    src = _FakeModel(_FakeTGN())
    save_model(src, str(tmp_path / "ckpt"), _Cfg())

    dst = _FakeModel(_FakeTGN())
    # sanity: the fresh model really is different before loading
    assert not torch.allclose(src.head.weight, dst.head.weight)

    load_model(dst, str(tmp_path / "ckpt"), _Cfg())

    for k, v in src.state_dict().items():
        assert torch.equal(v, dst.state_dict()[k]), f"{k} not restored"


def test_load_model_tolerates_absent_neighbor_loader_file(tmp_path, monkeypatch):
    """load_model must mirror save_model: no file written => no file expected."""
    import pidsmaker.utils.data_utils as du

    monkeypatch.setattr(du, "TGNEncoder", _FakeTGN)
    save_model(_FakeModel(_FakeTGN()), str(tmp_path / "ckpt"), _Cfg())
    load_model(_FakeModel(_FakeTGN()), str(tmp_path / "ckpt"), _Cfg())  # must not raise


# --------------------------------------------------------------------------------------------
# Phase 1: native class posteriors
# --------------------------------------------------------------------------------------------


class _IdentityDecoder(nn.Module):
    def forward(self, h=None, h_src=None, h_dst=None):
        return h if h is not None else h_src + h_dst


def _one_hot(idx, n):
    return torch.nn.functional.one_hot(idx, n).float()


def test_edge_type_posterior_shape_normalisation_and_argmax():
    from pidsmaker.objectives.predict_edge_type import EdgeTypePrediction

    E, C = 7, 10
    obj = EdgeTypePrediction(
        decoder=_IdentityDecoder(),
        loss_fn=lambda h, y, inference, class_weights: torch.zeros(len(h)),
        balanced_loss=False,
        edge_type_dim=C,
    )
    h_src, h_dst = torch.randn(E, C), torch.randn(E, C)
    edge_type = _one_hot(torch.randint(0, C, (E,)), C)

    out = obj(h_src=h_src, h_dst=h_dst, edge_type=edge_type, inference=True)

    assert out["posterior"].shape == (E, C)
    assert torch.allclose(out["posterior"].sum(1), torch.ones(E), atol=1e-5)
    assert torch.equal(out["posterior"].argmax(1), out["logits"].argmax(1))


def test_edge_type_posterior_absent_during_training():
    """Training must be untouched -- posteriors are an inference-only observable."""
    from pidsmaker.objectives.predict_edge_type import EdgeTypePrediction

    E, C = 7, 10
    obj = EdgeTypePrediction(
        decoder=_IdentityDecoder(),
        loss_fn=lambda h, y, inference, class_weights: torch.zeros(1),
        balanced_loss=False,
        edge_type_dim=C,
    )
    out = obj(
        h_src=torch.randn(E, C),
        h_dst=torch.randn(E, C),
        edge_type=_one_hot(torch.randint(0, C, (E,)), C),
        inference=False,
    )
    assert "posterior" not in out and "logits" not in out


def test_node_type_posterior_comes_from_decoder_logits_not_embedding():
    """The posterior must be softmax(h_hat). A non-identity decoder makes h and h_hat differ, so
    a posterior built from `h` (as the legacy "out" key is) would be wrong."""
    from pidsmaker.objectives.predict_node_type import NodeTypePrediction

    N, D, C = 6, 8, 3

    class _Proj(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(D, C)

        def forward(self, h):
            return self.lin(h)

    decoder = _Proj()
    obj = NodeTypePrediction(
        decoder=decoder,
        loss_fn=lambda h, y, inference, weight: torch.zeros(len(h)),
        balanced_loss=False,
        node_type_dim=C,
    )
    h = torch.randn(N, D)
    out = obj(h=h, node_type=_one_hot(torch.randint(0, C, (N,)), C), inference=True)

    assert out["posterior"].shape == (N, C)
    assert torch.allclose(out["posterior"].sum(1), torch.ones(N), atol=1e-5)
    expected = torch.softmax(decoder(h), dim=1)
    assert torch.allclose(out["posterior"], expected, atol=1e-6), "posterior must use h_hat"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
