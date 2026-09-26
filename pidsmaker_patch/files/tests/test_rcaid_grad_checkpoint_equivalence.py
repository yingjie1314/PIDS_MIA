"""PIDS_GRAD_CHECKPOINT=1 must change memory behaviour only: same forward output, same loss,
same gradients as the plain path. The GAT layers hold no dropout (RCaidGAT's only dropout sits
outside the checkpointed blocks and draws from the global RNG in the same order either way), so
on CPU the comparison is exact, not approximate.
"""

import contextlib
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pidsmaker.encoders.rcaid_encoder import RCaidGAT


def small_graph(seed=0, n_nodes=50, n_edges=200, in_dim=16):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_nodes, in_dim, generator=g)
    edge_index = torch.randint(0, n_nodes, (2, n_edges), generator=g)
    return x, edge_index


def run_once(checkpointed, monkeypatch, dropout, train_mode, no_grad=False):
    if checkpointed:
        monkeypatch.setenv("PIDS_GRAD_CHECKPOINT", "1")
    else:
        monkeypatch.delenv("PIDS_GRAD_CHECKPOINT", raising=False)

    torch.manual_seed(0)
    model = RCaidGAT(in_dim=16, hid_dim=8, out_dim=3, dropout=dropout)
    model.train(train_mode)

    x, edge_index = small_graph()
    torch.manual_seed(1)  # governs dropout1's mask, drawn in the same order on both paths
    with torch.no_grad() if no_grad else contextlib.nullcontext():
        out = model(x, edge_index)["h"]
        loss = out.pow(2).sum()
    if train_mode and not no_grad:
        loss.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters()}
    else:
        grads = {}
    return out.detach(), loss.detach(), grads


@pytest.mark.parametrize("dropout", [0.0, 0.5])
def test_training_forward_and_gradients_are_identical(monkeypatch, dropout):
    out_a, loss_a, grads_a = run_once(False, monkeypatch, dropout, train_mode=True)
    out_b, loss_b, grads_b = run_once(True, monkeypatch, dropout, train_mode=True)
    # Forward is bitwise identical (verified down to torch.equal). Gradients are mathematically
    # identical but accumulate per-head slices in a different order, so they match to float32
    # summation-order ulps (~1e-8 measured) -- the same class of difference CUDA scatter atomics
    # introduce on every production run.
    assert torch.equal(out_a, out_b)
    assert torch.equal(loss_a, loss_b)
    assert grads_a.keys() == grads_b.keys()
    for name in grads_a:
        assert torch.allclose(grads_a[name], grads_b[name], rtol=1e-5, atol=1e-7), name


@pytest.mark.parametrize("no_grad", [False, True])
def test_inference_output_is_unchanged_by_the_flag(monkeypatch, no_grad):
    """The flag must move memory, never inference output -- in eval, with or without no_grad.

    The no_grad case is the one that matters in production: the MIA dump calls this encoder under
    `@torch.no_grad()`, and it is the branch that takes `_gat_one_head` directly rather than
    through `checkpoint`. Before the head-split was extended to cover it, that branch built one
    (E, H, C) tensor whole and OOM'd a 32 GB card on CLEARSCOPE's 4.6M-edge windows.
    """
    out_a, *_ = run_once(False, monkeypatch, dropout=0.5, train_mode=False, no_grad=no_grad)
    out_b, *_ = run_once(True, monkeypatch, dropout=0.5, train_mode=False, no_grad=no_grad)
    assert torch.equal(out_a, out_b)


def test_no_grad_inference_matches_grad_enabled_inference(monkeypatch):
    """Head-splitting under no_grad must agree with the checkpointed grad-enabled eval path.

    Pins the new branch against the one the equivalence argument was originally made for, so a
    divergence between them cannot hide behind both being 'the checkpointed path'.
    """
    out_grad, *_ = run_once(True, monkeypatch, dropout=0.5, train_mode=False, no_grad=False)
    out_nograd, *_ = run_once(True, monkeypatch, dropout=0.5, train_mode=False, no_grad=True)
    assert torch.equal(out_grad, out_nograd)
