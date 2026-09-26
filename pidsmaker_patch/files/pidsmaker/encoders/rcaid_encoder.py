"""R-Caid system GAT encoder with residual MLP aggregation.

R-Caid-specific encoder combining multi-layer GAT with MLP-based aggregation
for root cause analysis and attack investigation in provenance graphs.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_scatter import scatter


def _grad_checkpoint_enabled() -> bool:
    """Opt-in via environment for the same reason as PIDS_LOW_MEMORY: it changes the execution
    strategy (recompute activations in backward instead of retaining them), never the math, so it
    must stay out of the artifact hash chain."""
    return os.environ.get("PIDS_GRAD_CHECKPOINT", "").strip().lower() not in ("", "0", "false")


def _gat_one_head(conv: GATConv, head: int, x, edge_index, num_nodes: int):
    """One attention head of `conv`, computed from slices of the conv's own parameters.

    Multi-head GAT with concat=True is exactly H independent single-head GATs whose outputs are
    concatenated: row block h of `lin.weight` and slice h of `att_src`/`att_dst` never interact
    with any other head before the concat. Re-expressing one head functionally (mirroring
    GATConv.forward of torch_geometric 2.5.3 for the dense, no-edge-attr case, with self-loops
    already added by the caller) lets the checkpointed backward materialise one head's per-edge
    tensors at a time -- (E, C) instead of (E, H, C). Gradients flow into the original parameters
    through the slices. Equivalence is pinned by tests/test_rcaid_grad_checkpoint_equivalence.py.
    """
    H, C = conv.heads, conv.out_channels
    W = conv.lin.weight[head * C:(head + 1) * C]
    x_h = F.linear(x, W)                                       # (N, C); GATConv's lin has no bias
    alpha_src = (x_h * conv.att_src[0, head]).sum(dim=-1)      # (N,)
    alpha_dst = (x_h * conv.att_dst[0, head]).sum(dim=-1)

    j, i = edge_index[0], edge_index[1]                        # source -> target
    alpha = F.leaky_relu(alpha_src[j] + alpha_dst[i], conv.negative_slope)
    alpha = softmax(alpha, i, num_nodes=num_nodes)
    alpha = F.dropout(alpha, p=conv.dropout, training=conv.training)

    msg = x_h[j] * alpha.unsqueeze(-1)                         # (E, C)
    return scatter(msg, i, dim=0, dim_size=num_nodes, reduce="sum")


def _gat_head_chunked(conv: GATConv, x, edge_index):
    """`conv(x, edge_index)`, one head at a time; checkpointed only when a backward will happen.

    Under `no_grad` there are no activations to trade for recompute, but the per-head split still
    caps the peak transient at (E, C) instead of (E, H, C), which is the whole problem during
    inference. Calling `_gat_one_head` directly there keeps `checkpoint`'s no-grad behaviour out
    of the picture entirely.
    """
    num_nodes = x.size(0)
    if conv.add_self_loops:
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    outs = [
        checkpoint(_gat_one_head, conv, h, x, edge_index, num_nodes, use_reentrant=False)
        if torch.is_grad_enabled()
        else _gat_one_head(conv, h, x, edge_index, num_nodes)
        for h in range(conv.heads)
    ]
    out = torch.cat(outs, dim=-1) if conv.concat else torch.stack(outs, dim=1).mean(dim=1)
    if conv.bias is not None:
        out = out + conv.bias
    return out


class _RcaidMLP(nn.Module):
    """Internal MLP for R-Caid aggregation."""

    def __init__(self, input_dim, output_dim):
        super(_RcaidMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class RCaidGAT(nn.Module):
    """R-Caid GAT encoder with 3-layer attention and MLP aggregation.

    Combines three GAT layers with an MLP that aggregates intermediate and final
    representations for improved node embeddings in causal analysis tasks.
    """

    def __init__(self, in_dim, hid_dim, out_dim, dropout, num_heads=4):
        super(RCaidGAT, self).__init__()
        self.gat1 = GATConv(in_dim, hid_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hid_dim * num_heads, hid_dim, heads=num_heads, concat=True)
        self.gat3 = GATConv(
            hid_dim * num_heads, out_dim, heads=1, concat=False
        )  # Output is not concatenated
        self.mlp = _RcaidMLP(hid_dim * num_heads + out_dim, out_dim)  # Input is concatenated
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x, edge_index, **kwargs):
        # GAT attention retains per-edge, per-head activations for backward: with 4 heads x 128
        # dims that is kilobytes of GPU per edge per layer, which the measured CADETS run confirms
        # (19.16 GB peak at 1.6M edges). R-CAID's pseudo-graph windows reach 5-6M edges on
        # CLEARSCOPE, past the 32 GB cards even with whole-layer checkpointing (measured 33.7 GB
        # true peak). So the checkpoint unit is one attention HEAD, an exact decomposition of
        # concat-mode GAT: per-edge transients shrink from (E, H, C) to (E, C), identical
        # gradients (the GAT layers hold no dropout, so recomputation is deterministic), ~1/3
        # extra forward compute.
        # Gated on the flag alone, not on `self.training`: the MIA dump runs this encoder under
        # no_grad on the same windows, where the eager branch builds one (E, H, C) message tensor
        # whole -- 11.68 GiB at CLEARSCOPE's largest window (4.6M edges), which OOM'd a 32 GB card
        # at epoch 0 even after training itself had been brought down to 11.33 GB.
        if _grad_checkpoint_enabled():
            x1 = F.relu(_gat_head_chunked(self.gat1, x, edge_index))
            x2 = F.relu(_gat_head_chunked(self.gat2, x1, edge_index))
            x3 = _gat_head_chunked(self.gat3, x2, edge_index)
        else:
            x1 = self.gat1(x, edge_index)
            x1 = F.relu(x1)
            # GAT Layer 2 with attention
            x2 = self.gat2(x1, edge_index)
            x2 = F.relu(x2)
            # Aggregation through attention in the third layer
            x3 = self.gat3(x2, edge_index)

        x3 = self.dropout1(x3)
        # Update through MLP (concatenate previous layer's output with the current output)
        mlp_input = torch.cat([x2, x3], dim=-1)

        out = self.mlp(mlp_input)

        return {"h": out}
