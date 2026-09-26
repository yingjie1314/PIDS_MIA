"""The preallocated `feat_inference` must produce exactly what the list-of-tensors version did.

That function was rewritten to stop R-CAID OOMing on E3-THEIA (side task S4). It feeds every
downstream task, so "uses less memory" is worth nothing unless the bytes are identical: a silent
change to the message-column layout would retrain every detector on differently-ordered features
and invalidate the detection tables without failing anything visibly.

The reference below is the original implementation, copied verbatim.
"""

import networkx as nx
import numpy as np
import pytest
import torch

from pidsmaker.tasks.feat_inference import feat_inference
from pidsmaker.utils.utils import gen_relation_onehot


def _reference(indexid2vec, etype2oh, ntype2oh, graph):
    """The pre-S4 implementation, kept here as the oracle."""
    src, dst, msg, t, y = [], [], [], [], []
    for u, v, k, attr in graph.edges(data=True, keys=True):
        src.append(int(u))
        dst.append(int(v))
        t.append(int(attr["time"]))
        y.append(int(attr.get("y", 0)))

        if "label" in attr:
            edge_label = etype2oh[attr["label"]]
        else:
            edge_label = torch.zeros_like(etype2oh[list(etype2oh.keys())[0]])

        if indexid2vec is None:
            msg.append(torch.cat([
                ntype2oh[graph.nodes[u]["node_type"]],
                edge_label,
                ntype2oh[graph.nodes[v]["node_type"]],
            ]))
        else:
            msg.append(torch.cat([
                ntype2oh[graph.nodes[u]["node_type"]],
                torch.from_numpy(indexid2vec[u]),
                edge_label,
                ntype2oh[graph.nodes[v]["node_type"]],
                torch.from_numpy(indexid2vec[v]),
            ]))

    return {
        "src": torch.tensor(src).to(torch.long),
        "dst": torch.tensor(dst).to(torch.long),
        "t": torch.tensor(t).to(torch.long),
        "msg": torch.vstack(msg).to(torch.float),
        "y": torch.tensor(y).to(torch.long),
    }


def _build(n_nodes=12, n_edges=40, seed=0, with_labels=True):
    rng = np.random.default_rng(seed)
    rel2id = {"read": 1, "write": 2, "execute": 3, 1: "read", 2: "write", 3: "execute"}
    ntype2id = {"file": 1, "process": 2, 1: "file", 2: "process"}
    etype2oh = gen_relation_onehot(rel2id)
    ntype2oh = gen_relation_onehot(ntype2id)

    g = nx.MultiDiGraph()
    for n in range(n_nodes):
        g.add_node(n, node_type="file" if n % 2 else "process")
    labels = ["read", "write", "execute"]
    for i in range(n_edges):
        u, v = int(rng.integers(n_nodes)), int(rng.integers(n_nodes))
        attr = {"time": 1000 + i, "y": int(i % 7 == 0)}
        if with_labels:
            attr["label"] = labels[i % len(labels)]
        g.add_edge(u, v, **attr)

    indexid2vec = {n: rng.normal(size=8).astype(np.float32) for n in range(n_nodes)}
    return g, etype2oh, ntype2oh, indexid2vec


@pytest.mark.parametrize("use_embeddings", [True, False])
@pytest.mark.parametrize("with_labels", [True, False])
def test_matches_reference(tmp_path, use_embeddings, with_labels):
    g, etype2oh, ntype2oh, indexid2vec = _build(with_labels=with_labels)
    vecs = indexid2vec if use_embeddings else None

    graph_path = tmp_path / "graph.pt"
    torch.save(g, graph_path)
    out_dir = tmp_path / "out"

    feat_inference(vecs, etype2oh, ntype2oh, [str(graph_path)], str(out_dir), cfg=None)
    got = torch.load(out_dir / "graph.pt.TemporalData.simple", weights_only=False)
    want = _reference(vecs, etype2oh, ntype2oh, g)

    for field in ("src", "dst", "t", "y"):
        assert torch.equal(getattr(got, field), want[field]), field
    assert got.msg.shape == want["msg"].shape
    assert got.msg.dtype == want["msg"].dtype == torch.float32
    assert torch.equal(got.msg, want["msg"]), "message matrix differs from the reference"


def test_empty_graph_keeps_the_column_layout(tmp_path):
    """A time window with no edges must still yield the right msg width, or concatenating it with
    other windows downstream would fail on a shape mismatch."""
    g, etype2oh, ntype2oh, indexid2vec = _build()
    empty = nx.MultiDiGraph()
    graph_path = tmp_path / "empty.pt"
    torch.save(empty, graph_path)
    out_dir = tmp_path / "out"

    feat_inference(indexid2vec, etype2oh, ntype2oh, [str(graph_path)], str(out_dir), cfg=None)
    got = torch.load(out_dir / "empty.pt.TemporalData.simple", weights_only=False)

    n_dim = len(next(v for v in ntype2oh.values() if torch.is_tensor(v)))
    e_dim = len(next(v for v in etype2oh.values() if torch.is_tensor(v)))
    assert got.msg.shape == (0, 2 * n_dim + e_dim + 2 * 8)
