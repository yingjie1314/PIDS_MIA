"""The low-memory loading path (PIDS_LOW_MEMORY=1) must hand the training loop the same windows
as the eager `load_all_datasets` pipeline, minus exactly the three attributes nothing reads after
reindexing in a non-TGN run: `msg`, `x_src`, `x_dst`.

Both paths run here on CPU against the same synthetic feat_inference artifact tree, and every
surviving tensor of every window is compared exactly. CPU matters: eager reindexing normally runs
on the training device, and CUDA scatter-mean is nondeterministic-atomics; on CPU both paths are
deterministic, so equality is exact rather than approximate.
"""

import os
import sys
from types import SimpleNamespace as NS

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pidsmaker.factory import get_dimensions_from_data_sample
from pidsmaker.utils.data_utils import (
    CollatableTemporalData,
    load_all_datasets,
)

EMB_DIM = 4
N_TYPES = 3
E_TYPES = 5
MSG_LEN = 2 * EMB_DIM + 2 * N_TYPES + E_TYPES


def make_cfg(edge_embeds_dir, encoder="rcaid_gat"):
    return NS(
        featurization=NS(emb_dim=EMB_DIM, used_method="doc2vec"),
        dataset=NS(num_node_types=N_TYPES, num_edge_types=E_TYPES, name="CADETS_E3"),
        batching=NS(
            node_features="node_emb",
            edge_features="none",
            fix_buggy_graph_reindexer=False,
            multi_dataset_training=False,
            save_on_disk=False,
            global_batching=NS(
                used_method="none",
                global_batching_batch_size=None,
                global_batching_batch_size_inference=None,
            ),
            intra_graph_batching=NS(used_methods="none"),
            inter_graph_batching=NS(used_method="none"),
        ),
        training=NS(
            encoder=NS(used_methods=encoder, x_is_tuple=False),
            decoder=NS(used_methods="predict_node_type"),
        ),
        feat_inference=NS(_edge_embeds_dir=str(edge_embeds_dir)),
    )


def make_window(seed, n_edges=40, id_pool=30):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, id_pool, (n_edges,), generator=g)
    dst = torch.randint(0, id_pool, (n_edges,), generator=g)
    t = torch.sort(torch.randint(1_000, 9_000, (n_edges,), generator=g)).values

    def onehot(n_classes):
        return torch.nn.functional.one_hot(
            torch.randint(0, n_classes, (n_edges,), generator=g), n_classes
        ).float()

    msg = torch.cat(
        [
            onehot(N_TYPES),                                   # src_type
            torch.randn(n_edges, EMB_DIM, generator=g),        # src_emb
            onehot(E_TYPES),                                   # edge_type
            onehot(N_TYPES),                                   # dst_type
            torch.randn(n_edges, EMB_DIM, generator=g),        # dst_emb
        ],
        dim=1,
    )
    assert msg.shape[1] == MSG_LEN
    return CollatableTemporalData(
        src=src, dst=dst, t=t, msg=msg, y=torch.zeros(n_edges, dtype=torch.long)
    )


@pytest.fixture
def artifact_tree(tmp_path):
    base = tmp_path / "edge_embeds"
    seed = 0
    for split, n_windows in [("train", 3), ("val", 2), ("test", 2)]:
        d = base / split
        d.mkdir(parents=True)
        for i in range(n_windows):
            torch.save(make_window(seed), d / f"window_{i:02d}.TemporalData.simple")
            seed += 1
    return base


def load_both(artifact_tree, monkeypatch):
    """Load the same artifact tree through both paths.

    The eager reference needs one adjustment to be comparable on CPU: with
    `fix_buggy_graph_reindexer=False`, `node_features_reshape` returns `x` as a *view of the
    reindexer's shared cache buffer*. On a CUDA run (every production run) the subsequent
    `batch.to("cpu")` device transfer materialises that view into a private tensor, but on a
    CPU-only run `.to("cpu")` is a no-op and later windows overwrite earlier windows' `x` in
    place. Cloning after each window reproduces exactly what the device transfer does on CUDA.
    The low-memory path needs no such fix: it builds a fresh reindexer per window.
    """
    from pidsmaker.utils.data_utils import GraphReindexer

    cfg = make_cfg(artifact_tree)
    monkeypatch.delenv("PIDS_LOW_MEMORY", raising=False)
    original = GraphReindexer.reindex_graph

    def reindex_then_copy(self, data, **kwargs):
        data = original(self, data, **kwargs)
        data.x = data.x.clone()
        data.node_type = data.node_type.clone()
        return data

    with monkeypatch.context() as m:
        m.setattr(GraphReindexer, "reindex_graph", reindex_then_copy)
        eager = load_all_datasets(cfg, torch.device("cpu"))

    monkeypatch.setenv("PIDS_LOW_MEMORY", "1")
    slim = load_all_datasets(cfg, torch.device("cpu"))
    return eager, slim


KEPT_ATTRS = [
    "x", "edge_index", "src", "dst", "t", "y", "edge_type",
    "node_type", "node_type_src", "node_type_dst",
    "original_edge_index", "original_n_id",
]
DROPPED_ATTRS = ["msg", "x_src", "x_dst"]


def test_windows_match_eager_exactly(artifact_tree, monkeypatch):
    (e_train, e_val, e_test, e_max), (s_train, s_val, s_test, s_max) = load_both(
        artifact_tree, monkeypatch
    )
    assert e_max == s_max

    for e_split, s_split in [(e_train, s_train), (e_val, s_val), (e_test, s_test)]:
        assert len(e_split) == len(s_split) == 1
        assert len(e_split[0]) == len(s_split[0])
        for e_g, s_g in zip(e_split[0], s_split[0]):
            for attr in KEPT_ATTRS:
                assert torch.equal(getattr(e_g, attr), getattr(s_g, attr)), attr


def test_only_the_payload_attributes_are_dropped(artifact_tree, monkeypatch):
    (e_train, *_), (s_train, *_) = load_both(artifact_tree, monkeypatch)
    e_g, s_g = e_train[0][0], s_train[0][0]
    for attr in DROPPED_ATTRS:
        assert hasattr(e_g, attr), f"eager path lost {attr}"
        assert not hasattr(s_g, attr), f"low-memory path kept {attr}"


def test_model_input_dimension_is_unchanged(artifact_tree, monkeypatch):
    """build_model infers in_dim from x_src on eager windows and falls back to x on slim ones;
    both must agree or the encoder would be built with the wrong input layer."""
    (e_train, *_), (s_train, *_) = load_both(artifact_tree, monkeypatch)
    *_, e_in = get_dimensions_from_data_sample(e_train[0][0])
    *_, s_in = get_dimensions_from_data_sample(s_train[0][0])
    assert e_in == s_in == EMB_DIM


def test_encoder_forward_is_identical(artifact_tree, monkeypatch):
    from pidsmaker.encoders.rcaid_encoder import RCaidGAT

    (e_train, *_), (s_train, *_) = load_both(artifact_tree, monkeypatch)
    e_g, s_g = e_train[0][0], s_train[0][0]

    def run(g):
        torch.manual_seed(0)
        gat = RCaidGAT(in_dim=EMB_DIM, hid_dim=8, out_dim=N_TYPES, dropout=0.0)
        gat.eval()
        with torch.no_grad():
            return gat(g.x, g.edge_index)["h"]

    assert torch.equal(run(e_g), run(s_g))


def test_dropping_msg_actually_frees_its_storage(artifact_tree, monkeypatch):
    """node_type_src/dst and edge_type come out of extraction as slices of msg. If they stay
    views, deleting msg frees nothing: each view pins the whole (E, msg_len) buffer, which is
    the exact allocation this path exists to drop."""
    _, (s_train, *_) = load_both(artifact_tree, monkeypatch)
    for g in s_train[0]:
        for attr in ["node_type_src", "node_type_dst", "edge_type"]:
            t = getattr(g, attr)
            storage_bytes = t.storage().size() * t.element_size()
            assert storage_bytes == t.numel() * t.element_size(), (
                f"{attr} still views a larger buffer"
            )


def test_incompatible_config_is_rejected_loudly(artifact_tree, monkeypatch):
    monkeypatch.setenv("PIDS_LOW_MEMORY", "1")
    cfg = make_cfg(artifact_tree, encoder="tgn")
    with pytest.raises(ValueError, match="incompatible with low-memory loading"):
        load_all_datasets(cfg, torch.device("cpu"))


def test_flag_off_means_eager(artifact_tree, monkeypatch):
    monkeypatch.setenv("PIDS_LOW_MEMORY", "0")
    cfg = make_cfg(artifact_tree)
    train_data, *_ = load_all_datasets(cfg, torch.device("cpu"))
    assert hasattr(train_data[0][0], "msg")
