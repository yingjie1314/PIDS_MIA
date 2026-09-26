import os

import numpy as np
import torch

from pidsmaker.config import update_cfg_for_multi_dataset
from pidsmaker.featurization.feat_inference_methods import (
    feat_inference_alacarte,
    feat_inference_doc2vec,
    feat_inference_fasttext,
    feat_inference_flash,
    feat_inference_HFH,
    feat_inference_TRW,
    feat_inference_word2vec,
)
from pidsmaker.utils.data_utils import CollatableTemporalData
from pidsmaker.utils.dataset_utils import get_node_map, get_rel2id
from pidsmaker.utils.utils import (
    gen_relation_onehot,
    get_multi_datasets,
    get_split_to_files,
    log_tqdm,
)


def feat_inference(indexid2vec, etype2oh, ntype2oh, sorted_paths, out_dir, cfg):
    # The message matrix is preallocated and filled row by row rather than built as a Python list
    # of one small tensor per edge. On the large graphs (R-CAID on E3-THEIA) the list form was the
    # OOM: a per-edge torch tensor costs several hundred bytes of object overhead on top of its
    # payload, and `torch.vstack` then allocates the whole matrix a second time while the list is
    # still alive. One preallocated array holds exactly the payload, once.
    # `gen_relation_onehot` returns a BIDIRECTIONAL map: label -> one-hot and one-hot -> label. Only
    # the forward direction is a lookup table, so the reverse entries are filtered out here.
    def _forward_only(table):
        return {k: v.numpy().astype(np.float32)
                for k, v in table.items() if torch.is_tensor(v)}

    etype_np = _forward_only(etype2oh)
    ntype_np = _forward_only(ntype2oh)

    e_dim = len(next(iter(etype_np.values())))
    n_dim = len(next(iter(ntype_np.values())))
    zero_etype = np.zeros(e_dim, dtype=np.float32)
    emb_dim = len(next(iter(indexid2vec.values()))) if indexid2vec else 0
    msg_dim = 2 * n_dim + e_dim + 2 * emb_dim

    # Column layout, kept byte-identical to the previous concatenation order so existing artifacts
    # and trained models stay comparable.
    if indexid2vec is None:
        off_ntype_u, off_etype = 0, n_dim
        off_ntype_v = off_etype + e_dim
        off_emb_u = off_emb_v = None
    else:
        off_ntype_u = 0
        off_emb_u = n_dim
        off_etype = off_emb_u + emb_dim
        off_ntype_v = off_etype + e_dim
        off_emb_v = off_ntype_v + n_dim

    for path in log_tqdm(sorted_paths, desc="Computing edge embeddings"):
        graph = torch.load(path)
        n_edges = graph.number_of_edges()

        src = np.empty(n_edges, dtype=np.int64)
        dst = np.empty(n_edges, dtype=np.int64)
        t = np.empty(n_edges, dtype=np.int64)
        y = np.empty(n_edges, dtype=np.int64)
        msg = np.zeros((n_edges, msg_dim), dtype=np.float32)

        for i, (u, v, k, attr) in enumerate(graph.edges(data=True, keys=True)):
            src[i] = int(u)
            dst[i] = int(v)
            t[i] = int(attr["time"])
            y[i] = int(attr.get("y", 0))

            # If the graph structure has been changed in transformation, we may loose
            # the edge label
            edge_label = etype_np[attr["label"]] if "label" in attr else zero_etype

            row = msg[i]
            row[off_ntype_u:off_ntype_u + n_dim] = ntype_np[graph.nodes[u]["node_type"]]
            row[off_etype:off_etype + e_dim] = edge_label
            row[off_ntype_v:off_ntype_v + n_dim] = ntype_np[graph.nodes[v]["node_type"]]
            if indexid2vec is not None:
                row[off_emb_u:off_emb_u + emb_dim] = indexid2vec[u]
                row[off_emb_v:off_emb_v + emb_dim] = indexid2vec[v]

        data = CollatableTemporalData(
            src=torch.from_numpy(src),
            dst=torch.from_numpy(dst),
            t=torch.from_numpy(t),
            msg=torch.from_numpy(msg),
            y=torch.from_numpy(y),
        )

        os.makedirs(out_dir, exist_ok=True)
        file = path.split("/")[-1]
        torch.save(data, os.path.join(out_dir, f"{file}.TemporalData.simple"))
        del data, msg, src, dst, t, y, graph


def get_indexid2vec(cfg):
    method = cfg.featurization.used_method.strip()
    if method in ["only_type", "only_ones"]:
        return None
    if method == "alacarte":
        return feat_inference_alacarte.main(cfg)
    if method == "doc2vec":
        return feat_inference_doc2vec.main(cfg)
    if method == "hierarchical_hashing":
        return feat_inference_HFH.main(cfg)
    if method == "word2vec":
        return feat_inference_word2vec.main(cfg)
    if method == "temporal_rw":
        return feat_inference_TRW.main(cfg)
    if method == "flash":
        return feat_inference_flash.main(cfg)
    if method == "fasttext":
        return feat_inference_fasttext.main(cfg)

    raise ValueError(f"Invalid node embedding method {method}")


def main_from_config(cfg):
    rel2id = get_rel2id(cfg)
    ntype2id = get_node_map()
    etype2onehot = gen_relation_onehot(rel2id=rel2id)
    ntype2onehot = gen_relation_onehot(rel2id=ntype2id)

    base_dir = cfg.transformation._graphs_dir
    split_to_files = get_split_to_files(cfg, base_dir)

    # Here we get a mapping {node_id => embedding vector}
    indexid2vec = get_indexid2vec(cfg)

    # Create edges for Train, Val, Test sets
    for split, sorted_paths in split_to_files.items():
        feat_inference(
            indexid2vec=indexid2vec,
            etype2oh=etype2onehot,
            ntype2oh=ntype2onehot,
            sorted_paths=sorted_paths,
            out_dir=os.path.join(cfg.feat_inference._edge_embeds_dir, f"{split}/"),
            cfg=cfg,
        )


def main(cfg):
    multi_dataset_training = cfg.batching.multi_dataset_training
    if not multi_dataset_training:
        main_from_config(cfg)

    # Multi-dataset mode
    else:
        trained_model_dir = cfg.featurization._model_dir
        multi_datasets = get_multi_datasets(cfg)
        for dataset in multi_datasets:
            updated_cfg, should_restart = update_cfg_for_multi_dataset(cfg, dataset)
            updated_cfg.featurization._model_dir = trained_model_dir

            if should_restart["feat_inference"]:
                main_from_config(updated_cfg)
