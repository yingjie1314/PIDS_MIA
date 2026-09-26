#!/usr/bin/env python
r"""Multi-hop re-query of a trained v2 detector, for the He et al. / Conti et al. attack ports.

WHY THIS EXISTS. `ours_v2.py` covers only the white-box attack. The other two methods in the
benchmark are both built on the same mechanism -- query the target for node v while varying how
much of v's neighbourhood it may see, and read membership off how the answer changes:

  He et al. 2021      per-node answer under 0-hop / 1-hop queries, fed to an attack classifier.
  Conti et al. 2022   the same ladder, but the adversary sees only a hard flagged/not-flagged bit,
                      so they distil a surrogate they can read a probability from.

Neither can be scored from the training dumps, which record the 1-hop (deployed) answer only. The
0-hop rung has to be re-queried from the saved checkpoint, which is what this produces.

DIFFERENCE FROM `requery_hops.py` (no-frac). That script hardcodes nodlink at its `sys.argv` line
and inlines one arm's flag list. Here the detector and arm are arguments and the flags are replayed
from the campaign's own `run_flags_<DATASET>.json`, which `run_arm.sh` wrote at training time for
exactly this purpose ("any later re-query must replay the exact same list or it silently resolves
to a different (empty) artifact tree").

THAT FAILURE IS THE MAIN HAZARD, so it is gated rather than trusted: PIDSMaker folds runtime args
into the artifact hash, and a mismatch does not raise -- it resolves to a fresh empty tree and
writes a plausible-looking dump against a randomly initialised model. `check_resolved` refuses to
run unless the resolved directory is the one already holding this arm's training dumps.

NODE DETECTORS ONLY. Hop 0 replaces the edge set with self-loops. An edge detector scores edges, so
hop 0 leaves it with nothing to score and the ladder is undefined -- velox / orthrus / kairos cannot
be covered by these two methods at all, and are skipped rather than approximated.

Hop 2a from the no-frac script is deliberately not ported: it was a bonus surface there, it blew the
GPU budget on the largest windows, and `REQUIRED_HOPS` in the scorer is ("0", "1").
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/yingjie/PIDSMaker")

from pidsmaker.config import get_runtime_required_args, get_yml_cfg  # noqa: E402
from pidsmaker.factory import build_model  # noqa: E402
from pidsmaker.tasks.batching import get_preprocessed_graphs  # noqa: E402
from pidsmaker.utils.data_utils import load_model  # noqa: E402

CAMPAIGN = "/data/pids_v2_shadow/clearscope_e3"
NODE_DETECTORS = ("nodlink", "magic", "flash", "threatrace", "rcaid")
EDGE_DETECTORS = ("velox", "orthrus_non_snooped", "kairos")
SPLITS = ("mia_train_in", "mia_heldout")


def replay_flags(detector: str, dataset: str) -> tuple[list[str], str]:
    p = os.path.join(CAMPAIGN, detector, f"run_flags_{dataset}.json")
    if not os.path.exists(p):
        raise SystemExit(f"ABORT: no recorded flags at {p}; cannot reproduce the artifact hash.")
    rec = json.load(open(p))
    return list(rec["flags"]), rec.get("dump_epochs", "")


def training_dump_dir(detector: str, dataset: str) -> str | None:
    """Where this arm's training dumps actually live, found by search rather than by hash."""
    root = os.path.join(CAMPAIGN, detector, "artifacts", "training", "training")
    if not os.path.isdir(root):
        return None
    hits = [os.path.join(root, h, dataset, "edge_losses")
            for h in os.listdir(root)
            if os.path.isdir(os.path.join(root, h, dataset, "edge_losses"))]
    return hits[0] if hits else None


def check_resolved(cfg, detector: str, dataset: str) -> str:
    """Refuse to proceed unless the config hash landed on the trained arm's own directory."""
    resolved = os.path.realpath(cfg.training._edge_losses_dir)
    expected = training_dump_dir(detector, dataset)
    if expected is None:
        raise SystemExit(f"ABORT: found no training dumps for {detector}/{dataset}.")
    if resolved != os.path.realpath(expected):
        raise SystemExit(
            "ABORT: the replayed flags resolve to a DIFFERENT artifact tree than the one this arm "
            "was trained in. Re-querying here would load a randomly initialised model and write a "
            f"dump that looks fine and means nothing.\n  resolved: {resolved}\n  expected: {expected}")
    return resolved


class _ZeroNoise:
    def __init__(self, ref):
        self.device = ref.loc.device

    def sample(self, shape):
        return torch.zeros(shape, device=self.device)


def hop_edge_index(data, hop: str, device):
    n = data.x.size(0)
    if hop == "0":
        idx = torch.arange(n, device=device)
        return torch.stack([idx, idx])
    if hop == "1":
        return data.edge_index
    raise ValueError(f"unknown hop {hop!r}; only 0 and 1 are part of the ladder")


@torch.no_grad()
def query(model, data, edge_index, samples: int) -> np.ndarray:
    """Mean per-node loss over `samples` passes, with edge_index temporarily replaced.

    Averaging is not cosmetic: NodLink's decoder is a VAE that resamples every forward pass even
    under eval, so a single pass disagrees with itself. A real adversary can average too -- it costs
    only repeated queries. Deterministic detectors are unaffected.
    """
    original = data.edge_index
    try:
        data.edge_index = edge_index
        acc = None
        for _ in range(samples):
            loss = model(data, inference=True, validation=False)["loss"].detach().float().cpu()
            acc = loss if acc is None else acc + loss
        return (acc / samples).numpy()
    finally:
        data.edge_index = original


def node_ids(data) -> np.ndarray:
    n_id = getattr(data, "original_n_id_tgn", getattr(data, "original_n_id"))
    return n_id.detach().cpu().numpy()


def iter_windows(datasets):
    for ds in datasets:
        for g in ds:
            yield g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--arm", required=True, choices=("TARGET", "SHADOW"))
    ap.add_argument("--epochs", default="49,99,149,199,299")
    ap.add_argument("--hops", default="0,1")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="windows per split; 0 = all")
    ap.add_argument("--dump-x", action="store_true",
                    help="write per-node input features and exit; no checkpoint is read")
    a = ap.parse_args()

    if a.detector in EDGE_DETECTORS:
        raise SystemExit(
            f"ABORT: {a.detector} scores edges. Hop 0 replaces the edge set with self-loops, which "
            "leaves an edge detector with nothing to score, so the He/Conti ladder is undefined "
            "for it. Not approximable -- report these two methods as N/A for edge detectors.")
    if a.detector not in NODE_DETECTORS:
        raise SystemExit(f"ABORT: unknown detector {a.detector!r}")

    dataset = f"CLEARSCOPE_E3_V2_{a.arm}"
    flags, dump_epochs = replay_flags(a.detector, dataset)
    if dump_epochs:
        os.environ.setdefault("PIDS_MIA_DUMP_EPOCHS", dump_epochs)

    sys.argv = ["x", a.detector, dataset] + flags
    cfg = get_yml_cfg(get_runtime_required_args())
    resolved = check_resolved(cfg, a.detector, dataset)
    print(f"[ok] hash resolves to the trained arm: {resolved}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data, _, max_node_num = get_preprocessed_graphs(cfg)
    splits = {"mia_train_in": list(train_data), "mia_heldout": list(val_data)}
    out_root = os.path.join(resolved, "mia_hops")

    if a.dump_x:
        # The raw input vector per node: a property of the provenance graph, not of any model, so
        # one pass serves every epoch and both arms. Conti's surrogate distils on this.
        d = os.path.join(out_root, "node_x")
        os.makedirs(d, exist_ok=True)
        for split, datasets in splits.items():
            n = 0
            for g in iter_windows(datasets):
                np.savez_compressed(os.path.join(d, f"{split}_w{n:05d}.npz"),
                                    node=node_ids(g),
                                    x=g.x.detach().cpu().numpy().astype(np.float32))
                n += 1
            print(f"node_x {split}: {n} windows -> {d}", flush=True)
        return

    model = build_model(data_sample=train_data[0][0], device=device, cfg=cfg,
                        max_node_num=max_node_num)

    for epoch in (int(e) for e in a.epochs.split(",") if e):
        ckpt = os.path.join(cfg.training._trained_models_dir, f"model_epoch_{epoch}")
        if not os.path.exists(ckpt):
            print(f"[skip] no checkpoint {ckpt}", flush=True)
            continue
        load_model(model, ckpt, cfg)
        model.to_device(device)
        model.eval()
        model.reset_state()
        print(f"loaded {ckpt}", flush=True)

        for hop in [h for h in a.hops.split(",") if h]:
            for split, datasets in splits.items():
                d = os.path.join(out_root, f"hop{hop}", split, f"model_epoch_{epoch}")
                os.makedirs(d, exist_ok=True)
                n = 0
                for g in iter_windows(datasets):
                    if a.limit and n >= a.limit:
                        break
                    g.to(device=device)
                    nid = node_ids(g)
                    loss = query(model, g, hop_edge_index(g, hop, device), a.samples)
                    if len(loss) != len(nid):
                        raise SystemExit(
                            f"ABORT: {a.detector} returned {len(loss)} loss values for {len(nid)} "
                            "nodes -- its loss is not per-node, so these rows cannot be keyed by "
                            "node and the ladder is not portable to it.")
                    np.savez_compressed(os.path.join(d, f"w{n:05d}.npz"), node=nid, loss=loss)
                    g.to("cpu")
                    n += 1
                print(f"  e{epoch} hop{hop} {split}: {n} windows -> {d}", flush=True)


if __name__ == "__main__":
    main()
