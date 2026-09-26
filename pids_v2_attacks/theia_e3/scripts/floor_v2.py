#!/usr/bin/env python
"""Structure-only floor: how well does membership predict from NO model output whatsoever?

`ours_v2.py` shows the embedding attack reading 0.64-0.95 and sitting flat from epoch 0 to 299.
Flatness is strong evidence that nothing is being learned, but it is not airtight on its own --
`model_epoch_0` is dumped at the END of the first loop body (training_loop.py:296), so it is a
model after one epoch, not a randomly-initialised one. A single epoch is not nothing.

This closes that gap from the other side. It throws away every quantity the model produced -- no
loss, no embedding, no posterior -- and keeps only properties of the graph that exist before any
training at all: how many windows a node appears in, its degree, and its type. If membership is
predictable from those alone, then members and non-members are structurally different populations
and any attack that "succeeds" is reading the calendar rather than the weights.

Members are 04-02/03/04 and non-members are 04-05, so this is exactly the confound to fear: three
days of traffic against one, with whatever diurnal and workload differences that carries.

Rows use the same seed scheme as ours_v2.py, so the floor and the attack are scored on the same
nodes and the two numbers are directly comparable.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import zlib

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# cadets_e3 path is deliberate: only compare_attacks' pure scorers are used
# (_within_cv/_transfer/he_features/conti_*). Nothing here reads its C.ROOT.
sys.path.insert(0, "/data/pids_nofrac_tuned/cadets_e3/scripts")
import compare_attacks as C  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ours_v2 import EDGE, MAX_PER_CLASS, SPLITS, run_dir  # noqa: E402

OUT = "/data/pids_v2_attacks/theia_e3/results/floor_v2.json"
DETECTORS = ("nodlink", "magic", "flash", "rcaid", "threatrace",
             "velox", "orthrus_non_snooped", "kairos")
ARMS = ("TARGET", "SHADOW")


def structure(split_dir: str, detector: str):
    """(nodes, [touch, degree, node_type]) -- nothing here depends on the model's weights."""
    files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
    touch: dict[int, float] = {}
    deg: dict[int, float] = {}
    ntype: dict[int, float] = {}
    for p in files:
        z = np.load(p)
        if detector in EDGE:
            if "srcnode" not in z.files or not len(z["srcnode"]):
                continue
            ends = np.concatenate([z["srcnode"], z["dstnode"]]).astype(np.int64)
            for n, c in zip(*np.unique(ends, return_counts=True)):
                touch[int(n)] = touch.get(int(n), 0) + 1
                deg[int(n)] = deg.get(int(n), 0) + float(c)
            continue
        if "node" not in z.files or not len(z["node"]):
            continue
        nodes = z["node"].astype(np.int64)
        d = np.zeros(len(nodes))
        if "edge_index" in z.files and z["edge_index"].size:
            ei = z["edge_index"].astype(np.int64).ravel()
            ei = ei[(ei >= 0) & (ei < len(nodes))]
            if len(ei):
                d = np.bincount(ei, minlength=len(nodes)).astype(float)
        nt = z["node_type"].astype(float) if "node_type" in z.files else np.zeros(len(nodes))
        for i, n in enumerate(nodes):
            k = int(n)
            touch[k] = touch.get(k, 0) + 1
            deg[k] = deg.get(k, 0) + float(d[i])
            ntype[k] = float(nt[i])
    if not touch:
        return None
    keys = np.array(sorted(touch), dtype=np.int64)
    X = np.stack([np.array([touch[int(k)] for k in keys], dtype=float),
                  np.array([deg[int(k)] for k in keys], dtype=float),
                  np.array([ntype.get(int(k), 0.0) for k in keys], dtype=float)], axis=1)
    # mean degree per appearance: separates "busy node" from "node seen on many days"
    X = np.hstack([X, (X[:, 1] / np.maximum(X[:, 0], 1.0)).reshape(-1, 1)])
    return keys, X


def metrics(y, s) -> dict:
    fpr, tpr, _ = roc_curve(y, s)
    return {"auc": float(roc_auc_score(y, s)),
            "bal_acc": float((1.0 + float(np.max(tpr - fpr))) / 2.0),
            "tpr_at_1": float(np.interp(0.01, fpr, tpr))}


def cell(detector: str, arm: str, epoch: int) -> dict | None:
    rng = np.random.default_rng(zlib.crc32(f"{detector}|{arm}".encode()))
    base = run_dir(detector, f"THEIA_E3_{arm}")
    if not base:
        return None
    per = {}
    for split in SPLITS:
        d = os.path.join(base, split, f"model_epoch_{epoch}")
        if not os.path.isdir(d):
            return None
        got = structure(d, detector)
        if got is None:
            return None
        per[split] = got

    (nm, xm), (nn, xn) = per["mia_train_in"], per["mia_heldout"]
    overlap = np.intersect1d(nm, nn)
    nm, xm = nm[~np.isin(nm, overlap)], xm[~np.isin(nm, overlap)]
    nn, xn = nn[~np.isin(nn, overlap)], xn[~np.isin(nn, overlap)]
    if len(nm) < 50 or len(nn) < 50:
        return None

    def sub(n, x):
        if len(n) <= MAX_PER_CLASS:
            return x
        return x[rng.choice(len(n), MAX_PER_CLASS, replace=False)]

    xm, xn = sub(nm, xm), sub(nn, xn)
    y = np.concatenate([np.ones(len(xm)), np.zeros(len(xn))]).astype(int)
    X = np.vstack([xm, xn])
    return {"n_members": int(len(xm)), "n_nonmembers": int(len(xn)),
            "floor_structure": metrics(y, C._within_cv(X, y))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=299,
                    help="structure is epoch-independent; any dumped epoch gives the same graph")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    rows = json.load(open(a.out)) if os.path.exists(a.out) else {}
    t0 = time.time()
    for det in DETECTORS:
        for arm in ARMS:
            key = f"{det}|{arm}"
            if key in rows:
                continue
            t = time.time()
            r = cell(det, arm, a.epoch)
            if r is None:
                print(f"  {key:<34} no dump", flush=True)
                continue
            rows[key] = r
            print(f"  {key:<34} floor {r['floor_structure']['auc']:.4f} / "
                  f"bACC {r['floor_structure']['bal_acc']:.4f}  "
                  f"n={r['n_members']}+{r['n_nonmembers']} [{time.time()-t:.0f}s]", flush=True)
            json.dump(rows, open(a.out, "w"), indent=1)
    print(f"{len(rows)} cells in {time.time()-t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
