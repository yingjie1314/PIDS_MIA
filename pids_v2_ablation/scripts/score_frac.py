#!/usr/bin/env python
"""Ablation 2 -- score the random-split (fraction) runs.

Reuses ours_v2's pure helpers (`collect`, `metrics`, and the npz schema readers) so the rows are
built exactly as in the published table: per-node mean over every window the node appears in,
float64 accumulation, nodes present in both splits dropped. Only the path layout and the arm name
differ, which is why this re-implements `cell()` rather than importing it -- ours_v2.cell hardcodes
`CLEARSCOPE_E3_V2_{arm}`.

WHAT IS DIFFERENT ABOUT THESE RUNS, AND WHY IT MATTERS. Under `mia_split_mode=random` the members
and non-members are windows from the SAME days, so an attack can no longer win by recognising the
calendar. That makes these the first cells in the campaign where a lift is attributable to
membership rather than to day-distribution shift. Expect the overlap drop to be far larger than
under `xrole`: the same process recurs across adjacent windows, so many nodes land in both splits
and have no well-defined label. `n_overlap_dropped` is reported per cell for exactly that reason --
if it swallows most of the graph, the surviving rows are a biased remnant and the cell is not
comparable to the day-disjoint ones.
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

sys.path.insert(0, "/data/pids_v2_attacks/clearscope_e3/scripts")
import ours_v2 as O  # noqa: E402

sys.path.insert(0, "/data/pids_nofrac_tuned/cadets_e3/scripts")
import compare_attacks as C  # noqa: E402

ROOT = "/data/pids_v2_ablation"
ARMS = {"cadets_e3": "CADETS_E3_V2_TARGET", "theia_e3": "THEIA_E3_TARGET",
        "clearscope_e3": "CLEARSCOPE_E3_V2_TARGET"}
LADDER = (0, 1, 2, 5, 10, 24, 49, 99, 149, 199, 299)
SPLITS = ("mia_train_in", "mia_heldout")
MAX_PER_CLASS = 30_000
DET = "nodlink"


def run_dir(ds: str, frac: str) -> str | None:
    root = os.path.join(ROOT, ds, "frac", f"{DET}_f{frac}", "artifacts", "training", "training")
    if not os.path.isdir(root):
        return None
    arm = ARMS[ds]
    hits = [os.path.join(root, h, arm, "edge_losses") for h in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, h, arm, "edge_losses"))]
    if len(hits) > 1:
        raise SystemExit(f"{ds}/f{frac}: {len(hits)} artifact hashes resolve -- ambiguous, "
                         f"refusing to guess: {hits}")
    return hits[0] if hits else None


def cell(ds: str, frac: str, epoch: int) -> dict | None:
    # Seeded by (dataset, frac) and NOT by epoch, so every checkpoint of one cell is scored on the
    # same subsampled nodes and the epoch curve is not moving underneath the comparison.
    rng = np.random.default_rng(zlib.crc32(f"{ds}|{frac}".encode()))
    base = run_dir(ds, frac)
    if not base:
        return None
    per = {}
    for split in SPLITS:
        d = os.path.join(base, split, f"model_epoch_{epoch}")
        if not os.path.isdir(d):
            return None
        got = O.collect(d, DET)
        if got is None:
            return None
        per[split] = got

    (nm, lm, fm), (nn, ln, fn) = per["mia_train_in"], per["mia_heldout"]
    overlap = np.intersect1d(nm, nn)
    keep_m, keep_n = ~np.isin(nm, overlap), ~np.isin(nn, overlap)
    nm, lm, fm = nm[keep_m], lm[keep_m], fm[keep_m]
    nn, ln, fn = nn[keep_n], ln[keep_n], fn[keep_n]
    if len(nm) < 50 or len(nn) < 50:
        return None

    def sub(n, l, f):
        if len(n) <= MAX_PER_CLASS:
            return l, f
        i = rng.choice(len(n), MAX_PER_CLASS, replace=False)
        return l[i], f[i]

    lm, fm = sub(nm, lm, fm)
    ln, fn = sub(nn, ln, fn)

    y = np.concatenate([np.ones(len(lm)), np.zeros(len(ln))]).astype(int)
    loss = np.concatenate([lm, ln])
    feat = np.vstack([fm, fn])
    X = np.hstack([loss.reshape(-1, 1), np.linalg.norm(feat, axis=1, keepdims=True), feat])
    return {"n_members": int(len(lm)), "n_nonmembers": int(len(ln)),
            "n_overlap_dropped": int(len(overlap)), "feat_dim": int(feat.shape[1]),
            "single_loss": O.metrics(y, -loss),
            "ours": O.metrics(y, C._within_cv(X, y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(ARMS))
    ap.add_argument("--fracs", default="0.1,0.25,0.5,0.75")
    ap.add_argument("--epochs", default=",".join(str(e) for e in LADDER))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "frac_ours.json"))
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for ds in a.datasets.split(","):
        for frac in a.fracs.split(","):
            for e in (int(x) for x in a.epochs.split(",")):
                k = f"{ds}|{frac}|{e}"
                if k in res:
                    continue
                t0 = time.time()
                try:
                    c = cell(ds, frac, e)
                except Exception as exc:
                    print(f"  {k:<34} ERROR {exc}", flush=True)
                    continue
                if not c:
                    continue
                res[k] = c
                json.dump(res, open(a.out, "w"), indent=1)
                print(f"  {k:<34} single {c['single_loss']['auc']:.4f} | "
                      f"ours {c['ours']['auc']:.4f} / bACC {c['ours']['bal_acc']:.4f} | "
                      f"n={c['n_members']}+{c['n_nonmembers']} drop={c['n_overlap_dropped']} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
    print(f"{len(res)} cells -> {a.out}")


if __name__ == "__main__":
    main()
