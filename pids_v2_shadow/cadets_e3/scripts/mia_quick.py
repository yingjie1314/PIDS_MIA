#!/usr/bin/env python
"""Single-loss node-level MIA on the v2 dumps, per arm, per dumped epoch.

WHAT THIS IS, AND WHAT IT IS NOT
    This scores each arm against ITS OWN membership labels, using only the per-node loss that
    training already dumped. That is the raw "signal" column and nothing more.

    It is NOT the controlled number the campaign is built to produce. DESIGN.md §3 defines

        signal = target scored against the target's labels   (members 04-02/03/04 vs non-member 04-05)
        null   = shadow scored against the TARGET's labels   (same nodes, model saw neither class)

    The null needs the shadow model RE-QUERIED over the target's nodes, because the shadow arm only
    ever dumps its own days (members 04-07/08/09, non-member 04-10). Replay
    `run_flags_CADETS_E3_V2_SHADOW.json` to build that pass. Until then the two columns printed here
    are two independent raw MIAs on disjoint day pairs, not an attack and its control.

    Reading the target column alone is the exact error the no-frac campaign was built to catch: its
    headline 0.91 AUC was matched at 0.93 by a model that had never seen the members.

Metric formulas are copied from `/data/pids_nofrac_multi/cadets_e3/scripts/sweep.py` so the numbers
are comparable across campaigns.
"""
from __future__ import annotations

import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

CAMPAIGN = "/data/pids_v2_shadow/cadets_e3"
ARMS = ("CADETS_E3_V2_TARGET", "CADETS_E3_V2_SHADOW")
SPLITS = {"mia_train_in": 1, "mia_heldout": 0}


def tpr_at_fpr(y, s, target_fpr=0.01) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    return float(tpr[max(idx, 0)])


def metrics(y, s) -> dict:
    y, s = np.asarray(y), np.asarray(s, dtype=np.float64)
    fpr, tpr, _ = roc_curve(y, s)
    return {"auc": float(roc_auc_score(y, s)),
            "bal_acc": float((1 + np.max(tpr - fpr)) / 2),
            "tpr_at_1": tpr_at_fpr(y, s, 0.01)}


def run_dir(detector: str, dataset: str) -> str:
    root = os.path.join(CAMPAIGN, detector, "artifacts", "training", "training")
    hits = [os.path.join(root, h, dataset, "edge_losses")
            for h in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, h, dataset, "edge_losses"))]
    if len(hits) != 1:
        raise RuntimeError(f"expected one run dir for {detector}/{dataset}, found {hits}")
    return hits[0]


def _pairs(z) -> tuple[np.ndarray, np.ndarray] | None:
    """(nodes, losses) from one dump, for either dump schema.

    Node detectors dump one row per node: keys `node`, `loss`.
    The TGN detectors (velox, orthrus, kairos) dump one row per EDGE instead: keys `srcnode`,
    `dstnode`, `loss`, and no `node` at all. v2 membership is defined per node by day, so an edge
    loss is credited to BOTH of its endpoints, and the per-node mean below then divides by that
    node's incident-edge count. Mean rather than sum matters: summing would make the score a proxy
    for degree, and the member days carry more edges than the single non-member day, which would
    manufacture separation out of split size alone.

    Note this is NOT the edge-membership convention of the earlier link-stealing campaign, which
    keyed on (src, dst, time, etype). These numbers are node-membership and are comparable to the
    five node detectors, not to that campaign's edge AUCs.
    """
    f = set(z.files)
    if "loss" not in f:
        return None
    if "node" in f:
        n = z["node"].astype(np.int64)
        return (n, z["loss"].astype(np.float64)) if len(n) else None
    if {"srcnode", "dstnode"} <= f:
        loss = z["loss"].astype(np.float64)
        if not len(loss):
            return None
        return (np.concatenate([z["srcnode"].astype(np.int64), z["dstnode"].astype(np.int64)]),
                np.concatenate([loss, loss]))
    return None


def mean_loss_by_node(d: str) -> dict:
    """{node: mean loss over every window/edge the node appears in}."""
    if not os.path.isdir(d):
        return {}
    nodes, losses = [], []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".npz"):
            continue
        got = _pairs(np.load(os.path.join(d, f), allow_pickle=True))
        if got is not None:
            nodes.append(got[0])
            losses.append(got[1])
    if not nodes:
        return {}
    # Vectorised: the edge dumps run to millions of rows per epoch, where a Python-level loop over
    # rows dominates the whole sweep.
    nodes, losses = np.concatenate(nodes), np.concatenate(losses)
    uniq, inv = np.unique(nodes, return_inverse=True)
    means = np.bincount(inv, weights=losses) / np.bincount(inv)
    return dict(zip(uniq.tolist(), means.tolist()))


def epochs_present(base: str) -> list[int]:
    d = os.path.join(base, "mia_train_in")
    if not os.path.isdir(d):
        return []
    return sorted(int(x.rsplit("_", 1)[1]) for x in os.listdir(d) if x.startswith("model_epoch_"))


def score(detector: str, dataset: str, epoch: int) -> dict | None:
    base = run_dir(detector, dataset)
    per = {s: mean_loss_by_node(os.path.join(base, s, f"model_epoch_{epoch}")) for s in SPLITS}
    members, nonmembers = per["mia_train_in"], per["mia_heldout"]
    if not members or not nonmembers:
        return None

    overlap = set(members) & set(nonmembers)
    # A node seen on both a member day and the non-member day has no well-defined label. DESIGN.md
    # §3 notes this is label noise that DEPRESSES measured AUC, so it never manufactures a leak.
    # Both numbers are reported so the effect is visible rather than assumed.
    # Memorisation means a member is reconstructed BETTER, i.e. lower loss, so the membership score
    # is the NEGATED loss (sweep.py:399 uses the same sign). Scoring raw loss inverts every AUC.
    y_all = [1] * len(members) + [0] * len(nonmembers)
    s_all = [-members[k] for k in members] + [-nonmembers[k] for k in nonmembers]

    m_dis = [k for k in members if k not in overlap]
    n_dis = [k for k in nonmembers if k not in overlap]
    out = {"n_members": len(members), "n_nonmembers": len(nonmembers), "n_overlap": len(overlap),
           "all": metrics(y_all, s_all)}
    if m_dis and n_dis:
        out["disjoint"] = metrics([1] * len(m_dis) + [0] * len(n_dis),
                                  [-members[k] for k in m_dis] + [-nonmembers[k] for k in n_dis])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--epochs", default="")
    args = ap.parse_args()

    print(f"=== {args.detector} | single-loss node MIA | higher loss => predicted NON-member ===")
    print("NOTE: each arm is scored against its OWN labels on its OWN days. This is NOT the")
    print("      controlled comparison -- that needs the shadow re-queried on the target's nodes.\n")

    for arm in ARMS:
        try:
            base = run_dir(args.detector, arm)
        except (RuntimeError, FileNotFoundError) as e:
            print(f"--- {arm}: not available ({e})\n")
            continue
        eps = ([int(x) for x in args.epochs.split(",")] if args.epochs else epochs_present(base))
        print(f"--- {arm}")
        print(f"    {'epoch':>6} {'AUC':>8} {'balACC':>8} {'TPR@1%':>8}   "
              f"{'AUC_dis':>8} {'balACC_d':>8}   n_mem/n_non/n_ovl")
        for e in eps:
            r = score(args.detector, arm, e)
            if r is None:
                print(f"    {e:>6}  (no dumps)")
                continue
            a, d = r["all"], r.get("disjoint")
            ds = (f"{d['auc']:>8.4f} {d['bal_acc']:>8.4f}" if d else f"{'-':>8} {'-':>8}")
            print(f"    {e:>6} {a['auc']:>8.4f} {a['bal_acc']:>8.4f} {a['tpr_at_1']:>8.4f}   {ds}   "
                  f"{r['n_members']}/{r['n_nonmembers']}/{r['n_overlap']}")
        print()


if __name__ == "__main__":
    main()
