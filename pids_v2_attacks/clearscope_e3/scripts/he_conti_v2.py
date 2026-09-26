#!/usr/bin/env python
"""He et al. 2021 and Conti et al. 2022 node-level MIAs on the v2 dumps.

Completes the three-method benchmark: `ours_v2.py` covers the white-box attack, this covers the two
published ones. All three are scored on the SAME rows with the SAME classifier, so a difference
between columns is a difference between attacks and not between harnesses.

PROTOCOL, AND WHY IT DIFFERS FROM THE NO-FRAC RUN. In `compare_attacks.py` He is scored by fitting
on the shadow and predicting the target -- the published threat model, valid there because the two
no-frac arms are exactly role-swapped (measured label agreement 0.0000 over 304,885 shared nodes).
v2 is NOT role-swapped: its arms train on disjoint day RANGES, so a node seen on both 04-02 and
04-07 is a member for both models. Measured agreement 0.6482 with 0.227 coverage. At intermediate
agreement a transfer row has no defined sign yet still reads high and looks like a strong leak -- a
synthetic replication at 0.65 scored 0.838 against 0.940 for a clean aligned design and 0.201 for a
clean swapped one. `compare_attacks.py` now aborts rather than print that table.

So He is scored here by within-arm cross-validation, the same upper bound `ours_v2.py` reports. It
is NOT a mountable attack -- the classifier sees the arm's own labels -- and that is exactly what
makes it a usable gate: if the upper bound sits at the structure floor, no weaker and more realistic
attack can beat it, and the question is closed.

CONTI NEEDS NO SUCH CHANGE. Its score is a threshold statistic, not a fitted classifier, so the only
adversary choice is the flag rate. That is selected on the OTHER arm exactly as the paper selects it
on the shadow: the arm being scored never participates in choosing its own threshold. Both v2 arms
are independent models, so the cross-selection is clean and this column stays mountable.

ROW ALIGNMENT. Seeded by (detector, arm) and not by epoch, and the member subsample is drawn before
the non-member one -- identical to `ours_v2.py`, so the same nodes are scored and the columns can be
read across.
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
from ours_v2 import MAX_PER_CLASS, SPLITS, run_dir  # noqa: E402

OUT = "/data/pids_v2_attacks/clearscope_e3/results/he_conti_v2.json"
DETECTORS = ("nodlink", "magic", "flash", "threatrace", "rcaid")  # node detectors only; CLEARSCOPE ran rcaid
ARMS = ("TARGET", "SHADOW")
CKPTS = (49, 99, 149, 199, 299)
HOPS = ("0", "1")
Q_GRID = (0.01, 0.05, 0.10, 0.25)
BATCH = 40


def _mean_by_node(files, key: str, uniq=None):
    """Per-node mean of `key` over the given npz windows, reduced in batches."""
    if uniq is None:
        ids = []
        for p in files:
            z = np.load(p)
            if len(z["node"]):
                ids.append(z["node"].astype(np.int64))
        if not ids:
            return None, None
        uniq = np.unique(np.concatenate(ids))
    n, dim, acc, cnt = len(uniq), None, None, np.zeros(n := len(uniq))
    for i in range(0, len(files), BATCH):
        nb, vb = [], []
        for p in files[i:i + BATCH]:
            z = np.load(p)
            if not len(z["node"]):
                continue
            nb.append(z["node"].astype(np.int64))
            v = z[key].astype(np.float64)
            vb.append(v.reshape(len(v), -1))
        if not nb:
            continue
        idx = np.searchsorted(uniq, np.concatenate(nb))
        vb = np.vstack(vb)
        if acc is None:
            dim = vb.shape[1]
            acc = np.zeros((n, dim))
        cnt += np.bincount(idx, minlength=n)
        for j in range(dim):
            acc[:, j] += np.bincount(idx, weights=vb[:, j], minlength=n)
    if acc is None:
        return None, None
    return uniq, acc / np.maximum(cnt, 1.0)[:, None]


def _geom(split_dir: str):
    """(nodes, [degree, node_type]) from the training dump -- Conti's surrogate inputs."""
    files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
    deg, nt = {}, {}
    for p in files:
        z = np.load(p)
        if "node" not in z.files or not len(z["node"]):
            continue
        nodes = z["node"].astype(np.int64)
        d = np.zeros(len(nodes))
        if "edge_index" in z.files and z["edge_index"].size:
            ei = z["edge_index"].astype(np.int64).ravel()
            ei = ei[(ei >= 0) & (ei < len(nodes))]
            if len(ei):
                d = np.bincount(ei, minlength=len(nodes)).astype(float)
        t = z["node_type"].astype(float) if "node_type" in z.files else np.zeros(len(nodes))
        for i, node in enumerate(nodes):
            k = int(node)
            deg[k] = deg.get(k, 0.0) + float(d[i])
            nt[k] = float(t[i])
    return deg, nt


def surfaces(detector: str, arm: str, epoch: int):
    """One arm's rows: labels, the hop ladder, raw features and geometry."""
    base = run_dir(detector, f"CLEARSCOPE_E3_V2_{arm}")
    if not base:
        return None
    hops_root = os.path.join(base, "mia_hops")
    per = {}
    for split in SPLITS:
        got = {}
        for h in HOPS:
            d = os.path.join(hops_root, f"hop{h}", split, f"model_epoch_{epoch}")
            files = sorted(glob.glob(os.path.join(d, "*.npz")))
            if not files:
                return None
            nodes, v = _mean_by_node(files, "loss")
            if nodes is None:
                return None
            got[h] = (nodes, v[:, 0])
        xf = sorted(glob.glob(os.path.join(hops_root, "node_x", f"{split}_w*.npz")))
        if not xf:
            return None
        xn, xv = _mean_by_node(xf, "x")
        deg, nt = _geom(os.path.join(base, split, f"model_epoch_{epoch}"))

        # A node counts only if every surface covers it; otherwise the ladder has a hole and the
        # row would silently compare a real query against a missing one.
        keys = got[HOPS[0]][0]
        for h in HOPS[1:]:
            keys = np.intersect1d(keys, got[h][0])
        keys = np.intersect1d(keys, xn)
        keys = keys[np.isin(keys, np.fromiter(deg.keys(), dtype=np.int64))] if deg else keys
        if not len(keys):
            return None
        hop_v = {h: got[h][1][np.searchsorted(got[h][0], keys)] for h in HOPS}
        x_v = xv[np.searchsorted(xn, keys)]
        g_v = np.stack([np.array([deg.get(int(k), 0.0) for k in keys]),
                        np.array([nt.get(int(k), 0.0) for k in keys])], axis=1)
        per[split] = (keys, hop_v, x_v, g_v)
    return per


def build_run(detector: str, arm: str, epoch: int):
    rng = np.random.default_rng(zlib.crc32(f"{detector}|{arm}".encode()))
    per = surfaces(detector, arm, epoch)
    if per is None:
        return None
    (km, hm, xm, gm), (kn, hn, xn_, gn) = per["mia_train_in"], per["mia_heldout"]
    overlap = np.intersect1d(km, kn)
    keep_m, keep_n = ~np.isin(km, overlap), ~np.isin(kn, overlap)
    km, kn = km[keep_m], kn[keep_n]
    hm = {h: v[keep_m] for h, v in hm.items()}
    hn = {h: v[keep_n] for h, v in hn.items()}
    xm, gm, xn_, gn = xm[keep_m], gm[keep_m], xn_[keep_n], gn[keep_n]
    if len(km) < 50 or len(kn) < 50:
        return None

    # Member subsample drawn first, then non-member -- same rng order as ours_v2.cell.
    im = rng.choice(len(km), MAX_PER_CLASS, replace=False) if len(km) > MAX_PER_CLASS \
        else np.arange(len(km))
    in_ = rng.choice(len(kn), MAX_PER_CLASS, replace=False) if len(kn) > MAX_PER_CLASS \
        else np.arange(len(kn))

    return {
        "y": np.concatenate([np.ones(len(im)), np.zeros(len(in_))]).astype(int),
        "hops": {h: np.concatenate([hm[h][im], hn[h][in_]]) for h in HOPS},
        "x": np.vstack([xm[im], xn_[in_]]),
        "geom": np.vstack([gm[im], gn[in_]]),
        "n_members": int(len(im)), "n_nonmembers": int(len(in_)),
        "n_overlap_dropped": int(len(overlap)),
    }


def metrics(y, s) -> dict:
    fpr, tpr, _ = roc_curve(y, s)
    return {"auc": float(roc_auc_score(y, s)),
            "bal_acc": float((1.0 + float(np.max(tpr - fpr))) / 2.0),
            "tpr_at_1": float(np.interp(0.01, fpr, tpr))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", default=",".join(DETECTORS))
    ap.add_argument("--epochs", default=",".join(str(e) for e in CKPTS))
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    rows = json.load(open(a.out)) if os.path.exists(a.out) else {}
    t0 = time.time()
    for det in a.detectors.split(","):
        for ep in (int(e) for e in a.epochs.split(",")):
            if all(f"{det}|{arm}|{ep}" in rows for arm in ARMS):
                continue
            t = time.time()
            runs = {}
            for arm in ARMS:
                try:
                    runs[arm] = build_run(det, arm, ep)
                except Exception as e:
                    print(f"  {det}|{arm}|{ep:<5} ERROR {e}", flush=True)
                    runs[arm] = None
            if any(v is None for v in runs.values()):
                print(f"  {det}|*|{ep:<5} incomplete -- need both arms to cross-select the flag "
                      "rate; skipping", flush=True)
                continue

            # Flag rate chosen on the OTHER arm, so the scored arm never sets its own threshold.
            q_for = {}
            for arm in ARMS:
                other = runs["SHADOW" if arm == "TARGET" else "TARGET"]
                sweep = [(roc_auc_score(other["y"], C.conti_label_only(other, q)), q)
                         for q in Q_GRID]
                q_for[arm] = max(sweep)[1]

            for arm in ARMS:
                r, q = runs[arm], q_for[arm]
                y = r["y"]
                rows[f"{det}|{arm}|{ep}"] = {
                    "n_members": r["n_members"], "n_nonmembers": r["n_nonmembers"],
                    "n_overlap_dropped": r["n_overlap_dropped"], "flag_rate": q,
                    "he2021": metrics(y, C._within_cv(C.he_features(r, HOPS), y)),
                    "conti2022_label_only": metrics(y, C.conti_label_only(r, q)),
                    "conti2022_distilled": metrics(y, C.conti_distilled(r, q)),
                }
                m = rows[f"{det}|{arm}|{ep}"]
                print(f"  {det}|{arm}|{ep:<5} he {m['he2021']['auc']:.4f} | "
                      f"conti-lo {m['conti2022_label_only']['auc']:.4f} | "
                      f"conti-dist {m['conti2022_distilled']['auc']:.4f} | q={q:g} "
                      f"n={r['n_members']}+{r['n_nonmembers']} [{time.time()-t:.0f}s]", flush=True)
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            json.dump(rows, open(a.out, "w"), indent=1)
    print(f"{len(rows)} cells in {time.time()-t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
