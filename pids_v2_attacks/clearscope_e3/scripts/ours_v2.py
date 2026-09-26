#!/usr/bin/env python
"""'Ours' (white-box) node-level MIA on the v2 dumps -- loss PLUS the full node embedding.

The published v2 table (`mia_quick.py`) scores a single scalar: the per-node loss. That is the
weakest of the three methods. This adds the embedding the detector actually computed, which is
everything a white-box attacker holds, and fits it with within-arm cross-validation.

WHAT THIS NUMBER IS. An UPPER BOUND, not a mountable attack: the classifier is fitted and scored on
the same arm via 5-fold CV, so it sees that arm's own labels. No real adversary has them. The bound
is what makes it useful as a gate -- if the upper bound sits at chance, no weaker and more realistic
attack can do better, and the question is closed without spending anything on the neutral control.

WHAT IT CANNOT SETTLE. Membership is collinear with the calendar here (members 04-02/03/04,
non-members 04-05), so a lift is equally consistent with memorisation and with the embedding simply
encoding which day a node was seen on. Only a model that never trained on either group, scoring
these same nodes on these same days, separates those. That is the re-inference pass, and it is
worth running ONLY for cells that lift here.

Row protocol matches mia_quick.py so the columns are comparable: per-node mean over every window the
node appears in, edge losses credited to both endpoints, nodes present in both splits dropped as
having no well-defined label. Sums are accumulated in float64 -- the dumps store emb as float16 and
a naive accumulation drifts by ~2% on the highest-touch nodes.
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

CAMPAIGN = "/data/pids_v2_shadow/clearscope_e3"
OUT = "/data/pids_v2_attacks/clearscope_e3/results/ours_v2.json"
DETECTORS = ("nodlink", "magic", "flash", "threatrace", "rcaid",
             "velox", "orthrus_non_snooped", "kairos")
EDGE = {"velox", "orthrus_non_snooped", "kairos"}
# Kairos's features are label-dependent under the two-pass dump (TGN memory + reset_state between
# splits), so its rows are computed for completeness and marked unreportable rather than quoted.
STATEFUL = {"kairos"}
ARMS = ("TARGET", "SHADOW")
SPLITS = {"mia_train_in": 1, "mia_heldout": 0}
CKPTS = (49, 99, 149, 199, 299)
MAX_PER_CLASS = 30_000  # 60k rows total, matching the no-frac protocol's MAX_KEYS


def run_dir(detector: str, dataset: str) -> str | None:
    root = os.path.join(CAMPAIGN, detector, "artifacts", "training", "training")
    hits = [os.path.join(root, h, dataset, "edge_losses")
            for h in os.listdir(root)
            if os.path.isdir(os.path.join(root, h, dataset, "edge_losses"))]
    return hits[0] if hits else None


def node_features(z, detector: str):
    """(nodes, loss, feat) for one dump window, for either schema."""
    if detector in EDGE:
        if "srcnode" not in z.files or not len(z["loss"]):
            return None
        n = np.concatenate([z["srcnode"], z["dstnode"]]).astype(np.int64)
        loss = np.concatenate([z["loss"], z["loss"]]).astype(np.float64)
        feat = np.concatenate([z["h_src"], z["h_dst"]]).astype(np.float64)
        return n, loss, feat
    if "node" not in z.files or not len(z["node"]):
        return None
    n = z["node"].astype(np.int64)
    parts = [z[k].astype(np.float64) for k in ("emb", "posterior", "logits") if k in z.files]
    return n, z["loss"].astype(np.float64), np.hstack(parts)


def node_ids(z, detector: str):
    """Just the node column. npz members decompress lazily, so this does not touch the features."""
    if detector in EDGE:
        if "srcnode" not in z.files or not len(z["srcnode"]):
            return None
        return np.concatenate([z["srcnode"], z["dstnode"]]).astype(np.int64)
    if "node" not in z.files or not len(z["node"]):
        return None
    return z["node"].astype(np.int64)


BATCH = 40  # windows reduced at a time; caps peak memory at roughly BATCH x rows x dim


def collect(split_dir: str, detector: str):
    """(nodes, mean loss, mean feature vector) over every window in one split.

    Two passes, reducing in batches. Concatenating every window first and reducing once is the
    obvious way to write this and it is what got the sweep OOM-killed: a 300-window split of 256-d
    edge features holds the whole dump in RAM before collapsing it to one row per node. Here peak
    memory is bounded by the output array plus one batch.
    """
    files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
    ids = []
    for p in files:
        try:
            z = np.load(p)
        except Exception as e:                       # a truncated dump must not look like no data
            raise RuntimeError(f"unreadable dump {p}: {e}") from e
        got = node_ids(z, detector)
        if got is not None:
            ids.append(got)
    if not ids:
        return None
    uniq = np.unique(np.concatenate(ids))
    del ids

    n, dim = len(uniq), None
    cnt = np.zeros(n)
    sloss = np.zeros(n)
    sfeat = None
    for i in range(0, len(files), BATCH):
        nb, lb, fb = [], [], []
        for p in files[i:i + BATCH]:
            got = node_features(np.load(p), detector)
            if got is None:
                continue
            nb.append(got[0]); lb.append(got[1]); fb.append(got[2])
        if not nb:
            continue
        idx = np.searchsorted(uniq, np.concatenate(nb))
        lb = np.concatenate(lb)
        fb = np.vstack(fb)
        if sfeat is None:
            dim = fb.shape[1]
            sfeat = np.zeros((n, dim))
        cnt += np.bincount(idx, minlength=n)
        sloss += np.bincount(idx, weights=lb, minlength=n)
        for j in range(dim):
            sfeat[:, j] += np.bincount(idx, weights=fb[:, j], minlength=n)
    if sfeat is None:
        return None
    cnt = np.maximum(cnt, 1.0)
    return uniq, sloss / cnt, sfeat / cnt[:, None]


def metrics(y, s) -> dict:
    auc = float(roc_auc_score(y, s))
    fpr, tpr, _ = roc_curve(y, s)
    return {"auc": auc, "bal_acc": float((1.0 + float(np.max(tpr - fpr))) / 2.0),
            "tpr_at_1": float(np.interp(0.01, fpr, tpr))}


def cell(detector: str, arm: str, epoch: int) -> dict | None:
    # Seeded by (detector, arm) and NOT by epoch, so every checkpoint of one arm is scored on the
    # SAME subsampled nodes. The epoch curve is the whole argument -- a signal that is flat from
    # epoch 0 to 299 cannot be memorisation -- and that reading is only clean if the rows do not
    # move underneath it. A sequential RNG shifted cells by ~0.005 depending on run order.
    rng = np.random.default_rng(zlib.crc32(f"{detector}|{arm}".encode()))
    base = run_dir(detector, f"CLEARSCOPE_E3_V2_{arm}")
    if not base:
        return None
    per = {}
    for split in SPLITS:
        d = os.path.join(base, split, f"model_epoch_{epoch}")
        if not os.path.isdir(d):
            return None
        got = collect(d, detector)
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
    # Memorisation means members are reconstructed BETTER, so the score is the NEGATED loss.
    X = np.hstack([loss.reshape(-1, 1), np.linalg.norm(feat, axis=1, keepdims=True), feat])
    return {"n_members": int(len(lm)), "n_nonmembers": int(len(ln)),
            "n_overlap_dropped": int(len(overlap)), "feat_dim": int(feat.shape[1]),
            "single_loss": metrics(y, -loss),
            "ours": metrics(y, C._within_cv(X, y)),
            "unreportable": detector in STATEFUL}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", default=",".join(DETECTORS))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--epochs", default=",".join(str(e) for e in CKPTS))
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    rows, t0 = {}, time.time()
    if os.path.exists(a.out):
        rows = json.load(open(a.out))
    for det in a.detectors.split(","):
        for arm in a.arms.split(","):
            for ep in (int(e) for e in a.epochs.split(",")):
                key = f"{det}|{arm}|{ep}"
                if key in rows:
                    continue
                t = time.time()
                try:
                    r = cell(det, arm, ep)
                except Exception as e:
                    print(f"  {key:<45} ERROR {e}", flush=True)
                    continue
                if r is None:
                    print(f"  {key:<45} no dump", flush=True)
                    continue
                rows[key] = r
                print(f"  {key:<45} single {r['single_loss']['auc']:.4f} | "
                      f"ours {r['ours']['auc']:.4f} / bACC {r['ours']['bal_acc']:.4f} | "
                      f"d={r['feat_dim']} n={r['n_members']}+{r['n_nonmembers']} "
                      f"[{time.time()-t:.0f}s]", flush=True)
                os.makedirs(os.path.dirname(a.out), exist_ok=True)
                json.dump(rows, open(a.out, "w"), indent=1)
    print(f"{len(rows)} cells in {time.time()-t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
