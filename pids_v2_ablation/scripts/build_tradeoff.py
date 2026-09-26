#!/usr/bin/env python
"""Ablation 1b -- is early stopping a usable defence, or does it just break the detector?

The epoch ladder alone cannot answer that. A checkpoint that leaks nothing is worthless if it also
detects nothing, so the question is not "when does leakage appear" but "does leakage appear before
or after the detector becomes useful". This pairs each MIA rung with the detection AUC of the same
checkpoint.

It also answers the standing question about weak detectors directly: by walking the checkpoints we
sweep detection AUC across its whole range and can read off the MIA AUC at each level, rather than
guessing what a 0.3/0.4/0.5 detector would leak.

The detection sweep evaluated every ODD epoch plus 0, while the MIA dump ladder is 0,1,2,5,10,24,...
-- so even rungs (2, 10, 24) have no exact detection counterpart and are matched to the nearest
evaluated epoch. The offset is at most one epoch and is printed, not hidden.
"""
from __future__ import annotations

import csv
import glob
import json
import os

import torch

ROOT = "/data/pids_v2_ablation"
SRC = "/data/pids_v2_attacks"
CAMP = "/data/pids_v2_shadow"
# THEIA's arms predate the _V2_ naming and are plain THEIA_E3_TARGET/SHADOW. Getting this wrong is
# silent: the glob simply matches nothing and the dataset vanishes from every aggregate below.
DATASETS = {"cadets_e3": "CADETS_E3_V2", "theia_e3": "THEIA_E3",
            "clearscope_e3": "CLEARSCOPE_E3_V2"}
NODE = ("nodlink", "magic", "flash", "threatrace", "rcaid")
ARMS = ("TARGET", "SHADOW")
LADDER = (0, 1, 2, 5, 10, 24, 49, 99, 149, 199, 299)


def det_epochs(ds, det, arm):
    d = glob.glob(os.path.join(CAMP, ds, det, "artifacts", "evaluation", "evaluation", "*",
                               f"{DATASETS[ds]}_{arm}", "precision_recall_dir"))
    if not d:
        return {}
    out = {}
    for f in glob.glob(os.path.join(d[0], "stats_model_epoch_*.pth")):
        try:
            out[int(f.rsplit("_", 1)[1].split(".")[0])] = f
        except ValueError:
            continue
    return out


def main():
    rows = []
    for ds in DATASETS:
        he = {}
        for n in ("he_conti_v2.json", "he_conti_v2_early.json", "he_conti_ablation.json"):
            for base in (os.path.join(ROOT, ds, "results"), os.path.join(SRC, ds, "results")):
                p = os.path.join(base, n)
                if os.path.exists(p):
                    he.update(json.load(open(p)))
        for det in NODE:
            for arm in ARMS:
                avail = det_epochs(ds, det, arm)
                if not avail:
                    continue
                for e in LADDER:
                    cell = he.get(f"{det}|{arm}|{e}")
                    if not cell or "he2021" not in cell:
                        continue
                    near = min(avail, key=lambda a: (abs(a - e), a))
                    s = torch.load(avail[near], map_location="cpu", weights_only=False)
                    rows.append({
                        "dataset": ds, "detector": det, "arm": arm, "mia_epoch": e,
                        "det_epoch": near, "offset": near - e,
                        "det_auc": s.get("auc"), "det_bal_acc": s.get("balanced_acc"),
                        "det_mcc": s.get("mcc"), "det_precision": s.get("precision"),
                        "det_recall": s.get("recall"),
                        "mia_he_auc": cell["he2021"]["auc"],
                        "mia_he_bal_acc": cell["he2021"]["bal_acc"],
                    })

    # A mis-named arm makes the glob match nothing, which drops a whole dataset from every
    # aggregate without raising. Fail loudly instead of publishing a mean over two thirds of the data.
    for ds in DATASETS:
        if not any(r["dataset"] == ds for r in rows):
            raise SystemExit(f"{ds} contributed 0 rows -- check the arm name {DATASETS[ds]}_*")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    cols = list(rows[0])
    with open(os.path.join(ROOT, "results", "tradeoff_detection_vs_mia.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    doc = ["# Ablation 1b -- detection quality vs membership leakage, same checkpoint",
           "",
           "Pairs every MIA rung with the detection AUC of the *same* checkpoint, so early stopping "
           "can be\npriced: a checkpoint is only a defence if it drops MIA without dropping "
           "detection.",
           "",
           "MIA is He 2021 (the mountable-adjacent numeric-answer attack). Detection AUC is "
           "node-level.",
           "Even MIA rungs (2, 10, 24) have no exact detection checkpoint and use the nearest "
           "evaluated\nepoch; `off` is that offset in epochs.",
           ""]

    # Only detectors whose MIA actually moves are worth a per-epoch curve.
    for ds in DATASETS:
        for det in NODE:
            for arm in ARMS:
                g = [r for r in rows if r["dataset"] == ds and r["detector"] == det
                     and r["arm"] == arm]
                if len(g) < 3:
                    continue
                lift = max(r["mia_he_auc"] for r in g) - g[0]["mia_he_auc"]
                if lift < 0.10:
                    continue
                doc += [f"\n## {ds} / {det} / {arm} -- MIA gains {lift:+.3f} over its "
                        f"epoch-0 baseline\n",
                        "| epoch | off | detection AUC | detection MCC | MIA AUC | MIA balACC |",
                        "|---|---|---|---|---|---|"]
                for r in g:
                    da = f"{r['det_auc']:.3f}" if r["det_auc"] is not None else "--"
                    mc = f"{r['det_mcc']:.3f}" if r["det_mcc"] is not None else "--"
                    doc.append(f"| e{r['mia_epoch']} | {r['offset']:+d} | {da} | {mc} | "
                               f"{r['mia_he_auc']:.3f} | {r['mia_he_bal_acc']:.3f} |")

    # Does a weaker detector leak less? Bin every (checkpoint, arm, dataset) by detection AUC.
    doc += ["",
            "## Does a weaker detector leak less?",
            "",
            "Every node-detector checkpoint across all three datasets, binned by its detection AUC. "
            "If\nleakage tracked detection quality, MIA AUC would climb with the bin.",
            "",
            "| detection AUC band | checkpoints | mean detection AUC | mean MIA AUC | max MIA AUC |",
            "|---|---|---|---|---|"]
    bands = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for lo, hi in bands:
        g = [r for r in rows if r["det_auc"] is not None and lo <= r["det_auc"] < hi]
        if not g:
            continue
        doc.append(f"| {lo:.1f}-{hi:.1f} | {len(g)} | "
                   f"{sum(r['det_auc'] for r in g)/len(g):.3f} | "
                   f"{sum(r['mia_he_auc'] for r in g)/len(g):.3f} | "
                   f"{max(r['mia_he_auc'] for r in g):.3f} |")

    n = len(rows)
    mx = sum(r["det_auc"] for r in rows if r["det_auc"] is not None)
    my = sum(r["mia_he_auc"] for r in rows)
    mx, my = mx / n, my / n
    cov = sum((r["det_auc"] - mx) * (r["mia_he_auc"] - my) for r in rows if r["det_auc"] is not None)
    vx = sum((r["det_auc"] - mx) ** 2 for r in rows if r["det_auc"] is not None) ** 0.5
    vy = sum((r["mia_he_auc"] - my) ** 2 for r in rows) ** 0.5
    doc += ["", f"**Correlation across all {n} checkpoints:** pearson r = "
                f"{cov/(vx*vy):.3f} between detection AUC and MIA AUC."]

    md = "\n".join(doc) + "\n"
    open(os.path.join(ROOT, "results", "ABLATION_1B_TRADEOFF.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
