#!/usr/bin/env python
"""Node-level detection results in the PIDSMaker Table-1 format, one table per checkpoint.

Layout mirrors the paper: detectors are rows, and where the paper groups columns by DATASET this
groups them by ARM (TARGET / SHADOW), since v2 trains the same eight detectors twice on disjoint
day splits. Columns within a group are the paper's: TP, FP, TN, FN, Prec, Rec, MCC, ADP.

One table per checkpoint in 49/99/149/199/299 -- the five the old no-frac campaign saved, so these
line up with that campaign even though v2 dumped eleven.

NOT the paper's selection rule. The paper reports the best of five training runs; v2 has one run per
arm, so there is no equivalent axis to maximise over. Each table is simply that checkpoint as-is.
"""
from __future__ import annotations

import csv
import glob
import os

import torch

CAMPAIGN = "/data/pids_v2_shadow/cadets_e3"
# Paper's row order, with PIDSMaker's internal names alongside.
DETECTORS = (("ThreaTrace", "threatrace"), ("NodLink", "nodlink"), ("Magic", "magic"),
             ("Kairos", "kairos"), ("Flash", "flash"), ("R-Caid", "rcaid"),
             ("Orthrus", "orthrus_non_snooped"), ("Velox", "velox"))
ARMS = (("TARGET", "CADETS_E3_V2_TARGET"), ("SHADOW", "CADETS_E3_V2_SHADOW"))
CKPTS = (49, 99, 149, 199, 299)
COUNTS = ("tp", "fp", "tn", "fn")
RATES = ("precision", "recall", "mcc", "adp_score")


def stats(detector: str, arm: str, epoch: int) -> dict:
    hits = glob.glob(os.path.join(
        CAMPAIGN, detector, "artifacts", "evaluation", "evaluation", "*", arm,
        "precision_recall_dir", f"stats_model_epoch_{epoch}.pth"))
    if not hits:
        return {}
    d = torch.load(hits[0], map_location="cpu", weights_only=False)
    return {k: d.get(k) for k in COUNTS + RATES} | {"auc": d.get("auc")}


def human(n) -> str:
    # numpy int64 is not a Python int, so an isinstance(n, int) guard silently blanks every count.
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "--"
    return f"{n/1000:.0f}k" if abs(n) >= 10000 else str(n)


def num(v) -> str:
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "--"


def main() -> None:
    out = os.path.join(CAMPAIGN, "results")
    os.makedirs(out, exist_ok=True)
    doc = ["# v2 CADETS_E3 -- node-level detection results, per checkpoint",
           "",
           "Columns grouped by arm (the paper groups by dataset). Prec=Precision, Rec=Recall,",
           "MCC=Matthews Correlation Coefficient, ADP=Attack Detection Precision.", ""]
    rows_csv = []

    per_arm = ["TP", "FP", "TN", "FN", "Prec", "Rec", "MCC", "ADP"]
    header = "| System | " + " | ".join(per_arm * len(ARMS)) + " |"
    rule = "|---" * (1 + len(per_arm) * len(ARMS)) + "|"
    # Group banner: one label cell per arm, padded so it sits over that arm's eight columns.
    sub = "| | " + " | ".join(
        " | ".join([f"**{label}**"] + [""] * (len(per_arm) - 1)) for label, _ in ARMS) + " |"

    for e in CKPTS:
        doc += [f"\n## Checkpoint: epoch {e}\n", sub, header, rule]
        for disp, det in DETECTORS:
            cells = []
            for label, arm in ARMS:
                s = stats(det, arm, e)
                cells += [human(s.get(k)) for k in COUNTS] + [num(s.get(k)) for k in RATES]
                rows_csv.append({"epoch": e, "system": disp, "detector": det, "arm": label,
                                 **{k: s.get(k, "") for k in COUNTS + RATES},
                                 "auc": s.get("auc", "")})
            doc.append(f"| {disp} | " + " | ".join(cells) + " |")

    open(os.path.join(out, "v2_detection_tables_5ckpt.md"), "w").write("\n".join(doc) + "\n")
    cols = ["epoch", "system", "detector", "arm", *COUNTS, *RATES, "auc"]
    with open(os.path.join(out, "v2_detection_tables_5ckpt.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows_csv:
            w.writerow({c: r.get(c, "") for c in cols})
    print("\n".join(doc))


if __name__ == "__main__":
    main()
