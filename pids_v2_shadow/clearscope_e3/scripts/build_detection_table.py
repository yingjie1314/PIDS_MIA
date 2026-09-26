#!/usr/bin/env python
"""Node-level DETECTION results for CLEARSCOPE_E3 v2, per checkpoint, both arms.

Layout follows the PIDSMaker Table-1 format (detectors as rows, the paper's TP/FP/TN/FN/Prec/Rec/
MCC/ADP columns), grouped by ARM rather than by dataset because v2 trains the same eight detectors
twice on disjoint day splits.

AUC AND BALANCED ACCURACY ARE CARRIED ALONGSIDE, not left to the CSV. On CLEARSCOPE's base rate a
confusion-matrix-only row is actively misleading: nodlink/TARGET/e299 reads precision 0.012 and
MCC 0.078, which looks like a broken detector, while its AUC is 0.946. Both numbers are true --
the ranking is good and the operating threshold is bad -- and quoting either alone misrepresents
the model.

NOT the paper's selection rule. The paper reports the best of five training runs; v2 has one run
per arm, so there is no axis to maximise over. Each table is that checkpoint as-is.

Cells read `--` when the evaluation sweep has not reached that checkpoint yet. That is a real state
here, not an error: the sweep walks ~150 checkpoints at ~2 min each and lags training by hours.
"""
from __future__ import annotations

import csv
import glob
import os

import torch

CAMPAIGN = "/data/pids_v2_shadow/clearscope_e3"
DETECTORS = (("ThreaTrace", "threatrace"), ("NodLink", "nodlink"), ("Magic", "magic"),
             ("Kairos", "kairos"), ("Flash", "flash"), ("R-Caid", "rcaid"),
             ("Orthrus", "orthrus_non_snooped"), ("Velox", "velox"))
ARMS = (("TARGET", "CLEARSCOPE_E3_V2_TARGET"), ("SHADOW", "CLEARSCOPE_E3_V2_SHADOW"))
CKPTS = (49, 99, 149, 199, 299)
COUNTS = ("tp", "fp", "tn", "fn")
RATES = ("precision", "recall", "mcc", "adp_score", "auc", "balanced_acc")


def stats(detector: str, arm: str, epoch: int) -> dict:
    hits = glob.glob(os.path.join(
        CAMPAIGN, detector, "artifacts", "evaluation", "evaluation", "*", arm,
        "precision_recall_dir", f"stats_model_epoch_{epoch}.pth"))
    if not hits:
        return {}
    d = torch.load(hits[0], map_location="cpu", weights_only=False)
    return {k: d.get(k) for k in COUNTS + RATES}


def human(n) -> str:
    # numpy int64 is not a Python int, so an isinstance(n, int) guard silently blanks every count.
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "--"
    return f"{n / 1000:.0f}k" if abs(n) >= 10000 else str(n)


def num(v) -> str:
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "--"


def main() -> None:
    out = os.path.join(CAMPAIGN, "results")
    os.makedirs(out, exist_ok=True)
    doc = ["# CLEARSCOPE_E3 v2 -- node-level detection results, per checkpoint",
           "",
           "Eight detectors x two day-disjoint arms x 300 epochs. Columns are grouped by arm.",
           "Prec=Precision, Rec=Recall, MCC=Matthews Correlation Coefficient,",
           "ADP=Attack Detection Precision, balACC=balanced accuracy.",
           "",
           "Read AUC and balACC together with the confusion counts: on this base rate a detector",
           "can rank almost perfectly (AUC 0.95) while its fixed threshold yields precision 0.01.",
           "`--` means the evaluation sweep has not reached that checkpoint yet.",
           ""]
    rows_csv = []

    per_arm = ["TP", "FP", "TN", "FN", "Prec", "Rec", "MCC", "ADP", "AUC", "balACC"]
    header = "| System | " + " | ".join(per_arm * len(ARMS)) + " |"
    rule = "|---" * (1 + len(per_arm) * len(ARMS)) + "|"
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
                                 **{k: s.get(k, "") for k in COUNTS + RATES}})
            doc.append(f"| {disp} | " + " | ".join(cells) + " |")

    open(os.path.join(out, "v2_detection_tables_5ckpt.md"), "w").write("\n".join(doc) + "\n")
    cols = ["epoch", "system", "detector", "arm", *COUNTS, *RATES]
    with open(os.path.join(out, "v2_detection_tables_5ckpt.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows_csv:
            w.writerow({c: r.get(c, "") for c in cols})
    print("\n".join(doc))


if __name__ == "__main__":
    main()
