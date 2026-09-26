#!/usr/bin/env python
"""v2 summary table: detection metrics + node MIA, 8 detectors x 2 arms x 5 checkpoints.

Checkpoints are 49/99/149/199/299 -- the five the old no-frac campaign saved, so rows here line up
with that campaign even though v2 dumped eleven.

KAIROS IS NOT REPORTABLE FROM THIS TABLE. `mia_dump._dump_split` scores members and non-members in
two separate passes with `model.reset_state()` between them. Kairos is the only detector in the set
whose trained weights carry TGN memory (`encoder.memory.*`), so its per-edge loss depends on which
pass the record was scored in -- i.e. on the label being predicted. An attacker cannot run that
protocol, and a prior campaign measured the inflation directly: Kairos gradient MIA read AUC 0.6776
two-pass against 0.5168 single-pass. Re-dump with PIDS_MIA_SINGLEPASS=1 before quoting any Kairos
number. The other seven have no memory buffers, so the reset is a no-op and their rows stand.
"""
from __future__ import annotations

import csv
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mia_quick import ARMS, CAMPAIGN, score  # noqa: E402

DETECTORS = ("nodlink", "velox", "rcaid", "magic", "threatrace",
             "orthrus_non_snooped", "kairos", "flash")
CKPTS = (49, 99, 149, 199, 299)
STATEFUL = {"kairos"}


def detection(detector: str, arm: str, epoch: int) -> dict:
    """Detection metrics for one checkpoint, or {} if evaluation has not reached it yet."""
    hits = glob.glob(os.path.join(
        CAMPAIGN, detector, "artifacts", "evaluation", "evaluation", "*", arm,
        "precision_recall_dir", f"stats_model_epoch_{epoch}.pth"))
    if not hits:
        return {}
    d = torch.load(hits[0], map_location="cpu", weights_only=False)
    return {"det_auc": d.get("auc"), "det_bal_acc": d.get("balanced_acc"),
            "det_ap": d.get("ap"), "det_fpr": d.get("fpr")}


def main() -> None:
    rows = []
    for det in DETECTORS:
        for arm in ARMS:
            for e in CKPTS:
                try:
                    m = score(det, arm, e)
                except (RuntimeError, FileNotFoundError):
                    m = None
                r = {"detector": det, "arm": arm.replace("CADETS_E3_V2_", ""), "epoch": e,
                     "stateful": det in STATEFUL, **detection(det, arm, e)}
                if m:
                    r.update(mia_auc=m["all"]["auc"], mia_bal_acc=m["all"]["bal_acc"],
                             mia_tpr_at_1=m["all"]["tpr_at_1"],
                             mia_auc_disj=(m.get("disjoint") or {}).get("auc"),
                             mia_bal_acc_disj=(m.get("disjoint") or {}).get("bal_acc"),
                             n_mem=m["n_members"], n_non=m["n_nonmembers"], n_ovl=m["n_overlap"])
                rows.append(r)
                print(f"  done {det:22s} {r['arm']:7s} e{e}", flush=True)

    cols = ["detector", "arm", "epoch", "stateful", "det_auc", "det_bal_acc", "det_ap", "det_fpr",
            "mia_auc", "mia_bal_acc", "mia_tpr_at_1", "mia_auc_disj", "mia_bal_acc_disj",
            "n_mem", "n_non", "n_ovl"]
    out = os.path.join(CAMPAIGN, "results")
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, "v2_table_5ckpt.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    def fmt(v, n=4):
        return f"{v:.{n}f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else "--"

    md = [f"# v2 CADETS_E3 -- detection + node MIA, 5 checkpoints\n",
          "MIA AUC/balACC score each arm against its OWN labels. Kairos rows are NOT reportable",
          "(two-pass `_dump_split` + TGN memory = label-dependent features).\n",
          "| detector | arm | epoch | det AUC | det balACC | MIA AUC | MIA balACC | MIA TPR@1% |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        flag = " !" if r["stateful"] else ""
        md.append(f"| {r['detector']}{flag} | {r['arm']} | {r['epoch']} | "
                  f"{fmt(r.get('det_auc'))} | {fmt(r.get('det_bal_acc'))} | "
                  f"{fmt(r.get('mia_auc'))} | {fmt(r.get('mia_bal_acc'))} | "
                  f"{fmt(r.get('mia_tpr_at_1'))} |")
    md_path = os.path.join(out, "v2_table_5ckpt.md")
    open(md_path, "w").write("\n".join(md) + "\n")
    print(f"\nwrote {csv_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
