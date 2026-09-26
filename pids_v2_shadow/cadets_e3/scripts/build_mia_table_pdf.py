#!/usr/bin/env python
"""Detection + node-MIA summary as PDF, one checkpoint per page.

Reads `results/v2_table_5ckpt.csv` for the MIA columns rather than recomputing them: scoring one
cell means re-reading the whole edge dump for that epoch, millions of rows, and all 80 cells are
already present. The detection columns ARE re-read from disk, because five of them were still
blank in the CSV when it was built -- evaluation finished after that run -- and a blank there is
staleness, not a real gap. The refreshed CSV and markdown are rewritten so every artifact agrees.

Drawn with matplotlib for the same reason as build_detection_pdf.py: no LaTeX engine on this box,
and LibreOffice's HTML importer drops the first body element and splits tables across pages.
"""
from __future__ import annotations

import csv
import glob
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

CAMPAIGN = "/data/pids_v2_shadow/cadets_e3"
RESULTS = os.path.join(CAMPAIGN, "results")
OUT_PDF = "/home/yingjie/PIDSMaker/v2_mia_table_5ckpt.pdf"
DETECTORS = (("ThreaTrace", "threatrace"), ("NodLink", "nodlink"), ("Magic", "magic"),
             ("Kairos", "kairos"), ("Flash", "flash"), ("R-Caid", "rcaid"),
             ("Orthrus", "orthrus_non_snooped"), ("Velox", "velox"))
ARMS = ("TARGET", "SHADOW")
CKPTS = (49, 99, 149, 199, 299)
STATEFUL = {"kairos"}
PER_ARM = ("det AUC", "det bACC", "MIA AUC", "MIA bACC", "TPR@1%")
KEYS = ("det_auc", "det_bal_acc", "mia_auc", "mia_bal_acc", "mia_tpr_at_1")
PAGE = (11.0, 8.5)
HDR_BG, GRP_BG, ALT_BG, FLAG_BG = "#e6e6e6", "#cfd8e6", "#f5f7fa", "#f7e3e3"


def detection(detector: str, arm: str, epoch: int) -> dict:
    hits = glob.glob(os.path.join(
        CAMPAIGN, detector, "artifacts", "evaluation", "evaluation", "*",
        f"CADETS_E3_V2_{arm}", "precision_recall_dir", f"stats_model_epoch_{epoch}.pth"))
    if not hits:
        return {}
    d = torch.load(hits[0], map_location="cpu", weights_only=False)
    return {"det_auc": d.get("auc"), "det_bal_acc": d.get("balanced_acc"),
            "det_ap": d.get("ap"), "det_fpr": d.get("fpr")}


def num(v, n=3) -> str:
    try:
        return f"{float(v):.{n}f}"
    except (TypeError, ValueError):
        return "--"


def load_and_refresh() -> tuple[dict, int]:
    src = os.path.join(RESULTS, "v2_table_5ckpt.csv")
    rows = list(csv.DictReader(open(src)))
    filled = 0
    for r in rows:
        fresh = detection(r["detector"], r["arm"], int(r["epoch"]))
        for k, v in fresh.items():
            if not r.get(k) and v is not None:
                filled += 1
            if v is not None:
                r[k] = v
    with open(src, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = ["# v2 CADETS_E3 -- detection + node MIA, 5 checkpoints\n",
          "MIA AUC/balACC score each arm against its OWN labels. Kairos rows are NOT reportable",
          "(two-pass `_dump_split` + TGN memory = label-dependent features).\n",
          "| detector | arm | epoch | det AUC | det balACC | MIA AUC | MIA balACC | MIA TPR@1% |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        flag = " !" if r["detector"] in STATEFUL else ""
        md.append(f"| {r['detector']}{flag} | {r['arm']} | {r['epoch']} | "
                  f"{num(r.get('det_auc'), 4)} | {num(r.get('det_bal_acc'), 4)} | "
                  f"{num(r.get('mia_auc'), 4)} | {num(r.get('mia_bal_acc'), 4)} | "
                  f"{num(r.get('mia_tpr_at_1'), 4)} |")
    open(os.path.join(RESULTS, "v2_table_5ckpt.md"), "w").write("\n".join(md) + "\n")
    return {(r["detector"], r["arm"], int(r["epoch"])): r for r in rows}, filled


def draw_table(ax, rows: dict, epoch: int) -> None:
    ncol = 1 + len(PER_ARM) * len(ARMS)
    w_sys = 0.13
    w_col = (1.0 - w_sys) / (ncol - 1)
    xs = [0.0, w_sys] + [w_sys + w_col * (i + 1) for i in range(ncol - 1)]
    nrow = 2 + len(DETECTORS)
    h = 1.0 / nrow
    ys = [1.0 - h * i for i in range(nrow + 1)]

    def cell(c0, c1, r, text, *, bg=None, bold=False, align="right", size=8.2):
        x0, x1, y0, y1 = xs[c0], xs[c1], ys[r + 1], ys[r]
        if bg:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=bg,
                                   edgecolor="none", zorder=0))
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                               edgecolor="#999999", linewidth=0.4, zorder=2))
        pad = 0.005
        tx = {"right": x1 - pad, "left": x0 + pad, "center": (x0 + x1) / 2}[align]
        ax.text(tx, (y0 + y1) / 2, text, ha=align, va="center", fontsize=size,
                fontweight="bold" if bold else "normal", zorder=3)

    cell(0, 1, 0, "", bg=HDR_BG)
    cell(0, 1, 1, "System", bg=HDR_BG, bold=True, align="left")
    for a, arm in enumerate(ARMS):
        base = 1 + a * len(PER_ARM)
        cell(base, base + len(PER_ARM), 0, arm, bg=GRP_BG, bold=True, align="center", size=10)
        for i, name in enumerate(PER_ARM):
            cell(base + i, base + i + 1, 1, name, bg=HDR_BG, bold=True, align="center", size=8)

    for d, (disp, det) in enumerate(DETECTORS):
        r = 2 + d
        flagged = det in STATEFUL
        bg = FLAG_BG if flagged else (ALT_BG if d % 2 else None)
        cell(0, 1, r, disp + (" !" if flagged else ""), bg=bg, bold=True, align="left")
        for a, arm in enumerate(ARMS):
            src = rows.get((det, arm, epoch), {})
            for i, k in enumerate(KEYS):
                cell(1 + a * len(PER_ARM) + i, 2 + a * len(PER_ARM) + i, r,
                     num(src.get(k)), bg=bg)

    for x0, y0, x1, y1 in [(0, ys[-1], 1, ys[0]), (0, ys[2], 1, ys[2]),
                           (xs[1 + len(PER_ARM)], ys[-1], xs[1 + len(PER_ARM)], ys[0])]:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                               edgecolor="#333333", linewidth=1.1, zorder=4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


NOTES = [
    "Each row pairs a detector's DETECTION quality with the node-level membership-inference (MIA) "
    "attack against that same checkpoint. det AUC / det bACC are the detector's own task. "
    "MIA AUC / MIA bACC / TPR@1% are the attack. All five checkpoints are shown as-is; there is "
    "one training run per arm, so nothing is selected or maximised over.",
    "",
    "HEADLINE: NO DETECTOR LEAKS. Across the clean cells, MIA AUC spans roughly 0.35-0.58 against "
    "a 0.50 chance line, and TPR at 1% FPR sits at or below the 1% chance rate everywhere. Several "
    "arms score BELOW chance. Magic SHADOW at epoch 0 (AUC 0.571) beats every trained TARGET arm, "
    "which memorisation cannot explain -- a barely-trained model cannot have memorised anything.",
    "",
    "WHAT THESE NUMBERS ARE NOT. Each arm is scored against its OWN membership labels on its OWN "
    "days, so membership is perfectly collinear with the calendar: members are 04-02/03/04 (TARGET) "
    "or 04-07/08/09 (SHADOW), non-members are the single held-out day 04-05 or 04-10. Any "
    "separation here can be day-to-day drift rather than memorisation. The controlled comparison -- "
    "the shadow model re-queried over the TARGET's nodes -- has not been run. A previous campaign "
    "is the cautionary case: a headline 0.91 AUC was matched at 0.93 by a model that had never seen "
    "the members.",
    "",
    "KAIROS (!) IS NOT REPORTABLE. mia_dump._dump_split scores members and non-members in two "
    "separate passes with model.reset_state() between them. Kairos is the only detector here whose "
    "trained weights carry TGN memory, so its per-edge loss depends on which pass a record was "
    "scored in -- that is, on the label being predicted. No attacker can run that protocol. A prior "
    "campaign measured the inflation directly: Kairos gradient MIA read 0.6776 two-pass against "
    "0.5168 single-pass. Re-dump with PIDS_MIA_SINGLEPASS=1 before quoting any Kairos number.",
    "",
    "CONVENTIONS. Memorisation means members are reconstructed BETTER, so the membership score is "
    "the NEGATED loss; scoring raw loss would invert every AUC. Balanced accuracy is the Youden "
    "statistic, (1 + max(TPR - FPR)) / 2 -- the single best-separating threshold, which can read "
    "high while AUC sits below 0.50. Flash's elevated MIA bACC (~0.65) is exactly this artifact: "
    "39,202 nodes, 40.5% of non-members and zero members, share one byte-identical loss, and the "
    "threshold parks on that block. It identifies NON-members, so it is useless to an attacker.",
    "",
    "The TARGET and SHADOW arms are trained on disjoint days and tested on different attack days, "
    "so their DETECTION columns are not head-to-head. For edge-based detectors (Velox, Orthrus, "
    "Kairos) each edge loss is credited to both endpoints and averaged per node, so these are "
    "node-membership numbers throughout.",
]


def main() -> None:
    rows, filled = load_and_refresh()
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        fig = plt.figure(figsize=PAGE)
        fig.text(0.06, 0.94, "CADETS_E3 v2 - detection quality and membership-inference leakage",
                 fontsize=15, fontweight="bold", va="top")
        y = 0.88
        for para in NOTES:
            if not para:
                y -= 0.014
                continue
            for line in textwrap.wrap(para, 122):
                fig.text(0.06, y, line, fontsize=8.2, va="top")
                y -= 0.0185
        fig.text(0.06, 0.05, f"Tables follow on pages 2-{1 + len(CKPTS)}, one per checkpoint: "
                             + ", ".join(f"epoch {e}" for e in CKPTS) + ".",
                 fontsize=8.5, style="italic", va="top")
        pdf.savefig(fig)
        plt.close(fig)

        for e in CKPTS:
            fig = plt.figure(figsize=PAGE)
            fig.text(0.5, 0.94, f"Checkpoint: epoch {e}", fontsize=14,
                     fontweight="bold", ha="center", va="top")
            ax = fig.add_axes([0.06, 0.18, 0.88, 0.72])
            draw_table(ax, rows, e)
            fig.text(0.06, 0.12, "!  Kairos is not reportable: two-pass dump + TGN memory makes "
                                 "its MIA features label-dependent (see page 1).",
                     fontsize=8, style="italic", va="top")
            pdf.savefig(fig)
            plt.close(fig)
        pdf.infodict()["Title"] = "CADETS_E3 v2 detection and MIA per checkpoint"
    print(f"backfilled {filled} stale detection values")
    print(OUT_PDF)


if __name__ == "__main__":
    main()
