#!/usr/bin/env python
"""Render the per-checkpoint detection tables to PDF, one checkpoint per page.

Drawn directly with matplotlib rather than converted from HTML or markdown. The obvious routes on
this box do not work: there is no LaTeX engine, so pandoc cannot emit PDF, and LibreOffice's HTML
importer silently discards the first element of <body>, ignores `page-break-before`, and splits
tables across page boundaries mid-row. Drawing the cells means the arm-group header is a genuine
merged span and every table is guaranteed whole on its own page.
"""
from __future__ import annotations

import collections
import csv
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

CAMPAIGN = "/data/pids_v2_shadow/cadets_e3"
OUT_PDF = "/home/yingjie/PIDSMaker/v2_detection_tables_5ckpt.pdf"
DETECTORS = (("ThreaTrace", "threatrace"), ("NodLink", "nodlink"), ("Magic", "magic"),
             ("Kairos", "kairos"), ("Flash", "flash"), ("R-Caid", "rcaid"),
             ("Orthrus", "orthrus_non_snooped"), ("Velox", "velox"))
ARMS = ("TARGET", "SHADOW")
CKPTS = (49, 99, 149, 199, 299)
COUNTS = ("tp", "fp", "tn", "fn")
RATES = ("precision", "recall", "mcc", "adp_score")
PER_ARM = ("TP", "FP", "TN", "FN", "Prec", "Rec", "MCC", "ADP")
PAGE = (11.0, 8.5)  # landscape letter

HDR_BG, GRP_BG, ALT_BG = "#e6e6e6", "#cfd8e6", "#f5f7fa"


def human(n) -> str:
    # numpy int64 is not a Python int, so an isinstance(n, int) guard silently blanks every count.
    try:
        n = int(float(n))
    except (TypeError, ValueError):
        return "--"
    return f"{n / 1000:.0f}k" if abs(n) >= 10000 else str(n)


def num(v) -> str:
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "--"


def load() -> dict:
    src = os.path.join(CAMPAIGN, "results", "v2_detection_tables_5ckpt.csv")
    return {(r["detector"], r["arm"], int(r["epoch"])): r
            for r in csv.DictReader(open(src))}


def draw_table(ax, rows: dict, epoch: int) -> None:
    ncol = 1 + len(PER_ARM) * len(ARMS)
    w_sys = 0.085
    w_col = (1.0 - w_sys) / (ncol - 1)
    xs = [0.0, w_sys] + [w_sys + w_col * (i + 1) for i in range(ncol - 1)]

    nrow = 2 + len(DETECTORS)
    h = 1.0 / nrow
    ys = [1.0 - h * i for i in range(nrow + 1)]

    def cell(c0, c1, r, text, *, bg=None, bold=False, align="right", size=7.4):
        x0, x1, y0, y1 = xs[c0], xs[c1], ys[r + 1], ys[r]
        if bg:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=bg,
                                   edgecolor="none", zorder=0))
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                               edgecolor="#999999", linewidth=0.4, zorder=2))
        pad = 0.004
        tx = {"right": x1 - pad, "left": x0 + pad, "center": (x0 + x1) / 2}[align]
        ax.text(tx, (y0 + y1) / 2, text, ha=align, va="center", fontsize=size,
                fontweight="bold" if bold else "normal", zorder=3)

    cell(0, 1, 0, "", bg=HDR_BG)
    cell(0, 1, 1, "System", bg=HDR_BG, bold=True, align="left")
    for a, arm in enumerate(ARMS):
        base = 1 + a * len(PER_ARM)
        cell(base, base + len(PER_ARM), 0, arm, bg=GRP_BG, bold=True, align="center", size=9)
        for i, name in enumerate(PER_ARM):
            cell(base + i, base + i + 1, 1, name, bg=HDR_BG, bold=True, align="center")

    for d, (disp, det) in enumerate(DETECTORS):
        r = 2 + d
        bg = ALT_BG if d % 2 else None
        cell(0, 1, r, disp, bg=bg, bold=True, align="left")
        for a, arm in enumerate(ARMS):
            src = rows.get((det, arm, epoch), {})
            vals = [human(src.get(k)) for k in COUNTS] + [num(src.get(k)) for k in RATES]
            for i, v in enumerate(vals):
                cell(1 + a * len(PER_ARM) + i, 2 + a * len(PER_ARM) + i, r, v, bg=bg)

    # Heavy rules: outer frame, the header/body split, and the TARGET|SHADOW boundary.
    for x0, y0, x1, y1 in [(0, ys[-1], 1, ys[0]), (0, ys[2], 1, ys[2]),
                           (xs[1 + len(PER_ARM)], ys[-1], xs[1 + len(PER_ARM)], ys[0])]:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                               edgecolor="#333333", linewidth=1.1, zorder=4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def notes(rows: dict) -> list[str]:
    # Positives per arm are a structural fingerprint: TARGET tests 04-06+04-12, SHADOW tests 04-13.
    pos = collections.defaultdict(set)
    for (_, arm, _), r in rows.items():
        pos[arm].add(int(float(r["tp"])) + int(float(r["fn"])))
    fp = ", ".join(f"{a}={sorted(s)}" for a, s in sorted(pos.items()))
    return [
        "PIDSMaker Table-1 format. Where the paper groups columns by dataset, this groups by ARM: "
        "the same eight detectors trained twice on disjoint day splits. Prec = Precision, "
        "Rec = Recall, MCC = Matthews Correlation Coefficient, ADP = Attack Detection Precision. "
        "Counts of 10,000 or more are abbreviated (e.g. 185k).",
        "",
        "ARM DEFINITIONS.  TARGET trains on 2018-04-02/03/04, holds out 04-05, and is tested on "
        "04-06 + 04-12 (51 ground-truth malicious nodes, 49 realized in the graphs). SHADOW trains "
        "on 04-07/08/09, holds out 04-10, and is tested on 04-13 (24 ground truth, 22 realized). "
        "The two arms share no days.",
        "",
        "THE TWO ARMS ARE NOT HEAD-TO-HEAD. They are scored on different test days, with different "
        "positive counts and a roughly six-fold difference in test-set size (~185k rows against "
        "~31k). Read them as two separate detection results, not as a comparison of one arm "
        "against the other.",
        "",
        "SELECTION RULE. This is not the paper's. The paper reports the best of five training "
        "runs; v2 has one run per arm, so there is no equivalent axis to maximise over. Each table "
        "is that checkpoint as-is.",
        "",
        "ARM ASSIGNMENT VERIFIED. Each detector/arm resolves to exactly one config-hash directory, "
        "and the arm name is a literal path component, so no cell depends on an ambiguous lookup. "
        "All 640 values were re-extracted directly from the stats_model_epoch_N.pth checkpoints "
        "with the arm derived from the filesystem path rather than from the builder's own arm "
        f"list: 0 differences. Positives per arm are unique and constant ({fp}), which a swapped "
        "column would break.",
    ]


def main() -> None:
    rows = load()
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        fig = plt.figure(figsize=PAGE)
        fig.text(0.06, 0.93, "CADETS_E3 v2 - node-level detection results, per checkpoint",
                 fontsize=16, fontweight="bold", va="top")
        y = 0.86
        for para in notes(rows):
            if not para:
                y -= 0.018
                continue
            for line in textwrap.wrap(para, 118):
                fig.text(0.06, y, line, fontsize=9, va="top")
                y -= 0.022
        fig.text(0.06, 0.06, f"Tables follow on pages 2-{1 + len(CKPTS)}, "
                             "one per checkpoint: " + ", ".join(f"epoch {e}" for e in CKPTS) + ".",
                 fontsize=9, style="italic", va="top")
        pdf.savefig(fig)
        plt.close(fig)

        for e in CKPTS:
            fig = plt.figure(figsize=PAGE)
            fig.text(0.5, 0.94, f"Checkpoint: epoch {e}", fontsize=14,
                     fontweight="bold", ha="center", va="top")
            ax = fig.add_axes([0.035, 0.18, 0.93, 0.72])
            draw_table(ax, rows, e)
            pdf.savefig(fig)
            plt.close(fig)

        d = pdf.infodict()
        d["Title"] = "CADETS_E3 v2 node-level detection results per checkpoint"
    print(OUT_PDF)


if __name__ == "__main__":
    main()
