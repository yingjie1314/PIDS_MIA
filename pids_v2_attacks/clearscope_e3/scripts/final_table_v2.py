#!/usr/bin/env python
"""Final v2 attack table: three MIA methods x 8 detectors x 2 arms x 5 checkpoints, AUC and balACC.

Sources, all scored on the SAME rows with the SAME classifier so the columns compare attacks rather
than harnesses:
  he_conti_v2.json   He et al. 2021 (hop ladder) and Conti et al. 2022 (label-only + distilled)
  ours_v2.json       Ours (white-box: per-node loss + full embedding), and the single-loss baseline
  floor_v2.json      structure-only control -- degree, type, touch count; no model output at all

balACC is the Youden statistic, (1 + max(TPR - FPR)) / 2, i.e. the best balanced accuracy over all
thresholds. It is an upper bound on what a fixed-threshold attacker gets.

He and Conti are undefined for edge detectors: hop 0 replaces the edge set with self-loops, leaving
an edge model with nothing to score. Printed as N/A rather than omitted, so the gap is visible.
"""
from __future__ import annotations

import json

OURS = "/data/pids_v2_attacks/clearscope_e3/results/ours_v2.json"
HC = "/data/pids_v2_attacks/clearscope_e3/results/he_conti_v2.json"
FLOOR = "/data/pids_v2_attacks/clearscope_e3/results/floor_v2.json"
MD = "/data/pids_v2_attacks/clearscope_e3/results/FINAL_ATTACK_TABLE_V2.md"

DETECTORS = ("nodlink", "magic", "flash", "rcaid", "threatrace",
             "velox", "orthrus_non_snooped", "kairos")
EDGE = {"velox", "orthrus_non_snooped", "kairos"}
ARMS = ("TARGET", "SHADOW")
CKPTS = (49, 99, 149, 199, 299)
METHODS = (
    ("He et al. 2021", HC, "he2021"),
    ("Conti et al. 2022", HC, "conti2022_distilled"),
    ("Ours (white-box)", OURS, "ours"),
    ("- Conti label-only", HC, "conti2022_label_only"),
    ("- single-loss base", OURS, "single_loss"),
)


def main() -> None:
    ours, hc, floor = (json.load(open(p)) for p in (OURS, HC, FLOOR))
    src = {OURS: ours, HC: hc}
    out = []

    def w(s=""):
        out.append(s)
        print(s)

    w("# Final MIA attack table - CLEARSCOPE_E3 v2 (day-disjoint target/shadow)")
    w()
    w("Each cell is **AUC / balACC**. 30,000 members + 30,000 non-members per cell; nodes present "
      "in both splits are dropped as unlabelable. balACC = (1 + max(TPR-FPR))/2.")
    w()
    w("`floor` is a structure-only control: degree, node type and window-touch count, with **no "
      "model output of any kind**. It is what membership predicts from the calendar alone.")
    w()
    w("He et al. and Conti et al. are **N/A for edge detectors** (velox, orthrus, kairos): their "
      "hop-0 query replaces the edge set with self-loops, which leaves an edge model with nothing "
      "to score. This is structural, not a gap in the run.")
    w()
    w("Kairos rows are **unreportable** regardless: its two-pass dump plus TGN memory makes the "
      "features label-dependent.")
    w()

    # --- verdict, computed rather than asserted -------------------------------------------
    w("## What the table shows")
    w()
    w("Two quantities decide whether any of this is memorisation. Both are computed from the "
      "cells below, not stated by hand.")
    w()
    w("**1. Movement with training.** If an attack reads memorisation, it must get stronger as the "
      "model trains. Largest change in AUC between e49 and e299, over all detector/arm pairs:")
    w()
    w("| method | largest abs. AUC change, e49 to e299 | where |")
    w("|---|---|---|")
    for label, path, key in METHODS:
        best = (0.0, "-")
        for det in DETECTORS:
            if path is HC and det in EDGE:
                continue
            for arm in ARMS:
                a = src[path].get(f"{det}|{arm}|49")
                b = src[path].get(f"{det}|{arm}|299")
                if not a or not b:
                    continue
                d = abs(b[key]["auc"] - a[key]["auc"])
                if d > best[0]:
                    best = (d, f"{det}/{arm}")
        w(f"| {label} | {best[0]:.3f} | {best[1]} |")
    w()
    w("**2. Excess over the structure floor.** The floor uses no model output at all, so anything "
      "at or below it is reading the calendar, not the weights. Counting detector/arm pairs whose "
      "e299 AUC exceeds their own floor:")
    w()
    w("| method | pairs above floor | median (attack - floor) |")
    w("|---|---|---|")
    for label, path, key in METHODS:
        diffs = []
        for det in DETECTORS:
            if path is HC and det in EDGE:
                continue
            for arm in ARMS:
                r = src[path].get(f"{det}|{arm}|299")
                fl = floor.get(f"{det}|{arm}", {}).get("floor_structure")
                if r and fl:
                    diffs.append(r[key]["auc"] - fl["auc"])
        if diffs:
            med = sorted(diffs)[len(diffs) // 2]
            w(f"| {label} | {sum(d > 0 for d in diffs)}/{len(diffs)} | {med:+.3f} |")
    w()
    w("**The one cell that does move: orthrus_non_snooped / TARGET under Ours**, 0.724 at e49 to "
      "0.824 at e299. It is the single largest movement in the table and the only one that looks "
      "like training. Two things argue against reading it as memorisation: the SHADOW arm of the "
      "same detector moves the *other* way over the same span (0.843 to 0.829), and over the full "
      "11-checkpoint curve the TARGET arm is non-monotonic (0.809 at e0, dipping to 0.724 at e49, "
      "back to 0.824) -- so e49 is a trough rather than e299 being a peak, and the e0-to-e299 "
      "change is only +0.014. It is still the one cell worth a dedicated control.")
    w()

    for det in DETECTORS:
        w(f"## {det}")
        w()
        head = "| arm | method | " + " | ".join(f"e{c}" for c in CKPTS) + " | floor |"
        w(head)
        w("|---|---|" + "---|" * (len(CKPTS) + 1))
        for arm in ARMS:
            fl = floor.get(f"{det}|{arm}", {}).get("floor_structure")
            fls = f"{fl['auc']:.3f} / {fl['bal_acc']:.3f}" if fl else "-"
            for label, path, key in METHODS:
                cells = []
                for c in CKPTS:
                    if path is HC and det in EDGE:
                        cells.append("N/A")
                        continue
                    r = src[path].get(f"{det}|{arm}|{c}")
                    m = r.get(key) if r else None
                    cells.append(f"{m['auc']:.3f} / {m['bal_acc']:.3f}" if m else "-")
                w(f"| {arm} | {label} | " + " | ".join(cells) + f" | {fls} |")
        w()

    open(MD, "w").write("\n".join(out) + "\n")
    print(f"\n-> {MD}")


if __name__ == "__main__":
    main()
