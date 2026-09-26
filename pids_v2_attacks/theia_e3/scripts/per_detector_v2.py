#!/usr/bin/env python
"""Per-detector final results: one table per detector, 3 methods x 5 checkpoints x 2 arms,
plus a verdict computed from the cells rather than asserted.

VERDICT RULE. Calling something a leak requires memorisation, and memorisation has two
signatures that a calendar artifact does not:
  (1) it GROWS with training  -- e299 AUC must exceed e49 by >= GROW on some method/arm;
  (2) it exceeds the structure-only floor (degree, node type, touch count; no model output).
A third diagnostic, not part of the rule but reported: the two arms are independently trained
on disjoint days, so if an attack reads weights the arms should DISAGREE. Arms landing within
ARM_TOL of each other point at something shared between them -- i.e. the calendar, not the model.
"""
from __future__ import annotations

import json

R = "/data/pids_v2_attacks/theia_e3/results"
OURS, HC, FLOOR = (json.load(open(f"{R}/{f}.json")) for f in ("ours_v2", "he_conti_v2", "floor_v2"))
SRC = {"ours": OURS, "he2021": HC, "conti2022_distilled": HC}

DETECTORS = ("nodlink", "magic", "flash", "rcaid", "threatrace",
             "velox", "orthrus_non_snooped", "kairos")
EDGE = {"velox", "orthrus_non_snooped", "kairos"}
ARMS = ("TARGET", "SHADOW")
CKPTS = (49, 99, 149, 199, 299)
METHODS = (("He et al. 2021", "he2021"),
           ("Conti et al. 2022", "conti2022_distilled"),
           ("Ours (white-box)", "ours"))
GROW, ARM_TOL = 0.05, 0.05

out = []


def w(s=""):
    out.append(s)
    print(s)


def cell(key, det, arm, ep):
    r = SRC[key].get(f"{det}|{arm}|{ep}")
    return r[key] if r else None


w("# Per-detector final results - THEIA_E3 v2")
w()
w("Each cell **AUC / balACC**. 30k members + 30k non-members. `floor` = structure only "
  "(degree, node type, touch count), **no model output**. He/Conti are structurally undefined on "
  "edge detectors: their hop-0 query replaces the edge set with self-loops, leaving an edge model "
  "nothing to score.")
w()
w(f"Verdict rule: LEAK requires AUC growth >= {GROW:+.2f} from e49 to e299 **and** e299 above the "
  f"detector's own floor. Arms agreeing within {ARM_TOL:.2f} is reported as a separate diagnostic "
  "-- two models trained on disjoint days should not agree if the attack reads weights.")
w()

summary = []
for det in DETECTORS:
    w(f"## {det}")
    w()
    w("| arm | method | " + " | ".join(f"e{c}" for c in CKPTS) + " | Δ(299−49) | floor | excess |")
    w("|---|---|" + "---|" * (len(CKPTS) + 3))
    grow_best, exc_best, arm_gaps = -9.0, -9.0, []
    for arm in ARMS:
        fl = FLOOR[f"{det}|{arm}"]["floor_structure"]
        for label, key in METHODS:
            if det in EDGE and key != "ours":
                w(f"| {arm} | {label} | " + " | ".join(["N/A"] * len(CKPTS))
                  + f" | N/A | {fl['auc']:.3f} / {fl['bal_acc']:.3f} | N/A |")
                continue
            vals = [cell(key, det, arm, c) for c in CKPTS]
            cells = [f"{v['auc']:.3f} / {v['bal_acc']:.3f}" if v else "-" for v in vals]
            d = vals[-1]["auc"] - vals[0]["auc"] if vals[0] and vals[-1] else None
            exc = vals[-1]["auc"] - fl["auc"] if vals[-1] else None
            grow_best = max(grow_best, d if d is not None else -9.0)
            exc_best = max(exc_best, exc if exc is not None else -9.0)
            w(f"| {arm} | {label} | " + " | ".join(cells)
              + f" | {d:+.3f} | {fl['auc']:.3f} / {fl['bal_acc']:.3f} | {exc:+.3f} |")
    for label, key in METHODS:
        if det in EDGE and key != "ours":
            continue
        t, s = cell(key, det, "TARGET", 299), cell(key, det, "SHADOW", 299)
        if t and s:
            arm_gaps.append((abs(t["auc"] - s["auc"]), label))
    w()
    leak = grow_best >= GROW and exc_best > 0
    verdict = "LEAK" if leak else "NO LEAK"
    gap_txt = ", ".join(f"{lbl} {g:.3f}" for g, lbl in arm_gaps)
    w(f"**Verdict: {verdict}.** Best growth over training {grow_best:+.3f} "
      f"(need {GROW:+.2f}); best e299 excess over floor {exc_best:+.3f}. "
      f"TARGET-vs-SHADOW gap at e299: {gap_txt}.")
    if det == "kairos":
        w()
        w("> Kairos rows are **unreportable** regardless of the numbers: its two-pass `_dump_split` "
          "plus TGN memory makes the dumped features label-dependent.")
    w()
    summary.append((det, verdict, grow_best, exc_best,
                    min(arm_gaps)[0] if arm_gaps else None))

w("## Summary")
w()
w("| detector | verdict | best growth e49→e299 | best excess over floor | closest arm gap |")
w("|---|---|---|---|---|")
for det, v, g, e, a in summary:
    w(f"| {det} | {v} | {g:+.3f} | {e:+.3f} | {a:.3f} |" if a is not None
      else f"| {det} | {v} | {g:+.3f} | {e:+.3f} | - |")
w()

open(f"{R}/PER_DETECTOR_V2.md", "w").write("\n".join(out) + "\n")
print(f"\n-> {R}/PER_DETECTOR_V2.md")
