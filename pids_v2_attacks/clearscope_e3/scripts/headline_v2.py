#!/usr/bin/env python
"""Consolidated v2 result: 3 methods x 5 checkpoints x {TARGET, SHADOW}, AUC and balACC.

Collapses the per-detector table into the single answer to "what does this attack achieve".
Restricted to the 5 NODE detectors for the three-method comparison, because He and Conti are
undefined on edge detectors -- averaging them over different detector sets would compare the
detector mix rather than the attack. Ours is additionally reported over all 8 for completeness.

mean = across detectors; max = the worst-case (most leaky) detector, which is the number an
attacker actually gets to pick.
"""
from __future__ import annotations

import json

R = "/data/pids_v2_attacks/clearscope_e3/results"
OURS, HC, FLOOR = (json.load(open(f"{R}/{f}.json")) for f in ("ours_v2", "he_conti_v2", "floor_v2"))
SRC = {"ours": OURS, "single_loss": OURS,
       "he2021": HC, "conti2022_distilled": HC, "conti2022_label_only": HC}

NODE = ("nodlink", "magic", "flash", "rcaid", "threatrace")
ALL8 = NODE + ("velox", "orthrus_non_snooped", "kairos")
ARMS = ("TARGET", "SHADOW")
CKPTS = (49, 99, 149, 199, 299)
MAIN = (("He et al. 2021", "he2021"),
        ("Conti et al. 2022", "conti2022_distilled"),
        ("Ours (white-box)", "ours"))


def agg(key, dets, arm, ep):
    v = [SRC[key][f"{d}|{arm}|{ep}"][key] for d in dets if f"{d}|{arm}|{ep}" in SRC[key]]
    if not v:
        return None
    n = len(v)
    return (sum(x["auc"] for x in v) / n, sum(x["bal_acc"] for x in v) / n,
            max(v, key=lambda x: x["auc"]), n)


def floor_agg(dets, arm):
    v = [FLOOR[f"{d}|{arm}"]["floor_structure"] for d in dets if f"{d}|{arm}" in FLOOR]
    n = len(v)
    return sum(x["auc"] for x in v) / n, sum(x["bal_acc"] for x in v) / n


out = []


def w(s=""):
    out.append(s)
    print(s)


w("# Final attack result - CLEARSCOPE_E3 v2")
w()
w("Cells are **AUC / balACC**, averaged over detectors. 30k members + 30k non-members per "
  "detector-arm-epoch cell.")
w()

for title, dets in (("Three methods, 5 node detectors (He/Conti undefined on edge detectors)", NODE),):
    w(f"## {title}")
    w()
    for arm in ARMS:
        fa, fb = floor_agg(dets, arm)
        w(f"### {arm} arm")
        w()
        w("| method | " + " | ".join(f"e{c}" for c in CKPTS) + " | e299 - e49 |")
        w("|---|" + "---|" * (len(CKPTS) + 1))
        for label, key in MAIN:
            cells, aucs = [], []
            for c in CKPTS:
                a = agg(key, dets, arm, c)
                cells.append(f"{a[0]:.3f} / {a[1]:.3f}" if a else "-")
                aucs.append(a[0] if a else None)
            d = f"{aucs[-1] - aucs[0]:+.3f}" if aucs[0] and aucs[-1] else "-"
            w(f"| {label} | " + " | ".join(cells) + f" | {d} |")
        w(f"| *structure floor (no model)* | " + " | ".join([f"*{fa:.3f} / {fb:.3f}*"] * len(CKPTS))
          + " | *0.000* |")
        w()

w("## Worst-case detector at e299 (the number an attacker gets to pick)")
w()
w("| method | arm | best detector | AUC / balACC | its floor | excess |")
w("|---|---|---|---|---|---|")
for label, key in MAIN:
    for arm in ARMS:
        best, bd = None, None
        for d in NODE:
            k = f"{d}|{arm}|299"
            if k in SRC[key]:
                m = SRC[key][k][key]
                if best is None or m["auc"] > best["auc"]:
                    best, bd = m, d
        fl = FLOOR[f"{bd}|{arm}"]["floor_structure"]
        w(f"| {label} | {arm} | {bd} | {best['auc']:.3f} / {best['bal_acc']:.3f} | "
          f"{fl['auc']:.3f} / {fl['bal_acc']:.3f} | {best['auc'] - fl['auc']:+.3f} |")
w()

w("## Ours only, all 8 detectors (He/Conti have no defined query here)")
w()
for arm in ARMS:
    fa, fb = floor_agg(ALL8, arm)
    cells, aucs = [], []
    for c in CKPTS:
        a = agg("ours", ALL8, arm, c)
        cells.append(f"{a[0]:.3f} / {a[1]:.3f}" if a else "-")
        aucs.append(a[0] if a else None)
    if arm == ARMS[0]:
        w("| arm | " + " | ".join(f"e{c}" for c in CKPTS) + " | e299 - e49 | floor |")
        w("|---|" + "---|" * (len(CKPTS) + 2))
    w(f"| {arm} | " + " | ".join(cells)
      + f" | {aucs[-1] - aucs[0]:+.3f} | {fa:.3f} / {fb:.3f} |")
w()

open(f"{R}/HEADLINE_V2.md", "w").write("\n".join(out) + "\n")
print(f"\n-> {R}/HEADLINE_V2.md")
