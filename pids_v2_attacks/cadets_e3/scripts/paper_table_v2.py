#!/usr/bin/env python
"""THE paper table: one row per target detector, one column per attack method.

Standard MIA reporting convention. The shadow model is adversary machinery, not a result -- it is
used to build the attack, and the attack is then evaluated against the TARGET model. So this table
is TARGET arm only, at the final checkpoint (e299), because the deployed model is what gets
attacked. The shadow arm and the 5-checkpoint sweep are controls/ablations, not the main table.

Columns: AUC, balanced accuracy, and TPR at 1% FPR. The last one matters -- Carlini et al. 2022
showed average-case metrics hide attacks that confidently identify a small number of members, and
conversely an AUC of 0.8 with chance-level low-FPR behaviour is not a meaningful attack.

The Baseline column is a structure-only classifier (degree, node type, window-touch count) with NO
model output of any kind. It is the number to beat: anything at or below it is inferring membership
from the data distribution rather than from the trained model.
"""
from __future__ import annotations

import json

R = "/data/pids_v2_attacks/cadets_e3/results"
OURS, HC, FLOOR = (json.load(open(f"{R}/{f}.json")) for f in ("ours_v2", "he_conti_v2", "floor_v2"))
SRC = {"ours": OURS, "he2021": HC, "conti2022_distilled": HC}

DETECTORS = ("nodlink", "magic", "flash", "rcaid", "threatrace",
             "velox", "orthrus_non_snooped", "kairos")
EDGE = {"velox", "orthrus_non_snooped", "kairos"}
METHODS = (("He et al. 2021", "he2021"), ("Conti et al. 2022", "conti2022_distilled"),
           ("Ours", "ours"))
ARM, EP = "TARGET", 299
out = []


def w(s=""):
    out.append(s)
    print(s)


def fmt(m):
    return (f"{m['auc']:.3f}", f"{m['bal_acc']:.3f}", f"{100 * m.get('tpr_at_1', float('nan')):.1f}")


w("# Table N: Membership inference against provenance IDS (CADETS_E3, DARPA TC E3)")
w()
w("Attack performance against each **target** detector at the final checkpoint (epoch 299). "
  "Shadow models are used to construct the attacks and are not reported. Each cell is computed on "
  "30,000 members and 30,000 non-members; nodes appearing in both splits are dropped as "
  "unlabelable. balACC is the best balanced accuracy over all thresholds; TPR@1%FPR is the true "
  "positive rate at 1% false positive rate. **Baseline** uses graph structure only (degree, node "
  "type, touch count) with no access to the model.")
w()
hdr = "| Detector | " + " | ".join(f"{lbl} AUC | {lbl} balACC | {lbl} TPR@1%" for lbl, _ in METHODS)
w(hdr + " | Baseline AUC | Baseline balACC |")
w("|---|" + "---|" * (3 * len(METHODS) + 2))
for det in DETECTORS:
    row = [det]
    for _, key in METHODS:
        if det in EDGE and key != "ours":
            row += ["n/a", "n/a", "n/a"]
            continue
        r = SRC[key].get(f"{det}|{ARM}|{EP}")
        row += list(fmt(r[key])) if r else ["-", "-", "-"]
    fl = FLOOR[f"{det}|{ARM}"]["floor_structure"]
    row += [f"{fl['auc']:.3f}", f"{fl['bal_acc']:.3f}"]
    w("| " + " | ".join(row) + " |")
w()
w("n/a: He et al. and Conti et al. query the model with the node's k-hop neighbourhood removed. "
  "For edge-scoring detectors (velox, orthrus, kairos) the 0-hop query deletes every edge, leaving "
  "the model with nothing to score, so these attacks are undefined rather than merely unrun.")
w()
w("Kairos is reported for completeness but its dump is label-dependent (two-pass split with TGN "
  "memory carry-over) and should not be cited.")
w()

open(f"{R}/PAPER_TABLE_V2.md", "w").write("\n".join(out) + "\n")
print(f"\n-> {R}/PAPER_TABLE_V2.md")
