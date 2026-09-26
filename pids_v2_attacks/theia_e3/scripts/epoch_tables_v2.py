#!/usr/bin/env python
"""Per-detector epoch tables in the requested paper format.

One table per detector. Rows = checkpoint epoch. Columns = the three attack methods, each cell
AUC/ACC at 4 decimals. TARGET arm only -- standard MIA reporting evaluates the attack against the
target model; the shadow arm is adversary machinery and is not a reported result.

ACC here is balanced accuracy at the best threshold, (1 + max(TPR - FPR)) / 2, which is what the
metric means throughout this campaign. Stated explicitly because "ACC" in MIA papers is ambiguous
between this and accuracy at a fixed 0.5 threshold, and the two differ substantially on skewed
score distributions.
"""
from __future__ import annotations

import json

R = "/data/pids_v2_attacks/theia_e3/results"
OURS, HC, FLOOR = (json.load(open(f"{R}/{f}.json")) for f in ("ours_v2", "he_conti_v2", "floor_v2"))
SRC = {"ours": OURS, "he2021": HC, "conti2022_distilled": HC}

DETECTORS = (("NodLink", "nodlink"), ("Magic", "magic"), ("Flash", "flash"),
             ("R-CAID", "rcaid"), ("ThreaTrace", "threatrace"), ("Velox", "velox"),
             ("Orthrus", "orthrus_non_snooped"), ("Kairos", "kairos"))
EDGE = {"velox", "orthrus_non_snooped", "kairos"}
METHODS = (("He 2021", "he2021"), ("Conti 2022", "conti2022_distilled"), ("Ours", "ours"))
ARM, CKPTS = "TARGET", (49, 99, 149, 199, 299)
out = []


def w(s=""):
    out.append(s)
    print(s)


w("# MIA attack results per detector - THEIA_E3 (target models, epoch 299 = final)")
w()
for name, det in DETECTORS:
    w(f"## {name}")
    w()
    w("| Epoch | " + " | ".join(f"{lbl}(AUC/ACC)" for lbl, _ in METHODS) + " |")
    w("|---|---|---|---|")
    for ep in CKPTS:
        cells = []
        for _, key in METHODS:
            if det in EDGE and key != "ours":
                cells.append("n/a")
                continue
            r = SRC[key].get(f"{det}|{ARM}|{ep}")
            m = r[key] if r else None
            cells.append(f"{m['auc']:.4f}/{m['bal_acc']:.4f}" if m else "-")
        w(f"| {ep} | " + " | ".join(cells) + " |")
    w()

w("## Baseline (no model access)")
w()
w("Structure-only classifier: node degree, node type, window-touch count. Constant across epochs "
  "because it never queries the model. This is the number each attack above must beat to be "
  "reading the trained model rather than the data distribution.")
w()
w("| Detector | Baseline(AUC/ACC) |")
w("|---|---|")
for name, det in DETECTORS:
    fl = FLOOR[f"{det}|{ARM}"]["floor_structure"]
    w(f"| {name} | {fl['auc']:.4f}/{fl['bal_acc']:.4f} |")
w()
w("n/a: He 2021 and Conti 2022 query the model with the node's neighbourhood removed. For "
  "edge-scoring detectors (Velox, Orthrus, Kairos) the 0-hop query deletes every edge, leaving the "
  "model nothing to score, so these attacks are undefined rather than unrun.")
w()
w("ACC is balanced accuracy at the best threshold. Kairos's dump is label-dependent (two-pass "
  "split with TGN memory carry-over) and should not be cited.")
w()

open(f"{R}/EPOCH_TABLES_V2.md", "w").write("\n".join(out) + "\n")
print(f"\n-> {R}/EPOCH_TABLES_V2.md")
