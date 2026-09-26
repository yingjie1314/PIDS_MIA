#!/usr/bin/env python
"""The three-method attack table for CLEARSCOPE_E3, in the format used for CADETS and THEIA.

Reads the three scorers' JSON and emits one table plus the two controls that decide how it reads:
the structure floor (what membership is worth with NO model output) and the epoch-0 column (what
the attack scores against an untrained model). A high AUC means nothing without both.

Conti is reported as the better of its label-only and distilled variants per cell, which favours
the attack -- the honest direction when the claim being tested is "these detectors do not leak".

Edge detectors get N/A for He/Conti rather than a blank or an approximation: hop 0 replaces the
edge set with self-loops, so an edge-scoring model has nothing to score and the ladder is undefined.
"""
from __future__ import annotations

import json
import os

R = "/data/pids_v2_attacks/clearscope_e3/results"
DETECTORS = ("nodlink", "magic", "flash", "threatrace", "rcaid",
             "velox", "orthrus_non_snooped", "kairos")
EDGE = {"velox", "orthrus_non_snooped", "kairos"}
STATEFUL = {"kairos"}
ARMS = ("TARGET", "SHADOW")
CKPTS = (49, 99, 149, 199, 299)


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def fmt(m):
    return f"{m['auc']:.3f} / {m['bal_acc']:.3f}" if m else "--"


def main():
    ours, early, floor, he = (load("ours_v2.json"), load("ours_v2_early.json"),
                              load("floor_v2.json"), load("he_conti_v2.json"))
    he_early = load("he_conti_v2_early.json")

    out = ["# Membership inference against provenance IDS -- CLEARSCOPE_E3 (DARPA TC E3)",
           "",
           "Day-disjoint target/shadow arms, 300 epochs, five checkpoints. Every cell is "
           "**AUC / balanced-ACC**.",
           "",
           "- **Ours** -- per-node GBDT on the dumped representation, within-arm 5-fold CV. An "
           "*upper bound, not a\n  mountable attack*: it sees the arm's own labels, which no real "
           "adversary has.",
           "- **He 2021** -- node-level MIA; adversary reads the model's numeric answer at each hop "
           "of the query ladder.",
           "- **Conti 2022** -- label-only; adversary reads one bit per query (flagged / not "
           "flagged).\n  Reported as the better of the paper's `label_only` and `distilled` "
           "variants per cell, which favours the attack.",
           "- **Floor** -- structure only (degree, window count, node type). No model output "
           "whatsoever.",
           "- **e0** -- Ours against the *untrained* epoch-0 checkpoint.",
           "",
           "| detector | arm | method | floor | e0 | " +
           " | ".join(f"e{e}" for e in CKPTS) + " |",
           "|---|---|---|---|---|" + "---|" * len(CKPTS)]

    lifts = {"Ours": [], "He 2021": [], "Conti 2022": []}
    lifts_ex = {k: [] for k in lifts}   # same, excluding rcaid's artifact floor
    trained = {k: [] for k in lifts}    # (untrained, trained) pairs -- what training actually buys
    e0_pairs = []

    for det in DETECTORS:
        for arm in ARMS:
            fl = floor.get(f"{det}|{arm}", {}).get("floor_structure")
            if fl is None:
                continue
            tag = f"{det} ᵃ" if det in STATEFUL else (f"{det} ᶜ" if det == "rcaid" else det)
            e0 = early.get(f"{det}|{arm}|0", {}).get("ours")
            row = [f"| {tag} | {arm} | Ours | {fmt(fl)} | {fmt(e0)}"]
            for e in CKPTS:
                m = ours.get(f"{det}|{arm}|{e}", {}).get("ours")
                row.append(fmt(m))
                if e == CKPTS[-1] and m:
                    lifts["Ours"].append(m["auc"] - fl["auc"])
                    if det != "rcaid":
                        lifts_ex["Ours"].append(m["auc"] - fl["auc"])
                    if e0:
                        e0_pairs.append((e0["auc"], m["auc"]))
                        trained["Ours"].append((e0["auc"], m["auc"], det, arm))
            out.append(" | ".join(row) + " |")

            if det in EDGE:
                out.append("|  |  | He 2021 / Conti 2022 |  |  | " +
                           " | ".join("N/A ᵇ" for _ in CKPTS) + " |")
                continue

            for label, keys in (("He 2021", ("he2021",)),
                                ("Conti 2022", ("conti2022_label_only", "conti2022_distilled"))):
                def best(src, e):
                    c = src.get(f"{det}|{arm}|{e}")
                    return max((c[k] for k in keys if k in c), key=lambda d: d["auc"]) if c else None

                z = best(he_early, 0)
                cells = []
                for e in CKPTS:
                    m = best(he, e)
                    cells.append(fmt(m))
                    if e == CKPTS[-1] and m:
                        lifts[label].append(m["auc"] - fl["auc"])
                        if det != "rcaid":
                            lifts_ex[label].append(m["auc"] - fl["auc"])
                        if z:
                            trained[label].append((z["auc"], m["auc"], det, arm))
                out.append(f"|  |  | {label} |  | {fmt(z)} | " + " | ".join(cells) + " |")

    out += ["",
            "ᵃ kairos rows flagged `unreportable` by the scorer -- its two-pass dump (TGN "
            "memory + `reset_state`\nbetween splits) makes features label-dependent.  ",
            "ᵇ Undefined for edge detectors: hop 0 replaces the edge set with self-loops, "
            "leaving an edge\ndetector with nothing to score. Not approximable.",
            "",
            "ᶜ **rcaid's floor (0.986) is a dump-geometry artifact and must not be read as a "
            "structure floor.**\nMeasured on TARGET/e299: no single structure feature separates "
            "the arms (touch 0.500, degree 0.530,\nnode_type 0.519 by AUC), but a GBDT on degree "
            "ALONE reaches 0.977. The cause is that the floor's\n`degree` accumulates over "
            "windows, and rcaid is the one detector here running the windowed "
            "low-memory\nloader, whose windows vary enormously in size -- only 678 distinct "
            "degree values over 63,658 rows,\n348 of them exclusive to members and 211 exclusive "
            "to non-members. Degree is therefore acting as a\nwindow-size fingerprint, and "
            "windows partition cleanly by day, so the GBDT is recovering the\ncalendar rather "
            "than anything structural. rcaid's own attacks sit at 0.53 -- *below* this floor -- "
            "so\nthe artifact does not manufacture a leak, but it does corrupt any mean taken "
            "over it.",
            "",
            "## Summary -- mean lift over the structure floor at epoch 299",
            "",
            "| method | arms | mean lift vs floor | excl. rcaid ᶜ |", "|---|---|---|---|"]
    for k, v in lifts.items():
        if v:
            w = lifts_ex[k]
            ex = f"{sum(w)/len(w):+.4f} (n={len(w)})" if w else "--"
            out.append(f"| {k} | {len(v)} | {sum(v)/len(v):+.4f} | {ex} |")

    if e0_pairs:
        u = sum(a for a, _ in e0_pairs) / len(e0_pairs)
        t = sum(b for _, b in e0_pairs) / len(e0_pairs)
        tail = ("training does not add separability at all -- the untrained model is the "
                "*better* attack surface" if t < u else
                f"{100*(t-u)/max(t-0.5, 1e-9):.1f}% of the lift over chance")
        out += ["", f"**Epoch-0 control (Ours, {len(e0_pairs)} arms):** untrained {u:.4f} vs "
                    f"trained {t:.4f} -- training contributes **{t-u:+.4f}**; {tail}."]

    out += ["", "## Training-attributable lift (e299 minus e0), per method and arm",
            "",
            "The only number that can distinguish memorisation from geometry. A cell is evidence of "
            "a leak\nonly if training MOVED it: an attack that already scores high against the "
            "untrained model is\nreading the input representation, not the weights.",
            "",
            "| method | detector | arm | e0 (untrained) | e299 (trained) | delta |", "|---|---|---|---|---|---|"]
    flagged = []
    for label, pairs in trained.items():
        for z, m, det, arm in sorted(pairs, key=lambda p: p[1] - p[0], reverse=True):
            mark = " **<-**" if m - z >= 0.10 and m > 0.60 else ""
            if mark:
                flagged.append((label, det, arm, z, m))
            out.append(f"| {label} | {det} | {arm} | {z:.3f} | {m:.3f} | {m-z:+.3f}{mark} |")

    out += [""]
    if flagged:
        out.append("**Cells where training adds >= 0.10 AUC and the trained attack beats 0.60 "
                   "-- the only\ncandidates for genuine memorisation on this dataset:**")
        out.append("")
        for label, det, arm, z, m in flagged:
            out.append(f"- `{det}` / {arm} / {label}: {z:.3f} -> {m:.3f} (**{m-z:+.3f}**)")
    else:
        out.append("**No cell clears the bar.** Every attack that scores high scores just as high "
                   "against the\nuntrained model, so nothing here is memorisation.")

    md = "\n".join(out) + "\n"
    open(os.path.join(R, "ATTACK_TABLE_CLEARSCOPE_3METHOD.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
