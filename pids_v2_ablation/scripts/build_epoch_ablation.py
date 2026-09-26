#!/usr/bin/env python
"""Ablation 1 -- early-stop ladder. Does training longer make a detector leak more?

Scores the full epoch ladder 0..299 for all three datasets and both attack families, then reports
the only quantity that answers the question: AUC(e) - AUC(0), the *training-attributable* lift.

Why the raw curve is not enough. An attack scoring 0.94 at epoch 299 looks like catastrophic
leakage until you see it scored 0.94 at epoch 0 as well, after a single training epoch.
That cell is reading the input representation -- node degree, type, which window a node fell in --
not anything the model learned. Early stopping cannot remove a signal that was never learned, so
subtracting e0 is what separates "stop early to leak less" from "this axis does nothing".

The e0 column doubles as the ladder's own zero point, so no separate control file is needed.
"""
from __future__ import annotations

import json
import os

ROOT = "/data/pids_v2_ablation"
SRC = "/data/pids_v2_attacks"
DATASETS = ("cadets_e3", "theia_e3", "clearscope_e3")
LADDER = (0, 1, 2, 5, 10, 24, 49, 99, 149, 199, 299)
NODE = ("nodlink", "magic", "flash", "threatrace", "rcaid")
ALL = NODE + ("velox", "orthrus_non_snooped", "kairos")
ARMS = ("TARGET", "SHADOW")
# kairos dumps its features in two passes with TGN memory carried across them, which makes the
# features label-dependent; its rows are excluded from every aggregate rather than quietly averaged.
STATEFUL = {"kairos"}


def load(ds, *names):
    """Later names win. Pass the ablation file LAST so its freshly scored ladder overrides the
    published 49..299 cells rather than being silently overwritten by them."""
    merged = {}
    for n in names:
        for base in (os.path.join(ROOT, ds, "results"), os.path.join(SRC, ds, "results")):
            p = os.path.join(base, n)
            if os.path.exists(p):
                merged.update(json.load(open(p)))
    return merged


def he_best(cell, keys):
    if not cell:
        return None
    c = [cell[k] for k in keys if k in cell]
    return max(c, key=lambda d: d["auc"]) if c else None


def main():
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    doc = ["# Ablation 1 -- early stopping vs membership leakage",
           "",
           "Three datasets, the full checkpoint ladder "
           f"`{', '.join(str(e) for e in LADDER)}`, both arms.",
           "Every cell is **AUC / balanced-ACC**. `e0` is the checkpoint dumped at the END of the first training epoch (training_loop.py:283-293), not a randomly-initialised model.",
           "",
           "**Read the delta column, not the curve.** An attack that already scores high at epoch 0 "
           "is reading\nthe input representation, not the weights -- early stopping cannot take away "
           "a signal training\nnever put there. Only `e299 - e0` is attributable to training, and "
           "only a ladder that *rises*\nwith epoch makes early stopping a usable defence.",
           ""]

    summary = []   # (ds, det, arm, method, e0, peak_e, peak_auc, final_auc)

    for ds in DATASETS:
        ours = load(ds, "ours_v2.json", "ours_v2_early.json", "ours_epochs.json")
        he = load(ds, "he_conti_v2.json", "he_conti_v2_early.json", "he_conti_ablation.json")
        floor = load(ds, "floor_v2.json")
        doc += [f"\n## {ds.upper()}\n",
                "| detector | arm | method | floor | " +
                " | ".join(f"e{e}" for e in LADDER) + " | e299-e0 |",
                "|---|---|---|" + "---|" * (len(LADDER) + 1)]

        for det in ALL:
            for arm in ARMS:
                fl = floor.get(f"{det}|{arm}", {}).get("floor_structure")
                if fl is None:
                    continue
                tag = f"{det} *" if det in STATEFUL else det
                methods = [("Ours", lambda e, d=det, a=arm:
                            (ours.get(f"{d}|{a}|{e}") or {}).get("ours"))]
                if det in NODE:
                    methods += [
                        ("He 2021", lambda e, d=det, a=arm:
                         he_best(he.get(f"{d}|{a}|{e}"), ("he2021",))),
                        ("Conti 2022", lambda e, d=det, a=arm:
                         he_best(he.get(f"{d}|{a}|{e}"),
                                 ("conti2022_label_only", "conti2022_distilled"))),
                    ]
                for mi, (label, get) in enumerate(methods):
                    vals = [get(e) for e in LADDER]
                    cells = [f"{v['auc']:.3f} / {v['bal_acc']:.3f}" if v else "--" for v in vals]
                    z, f_ = vals[0], vals[-1]
                    delta = f"{f_['auc'] - z['auc']:+.3f}" if (z and f_) else "--"
                    head = f"| {tag} | {arm} |" if mi == 0 else "|  |  |"
                    fcell = f"{fl['auc']:.3f} / {fl['bal_acc']:.3f}" if mi == 0 else ""
                    doc.append(f"{head} {label} | {fcell} | " + " | ".join(cells) +
                               f" | {delta} |")
                    if z and f_ and det not in STATEFUL:
                        seen = [(e, v) for e, v in zip(LADDER, vals) if v]
                        pe, pv = max(seen, key=lambda t: t[1]["auc"])
                        summary.append((ds, det, arm, label, z["auc"], pe, pv["auc"], f_["auc"]))
                if det not in NODE:
                    doc.append("|  |  | He 2021 / Conti 2022 | | " +
                               " | ".join("N/A" for _ in LADDER) + " | N/A |")

    doc += ["",
            "`*` kairos is excluded from every aggregate below: its two-pass dump carries TGN "
            "memory across\nthe member/non-member split, so its features are label-dependent.  ",
            "`N/A` for edge detectors under He/Conti: hop 0 replaces the edge set with self-loops, "
            "leaving an\nedge-scoring model nothing to answer. Undefined, not missing.",
            "",
            "## Does training longer increase leakage?",
            "",
            "| dataset | method | arms | mean e0 | mean e299 | mean lift | arms where training "
            "helped the attack |",
            "|---|---|---|---|---|---|---|"]

    for ds in DATASETS:
        for label in ("Ours", "He 2021", "Conti 2022"):
            g = [s for s in summary if s[0] == ds and s[3] == label]
            if not g:
                continue
            z = sum(s[4] for s in g) / len(g)
            f_ = sum(s[7] for s in g) / len(g)
            up = sum(1 for s in g if s[7] - s[4] > 0.02)
            doc.append(f"| {ds} | {label} | {len(g)} | {z:.3f} | {f_:.3f} | {f_-z:+.4f} | "
                       f"{up}/{len(g)} |")

    rose = [s for s in summary if s[7] - s[4] >= 0.10 and s[7] > 0.60]
    doc += ["",
            "## Cells where early stopping would actually help",
            "",
            "A defence only exists where training *created* the signal: `e299 - e0 >= 0.10` and the "
            "trained\nattack beats 0.60. Everywhere else the ladder is flat and stopping early "
            "changes nothing.",
            ""]
    if rose:
        doc.append("| dataset | detector | arm | method | e0 | peak | e299 | lift |")
        doc.append("|---|---|---|---|---|---|---|---|")
        for ds, det, arm, label, z, pe, pv, f_ in sorted(rose, key=lambda s: s[4] - s[7]):
            doc.append(f"| {ds} | {det} | {arm} | {label} | {z:.3f} | {pv:.3f} @e{pe} | "
                       f"{f_:.3f} | **{f_-z:+.3f}** |")
    else:
        doc.append("**None.** No cell on any dataset gains >= 0.10 AUC from training, so early "
                   "stopping has no\nleakage to remove.")

    md = "\n".join(doc) + "\n"
    open(os.path.join(ROOT, "results", "ABLATION_1_EARLY_STOP.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
