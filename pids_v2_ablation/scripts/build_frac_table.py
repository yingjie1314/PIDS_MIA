#!/usr/bin/env python
"""Ablation 2 -- training-set size vs membership leakage, under the calendar control.

These are the campaign's first cells where members and non-members come from the SAME days
(`mia_split_mode=random`), so a lift here cannot be explained by day-distribution shift the way
every `xrole` result can. That makes the epoch-0 column do double duty: it removes the graph-
geometry floor, and the split removes the calendar. What survives both is membership.

Two scores per cell:
  single  -- per-node loss alone, the weakest and most mountable signal.
  Ours    -- loss + full embedding, within-arm 5-fold CV. An UPPER BOUND, not mountable.

`members` is reported per cell because it is the ablation's whole point and because it is badly
unbalanced at small fractions: frac=0.1 leaves ~1.1k member nodes against 30k non-members on
CLEARSCOPE. AUC is insensitive to that, but the small-frac cells rest on far fewer rows and should
not be read as equal in weight to the large-frac ones.
"""
from __future__ import annotations

import json
import os

ROOT = "/data/pids_v2_ablation"
DATASETS = ("cadets_e3", "theia_e3", "clearscope_e3")
FRACS = ("0.1", "0.25", "0.5", "0.75")
LADDER = (0, 1, 2, 5, 10, 24, 49, 99, 149, 199, 299)


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):  # average ranks within a tie group
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan")


def main():
    res = json.load(open(os.path.join(ROOT, "results", "frac_ours.json")))
    # True per-cell sizes. score_frac's own n_members is post-subsample and pins at 30000, which
    # hides the very axis this ablation varies; frac_sizes.py recovers the real counts.
    sz_path = os.path.join(ROOT, "results", "frac_sizes.json")
    sz = json.load(open(sz_path)) if os.path.exists(sz_path) else {}
    doc = ["# Ablation 2 -- training-set fraction vs membership leakage",
           "",
           "NodLink, `mia_split_mode=random`, seed 0, 300 epochs. Members and non-members are "
           "windows from\nthe **same days**, so the calendar confound that limits every `xrole` "
           "result in this project\ndoes not apply here.",
           "",
           "Every cell is **AUC / balanced-ACC**. `e0` is the checkpoint after ONE training epoch, "
           "not an\nuntrained model: the lift over it is what epochs 1-299 add.",
           ""]

    rows = []
    for ds in DATASETS:
        have = [f for f in FRACS if any(k.startswith(f"{ds}|{f}|") for k in res)]
        if not have:
            continue
        doc += [f"\n## {ds}\n",
                "| frac | members | non-members | score | " +
                " | ".join(f"e{e}" for e in LADDER) + " | last-e0 |",
                "|---|---|---|---|" + "---|" * (len(LADDER) + 1)]
        for f in have:
            cells = {e: res.get(f"{ds}|{f}|{e}") for e in LADDER}
            any_c = next((c for c in cells.values() if c), None)
            if not any_c:
                continue
            # A cell whose run is still training has no e299, and reporting its e10-e0 delta under
            # an "e299-e0" heading would understate a curve that has not finished rising. The last
            # scored rung is named whenever it is not 299.
            last_e = next((e for e in reversed(LADDER) if cells[e]), None)
            for i, key in enumerate(("single_loss", "ours")):
                label = "single" if key == "single_loss" else "**Ours**"
                vals = [cells[e][key] if cells[e] else None for e in LADDER]
                txt = [f"{v['auc']:.3f} / {v['bal_acc']:.3f}" if v else "--" for v in vals]
                z, last = vals[0], next((v for v in reversed(vals) if v), None)
                if z and last:
                    d = f"{last['auc'] - z['auc']:+.3f}"
                    if last_e != LADDER[-1]:
                        d += f" (to e{last_e})"
                else:
                    d = "--"
                head = (f"| {f} | {any_c['n_members']} | {any_c['n_nonmembers']} |"
                        if i == 0 else "|  |  |  |")
                doc.append(f"{head} {label} | " + " | ".join(txt) + f" | {d} |")
                if z and last:
                    rows.append((ds, f, key, z["auc"], last["auc"],
                                 any_c["n_members"], last_e))

    doc += ["",
            "## Does a smaller training set leak more?",
            "",
            "Per-sample memorisation should grow as the training set shrinks. If leakage is instead "
            "an\nartifact of representation geometry, the lift stays flat across fractions.",
            "",
            "**`members` here is the true count of labelable member nodes**, not the 30k-per-class "
            "subsample\nused for fitting -- the subsample saturates on CADETS and THEIA and would "
            "hide the size axis\nentirely. `frac` is only a within-dataset proxy; the count is what "
            "compares across datasets.",
            "",
            "| dataset | frac | members | non-members | last rung | single: e0 -> last | "
            "Ours: e0 -> last |",
            "|---|---|---|---|---|---|---|"]
    pooled = []
    for ds in DATASETS:
        for f in FRACS:
            s = next((r for r in rows if r[0] == ds and r[1] == f and r[2] == "single_loss"), None)
            o = next((r for r in rows if r[0] == ds and r[1] == f and r[2] == "ours"), None)
            if not s and not o:
                continue
            sc = f"{s[3]:.3f} -> {s[4]:.3f} ({s[4]-s[3]:+.3f})" if s else "--"
            oc = f"{o[3]:.3f} -> {o[4]:.3f} ({o[4]-o[3]:+.3f})" if o else "--"
            n, le = (s[5], s[6]) if s else (o[5], o[6])
            z = sz.get(f"{ds}|{f}")
            nm = f"{z['n_members_kept']:,}" if z else f"{n} (capped)"
            nn = f"{z['n_nonmembers_kept']:,}" if z else "--"
            flag = f"e{le}" + ("" if le == LADDER[-1] else " **(incomplete)**")
            doc.append(f"| {ds} | {f} | {nm} | {nn} | {flag} | {sc} | {oc} |")
            if z and s and le == LADDER[-1]:
                pooled.append((z["n_members_kept"], ds, f, s[4] - s[3],
                               (o[4] - o[3]) if o else None))

    if pooled:
        doc += ["",
                "### Pooled across datasets, ordered by training-set size",
                "",
                "The same cells sorted by member count rather than grouped by dataset. If the lift "
                "were a\nproperty of a dataset it would cluster by dataset; if it is per-sample "
                "memorisation it should\ndecay with size regardless of which dataset the cell came "
                "from.",
                "",
                "| members | dataset | frac | single-loss lift | Ours lift |",
                "|---|---|---|---|---|"]
        for n, ds, f, sl, ol in sorted(pooled):
            oc = f"{ol:+.3f}" if ol is not None else "--"
            doc.append(f"| {n:,} | {ds} | {f} | **{sl:+.3f}** | {oc} |")
        rho = spearman([p[0] for p in pooled], [p[3] for p in pooled])
        doc += ["", f"**Spearman rho = {rho:+.3f}** between training-set size and single-loss "
                    f"lift, over all {len(pooled)} cells.",
                "Rank, not pearson, because the relationship is expected to be monotone but not "
                "linear and the\nsizes span two orders of magnitude."]

    # An incomplete cell cannot be called a non-leaker: its curve may still be rising. Only cells
    # that reached e299 are eligible for the verdict.
    n_partial = len({(r[0], r[1]) for r in rows if r[6] != LADDER[-1]})
    rows = [r for r in rows if r[6] == LADDER[-1]]
    lifts = [(r[0], r[1], r[2], r[4] - r[3]) for r in rows]
    big = [l for l in lifts if l[3] >= 0.10]
    doc += ["",
            "## Verdict",
            ""]
    if n_partial:
        doc += [f"_{n_partial} cell(s) have not reached e299 and are excluded from the verdict "
                "below; their\ncurves may still be rising._", ""]
    if big:
        doc.append("Cells gaining >= 0.10 AUC from training **with members and non-members drawn "
                   "from the same days** --\nthese cannot be explained by the calendar:")
        doc.append("")
        for ds, f, key, d in sorted(big, key=lambda x: -x[3]):
            doc.append(f"- `{ds}` frac={f} / {'single loss' if key == 'single_loss' else 'Ours'}: "
                       f"**{d:+.3f}**")
    else:
        doc.append("**No cell gains >= 0.10 AUC from training once members and non-members share "
                   "the same days.**\nThe leakage visible in the day-disjoint tables does not "
                   "survive the calendar control.")

    md = "\n".join(doc) + "\n"
    open(os.path.join(ROOT, "results", "ABLATION_2_FRACTION.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
