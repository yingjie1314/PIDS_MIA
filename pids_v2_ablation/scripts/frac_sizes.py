#!/usr/bin/env python
"""Ablation 2 -- recover the TRUE training-set size per cell.

score_frac.py reports `n_members` AFTER the 30k-per-class subsample, so it pins at 30000 for every
CADETS and THEIA cell and the size axis -- the whole point of this ablation -- is invisible there.
The node sets do not depend on the checkpoint, so one epoch per cell is enough to recover the real
counts, and doing it separately avoids re-running 121 GBDT fits to add a column.

Reports the member count before AND after the overlap drop, because the drop is not incidental: the
same process recurs across adjacent windows, so under `random` a large share of nodes appear in both
splits and are discarded as unlabelable. A cell whose surviving members are a small remnant of its
training set is not comparable to one where the drop was mild.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_frac as S  # noqa: E402

PROBE_EPOCH = 0
OUT = os.path.join(S.ROOT, "results", "frac_sizes.json")


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for ds in S.ARMS:
        for frac in ("0.1", "0.25", "0.5", "0.75"):
            key = f"{ds}|{frac}"
            if key in res:
                continue
            base = S.run_dir(ds, frac)
            if not base:
                continue
            per = {}
            for split in S.SPLITS:
                d = os.path.join(base, split, f"model_epoch_{PROBE_EPOCH}")
                got = S.O.collect(d, S.DET) if os.path.isdir(d) else None
                if got is None:
                    break
                per[split] = got
            if len(per) != len(S.SPLITS):
                continue
            nm, nn = per["mia_train_in"][0], per["mia_heldout"][0]
            ov = np.intersect1d(nm, nn)
            res[key] = {
                "n_members_raw": int(len(nm)), "n_nonmembers_raw": int(len(nn)),
                "n_overlap": int(len(ov)),
                "n_members_kept": int((~np.isin(nm, ov)).sum()),
                "n_nonmembers_kept": int((~np.isin(nn, ov)).sum()),
                "n_windows_in": len(os.listdir(os.path.join(
                    base, "mia_train_in", f"model_epoch_{PROBE_EPOCH}"))),
                "n_windows_held": len(os.listdir(os.path.join(
                    base, "mia_heldout", f"model_epoch_{PROBE_EPOCH}"))),
            }
            json.dump(res, open(OUT, "w"), indent=1)
            r = res[key]
            print(f"  {key:<22} members {r['n_members_raw']:>7} "
                  f"(kept {r['n_members_kept']:>7}) | non-members {r['n_nonmembers_raw']:>7} "
                  f"(kept {r['n_nonmembers_kept']:>7}) | overlap {r['n_overlap']:>7} | "
                  f"windows {r['n_windows_in']}/{r['n_windows_held']}", flush=True)
    print(f"{len(res)} cells -> {OUT}")


if __name__ == "__main__":
    main()
