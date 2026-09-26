#!/usr/bin/env python
"""Ground-truth tests for the three ported MIAs, before they are pointed at the v2 campaign.

Reading the code cannot tell you whether an attack is oriented correctly, whether a cross-validated
score leaks the labels it is predicting, or whether a shadow-transfer row means what its docstring
claims. Each of those is decidable by feeding the method data whose answer is known in advance:

  NULL       labels carry no information  -> every method must land on AUC 0.50
  POSITIVE   members reconstruct better   -> every method must land ABOVE 0.50
  INVERTED   members reconstruct worse    -> threshold methods must land BELOW 0.50
  LEAKAGE    features are pure noise      -> within-CV must land on 0.50, not above

  TRANSFER   shadow model scores target    -> sign is set by the DESIGN, not by the attack

The INVERTED case is the one worth stating plainly: a method that reported |AUC - 0.5| would pass
NULL and POSITIVE and still be wrong, because it would call a detector that reconstructs members
WORSE a leak. Only a signed test catches that.

The TRANSFER cases need a node population that carries DAY IDENTITY separately from membership --
see make_paired. Without it a transfer test reads high in every design and proves nothing.

Tests the real functions from the no-frac campaign, imported rather than copied, so what is
validated here is the code that would actually run.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/data/pids_nofrac_tuned/cadets_e3/scripts")
import compare_attacks as C  # noqa: E402

N = 20_000
EMB_DIM = 16
TOL = 0.03  # at n=20k a true 0.50 sits well inside +/-0.02; 0.03 leaves headroom without excusing a real bias
DAY_AMP = 0.5  # day identity must dominate memorisation, as it does in the real no-frac dumps


def make_run(rng, y, *, sep: float, noise_only: bool = False) -> dict:
    """A synthetic run dict shaped exactly like `load_run`'s output.

    `sep` is how much LOWER a member's loss runs. Positive sep = the memorisation direction
    (members reconstruct better). Negative sep = the inverted world. sep = 0 = no signal at all.
    """
    base = rng.normal(0.0, 1.0, N) - sep * y
    hops = {h: base + rng.normal(0, 0.15, N) for h in C.HOPS}
    if noise_only:
        emb = rng.normal(0, 1, (N, EMB_DIM))
        x = rng.normal(0, 1, (N, 8))
    else:
        emb = rng.normal(0, 1, (N, EMB_DIM)) - (sep * y)[:, None] * 0.5
        x = rng.normal(0, 1, (N, 8)) - (sep * y)[:, None] * 0.5
    return {
        "y": y, "loss": base.copy(), "hops": hops, "emb": emb, "x": x,
        "geom": np.stack([rng.integers(1, 50, N).astype(float),
                          rng.integers(0, 3, N).astype(float)], axis=1),
    }


def make_paired(rng, d, *, y_t, y_s, sep: float = 1.0) -> tuple[dict, dict]:
    """Two models scored over the SAME nodes, for the transfer tests.

    `d` is the node's day group. It is a property of the NODE, so both models see the same value.
    Membership is a property of the MODEL: y_t and y_s say which nodes each one trained on. Keeping
    those two things separate is the whole point. The loss carries memorisation and so flips with
    the roles; the embedding carries day identity and does not. Day identity is therefore what
    survives a transfer, and its sign is set entirely by how the design maps days onto membership.
    """
    day_emb = rng.normal(0, 1, (N, EMB_DIM)) + DAY_AMP * d[:, None]
    day_x = rng.normal(0, 1, (N, 8)) + DAY_AMP * d[:, None]

    def one(y: np.ndarray) -> dict:
        base = rng.normal(0, 1, N) - sep * y
        return {
            "y": y, "loss": base.copy(),
            "hops": {h: base + rng.normal(0, 0.15, N) for h in C.HOPS},
            "emb": day_emb + rng.normal(0, 0.3, (N, EMB_DIM)),
            "x": day_x + rng.normal(0, 0.3, (N, 8)),
            "geom": np.stack([rng.integers(1, 50, N).astype(float),
                              rng.integers(0, 3, N).astype(float)], axis=1),
        }

    return one(np.asarray(y_t)), one(np.asarray(y_s))


def ours_features(run: dict) -> np.ndarray:
    return np.hstack([run["loss"].reshape(-1, 1),
                      np.linalg.norm(run["emb"], axis=1, keepdims=True), run["emb"]])


def score_all(run: dict, y: np.ndarray, q: float = 0.10) -> dict:
    """The three attacks plus the plain loss threshold, scored the way the campaign scores them."""
    return {
        "single_loss": roc_auc_score(y, -run["hops"]["1"]),
        "he2021": roc_auc_score(y, C._within_cv(C.he_features(run), y)),
        "conti2022": roc_auc_score(y, C.conti_label_only(run, q)),
        "conti2022_distilled": roc_auc_score(y, C.conti_distilled(run, q)),
        "ours": roc_auc_score(y, C._within_cv(ours_features(run), y)),
    }


FAILURES: list[str] = []


def check(name: str, got: float, lo: float, hi: float) -> None:
    ok = lo <= got <= hi
    print(f"    {'PASS' if ok else 'FAIL'}  {name:<22} {got:.4f}   expected [{lo:.2f}, {hi:.2f}]")
    if not ok:
        FAILURES.append(f"{name}: {got:.4f} not in [{lo:.2f}, {hi:.2f}]")


def main() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(N) < 0.5).astype(int)

    print("\n[1] NULL -- features carry real signal, labels shuffled. Everything must read 0.50.")
    print("    Anything above chance here is the evaluation path leaking, not an attack.")
    run = make_run(rng, y, sep=1.0)
    y_shuf = rng.permutation(y)
    for k, v in score_all(run, y_shuf).items():
        check(k, v, 0.5 - TOL, 0.5 + TOL)

    print("\n[2] LEAKAGE -- features are pure noise, labels real. Within-CV must not self-fit.")
    noise = make_run(rng, y, sep=0.0, noise_only=True)
    check("he2021_within_cv", roc_auc_score(y, C._within_cv(C.he_features(noise), y)),
          0.5 - TOL, 0.5 + TOL)
    check("ours_within_cv", roc_auc_score(y, C._within_cv(ours_features(noise), y)),
          0.5 - TOL, 0.5 + TOL)

    print("\n[3] POSITIVE -- members reconstruct BETTER (the memorisation direction).")
    print("    Every method must read ABOVE 0.50. Conti is coarse by construction (3 score levels)")
    print("    so it gets a weaker bound than the continuous methods.")
    pos = make_run(rng, y, sep=1.0)
    s = score_all(pos, y)
    check("single_loss", s["single_loss"], 0.70, 1.0)
    check("he2021", s["he2021"], 0.70, 1.0)
    check("conti2022", s["conti2022"], 0.53, 1.0)
    check("conti2022_distilled", s["conti2022_distilled"], 0.52, 1.0)
    check("ours", s["ours"], 0.70, 1.0)

    print("\n[4] INVERTED -- members reconstruct WORSE. Threshold methods must read BELOW 0.50.")
    print("    A method scoring |AUC-0.5| would pass tests 1 and 3 and still be wrong here.")
    neg = make_run(rng, y, sep=-1.0)
    check("single_loss", roc_auc_score(y, -neg["hops"]["1"]), 0.0, 0.30)
    check("conti2022", roc_auc_score(y, C.conti_label_only(neg, 0.10)), 0.0, 0.47)

    # Transfer can only be tested with a DAY-IDENTITY component that is shared by both models.
    # Memorisation lives in the loss and flips with the roles; day identity lives in the node's own
    # content and does NOT, so it is the part that transfers with a sign set by the design. A test
    # that models only the loss signal measures nothing about transfer -- it reads high in every
    # design, which is how the first version of this file passed while testing the wrong thing.
    print("\n[5] TRANSFER, INVERTED DESIGN -- shadow's members ARE the target's non-members")
    print("    (the no-frac design). Day identity transfers backwards, so this reads BELOW 0.50.")
    d = (rng.random(N) < 0.5).astype(int)
    tgt_i, shd_i = make_paired(rng, d, y_t=d, y_s=1 - d)
    check("transfer_inverted",
          roc_auc_score(d, C._transfer(ours_features(shd_i), 1 - d, ours_features(tgt_i))),
          0.0, 0.40)

    print("\n[6] TRANSFER, ALIGNED DESIGN -- both arms call the same nodes members.")
    print("    Identical code, opposite reading.")
    tgt_a, shd_a = make_paired(rng, d, y_t=d, y_s=d)
    check("transfer_aligned",
          roc_auc_score(d, C._transfer(ours_features(shd_a), d, ours_features(tgt_a))), 0.60, 1.0)

    print("\n[7] TRANSFER, v2's ACTUAL DESIGN -- labels agree on 65% of common nodes (measured).")
    print("    Lands between the two, and no property of the OUTPUT distinguishes the three cases.")
    print("    The design therefore has to be declared and checked, never inferred from the score.")
    flip = rng.random(N) < 0.35
    y_s65 = np.where(flip, 1 - d, d)
    tgt_v, shd_v = make_paired(rng, d, y_t=d, y_s=y_s65)
    auc_v2 = roc_auc_score(d, C._transfer(ours_features(shd_v), y_s65, ours_features(tgt_v)))
    print(f"    ----  transfer_v2_like     {auc_v2:.4f}   (agreement "
          f"{float((y_s65 == d).mean()):.2f}; no expected band -- the reading is undefined)")

    print("\n[8] DETERMINISM -- identical inputs must give identical scores.")
    a1 = C._within_cv(ours_features(pos), y)
    a2 = C._within_cv(ours_features(pos), y)
    same = bool(np.array_equal(a1, a2))
    print(f"    {'PASS' if same else 'FAIL'}  within_cv reproducible  {same}")
    if not same:
        FAILURES.append("within_cv is not deterministic")

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print("  -", f)
        raise SystemExit(1)
    print("all checks passed -- the three methods are correctly oriented and do not self-leak")


if __name__ == "__main__":
    main()
