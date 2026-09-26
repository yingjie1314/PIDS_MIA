#!/usr/bin/env python
r"""Head-to-head comparison of published node-level MIAs against the NodLink no-frac target.

WHAT IS BEING COMPARED, AND WHY IT NEEDED PORTING
-------------------------------------------------
Two published node-level membership inference attacks on GNNs:

  He et al. 2021   "Node-Level Membership Inference Attacks Against Graph Neural Networks"
                   github.com/fight-think/Node-level-Membership-Inference-Attacks  (code-2)
  Conti et al.2022 "Label-Only Membership Inference Attack against Node-Level GNNs", AISec'22
                   arXiv:2207.13766, github.com/fight-think/Label-Only_MIA_for_GNNs

Both were written against SUPERVISED NODE CLASSIFIERS. Their attack surface is the class posterior
p(y|v): He et al. feed an attack MLP the top-2 posterior entries obtained under 0/1/2-hop queries,
and Conti et al. assume the adversary sees only argmax p(y|v) and recover a soft signal by distilling
a surrogate. NodLink has no classes. It is an unsupervised reconstruction detector whose entire
per-node output is a scalar loss. So neither attack runs unmodified, and the port has to decide what
plays the role of the posterior. Two adaptations, applied to both papers identically:

  1. THE PER-QUERY OBSERVATION is the scalar reconstruction loss instead of a posterior vector. This
     costs the attacks feature width -- He et al.'s ladder yields 2 numbers here, not ~10 -- and that
     narrowing is a consequence of the detector being unsupervised, not a weakened port.
  2. THE HARD LABEL, which Conti et al. need, is the detector's own verdict: a node is flagged
     anomalous when its loss exceeds the operating threshold. The adversary picks the flag rate; it
     is swept and selected on the shadow model, never on the target.

A THIRD ADAPTATION IS FORCED BY THE ARCHITECTURE. NodLink's encoder runs exactly one `propagate`, so
loss(v) depends only on v's immediate neighbours. `requery_hops.py --verify` measured this: deleting
97 % of a window's edges, keeping only those incident to the probed nodes, moves their loss by
<= 3.6e-07. The faithful 2-hop query is therefore ARITHMETICALLY THE SAME as the 1-hop query, and
both papers' 3-level ladder collapses to 2 levels. `hop2a` (A union A^2) is reported alongside as a
strictly stronger query the adversary could still make; it is NOT the papers' 2-hop and is never
labelled as such.

THE HONESTY LINE THAT ORGANISES THE TABLE
-----------------------------------------
Both published attacks are SHADOW-MODEL attacks: fit on a model whose membership the adversary
knows, transfer to the target. Scoring the target under cross-validation on its OWN membership
labels is not an attack -- no adversary has those labels -- it is a separability upper bound. The
table keeps the two apart explicitly, because the campaign's headline 0.93 AUC is of the second kind
and would flatter itself if listed next to shadow-transfer numbers without comment.

Every row is scored on the SAME node set with the SAME metrics, and every row is read against the
geometry floor: a structure-only classifier (degree, node type) that never sees the model. An attack
that does not beat the floor on AUC *and* TPR@1% has found a property of the provenance graph, not
memorisation.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

ROOT = "/data/pids_nofrac_tuned/cadets_e3/artifacts/training/training"
TARGET_DS = "CADETS_E3_NOFRAC"
SHADOW_DS = "CADETS_E3_NOFRAC_SHADOW"
HOPS = ("0", "1", "2a")
# Only 0-hop and 1-hop decide which nodes are eligible. Hop 2a skips windows whose squared adjacency
# would not fit in GPU memory, and those are precisely the largest windows -- requiring it cost 55 %
# of eligible nodes and biased the row set towards small windows (single_loss AUC fell 0.646 ->
# 0.573 on the survivors alone). Since the faithful 2-hop query is provably identical to the 1-hop
# one for this single-propagate model, hop 2a is a bonus surface, not part of the ladder, and is
# reported separately on whatever rows it covers.
REQUIRED_HOPS = ("0", "1")
MAX_KEYS = 60_000
SPLITS = {"mia_train_in": 1, "mia_heldout": 0}


# --------------------------------------------------------------------------------------------
# metrics -- inlined verbatim from robustness_bracket/phase_a_loss_strong.py so the numbers here
# are computed by the same code as the rest of the campaign regardless of which env this runs in.
# --------------------------------------------------------------------------------------------
def tpr_at_fpr(y, s, target_fpr=0.01) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    return float(tpr[max(idx, 0)])


def _metrics(y, s) -> dict:
    y, s = np.asarray(y), np.asarray(s, dtype=np.float64)
    fpr, tpr, _ = roc_curve(y, s)
    return {
        "auc": float(roc_auc_score(y, s)),
        "bal_acc": float((1 + np.max(tpr - fpr)) / 2),
        "tpr_at_1": tpr_at_fpr(y, s, 0.01),
    }


def _fit_predict(clf, X_train, y_train, X_eval) -> np.ndarray:
    """Standardise on the training model's own statistics, fit, score the eval model's rows."""
    X_train, X_eval = np.asarray(X_train, float), np.asarray(X_eval, float)
    mu, sigma = X_train.mean(0, keepdims=True), X_train.std(0, keepdims=True) + 1e-8
    clf.fit((X_train - mu) / sigma, np.asarray(y_train))
    return clf.predict_proba((X_eval - mu) / sigma)[:, 1]


def _gbdt(X_train, y_train, X_eval, seed=0) -> np.ndarray:
    return _fit_predict(
        HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed),
        X_train, y_train, X_eval)


def _within_cv(X, y, seed=0) -> np.ndarray:
    """Out-of-fold scores from the target's OWN labels. An upper bound, not a mountable attack."""
    X, y = np.asarray(X, float), np.asarray(y)
    out = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y):
        out[te] = _gbdt(X[tr], y[tr], X[te], seed)
    return out


def _transfer(X_shadow, y_shadow, X_target, kind="gbdt", seed=0) -> np.ndarray:
    """Fit on the shadow model's rows, score the target's. The published threat model."""
    if kind == "mlp":
        # He et al.'s attack model is a small MLP; theirs is 3 layers over a ~10-dim input.
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed)
    else:
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
    # Each model is standardised by its own pooled statistics. The adversary can do this: it needs
    # the query outputs, which they have, and not the labels, which they do not.
    Xs, Xt = np.asarray(X_shadow, float), np.asarray(X_target, float)
    mu_s, sd_s = Xs.mean(0, keepdims=True), Xs.std(0, keepdims=True) + 1e-8
    mu_t, sd_t = Xt.mean(0, keepdims=True), Xt.std(0, keepdims=True) + 1e-8
    clf.fit((Xs - mu_s) / sd_s, np.asarray(y_shadow))
    return clf.predict_proba((Xt - mu_t) / sd_t)[:, 1]


# --------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------
def run_dir(dataset: str) -> str:
    """Locate a run's artifacts by dataset name rather than by a hardcoded config hash.

    The two runs do NOT share a hash: `set_task_paths` folds the run tag into it, and the shadow
    carries its own tag. The dataset name is a separate path component and identifies the run
    uniquely here, so resolving on it survives any future config change that shifts the hash.
    """
    hits = [os.path.join(ROOT, h, dataset, "edge_losses") for h in sorted(os.listdir(ROOT))
            if os.path.isdir(os.path.join(ROOT, h, dataset, "edge_losses"))]
    # A smoke-test tree also carries CADETS_E3_NOFRAC; it is marked and must never be scored.
    hits = [d for d in hits
            if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(d)),
                                               "SMOKE_TEST_IGNORE.txt"))]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one run dir for {dataset}, found {hits}")
    return hits[0]



def _mean_by_key(paths, fields):
    """{node: {field: mean over every window the node appears in}} plus a per-node touch count."""
    sums, counts = defaultdict(dict), defaultdict(int)
    for p in paths:
        try:
            z = np.load(p)
        except Exception:
            continue
        if "node" not in z.files or len(z["node"]) == 0:
            continue
        keys = z["node"].astype(np.int64)
        have = {f: z[f] for f in fields if f in z.files}
        for i, k in enumerate(keys):
            k = int(k)
            acc = sums[k]
            for f, arr in have.items():
                acc[f] = acc.get(f, 0) + arr[i]
            counts[k] += 1
    return ({k: {f: v / counts[k] for f, v in acc.items()} for k, acc in sums.items()},
            dict(counts))


def _glob(d):
    return sorted(os.path.join(d, f) for f in os.listdir(d)) if os.path.isdir(d) else []


_X_TABLE: dict = {}


def x_table() -> dict:
    """node id -> raw fastText feature vector, shared by both runs.

    A node's input features are a property of the provenance graph, not of any model, and the two
    runs cover the same six days -- they differ only in which three are labelled member. So one
    table, dumped once from the target's artifact tree, serves both.
    """
    global _X_TABLE
    if not _X_TABLE:
        d = os.path.join(run_dir(TARGET_DS), "mia_hops", "node_x")
        _X_TABLE, _ = _mean_by_key(_glob(d), ("x",))
    return _X_TABLE


_SURFACES: dict = {}


def _surfaces(dataset: str, epoch: int):
    """Parsed per-node surfaces for one run, cached so a run can be loaded twice for free."""
    hit = _SURFACES.get((dataset, epoch))
    if hit is not None:
        return hit
    base = run_dir(dataset)
    rich, touch, hop = {}, {}, {h: {} for h in HOPS}
    for split in SPLITS:
        rich[split], touch[split] = _mean_by_key(
            _glob(os.path.join(base, split, f"model_epoch_{epoch}")),
            ("loss", "emb", "node_type"))
        for h in HOPS:
            hop[h][split], _ = _mean_by_key(
                _glob(os.path.join(base, "mia_hops", f"hop{h}", split, f"model_epoch_{epoch}")),
                ("loss",))
    _SURFACES[(dataset, epoch)] = (rich, touch, hop)
    return rich, touch, hop


def load_run(dataset: str, epoch: int, need_x: bool = True,
             restrict: set | None = None, max_keys: int | None = MAX_KEYS,
             keys_only: bool = False) -> dict:
    """One model's per-node surfaces, with ambiguous nodes dropped exactly as d2_collect does.

    `restrict` pins the row set to a caller-supplied node set so two runs can be loaded on the
    identical rows in the identical order -- required for any cross-model comparison.
    """
    rich, touch, hop = _surfaces(dataset, epoch)
    xtab = x_table() if need_x else {}

    # A node present on both sides is genuinely member AND non-member and cannot label an example.
    raw = {s: len(rich[s]) for s in SPLITS}
    ambiguous = set(rich["mia_train_in"]) & set(rich["mia_heldout"])

    keys = []
    for split, label in SPLITS.items():
        for k in rich[split]:
            if k in ambiguous:
                continue
            if any(k not in hop[h][split] for h in REQUIRED_HOPS):
                continue
            if need_x and k not in xtab:
                continue
            if restrict is not None and k not in restrict:
                continue
            keys.append((k, label, split))
    # Sorting by node id alone (not by label) keeps the order identical across two runs whose
    # labels are swapped, which is what makes the cross-model join positional.
    keys.sort(key=lambda t: t[0])
    n_eligible = len(keys)
    if max_keys is not None and n_eligible > max_keys:
        rng = np.random.default_rng(0)
        keys = [keys[i] for i in np.sort(rng.choice(n_eligible, max_keys, replace=False))]

    if keys_only:
        return {"keys": keys, "n_eligible": n_eligible}

    y = np.array([lab for _, lab, _ in keys], dtype=int)
    loss = np.array([rich[s][k]["loss"] for k, _, s in keys], dtype=np.float64)
    emb = np.stack([np.atleast_1d(rich[s][k]["emb"]) for k, _, s in keys]).astype(np.float64)
    ntype = np.array([float(rich[s][k]["node_type"]) for k, _, s in keys])
    deg = np.array([touch[s].get(k, 0) for k, _, s in keys], dtype=np.float64)
    # NaN marks a node whose window was too large for the hop-2a query; only hop 2a can be missing.
    hops = {h: np.array([hop[h][s].get(k, {}).get("loss", np.nan) for k, _, s in keys],
                        dtype=np.float64) for h in HOPS}

    out = {
        "dataset": dataset, "epoch": epoch, "keys": keys, "y": y, "loss": loss, "emb": emb,
        "geom": np.stack([deg, ntype], axis=1), "hops": hops,
        "n_ambiguous": len(ambiguous), "n_raw": raw, "n_eligible": n_eligible, "n_rows": len(y),
    }
    if need_x:
        out["x"] = np.stack([np.atleast_1d(xtab[k]["x"]) for k, _, _ in keys]).astype(np.float64)
    return out


# --------------------------------------------------------------------------------------------
# the ported attacks
# --------------------------------------------------------------------------------------------
def he_features(run: dict, hops=REQUIRED_HOPS) -> np.ndarray:
    """He et al.'s query ladder: the model's answer at each hop, concatenated.

    Their version stacks the top-2 posterior entries per hop. NodLink answers each query with one
    number, so the ladder is 2-dimensional here rather than ~10 -- 2 and not 3 because its faithful
    2-hop query returns the 1-hop answer exactly.
    """
    return np.stack([run["hops"][h] for h in hops], axis=1)


def conti_label_only(run: dict, flag_rate: float) -> np.ndarray:
    """Conti et al.'s threshold family: how often the node survives the detector's verdict.

    The adversary sees one bit per query -- flagged anomalous, or not -- and nothing else. A node
    the model has memorised should read benign under every query; a node it has not should flip on
    at least one. Scoring the agreement across the hop ladder is the `all_cor` / `p_o_g_t` idea from
    the paper, transposed from 'predicted the right class' to 'was not flagged'.

    The threshold is a quantile of THIS model's own loss distribution, so the flag rate means the
    same thing on the shadow and on the target even though their loss scales differ.
    """
    bits = []
    for h in REQUIRED_HOPS:
        v = run["hops"][h]
        bits.append((v <= np.quantile(v, 1.0 - flag_rate)).astype(float))
    return np.sum(bits, axis=0)


def conti_distilled(run: dict, flag_rate: float, seed=0) -> np.ndarray:
    """Conti et al.'s distillation step, which is what makes a label-only attack sharp.

    Three bits give only four score levels, far too coarse to rank nodes. The paper's answer is to
    relabel data with the model's HARD verdicts, train a surrogate on those, and read the
    surrogate's soft output in place of the posterior the adversary cannot see. Ported here:

      surrogate    : (raw node features, degree, node type) -> the model's hard verdict
      attack score : -cross_entropy(hard verdict, surrogate probability), their `cro_entropy`

    A node the model memorised carries a verdict the surrogate can reproduce from generic features,
    so its cross-entropy is low. Only hard verdicts ever cross the boundary, so this stays
    label-only. The surrogate is fit OUT OF FOLD: trained on four fifths of the rows and asked about
    the fifth it has not seen. Fitting and scoring the same rows would let it echo the verdicts back
    verbatim and collapse the signal. The folds are cut on the verdicts alone -- no membership label
    takes part -- so the adversary can do exactly this.
    """
    v = run["hops"]["1"]
    verdict = (v > np.quantile(v, 1.0 - flag_rate)).astype(int)
    if verdict.sum() < 50 or (1 - verdict).sum() < 50:
        return np.zeros(len(verdict))

    X = np.hstack([run["x"], run["geom"]])
    p = np.zeros(len(verdict))
    for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X, verdict):
        p[te] = _gbdt(X[tr], verdict[tr], X[te], seed)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ce = -(verdict * np.log(p) + (1 - verdict) * np.log(1 - p))
    return -ce


def calibrated(tgt: dict, shd: dict) -> dict:
    """The target's surfaces with the shadow model's answer to the same query subtracted.

    This is difficulty calibration (Watson et al. 2022; the offline arm of LiRA), and against this
    target it is not optional. Members and non-members here are different calendar days, and the
    two day sets are not equally easy to reconstruct -- so a model that never trained on the
    members still ranks them apart. Subtracting a second model's answer to the same query cancels
    whatever is a property of the node rather than of the training set, and leaves only what is
    specific to THIS model having seen THIS node.

    Embedding coordinates are deliberately not differenced: coordinate k of two independently
    trained latent spaces refers to nothing in common, so their difference is noise, not a signal.
    Only the norm survives.
    """
    out = dict(tgt)
    out["hops"] = {h: tgt["hops"][h] - shd["hops"][h] for h in HOPS}
    out["loss"] = tgt["loss"] - shd["loss"]
    out["emb_norm_delta"] = (np.linalg.norm(tgt["emb"], axis=1)
                             - np.linalg.norm(shd["emb"], axis=1))
    return out


# --------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=299)
    ap.add_argument("--shadow-epoch", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap the row set for a quick run; default is the full eligible population")
    ap.add_argument("--out", default="/data/pids_nofrac_tuned/cadets_e3/results/attack_comparison.json")
    ap.add_argument("--design", choices=("swapped", "aligned"), default="swapped",
                    help="how the campaign maps days onto membership. 'swapped': the shadow's "
                         "members are exactly the target's non-members (agreement 0.0), which is "
                         "what every transfer row below assumes. 'aligned': both arms call the "
                         "same nodes members (agreement 1.0). A campaign that is neither cannot "
                         "be read by this script -- see the gate in main().")
    ap.add_argument("--design-tol", type=float, default=0.02,
                    help="how far measured agreement may sit from the declared design")
    ap.add_argument("--min-coverage", type=float, default=0.90,
                    help="minimum share of each arm's eligible nodes that must survive the "
                         "intersection; below this the floors sit on a biased subset")
    a = ap.parse_args()
    shadow_epoch = a.shadow_epoch if a.shadow_epoch is not None else a.epoch

    # Both runs must be loaded on the SAME nodes in the SAME order, or no cross-model quantity
    # below means anything. Take each run's eligible set, intersect, then sample once.
    print("aligning target and shadow on a common node set...", flush=True)
    kt = {k for k, _, _ in load_run(TARGET_DS, a.epoch, max_keys=None, keys_only=True)["keys"]}
    ks = {k for k, _, _ in load_run(SHADOW_DS, shadow_epoch, max_keys=None, keys_only=True)["keys"]}
    common = kt & ks
    cov_t, cov_s = len(common) / max(len(kt), 1), len(common) / max(len(ks), 1)
    print(f"  target eligible {len(kt)} | shadow eligible {len(ks)} | common {len(common)} "
          f"| coverage target {cov_t:.3f} shadow {cov_s:.3f}", flush=True)
    # Every floor below is measured on `common`. If the intersection keeps only part of an arm,
    # the floor describes that subset and not the population the raw rows are compared against.
    if min(cov_t, cov_s) < a.min_coverage:
        raise SystemExit(
            f"ABORT: the two arms overlap on only {min(cov_t, cov_s):.3f} of the smaller eligible "
            f"set (need >= {a.min_coverage}). Every floor below would be measured on a biased "
            f"subset while the raw rows describe the full population, so the two are not "
            f"comparable. This is what a disjoint-day-range campaign looks like.")

    tgt = load_run(TARGET_DS, a.epoch, restrict=common, max_keys=a.max_rows)
    shd = load_run(SHADOW_DS, shadow_epoch, restrict=common, max_keys=a.max_rows)
    assert [k for k, _, _ in tgt["keys"]] == [k for k, _, _ in shd["keys"]], "row alignment failed"

    yt, ys = tgt["y"], shd["y"]
    # Every transfer row below reads the shadow as a role-swapped copy of the target, and its
    # caveat text states that outright. That is a property of the CAMPAIGN, not of these dumps,
    # so it cannot be inferred from the numbers -- it has to be declared and then checked. A
    # campaign that violates it still runs to completion and still prints a plausible table.
    agree = float((yt == ys).mean())
    expect = 0.0 if a.design == "swapped" else 1.0
    print(f"  rows {len(yt)} | target members {int(yt.sum())} | shadow members {int(ys.sum())} | "
          f"label agreement {agree:.4f} (declared '{a.design}', expects {expect:.1f})", flush=True)
    if abs(agree - expect) > a.design_tol:
        raise SystemExit(
            f"ABORT: declared design '{a.design}' expects label agreement {expect:.1f} "
            f"(+/-{a.design_tol}), measured {agree:.4f}. The two arms are neither role-swapped "
            f"nor identical, so no transfer row has a defined sign: a shadow-trained classifier "
            f"is fitted partly on nodes that are members for BOTH models and partly on nodes "
            f"that swap, and the resulting AUC mixes the two with no way to separate them. "
            f"Use a neutral arm that never saw either day set as the transfer source instead.")

    cal = calibrated(tgt, shd)
    rows = {}

    # --- reference lines ------------------------------------------------------------------
    # A floor answers one question: what would this attack score if the model had memorised
    # NOTHING? Two independent answers, and every raw row is held to the stricter of them.
    rows["floor_geometry"] = dict(
        _metrics(yt, _within_cv(tgt["geom"], yt)),
        knows="structure only (degree, node type); no model access at all",
        fit="target's own labels, 5-fold out-of-fold",
        role="floor", protocol="floor",
        note="how well the day sets can be told apart from graph structure alone")
    # The decisive control. The shadow model never trained on the target's members, so whatever it
    # scores against the target's labels is by construction NOT memorisation of them -- it is the
    # two day sets being unequally easy. An attack has to beat this to have found anything.
    rows["floor_shadow_model"] = dict(
        _metrics(yt, -shd["hops"]["1"]),
        knows="one loss value per node, from a model that NEVER trained on these members",
        fit="none (threshold)",
        role="floor", protocol="floor",
        note="day-difficulty floor: the same attack mounted on a model with no access to the "
             "target's training set")

    rows["single_loss"] = dict(
        _metrics(yt, -tgt["hops"]["1"]),
        knows="one loss value per node", fit="none (threshold)", protocol="raw")

    # --- He et al. 2021 -------------------------------------------------------------------
    Xt_he, Xs_he = he_features(tgt), he_features(shd)
    rows["he2021_hop_ladder"] = dict(
        _metrics(yt, _transfer(Xs_he, ys, Xt_he, kind="mlp")),
        knows="loss under 0-hop and 1-hop queries (their full ladder for a 1-layer model)",
        fit="shadow model, MLP attack classifier -> transferred to target", protocol="transfer")
    rows["he2021_hop0_only"] = dict(
        _metrics(yt, _transfer(Xs_he[:, :1], ys, Xt_he[:, :1], kind="mlp")),
        knows="loss under the 0-hop query only (their attack-0 analogue)",
        fit="shadow model, MLP attack classifier -> transferred to target", protocol="transfer")

    # Bonus surface, NOT the papers' 2-hop: A union A^2 makes v aggregate over its 2-hop
    # neighbourhood. Scored only where both models have it, so it is a different -- and
    # smaller-window-biased -- row set; the count is reported so it is never read as comparable.
    ok_t = ~np.isnan(tgt["hops"]["2a"])
    ok_s = ~np.isnan(shd["hops"]["2a"])
    if ok_t.sum() > 1000 and ok_s.sum() > 1000:
        Xt3, Xs3 = he_features(tgt, HOPS)[ok_t], he_features(shd, HOPS)[ok_s]
        rows["he2021_plus_adapted_2hop"] = dict(
            _metrics(yt[ok_t], _transfer(Xs3, ys[ok_s], Xt3, kind="mlp")),
            knows="loss under 0-hop / 1-hop / adapted A-union-A^2 queries",
            fit="shadow model, MLP attack classifier -> transferred to target", protocol="transfer",
            note=("different row set: hop 2a skips the largest windows, so these rows are biased "
                  "towards small windows and are not comparable to the rows above"),
            n_rows=int(ok_t.sum()))

    # --- Conti et al. 2022 ----------------------------------------------------------------
    # Flag rate is the adversary's choice, so it is selected where they are allowed to look: on the
    # shadow model, whose membership they know. The target never participates in the choice.
    sweep = []
    for q in (0.01, 0.05, 0.10, 0.25):
        s_shadow = conti_label_only(shd, q)
        sweep.append((roc_auc_score(ys, s_shadow), q))
    best_auc_shadow, q_star = max(sweep)
    rows["conti2022_label_only"] = dict(
        _metrics(yt, conti_label_only(tgt, q_star)),
        knows=f"one bit per query (flagged / not) at flag rate {q_star:g}",
        fit=f"threshold selected on shadow (shadow AUC {best_auc_shadow:.4f})",
        protocol="raw",
        flag_rate=q_star, sweep=[{"flag_rate": q, "shadow_auc": v} for v, q in sweep])

    rows["conti2022_distilled"] = dict(
        _metrics(yt, conti_distilled(tgt, q_star)),
        knows=f"hard verdicts only, at flag rate {q_star:g}, plus raw node features",
        fit="surrogate distilled from the target's hard verdicts; flag rate set on shadow",
        protocol="raw",
        shadow_auc=float(roc_auc_score(ys, conti_distilled(shd, q_star))))

    # --- ours ------------------------------------------------------------------------------
    Xt_ours = np.hstack([tgt["loss"].reshape(-1, 1),
                         np.linalg.norm(tgt["emb"], axis=1, keepdims=True), tgt["emb"]])
    Xs_ours = np.hstack([shd["loss"].reshape(-1, 1),
                         np.linalg.norm(shd["emb"], axis=1, keepdims=True), shd["emb"]])
    rows["ours_shadow_transfer"] = dict(
        _metrics(yt, _transfer(Xs_ours, ys, Xt_ours)),
        knows="loss + 256-d node embedding",
        fit="shadow model, GBDT -> transferred to target (mountable)", protocol="transfer")
    # Embedding coordinate k means something different in two independently trained models -- their
    # latent spaces are never aligned -- so a classifier fit on the shadow's raw dimensions has
    # little reason to transfer. These are the model-agnostic scalars instead, directly comparable
    # once each model is standardised by its own statistics.
    rows["ours_shadow_transfer_scalar"] = dict(
        _metrics(yt, _transfer(Xs_ours[:, :2], ys, Xt_ours[:, :2])),
        knows="loss + embedding norm (no raw latent coordinates)",
        fit="shadow model, GBDT -> transferred to target (mountable)", protocol="transfer")
    rows["ours_within_cv_upper_bound"] = dict(
        _metrics(yt, _within_cv(Xt_ours, yt)),
        knows="loss + 256-d node embedding",
        fit="target's OWN labels, 5-fold out-of-fold -- UPPER BOUND, not a mountable attack",
        protocol="within_cv")
    # The matched control for the row above, and the one the campaign was missing. Identical
    # classifier, identical labels, identical folds -- but reading the SHADOW model's embedding,
    # which cannot contain memorisation of the target's members. Whatever this scores is what a
    # NodLink embedding reveals about a node irrespective of who trained on it.
    rows["floor_shadow_within_cv"] = dict(
        _metrics(yt, _within_cv(Xs_ours, yt)),
        knows="loss + 256-d embedding of a model that NEVER trained on these members",
        fit="target's own labels, 5-fold out-of-fold",
        role="floor", protocol="floor",
        note="matched control for every within-CV row")
    rows["floor_raw_features_only"] = dict(
        _metrics(yt, _within_cv(tgt["x"], yt)),
        knows="raw fastText node features; NO model of any kind",
        fit="target's own labels, 5-fold out-of-fold",
        role="floor", protocol="floor",
        note="how much of membership is written in the node's own content")

    # --- the same attacks, difficulty-calibrated ------------------------------------------
    # Every row above reads the target model's answer in absolute terms, which on this split
    # confounds "the model memorised this node" with "this node's day is easier". These subtract
    # the shadow model's answer to the identical query first.
    rows["cal_single_loss"] = dict(
        _metrics(yt, -cal["hops"]["1"]),
        knows="target loss minus shadow loss, per node",
        fit="none (threshold); mountable -- both papers already assume a shadow model",
        protocol="calibrated")
    rows["cal_he2021_hop_ladder"] = dict(
        _metrics(yt, _within_cv(he_features(cal), yt)),
        knows="per-hop loss differences (target minus shadow) across the ladder",
        fit="target's OWN labels, 5-fold out-of-fold -- UPPER BOUND on the calibrated ladder",
        protocol="calibrated_within_cv")
    rows["cal_conti2022_label_only"] = dict(
        _metrics(yt, conti_label_only(cal, q_star)),
        knows=f"one bit per query on the calibrated surface, flag rate {q_star:g}",
        fit="threshold selected on shadow", protocol="calibrated")
    rows["cal_ours_within_cv_upper_bound"] = dict(
        _metrics(yt, _within_cv(np.stack([cal["loss"], cal["emb_norm_delta"]]
                                         + [cal["hops"][h] for h in REQUIRED_HOPS], axis=1), yt)),
        knows="calibrated loss, calibrated embedding norm, calibrated hop ladder",
        fit="target's OWN labels, 5-fold out-of-fold -- UPPER BOUND, not a mountable attack",
        protocol="calibrated_within_cv")

    # --- verdicts ---------------------------------------------------------------------------
    # Raw rows are held to the stricter of the two floors. Calibrated rows are held to chance,
    # because calibration has already removed the component both floors measure: differencing two
    # models that saw the same graph cancels graph structure and day difficulty alike, so a
    # non-memorising target would score 0.5 by construction.
    chance = {"auc": 0.5, "bal_acc": 0.5, "tpr_at_1": 0.01, "name": "chance"}
    # A floor is only a floor if it was measured under the same protocol as the row it judges. An
    # unsupervised threshold is judged against an unsupervised threshold on a non-member-trained
    # model; a supervised within-CV row against the SAME supervised classifier on a
    # non-member-trained model. Mixing them is how a structure-only floor of 0.5975 came to certify
    # a 0.91 row as a leak when its own matched control scores 0.93.
    floor_by_protocol = {
        "raw": ("floor_shadow_model", rows["floor_shadow_model"]),
        "within_cv": ("floor_shadow_within_cv", rows["floor_shadow_within_cv"]),
        # A threshold on a difference cannot manufacture a day signal, so chance is a fair floor.
        "calibrated": ("chance", chance),
        # A supervised classifier on a difference CAN: even after the day effect cancels on average,
        # the magnitude of the two models' disagreement still varies by day, and a flexible model
        # picks that up. Chance is therefore NOT a valid floor here. The matched control is the same
        # classifier on (neutral - shadow), which requires the third model trained on days neither
        # role used; until that lands these rows are reported without a verdict.
        "calibrated_within_cv": (None, None),
        "transfer": ("floor_shadow_model", rows["floor_shadow_model"]),
    }

    if "he2021_plus_adapted_2hop" in rows:
        # Its rows are a biased subset, so it gets a floor built on that same subset rather than
        # borrowing one measured on a different population.
        rows["he2021_plus_adapted_2hop"]["floor_on_same_rows"] = _metrics(
            yt[ok_t], -shd["hops"]["1"][ok_t])

    inverted = ("this pilot swaps the two day sets between target and shadow, so the shadow's "
                "members ARE the target's non-members. Anything day-specific the attack "
                "classifier learns therefore transfers with inverted sign, and an AUC below 0.5 "
                "is the signature of exactly that. These rows measure the pilot's split design, "
                "not the published attack's strength.")
    for r in rows.values():
        if r.get("protocol") == "transfer":
            r["caveat"] = inverted

    for name, r in rows.items():
        if r.get("role") == "floor":
            continue
        if r.get("floor_on_same_rows"):
            f, ref = r["floor_on_same_rows"], "same_rows"
        else:
            ref, f = floor_by_protocol[r["protocol"]]
        if f is None:
            r["floor_ref"], r["floor_auc"], r["beats_floor"] = "no valid floor yet", None, None
            continue
        r["floor_ref"] = ref
        r["floor_auc"] = float(f["auc"])
        # Three conditions, all required. Beating a floor while scoring below chance is not a
        # leak -- it means the floor itself collapsed, which is what a sub-chance transfer row
        # does. Requiring the attack to clear chance in its own right blocks that.
        r["beats_floor"] = bool(r["auc"] > 0.5
                                and r["auc"] > f["auc"]
                                and r["tpr_at_1"] > f["tpr_at_1"])

    width = max(len(k) for k in rows)
    order = ["floor", "raw", "transfer", "within_cv", "calibrated", "calibrated_within_cv"]
    header = f"\n{'attack':<{width}}  {'AUC':>7} {'balACC':>7} {'TPR@1%':>7}  " \
             f"{'floor':>22} {'vs':>7}  verdict"
    print(header)
    print("-" * (width + 60))
    for group in order:
        members = [(n, r) for n, r in rows.items() if r["protocol"] == group]
        if not members:
            continue
        print(f"-- {group} " + "-" * (width + 55 - len(group)))
        for name, r in members:
            if r.get("role") == "floor":
                print(f"{name:<{width}}  {r['auc']:7.4f} {r['bal_acc']:7.4f} "
                      f"{r['tpr_at_1']:7.4f}  {'--':>22} {'--':>7}  --")
                continue
            if r["beats_floor"] is None:
                print(f"{name:<{width}}  {r['auc']:7.4f} {r['bal_acc']:7.4f} "
                      f"{r['tpr_at_1']:7.4f}  {r['floor_ref']:>22} {'--':>7}  ?")
                continue
            v = "LEAK" if r["beats_floor"] else "no"
            print(f"{name:<{width}}  {r['auc']:7.4f} {r['bal_acc']:7.4f} {r['tpr_at_1']:7.4f}  "
                  f"{r['floor_ref']:>22} {r['floor_auc']:7.4f}  {v}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({
            "epoch": a.epoch, "shadow_epoch": shadow_epoch,
            "target": {k: tgt[k] for k in ("dataset", "n_rows", "n_eligible", "n_ambiguous")},
            "shadow": {k: shd[k] for k in ("dataset", "n_rows", "n_eligible", "n_ambiguous")},
            "target_members": int(yt.sum()), "shadow_members": int(ys.sum()),
            "rows": rows,
        }, fh, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
