# Prov-MIA v2 — membership inference on PIDSMaker provenance detectors (code only)

Code for the **v2 shadow-model campaign** and the **ablation studies** built on it: training
PIDSMaker intrusion detectors on day-disjoint TARGET/SHADOW arms, dumping per-node MIA artifacts
along a checkpoint ladder, and scoring them with three membership-inference attacks.

This repository contains **code only**. No results, tables, trained models, checkpoints, MIA dumps
or datasets are included.

## Layout

Folder names mirror the `/data/...` directories the scripts were run from, so the absolute paths
hard-coded in the scripts map one-to-one onto this tree.

| folder | was `/data/...` | what it does |
|---|---|---|
| `pidsmaker_patch/` | — | the PIDSMaker fork the runs used (see below) |
| `pids_v2_shadow/` | `/data/pids_v2_shadow` | training campaign: `run_arm.sh` trains one detector on one arm; `queue*.sh` schedule arms; `resolve_params.py` prints each arm's effective config |
| `pids_v2_attacks/` | `/data/pids_v2_attacks` | attacks and tables: `ours_v2.py` (white-box), `requery_hops_v2.py` + `he_conti_v2.py` (He 2021, Conti 2022), `floor_v2.py` (structure-only floor), `*table*.py` |
| `pids_v2_ablation/` | `/data/pids_v2_ablation` | Ablation 1 epoch ladder (`queue_ablation_hops.sh`, `build_epoch_ablation.py`), 1b detection-vs-MIA (`build_tradeoff.py`), 2 training-set fraction (`run_frac.sh`, `score_frac.py`, `frac_sizes.py`, `build_frac_table.py`) |
| `pids_nofrac_tuned/cadets_e3/scripts/compare_attacks.py` | same | shared attack helpers imported by the v2 attack scripts |

## PIDSMaker fork

`pidsmaker_patch/pidsmaker_v2.patch` applies cleanly on top of upstream
[ubc-provenance/PIDSMaker](https://github.com/ubc-provenance/PIDSMaker) commit
`32602734bc9f896be5fc0f03f0a185c967cd6624`:

```bash
git clone https://github.com/ubc-provenance/PIDSMaker.git && cd PIDSMaker
git checkout 32602734bc9f896be5fc0f03f0a185c967cd6624
git apply /path/to/pidsmaker_patch/pidsmaker_v2.patch
```

`pidsmaker_patch/files/` holds full copies of the 28 changed files (listed in
`CHANGED_FILES.txt`). The patch adds:

- **MIA dataset roles** in `pidsmaker/config/config.py`: `*_V2_TARGET` / `*_V2_SHADOW` for CADETS_E3
  and CLEARSCOPE_E3, `THEIA_E3_TARGET` / `THEIA_E3_SHADOW`, and the `mia_split_mode`
  (`xrole` / `random`) and `mia_split_frac` / `mia_split_seed` options.
- **Checkpoint-ladder MIA dumps** (`detection/training_methods/mia_dump.py`, `training_loop.py`):
  `PIDS_MIA_DUMP_EPOCHS=0,1,2,5,10,24,49,99,149,199,299` dumps per-node inference losses (plus embeddings where
  the detector exposes them) for members and non-members at those epochs. The epoch-`e` dump is written at the **end** of that
  epoch's loop body, so `epoch 0` is the model after one training epoch, not a random init.
- **Memory fixes** needed for R-CAID and large datasets (streamed featurization, low-memory window
  loading, gradient checkpointing), each with an equivalence test under `tests/`.

## Pipeline

1. **Train** (per dataset, per detector, per arm):
   `pids_v2_shadow/<dataset>/scripts/run_arm.sh <detector> <DATASET_ARM> [gpu]`
2. **White-box attack**: `pids_v2_attacks/<dataset>/scripts/ours_v2.py`
3. **He 2021 / Conti 2022**: re-query hop 0/1 with `requery_hops_v2.py` (driven by
   `sweep_hops_v2.sh` / `queue_hops*.sh`), then score with `he_conti_v2.py`
4. **Floor**: `floor_v2.py` (structure only, no model output)
5. **Ablations**: see `pids_v2_ablation/scripts/`; Ablation 2 runs as
   `DATASET=<dataset> run_frac.sh`, then `score_frac.py` → `frac_sizes.py` → `build_frac_table.py`.

## Notes

- Paths are hard-coded to the original machine (`/data/...`, `/home/yingjie/PIDSMaker`, a conda env
  at `/home/yingjie/anaconda3/envs/pids`). Edit the `ROOT` / `REPO` / `PY` constants at the top of
  each script, or recreate the same directory layout.
- Set `OMP_NUM_THREADS=8` (or similar) for the scikit-learn scorers; an unbounded OpenMP pool is
  ~80x slower on a many-core machine.
- `ours_v2.py` and `he_conti_v2.py` score with within-arm 5-fold CV and are **upper bounds**, not
  attacks a real adversary can mount. Conti 2022 selects its flag rate on the other arm and is
  mountable.
