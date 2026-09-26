#!/usr/bin/env bash
# Train ONE detector on ONE arm of the shadow-model MIA design (THEIA_E3).
#
# WHAT THIS CAMPAIGN IS
#   Eight PIDSMaker detectors x two arms, on day-disjoint roles:
#     target  = THEIA_E3_TARGET   train 04-03/04  val 04-05  test 04-10
#     shadow  = THEIA_E3_SHADOW   train 04-09/11  val 04-13  test 04-12
#   There is NO third (neutral) arm and none is possible: THEIA_E3 has only six benign days with
#   data, which is exactly what two arms of two member days plus one private val day each consume.
#   See ../DESIGN.md for the measured day inventory and the known confound.
#
# TWO MEMBER DAYS PER ARM IS FORCED, NOT CHOSEN. Four days listed in THEIA_E3.train_dates in
# config.py (04-02, 04-06, 04-07, 04-08) contain ZERO events -- verified against the database, not
# read from the config. Six benign days remain, and two arms need 2M+2 of them, so M=2.
#
# THE DATASET DICTS ALREADY EXIST. Unlike the CADETS_E3 v2 campaign, which needed new _V2_ dicts
# because its predecessor pair was mirrored, the committed THEIA_E3_TARGET / THEIA_E3_SHADOW
# (config.py:515 and :541) already encode exactly the split above. Verify, do not re-author.
#
# `mia_split_mode=xrole` takes members from train_dates and non-members from val_dates. val_dates is
# therefore consumed DURING TRAINING (it is what the per-node dumps are labelled against), which is
# why changing it forced a retrain rather than a re-evaluation.
#
# HYPERPARAMETERS. Velox / Orthrus-non-snooped / Kairos / NodLink have grid-searched settings in
# docs/docs/tuned_systems.md and use them verbatim. MAGIC / FLASH / ThreaTrace / R-CAID have none
# published, so they run the shipped config/<detector>.yml defaults -- inventing values would make
# "tuned" a false label. The single deviation everywhere is num_epochs 12 -> 300, which is what makes
# the 49/99/149/199/299 checkpoint ladder possible.
#
# EVERY FLAG IS RECORDED to <det>/run_flags_<DATASET>.json. PIDSMaker folds runtime args into the
# artifact hash, so any later re-query must replay the exact same list or it silently resolves to a
# different (empty) artifact tree.
set -euo pipefail

DETECTOR="${1:?usage: run_arm.sh <detector> <dataset> [gpu]}"
DATASET="${2:?usage: run_arm.sh <detector> <dataset> [gpu]}"
GPU="${3:-0}"

REPO=/home/yingjie/PIDSMaker
ROOT=/data/pids_v2_shadow/theia_e3/$DETECTOR
EPOCHS="${EPOCHS:-300}"
# 0-indexed. The early rungs are the naturally weak detectors, for "does a worse detector leak
# more?"; 49..299 is the ladder the summary tables report.
DUMP_EPOCHS="${DUMP_EPOCHS:-0,1,2,5,10,24,49,99,149,199,299}"
TAG="${TAG:-v2_${DETECTOR}}"

mkdir -p "$ROOT"/{artifacts,logs}
LOG="$ROOT/logs/${DATASET}_e${EPOCHS}.log"

# Common to every detector. mia_dump is ON for all eight: this campaign exists to be attacked.
FLAGS=(
  --artifact_dir "$ROOT/artifacts"
  --database_host localhost
  --database_port 8888
  --training.num_epochs="$EPOCHS"
  --training.save_final_model=true
  --training.mia_dump=true
  --training.mia_rich_dump=true
  --training.mia_split_mode=xrole
  --tags "$TAG"
)

# Published tuned settings, where they exist.
case "$DETECTOR" in
  velox)
    FLAGS+=(--training.encoder.dropout=0.3 --training.lr=0.0001
            --training.node_hid_dim=256 --training.node_out_dim=256 --featurization.emb_dim=256) ;;
  orthrus_non_snooped|kairos)
    FLAGS+=(--training.encoder.dropout=0.3 --training.lr=0.001
            --training.node_hid_dim=256 --training.node_out_dim=256 --featurization.emb_dim=256) ;;
  nodlink)
    FLAGS+=(--training.encoder.dropout=0.3 --training.lr=0.001
            --training.node_hid_dim=320 --training.node_out_dim=320
            --featurization.emb_dim=128 --featurization.epochs=20
            --construction.time_window_size=15.0 --transformation.used_methods="none") ;;
  magic|flash|threatrace|rcaid)
    : ;;  # shipped config defaults, deliberately
  *)
    echo "unknown detector '$DETECTOR'" >&2; exit 2 ;;
esac

# Keyed by dataset: the three arms share this directory and must not overwrite each other's record.
python - "$ROOT/run_flags_${DATASET}.json" "$DETECTOR" "$DATASET" "$EPOCHS" "$TAG" "$DUMP_EPOCHS" "${FLAGS[@]}" <<'PY'
import json, os, sys
out, det, ds, epochs, tag, dumps = sys.argv[1:7]
tmp = f"{out}.{os.getpid()}.tmp"
with open(tmp, "w") as fh:
    json.dump({"detector": det, "dataset": ds, "epochs": int(epochs), "tag": tag,
               "dump_epochs": dumps, "flags": sys.argv[7:]}, fh, indent=2)
os.replace(tmp, out)
print(f"recorded flags -> {out}")
PY

source /home/yingjie/anaconda3/etc/profile.d/conda.sh
conda activate pids
cd "$REPO"

echo "=== $DETECTOR | $DATASET | epochs=$EPOCHS dumps=$DUMP_EPOCHS gpu=$GPU ==="
echo "=== log: $LOG ==="

CUDA_VISIBLE_DEVICES="$GPU" \
PIDS_MIA_DUMP_EPOCHS="$DUMP_EPOCHS" \
python -u -m pidsmaker.main "$DETECTOR" "$DATASET" "${FLAGS[@]}" 2>&1 | tee "$LOG"

echo "=== done: $DETECTOR / $DATASET ==="
