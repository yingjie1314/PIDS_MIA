#!/usr/bin/env bash
# Train ONE detector on ONE arm of the shadow-model MIA design (CLEARSCOPE_E3).
#
# WHAT THIS CAMPAIGN IS
#   Eight PIDSMaker detectors x two arms, on day-disjoint roles:
#     target = CLEARSCOPE_E3_V2_TARGET  train 04-08/09  val 04-03/06  test 04-11
#     shadow = CLEARSCOPE_E3_V2_SHADOW  train 04-05/10  val 04-04/07  test 04-11
#   See ../DESIGN.md for the day-assignment search and the measured pool sizes.
#
# TWO MEMBER DAYS PER ARM IS CHOSEN, NOT FORCED. `mia_split_mode=xrole` makes `val_dates` the
# NON-MEMBER class, so the structure floor is the within-role train-vs-val composition distance.
# Searching all 560 (3 train + 1 val) and 1260 (2+2) assignments of the eight usable benign days put
# the best 2+2 at floor 0.2834 against 0.4254 for the best 3+1 and 0.8313 for the committed
# CLEARSCOPE_E3_* pair. 2+2 also buys near-1:1 class balance (shadow 1.03:1).
#
# BOTH ARMS SHARE TEST DAY 04-11 and no alternative exists: node_clearscope_e3_firefox_0412.csv is
# an empty file, so 04-11 is CLEARSCOPE's only labelled attack day. This does not touch the MIA --
# `xrole` deliberately excludes test days from the non-member pool (training_loop.py:72).
#
# HYPERPARAMETERS ARE THE CLEARSCOPE_E3 ROWS, NOT CADETS'. Velox / Orthrus-non-snooped / Kairos /
# NodLink have grid-searched settings in docs/docs/tuned_systems.md and the values below are that
# file's CLEARSCOPE_E3 lines. They differ a lot from the other datasets (velox hid 64 here vs 256 on
# CADETS and 320 on THEIA), so do not copy another campaign's block into this one -- the THEIA v2
# runner did exactly that and ran CADETS values on THEIA. MAGIC / FLASH / ThreaTrace / R-CAID have
# no published row and run the shipped config/<detector>.yml defaults; inventing values would make
# "tuned" a false label.
#
# ONE DELIBERATE DEVIATION FROM THE TUNED ROWS: `construction.time_window_size`. All four CLEARSCOPE
# rows specify 1.0 minute; this script leaves the shipped 15.0 unless TWS is set. Reasons: (a) 1.0
# gives 1440 windows/day instead of 96, a 15x multiplier on graphs, checkpoint payloads and the 11
# per-epoch MIA dumps, against 1.1 TB free with R-CAID alone needing ~250 GB; (b) CADETS v2 and
# THEIA v2 both ran at 15.0, and the D1 CLEARSCOPE detection table did too, so 15.0 is what keeps
# this campaign comparable to all three. Override with TWS=1 to follow the tuned rows verbatim.
#
# EVERY FLAG IS RECORDED to <det>/run_flags_<DATASET>.json. PIDSMaker folds runtime args into the
# artifact hash, so any later re-query must replay the exact same list or it silently resolves to a
# different (empty) artifact tree.
set -euo pipefail

DETECTOR="${1:?usage: run_arm.sh <detector> <dataset> [gpu]}"
DATASET="${2:?usage: run_arm.sh <detector> <dataset> [gpu]}"
GPU="${3:-0}"

REPO=/home/yingjie/PIDSMaker
ROOT=/data/pids_v2_shadow/clearscope_e3/$DETECTOR
EPOCHS="${EPOCHS:-300}"
# 0-indexed. The early rungs are the naturally weak detectors, for "does a worse detector leak
# more?"; 49..299 is the ladder the summary tables report.
DUMP_EPOCHS="${DUMP_EPOCHS:-0,1,2,5,10,24,49,99,149,199,299}"
TAG="${TAG:-v2cs_${DETECTOR}}"
TWS="${TWS:-}"

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

# docs/docs/tuned_systems.md, CLEARSCOPE_E3 rows, verbatim except time_window_size (see header).
# NodLink's CLEARSCOPE row carries neither `featurization.epochs=20` nor
# `transformation.used_methods=none`, unlike its CADETS/THEIA rows -- do not add them back.
case "$DETECTOR" in
  velox)
    FLAGS+=(--training.encoder.dropout=0.3 --training.lr=0.001
            --training.node_hid_dim=64 --training.node_out_dim=64 --featurization.emb_dim=128) ;;
  orthrus_non_snooped)
    FLAGS+=(--training.encoder.dropout=0.3 --training.lr=0.0001
            --training.node_hid_dim=64 --training.node_out_dim=64 --featurization.emb_dim=128) ;;
  nodlink)
    FLAGS+=(--training.encoder.dropout=0.3 --training.lr=0.0001
            --training.node_hid_dim=64 --training.node_out_dim=64 --featurization.emb_dim=128) ;;
  kairos)
    FLAGS+=(--training.encoder.dropout=0.3 --training.lr=0.0001
            --training.node_hid_dim=256 --training.node_out_dim=256 --featurization.emb_dim=128) ;;
  magic|flash|threatrace|rcaid)
    : ;;  # shipped config defaults, deliberately
  *)
    echo "unknown detector '$DETECTOR'" >&2; exit 2 ;;
esac

[ -n "$TWS" ] && FLAGS+=(--construction.time_window_size="$TWS")

# Keyed by dataset: the two arms share this directory and must not overwrite each other's record.
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

echo "=== $DETECTOR | $DATASET | epochs=$EPOCHS dumps=$DUMP_EPOCHS gpu=$GPU tws=${TWS:-default} ==="
echo "=== log: $LOG ==="

CUDA_VISIBLE_DEVICES="$GPU" \
PIDS_MIA_DUMP_EPOCHS="$DUMP_EPOCHS" \
python -u -m pidsmaker.main "$DETECTOR" "$DATASET" "${FLAGS[@]}" 2>&1 | tee "$LOG"

echo "=== done: $DETECTOR / $DATASET ==="
