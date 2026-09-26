#!/usr/bin/env bash
# Ablation 2 -- training-set size, via mia_split_mode=random.
#
# WHY THIS MODE IS TWO EXPERIMENTS AT ONCE. In `xrole` (what the v2 campaign used) members come from
# train_dates and non-members from val_dates, so the two classes are different CALENDAR DAYS. Any
# attack can then win by recognising the day rather than the member, and that confound is exactly
# what blocks the NodLink/CLEARSCOPE result from being a leak claim. In `random`, split_train_graphs
# assigns each training graph to in/out by a seeded coin flip and training_loop then sets
# `train_data = mia_in_data` -- so the model trains only on members, and non-members are interleaved
# among the SAME days. Varying `frac` therefore measures dataset size AND removes the calendar
# confound in one pass.
#
# ONE ARM PER CELL. The target/shadow pair existed to supply non-members from a disjoint role.
# `random` supplies its own non-members from inside the arm, so a second arm would buy nothing here.
#
# THEIA'S ARMS PREDATE THE _V2_ NAMING and are plain THEIA_E3_TARGET -- not THEIA_E3_V2_TARGET,
# which does not exist in config.py and would fail to resolve.
set -uo pipefail

REPO=/home/yingjie/PIDSMaker
ROOT=/data/pids_v2_ablation
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-300}"
DUMP_EPOCHS="${DUMP_EPOCHS:-0,1,2,5,10,24,49,99,149,199,299}"
FRACS="${FRACS:-0.1 0.25 0.5 0.75}"
DET=nodlink
MIN_FREE_GB="${MIN_FREE_GB:-250}"

case "${DATASET:?set DATASET to cadets_e3|theia_e3|clearscope_e3}" in
  cadets_e3)     ARM=CADETS_E3_V2_TARGET ;;
  theia_e3)      ARM=THEIA_E3_TARGET ;;
  clearscope_e3) ARM=CLEARSCOPE_E3_V2_TARGET ;;
  *) echo "unknown DATASET '$DATASET'" >&2; exit 2 ;;
esac

source /home/yingjie/anaconda3/etc/profile.d/conda.sh
conda activate pids
cd "$REPO" || exit 1

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $DATASET $*" >&2; }
free_gb(){ df -BG --output=avail /data | tail -1 | tr -dc '0-9'; }

log "=== ablation 2: $DET / $ARM / fracs $FRACS on GPU $GPU ==="

for frac in $FRACS; do
  TAG="abl2_${DET}_f${frac}"
  OUT="$ROOT/$DATASET/frac/${DET}_f${frac}"
  mkdir -p "$OUT/logs"

  # Resumable: an epoch-299 checkpoint means this cell already finished.
  if find "$OUT/artifacts/training" -path "*/$ARM/trained_models/model_epoch_299/state_dict.pkl" \
       2>/dev/null | grep -q .; then
    log "frac=$frac: already trained -- skipping"
    continue
  fi
  avail=$(free_gb)
  if [ "$avail" -lt "$MIN_FREE_GB" ]; then
    log "ABORT: only ${avail}G free on /data, below MIN_FREE_GB=${MIN_FREE_GB}G"
    exit 1
  fi

  # NodLink's published tuned settings, verbatim from docs/docs/tuned_systems.md -- identical to the
  # v2 campaign so this ablation differs from it in the split alone.
  FLAGS=(
    --artifact_dir "$OUT/artifacts"
    --database_host localhost
    --database_port 8888
    --training.num_epochs="$EPOCHS"
    --training.save_final_model=true
    --training.mia_dump=true
    --training.mia_rich_dump=true
    --training.mia_split_mode=random
    --training.mia_split_frac="$frac"
    --training.mia_split_seed=0
    --training.encoder.dropout=0.3
    --training.lr=0.001
    --training.node_hid_dim=320
    --training.node_out_dim=320
    --featurization.emb_dim=128
    --featurization.epochs=20
    --construction.time_window_size=15.0
    --transformation.used_methods=none
    --tags "$TAG"
  )

  # PIDSMaker folds runtime args into the artifact hash, and a mismatch does not raise -- it
  # resolves to a fresh empty tree. Record the exact list so any re-query can replay it.
  python - "$OUT/run_flags.json" "$DET" "$ARM" "$frac" "$DUMP_EPOCHS" "${FLAGS[@]}" <<'PY'
import json, os, sys
out, det, arm, frac, dumps = sys.argv[1:6]
tmp = f"{out}.{os.getpid()}.tmp"
with open(tmp, "w") as fh:
    json.dump({"detector": det, "arm": arm, "mia_split_mode": "random",
               "mia_split_frac": float(frac), "dump_epochs": dumps,
               "flags": sys.argv[6:]}, fh, indent=2)
os.replace(tmp, out)
PY

  log "frac=$frac: starting (${avail}G free)"
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES="$GPU" PIDS_MIA_DUMP_EPOCHS="$DUMP_EPOCHS" \
    python -u -m pidsmaker.main "$DET" "$ARM" "${FLAGS[@]}" \
    > "$OUT/logs/train.log" 2>&1
  rc=$?
  log "frac=$frac: exit $rc after $(( ($(date +%s)-t0)/60 )) min"
done
log "=== done ==="
