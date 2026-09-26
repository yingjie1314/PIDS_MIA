#!/usr/bin/env bash
# Run the whole CLEARSCOPE_E3 shadow-model campaign, STRICTLY ONE ARM AT A TIME.
#
# WHY SERIAL. RAM, not GPU, is the binding constraint: Kairos and R-CAID both peak far above what a
# second concurrent arm leaves free on the 251 GB box, and an R-CAID spike has twice caused the
# harness to kill EVERY background task in a session, not just the offending one. Two GPUs do not
# help if the box swaps.
#
# ORDER: R-CAID RUNS LAST, and that differs from the THEIA/CADETS queues on purpose. DISK is the
# second binding constraint here. Measured on the D1 CLEARSCOPE campaign, R-CAID's feat_inference
# payload alone was 125 GB per arm -- 235 GB of that campaign's 281 GB total, against ~46 GB for the
# other seven detectors combined. Running it last means an out-of-disk abort costs only R-CAID and
# leaves seven complete detector rows rather than truncating the table at random.
#
# DISK GUARD. Each arm is gated on free space before it starts, with a higher bar for R-CAID
# (MIN_FREE_RCAID_GB) than for everything else (MIN_FREE_GB), because the guard is checked once up
# front and R-CAID's growth after that point is an order of magnitude larger.
#
# RESUMABLE. An arm is "done" when its epoch-299 checkpoint exists, so re-running this script after
# an interruption skips what already finished. Logs are per (detector, dataset) and are NOT
# overwritten by the skip path.
set -uo pipefail

ROOT=/data/pids_v2_shadow/clearscope_e3
GPU="${GPU:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-150}"
MIN_FREE_RCAID_GB="${MIN_FREE_RCAID_GB:-350}"
LOCK="$ROOT/logs/queue.lock"
mkdir -p "$ROOT/logs"

exec 9>"$LOCK"
flock -n 9 || { echo "another queue.sh holds $LOCK -- exiting"; exit 0; }

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# cheapest-first, R-CAID displaced to the end (see header)
DETS=(nodlink velox magic threatrace orthrus_non_snooped flash kairos rcaid)

arm_done() {  # <detector> <dataset>
  find "$ROOT/$1/artifacts/training" \
    -path "*/$2/trained_models/model_epoch_299/state_dict.pkl" 2>/dev/null | grep -q .
}

free_gb() { df -BG --output=avail /data | tail -1 | tr -dc '0-9'; }

run_one() {  # <detector> <dataset>
  local det="$1" ds="$2" start mins avail need
  if arm_done "$det" "$ds"; then
    log "$det / $ds: already trained -- skipping"
    return 0
  fi
  need="$MIN_FREE_GB"
  [ "$det" = rcaid ] && need="$MIN_FREE_RCAID_GB"
  avail=$(free_gb)
  if [ "$avail" -lt "$need" ]; then
    log "ABORT: only ${avail}G free on /data, below the ${need}G required for $det"
    exit 1
  fi
  log "$det / $ds: starting (${avail}G free, this is the only job running)"
  start=$(date +%s)
  "$ROOT/scripts/run_arm.sh" "$det" "$ds" "$GPU" >/dev/null 2>&1
  mins=$(( ($(date +%s) - start) / 60 ))
  if arm_done "$det" "$ds"; then
    log "$det / $ds: COMPLETE (${mins} min)"
  else
    log "$det / $ds: FAILED, no epoch-299 checkpoint (${mins} min) -- see $ROOT/$det/logs/${ds}_e300.log"
  fi
}

log "=== queue.sh starting (pid $$, gpu $GPU) ==="
log "--- target + shadow per detector, R-CAID last ---"
for det in "${DETS[@]}"; do
  run_one "$det" CLEARSCOPE_E3_V2_TARGET
  run_one "$det" CLEARSCOPE_E3_V2_SHADOW
done

log "=== queue.sh done ==="
