#!/usr/bin/env bash
# Run the whole v2 shadow-model campaign, STRICTLY ONE ARM AT A TIME.
#
# WHY SERIAL. RAM, not GPU, is the binding constraint: Kairos peaks near 79.5 GB and R-CAID near
# 47 GB against 251 GB total. An R-CAID spike has twice caused the harness to kill EVERY background
# task in a session, not just the offending one. Two GPUs do not help if the box swaps.
#
# ORDER. Walks the eight detectors cheapest-first and runs target+shadow back to back, so the
# detection table fills in complete rows rather than half-rows.
#
# NO NEUTRAL ARM. `CADETS_E3_V2_CAL` is defined in config.py but deliberately not run here. In this
# design the SHADOW already is the neutral reference: it trains on 04-07/08/09, which are neither
# the target's members (04-02/03/04) nor its non-members (04-05), so scoring the shadow against the
# target's membership labels gives a model that never saw either class -- exactly the zero point the
# separate neutral arm existed to provide. That was not true of the old no-frac design, where the
# shadow trained ON the target's non-member days and so was biased towards one half.
#
# RESUMABLE. An arm is "done" when its epoch-299 checkpoint exists, so re-running this script after
# an interruption skips what already finished. Logs are per (detector, dataset) and are NOT
# overwritten by the skip path.
set -uo pipefail

ROOT=/data/pids_v2_shadow/cadets_e3
GPU="${GPU:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-300}"
LOCK="$ROOT/logs/queue.lock"
mkdir -p "$ROOT/logs"

exec 9>"$LOCK"
flock -n 9 || { echo "another queue.sh holds $LOCK -- exiting"; exit 0; }

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

DETS=(nodlink velox rcaid magic threatrace orthrus_non_snooped flash kairos)

arm_done() {  # <detector> <dataset>
  find "$ROOT/$1/artifacts/training" \
    -path "*/$2/trained_models/model_epoch_299/state_dict.pkl" 2>/dev/null | grep -q .
}

free_gb() { df -BG --output=avail /data | tail -1 | tr -dc '0-9'; }

run_one() {  # <detector> <dataset>
  local det="$1" ds="$2" start mins avail
  if arm_done "$det" "$ds"; then
    log "$det / $ds: already trained -- skipping"
    return 0
  fi
  avail=$(free_gb)
  if [ "$avail" -lt "$MIN_FREE_GB" ]; then
    log "ABORT: only ${avail}G free on /data, below MIN_FREE_GB=${MIN_FREE_GB}G"
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

log "--- target + shadow, cheapest detector first ---"
for det in "${DETS[@]}"; do
  run_one "$det" CADETS_E3_V2_TARGET
  run_one "$det" CADETS_E3_V2_SHADOW
done

log "=== queue.sh done ==="
