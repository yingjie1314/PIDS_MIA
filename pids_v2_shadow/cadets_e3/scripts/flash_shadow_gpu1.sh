#!/usr/bin/env bash
# Run flash / CADETS_E3_V2_SHADOW on GPU 1, in parallel with the serial queue on GPU 0.
#
# WHY IT STARTS IMMEDIATELY RATHER THAN WAITING FOR GPU 1 TO CLEAR. flash is the last detector in
# queue.sh's list, so the queue will itself reach flash SHADOW once it finishes flash TARGET
# (projected ~15:03). Whichever copy gets there second must find an epoch-299 checkpoint already in
# place, because `arm_done` is the only thing that stops the queue from launching a duplicate into
# the same artifact tree. So the margin is what buys safety, and starting now maximises it:
#
#   start now (~09:40), sharing GPU 1 with kairos SHADOW for its last ~1.8 h
#     -> finishes ~13:45 even on the pessimistic 245 min estimate, ~80 min before the queue arrives
#   start at ~11:15 when kairos frees GPU 1
#     -> finishes ~15:20 on the same estimate, i.e. AFTER the queue arrives. Not acceptable.
#
# The alternative -- stopping queue.sh so it can never reach this arm -- is cleaner but means
# killing a running campaign process, so it is not done here.
#
# CONTENTION. Sharing GPU 1 with kairos costs both jobs some throughput for ~1.8 h. VRAM is not the
# constraint (kairos holds ~6 GB of 32 GB). RAM is, hence the guard below.
set -uo pipefail

ROOT=/data/pids_v2_shadow/cadets_e3
DS=CADETS_E3_V2_SHADOW
GPU=1
MIN_FREE_RAM_GB="${MIN_FREE_RAM_GB:-90}"
LOCK=$ROOT/logs/flash_gpu1.lock
mkdir -p "$ROOT/logs"

exec 9>"$LOCK"
flock -n 9 || { echo "another flash_shadow_gpu1.sh holds $LOCK -- exiting"; exit 0; }

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

arm_done() {  # identical predicate to queue.sh
  find "$ROOT/flash/artifacts/training" \
    -path "*/$DS/trained_models/model_epoch_299/state_dict.pkl" 2>/dev/null | grep -q .
}

log "=== flash_shadow_gpu1.sh starting (pid $$, gpu $GPU) ==="

if arm_done; then
  log "flash / $DS already trained -- nothing to do"
  exit 0
fi
if pgrep -f "pidsmaker\.main flash $DS" >/dev/null 2>&1; then
  log "ABORT: flash / $DS is ALREADY running somewhere -- refusing to start a second copy"
  exit 1
fi

ram=$(free -g | awk 'NR==2{print $7}')
if [ "$ram" -lt "$MIN_FREE_RAM_GB" ]; then
  log "ABORT: only ${ram}G RAM available, below MIN_FREE_RAM_GB=${MIN_FREE_RAM_GB}G"
  exit 1
fi

log "starting flash / $DS on GPU $GPU (${ram}G RAM free)"
start=$(date +%s)
"$ROOT/scripts/run_arm.sh" flash "$DS" "$GPU" >/dev/null 2>&1
mins=$(( ($(date +%s) - start) / 60 ))

if arm_done; then
  log "flash / $DS: COMPLETE (${mins} min)"
  log "queue.sh will now skip this arm via its own arm_done check"
else
  log "flash / $DS: FAILED (${mins} min) -- see $ROOT/flash/logs/${DS}_e300.log"
fi
log "=== flash_shadow_gpu1.sh done ==="
