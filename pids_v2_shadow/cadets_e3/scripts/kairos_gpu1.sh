#!/usr/bin/env bash
# Run the two Kairos arms on GPU 1, concurrently with the serial queue.sh on GPU 0.
#
# WHY. Kairos is the campaign's long pole (~11.4 h for both arms) and queue.sh runs it LAST, so in
# the serial order it sits on the critical path. Running it on the otherwise idle GPU 1 removes it
# from that path and brings the campaign in roughly 11.5 h earlier.
#
# WHY THIS IS SAFE NOW, HAVING BEEN UNSAFE BEFORE. The serial design exists because RAM is the
# binding constraint and an R-CAID spike has twice mass-killed every background task in a session.
# BOTH R-CAID arms are already complete, so that detector is behind us. What remains on GPU 0 is
# magic / threatrace / orthrus / flash -- magic was measured at 4.1 GB RSS -- against Kairos at
# ~79.5 GB. Peak concurrent RAM is therefore ~110 GB of 251 GB, not the ~159 GB that pairing two
# heavy arms would need.
#
# NO COLLISION WITH queue.sh. Separate lock, separate GPU, and a detector queue.sh will not reach
# for ~22 h. `arm_done` here is the SAME predicate queue.sh uses, so when queue.sh does reach Kairos
# it finds the epoch-299 checkpoint and skips. Artifacts, logs and run_flags json are all per
# (detector, dataset), so nothing is shared with the arms running on GPU 0.
#
# NEVER edit queue.sh while it runs -- bash reads scripts incrementally. This is a separate file for
# that reason.
set -uo pipefail

ROOT=/data/pids_v2_shadow/cadets_e3
GPU=1
MIN_FREE_GB="${MIN_FREE_GB:-300}"
# Refuse to start an arm unless this much RAM is free, so a co-running GPU-0 arm cannot be dragged
# into OOM by Kairos starting underneath it.
MIN_FREE_RAM_GB="${MIN_FREE_RAM_GB:-120}"
LOCK="$ROOT/logs/kairos_gpu1.lock"
mkdir -p "$ROOT/logs"

exec 9>"$LOCK"
flock -n 9 || { echo "another kairos_gpu1.sh holds $LOCK -- exiting"; exit 0; }

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
free_gb() { df -BG --output=avail /data | tail -1 | tr -dc '0-9'; }
free_ram_gb() { free -g | awk 'NR==2{print $7}'; }

arm_done() {  # <detector> <dataset> -- identical to queue.sh's predicate
  find "$ROOT/$1/artifacts/training" \
    -path "*/$2/trained_models/model_epoch_299/state_dict.pkl" 2>/dev/null | grep -q .
}

run_one() {  # <dataset>
  local ds="$1" start mins avail ram
  if arm_done kairos "$ds"; then
    log "kairos / $ds: already trained -- skipping"
    return 0
  fi
  avail=$(free_gb)
  if [ "$avail" -lt "$MIN_FREE_GB" ]; then
    log "ABORT: only ${avail}G free on /data, below MIN_FREE_GB=${MIN_FREE_GB}G"
    exit 1
  fi
  ram=$(free_ram_gb)
  if [ "$ram" -lt "$MIN_FREE_RAM_GB" ]; then
    log "ABORT: only ${ram}G RAM available, below MIN_FREE_RAM_GB=${MIN_FREE_RAM_GB}G -- refusing"
    log "       to start Kairos next to whatever GPU 0 is running."
    exit 1
  fi
  log "kairos / $ds: starting on GPU $GPU (${avail}G disk, ${ram}G RAM free)"
  start=$(date +%s)
  "$ROOT/scripts/run_arm.sh" kairos "$ds" "$GPU" >/dev/null 2>&1
  mins=$(( ($(date +%s) - start) / 60 ))
  if arm_done kairos "$ds"; then
    log "kairos / $ds: COMPLETE (${mins} min)"
  else
    log "kairos / $ds: FAILED, no epoch-299 checkpoint (${mins} min) -- see $ROOT/kairos/logs/${ds}_e300.log"
  fi
}

log "=== kairos_gpu1.sh starting (pid $$, gpu $GPU) ==="
run_one CADETS_E3_V2_TARGET
run_one CADETS_E3_V2_SHADOW
log "=== kairos_gpu1.sh done ==="
