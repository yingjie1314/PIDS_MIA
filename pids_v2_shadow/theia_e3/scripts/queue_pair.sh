#!/usr/bin/env bash
# Run the THEIA_E3 shadow-model campaign with BOTH GPUs: the two arms of one detector run
# concurrently, target on GPU 0 and shadow on GPU 1. Replaces the strictly-serial queue.sh.
#
# WHY PAIRING THE ARMS IS THE SAFE FORM OF PARALLELISM. The two arms of a detector are fully
# independent -- disjoint days, disjoint artifact subtrees (everything is keyed by dataset name),
# separate logs and separate run_flags_<DATASET>.json. Nothing is shared but the source database,
# which is read-only here. Running two DIFFERENT detectors at once would instead have them contend
# for the same RAM ceiling at unpredictable phases, which is what the serial queue existed to avoid.
#
# RAM, NOT GPU, REMAINS THE BINDING CONSTRAINT. 251 GB total. On CADETS_E3 Kairos peaked near
# 79.5 GB and R-CAID near 47 GB; THEIA's member days are 2-4x larger in events, so those peaks are
# lower bounds here, and doubling either one is not worth the risk -- an R-CAID spike has twice
# caused the harness to kill every background task in a session, not just the offending one.
# SERIAL_DETS therefore fall back to one-arm-at-a-time on GPU 0. Everything else pairs.
#
# RESUMABLE AND ATTACH-SAFE. An arm is "done" when its epoch-299 checkpoint exists, and an arm that
# is already running (e.g. started by the earlier serial queue.sh) is waited on rather than launched
# a second time. Two processes writing one artifact tree would corrupt it silently, so this check
# is not optional.
set -uo pipefail

ROOT=/data/pids_v2_shadow/theia_e3
MIN_FREE_GB="${MIN_FREE_GB:-300}"
MIN_RAM_GB="${MIN_RAM_GB:-60}"
# Deliberately NOT queue.sh's lock. `exec 9>` leaks the descriptor into every child, so a training
# process launched by the old serial queue keeps that lock held long after the queue itself is gone
# -- it tracks the run, not the queue. The real guard against two writers on one artifact tree is
# the per-arm arm_pid() check in start_arm, which is finer-grained than either lock.
LOCK="$ROOT/logs/queue_pair.lock"
mkdir -p "$ROOT/logs"

exec 9>"$LOCK"
flock -n 9 || { echo "another queue holds $LOCK -- exiting"; exit 0; }

# stderr, not stdout: start_arm returns a pid on stdout via command substitution, so anything else
# printed there would be swallowed into the caller's variable instead of reaching the log.
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >&2; }

DETS=(nodlink velox magic threatrace orthrus_non_snooped flash rcaid kairos)
SERIAL_DETS=" rcaid kairos "   # too RAM-hungry to double up

arm_done() {  # <detector> <dataset>
  find "$ROOT/$1/artifacts/training" \
    -path "*/$2/trained_models/model_epoch_299/state_dict.pkl" 2>/dev/null | grep -q .
}

arm_pid() {  # <detector> <dataset> -> pid on stdout, empty if not running
  pgrep -f "[p]idsmaker[.]main $1 $2 " | head -1
}

free_gb() { df -BG --output=avail /data | tail -1 | tr -dc '0-9'; }
ram_gb()  { free -g | awk '/^Mem:/{print $7}'; }

gate() {  # aborts the whole queue if the box is too full to continue safely
  local avail ram
  avail=$(free_gb); ram=$(ram_gb)
  if [ "$avail" -lt "$MIN_FREE_GB" ]; then
    log "ABORT: only ${avail}G free on /data, below MIN_FREE_GB=${MIN_FREE_GB}G"; exit 1
  fi
  if [ "$ram" -lt "$MIN_RAM_GB" ]; then
    log "ABORT: only ${ram}G RAM available, below MIN_RAM_GB=${MIN_RAM_GB}G"; exit 1
  fi
}

# Starts an arm in the background and echoes its pid, or echoes the pid of the one already running,
# or echoes nothing if the arm is already trained.
start_arm() {  # <detector> <dataset> <gpu>
  local det="$1" ds="$2" gpu="$3" pid
  if arm_done "$det" "$ds"; then
    log "$det / $ds: already trained -- skipping"; return 0
  fi
  pid=$(arm_pid "$det" "$ds")
  if [ -n "$pid" ]; then
    log "$det / $ds: already running (pid $pid) -- attaching instead of relaunching"
    echo "$pid"; return 0
  fi
  "$ROOT/scripts/run_arm.sh" "$det" "$ds" "$gpu" >/dev/null 2>&1 &
  log "$det / $ds: started on GPU $gpu (wrapper pid $!)"
  echo "$!"
}

# Waits on a pid whether or not it is our child. `wait` only works for children; an arm inherited
# from the previous queue is not one, so fall back to tail --pid.
wait_pid() {
  local pid="$1"
  [ -z "$pid" ] && return 0
  wait "$pid" 2>/dev/null || tail --pid="$pid" -f /dev/null
}

report() {  # <detector> <dataset> <start_epoch_seconds>
  local mins=$(( ($(date +%s) - $3) / 60 ))
  if arm_done "$1" "$2"; then
    log "$1 / $2: COMPLETE (${mins} min)"
  else
    log "$1 / $2: FAILED, no epoch-299 checkpoint (${mins} min) -- see $ROOT/$1/logs/$2_e300.log"
  fi
}

run_paired() {  # <detector>
  local det="$1" start p0 p1
  gate
  log "--- $det: target(GPU0) + shadow(GPU1) in parallel ---"
  start=$(date +%s)
  p0=$(start_arm "$det" THEIA_E3_TARGET 0 | tail -1)
  p1=$(start_arm "$det" THEIA_E3_SHADOW 1 | tail -1)
  wait_pid "$p0"; wait_pid "$p1"
  report "$det" THEIA_E3_TARGET "$start"
  report "$det" THEIA_E3_SHADOW "$start"
}

run_serial() {  # <detector>
  local det="$1" start p
  log "--- $det: serial, one arm at a time (RAM ceiling) ---"
  for ds in THEIA_E3_TARGET THEIA_E3_SHADOW; do
    gate
    start=$(date +%s)
    p=$(start_arm "$det" "$ds" 0 | tail -1)
    wait_pid "$p"
    report "$det" "$ds" "$start"
  done
}

log "=== queue_pair.sh starting (pid $$) ==="
log "paired on 2 GPUs: ${DETS[*]}  |  forced serial:${SERIAL_DETS}"

for det in "${DETS[@]}"; do
  if [[ "$SERIAL_DETS" == *" $det "* ]]; then
    run_serial "$det"
  else
    run_paired "$det"
  fi
done

log "=== queue_pair.sh done ==="
