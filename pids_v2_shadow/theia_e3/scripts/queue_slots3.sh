#!/usr/bin/env bash
# Run the THEIA_E3 shadow-model campaign as a two-slot scheduler: one slot per GPU, each slot pulling
# the next unfinished arm off a flat work list. Supersedes queue_pair.sh.
#
# WHY NOT queue_pair.sh. That script ran the two arms of a detector concurrently but then waited for
# BOTH before starting the next detector. The arms are badly unbalanced -- the shadow's test day
# (04-12, 455,154 nodes) is 3x the target's (04-10, 148,687) -- and the per-epoch evaluation ladder
# is 151 evaluations that scale with test-set size. Measured on nodlink: target finished its whole
# run at 17:05 while shadow ran until ~18:44, leaving GPU 0 idle 1h35m. Across six paired detectors
# that barrier throws away most of a working day. Slots have no barrier: a GPU takes new work the
# moment it frees.
#
# RAM IS STILL THE CEILING, AND IT IS ENFORCED DYNAMICALLY, NOT BY SCHEDULE. 251 GB total. The old
# rule "never two different detectors at once" was a blunt proxy for "never blow the RAM ceiling".
# Here the proxy is replaced by the thing it stood for: MIN_RAM_GB free is required before any launch,
# so if a detector turns out to be a memory hog the second slot simply refuses to start and the
# campaign degrades gracefully to serial instead of OOMing. This matters -- an R-CAID spike has twice
# caused the harness to kill every background task in a session, not just the offending one.
#
# EXCLUSIVE DETECTORS run alone regardless of the gate, because a gate reading *current* RAM cannot
# see a spike that has not happened yet.
#
# rcaid WAS exclusive here and no longer is, on measured grounds. Holding it back was costing a whole
# idle GPU: flash/SHADOW occupies only 10.7 GB, 224 GB is free, and the rule was stalling rcaid behind
# it for ~5 h to guard against a peak that cannot land in that window. The orthrus pair ran at
# 95 + 74 = 169 GB combined with 73 GB still free, which is the real evidence for what this box
# tolerates; rcaid was ~47 GB on CADETS, so even at THEIA's ~2x inflation it lands near 95 GB. The
# MIN_RAM_GB gate stays the backstop -- if rcaid does surprise us the second slot refuses to launch
# and the tail degrades to serial by itself.
#
# kairos STAYS exclusive. It was ~79.5 GB on CADETS and is the one detector whose THEIA peak is both
# unmeasured and plausibly the largest -- orthrus already reached 95 GB here. It gets the whole box.
#
# ATTACH-SAFE. An arm already running (e.g. started by a previous queue) is bound to the slot matching
# its actual GPU, read from /proc/<pid>/environ, and waited on rather than relaunched. Two processes
# writing one artifact tree would corrupt it silently.
set -uo pipefail

ROOT=/data/pids_v2_shadow/theia_e3
MIN_FREE_GB="${MIN_FREE_GB:-300}"   # disk
MIN_RAM_GB="${MIN_RAM_GB:-100}"     # refuse to start an arm below this much free RAM
POLL="${POLL:-20}"
LOCK="$ROOT/logs/queue_slots3.lock"
mkdir -p "$ROOT/logs"

exec 9>"$LOCK"
flock -n 9 || { echo "another queue_slots holds $LOCK -- exiting"; exit 0; }

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >&2; }

DETS=(nodlink velox magic threatrace orthrus_non_snooped flash rcaid kairos)
EXCLUSIVE=" kairos "
GPUS=(0 1)

WORK=()
for d in "${DETS[@]}"; do WORK+=("$d|THEIA_E3_TARGET" "$d|THEIA_E3_SHADOW"); done

declare -A SLOT_PID SLOT_ARM SLOT_T0

arm_done() {  # <detector> <dataset>
  find "$ROOT/$1/artifacts/training" \
    -path "*/$2/trained_models/model_epoch_299/state_dict.pkl" 2>/dev/null | grep -q .
}
arm_pid()  { pgrep -f "[p]idsmaker[.]main $1 $2 " | head -1; }
free_gb()  { df -BG --output=avail /data | tail -1 | tr -dc '0-9'; }
ram_gb()   { free -g | awk '/^Mem:/{print $7}'; }
is_excl()  { [[ "$EXCLUSIVE" == *" $1 "* ]]; }
busy()     { local g; for g in "${GPUS[@]}"; do [ -n "${SLOT_PID[$g]:-}" ] && return 0; done; return 1; }

gpu_of_pid() {  # read CUDA_VISIBLE_DEVICES out of a running process
  tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1
}

report() {  # <det> <ds> <t0>
  local mins=$(( ($(date +%s) - $3) / 60 ))
  if arm_done "$1" "$2"; then log "$1 / $2: COMPLETE (${mins} min)"
  else log "$1 / $2: FAILED, no epoch-299 checkpoint (${mins} min) -- see $ROOT/$1/logs/$2_e300.log"; fi
}

# Bind arms that are already running to their real GPU slot before scheduling anything new.
for item in "${WORK[@]}"; do
  det="${item%%|*}"; ds="${item##*|}"
  pid=$(arm_pid "$det" "$ds")
  [ -z "$pid" ] && continue
  g=$(gpu_of_pid "$pid"); [[ "$g" =~ ^[01]$ ]] || g=0
  SLOT_PID[$g]=$pid; SLOT_ARM[$g]="$item"; SLOT_T0[$g]=$(date +%s)
  log "$det / $ds: already running (pid $pid, GPU $g) -- attached, not relaunched"
done

log "=== queue_slots.sh starting (pid $$) ==="
log "order: ${DETS[*]}  |  exclusive (run alone):${EXCLUSIVE}"
log "gates: >=${MIN_RAM_GB}G RAM and >=${MIN_FREE_GB}G disk per launch"

idx=0
while :; do
  # reap
  for g in "${GPUS[@]}"; do
    pid="${SLOT_PID[$g]:-}"
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      item="${SLOT_ARM[$g]}"
      report "${item%%|*}" "${item##*|}" "${SLOT_T0[$g]}"
      unset 'SLOT_PID[$g]' 'SLOT_ARM[$g]' 'SLOT_T0[$g]'
    fi
  done

  # dispatch
  for g in "${GPUS[@]}"; do
    [ -n "${SLOT_PID[$g]:-}" ] && continue
    [ "$idx" -ge "${#WORK[@]}" ] && continue

    item="${WORK[$idx]}"; det="${item%%|*}"; ds="${item##*|}"

    if arm_done "$det" "$ds"; then
      log "$det / $ds: already trained -- skipping"; idx=$((idx+1)); continue
    fi
    # An arm that is mid-flight is NOT done (no epoch-299 checkpoint yet) but must never be launched
    # a second time. The startup attach loop binds inherited arms to slots, but that alone is not
    # enough: the dispatch walk reaches the same work item later and, seeing only arm_done=false,
    # would start a duplicate on the other GPU. That happened on 2026-09-21 17:55 -- a second
    # flash/THEIA_E3_SHADOW ran for 79 s against the artifact tree of the live one before being
    # killed. Two writers on one tree corrupt it silently, and both arms share a log that run_arm.sh
    # opens through `tee`, which truncates. Check liveness, not just completion.
    running_pid=$(arm_pid "$det" "$ds")
    if [ -n "$running_pid" ]; then
      log "$det / $ds: already running (pid $running_pid) -- not launching a duplicate"
      idx=$((idx+1)); continue
    fi
    # An exclusive detector waits for the box to drain, and holds it while it runs.
    if is_excl "$det" && busy; then continue; fi
    # Never start a non-exclusive arm alongside a running exclusive one.
    if busy; then
      other="${SLOT_ARM[0]:-${SLOT_ARM[1]:-}}"
      is_excl "${other%%|*}" && continue
    fi

    disk=$(free_gb); ram=$(ram_gb)
    if [ "$disk" -lt "$MIN_FREE_GB" ]; then
      log "HOLD: ${disk}G disk free, below ${MIN_FREE_GB}G -- not launching $det / $ds"; continue
    fi
    if [ "$ram" -lt "$MIN_RAM_GB" ]; then
      log "HOLD: ${ram}G RAM free, below ${MIN_RAM_GB}G -- not launching $det / $ds"; continue
    fi

    "$ROOT/scripts/run_arm.sh" "$det" "$ds" "$g" >/dev/null 2>&1 &
    SLOT_PID[$g]=$!; SLOT_ARM[$g]="$item"; SLOT_T0[$g]=$(date +%s)
    log "$det / $ds: started on GPU $g (pid $!, ${ram}G RAM / ${disk}G disk free)"
    idx=$((idx+1))
  done

  [ "$idx" -ge "${#WORK[@]}" ] && ! busy && break
  sleep "$POLL"
done

log "=== queue_slots.sh done ==="
