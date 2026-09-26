#!/usr/bin/env bash
# Two-slot parallel scheduler for the CLEARSCOPE_E3 shadow-model campaign, one slot per GPU.
# Mirrors ../../theia_e3/scripts/queue_slots5.sh; see that file for the pattern's history.
#
# WHY PARALLEL IS SAFE HERE AND WAS NOT ON THEIA. CLEARSCOPE's days are far smaller: 0.94-2.81M
# events per day against THEIA's 5-8M, and ~45k member nodes per arm against THEIA's 148k/455k.
# The measured THEIA per-arm peaks were rcaid 172 GB (VmHWM), orthrus 95-97 GB, velox 49 GB,
# nodlink 17 GB, flash 10.7 GB, magic 5.5 GB, and the note attached to them is that CADETS -- which
# is closer to CLEARSCOPE in size -- ran about 3x lower still (rcaid 47 GB there vs 172 GB here).
# Two light arms therefore fit comfortably in 251 GB. This is a size argument, not an optimistic
# one, but it is still a PREDICTION: the two heavyweights stay exclusive regardless.
#
# EXCLUSIVE = kairos, rcaid. Each gets the whole box.
#   - rcaid because DISK, not RAM, is its constraint: measured on the D1 CLEARSCOPE campaign its
#     feat_inference payload was 125 GB per arm, 235 GB of that campaign's 281 GB total against
#     ~46 GB for the other seven detectors combined. It is also queued LAST, so an out-of-disk abort
#     costs only rcaid and still leaves seven complete detector rows.
#   - kairos because it is the RAM hog and carries TGN memory buffers.
# The standing lesson from THEIA is that one detector's peak does not predict another's -- relaxing
# rcaid from EXCLUSIVE there killed an arm 175 minutes in. Do not widen this list from a successful
# light-arm pair.
#
# DISK GATE IS A LAUNCH GATE ONLY. It is checked before an arm starts and never again, so it cannot
# stop an arm already in flight. That is why rcaid gets its own higher bar (MIN_FREE_RCAID_GB)
# rather than sharing the general one.
#
# ATTACH-SAFE and duplicate-safe: an arm already running is bound to the slot matching its real GPU
# rather than relaunched, and the dispatch walk checks liveness as well as completion, because an
# arm mid-flight is not "done" but must never be started twice.
set -uo pipefail

ROOT=/data/pids_v2_shadow/clearscope_e3
MIN_FREE_GB="${MIN_FREE_GB:-150}"          # disk, general
MIN_FREE_RCAID_GB="${MIN_FREE_RCAID_GB:-350}"   # disk, before rcaid specifically
MIN_RAM_GB="${MIN_RAM_GB:-100}"            # refuse to start an arm below this much free RAM
POLL="${POLL:-20}"
LOCK="$ROOT/logs/queue_slots.lock"
mkdir -p "$ROOT/logs"

exec 9>"$LOCK"
flock -n 9 || { echo "another queue_slots holds $LOCK -- exiting"; exit 0; }

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >&2; }

# light detectors first so the two slots stay busy, then the exclusives, rcaid last
DETS=(nodlink magic threatrace flash velox orthrus_non_snooped kairos rcaid)
EXCLUSIVE=" kairos rcaid "
GPUS=(0 1)

WORK=()
for d in "${DETS[@]}"; do
  WORK+=("$d|CLEARSCOPE_E3_V2_TARGET" "$d|CLEARSCOPE_E3_V2_SHADOW")
done

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
disk_need() { if [ "$1" = rcaid ]; then echo "$MIN_FREE_RCAID_GB"; else echo "$MIN_FREE_GB"; fi; }

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
log "order: ${DETS[*]}"
log "exclusive (run alone):${EXCLUSIVE}"
log "gates: >=${MIN_RAM_GB}G RAM, >=${MIN_FREE_GB}G disk (>=${MIN_FREE_RCAID_GB}G before rcaid)"

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
    running_pid=$(arm_pid "$det" "$ds")
    if [ -n "$running_pid" ]; then
      log "$det / $ds: already running (pid $running_pid) -- not launching a duplicate"
      idx=$((idx+1)); continue
    fi
    # an exclusive arm waits for an empty box; a normal arm waits out an exclusive one
    if is_excl "$det" && busy; then continue; fi
    if busy; then
      other="${SLOT_ARM[0]:-${SLOT_ARM[1]:-}}"
      is_excl "${other%%|*}" && continue
    fi

    need=$(disk_need "$det")
    disk=$(free_gb); ram=$(ram_gb)
    if [ "$disk" -lt "$need" ]; then
      log "HOLD: ${disk}G disk free, below the ${need}G required for $det -- not launching $det / $ds"
      continue
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
