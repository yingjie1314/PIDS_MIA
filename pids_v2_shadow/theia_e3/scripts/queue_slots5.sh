#!/usr/bin/env bash
# Final generation of the THEIA_E3 scheduler. Supersedes queue_slots4.sh. Only kairos remains.
#
# R-CAID IS EXCLUDED FROM THE THEIA CAMPAIGN. It is not a scheduling decision that can be tuned --
# R-CAID does not fit on this partition, by a factor of about three, and no amount of pruning or
# serialisation changes that. The numbers, all measured on 2026-09-21 and independently re-derived
# by a second pass before acting:
#
#   R-CAID's feat_inference output is 11.13x its transformation output. This was derived from
#   scratch by pairing every window file under transformation/.../nx/graph_2018-04-10 against its
#   counterpart under feat_inference/.../edge_embeds/test: 7.77 GB of graphs -> 86.42 GB of
#   embeddings, ratio 11.128. On CADETS the same detector produced 13.2 GB of feat_inference per
#   arm; on THEIA the transformation alone is ~49 GB (TARGET, complete) and ~244 GB projected
#   (SHADOW), because THEIA's days carry 5-8M events against CADETS's far smaller ones.
#
#     rcaid/SHADOW feat_inference   ~2700 GB   (244 GB transformation x 11.13)
#     rcaid/TARGET re-run, net      ~ 381 GB
#     total remaining demand        ~3100 GB
#     free on /data                   987 GB
#     usable above the 300 GB floor   687 GB
#     reclaimable by pruning the 6 finished detectors' intermediates   ~280 GB
#     best case available             ~967 GB
#
#   There is no second filesystem to spill to: / has 119 GB.
#
# WHY THIS HAD TO BE STOPPED RATHER THAN LEFT TO FAIL. MIN_FREE_GB is a *launch* gate. It is checked
# before an arm starts and never again, so it cannot stop an arm already in flight. rcaid/SHADOW
# would have written until ENOSPC -- roughly 6-10 h of doomed compute -- and left the partition too
# full for kairos to pass its own launch gate. Killing it cost the 1h45m it had spent and protected
# the arms that can actually finish. rcaid/SHADOW was NOT failing on memory: its RSS is a
# per-window sawtooth with VmHWM pinned at 172 GB, cycling 17->158 GB and surviving each peak.
# Disk was always the binding constraint; the memory scare was a misread of a mid-write sample.
#
# The exclusion is reportable, not just operational: an 11x feat_inference amplification is a real
# scalability property of R-CAID, and the THEIA write-up covers 7 detectors with this stated.
#
# KAIROS IS EXCLUSIVE and gets the whole box. Predicted peak is ~120-140 GB on the SHADOW arm,
# anchored on the MEASURED orthrus THEIA_SHADOW figure of ~97 GB at the same emb_dim=256 on the same
# data, plus a kairos-specific increment for edge_features=msg (~13 GB for g.edge_feats at
# 6.354M x 528 x 4B) and the TGN memory module that orthrus does not enable. The "~79.5 GB" figure
# quoted in every previous generation of this script is unsourced -- no telemetry file backing it
# exists anywhere under /data -- and it came from a NOFRAC run holding ~1.47x more graph than either
# v2 arm, so it was never a v2 baseline. Do not propagate it further.
#
# ATTACH-SAFE and duplicate-safe, unchanged from queue_slots4.sh: an inherited arm is bound to the
# slot matching its real GPU rather than relaunched, and the dispatch walk checks liveness as well as
# completion, because an arm mid-flight is not "done" but must never be started twice.
set -uo pipefail

ROOT=/data/pids_v2_shadow/theia_e3
MIN_FREE_GB="${MIN_FREE_GB:-300}"   # disk
MIN_RAM_GB="${MIN_RAM_GB:-100}"     # refuse to start an arm below this much free RAM
POLL="${POLL:-20}"
LOCK="$ROOT/logs/queue_slots5.lock"
mkdir -p "$ROOT/logs"

exec 9>"$LOCK"
flock -n 9 || { echo "another queue_slots holds $LOCK -- exiting"; exit 0; }

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >&2; }

DETS=(nodlink velox magic threatrace orthrus_non_snooped flash kairos)
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

log "=== queue_slots5.sh starting (pid $$) ==="
log "order: ${DETS[*]}  |  exclusive (run alone):${EXCLUSIVE}"
log "rcaid EXCLUDED -- does not fit on /data (see header)"
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
    running_pid=$(arm_pid "$det" "$ds")
    if [ -n "$running_pid" ]; then
      log "$det / $ds: already running (pid $running_pid) -- not launching a duplicate"
      idx=$((idx+1)); continue
    fi
    if is_excl "$det" && busy; then continue; fi
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

log "=== queue_slots5.sh done ==="
