#!/usr/bin/env bash
# Re-query the hop ladder for every v2 node detector x arm, to feed the He et al. and Conti et al.
# attack ports. Serial on one GPU: R-CAID peaks around 47 GB RSS and an R-CAID spike has twice
# mass-killed a session, so nothing here runs concurrently with anything else.
#
# Edge detectors are absent on purpose -- hop 0 replaces the edge set with self-loops, which leaves
# an edge model with nothing to score. The driver refuses them rather than approximating.
set -uo pipefail

REPO=/home/yingjie/PIDSMaker
PY=/home/yingjie/anaconda3/envs/pids/bin/python
DRIVER=/data/pids_v2_attacks/cadets_e3/scripts/requery_hops_v2.py
LOG=/data/pids_v2_attacks/cadets_e3/results/sweep_hops_v2.log
EPOCHS="${EPOCHS:-49,99,149,199,299}"
GPU="${GPU:-0}"

mkdir -p "$(dirname "$LOG")"
cd "$REPO" || exit 1

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "=== sweep_hops_v2 starting (pid $$, gpu $GPU, epochs $EPOCHS) ==="
start_all=$(date +%s)

for det in nodlink magic flash rcaid threatrace; do
  for arm in TARGET SHADOW; do
    ram=$(free -g | awk 'NR==2{print $7}')
    if [ "$ram" -lt 60 ]; then
      log "ABORT: only ${ram}G RAM free -- refusing to start $det/$arm"
      exit 1
    fi
    log "--- $det / $arm (${ram}G RAM free)"
    t=$(date +%s)
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$DRIVER" \
      --detector "$det" --arm "$arm" --epochs "$EPOCHS" --hops 0,1 \
      >>"$LOG.$det.$arm" 2>&1
    rc=$?
    log "    hops rc=$rc in $(( ($(date +%s) - t) / 60 ))m"
    [ "$rc" -ne 0 ] && { log "    FAILED -- see $LOG.$det.$arm"; continue; }

    # node_x is model-independent, so it is dumped once per arm and reused by every epoch.
    t=$(date +%s)
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$DRIVER" \
      --detector "$det" --arm "$arm" --dump-x \
      >>"$LOG.$det.$arm" 2>&1
    log "    node_x rc=$? in $(( ($(date +%s) - t) / 60 ))m"
  done
done

log "=== sweep_hops_v2 done in $(( ($(date +%s) - start_all) / 60 ))m ==="
