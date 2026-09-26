#!/usr/bin/env bash
# Hop re-query for the He-2021 / Conti-2022 ladder. Node detectors only -- hop 0 self-loops leave an
# edge detector with nothing to score, so velox/orthrus/kairos are N/A for those two methods.
# Serial on one GPU: each run rebuilds the model and replays every window at 5 checkpoints.
set -uo pipefail
ROOT=/data/pids_v2_attacks/theia_e3
cd /home/yingjie/PIDSMaker
mkdir -p "$ROOT/logs"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >&2; }

for det in nodlink magic flash threatrace; do
  for arm in TARGET SHADOW; do
    L="$ROOT/logs/hops_${det}_${arm}.log"
    t0=$(date +%s)
    log "$det/$arm: node_x pass"
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 \
      python3 "$ROOT/scripts/requery_hops_v2.py" --detector "$det" --arm "$arm" --dump-x \
      > "$L" 2>&1
    log "$det/$arm: hop ladder, epochs 49,99,149,199,299"
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 \
      python3 "$ROOT/scripts/requery_hops_v2.py" --detector "$det" --arm "$arm" \
        --epochs 49,99,149,199,299 --hops 0,1 \
      >> "$L" 2>&1
    rc=$?
    log "$det/$arm: exit $rc after $(( ($(date +%s)-t0)/60 )) min"
  done
done
log "=== queue_hops.sh done ==="
