#!/usr/bin/env bash
# Ablation 1 (early-stop ladder): extend the He-2021 / Conti-2022 hop re-query down to the early
# epochs. The published tables only ever scored 49/99/149/199/299, so the question "does stopping
# early reduce leakage" currently has no answer below epoch 49 for these two methods -- and epoch 0
# is the control that decides whether any of it is memorisation at all.
#
# Node detectors only: hop 0 replaces the edge set with self-loops, so velox/orthrus/kairos have
# nothing to score and are N/A by construction, not by omission.
#
# The node_x pass is NOT repeated -- it is epoch-independent and was written when the 49..299 ladder
# was built. Re-running it would only rewrite identical features.
#
# Usage: DATASET=cadets_e3 GPU=0 ./queue_ablation_hops.sh
set -uo pipefail

DATASET="${DATASET:?set DATASET to cadets_e3|theia_e3|clearscope_e3}"
GPU="${GPU:-0}"
SRC=/data/pids_v2_attacks/$DATASET
ROOT=/data/pids_v2_ablation/$DATASET
mkdir -p "$ROOT/logs"
cd /home/yingjie/PIDSMaker || exit 1

# Explicit interpreter, not a bare `python3`. Launched from a background job the system python has
# no yacs and every run dies in under a second with exit 1 -- which looks like a completed queue.
PY=/home/yingjie/anaconda3/envs/pids/bin/python

# clearscope already has its epoch-0 rung; the other two do not.
if [ "$DATASET" = clearscope_e3 ]; then EPOCHS=1,2,5,10,24; else EPOCHS=0,1,2,5,10,24; fi

# theia dropped rcaid for an 11x disk blowup, so it ships four node detectors.
if [ "$DATASET" = theia_e3 ]; then DETS="nodlink magic flash threatrace"
else DETS="nodlink magic flash threatrace rcaid"; fi

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $DATASET $*" >&2; }
log "=== ablation hops: epochs $EPOCHS on GPU $GPU ==="

for det in $DETS; do
  # Opt-in execution-strategy flags, deliberately outside the artifact hash so they cannot
  # re-resolve a run to a different tree. Inert for everything except rcaid, whose eager loader
  # wants ~234 GB of host RAM for clearscope's windows.
  if [ "$det" = rcaid ] && [ "$DATASET" = clearscope_e3 ]; then
    MEMFLAGS=(PIDS_LOW_MEMORY=1 PIDS_GRAD_CHECKPOINT=1)
  else
    MEMFLAGS=()
  fi
  for arm in TARGET SHADOW; do
    L="$ROOT/logs/ablhops_${det}_${arm}.log"
    t0=$(date +%s)
    log "$det/$arm: hop ladder $EPOCHS"
    env "${MEMFLAGS[@]}" OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="$GPU" \
      "$PY" "$SRC/scripts/requery_hops_v2.py" --detector "$det" --arm "$arm" \
        --epochs "$EPOCHS" --hops 0,1 \
      > "$L" 2>&1
    log "$det/$arm: exit $? after $(( ($(date +%s)-t0)/60 )) min"
  done
done
log "=== done ==="
