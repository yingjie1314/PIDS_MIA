#!/usr/bin/env bash
# Hop re-query for the He-2021 / Conti-2022 ladder. Node detectors only -- hop 0 self-loops leave an
# edge detector with nothing to score, so velox/orthrus/kairos are N/A for those two methods.
# Serial on one GPU: each run rebuilds the model and replays every window at 5 checkpoints.
#
# R-CAID IS INCLUDED HERE AND WAS NOT ON THEIA. THEIA dropped it for an 11x disk blowup; CLEARSCOPE
# trained both arms successfully, so it gets the full ladder. It needs the same two memory flags its
# training did (see /data/pids_v2_shadow/clearscope_e3/RCAID_RETRY.md): the eager loader would want
# ~234 GB of host RAM for these windows, and this pass runs the encoder under no_grad on the same
# 4.6M-edge windows that OOM'd a 32 GB card during the epoch-0 MIA dump. Without both flags this
# pass fails the same way training did -- and failing here is worse, because a partially written
# ladder produces a hole rather than an error in the scorers.
set -uo pipefail
ROOT=/data/pids_v2_attacks/clearscope_e3
cd /home/yingjie/PIDSMaker
mkdir -p "$ROOT/logs"

# Activate explicitly rather than inheriting the caller's shell. THEIA's copy used a bare `python3`
# and happened to work because it was launched from an already-activated terminal; launched from a
# background job here, `python3` is the system interpreter and every run died on `No module named
# 'yacs'` in under a second -- twelve "completed" runs, no dumps, exit 1 each.
source /home/yingjie/anaconda3/etc/profile.d/conda.sh
conda activate pids
PY=/home/yingjie/anaconda3/envs/pids/bin/python
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >&2; }

# GPU 0: rcaid's SHADOW detection sweep still holds GPU 1 until ~21:45.
GPU="${GPU:-0}"

for det in nodlink magic flash threatrace rcaid; do
  # Opt-in execution-strategy flags; inert for every detector except rcaid, and deliberately kept
  # out of the artifact hash so they cannot re-resolve these runs to a different tree.
  if [ "$det" = rcaid ]; then
    MEMFLAGS=(PIDS_LOW_MEMORY=1 PIDS_GRAD_CHECKPOINT=1)
  else
    MEMFLAGS=()
  fi
  for arm in TARGET SHADOW; do
    L="$ROOT/logs/hops_${det}_${arm}.log"
    t0=$(date +%s)
    log "$det/$arm: node_x pass"
    env "${MEMFLAGS[@]}" OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="$GPU" \
      "$PY" "$ROOT/scripts/requery_hops_v2.py" --detector "$det" --arm "$arm" --dump-x \
      > "$L" 2>&1
    log "$det/$arm: hop ladder, epochs 49,99,149,199,299"
    env "${MEMFLAGS[@]}" OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="$GPU" \
      "$PY" "$ROOT/scripts/requery_hops_v2.py" --detector "$det" --arm "$arm" \
        --epochs 49,99,149,199,299 --hops 0,1 \
      >> "$L" 2>&1
    rc=$?
    log "$det/$arm: exit $rc after $(( ($(date +%s)-t0)/60 )) min"
  done
done
log "=== queue_hops.sh done ==="
