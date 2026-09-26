#!/usr/bin/env bash
# Epoch-0 rung of the hop ladder, so He/Conti get the same untrained control that `ours_v2` has.
# Without it the strongest cell in the table (flash He 0.986) has no null: a high number against a
# trained model only means memorisation if the UNTRAINED model does not produce it too, and on this
# campaign the untrained model has matched the trained one every time.
set -uo pipefail
ROOT=/data/pids_v2_attacks/clearscope_e3
cd /home/yingjie/PIDSMaker
source /home/yingjie/anaconda3/etc/profile.d/conda.sh
conda activate pids
PY=/home/yingjie/anaconda3/envs/pids/bin/python
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >&2; }
GPU="${GPU:-0}"
for det in nodlink magic flash threatrace rcaid; do
  if [ "$det" = rcaid ]; then MEMFLAGS=(PIDS_LOW_MEMORY=1 PIDS_GRAD_CHECKPOINT=1); else MEMFLAGS=(); fi
  for arm in TARGET SHADOW; do
    t0=$(date +%s)
    env "${MEMFLAGS[@]}" OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="$GPU" \
      "$PY" "$ROOT/scripts/requery_hops_v2.py" --detector "$det" --arm "$arm" \
        --epochs 0 --hops 0,1 >> "$ROOT/logs/hops_${det}_${arm}.log" 2>&1
    log "$det/$arm e0: exit $? after $(( ($(date +%s)-t0)/60 )) min"
  done
done
log "=== queue_hops_e0.sh done ==="
