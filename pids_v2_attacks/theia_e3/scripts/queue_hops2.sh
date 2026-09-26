#!/usr/bin/env bash
# Hop re-query for the He-2021 / Conti-2022 ladder. Supersedes queue_hops.sh, which used the system
# python and died on `No module named yacs` -- the re-query imports pidsmaker itself, so it needs the
# same `pids` conda env the training arms ran under (run_arm.sh:95). The pure scorers (ours_v2 etc.)
# do not, which is why they ran fine without it.
#
# NODE DETECTORS ONLY. Hop 0 replaces the edge set with self-loops, which leaves an edge detector
# with nothing to score, so the ladder is undefined for velox / orthrus / kairos and those two
# methods are reported N/A for them. That is a property of the attack, not a gap in this run.
set -uo pipefail
ROOT=/data/pids_v2_attacks/theia_e3
source /home/yingjie/anaconda3/etc/profile.d/conda.sh
conda activate pids
cd /home/yingjie/PIDSMaker
mkdir -p "$ROOT/logs"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >&2; }
log "python: $(command -v python)"

for det in nodlink magic flash threatrace; do
  for arm in TARGET SHADOW; do
    L="$ROOT/logs/hops_${det}_${arm}.log"; t0=$(date +%s)
    log "$det/$arm: node_x pass"
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 \
      python -u "$ROOT/scripts/requery_hops_v2.py" --detector "$det" --arm "$arm" --dump-x > "$L" 2>&1
    log "$det/$arm: node_x exit $? ($(( ($(date +%s)-t0)/60 )) min)"
    log "$det/$arm: hop ladder 49,99,149,199,299"
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 \
      python -u "$ROOT/scripts/requery_hops_v2.py" --detector "$det" --arm "$arm" \
        --epochs 49,99,149,199,299 --hops 0,1 >> "$L" 2>&1
    log "$det/$arm: ladder exit $? (total $(( ($(date +%s)-t0)/60 )) min)"
  done
done
log "=== queue_hops2.sh done ==="
