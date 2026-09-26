#!/usr/bin/env bash
# The SHADOW half of the R-CAID retry, started in PARALLEL with TARGET rather than after it.
#
# WHY THIS EXISTS SEPARATELY FROM retry_rcaid_lowmem.sh
#   That script ran the two arms serially, on the standing rule that R-CAID is the detector which
#   already took the campaign down once. The serial choice was made before either arm's cost was
#   known. Measured on TARGET, it is wrong: training peaks at 20 GB host RSS and 22.7 GB on one
#   card, so a second arm on the other card is comfortably inside 251 GB / 2x32 GB.
#
#   What made serial expensive is that training is the SHORT phase. TARGET trained 300 epochs in
#   1h13m (09:49:46 -> 11:03:04) and has spent 5h+ since in the post-training detection sweep,
#   which walks ~150 saved checkpoints at ~2m22s each. Serial would have idled GPU 1 through all
#   of that and finished both arms around 23:30; in parallel they finish around 17:30.
#
#   So retry_rcaid_lowmem.sh's scheduler (and its sampler) were killed by PID while TARGET's own
#   python was deliberately left running and reparented -- TARGET's epoch-299 checkpoint and all
#   11 MIA dumps were already on disk, so the only thing still at risk there was detection
#   metrics, not membership data.
#
# BOTH MEMORY FLAGS ARE REQUIRED. See retry_rcaid_lowmem.sh's header for the three stacked walls;
# in short PIDS_LOW_MEMORY=1 keeps host RAM at ~20 GB instead of ~234 GB, and PIDS_GRAD_CHECKPOINT=1
# caps GAT per-edge transients in both training and the no_grad MIA dump.
set -euo pipefail

ROOT=/data/pids_v2_shadow/clearscope_e3
DATASET=CLEARSCOPE_E3_V2_SHADOW
GPU="${GPU:-1}"

export PIDS_LOW_MEMORY=1
export PIDS_GRAD_CHECKPOINT=1

sample() {
  local peak=0 pid rss gpu_mb
  while :; do
    pid=$(pgrep -f "[p]idsmaker[.]main rcaid $DATASET " | head -1 || true)
    if [ -n "$pid" ]; then
      rss=$(awk '/^VmRSS:/{print int($2/1048576)}' "/proc/$pid/status" 2>/dev/null || echo 0)
      [ -n "$rss" ] && [ "$rss" -gt "$peak" ] && peak="$rss"
      gpu_mb=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
                 2>/dev/null | awk -F', ' -v p="$pid" '$1==p{print $2}' | head -1)
      printf '%s rss=%sGB peak=%sGB free=%sGB gpu=%sMiB\n' "$(date +%H:%M:%S)" "$rss" "$peak" \
        "$(free -g | awk '/^Mem:/{print $7}')" "${gpu_mb:-0}" \
        >> "$ROOT/logs/rss_rcaid_${DATASET}.log"
    fi
    sleep 30
  done
}

sample &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null || true' EXIT

echo "=== $(date '+%F %T') starting rcaid / $DATASET on gpu $GPU (parallel with TARGET) ==="
bash "$ROOT/scripts/run_arm.sh" rcaid "$DATASET" "$GPU"

if find "$ROOT/rcaid/artifacts/training" \
     -path "*/$DATASET/trained_models/model_epoch_299/state_dict.pkl" 2>/dev/null | grep -q .; then
  echo "=== $(date '+%F %T') rcaid / $DATASET COMPLETE (epoch_299 checkpoint present) ==="
else
  echo "=== $(date '+%F %T') rcaid / $DATASET EXITED 0 BUT NO epoch_299 CHECKPOINT ==="
  exit 1
fi
