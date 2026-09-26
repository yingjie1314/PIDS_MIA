#!/usr/bin/env bash
# Ablation 2 -- wait for the training runners, then score every cell and rebuild the table.
#
# WHY A CHAIN RATHER THAN A POLL. score_frac.py makes a single pass and silently skips cells whose
# dumps do not exist yet, so running it early leaves permanent holes that only another full pass
# fills. Blocking on the runners means exactly one pass is needed and it sees every cell.
#
# `tail --pid= -f /dev/null` is used instead of `wait` (these are not our children) and instead of a
# pgrep loop (a pgrep pattern for run_frac.sh also matches this script's own command line).
set -uo pipefail

ROOT=/data/pids_v2_ablation
PY=/home/yingjie/anaconda3/envs/pids/bin/python
LOG="$ROOT/results/finish_frac.log"
mkdir -p "$ROOT/results"

exec >>"$LOG" 2>&1
echo "=== $(date '+%F %T') waiting on runners ${RUNNERS:?set RUNNERS to a space-separated PID list} ==="
for p in $RUNNERS; do
  [ -d "/proc/$p" ] || { echo "pid $p already gone"; continue; }
  tail --pid="$p" -f /dev/null
  echo "$(date '+%F %T') runner $p exited"
done

# sklearn's GBDT under an unbounded OpenMP pool is ~80x slower on this 48-core box.
cd "$ROOT" || exit 1
echo "=== $(date '+%F %T') scoring ==="
OMP_NUM_THREADS=8 "$PY" scripts/score_frac.py
# Must run before the table: score_frac's n_members is post-subsample and pins at 30000, so without
# this the new cells report a capped size and the size axis the ablation exists to measure is lost.
echo "=== $(date '+%F %T') recovering true sizes ==="
OMP_NUM_THREADS=8 "$PY" scripts/frac_sizes.py
echo "=== $(date '+%F %T') building table ==="
"$PY" scripts/build_frac_table.py
echo "=== $(date '+%F %T') done ==="
