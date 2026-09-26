#!/usr/bin/env bash
# Retry the two R-CAID arms that were OOM-killed on 2026-09-24, using the low-memory loader.
#
# WHY THE FIRST ATTEMPT DIED
#   Both arms stopped inside training_loop's "Computing graphs..." step with no Python traceback
#   (the OOM signature on this box -- dmesg is unreadable, journalctl empty). Cause is size, not
#   contention: R-CAID's pseudo-graph inflates feat_inference to 124 GB (target) + 108 GB (shadow)
#   on CLEARSCOPE against 27 GB for BOTH CADETS arms combined. The eager loader holds every
#   window's `msg` plus the x_src/x_dst slices derived from it resident across all three splits,
#   ~1.9x the on-disk payload, which overruns the ~234 GB that was free. Re-running unchanged on an
#   idle box would fail the same way.
#
# WHY PIDS_LOW_MEMORY=1 IS THE RIGHT FIX AND NOT A WORKAROUND
#   pidsmaker/utils/data_utils.py ships an opt-in windowed loader written for this exact failure;
#   its docstring names R-CAID, CLEARSCOPE and "~110 GB per role". It loads one window at a time
#   and drops msg/x_src/x_dst after reindexing, which nothing in a non-TGN run reads again.
#   All eight _assert_low_memory_compatible gates were checked against the resolved config of both
#   arms before this script was written: encoder=rcaid_gat (no tgn), multi_dataset_training=None,
#   save_on_disk=False, global/intra/inter batching all 'none', node_features='node_emb' (no
#   edges_distribution), edge_features='none' (no msg). If any of that drifts the loader raises
#   rather than silently falling back to the path that OOMs.
#
# THERE ARE TWO SIZE WALLS, NOT ONE. The 09:32 attempt with PIDS_LOW_MEMORY=1 alone proved the
# host-RAM fix works -- peak RSS 11 GB against the ~234 GB the eager path consumed -- and then died
# in R-CAID's second GAT layer with `torch.cuda.OutOfMemoryError: Tried to allocate 9.38 GiB
# (31.48 GiB total; 20.11 GiB allocated; 29.25 GiB reserved)`. The host-RAM wall had simply been
# hit first and was masking a GPU wall behind it.
#
#   Defragmentation alone does NOT fix this first wall, however tempting the numbers look: the
#   reserved-vs-allocated gap (~9.14 GiB) appears to cover the failed 9.38 GiB allocation exactly.
#   rcaid_encoder.py:99 records a measured 33.7 GB TRUE peak for these windows, past a 32 GB card
#   even with zero fragmentation. Perfect defrag would move that crash, not remove it.
#
#   PIDS_GRAD_CHECKPOINT=1 is the real fix and, like PIDS_LOW_MEMORY, already ships for this exact
#   case. It checkpoints GAT one attention HEAD at a time -- an exact decomposition of concat-mode
#   multi-head GAT, since head h touches only its own slice of lin.weight/att_src/att_dst before
#   the concat -- so per-edge transients drop from (E, H, C) to (E, C), 4x with 4 heads, at ~1/3
#   extra forward compute. Equivalence of forward AND gradients is pinned by
#   tests/test_rcaid_grad_checkpoint_equivalence.py, which was run green before this relaunch.
#
# AND A THIRD WALL, IN VALIDATION. With both flags set, epoch 0 trained clean at 11.33 GB GPU --
# head-chunking works -- and then died in VALIDATION allocating 11.68 GiB. Cause: rcaid_encoder.py
# gates the chunked path on `self.training and torch.is_grad_enabled()`, so eval takes the eager
# GAT and materialises one (E, H, C) message tensor whole. CLEARSCOPE's largest test window has
# 4,625,075 edges; x 4 heads x 128 dims x 4 B is ~9.5 GB before self-loops, hence 11.68 GiB.
#
#   It is specifically the epoch-0 MIA DUMP, not ordinary validation: mia_dump.py::_dump_split ->
#   inference_loop.test_node_level, which is decorated @torch.no_grad(). That is why the gate
#   missed -- `torch.is_grad_enabled()` is False there, so the eager branch ran even with the flag
#   set. Under no_grad there is nothing to checkpoint, but the per-head SPLIT still matters: it
#   caps the peak transient at (E, C) per head instead of one (E, H, C) tensor, ~4x.
#
#   An allocator tweak was tried first and is a dead end on this box. PYTORCH_CUDA_ALLOC_CONF=
#   expandable_segments:True is rejected outright by torch 1.13.1 ("Unrecognized CachingAllocator
#   option"); it landed in torch 2.1. The arithmetic had looked right -- 12.61 GiB allocated
#   against 21.67 GiB reserved strands ~9 GiB, and the true requirement of ~24.3 GiB fits the
#   31.48 GiB card -- but the option does not exist to be set.
#
#   So rcaid_encoder.py was changed instead, in two places: the forward gate is now
#   `_grad_checkpoint_enabled()` alone, and `_gat_head_chunked` calls `_gat_one_head` directly
#   when grad is off rather than routing through `checkpoint`. Still inert unless the flag is set,
#   so no other campaign's numbers move. tests/test_rcaid_grad_checkpoint_equivalence.py was
#   extended to cover eval with and without no_grad (5 tests, green before this relaunch).
#
# THE ONE CAVEAT WORTH RECORDING
#   Reindexing moves to CPU on this path (the eager path uses the device). Scatter-mean is
#   deterministic on CPU and only differs from CUDA by nondeterministic atomics, so results are
#   equivalent up to float ordering -- but R-CAID's CLEARSCOPE rows are produced under a different
#   loading path than the other seven detectors here and than R-CAID on CADETS. Say so in the
#   write-up rather than letting a reader assume uniformity.
#
# NO PREPROCESSING IS REDONE. PIDS_LOW_MEMORY is an environment variable, deliberately kept out of
# the artifact hash chain (data_utils.py:114), so the 231 GB of construction/transformation/
# featurization/feat_inference already on disk resolves to the same hash and is reused. These runs
# start at training.
#
# SERIAL BY CHOICE. The low-memory path should make each arm cheap enough to pair, but R-CAID is
# the detector that just took down the campaign and one detector's peak has never predicted
# another's here. Arm 1's measured peak (see the sampler below) is the evidence for whether
# pairing is ever safe; until then, one at a time.
set -euo pipefail

ROOT=/data/pids_v2_shadow/clearscope_e3
cd "$ROOT"

export PIDS_LOW_MEMORY=1        # host RAM: windowed loading, drops msg/x_src/x_dst after reindex
export PIDS_GRAD_CHECKPOINT=1   # GPU RAM, training: per-head checkpointing in the GAT layers
# (no PYTORCH_CUDA_ALLOC_CONF: torch 1.13.1 rejects expandable_segments outright -- see below)

# Sample RSS of the arm's python process so the peak is measured, not guessed. The first attempt
# left no memory evidence at all, which is why its diagnosis had to be reconstructed from payload
# sizes on disk.
sample_rss() {
  local dataset="$1" out="$2"
  local peak=0
  while :; do
    local pid rss
    pid=$(pgrep -f "[p]idsmaker[.]main rcaid $dataset " | head -1 || true)
    if [ -n "$pid" ]; then
      rss=$(awk '/^VmRSS:/{print int($2/1048576)}' "/proc/$pid/status" 2>/dev/null || echo 0)
      [ -n "$rss" ] && [ "$rss" -gt "$peak" ] && peak="$rss"
      # GPU is the binding constraint now, so sample it too rather than inferring it from a crash.
      local gpu_mb
      gpu_mb=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
                 2>/dev/null | awk -F', ' -v p="$pid" '$1==p{print $2}' | head -1)
      printf '%s rss=%sGB peak=%sGB free=%sGB gpu=%sMiB\n' "$(date +%H:%M:%S)" "$rss" "$peak" \
        "$(free -g | awk '/^Mem:/{print $7}')" "${gpu_mb:-0}" >> "$out"
    fi
    sleep 30
  done
}

run_arm() {
  local dataset="$1" gpu="$2"
  echo "=== $(date '+%F %T') starting rcaid / $dataset on gpu $gpu (PIDS_LOW_MEMORY=1) ==="
  sample_rss "$dataset" "$ROOT/logs/rss_rcaid_${dataset}.log" &
  local sampler=$!
  local rc=0
  bash "$ROOT/scripts/run_arm.sh" rcaid "$dataset" "$gpu" || rc=$?
  kill "$sampler" 2>/dev/null || true
  if [ "$rc" -ne 0 ]; then
    echo "=== $(date '+%F %T') rcaid / $dataset FAILED rc=$rc ==="
    return "$rc"
  fi
  # Trust the checkpoint on disk, not the exit code: the first attempt's wrapper also exited 0.
  if find "$ROOT/rcaid/artifacts/training" \
       -path "*/$dataset/trained_models/model_epoch_299/state_dict.pkl" 2>/dev/null | grep -q .; then
    echo "=== $(date '+%F %T') rcaid / $dataset COMPLETE (epoch_299 checkpoint present) ==="
  else
    echo "=== $(date '+%F %T') rcaid / $dataset EXITED 0 BUT NO epoch_299 CHECKPOINT ==="
    return 1
  fi
}

mkdir -p "$ROOT/logs"
run_arm CLEARSCOPE_E3_V2_TARGET 0
run_arm CLEARSCOPE_E3_V2_SHADOW 1
echo "=== $(date '+%F %T') both rcaid arms done ==="
