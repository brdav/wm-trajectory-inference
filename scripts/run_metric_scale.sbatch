#!/bin/bash
#SBATCH --job-name=natix_metric_scale
#SBATCH --account=ab036
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --output=/capstor/store/cscs/swissai/ab036/dbrggema/natix_metric_scale/logs/%x_%j.log
#SBATCH --signal=B:SIGUSR1@300
#
# Post-processing after scripts/batch_infer.py (run_full_dataset.sbatch). Full order:
#   1. batch_infer.py                  -> result.npz per clip + manifest.jsonl     (run_full_dataset.sbatch)
#   2. clip_qc.py                      -> clip_qc.csv: per clip route use_fused / needs_scale_check / needs_metric_scale / static / ego_failed
#   3. estimate_metric_scale.py        -> depth-based scale for needs_metric_scale + needs_scale_check clips (GPU)
#   4. integrate_metric_scale.py       -> metric_scale/metric_poses into each clip's result.npz (session zips)
#   5. update_manifest_metric_scale.py -> qc + metric_scale entry on every manifest.jsonl row
# This script is steps 2-5 (step 1 is its own job). Submit once, after step 1 is done:
#
#   sbatch scripts/run_metric_scale.sbatch
#
# Step 2 is skipped when clip_qc.csv is newer than manifest.jsonl (override: RERUN_QC=1); steps 4-5
# only run once step 3 reports nothing left to process.
#
# Step 3 self-resubmits like run_full_dataset.sbatch (see its header for the SIGUSR1/walltime
# reasoning): estimate_metric_scale.py skips clips that already have a metric_scale.npz and
# clips that already failed (recorded in manifest_metric_scale.jsonl), so a resubmitted job
# continues where the last one stopped. It exits 0 only when nothing is left to process (or
# non-zero if all workers died), so any other exit triggers a resubmit.
#
# To test first on a handful of random clips, run the python line below by hand with
# `--limit 8 --shuffle --out_root <somewhere else>` inside an allocation.
#
# To stop the chain:  touch /capstor/store/cscs/swissai/ab036/dbrggema/natix_metric_scale/STOP
# To retry clips that failed before (e.g. after lowering MIN_FRAMES_USED): add --retry_failed
# to EXTRA_ARGS below (or run once by hand).
#
# --workers_per_gpu 3 (not batch_infer's 4): each worker holds the SLAM model (~16GB) plus the
# metric model, a point-cloud copy on the GPU and ~0.5-1GB of host RAM for the PCD read, and this
# hasn't been validated at 4 on a full node. Raise it only after a --limit run shows headroom in
# nvidia-smi and host RAM.

set -uo pipefail

OUT_ROOT=/capstor/store/cscs/swissai/ab036/dbrggema/natix_metric_scale
REPO_DIR=/users/dbrggema/wm_trajectory_inference
SELF="$REPO_DIR/scripts/run_metric_scale.sbatch"
STOP_FILE="$OUT_ROOT/STOP"
RESUBMIT_COUNT_FILE="$OUT_ROOT/.resubmit_count"
MAX_RESUBMITS=20
TRAJ_ROOT=/capstor/store/cscs/swissai/ab036/processed_datasets/natix/trajectories
QC_CSV="$OUT_ROOT/clip_qc.csv"
CPU_PY=/users/dbrggema/miniconda3/envs/wm_env/bin/python  # CPU-only steps (clip_qc, zip integration, manifest)
EXTRA_ARGS="${EXTRA_ARGS:-}"  # override: sbatch --export=ALL,EXTRA_ARGS=--retry_failed scripts/run_metric_scale.sbatch

# Torch hub cache with dino_salad.ckpt + the DINOv2 hub repo (the SLAM loop-closure model needs
# both; ~/.cache/torch/hub currently lacks them) and the HF cache holding DA3METRIC-LARGE (it is
# downloaded there on first use if missing and the node has internet).
export TORCH_HOME=/iopsstor/scratch/cscs/dbrggema/metric_scale/torch_home
export HF_HOME=/iopsstor/scratch/cscs/dbrggema/hf_home

mkdir -p "$OUT_ROOT/logs"

# Read once up front, while the filesystem is quiet (same reason as run_full_dataset.sbatch).
RESUBMIT_COUNT=$(cat "$RESUBMIT_COUNT_FILE" 2>/dev/null || echo 0)

resubmit_or_stop() {
    local reason="$1"
    if [ -f "$STOP_FILE" ]; then
        echo "=== $STOP_FILE present, not resubmitting ($reason) ==="
        return
    fi
    RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1))
    if [ "$RESUBMIT_COUNT" -gt "$MAX_RESUBMITS" ]; then
        echo "=== hit MAX_RESUBMITS=$MAX_RESUBMITS -- stopping instead of retrying ($reason); investigate before resetting $RESUBMIT_COUNT_FILE ===" >&2
        return
    fi
    echo "=== resubmitting (#$RESUBMIT_COUNT) to continue processing remaining clips ($reason) ==="
    sbatch "$SELF"
    echo "$RESUBMIT_COUNT" > "$RESUBMIT_COUNT_FILE" 2>/dev/null || true
}

trap 'echo "=== $(date -Iseconds) job $SLURM_JOB_ID: caught SIGUSR1, walltime approaching -- resubmitting before hard kill ==="; resubmit_or_stop "walltime signal"; exit 0' SIGUSR1

echo "=== $(date -Iseconds) job $SLURM_JOB_ID starting on $SLURMD_NODENAME ==="

# Step 2: per-clip QC, unless the CSV is already newer than the manifest it was built from
if [ -n "${RERUN_QC:-}" ] || [ ! -f "$QC_CSV" ] || [ "$TRAJ_ROOT/manifest.jsonl" -nt "$QC_CSV" ]; then
    echo "=== $(date -Iseconds) step 2: clip_qc.py"
    "$CPU_PY" "$REPO_DIR/scripts/clip_qc.py" --out "$QC_CSV" --workers 32 || { echo "clip_qc.py failed" >&2; exit 1; }
else
    echo "=== step 2 skipped: $QC_CSV is up to date"
fi

srun --environment=vggtslam_env python3 -u "$REPO_DIR/scripts/estimate_metric_scale.py" \
    --out_root "$OUT_ROOT" \
    --qc_csv "$QC_CSV" \
    --num_gpus 4 \
    --workers_per_gpu 3 \
    $EXTRA_ARGS &
SRUN_PID=$!
wait "$SRUN_PID"
EXIT_CODE=$?

echo "=== $(date -Iseconds) job $SLURM_JOB_ID: estimate_metric_scale.py exited $EXIT_CODE ==="

if [ "$EXIT_CODE" -eq 0 ]; then
    echo "=== nothing left to estimate, not resubmitting ==="
    echo "=== $(date -Iseconds) step 4: integrate_metric_scale.py"
    "$CPU_PY" "$REPO_DIR/scripts/integrate_metric_scale.py" --metric_root "$OUT_ROOT" --traj_root "$TRAJ_ROOT" --workers 24 \
        || { echo "integration had failures -- NOT updating the manifest" >&2; exit 1; }
    # The per-clip metric_scale.npz files are redundant once integrated (the scale is in the clip's result.npz and
    # the manifest) and cost ~5 inodes each. Archive them; deleting is left to a human on purpose.
    PACK="$OUT_ROOT/backups/standalone_metric_scale_$(date +%Y%m%dT%H%M%S)"
    ( cd "$OUT_ROOT" && find . -mindepth 1 -maxdepth 1 -type d ! -name backups ! -name logs ! -name qc_validation -printf '%P\n' > "$PACK.roots.txt" \
        && [ -s "$PACK.roots.txt" ] && tar czf "$PACK.tar.gz" -T "$PACK.roots.txt" && gzip -t "$PACK.tar.gz" ) \
        && echo "=== archived the per-clip score files to $PACK.tar.gz; once checked, free the inodes with: (cd $OUT_ROOT && xargs -d '\n' rm -rf -- < $PACK.roots.txt)" \
        || echo "=== WARNING: could not archive the per-clip score files (nothing removed)" >&2
    echo "=== $(date -Iseconds) step 5: update_manifest_metric_scale.py"
    "$CPU_PY" "$REPO_DIR/scripts/update_manifest_metric_scale.py" --traj_root "$TRAJ_ROOT" --qc_csv "$QC_CSV" \
        --metric_manifest "$OUT_ROOT/manifest_metric_scale.jsonl" --integration_log "$OUT_ROOT/integration_log.jsonl" --workers 24
    echo "=== $(date -Iseconds) all steps done"
    exit 0
fi

resubmit_or_stop "non-zero exit $EXIT_CODE"
