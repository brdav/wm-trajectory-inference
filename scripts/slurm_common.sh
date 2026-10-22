# Common setup of the self-resubmitting pipeline jobs (scripts/slam/run_full_dataset.sbatch,
# scripts/metric_scale/run_metric_scale.sbatch). Sourced after the job script has set REPO_DIR, SELF (its own path
# in the checkout; sbatch runs a copy of the script, so `$0` is not it) and MAX_RESUBMITS (a backstop against
# silently burning allocation hours if the job crash-loops on the same clip). Everything can be overridden from
# the environment.

TRAJ_ROOT="${TRAJ_ROOT:-/capstor/store/cscs/swissai/ab036/processed_datasets/natix/trajectories}"
PIPELINE_DIR="$TRAJ_ROOT/_pipeline"
EDF_ENV="${EDF_ENV:-vggtslam_env_ext}"  # Pyxis environment (container image) the GPU steps run in; includes osmium for GPS fusion

# Torch hub cache with dino_salad.ckpt + the DINOv2 hub repo (the SLAM loop-closure model needs both; see the README,
# "Model weights"), and the HF cache for the depth models (downloaded on first use).
export TORCH_HOME="${TORCH_HOME:-/iopsstor/scratch/cscs/dbrggema/metric_scale/torch_home}"
export HF_HOME="${HF_HOME:-/iopsstor/scratch/cscs/dbrggema/hf_home}"

STOP_FILE="$PIPELINE_DIR/STOP"                       # touch it to stop the chain
RESUBMIT_COUNT_FILE="$PIPELINE_DIR/.resubmit_count"
mkdir -p "$PIPELINE_DIR/logs"

# Read once here, at script start, before the GPU workers start loading the shared filesystem: a read inside the
# SIGUSR1 trap once blocked the resubmit (memory/performance-investigations.md).
RESUBMIT_COUNT=$(cat "$RESUBMIT_COUNT_FILE" 2>/dev/null || echo 0)

# Used by both the SIGUSR1 (walltime-approaching) handler and the normal post-exit path, so STOP_FILE and
# MAX_RESUBMITS are honored whichever triggers the resubmit.
resubmit_or_stop() {
    local reason="$1"
    if [ -f "$STOP_FILE" ]; then
        echo "=== $STOP_FILE present, not resubmitting ($reason) ==="
        return
    fi
    RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1))
    if [ "$RESUBMIT_COUNT" -gt "$MAX_RESUBMITS" ]; then
        echo "=== hit MAX_RESUBMITS=$MAX_RESUBMITS (this is resubmission #$RESUBMIT_COUNT) --" \
             "stopping instead of continuing to retry ($reason); investigate before resetting" \
             "$RESUBMIT_COUNT_FILE ===" >&2
        return
    fi
    echo "=== resubmitting (#$RESUBMIT_COUNT) to continue processing remaining clips ($reason) ==="
    sbatch "$SELF"
    # Written AFTER sbatch, best effort: a slow write must never block the chain.
    echo "$RESUBMIT_COUNT" > "$RESUBMIT_COUNT_FILE" 2>/dev/null || true
}

trap 'echo "=== $(date -Iseconds) job $SLURM_JOB_ID: caught SIGUSR1, walltime approaching -- resubmitting before hard kill ==="; resubmit_or_stop "walltime signal"; exit 0' SIGUSR1
