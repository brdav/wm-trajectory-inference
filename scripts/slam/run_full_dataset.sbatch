#!/bin/bash
#SBATCH --job-name=vggt_slam_natix
#SBATCH --account=ab036
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --output=/capstor/store/cscs/swissai/ab036/processed_datasets/natix/trajectories/_pipeline/logs/%x_%j.log
#SBATCH --signal=B:SIGUSR1@300
#
# Runs scripts/slam/batch_infer.py over every source video listed in the dataset's sample manifest (parquet; see
# vggt_slam/sample_manifest.py -- the dataset tree is not walked), self-resubmitting as a new job every time the
# walltime above runs out, until everything is done; then packs the output into session zips
# (scripts/slam/pack_sessions.py). Submit once:
#
#   sbatch scripts/slam/run_full_dataset.sbatch
#
# On walltime timeout, SLURM kills every process in the job's cgroup -- including this script's
# own bash process, not just the srun child -- so code placed *after* srun would never run in
# that case without help. --signal=B:SIGUSR1@300 asks SLURM to deliver SIGUSR1 300s (5 min)
# before the hard TimeLimit kill; the trap below catches it, resubmits, and exits cleanly while
# there's still time. srun is backgrounded (`&` + `wait`) because bash only services traps
# promptly while not blocked on a synchronous foreground command. 300s of warning (not 120s) gives the trap
# slack on a contended filesystem (memory/performance-investigations.md).
#
# Why self-resubmit at all: the full ~120k-clip dataset takes about 90 hours (memory/performance-investigations.md),
# far longer than the "normal" partition's 12h limit, so one job can only make partial progress.
# batch_infer.py is explicitly designed for this: it skips every manifest video that is already
# processed (a loose result.npz, an ok row in $TRAJ_ROOT/manifest.jsonl, or its result inside the session's zip --
# see clips_to_process's docstring), so simply resubmitting picks up exactly where the last job left off. This script automates that: it
# resubmits itself unless batch_infer.py exited 0, which only happens when clips_to_process found
# nothing left to process (i.e. the dataset is actually done) -- any other exit (walltime hit,
# a transient crash, ...) means there's still work left, so it retries rather than requiring
# someone to notice and resubmit by hand.
#
# --workers_per_gpu 4 is the highest setting validated safe on this node (host RAM and per-GPU memory limits;
# throughput is bound by CPU work per clip, not GPU concurrency). Don't raise it without re-validating on a small
# --limit run. Evidence and the thread-cap investigation: memory/performance-investigations.md.
#
# To stop the chain (e.g. to investigate a repeated crash instead of letting it keep retrying):
#   touch /capstor/store/cscs/swissai/ab036/processed_datasets/natix/trajectories/_pipeline/STOP
# A currently-running job finishes its current 12h slice either way; it just won't resubmit.
#
# To process more data: add its source videos to the sample manifest (same columns) and submit this script
# again. Videos already processed are skipped, new clips of an already-packed session are processed and
# merged into that session's zip by the packing step. Afterwards run scripts/metric_scale/run_metric_scale.sbatch
# (QC, depth-based scale for clips without a trusted GPS scale, manifest fields).
#
# Everything is written under $TRAJ_ROOT (the trajectories folder): per-clip results (session zips after the
# packing step), manifest.jsonl, and job logs / STOP file / resubmit counter in $TRAJ_ROOT/_pipeline/.
# TRAJ_ROOT, DATASET_ROOT and SAMPLE_MANIFEST can be overridden from the environment, e.g. to test on a few
# clips: sbatch --export=ALL,TRAJ_ROOT=$SCRATCH/test_out,SAMPLE_MANIFEST=$SCRATCH/small.parquet scripts/slam/run_full_dataset.sbatch

set -uo pipefail

REPO_DIR="${REPO_DIR:-/users/dbrggema/wm_trajectory_inference}"  # sbatch runs a copy of this script, so it cannot locate the checkout itself
SELF="$REPO_DIR/scripts/slam/run_full_dataset.sbatch"
MAX_RESUBMITS=40  # generous headroom over the ~8 links a 90h run at 12h/job would need
source "$REPO_DIR/scripts/slurm_common.sh"  # TRAJ_ROOT, PIPELINE_DIR, EDF_ENV, TORCH_HOME, HF_HOME, resubmit_or_stop, SIGUSR1 trap
DATASET_ROOT="${DATASET_ROOT:-/capstor/store/cscs/swissai/ab036/datasets/natix}"
SAMPLE_MANIFEST="${SAMPLE_MANIFEST:-/capstor/store/cscs/swissai/ab036/processed_datasets/natix/manifest/samples_24fps.parquet}"

echo "=== $(date -Iseconds) job $SLURM_JOB_ID starting on $SLURMD_NODENAME ==="

srun --environment="$EDF_ENV" python3 -u "$REPO_DIR/scripts/slam/batch_infer.py" \
    --dataset_root "$DATASET_ROOT" \
    --out_root "$TRAJ_ROOT" \
    --sample_manifest "$SAMPLE_MANIFEST" \
    --fps 10 \
    --num_gpus 4 \
    --workers_per_gpu 4 &
SRUN_PID=$!
wait "$SRUN_PID"
EXIT_CODE=$?

echo "=== $(date -Iseconds) job $SLURM_JOB_ID: batch_infer.py exited $EXIT_CODE ==="

if [ "$EXIT_CODE" -eq 0 ]; then
    echo "=== everything in the sample manifest is processed, not resubmitting ==="
    # Archive finished sessions into one zip each (inode count; this is the layout every later stage reads).
    # --min_age_days 0: this job's batch_infer.py has just finished, nothing is still writing. Sessions that
    # already have a zip get the new clips merged in; sessions with a clip still missing stay loose.
    echo "=== $(date -Iseconds) packing sessions into zips"
    srun --environment="$EDF_ENV" python3 -u "$REPO_DIR/scripts/slam/pack_sessions.py" \
        --dataset_root "$DATASET_ROOT" \
        --out_root "$TRAJ_ROOT" \
        --sample_manifest "$SAMPLE_MANIFEST" \
        --min_age_days 0 || { echo "pack_sessions.py failed -- output is intact but not packed; rerun it by hand" >&2; exit 1; }
    echo "=== $(date -Iseconds) done. Next: sbatch scripts/metric_scale/run_metric_scale.sbatch"
    exit 0
fi

resubmit_or_stop "non-zero exit $EXIT_CODE"
exit "$EXIT_CODE"
