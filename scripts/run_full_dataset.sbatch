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
# Runs scripts/batch_infer.py over every source video listed in the dataset's sample manifest (parquet; see
# vggt_slam/sample_manifest.py -- the dataset tree is not walked), self-resubmitting as a new job every time the
# walltime above runs out, until everything is done; then packs the output into session zips
# (scripts/pack_sessions.py). Submit once:
#
#   sbatch scripts/run_full_dataset.sbatch
#
# On walltime timeout, SLURM kills every process in the job's cgroup -- including this script's
# own bash process, not just the srun child -- so code placed *after* srun would never run in
# that case without help. --signal=B:SIGUSR1@300 asks SLURM to deliver SIGUSR1 300s (5 min)
# before the hard TimeLimit kill; the trap below catches it, resubmits, and exits cleanly while
# there's still time. srun is backgrounded (`&` + `wait`) because bash only services traps
# promptly while not blocked on a synchronous foreground command. 300s (up from an original
# 120s) leaves a much bigger margin against exactly the failure mode that killed job 3198885's
# own resubmission (see resubmit_or_stop's own comment below) -- the 16 GPU workers don't stop
# running just because the wrapper script caught a signal, so anything in the trap's own path
# that touches the same contended filesystem needs real slack to still complete in time.
#
# Why self-resubmit at all: at the throughput measured for --workers_per_gpu 4 (~22 clips/min on
# this node, see conversation notes), the full ~120k-clip dataset takes on the order of 90 hours
# -- far longer than the "normal" partition's 12h limit -- so one job can only ever make partial
# progress. batch_infer.py is explicitly designed for this: it skips every manifest video that is already
# processed (a loose result.npz, an ok row in $OUT_ROOT/manifest.jsonl, or its result inside the session's zip --
# see discover_clips's docstring), so simply resubmitting picks up exactly where the last job left off. This script automates that: it
# resubmits itself unless batch_infer.py exited 0, which only happens when discover_clips found
# nothing left to process (i.e. the dataset is actually done) -- any other exit (walltime hit,
# a transient crash, ...) means there's still work left, so it retries rather than requiring
# someone to notice and resubmit by hand.
#
# --workers_per_gpu 4 was the highest setting validated safe on this node: it uses ~65GB/GPU
# (well under the 98GB/GPU cap) but sits close to the node's 450GB host RAM ceiling (16
# concurrent worker processes, each holding its own model copy in host RAM before the GPU
# transfer) -- --workers_per_gpu 5 outright killed the entire job with SIGKILL (a host-RAM OOM,
# not a graceful per-clip failure: GPU memory still had headroom when it happened). Don't raise
# this without re-validating host RAM headroom on a small --limit run first.
#
# 2026-08-24 efficiency investigation (live profiling on an idle 4-GPU allocation): confirmed
# nvidia-smi showed only ~41% average SM utilization at --workers_per_gpu 4, and confirmed via
# `nvidia-smi dmon` + load-average sampling that the root cause is NOT insufficient worker
# concurrency -- it's that every GPU worker process's PyTorch/MKL/OpenBLAS/OpenMP independently
# sizes its CPU thread pool off the node's full 288-core count, unaware it's one of many
# co-resident worker processes. batch_infer.py now caps each worker to
# max(1, cpu_count // (num_gpus*workers_per_gpu + 1)) threads (gpu_worker's docstring has the
# detail; override via --cpu_threads_per_worker if needed) -- this alone raised average SM
# utilization to ~75% at the SAME --workers_per_gpu 4, with no extra memory risk.
#
# Tried raising --workers_per_gpu to 6 instead (36 workers): it also reached ~74-75% SM
# utilization, but (a) transient host RAM during simultaneous model loading hit 81% of the 450GB
# cgroup cap (vs ~50% at 4), and (b) it triggered real, reproducible per-clip CUDA OOM crashes --
# 6 workers x ~15.7GB packs a 95GB GPU with ~0 headroom, so a single larger-than-average clip
# overflows it (caught gracefully per-clip, but wastes GPU/CPU time on the failed clip and
# depends on a later resubmit's resume-by-result.npz-skip to retry it). Do not raise
# --workers_per_gpu above 4 on this node.
#
# The unintuitive result: despite ~41%->~75% average SM utilization, measured throughput stayed
# flat at ~22-23 clips/min in both cases (with or without the thread-cap fix, at either worker
# count). GPU compute was never the binding constraint -- each clip's wall-clock is dominated by
# a CPU-bound, largely serial critical path (per-submap gtsam graph.optimize() batch
# re-optimization, GPS/OSM road-matching, keyframe selection), and no amount of extra GPU-side
# concurrency shortens that. The thread-cap fix is still worth keeping (it's free, and it removes
# scheduling-contention risk that gets much worse at higher worker/clip-resolution combinations --
# see the 232s outlier clip time observed pre-fix at --workers_per_gpu 6), but a real throughput
# win over the current ~22-23 clips/min would require either reducing that per-clip CPU-bound work
# (e.g. incremental instead of full-batch graph optimization) or scaling out to more nodes in
# parallel (safe today: every node/job independently skips whatever already has a result.npz).
#
# To stop the chain (e.g. to investigate a repeated crash instead of letting it keep retrying):
#   touch /capstor/store/cscs/swissai/ab036/processed_datasets/natix/trajectories/_pipeline/STOP
# A currently-running job finishes its current 12h slice either way; it just won't resubmit.
#
# To process more data: add its source videos to the sample manifest (same columns) and submit this script
# again. Videos already processed are skipped, new clips of an already-packed session are processed and
# merged into that session's zip by the packing step. Afterwards run scripts/run_metric_scale.sbatch
# (QC, depth-based scale for clips without a trusted GPS scale, manifest fields).
#
# Everything is written under $OUT_ROOT (the trajectories folder): per-clip results (session zips after the
# packing step), manifest.jsonl, and job logs / STOP file / resubmit counter in $OUT_ROOT/_pipeline/.
# OUT_ROOT, DATASET_ROOT and SAMPLE_MANIFEST can be overridden from the environment, e.g. to test on a few
# clips: sbatch --export=ALL,OUT_ROOT=$SCRATCH/test_out,SAMPLE_MANIFEST=$SCRATCH/small.parquet scripts/run_full_dataset.sbatch

set -uo pipefail

OUT_ROOT="${OUT_ROOT:-/capstor/store/cscs/swissai/ab036/processed_datasets/natix/trajectories}"
DATASET_ROOT="${DATASET_ROOT:-/capstor/store/cscs/swissai/ab036/datasets/natix}"
SAMPLE_MANIFEST="${SAMPLE_MANIFEST:-/capstor/store/cscs/swissai/ab036/processed_datasets/natix/manifest/samples_24fps.parquet}"
PIPELINE_DIR="$OUT_ROOT/_pipeline"

# The SLAM loop-closure model (SALAD) needs dino_salad.ckpt and the DINOv2 torch.hub repo in a torch hub cache
# (README, "Model weights"); the depth model weights are downloaded into HF_HOME on first use.
export TORCH_HOME="${TORCH_HOME:-/iopsstor/scratch/cscs/dbrggema/metric_scale/torch_home}"
export HF_HOME="${HF_HOME:-/iopsstor/scratch/cscs/dbrggema/hf_home}"
REPO_DIR=/users/dbrggema/wm_trajectory_inference
STOP_FILE="$PIPELINE_DIR/STOP"
RESUBMIT_COUNT_FILE="$PIPELINE_DIR/.resubmit_count"
MAX_RESUBMITS=40  # generous headroom over the ~8 links a 90h run at 12h/job would need -- a
                   # backstop against silently burning allocation hours if something is stuck
                   # crash-looping on the same clip (see batch_infer.py's per-clip try/except:
                   # a crash that kills the whole process, e.g. an OOM, doesn't mark that clip as
                   # failed, so a resubmitted job retries the exact same clip from scratch).

mkdir -p "$PIPELINE_DIR/logs"

# Read once here, at script start (before batch_infer.py's 16 GPU workers start hammering the
# shared Lustre filesystem with their own result.npz/osm_network writes), NOT inside
# resubmit_or_stop as it used to be. 2026-08-27: job 3198885 caught its SIGUSR1 right on schedule
# (confirmed in its own log) but the chain still silently died there -- sacct shows no follow-up
# job was EVER submitted. The workers kept running full-tilt straight through the trap firing
# (confirmed: later clips' own build_network log lines appear interleaved after the SIGUSR1
# message), so this tiny counter file's read+write almost certainly got queued behind that same
# filesystem contention and never returned before the hard TimeLimit kill landed -- meaning
# `sbatch "$0"` right after it was never even reached. Reading once up front (when the
# filesystem is quiet) and moving the write to run AFTER sbatch (see below) removes both
# blocking filesystem ops from ahead of the one call that actually matters.
RESUBMIT_COUNT=$(cat "$RESUBMIT_COUNT_FILE" 2>/dev/null || echo 0)

# Shared by both the SIGUSR1 (walltime-approaching) handler and the normal post-exit path below,
# so STOP_FILE / MAX_RESUBMITS are honored no matter which path triggers the resubmit.
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
    sbatch "$0"
    # Written AFTER sbatch, best-effort (a failed/slow write here no longer blocks the chain --
    # worst case MAX_RESUBMITS undercounts by one link, far better than the chain silently dying
    # because this write never got a chance to run at all).
    echo "$RESUBMIT_COUNT" > "$RESUBMIT_COUNT_FILE" 2>/dev/null || true
}

trap 'echo "=== $(date -Iseconds) job $SLURM_JOB_ID: caught SIGUSR1, walltime approaching -- resubmitting before hard kill ==="; resubmit_or_stop "walltime signal"; exit 0' SIGUSR1

echo "=== $(date -Iseconds) job $SLURM_JOB_ID starting on $SLURMD_NODENAME ==="

srun --environment=vggtslam_env python3 -u "$REPO_DIR/scripts/batch_infer.py" \
    --dataset_root "$DATASET_ROOT" \
    --out_root "$OUT_ROOT" \
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
    srun --environment=vggtslam_env python3 -u "$REPO_DIR/scripts/pack_sessions.py" \
        --dataset_root "$DATASET_ROOT" \
        --out_root "$OUT_ROOT" \
        --sample_manifest "$SAMPLE_MANIFEST" \
        --min_age_days 0 || { echo "pack_sessions.py failed -- output is intact but not packed; rerun it by hand" >&2; exit 1; }
    echo "=== $(date -Iseconds) done. Next: sbatch scripts/run_metric_scale.sbatch"
    exit 0
fi

resubmit_or_stop "non-zero exit $EXIT_CODE"
exit "$EXIT_CODE"
