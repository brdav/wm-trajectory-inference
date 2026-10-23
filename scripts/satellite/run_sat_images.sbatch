#!/bin/bash
#SBATCH --job-name=sat_images
#SBATCH --account=ab036
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --output=/capstor/store/cscs/swissai/ab036/processed_datasets/natix/satellite/logs/%x_%j.log
#
# Render the aerial image of every sample that has an OSM map (scripts/satellite/make_sat_images.py) from the tile cache of
# run_fetch_tiles.sbatch. Run it after the tile download has finished. One stored zip per country folder and month in OUT, plus
# summary.jsonl. Resumable: finished zips are skipped, so after a timeout or crash just submit the SAME command again. For several
# nodes use --array=0-(N-1) together with N_SHARDS=N.
#
#   sbatch scripts/satellite/run_sat_images.sbatch
#   sbatch --partition=preemptable scripts/satellite/run_sat_images.sbatch   # shorter queue; --requeue restarts it if preempted (safe: finished zips are skipped)
#   COUNTRIES="Switzerland" LIMIT=50 sbatch --partition=debug --time=00:30:00 scripts/satellite/run_sat_images.sbatch   # small test first
set -u
REPO_DIR="${REPO_DIR:-/users/dbrggema/wm_trajectory_inference}"
SAT_ROOT="${SAT_ROOT:-/capstor/store/cscs/swissai/ab036/processed_datasets/natix/satellite}"
MAPS="${MAPS:-/capstor/store/cscs/swissai/ab036/processed_datasets/natix/osm_bev/maps_predicted}"
CACHE="${CACHE:-$SAT_ROOT/tiles}"
OUT="${OUT:-$SAT_ROOT/images}"
N_SHARDS="${N_SHARDS:-1}"
FORMAT="${FORMAT:-webp}"
QUALITY="${QUALITY:-90}"
MAX_MISSING="${MAX_MISSING:-0.2}"   # a crop with more than this uncovered fraction gets no image
WORKERS="${WORKERS:-60}"
EDF_ENV="${EDF_ENV:-vggtslam_env_ext}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
EXTRA=()
[ -n "${COUNTRIES:-}" ] && { IFS=, read -ra CL <<< "$COUNTRIES"; EXTRA+=(--countries "${CL[@]}"); }   # comma-separated folders
[ -n "${LIMIT:-}" ] && EXTRA+=(--limit "$LIMIT")
[ -n "${REDO_REGIONS:-}" ] && { IFS=, read -ra RL <<< "$REDO_REGIONS"; EXTRA+=(--redo-regions "${RL[@]}"); }   # e.g. REDO_REGIONS="Italy,Croatia"

CMD=(python3 -u "$REPO_DIR/scripts/satellite/make_sat_images.py" --maps "$MAPS" --cache "$CACHE" --out "$OUT" --format "$FORMAT"
     --quality "$QUALITY" --max-missing "$MAX_MISSING" --workers "$WORKERS" --shard "$SHARD/$N_SHARDS" "${EXTRA[@]}")
if [ -n "${DRY_RUN:-}" ]; then echo srun --environment="$EDF_ENV" "${CMD[@]}"; exit 0; fi
mkdir -p "$OUT"
srun --environment="$EDF_ENV" "${CMD[@]}"
