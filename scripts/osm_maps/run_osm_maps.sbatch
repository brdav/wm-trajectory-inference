#!/bin/bash
#SBATCH --job-name=osm_maps
#SBATCH --account=ab036
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --array=0-3
#SBATCH --output=/capstor/store/cscs/swissai/ab036/processed_datasets/natix/osm_bev/logs/%x_%A_%a.log
#
# Step 2 of 2: write the OSM map images (with the ego's next 5 s) for every start frame of the sample manifest, for every
# clip with a fused trajectory. The sessions are split over the array tasks (--shard i/N, N = the array size below): each task
# writes its own sessions' zips and its own index / unplottable / done_sessions files, so tasks never share a file.
# Run scripts/osm_maps/run_osm_prepare.sbatch first (needs $OSM_DATA/<region>/ and _regions.json).
#
#   sbatch scripts/osm_maps/run_osm_maps.sbatch                    # 4 tasks; use --array=0-(N-1) and N_SHARDS=N together
#   N_SHARDS=8 sbatch --array=0-7 scripts/osm_maps/run_osm_maps.sbatch
#   COUNTRIES="Switzerland" LIMIT_CLIPS=200 sbatch ...    # small test first
#   PATH_KIND=map_matched sbatch scripts/osm_maps/run_osm_maps.sbatch   # same samples with the map-matched path -> maps_map_matched/
#   The route text is written next to the images (OUT=.../maps_predicted -> .../text_predicted; override with TEXT_OUT).
#   TEXT_ONLY=1 OUT=<dir>/text_predicted sbatch ...          # route text only, for a dataset whose images exist already
#   For another dataset set DATASET_ROOT, TRAJ_ROOT, SAMPLE_MANIFEST, OSM_DATA and OUT (and sbatch --output).
#
# Resumable: a session listed in done_sessions.shard<i>of<N>.txt is skipped, so after a timeout or crash just submit the SAME
# command again (same N_SHARDS). Measured only on a few hundred Swiss clips: ~80 images/s on 64 workers, i.e. roughly an
# hour of one node for all ~1.2M samples if that held -- check the first task's log before launching the rest.
#
# Storage per image: ~250 KB png, ~140 KB lossless webp (identical pixels), choose with FORMAT: ~300 GB / ~170 GB for 1.2M.
set -u
REPO_DIR="${REPO_DIR:-/users/dbrggema/wm_trajectory_inference}"
PATH_KIND="${PATH_KIND:-predicted}"        # predicted | map_matched | none : which ego path is drawn (output goes to maps_<PATH_KIND>)
OSM_DATA="${OSM_DATA:-/capstor/store/cscs/swissai/ab036/processed_datasets/natix/osm_bev/data}"
OUT="${OUT:-/capstor/store/cscs/swissai/ab036/processed_datasets/natix/osm_bev/maps_${PATH_KIND:-predicted}}"   # folder named after the drawn path
N_SHARDS="${N_SHARDS:-4}"
FORMAT="${FORMAT:-webp}"
WORKERS="${WORKERS:-256}"
EDF_ENV="${EDF_ENV:-vggtslam_env_ext}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
EXTRA=()
[ -n "${COUNTRIES:-}" ] && EXTRA+=(--countries $COUNTRIES)
[ -n "${LIMIT_CLIPS:-}" ] && EXTRA+=(--limit-clips "$LIMIT_CLIPS")
[ -n "${TEXT_OUT:-}" ] && EXTRA+=(--text-out "$TEXT_OUT")
[ -n "${TEXT_ONLY:-}" ] && EXTRA+=(--text-only)     # only the route text, no images (OUT then needs its own folder, e.g. text_predicted)

CMD=(python3 -u "$REPO_DIR/scripts/osm_maps/make_osm_maps.py" --osm-data "$OSM_DATA" --out "$OUT" --path "$PATH_KIND"
     --format "$FORMAT" --workers "$WORKERS" --shard "$SHARD/$N_SHARDS" "${EXTRA[@]}")
if [ -n "${DRY_RUN:-}" ]; then echo srun ${SRUN_EXTRA:-} --environment="$EDF_ENV" "${CMD[@]}"; exit 0; fi
mkdir -p "$OUT"
srun ${SRUN_EXTRA:-} --environment="$EDF_ENV" "${CMD[@]}"
