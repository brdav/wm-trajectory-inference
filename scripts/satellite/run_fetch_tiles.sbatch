#!/bin/bash
#SBATCH --job-name=sat_tiles
#SBATCH --account=ab036
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --output=/capstor/store/cscs/swissai/ab036/processed_datasets/natix/satellite/logs/%x_%j.log
#
# Download the aerial tiles the satellite crops need (see scripts/satellite/fetch_tiles.py, sources.py, LICENSES.md) into the tile
# cache: every region that has a source, one fetch_tiles.py process per region (the US states are split over US_GROUPS processes),
# PAR processes at a time. Each process rate-limits its own requests (RATE per second); the US (NAIP COGs) and Poland (bulk sheets)
# are not request-limited. Network-bound, little CPU. Resumable: tiles already in the cache are skipped, so after a timeout or
# crash just submit the SAME command again.
#
#   sbatch scripts/satellite/run_fetch_tiles.sbatch
#   REGIONS="Switzerland,Austria,United States_Nevada" LIMIT_SAMPLES=20 sbatch --partition=debug --time=00:30:00 \
#          scripts/satellite/run_fetch_tiles.sbatch                # small test first (comma-separated regions, random subset of samples)
#   DRY_RUN=1 bash scripts/satellite/run_fetch_tiles.sbatch      # print the process list (in the wm_env conda env on the login node)
#
# Expect several hours: ~1.4M tiles at <= 10 requests/s per process, plus ~25 GB of Polish GeoTIFF sheets (kept in <CACHE>/_sheets).
set -u
REPO_DIR="${REPO_DIR:-/users/dbrggema/wm_trajectory_inference}"
SAT_ROOT="${SAT_ROOT:-/capstor/store/cscs/swissai/ab036/processed_datasets/natix/satellite}"
MAPS="${MAPS:-/capstor/store/cscs/swissai/ab036/processed_datasets/natix/osm_bev/maps_predicted}"   # sample index: where the crops are
OSM_DATA="${OSM_DATA:-/capstor/store/cscs/swissai/ab036/processed_datasets/natix/osm_bev/data}"       # _regions.json: the region list
CACHE="${CACHE:-$SAT_ROOT/tiles}"
PAR="${PAR:-8}"                  # processes at a time
RATE="${RATE:-10}"               # requests per second, per process (tile/WMS sources)
WORKERS="${WORKERS:-16}"         # threads per process
US_WORKERS="${US_WORKERS:-24}"
US_GROUPS="${US_GROUPS:-3}"
EDF_ENV="${EDF_ENV:-vggtslam_env_ext}"
# rasterio (NAIP, Niedersachsen, Poland) is not in the image yet (it is in requirements.txt): use the --target install until the image is rebuilt
export PYTHONPATH="${PYLIBS:-/capstor/store/cscs/swissai/ab036/dbrggema/pylibs}${PYTHONPATH:+:$PYTHONPATH}"
EXTRA=()
[ -n "${LIMIT_SAMPLES:-}" ] && EXTRA+=(--limit-samples "$LIMIT_SAMPLES")

LOGS="$SAT_ROOT/logs/fetch_${SLURM_JOB_ID:-local}"
LIST=(python3 "$REPO_DIR/scripts/satellite/list_fetch_jobs.py" --osm-data "$OSM_DATA" --us-groups "$US_GROUPS")
if [ -n "${REGIONS:-}" ]; then IFS=, read -ra RLIST <<< "$REGIONS"; LIST+=(--regions "${RLIST[@]}"); fi
if [ -n "${DRY_RUN:-}" ]; then "${LIST[@]}"; exit 0; fi    # needs numpy/shapely: run it in the env (conda wm_env) on the login node
mkdir -p "$CACHE" "$LOGS"
srun --environment="$EDF_ENV" "${LIST[@]}" > "$LOGS/jobs.txt"
cat "$LOGS/jobs.txt"
export REPO_DIR MAPS CACHE RATE WORKERS US_WORKERS LOGS
export EXTRA_ARGS="${EXTRA[*]:-}"

run_one() {   # $1 = name, $2 = regions joined by | (spaces written as ~)
    local name="$1" regs w="$WORKERS"
    IFS='|' read -ra regs <<< "${2//\~/ }"
    [[ "$name" == US* ]] && w="$US_WORKERS"
    echo "$(date -Iseconds) start $name (${#regs[@]} region(s))"
    python3 -u "$REPO_DIR/scripts/satellite/fetch_tiles.py" --maps "$MAPS" --cache "$CACHE" --regions "${regs[@]}" \
        --workers "$w" --rate "$RATE" $EXTRA_ARGS > "$LOGS/$name.log" 2>&1
    local rc=$?
    local failed; failed=$(grep -c "FAILED block" "$LOGS/$name.log")
    echo "$(date -Iseconds) done $name (exit $rc, $failed failed blocks: rerun this job to redo them)"
}
export -f run_one

# PAR processes at a time; PAR is passed through the environment (srun --environment keeps the exported variables)
export PAR
srun --environment="$EDF_ENV" bash -c 'xargs -a "$LOGS/jobs.txt" -P "$PAR" -L 1 bash -c '"'"'run_one "$0" "$1"'"'"''
