#!/bin/bash
#SBATCH --job-name=osm_prepare
#SBATCH --account=ab036
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/capstor/store/cscs/swissai/ab036/processed_datasets/natix/osm_bev/logs/%x_%j.log
#
# Step 1 of 2 for the OSM map images: prepare (PBF -> tile parquet files) EVERY region extract of the dataset, plus the
# region boundary polygons that map a GPS position to its extract. One CPU process per region, several at once; the
# biggest regions start first. Safe to re-run: a region whose directory already has _info.json is skipped.
#
#   sbatch scripts/osm_maps/run_osm_prepare.sbatch                      # all regions
#   REGIONS="Switzerland,Japan,United States_Nevada" sbatch   # comma-separated, names = extract file stems scripts/osm_maps/run_osm_prepare.sbatch
#   DRY_RUN=1 bash scripts/osm_maps/run_osm_prepare.sbatch              # just print what would run
#
# Measured (first full run): per region 5-106 min (Canada 26, Germany ~60-100, Japan ~1 h), peak memory 18.9 GB; small regions 5-10 min.
# If a region is OOM-killed raise MEM_PER_REGION (and lower PARALLEL accordingly: PARALLEL x MEM_PER_REGION must stay below 450G).
#
# SRUN_EXTRA="--overlap" lets the script run inside an existing allocation (testing); not needed under sbatch.
# Output (override with the env vars below):
#   $OSM_DATA/<region>/tile_<tx>_<ty>.parquet, $OSM_DATA/<region>/_info.json   (region "United States/Nevada" -> United States_Nevada)
#   $OSM_DATA/_regions.json
set -u
REPO_DIR="${REPO_DIR:-/users/dbrggema/wm_trajectory_inference}"
EXTRACTS="${EXTRACTS:-/capstor/store/cscs/swissai/ab036/processed_datasets/natix/trajectories/_osm_extracts}"
OSM_DATA="${OSM_DATA:-/capstor/store/cscs/swissai/ab036/processed_datasets/natix/osm_bev/data}"
PARALLEL="${PARALLEL:-10}"
MEM_PER_REGION="${MEM_PER_REGION:-40G}"      # REQUIRED: a step without --mem reserves the whole node memory, so only ONE step runs at a time
                                             # (that is what happened in the first full run: 12 h, concurrency 1). Measured peak RSS of one
                                             # region: 18.9 GB (of the 53 regions that finished); 10 x 40 GB fits in the 450 GB of a node.
EDF_ENV="${EDF_ENV:-vggtslam_env_ext}"
RUN="srun ${SRUN_EXTRA:-} --environment=$EDF_ENV"                      # a string, not an array: arrays cannot be exported to xargs' bash
[ -n "${DRY_RUN:-}" ] && RUN="echo srun ${SRUN_EXTRA:-} --environment=$EDF_ENV"
[ -z "${DRY_RUN:-}" ] && mkdir -p "$OSM_DATA"

# regions: file stems of the extracts ("United States_Nevada.osm.pbf" -> key "United States/Nevada"), biggest file first
if [ -n "${REGIONS:-}" ]; then
  LIST=$(echo "$REGIONS" | tr ',' '\n')
else
  LIST=$(ls -S "$EXTRACTS"/*.osm.pbf | xargs -n1 -d '\n' basename | sed 's/\.osm\.pbf$//')
fi

$RUN python3 -u "$REPO_DIR/scripts/osm_maps/osm_bev.py" regions --root "$OSM_DATA"

prep_one() {
  local name="$1" key dir
  key="${name/United States_/United States/}"; dir="${name//\//_}"
  if [ -f "$OSM_DATA/$dir/_info.json" ]; then echo "skip $name (done)"; return 0; fi
  echo "start $name"
  local log="$OSM_DATA/_prepare_$dir.log"; [ -n "${DRY_RUN:-}" ] && log=/dev/null
  $RUN -n1 -c1 --mem="$MEM_PER_REGION" --exact python3 -u "$REPO_DIR/scripts/osm_maps/osm_bev.py" prepare --country "$key" \
      --pbf "$EXTRACTS/$name.osm.pbf" --root "$OSM_DATA" > "$log" 2>&1 \
    && echo "ok $name" || echo "FAILED $name (see $OSM_DATA/_prepare_$dir.log)"
}
export -f prep_one; export RUN OSM_DATA EXTRACTS REPO_DIR DRY_RUN MEM_PER_REGION
echo "$LIST" | xargs -d '\n' -P "$PARALLEL" -I{} bash -c 'prep_one "$1"' _ {}
