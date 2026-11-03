#!/bin/bash
#SBATCH --job-name=osm_places
#SBATCH --account=ab036
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=/capstor/store/cscs/swissai/ab036/processed_datasets/natix/osm_bev/logs/%x_%j.log
#
# The two lookups of the route text (scripts/osm_maps/extract_places.py): cities / towns / villages and the ids of the
# roundabout ways, from every region extract. Written to $OSM_DATA, where make_osm_maps.py picks them up.
#
#   sbatch scripts/osm_maps/run_extract_places.sbatch
#   EXTRACTS=<dir> OSM_DATA=<dir> REGIONS="Japan,United Kingdom" sbatch --output=<log> scripts/osm_maps/run_extract_places.sbatch
set -u
REPO_DIR="${REPO_DIR:-/users/dbrggema/wm_trajectory_inference}"
EXTRACTS="${EXTRACTS:-/capstor/store/cscs/swissai/ab036/processed_datasets/natix/trajectories/_osm_extracts}"
OSM_DATA="${OSM_DATA:-/capstor/store/cscs/swissai/ab036/processed_datasets/natix/osm_bev/data}"
EDF_ENV="${EDF_ENV:-vggtslam_env_ext}"
EXTRA=()
[ -n "${REGIONS:-}" ] && { IFS=, read -ra RL <<< "$REGIONS"; EXTRA+=(--regions "${RL[@]}"); }
srun --environment="$EDF_ENV" python3 -u "$REPO_DIR/scripts/osm_maps/extract_places.py" --extracts "$EXTRACTS" --out-dir "$OSM_DATA" \
     --workers "${SLURM_CPUS_PER_TASK:-8}" "${EXTRA[@]}"
