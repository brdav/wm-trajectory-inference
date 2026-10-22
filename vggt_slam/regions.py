"""Which pre-downloaded OSM region extract(s) to use for a clip's GPS fusion.

Clips are sometimes filed under the wrong country/state folder, so the region is chosen by where the
clip's GPS fixes are (distance to the centroid tables in country_regions.py / us_regions.py), falling
back to the folder-labeled region. Used by scripts/batch_infer.py and scripts/tools/refuse_gps.py.
"""
import csv
import math
from pathlib import Path

from vggt_slam import country_regions, us_regions

# Clips filed under the wrong country folder (e.g. an "Austria" recording near the Croatian/Bosnian border whose
# own crop finds no roads in Austria's extract): dataset-folder-name -> ordered fallback region keys to try (same
# keying as folder_region / download_osm_extracts.py). Add entries as mislabeled clips are found.
FALLBACK_REGIONS = {
    "Austria": ["Croatia", "Bosnia and Herzegovina"],
}


def folder_region(rel_path):
    country = rel_path.parts[0]
    return f"{country}_{rel_path.parts[1]}" if country == "United States" else country


def region_pbf_path(rel_path, osm_extracts_dir, region_name=None):
    """rel_path: a clip's video path relative to dataset_root. Returns the pre-downloaded
    region extract's path (see scripts/download_osm_extracts.py's naming -- "United States"
    clips are keyed by "United States_<State>", every other country by its own name directly)
    if osm_extracts_dir is set AND that region's extract actually exists there, else None (GPS
    fusion is required to have a region extract -- see main.py's --osm_region_pbf_path -- so a
    clip whose region isn't covered here just has its GPS fusion skipped, non-fatally).

    region_name: override the region key resolved from rel_path -- used to look up a fallback
    region (see FALLBACK_REGIONS) instead of the clip's own labeled region.
    """
    if not osm_extracts_dir:
        return None
    region_name = region_name or folder_region(rel_path)
    candidate = Path(osm_extracts_dir) / f"{region_name}.osm.pbf"
    return candidate if candidate.exists() else None


def fallback_region_pbf_paths(rel_path, osm_extracts_dir):
    """Returns the ordered list of candidate fallback extract paths that actually exist on disk
    (see FALLBACK_REGIONS) -- align_trajectory_to_roads tries each in turn until one has roads
    near the clip's GPS fixes. Empty list if this region has no configured fallback."""
    fallback_regions = FALLBACK_REGIONS.get(folder_region(rel_path), [])
    paths = (region_pbf_path(rel_path, osm_extracts_dir, region_name=r) for r in fallback_regions)
    return [p for p in paths if p is not None]


def read_latlon_centroid(csv_path):
    """Cheap best-effort mean (lat, lon) of a natix FRONT_*.csv's real (non-'na') GPS fixes, or
    None if the file is missing/unreadable or has no real fixes. Deliberately NOT
    vggt_slam.gps.load_gps_fixes -- that also does frame-number/fps alignment and a UTM
    projection that this doesn't need, and discover_clips calls this for every not-yet-processed
    US clip on every run, so keeping it a minimal single-pass CSV scan matters.
    """
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            lats, lons = [], []
            for row in reader:
                if row.get("GPS_latitude_deg", "na") == "na":
                    continue
                lats.append(float(row["GPS_latitude_deg"]))
                lons.append(float(row["GPS_longitude_deg"]))
    except (OSError, KeyError, ValueError):
        return None
    if not lats:
        return None
    return sum(lats) / len(lats), sum(lons) / len(lons)


def all_region_candidates(video_path, osm_extracts_dir):
    """GPS-centroid-based region resolution across BOTH US states and every other downloaded
    country, searched TOGETHER as one combined, distance-ranked list -- regardless of which
    folder (US state or country) the clip happens to be filed under. Some clips are filed under
    the wrong region entirely, including across the US/country boundary itself (confirmed on real
    clips: several "United States"-labeled clips actually in Puerto Rico -- which lives in
    vggt_slam/country_regions.py's table since its extract is named like a country ("Puerto
    Rico.osm.pbf"), not a state, so a search restricted to US states alone could never reach it;
    also a "Canada"-labeled clip sitting right on the US border). Searching one merged space
    instead of two separate ones (US-only vs. country-only) means neither direction of that
    mislabeling needs its own special case.

    Returns (primary_path, fallback_paths) ordered by distance from the clip's own GPS centroid,
    restricted to regions that actually have a downloaded extract; (None, []) if the clip's GPS
    centroid can't be determined (missing/empty/all-'na' CSV) -- caller falls back to folder-based
    resolution in that case.
    """
    centroid = read_latlon_centroid(video_path.with_suffix(".csv"))
    if centroid is None:
        return None, []
    lat, lon = centroid
    coslat = math.cos(math.radians(lat))

    def dist2(clat, clon):
        dlat, dlon = lat - clat, (lon - clon) * coslat
        return dlat * dlat + dlon * dlon

    entries = [(dist2(clat, clon), f"United States_{state}.osm.pbf")
               for state, (clat, clon) in us_regions.US_STATE_CENTROIDS.items()]
    entries += [(dist2(clat, clon), f"{country}.osm.pbf")
                for country, (clat, clon) in country_regions.COUNTRY_CENTROIDS.items()]
    entries.sort(key=lambda e: e[0])

    candidates = []
    for _, name in entries:
        p = Path(osm_extracts_dir) / name
        if p.exists():
            candidates.append(p)
    if not candidates:
        return None, []
    return candidates[0], candidates[1:]


def region_candidates(rel_path, video_path, osm_extracts_dir):
    """Unified region-selection entry point used by both discover_clips (batch_infer.py) and
    refuse_gps.py's --resume re-fusion path. Returns (primary_path_or_None, fallback_paths).
    """
    if not osm_extracts_dir:
        return None, []
    primary, fallback = all_region_candidates(video_path, osm_extracts_dir)
    if primary is not None:
        return primary, fallback
    # No usable GPS centroid (e.g. all-'na' CSV) -- degrade to the folder-labeled region rather
    # than skipping region resolution entirely; main.py/refuse_gps.py's own "No real GPS fixes"
    # handling already covers the case where GPS fusion can't run at all, so this fallback (and
    # FALLBACK_REGIONS, e.g. Austria->Croatia/Bosnia) is only actually exercised for the rarer
    # partial-GPS-data case now that the GPS-centroid path above handles the common one.
    return region_pbf_path(rel_path, osm_extracts_dir), fallback_region_pbf_paths(rel_path, osm_extracts_dir)
