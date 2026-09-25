"""Approximate geographic centroids of every non-US country/region this dataset spans (see
scripts/download_osm_extracts.py's EUROPE_COUNTRIES/OTHER_REGIONS), used to order OSM-region
candidates by proximity to a clip's own GPS fixes -- see scripts/batch_infer.py's
`_region_candidates`. Mirrors vggt_slam/us_regions.py's approach and rationale for US states,
generalized to countries once the same folder-mislabeling pattern was confirmed dataset-wide, not
just within the US: e.g. real clips filed under "Austria" whose GPS is actually in Germany,
"Croatia" actually in Germany (near Munich), "Switzerland" actually in central Italy (~700km from
the Swiss border -- too far to be a mere border mixup), "Poland" actually in Berlin, Germany.

As with us_regions.py, precision doesn't matter much -- centroids only decide try-order among
candidates that are each independently verified against the real road network before being
accepted, and align_trajectory_to_roads's existing fallback-region mechanism tries every
candidate with a downloaded extract, not just the nearest one.
"""

COUNTRY_CENTROIDS = {
    "Austria": (47.5, 14.5), "Belgium": (50.6, 4.7),
    "Bosnia and Herzegovina": (44.2, 17.8), "Canada": (56.0, -106.0),
    "Croatia": (45.3, 16.0), "Czech Republic": (49.8, 15.5), "Finland": (64.0, 26.0),
    "France": (46.6, 2.5), "Germany": (51.2, 10.4), "Greece": (39.0, 22.0),
    "Hungary": (47.0, 19.5), "Iceland": (65.0, -19.0), "Ireland": (53.4, -8.0),
    "Italy": (42.8, 12.8), "Japan": (36.5, 138.0), "Latvia": (56.9, 24.6),
    "Liechtenstein": (47.17, 9.53), "Netherlands": (52.2, 5.5), "Poland": (52.0, 19.5),
    "Portugal": (39.6, -8.0), "Puerto Rico": (18.2, -66.5), "Romania": (45.9, 24.9),
    "Slovenia": (46.1, 14.8), "Spain": (40.0, -4.0), "Sweden": (62.0, 15.0),
    "Switzerland": (46.8, 8.2),
}


def nearest_countries(lat, lon, candidates=None):
    """Returns country names from `candidates` (default: all of COUNTRY_CENTROIDS), ordered by
    approximate distance from (lat, lon) -- nearest first. Same equirectangular approximation as
    vggt_slam/us_regions.py's nearest_states (good enough for ordering purposes, see module
    docstring), not true geodesic distance."""
    import math

    names = candidates if candidates is not None else COUNTRY_CENTROIDS.keys()
    coslat = math.cos(math.radians(lat))

    def dist2(name):
        clat, clon = COUNTRY_CENTROIDS[name]
        dlat = lat - clat
        dlon = (lon - clon) * coslat
        return dlat * dlat + dlon * dlon

    return sorted(names, key=dist2)
