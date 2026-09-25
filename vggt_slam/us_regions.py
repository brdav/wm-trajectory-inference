"""Approximate geographic centroids of US states/DC, used to order OSM-region candidates by
proximity to a clip's own GPS fixes -- see scripts/batch_infer.py's `_region_candidates`.

Motivation: natix dataset clips are sometimes filed under the wrong state folder (confirmed on
several real clips: e.g. a clip under "Virginia" whose GPS fixes are actually in San Diego,
California; one under "Rhode Island" actually near Concord, New Hampshire). The folder name is
therefore not a reliable region label. These centroids let region selection instead be driven by
where a clip's GPS fixes actually are, trying the nearest state(s) first and falling through
(via align_trajectory_to_roads's existing fallback-region mechanism) until one actually has roads
nearby -- self-correcting regardless of how the clip happens to be filed.

Values are approximate (each within roughly half a degree of the state's true geographic center)
-- precision doesn't matter much here since they only decide try-order among candidates that are
each independently verified against the real road network before being accepted; a rough centroid
just means a few extra (cheap, indexed) probes before landing on the right state.
"""

US_STATE_CENTROIDS = {
    "Alabama": (32.8, -86.8), "Alaska": (64.0, -152.0), "Arizona": (34.3, -111.7),
    "Arkansas": (34.9, -92.4), "California": (37.2, -119.7), "Colorado": (39.0, -105.5),
    "Connecticut": (41.6, -72.7), "Delaware": (39.0, -75.5),
    "District of Columbia": (38.9, -77.0), "Florida": (28.6, -82.4), "Georgia": (32.6, -83.4),
    "Hawaii": (20.3, -156.4), "Idaho": (44.4, -114.6), "Illinois": (40.0, -89.2),
    "Indiana": (39.9, -86.3), "Iowa": (42.1, -93.5), "Kansas": (38.5, -98.4),
    "Kentucky": (37.5, -85.3), "Louisiana": (31.0, -92.0), "Maine": (45.4, -69.2),
    "Maryland": (39.0, -76.7), "Massachusetts": (42.3, -71.8), "Michigan": (44.3, -85.4),
    "Minnesota": (46.3, -94.3), "Mississippi": (32.7, -89.7), "Missouri": (38.4, -92.5),
    "Montana": (47.0, -109.6), "Nebraska": (41.5, -99.8), "Nevada": (39.3, -116.6),
    "New Hampshire": (43.7, -71.6), "New Jersey": (40.1, -74.7), "New Mexico": (34.4, -106.1),
    "New York": (42.9, -75.5), "North Carolina": (35.5, -79.4), "North Dakota": (47.5, -100.3),
    "Ohio": (40.3, -82.8), "Oklahoma": (35.5, -97.5), "Oregon": (44.0, -120.6),
    "Pennsylvania": (40.9, -77.7), "Rhode Island": (41.7, -71.5), "South Carolina": (33.9, -80.9),
    "South Dakota": (44.4, -100.3), "Tennessee": (35.8, -86.4), "Texas": (31.5, -99.3),
    "Utah": (39.3, -111.6), "Vermont": (44.0, -72.7), "Virginia": (37.5, -78.9),
    "Washington": (47.4, -120.5), "West Virginia": (38.6, -80.6), "Wisconsin": (44.6, -89.9),
    "Wyoming": (43.0, -107.5),
}


def nearest_states(lat, lon, candidates=None):
    """Returns state names from `candidates` (default: all of US_STATE_CENTROIDS), ordered by
    approximate distance from (lat, lon) -- nearest first. Distance is a simple equirectangular
    approximation (good enough for ordering purposes, see module docstring), not true geodesic
    distance.
    """
    import math

    names = candidates if candidates is not None else US_STATE_CENTROIDS.keys()
    coslat = math.cos(math.radians(lat))

    def dist2(name):
        clat, clon = US_STATE_CENTROIDS[name]
        dlat = lat - clat
        dlon = (lon - clon) * coslat
        return dlat * dlat + dlon * dlon

    return sorted(names, key=dist2)
