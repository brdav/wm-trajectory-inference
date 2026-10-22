"""Catalogue of the OSM region extracts this dataset needs, on Geofabrik (https://download.geofabrik.de).
Region keys are what the rest of the pipeline uses: the dataset's country folder name, or "United States_<State>".
Used by scripts/setup/download_osm_extracts.py (downloads them) and scripts/osm_maps/osm_bev.py (boundary polygons)."""

BASE = "https://download.geofabrik.de"

# natix dataset country-folder name -> Geofabrik path (relative to BASE, without -latest.osm.pbf).
# Ireland has no Republic-of-Ireland-only extract on Geofabrik; "ireland-and-northern-ireland" is
# the only/correct option (also covers Northern Ireland/UK territory).
#
# Czech Republic and Liechtenstein aren't natix dataset folders -- added once real clips filed
# under other countries turned out to actually be located there (confirmed: a "Spain"-labeled
# clip actually near Brno, Czech Republic; several "Austria"-labeled clips actually in
# Liechtenstein, near Vaduz) -- same mislabeling pattern as vggt_slam/country_regions.py's other
# entries, just two more neighbors that happened not to have any of their OWN natix folder to
# already guarantee coverage.
EUROPE_COUNTRIES = {
    "Austria": "austria", "Belgium": "belgium", "Croatia": "croatia",
    "Czech Republic": "czech-republic", "Finland": "finland", "France": "france",
    "Germany": "germany", "Greece": "greece", "Hungary": "hungary", "Iceland": "iceland",
    "Ireland": "ireland-and-northern-ireland", "Italy": "italy",
    "Liechtenstein": "liechtenstein", "Latvia": "latvia", "Netherlands": "netherlands",
    "Poland": "poland", "Portugal": "portugal", "Romania": "romania", "Slovenia": "slovenia",
    "Spain": "spain", "Sweden": "sweden", "Switzerland": "switzerland",
}
EUROPE_PREFIX = "europe"

OTHER_REGIONS = {
    "Canada": "north-america/canada",
    "Puerto Rico": "north-america/us/puerto-rico",
    "Japan": "asia/japan",
    # Not a natix dataset country folder -- a fallback-only region for clips mislabeled as
    # Austria (confirmed: GPS fixes ~300km south of Austria, in Bosnia and Herzegovina). See
    # vggt_slam/regions.py's FALLBACK_REGIONS.
    "Bosnia and Herzegovina": "europe/bosnia-herzegovina",
}

# "United States" clips are nested one level deeper (natix's own layout: United States/<State>/...,
# unlike every other country here) -- each state maps to its own Geofabrik us/<slug> extract.
#
# Includes every US state + DC, not just the ones natix's own folder layout happens to have --
# clips are sometimes filed under the wrong state (confirmed on real clips, see
# vggt_slam/us_regions.py's docstring), so a clip's *true* location can land in a state that has
# no folder of its own in the dataset at all (e.g. a clip filed under "Rhode Island" but actually
# in New Hampshire, which otherwise wouldn't have an extract downloaded here). GPS-centroid-based
# region selection (vggt_slam/regions.py's `region_candidates`) needs the full set to have
# anywhere to actually resolve a mislabeled clip to.
US_STATE_SLUGS = {
    "Alabama": "alabama", "Alaska": "alaska", "Arizona": "arizona", "Arkansas": "arkansas",
    "California": "california", "Colorado": "colorado", "Connecticut": "connecticut",
    "Delaware": "delaware", "District of Columbia": "district-of-columbia", "Florida": "florida",
    "Georgia": "georgia", "Hawaii": "hawaii", "Idaho": "idaho", "Illinois": "illinois",
    "Indiana": "indiana", "Iowa": "iowa", "Kansas": "kansas", "Kentucky": "kentucky",
    "Louisiana": "louisiana", "Maine": "maine", "Maryland": "maryland",
    "Massachusetts": "massachusetts", "Michigan": "michigan", "Minnesota": "minnesota",
    "Mississippi": "mississippi", "Missouri": "missouri", "Montana": "montana",
    "Nebraska": "nebraska", "Nevada": "nevada", "New Hampshire": "new-hampshire",
    "New Jersey": "new-jersey", "New Mexico": "new-mexico", "New York": "new-york",
    "North Carolina": "north-carolina", "North Dakota": "north-dakota", "Ohio": "ohio",
    "Oklahoma": "oklahoma", "Oregon": "oregon", "Pennsylvania": "pennsylvania",
    "Rhode Island": "rhode-island", "South Carolina": "south-carolina",
    "South Dakota": "south-dakota", "Tennessee": "tennessee", "Texas": "texas", "Utah": "utah",
    "Vermont": "vermont", "Virginia": "virginia", "Washington": "washington",
    "West Virginia": "west-virginia", "Wisconsin": "wisconsin", "Wyoming": "wyoming",
}


def region_key_for_clip(country, state=None):
    """country/state as they appear in the dataset's own folder names. Returns the same
    region_key build_network's region-extract lookup uses -- keep any change here in sync with
    vggt_slam/map_match.py.
    """
    if country == "United States":
        return f"United States/{state}"
    return country


def all_targets():
    """Yields (region_key, geofabrik_path) for every extract this dataset needs."""
    for name, slug in EUROPE_COUNTRIES.items():
        yield name, f"{EUROPE_PREFIX}/{slug}"
    for name, path in OTHER_REGIONS.items():
        yield name, path
    for state, slug in US_STATE_SLUGS.items():
        yield region_key_for_clip("United States", state), f"north-america/us/{slug}"
