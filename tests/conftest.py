import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
for d in ("slam", "metric_scale", "osm_maps"):  # scripts are imported by module name
    sys.path.insert(0, str(REPO / "scripts" / d))
