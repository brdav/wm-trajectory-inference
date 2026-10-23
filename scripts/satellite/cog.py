"""Aerial tiles from cloud-optimised GeoTIFFs found via a STAC API (US NAIP on the Microsoft Planetary Computer).

paint_tiles() paints the web-mercator window of a set of z-level tiles from rasters (newest first, older ones only fill the nodata
left by newer ones), then cuts it into 256 px JPEG tiles for the tile cache; render_tiles() gets the rasters from a STAC search.
sheets.py reuses paint_tiles for downloaded sheets. Needs rasterio.
"""
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import requests

from tiles import TILE_PX, WORLD_M

SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}"     # Planetary Computer asset-signing tokens
_tok = {}
_lock = threading.Lock()


def _token(collection):
    with _lock:
        t = _tok.get(collection)
        if t is None or t[1] - time.time() < 600:
            j = requests.get(SAS.format(collection=collection), timeout=60).json()
            exp = datetime.fromisoformat(j["msft:expiry"].replace("Z", "+00:00")).replace(tzinfo=timezone.utc).timestamp()
            t = _tok[collection] = (j["token"], exp)
        return t[0]


def _items(cfg, bbox, tries=5):
    """STAC items of the source `cfg` (dict from sources.py) intersecting bbox, newest first."""
    for k in range(tries):
        try:
            if cfg.get("method", "POST") == "POST":
                r = requests.post(cfg["stac"], json=dict(collections=[cfg["collection"]], bbox=bbox, limit=100), timeout=120)
            else:
                r = requests.get(cfg["stac"], params=dict(collections=cfg["collection"], bbox=",".join(map(str, bbox)), limit=100), timeout=120)
            r.raise_for_status()
            return sorted(r.json()["features"], key=lambda f: f["properties"]["datetime"], reverse=True)
        except requests.RequestException:
            time.sleep(2 ** k)
    raise RuntimeError(f"STAC search failed for {bbox}")


def paint_tiles(z, tiles, hrefs, quality=90):
    """tiles: [(x, y)] at zoom z; hrefs: iterable (consumed lazily) of rasters, newest first -> {(x, y): jpeg bytes or None}.
    Each raster is warped to the window of the tiles; older ones only fill what newer ones left empty (all-zero = no data)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT

    x0, x1 = min(x for x, _ in tiles), max(x for x, _ in tiles) + 1
    y0, y1 = min(y for _, y in tiles), max(y for _, y in tiles) + 1
    side = WORLD_M / 2 ** z
    west, north = -WORLD_M / 2 + x0 * side, WORLD_M / 2 - y0 * side
    w, h = (x1 - x0) * TILE_PX, (y1 - y0) * TILE_PX
    dst = from_origin(west, north, side / TILE_PX, side / TILE_PX)
    img = np.zeros((h, w, 3), np.uint8)
    todo = np.ones((h, w), bool)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY=5,
                      GDAL_HTTP_RETRY_DELAY=2, GDAL_CACHEMAX=512):
        for href in hrefs:
            if not todo.any():
                break
            with rasterio.open(href) as src:
                with WarpedVRT(src, crs="EPSG:3857", transform=dst, width=w, height=h, resampling=Resampling.bilinear) as vrt:
                    a = vrt.read([1, 2, 3]).transpose(1, 2, 0)
            ok = todo & a.any(axis=2)                       # raster edges / gaps are 0 in all bands
            img[ok], todo[ok] = a[ok], False
    out = {}
    for x, y in tiles:
        sl = (slice((y - y0) * TILE_PX, (y - y0 + 1) * TILE_PX), slice((x - x0) * TILE_PX, (x - x0 + 1) * TILE_PX))
        if todo[sl].mean() > 0.5:                           # mostly no imagery: treat as missing
            out[(x, y)] = None
        else:
            out[(x, y)] = cv2.imencode(".jpg", img[sl][:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
    return out


def window_bbox_lonlat(z, tiles):
    """(west, south, east, north) in degrees of the window of `tiles` at zoom z."""
    from rasterio.warp import transform_bounds
    x0, x1 = min(x for x, _ in tiles), max(x for x, _ in tiles) + 1
    y0, y1 = min(y for _, y in tiles), max(y for _, y in tiles) + 1
    side = WORLD_M / 2 ** z
    return transform_bounds("EPSG:3857", "EPSG:4326", -WORLD_M / 2 + x0 * side, WORLD_M / 2 - y1 * side,
                            -WORLD_M / 2 + x1 * side, WORLD_M / 2 - y0 * side)


def render_tiles(cfg, z, tiles, quality=90):
    """STAC source (sources.py entry with kind "cog") -> {(x, y): jpeg or None}."""
    sfx = (lambda: "?" + _token(cfg["collection"])) if cfg.get("sas") else (lambda: "")
    items = _items(cfg, list(window_bbox_lonlat(z, tiles)))
    return paint_tiles(z, tiles, (f["assets"][cfg.get("asset", "image")]["href"] + sfx() for f in items), quality)
