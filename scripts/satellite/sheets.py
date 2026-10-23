"""Aerial imagery from bulk GeoTIFF sheets listed in a WFS index (Poland, GUGiK; the WMS/WMTS forbid harvesting, the sheets are the
bulk route). Sheets (25 cm, 1:5000, ~2.3 x 2.1 km, ~35 MB) are downloaded once into <cache>/_sheets/<source>/ and painted with
cog.paint_tiles. Only RGB sheets at 0.2-0.5 m/px are used (CIR = false colour; the 10 cm sheets are ~300 MB), newest year first.
"""
import re
import threading
import time
from pathlib import Path

import requests

import cog

WFS = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WFS/Skorowidze"
YEARS = range(2026, 2014, -1)
_locks, _glock = {}, threading.Lock()


def _field(block, k):
    m = re.search(rf"<gugik:{k}>([^<]*)<", block)
    return m.group(1) if m else None


def find_sheets(bbox, sess, tries=4):
    """Download URLs of the usable sheets intersecting bbox = (west, south, east, north), newest first."""
    out = []
    for y in YEARS:
        url = (f"{WFS}?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature&TYPENAMES=gugik:SkorowidzOrtofomapy{y}&COUNT=200"
               f"&BBOX={bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]},urn:ogc:def:crs:EPSG::4326")
        for k in range(tries):
            try:
                r = sess.get(url, timeout=120)
                r.raise_for_status()
                break
            except requests.RequestException:
                if k == tries - 1:
                    raise
                time.sleep(2 ** k)
        for m in re.finditer(rf"<gugik:SkorowidzOrtofomapy{y} .*?</gugik:SkorowidzOrtofomapy{y}>", r.text, re.S):
            b = m.group(0)
            px = _field(b, "piksel")
            if _field(b, "kolor") == "RGB" and px and 0.2 <= float(px) <= 0.5 and _field(b, "url_do_pobrania"):
                out.append(_field(b, "url_do_pobrania"))
    return out


def fetch_sheet(url, root, sess, tries=5):
    """Local path of the sheet, downloaded once (atomic, safe for several threads)."""
    f = Path(root) / url.rsplit("/", 1)[1]
    with _glock:
        lock = _locks.setdefault(f.name, threading.Lock())
    with lock:
        if f.exists():
            return str(f)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f.name + ".part")
        for k in range(tries):
            try:
                with sess.get(url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(tmp, "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                tmp.rename(f)
                return str(f)
            except requests.RequestException:
                if k == tries - 1:
                    raise
                time.sleep(2 ** k * 5)


def render_tiles(cache, source, z, tiles, sess):
    sheets_dir = Path(cache) / "_sheets" / source
    urls = find_sheets(cog.window_bbox_lonlat(z, tiles), sess)
    return cog.paint_tiles(z, tiles, (fetch_sheet(u, sheets_dir, sess) for u in urls))
