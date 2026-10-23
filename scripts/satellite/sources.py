"""Open aerial-imagery sources and which region each one serves. All tiles are cached at web-mercator zoom Z = 18 (0.60 m/px at the
equator, ~0.4 m/px in Switzerland, ~0.5 m/px in Nevada), next to the 0.5 m/px target of the map crops. Licences: see LICENSES.md.

  kind "xyz"  : plain {z}/{x}/{y} tile URL (any WMTS KVP URL with {z}/{x}/{y} placeholders works too)
  kind "wms"  : WMS GetMap of the tile's web-mercator bbox, 256 px
  kind "sheets": bulk GeoTIFF sheets listed in a WFS index, downloaded once (sheets.py)
  `crop_left` (wms only): the service burns a text strip into the left edge of every image; the tile is requested with that many
              extra columns on the left, which are cut off again
  kind "imported": tiles somebody else fetched, brought into our cache by import_tiles.py (nothing to download)
  kind "cog"  : cloud-optimised GeoTIFFs found through a STAC API, read with rasterio (cog.py)

A region lists its sources in priority order. `poly` (a Geofabrik path) limits a source to that area, e.g. one German state; a
source without `poly` is tried for every tile of its region. A tile is fetched from the first source that has it.
`bbox` (west, south, east, north) limits a partial source (one city / province) the same way.
`release` "check" marks sources whose licence we could not confirm to allow redistributing derived images (note in `release_note`);
sources_of(region, strict=True) leaves them out. LICENSES.md has the details.
`status` "open" sources are used; "unclear" ones are kept here but never routed (licence not confirmed, see LICENSES.md).
"""
import json
import urllib.request
from pathlib import Path

import numpy as np

ZOOM = 18
GEOFABRIK = "https://download.geofabrik.de"


def wms(base, layer, fmt="image/jpeg", ver="1.3.0", pad=0):
    """GetMap URL template for a 256 px tile; pad > 0 asks for `pad` extra columns on the left (see `crop_left`)."""
    crs = "CRS" if ver == "1.3.0" else "SRS"
    return (f"{base}?SERVICE=WMS&VERSION={ver}&REQUEST=GetMap&LAYERS={layer}&STYLES=&{crs}=EPSG:3857&BBOX={{bbox}}"
            f"&WIDTH={256 + pad}&HEIGHT=256&FORMAT={fmt}")


SOURCES = {
    "swisstopo": dict(kind="xyz", regions=["Switzerland"], license="swisstopo OGD",
                      url="https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage/default/current/3857/{z}/{x}/{y}.jpeg"),
    "basemap_at": dict(kind="xyz", regions=["Austria"], license="CC BY 4.0",
                       url="https://mapsneu.wien.gv.at/basemap/bmaporthofoto30cm/normal/google3857/{z}/{y}/{x}.jpeg"),
    "gsi": dict(kind="xyz", regions=["Japan"], license="GSI terms", 
                url="https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg"),
    "naip": dict(kind="cog", regions=["Puerto Rico"], prefixes=["United States_"], license="public domain",
                stac="https://planetarycomputer.microsoft.com/api/stac/v1/search", collection="naip", sas=True),
    "pdok_nl": dict(kind="xyz", regions=["Netherlands"], license="CC0",
                    url="https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/Actueel_orthoHR/EPSG:3857/{z}/{x}/{y}.jpeg"),
    "ign_fr": dict(kind="xyz", regions=["France"], license="Licence Ouverte 2.0",
                   url="https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=ORTHOIMAGERY.ORTHOPHOTOS"
                       "&STYLE=normal&TILEMATRIXSET=PM&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&FORMAT=image/jpeg"),
    "pnoa_es": dict(kind="xyz", regions=["Spain"], license="CC BY 4.0",
                    url="https://www.ign.es/wmts/pnoa-ma?service=WMTS&request=GetTile&version=1.0.0&layer=OI.OrthoimageCoverage"
                        "&style=default&tilematrixset=GoogleMapsCompatible&tilematrix={z}&tilerow={y}&tilecol={x}&format=image/jpeg"),
    "flanders": dict(kind="xyz", regions=["Belgium"], license="Flanders geowebservices terms (free reuse), release confirmed by the user 2026-10-23, attribution required",
                     url="https://geo.api.vlaanderen.be/OMW/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=omwrgb25vl&STYLE="
                         "&TILEMATRIXSET=GoogleMapsVL&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&FORMAT=image/png"),
    "cuzk_cz": dict(kind="wms", regions=["Czech Republic"], license="CC BY 4.0 (check)", 
                    url=wms("https://ags.cuzk.gov.cz/arcgis1/services/ORTOFOTO/MapServer/WMSServer", "0")),
    "gurs_si": dict(kind="wms", regions=["Slovenia"], license="CC BY 4.0",
                    url=wms("https://storitve.eprostor.gov.si/ows-pub-wms/SI.GURS.ZPDZ/ows", "DOF5")),
    "gugik_pl": dict(kind="sheets", regions=["Poland"], license="GUGiK: free for any purpose (no named licence); release confirmed by the user 2026-10-23, attribution required; WMS/WMTS forbid harvesting"),
    "dgt_pt": dict(kind="wms", regions=["Portugal"], license="CC BY (version to confirm)",
                   url=wms("https://cartografia.dgterritorio.gov.pt/wms/ortos2025", "Ortos2025-RGB")),
    "ontario": dict(kind="xyz", regions=["Canada"], poly="north-america/canada/ontario", license="OGL-Ontario (check partner imagery)", 
                    url="https://ws.lioservices.lrc.gov.on.ca/arcgis2/rest/services/LIO_Imagery/Ontario_Imagery_Web_Map_Service/MapServer/tile/{z}/{y}/{x}"),
    "quebec": dict(kind="wms", regions=["Canada"], poly="north-america/canada/quebec", license="CC BY 4.0 (partner imagery excluded)",
                   url=wms("https://servicesmatriciels.mern.gouv.qc.ca/erdas-iws/ogc/wms/Imagerie_Aeroportee_Forestiere_Historique",
                           "Inventaire_Ecoforestier_2018_2018_Inv_Ecofor_20cm_RVB")),
    # Germany: one WMS per Bundesland (BKG's national DOP is blocked by a WAF)
    "de_he": dict(kind="wms", regions=["Germany"], poly="europe/germany/hessen", license="dl-de/zero-2-0",
                  url=wms("https://inspire-hessen.de/raster/dop20/ows", "he_dop20", "image/png")),
    "de_nw": dict(kind="wms", regions=["Germany"], poly="europe/germany/nordrhein-westfalen", license="dl-de/zero-2-0",
                  url=wms("https://www.wms.nrw.de/geobasis/wms_nw_dop", "nw_dop_rgb")),
    "de_by": dict(kind="wms", regions=["Germany"], poly="europe/germany/bayern", license="CC BY 4.0",
                  url=wms("https://geoservices.bayern.de/od/wms/dop/v1/dop20", "by_dop20c")),
    "de_be": dict(kind="wms", regions=["Germany"], poly="europe/germany/berlin", license="dl-de/zero-2-0",
                  url=wms("https://gdi.berlin.de/services/wms/dop_2025_fruehjahr", "dop_2025")),
    "de_bb": dict(kind="wms", regions=["Germany"], poly="europe/germany/brandenburg", license="dl-de/by-2-0",
                  url=wms("https://isk.geobasis-bb.de/mapproxy/dop20c/service/wms", "bebb_dop20c")),
    "de_st": dict(kind="wms", regions=["Germany"], poly="europe/germany/sachsen-anhalt", license="dl-de/by-2-0",
                  url=wms("https://www.geodatenportal.sachsen-anhalt.de/wss/service/ST_LVermGeo_DOP_WMS_OpenData/guest", "lsa_lvermgeo_dop20_2")),
    "de_sn": dict(kind="wms", regions=["Germany"], poly="europe/germany/sachsen", license="Sachsen terms (free)", 
                  url=wms("https://geodienste.sachsen.de/wms_geosn_dop-rgb/guest", "sn_dop_020")),
    "de_th": dict(kind="wms", regions=["Germany"], poly="europe/germany/thueringen", license="CC BY 4.0",
                  url=wms("https://www.geoproxy.geoportal-th.de/geoproxy/services/DOP20", "th_dop")),
    "de_mv": dict(kind="wms", regions=["Germany"], poly="europe/germany/mecklenburg-vorpommern", license="open, credit GeoBasis-DE/M-V", 
                  url=wms("https://www.geodaten-mv.de/dienste/adv_dop", "mv_dop")),
    "de_sh": dict(kind="wms", regions=["Germany"], poly="europe/germany/schleswig-holstein", license="CC BY 4.0",
                  url=wms("https://dienste.gdi-sh.de/WMS_SH_DOP20col_OpenGBD", "sh_dop20_rgb")),
    "de_sl": dict(kind="wms", regions=["Germany"], poly="europe/germany/saarland", license="no limitations (check)", 
                  url=wms("https://geoportal.saarland.de/freewms/dop2023", "sl_dop20_rgb")),
    "de_hb": dict(kind="wms", regions=["Germany"], poly="europe/germany/bremen", license="CC BY", 
                  url=wms("https://geodienste.bremen.de/wms_dop20_2023", "DOP20_2023_HB")),
    # Bremen and Hamburg lie inside Geofabrik's Niedersachsen / Schleswig-Holstein polygons: de_hb comes first
    "de_ni": dict(kind="cog", regions=["Germany"], poly="europe/germany/niedersachsen", license="CC BY 4.0 (credit LGLN)", method="GET",
                  stac="https://dop.stac.lgln.niedersachsen.de/search", collection="DOP", asset="dop20_rgb"),
    "de_bw": dict(kind="xyz", regions=["Germany"], poly="europe/germany/baden-wuerttemberg", license="dl-de/by-2-0",
                  url="https://owsproxy.lgl-bw.de/owsproxy/ows/WMTS_LGL-BW_ATKIS_DOP_20_C?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
                      "&LAYER=DOP_20_C&STYLE=default&TILEMATRIXSET=GoogleMapsCompatible&TILEMATRIX=GoogleMapsCompatible:{z}"
                      "&TILEROW={y}&TILECOL={x}&FORMAT=image/jpeg"),
    "de_rp": dict(kind="wms", regions=["Germany"], poly="europe/germany/rheinland-pfalz", license="dl-de/by-2-0 (confirm: DOP20 was paid until recently)", 
                  url=wms("https://geo4.service24.rlp.de/wms/rp_dop20.fcgi", "rp_dop20")),
    # partial sources: a city / province only
    "hsy_fi": dict(kind="wms", regions=["Finland"], bbox=(24.5, 60.05, 25.35, 60.5), license="CC BY 4.0 (credit municipalities/HSY/HSL/Finnish Defence Forces)",
                   url=wms("https://kartta.hsy.fi/geoserver/wms", "taustakartat_ja_aluejaot:Ortoilmakuva_2023", ver="1.1.1")),
    # national NLS orthophotos through Kapsi's keyless mirror (the official WMTS needs an API key); after HSY so Helsinki keeps the 25 cm layer
    "nls_fi": dict(kind="wms", regions=["Finland"], license="CC BY 4.0 (NLS open data; the Kapsi mirror states no terms of its own)",
                   url=wms("https://tiles.kartat.kapsi.fi/ortokuva", "ortokuva")),
    "bolzano": dict(kind="wms", regions=["Italy"], bbox=(10.35, 46.2, 12.5, 47.1), license="CC BY 4.0",
                    url=wms("https://geoservices.buergernetz.bz.it/mapproxy/p_bz-Orthoimagery/ows", "Aerial-2020-RGB")),
    "trento": dict(kind="wms", regions=["Italy"], bbox=(10.4, 45.65, 11.5, 46.55), status="unclear", license="layer licence not stated (PAT portals: CC BY 4.0)",
                   url=wms("https://siat.provincia.tn.it/geoserver/stem/ecw-rgb-2015/wms", "ecw-rgb-2015")),
    "toscana": dict(kind="wms", regions=["Italy"], bbox=(9.7, 42.3, 12.4, 44.5), crop_left=24, license="CC BY 4.0 (RNDT record)",
                    url=wms("https://www502.regione.toscana.it/ows_ofc/com.rt.wms.RTmap/wms?map=owsofc_rt&", "rt_ofc.5k23.32bit", pad=24)),
    # imported from a colleague's tile cache (/iopsstor/scratch/cscs/qzou/natix_aerial/tiles, import_tiles.py); the user confirmed the
    # licences on 2026-10-23 (all except PCN 2012, which is not used). AGEA: flights 2022-2024 are CC BY 4.0 on the AGEA portal.
    "li_swisstopo": dict(kind="imported", regions=["Liechtenstein"], year="latest", gsd_m=0.10, license="swisstopo OGD"),
    "it_trento_pat2015": dict(kind="imported", regions=["Italy"], year=2015, gsd_m=0.20, license="CC BY 4.0 (PAT portals)"),
    "it_veneto_agea2024": dict(kind="imported", regions=["Italy"], year=2024, gsd_m=0.20, license="AGEA CC BY 4.0 (regional WMS copy)"),
    "it_emiliaromagna_agea2023": dict(kind="imported", regions=["Italy"], year=2023, gsd_m=0.20, license="AGEA CC BY 4.0 (regional WMS copy)"),
    "it_lazio_agea2023": dict(kind="imported", regions=["Italy"], year=2023, gsd_m=0.20, license="AGEA CC BY 4.0 (regional WMS copy)"),
    "hr_dgu2024": dict(kind="imported", regions=["Croatia"], year=2024, gsd_m=0.50, license="Otvorena dozvola (DGU registered WMS)"),
    "hr_dgu2023": dict(kind="imported", regions=["Croatia"], year=2023, gsd_m=0.50, license="Otvorena dozvola (DGU registered WMS)"),
    "ro_mapn2020": dict(kind="imported", regions=["Romania"], year="2017-2020", gsd_m=1.5,
                        license="CC BY 4.0 (data.gov.ro record, MApN / AIGA)",
                        note="data.gov.ro 'Ortoimagini scara 1:5000 pentru teritoriul Romaniei', GeoTIFF 1.5 m EPSG:31700; native 1.5 m, so the "
                             "0.5 m/px crops are ~3x upsampled (blurry); fetched at z17, upsampled to z18"),
    "lv_lgia": dict(kind="imported", regions=["Latvia"], year="cycle 3", gsd_m=1.32, license="LGIA open data (old cycles); confirmed by the user 2026-10-23",
                    note="LGIA orthophoto cycle 3 open ArcGIS tile cache, native 1.32 m, fetched at z16 and upsampled 4x to z18 (blurry)"),
    # not routed: licence not confirmed
    "pcn_it": dict(kind="wms", regions=["Italy"], status="unclear", license="PCN/MASE 2012 AGEA ortho: no licence stated, reportedly excluded from open licence",
                   url=wms("http://wms.pcn.minambiente.it/ogc?map=/ms_ogc/WMS_v1.3/raster/ortofoto_colore_12.map", "OI.ORTOIMMAGINI.2012.32")),
}


# credit line per source, for the attribution of the released images (the index of every image lists the sources it used)
ATTRIBUTION = {
    "swisstopo": "(c) swisstopo", "basemap_at": "basemap.at", "gsi": "GSI Japan", "naip": "USGS NAIP", "pdok_nl": "PDOK / Kadaster / Beeldmateriaal NL",
    "ign_fr": "IGN", "pnoa_es": "PNOA (c) Instituto Geografico Nacional de Espana, CC BY 4.0 scne.es", "flanders": "Digitaal Vlaanderen",
    "cuzk_cz": "CUZK", "gurs_si": "GURS (Geodetska uprava Republike Slovenije)", "gugik_pl": "GUGiK", "dgt_pt": "DGT (Direcao-Geral do Territorio)",
    "ontario": "Ontario Ministry of Natural Resources / LIO (OGL-Ontario)", "quebec": "Gouvernement du Quebec (MRNF)",
    "de_he": "HVBG", "de_nw": "Geobasis NRW", "de_by": "Bayerische Vermessungsverwaltung", "de_be": "Geoportal Berlin", "de_bb": "GeoBasis-DE/LGB",
    "de_st": "GeoBasis-DE/LVermGeo ST", "de_sn": "GeoSN", "de_th": "GDI-Th", "de_mv": "GeoBasis-DE/M-V", "de_sh": "GeoBasis-DE/LVermGeo SH (GDI-SH)",
    "de_sl": "LVGL Saarland", "de_hb": "GeoInformation Bremen", "de_ni": "Quelle: LGLN", "de_bw": "LGL-BW, dl-de/by-2-0",
    "de_rp": "GeoBasis-DE / LVermGeoRP, dl-de/by-2-0", "hsy_fi": "Municipalities of the area, HSY, HSL, Finnish Defence Forces 2023",
    "nls_fi": "Contains data from the National Land Survey of Finland Orthophotos, CC BY 4.0",
    "bolzano": "Provincia Autonoma di Bolzano - Alto Adige", "toscana": "Regione Toscana", "li_swisstopo": "(c) swisstopo",
    "it_trento_pat2015": "Provincia autonoma di Trento", "it_veneto_agea2024": "AGEA / Regione del Veneto",
    "it_emiliaromagna_agea2023": "AGEA / Regione Emilia-Romagna", "it_lazio_agea2023": "AGEA / Regione Lazio",
    "hr_dgu2024": "Drzavna geodetska uprava (DGU)", "hr_dgu2023": "Drzavna geodetska uprava (DGU)", "lv_lgia": "LGIA (Latvijas Geotelpiskas informacijas agentura)", "ro_mapn2020": "Ministerul Apararii Nationale / Agentia de Informatii Geospatiale a Apararii (AIGA), data.gov.ro, CC BY 4.0",
}


def sources_of(region, strict=False):
    """Names of the open sources of a region, in priority order (strict: only those confirmed to allow releasing derived images)."""
    return [n for n, s in SOURCES.items() if s.get("status", "open") == "open" and not (strict and s.get("release") == "check")
            and (region in s["regions"] or any(region.startswith(p) for p in s.get("prefixes", ())))]


def tile_url(source, z, x, y):
    from tiles import WORLD_M
    s = SOURCES[source]
    assert s["kind"] in ("xyz", "wms"), f"{source}: no per-tile url"
    if s["kind"] == "xyz":
        return s["url"].format(z=z, x=x, y=y)
    side = WORLD_M / 2 ** z
    x0, y1 = -WORLD_M / 2 + x * side, WORLD_M / 2 - y * side
    x0 -= s.get("crop_left", 0) * side / 256                # extra columns on the left
    return s["url"].format(bbox=f"{x0},{y1 - side},{x0 + side},{y1}")


def _parse_poly(txt):
    import shapely
    lines, outers, holes, i = [ln.strip() for ln in txt.splitlines()], [], [], 1
    while i < len(lines):
        head, i, ring = lines[i], i + 1, []
        if not head or head == "END":
            continue
        while i < len(lines) and lines[i] != "END":
            if lines[i]:
                ring.append(tuple(float(v) for v in lines[i].split()[:2]))
            i += 1
        i += 1
        if len(ring) >= 4:
            (holes if head.startswith("!") else outers).append(shapely.Polygon(ring))
    geom = shapely.unary_union(outers)
    return geom.difference(shapely.unary_union(holes)) if holes else geom


def load_poly(cache, path):
    """Geofabrik boundary of `path`, downloaded once into <cache>/_poly/."""
    import shapely
    f = Path(cache) / "_poly" / (path.replace("/", "__") + ".wkt")
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        txt = urllib.request.urlopen(f"{GEOFABRIK}/{path}.poly", timeout=120).read().decode()
        f.write_text(shapely.simplify(_parse_poly(txt), 0.001).wkt)
    return shapely.from_wkt(f.read_text())


def tiles_in_source(cache, source, lon, lat):
    """Boolean mask: which points (tile centres) the source may serve (inside its `poly`, or all if it has none)."""
    import shapely
    cfg = SOURCES[source]
    m = np.ones(len(lon), bool)
    if "bbox" in cfg:
        w, s, e, n = cfg["bbox"]
        m &= (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
    if cfg.get("poly"):
        m &= shapely.contains_xy(load_poly(cache, cfg["poly"]), lon, lat)
    return m
