# Aerial imagery: sources and licences (NOT legally reviewed)

Status as of 2026-10-23. This is a working note for whoever decides whether the satellite crops can be released. Licence
names below come from service capabilities, catalogue pages and web searches, **not** from the official licence texts, and
were not checked by a lawyer. Re-check each source before any release. Commercial and TOS-restricted imagery (Google, Bing,
Mapbox, Esri World Imagery) is deliberately not used. The crops are derived from the imagery (resampled, rotated), so the
attribution/share-alike terms of each source probably still apply to them.

Release context (user, 2026-10-23): the dataset is non-commercial, research only; whether it will be released is undecided, but
the sources should allow releasing our processed crops (attribution is fine). Non-commercial release terms are compatible with
all the CC BY / CC0 / OGL / dl-de sources below. That does not settle redistribution of derived images for sources whose terms are
unnamed or unconfirmed; those carry `release="check"` in `sources.py` (see the list at the end), and
`sources_of(region, strict=True)` leaves them out. GDPR: the high-resolution crops can show identifiable people and vehicles
(Lantmaeteriet treats its orthophotos as personal data), which is a separate question from the copyright licence.

Code: `sources.py` (region -> source), `fetch_tiles.py` (download into the tile cache), `render_sat.py` (crops).

## Used (routed in sources.py)

| Region | Source | Licence (as found) | Attribution | Notes |
|---|---|---|---|---|
| Switzerland | swisstopo SWISSIMAGE (WMTS) | swisstopo open government data | "(c) swisstopo" | 10 cm |
| Austria | basemap.at orthophoto 30 cm | CC BY 4.0 | basemap.at | |
| Japan | GSI seamless photo | GSI tile terms | "GSI Japan" | terms of use not read in detail |
| USA, Puerto Rico | USGS NAIP via Microsoft Planetary Computer | public domain (USGS) | USGS NAIP | 0.3-1 m; read as COGs |
| Netherlands | PDOK Luchtfoto Actueel_orthoHR | CC0 (PDOK catalogue) | none required | 8 cm |
| France | IGN Geoplateforme ORTHOIMAGERY.ORTHOPHOTOS | Licence Ouverte 2.0 (IGN terms) | IGN | capabilities only link to cartes.gouv.fr CGU |
| Spain | PNOA-ma, IGN/CNIG | CC BY 4.0 "scne.es" | required | 25-50 cm |
| Belgium (Flanders) | Digitaal Vlaanderen OMW omwrgb25vl | free reuse, Flanders geowebservices terms; user confirmed on 2026-10-23 that release is fine | state the source (Digitaal Vlaanderen) | |
| Czech Republic | CUZK Ortofoto CR (WMS) | CC BY 4.0 since 2023-07 per CUZK; WMS terms unclear in 2025 | required | confirm |
| Slovenia | GURS DOF5 (WMS) | CC BY 4.0 per eprostor metadata | required | |
| Germany (other states) | per-Bundesland WMS, see sources.py `de_*` | Hessen, NRW: dl-de/zero-2-0; Bayern, Thueringen, Schleswig-Holstein: CC BY 4.0; Brandenburg: dl-de/by-2-0; Sachsen, MV, Saarland, Bremen: free / open with credit (not read in detail) | per state | states without a verified open endpoint are not covered |
| Niedersachsen | LGLN DOP20 RGB, STAC of COGs (`de_ni`) | CC-BY-4.0 declared in the STAC collection; credit "Quelle: LGLN" + year | per terms | 20 cm |
| Baden-Wuerttemberg | LGL-BW ATKIS DOP 20 WMTS (`de_bw`) | dl-de/by-2-0 ("Open Data Lizenz") | LGL-BW (year), dl-de/by-2-0 | 20 cm |
| Rheinland-Pfalz | LVermGeoRP DOP20 WMS (`de_rp`) | dl-de/by-2-0 per capabilities/Geoshop; a 2024 news page still called DOP20 paid, so confirm with LVermGeoRP | GeoBasis-DE / LVermGeoRP (year) | 20 cm |
| Finland (Helsinki region only) | HSY Ortoilmakuva 2023 WMS (`hsy_fi`) | CC-BY-4.0 (hri.fi); credit "Municipalities of the area, HSY, HSL, Finnish Defence Forces 2023" | required | 25 cm open version; ~33 % of Finland's tiles |
| Finland (rest) | NLS orthophotos via Kapsi's keyless WMS mirror `tiles.kartat.kapsi.fi/ortokuva` (`nls_fi`), updated 2024-09 | CC BY 4.0 (NLS open data); the Kapsi mirror states no terms of its own (`AccessConstraints: none`) | "Contains data from the National Land Survey of Finland Orthophotos, CC BY 4.0" | 0.25-0.5 m; community-run server, fetch at a low rate. The official NLS WMTS needs a free API key |
| Italy (South Tyrol only) | Provincia di Bolzano orthophoto 2020 WMS (`bolzano`) | CC BY 4.0 (INSPIRE/geoportal records) | required | 20 cm; ~12 % of Italy's tiles |
| Italy (Toscana) | Regione Toscana GEOscopio OFC Ortofoto 2023 WMS (`toscana`) | CC BY 4.0 per the RNDT record (2026-09-02); the service page itself names no licence | required | 20 cm; the service burns a text strip into each image's left edge, cropped off by requesting extra columns |
| Latvia | LGIA orthophoto cycle 3, open ArcGIS tile cache (imported from a colleague's cache, `lv_lgia`) | "LGIA open data (old cycles)"; user confirmed on 2026-10-23 that release is fine | LGIA | native 1.32 m/px, fetched at z16 and upsampled 4x: blurry; only ~1.3k tiles |
| Romania | MApN / AIGA "Ortoimagini scara 1:5000 pentru teritoriul Romaniei" (data.gov.ro; imported from a colleague's cache, `ro_mapn2020`) | CC BY 4.0 (`license_id` CC-BY-4.0 in the data.gov.ro CKAN record; flagged as an EU high value dataset) | Ministerul Apararii Nationale / AIGA, data.gov.ro | captured 2017-2020, **native 1.5 m/px** (10x10 km GeoTIFF tiles, EPSG:31700), so the 0.5 m/px crops are ~3x upsampled and blurry; use `gsd_m` in `sources_meta.json` to filter |
| Portugal | DGT Ortofotos 25 cm 2025 (WMS `ortos2025`) | CC BY per dados.gov.pt (version not stated; confirm 4.0); capabilities: no conditions | required | other DGT vintages show "notspecified": not used; Madeira/Azores not covered |
| Poland | GUGiK Geoportal ortofotomapa, bulk GeoTIFF sheets (25 cm RGB; `sheets.py`) | "free of charge, any purpose" (no named licence); user confirmed on 2026-10-23 that release is fine | state the source (GUGiK) | the WMS/WMTS metadata excludes automated harvesting, so only the bulk download is used |
| Berlin | FIS-Broker / gdi.berlin.de DOP 2025 | dl-de/zero-2-0 | | 20 cm, leaf-off |
| Sachsen-Anhalt | LVermGeo DOP20 OpenData WMS | dl-de/by-2-0 | GeoBasis-DE/LVermGeo ST | |
| Canada (Quebec) | MRNF Imagerie aeroportee forestiere historique, layer Inventaire_Ecoforestier_2018 20 cm | CC BY 4.0 per Donnees Quebec; partnership imagery excluded | required | 2018 layer covers Monteregie/Montreal/Estrie/Laurentides only |
| Canada (Ontario) | Ontario Imagery Web Map Service (LIO) | OGL-Ontario, only where Ontario owns the IP; the service may include partner imagery that is not open | per OGL | confirm per layer/year |

## Found but NOT used

| Region | Source | Why not |
|---|---|---|
| Sweden | Lantmaeteriet Ortofoto Nedladdning (STAC, CC-BY-4.0) | Downloads need free Geotorget credentials (401 without); waiting for credentials. |
| Finland | NLS official WMTS | Needs a free API key; the Kapsi mirror (see Used) is routed instead. |
| Italy | PCN/MASE AGEA 2012 WMS (50 cm, `pcn_it` in sources.py, status "unclear") | No licence stated; reportedly excluded from the open licence. Regional open data only for parts of Toscana (on request), Emilia-Romagna flood layers, Marche (B/W). Lombardia/Veneto/Piemonte are AGEA-derived and restricted. AGEA national portal: licence unclear, per-request download. |
| Canada (Vancouver, Winnipeg) | Data for Canada PMTiles of the cities' open orthophotos (7.5 cm; community repackager, v0.1.0-beta), Metro Vancouver 2025 MrSID zips | City open-data licences (OGL-Vancouver, Winnipeg Open Data Licence), repackaging terms and exact licence unverified | ~7.5k + 1.6k tiles (0.5 % of the total); would need a PMTiles reader or a MrSID decoder; not done |
| Italy (Trento) | PAT orthophoto 2015 WMS (`trento`, status "unclear") | layer licence not stated (PAT portals: CC BY 4.0) | confirm before routing |
| Italy (Lombardia, Veneto, Emilia-Romagna AGEA 2020) | regional WMS of AGEA orthophotos | Not open. Lombardia: "copyright AGEA, licence of use granted to Regione Lombardia" (RNDT 2026-10-06), institutional use; Veneto: terms forbid any reproduction ("solo per la visualizzazione"); Emilia-Romagna AGEA 2020: "utilizzo ristretto". Not routed. Lombardia alone is ~98k tiles (5 % of the dataset). |
| Italy (Lazio) | Geoportale Lazio Ortofoto AGEA 2023 WMS | catalogue says "creativecommons" without a version, imagery is AGEA's under a Region-AGEA agreement; not routed until confirmed (~7.6k tiles) |
| Italy (Emilia-Romagna 2022) | RER 2022 WMS, CC BY 4.0 | covers only a patch around Imola; not added |
| Italy (national, AGEA) | GeoPortale AGEA, ~20 cm RGB GeoTIFF | The portal's own legal notes (i18n file) say the downloadable RGB GeoTIFFs are available under CC BY 4.0, citing EU 2023/138; this contradicts the regional documents above, and the regional WMS copies are not covered. Downloads: hCaptcha + email OTP, max 20 tiles of ~1 GB per request, per-email/IP/daily caps; not automated. A bulk route would need written agreement from AGEA (sian@agea.gov.it), which would cover Lombardia/Veneto 2024. |
| Sweden (Goeteborg) | Goeteborg open WMS, CC0 25 cm | none of our Swedish samples lie in the city (0 tiles), so not routed |
| Liechtenstein | LLV WMS | The WMS says "free for non-commercial use"; the OGD GeoTIFF download (CC BY) would be the way. 7.5 km2, not done. |
| Croatia | DGU WMS | Anonymous WMS images carry a large "GEOPORTAL" watermark. |
| Ireland, Greece, Romania, Hungary, Latvia | national mapping agencies | No open licence or open access at the needed resolution found. |
| Bosnia and Herzegovina | | Nothing found (not searched in depth). |
| Iceland | LMI mapcache | Only EPSG:3057, unverified blank image, licence reported CC BY 4.0. |
| Puerto Rico | USACE San Juan 2013 S3 tiles, USGSImageryOnly | partial area / z16 only; NAIP covers PR instead. |

Samples whose region (or part of a region) has no open imagery get no satellite image.

## Not covered yet (tiles at zoom 18, whole dataset, 2026-10-23)

Italy 167k, Finland 77k and Sweden 47k (credentials/key needed), Ireland 36k, Croatia 28k, Greece 21k, Romania 20k, Germany 18k
(Hamburg: no working open endpoint; BKG's national DOP is blocked by a WAF and not open), Hungary 14k, Canada 10k
(Vancouver, Winnipeg), Iceland 8k, Latvia 1.3k, Liechtenstein 0.6k, of 1.83M tiles in total (about 25 %).
Belgium: Wallonia (SPW ORTHO_LAST, no open licence found) and Brussels (UrbIS, CC0 for 2022/2024) are not routed;
Flanders covers the samples seen so far.

## Release confirmation

On 2026-10-23 the user confirmed that releasing the processed crops is fine, with attribution, for every routed source, including the ones
flagged earlier (Poland, Flanders, Ontario, Japan, Czech Republic, Rheinland-Pfalz, Sachsen, Mecklenburg-Vorpommern, Saarland, Bremen).
`release="check"` is no longer set on any source in `sources.py` (the `strict` option of `sources_of` stays for future sources).
Not routed because their terms restrict use: Liechtenstein WMS (non-commercial), Wallonia, Italy PCN, Croatia (watermark),
Lombardia/Veneto/Emilia-Romagna AGEA 2020 (see above).
