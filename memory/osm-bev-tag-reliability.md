# OSM tag reliability for the map images (`scripts/osm_maps/osm_bev.py`)

- **Traffic signs:** `traffic_sign` is free text with country prefixes, so only `stop` and
  `give_way` are drawn. A co-tag check (script since removed) compared each national code with the
  core OSM tags on the same nodes: each mapped code is co-tagged `highway=stop` / `give_way` on
  45-100% of its nodes (IT:II.37 / II.36 the lowest, 45% / 49%) and on ~0% with the opposite class.
  Everything else (no-entry, warning, priority-road, no-parking, city-limit, speed limits, ...) is
  deliberately not drawn: per-country codes that cannot be validated this way.
- **Speed limits:** tagged limits are used; otherwise inferred from a touching same-class road
  (hold-out test on roads with mapped limits: right within 3 km/h for 79-96% of roads in
  CH/IT/ES/SE/PL). Statutory defaults were dropped: on residential streets they are right only
  6-85% of the time (many 30 zones).
- **Lanes:** tagged counts, or the neighbour's count if all same-class same-direction neighbours
  agree. A default count (1 one-way / 2 two-way) was dropped: right only 36-90% of the time, and
  its only visible effect was a guessed centre line.
