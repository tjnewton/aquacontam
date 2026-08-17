# paper/assets

## us_states_cb2023_5m.geojson

US state (plus DC and territory) boundary polygons used by `paper/_geo_flags.py` for the
Extended Data Fig. 1 mis-geocode exclusion flag and for coloring systems whose PWSID prefix
is not a state code. 56 features with `STUSPS` and `NAME` properties.

- **Source**: US Census Bureau cartographic boundary files, 2023 vintage, 1:5,000,000 scale.
  <https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_5m.zip>
  (retrieved 2026-07-12)
- **License**: public domain (work of the US federal government, 17 U.S.C. 105).
- **Transform**: source CRS NAD83 (EPSG:4269), treated as WGS84-equivalent for CONUS
  (difference < ~2 m, far below the 1:5M generalization error). Coordinates rounded to
  4 decimal places (~11 m); geometries failing `is_valid` after rounding repaired with
  `shapely.make_valid` (4 features). Features sorted by `STUSPS`; compact separators;
  LF-only; no timestamps (byte-reproducible).
- **Reproduce**:

```bash
curl -sLO https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_5m.zip
unzip cb_2023_us_state_5m.zip -d cb_5m
python - <<'PY'
import json
import geopandas as gpd
from shapely import make_valid
from shapely.geometry import mapping
from shapely.ops import transform

gdf = gpd.read_file("cb_5m/cb_2023_us_state_5m.shp")
feats = []
for _, row in gdf.sort_values("STUSPS").iterrows():
    geom = transform(lambda x, y, z=None: (round(x, 4), round(y, 4)), row.geometry)
    if not geom.is_valid:
        geom = make_valid(geom)
    feats.append({"type": "Feature",
                  "properties": {"STUSPS": str(row.STUSPS), "NAME": str(row.NAME)},
                  "geometry": json.loads(json.dumps(mapping(geom)))})
fc = {"type": "FeatureCollection", "name": "us_states_cb2023_5m",
      "description": ("US Census Bureau cartographic boundary file cb_2023_us_state_5m "
                      "(1:5,000,000), coordinates rounded to 4 decimals, geometries "
                      "validity-repaired where needed. Public domain (US government work). "
                      "See paper/assets/README.md for provenance and reproduction commands."),
      "features": feats}
with open("us_states_cb2023_5m.geojson", "w", encoding="utf-8", newline="\n") as f:
    json.dump(fc, f, separators=(",", ":"))
    f.write("\n")
PY
```

The polygon lookup itself (point-in-state via shapely STRtree, nearest-state fallback within
3.0 degrees for water/offshore points) lives in `paper/_geo_flags.py`.
