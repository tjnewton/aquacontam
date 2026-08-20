# AquaContam Data Sources

This directory contains raw, interim, and processed data for the AquaContam benchmark.
Raw data is immutable; all transformations write to `interim/` or `processed/`.

## Data Sources

| Source | Records | Analytes | Format | License |
|--------|---------|----------|--------|---------|
| UCMR5 | ~1.9M | 29 PFAS + lithium | CSV (Latin-1) | Public |
| UCMR3 | ~1.1M | 6 PFAS + 32 other | CSV (Latin-1) | Public |
| SDWIS | ~160K systems | Lead, copper | CSV | Public |
| EPA FRS | ~40K sites | Facility locations | CSV/ZIP | Public |
| MI MPART | ~5.6K | 5 PFAS | ArcGIS REST | Public |
| CA GeoTracker | ~324K | 29 PFAS | CSV | Public |
| MN MDH | ~247K | 27 PFAS | Excel (written request) | Provided on request; redistribution pending confirmation — see `docs/DATA_TERMS.md` |
| NJ DEP | ~248K | 25 PFAS | Manual CSV export | Public |
| NC DEQ | ~2K | 5 PFAS (GenX focus) | Excel/CSV | Public |
| WQP | ~35K | 13 PFAS | REST API | Public |
| MO DNR | ~76K | 29 PFAS | ArcGIS REST | Public |
| OH EPA | 26,554 | 6 PFAS | ArcGIS REST | Public; excluded from the merged dataset (PWSID overlap with SDWIS) |
| WA DOH | ~9.3K | 14 PFAS | ArcGIS REST | Public |
| TRI PFAS | ~2.5K facilities | PFAS releases | CSV | Public |
| DoD PFAS | ~700 sites | Military PFAS sites | CSV | Public |
| NJ Private Wells | — | PFAS, metals | CSV | Unavailable (provider removed bulk download) |
| EJScreen | ~220K block groups | Demographics, EJ indices | CSV | Public |
| NLCD | CONUS raster | Land cover classes | GeoTIFF (~1 GB) | Public |
| USGS Aquifers | CONUS polygons | Aquifer properties | Shapefile | Public |

Record counts are indicative snapshots; `docs/DATA_TERMS.md` is the authoritative
per-source attribution, terms, and redistribution record.

## Directory Structure

```
data/
  raw/           # Immutable downloaded files
  interim/       # Intermediate processed files
    merged_wq.parquet
    features/
      proximity.parquet
      land_use.parquet
      hydrogeology.parquet
      demographics.parquet
      dod_proximity.parquet
      tri_proximity.parquet
      system_characteristics.parquet
  processed/     # Final ready-to-use datasets
  checksums.sha256
```

## Download

Use the reproducibility script to download all sources:

```bash
python scripts/reproduce.py --download-only
python scripts/reproduce.py --download-only --skip-large  # Skip NLCD raster
```

Or download individual sources via Python:

```python
from pathlib import Path
from aquacontam.data.ucmr5 import UCMR5Source
source = UCMR5Source(
    raw_dir=Path("data/raw"),
    interim_dir=Path("data/interim"),
    processed_dir=Path("data/processed"),
)
parquet_path = source.run()
```

## Notes

- UCMR data files use Latin-1 encoding (not UTF-8)
- PWSID is always a 9-character string — never cast to int
- All coordinates stored in EPSG:4326 (WGS 84)
- Distance calculations use EPSG:5070 (CONUS Albers Equal Area)
