"""Ohio EPA PFAS sampling results data source.

Downloads PFAS sampling results from the Ohio EPA ArcGIS MapServer.
Results table (layer 2) has analyte-level records; systems layer (layer 0)
has coordinates for joining.

Data is in long format with explicit PWSID, analyte name, concentration
in NG/L, and a ``lessthandetect`` flag (Y/N).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from aquacontam._config import load_data_config
from aquacontam._constants import PPT_TO_UGL
from aquacontam.data._arcgis import fetch_arcgis_features
from aquacontam.data.base import DataSource
from aquacontam.data.schema import validate_schema

logger = logging.getLogger(__name__)

# Ohio EPA analyte names → standardized names
_ANALYTE_MAP: dict[str, str] = {
    "PERFLUOROCTANOIC ACID (PFOA)": "PFOA",
    "PERFLUOROCTANE SULFONIC ACID (PFOS)": "PFOS",
    "PERFLUOROHEXANE SULFONIC ACID (PFHxS)": "PFHxS",
    "PERFLUOROBUTANE SULFONIC ACID (PFBS)": "PFBS",
    "PERFLUORONONANOIC ACID (PFNA)": "PFNA",
    "HEXAFLUOROPROPYLENE OXIDE DIMER ACID (HFPO-DA)": "HFPO-DA",
}

OH_EPA_ANALYTES: tuple[str, ...] = tuple(sorted(_ANALYTE_MAP.values()))
"""Ohio EPA PFAS compounds."""


class OhEpaSource(DataSource):
    """Ohio EPA PFAS sampling data loader.

    Downloads PFAS sampling results from the Ohio EPA ArcGIS MapServer.
    The results table has ~26K records with PWSID, analyte, concentration,
    and detection indicator. Coordinates come from a separate systems layer.

    Parameters
    ----------
    raw_dir : Path
        Directory for raw downloaded files.
    interim_dir : Path
        Directory for intermediate artifacts.
    processed_dir : Path
        Directory for final Parquet output.
    config_path : Path | str | None
        Path to the data config YAML.
    """

    def __init__(
        self,
        raw_dir: Path,
        interim_dir: Path,
        processed_dir: Path,
        config_path: Path | str | None = None,
    ) -> None:
        super().__init__(raw_dir, interim_dir, processed_dir)
        self._config = load_data_config(config_path, source_name="oh_epa")

    @property
    def name(self) -> str:
        return "oh_epa"

    def download(self, *, force: bool = False, **kwargs: Any) -> list[Path]:
        """Download OH EPA PFAS data from ArcGIS REST API."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        dest = self.raw_dir / self._config["expected_files"][0]

        if dest.exists() and not force:
            logger.info("OH EPA data already exists: %s", dest)
            return [dest]

        if self._config.get("unavailable", False):
            reason = self._config.get("unavailable_reason", "source unavailable")
            raise RuntimeError(f"OH EPA download skipped: {reason}")

        base_url = self._config["arcgis_base_url"]
        results_layer = self._config.get("results_layer_id", 2)
        systems_layer = self._config.get("systems_layer_id", 0)

        # Download results table (no geometry)
        logger.info("Fetching OH EPA PFAS results (layer %d)…", results_layer)
        results = fetch_arcgis_features(base_url, results_layer)
        if not results:
            raise RuntimeError("No features returned from OH EPA results layer")

        # Download systems layer (has point geometry → lat/lon)
        logger.info("Fetching OH EPA systems (layer %d)…", systems_layer)
        systems = fetch_arcgis_features(base_url, systems_layer)

        # Build PWSID → (lat, lon) lookup from systems layer
        coord_lookup: dict[str, tuple[float, float]] = {}
        for sys_feat in systems:
            pwsid = str(sys_feat.get("pwsid", "")).strip()
            lat = sys_feat.get("_latitude")
            lon = sys_feat.get("_longitude")
            if pwsid and lat is not None and lon is not None:
                coord_lookup[pwsid] = (float(lat), float(lon))

        # Merge coordinates into results
        for feat in results:
            pwsid = str(feat.get("pwsid", "")).strip()
            if pwsid in coord_lookup:
                feat["_latitude"], feat["_longitude"] = coord_lookup[pwsid]

        df = pd.DataFrame(results)
        df.to_csv(dest, index=False)
        logger.info("Saved %d OH EPA records to %s", len(df), dest)
        return [dest]

    def parse(self, **kwargs: Any) -> pd.DataFrame:
        """Parse OH EPA CSV into standardized schema."""
        csv_path = self.raw_dir / self._config["expected_files"][0]
        df = pd.read_csv(csv_path, dtype=str, low_memory=False)
        logger.info("Read %d rows from OH EPA", len(df))

        result = pd.DataFrame(index=df.index)

        # 1. PWSID — already in OH format (e.g. "OH7762812")
        result["pwsid"] = df.get("pwsid", pd.Series("", index=df.index)).astype(str).str.strip()

        # 2. Analyte — standardize names
        raw_analyte = df.get("analyte", pd.Series("", index=df.index)).astype(str).str.strip()
        result["analyte"] = raw_analyte.map(lambda x: _ANALYTE_MAP.get(x, x))

        # 3. Concentration — parse from string, convert NG/L → ug/L
        conc_raw = pd.to_numeric(
            df.get("sample_result", pd.Series(dtype=float)).astype(str).str.strip(),
            errors="coerce",
        )
        conc_raw = conc_raw.fillna(0.0)

        # 4. Censoring — lessthandetect flag
        lt_flag = df.get("lessthandetect", pd.Series("", index=df.index)).astype(str).str.strip()
        censored_mask = lt_flag.str.upper() == "Y"

        # Also flag zero-concentration as censored
        censored_mask = censored_mask | (conc_raw == 0.0)
        result["censored"] = censored_mask

        # 5. Unit conversion NG/L → ug/L
        conc_ugl = conc_raw * PPT_TO_UGL
        conc_ugl[censored_mask] = 0.0
        result["concentration"] = conc_ugl

        # 6. Detection limit
        dl_ugl = conc_raw * PPT_TO_UGL
        dl_ugl[censored_mask & (conc_raw == 0.0)] = np.nan
        dl_ugl[~censored_mask] = conc_ugl[~censored_mask]
        result["detection_limit"] = dl_ugl

        result["unit"] = "ug/L"

        # 7. Coordinates
        lat_col = "_latitude" if "_latitude" in df.columns else "latitude"
        lon_col = "_longitude" if "_longitude" in df.columns else "longitude"
        result["latitude"] = pd.to_numeric(
            df.get(lat_col, pd.Series(dtype=float)), errors="coerce"
        )
        result["longitude"] = pd.to_numeric(
            df.get(lon_col, pd.Series(dtype=float)), errors="coerce"
        )

        # 8. Sample date
        if "samp_date" in df.columns:
            result["sample_date"] = pd.to_datetime(df["samp_date"].astype(str), errors="coerce")
        else:
            result["sample_date"] = pd.NaT

        # Drop rows with empty analyte
        result = result[result["analyte"].str.len() > 0]

        # Deterministic row ordering
        sort_cols = [c for c in ["pwsid", "analyte", "sample_date"] if c in result.columns]
        if sort_cols:
            result = result.sort_values(by=sort_cols).reset_index(drop=True)
        else:
            result = result.reset_index(drop=True)

        logger.info("Parsed %d OH EPA samples", len(result))
        return cast(pd.DataFrame, result)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OH EPA data."""
        has_coords = df["latitude"].notna().any() if "latitude" in df.columns else False
        df = validate_schema(df, skip_coord_check=not has_coords, drop_invalid_rows=True)

        if "analyte" in df.columns:
            unique = set(df["analyte"].unique())
            expected = set(OH_EPA_ANALYTES)
            unexpected = unique - expected
            if unexpected:
                logger.warning("Unexpected OH EPA analytes: %s", sorted(unexpected))

        return df
