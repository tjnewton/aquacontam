"""Missouri DNR PFAS data source — ArcGIS FeatureServer.

Downloads PFAS sampling results from the Missouri DNR PFAS Viewer
ArcGIS REST API and maps them to the standard water quality schema.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from aquacontam._config import load_data_config
from aquacontam._constants import MO_DNR_ANALYTES
from aquacontam.data._arcgis import fetch_arcgis_features
from aquacontam.data._utils import convert_to_ugl, normalize_pwsid
from aquacontam.data.base import DataSource
from aquacontam.data.schema import validate_schema

logger = logging.getLogger(__name__)

# Map verbose MO DNR analyte names to standard short codes.
# The parenthesized abbreviation is extracted first; fallback to exact match.
_MO_DNR_ANALYTE_MAP: dict[str, str] = {
    "11CL-PF3OUDS": "11Cl-PF3OUdS",
    "9CL-PF3ONS": "9Cl-PF3ONS",
    "ADONA": "ADONA",
    "HFPO-DA": "HFPO-DA",
    "NETFOSAA": "NEtFOSAA",
    "NMEFOSAA": "NMeFOSAA",
    "NONAFLUORO NFDHA": "NFDHA",
    "PERFLUORO PFEESA": "PFEESA",
    "PERFLUORO PFMBA": "PFMBA",
    "PERFLUORO PFMPA": "PFMPA",
    "PERFLUOROBUTANE SULFONIC ACID (PFBS)": "PFBS",
    "PERFLUOROBUTANOIC ACID (PFBA)": "PFBA",
    "PERFLUOROCTANE SULFONIC ACID (PFOS)": "PFOS",
    "PERFLUOROCTANOIC ACID (PFOA)": "PFOA",
    "PERFLUORODECANE SULFONIC ACID 8:2 FTS": "8:2 FTS",
    "PERFLUORODECANOIC ACID (PFDA)": "PFDA",
    "PERFLUORODODECANOIC ACID (PFDOA)": "PFDoA",
    "PERFLUOROHEPTANESULFONIC ACID (PFHPS)": "PFHpS",
    "PERFLUOROHEPTANOIC ACID (PFHPA)": "PFHpA",
    "PERFLUOROHEXANE SULFONIC ACID (PFHxS)": "PFHxS",
    "PERFLUOROHEXANE SULFONIC ACID 4:2 FTS": "4:2 FTS",
    "PERFLUOROHEXANOIC ACID (PFHXA)": "PFHxA",
    "PERFLUORONONANOIC ACID (PFNA)": "PFNA",
    "PERFLUOROOCTANE SULFONIC ACID 6:2 FTS": "6:2 FTS",
    "PERFLUOROPENTANESULFONIC ACID (PFPES)": "PFPeS",
    "PERFLUOROPENTANOIC ACID (PFPEA)": "PFPeA",
    "PERFLUOROTETRADECANOIC ACID (PFTA)": "PFTA",
    "PERFLUOROTRIDECANOIC ACID (PFTRDA)": "PFTrDA",
    "PERFLUOROUNDECANOIC ACID (PFUNA)": "PFUnA",
}


def _normalize_mo_dnr_analyte(name: str) -> str:
    """Normalize verbose MO DNR analyte names to standard short codes."""
    return _MO_DNR_ANALYTE_MAP.get(name, name)


class MoDnrSource(DataSource):
    """Missouri DNR PFAS sampling data loader.

    Downloads PFAS sampling results from the Missouri DNR PFAS Samples
    ArcGIS FeatureServer. Includes individual sample records with
    concentrations, censoring indicators, and coordinates.

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
        self._config = load_data_config(config_path, source_name="mo_dnr")

    @property
    def name(self) -> str:
        return "mo_dnr"

    def download(self, *, force: bool = False, **kwargs: Any) -> list[Path]:
        """Download MO DNR PFAS data from ArcGIS REST API."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        dest = self.raw_dir / self._config["expected_files"][0]

        if dest.exists() and not force:
            logger.info("MO DNR data already exists: %s", dest)
            return [dest]

        if self._config.get("unavailable", False):
            reason = self._config.get("unavailable_reason", "source unavailable")
            raise RuntimeError(f"MO DNR download skipped: {reason}")

        base_url = self._config["arcgis_base_url"]
        layer_id = self._config.get("layer_id", 0)
        features = fetch_arcgis_features(base_url, layer_id)

        if not features:
            raise RuntimeError("No features returned from MO DNR ArcGIS API")

        df = pd.DataFrame(features)
        df.to_csv(dest, index=False)
        logger.info("Saved %d MO DNR records to %s", len(df), dest)
        return [dest]

    def parse(self, **kwargs: Any) -> pd.DataFrame:
        """Parse MO DNR CSV into standardized schema.

        MO DNR data has explicit columns for PWSID, analyte, concentration,
        censoring indicator (``LESS_THAN_IND``), detection limit, and
        coordinates.
        """
        csv_path = self.raw_dir / self._config["expected_files"][0]
        df = pd.read_csv(csv_path, dtype={"PWSID": str}, low_memory=False)
        logger.info("Read %d rows from MO DNR", len(df))

        result = pd.DataFrame(index=df.index)

        # 1. PWSID normalization
        if "PWSID" in df.columns:
            result["pwsid"] = df["PWSID"].astype(str).apply(lambda x: normalize_pwsid(x, "MO"))
        else:
            result["pwsid"] = ""

        # 2. Analyte — normalize verbose names to standard short codes
        if "ANALYTE" in df.columns:
            result["analyte"] = (
                df["ANALYTE"].astype(str).str.strip().map(_normalize_mo_dnr_analyte)
            )
        else:
            result["analyte"] = ""

        # 3. Concentration (numeric coerce)
        conc = pd.to_numeric(df.get("CONCENTRAT", pd.Series(dtype=float)), errors="coerce")
        conc = conc.fillna(0.0)

        # 4. Unit — convert to ug/L
        unit_col = df.get("CONCEN_UOM", pd.Series("", index=df.index)).astype(str).str.strip()

        # 5. Censoring — LESS_THAN_IND "<" or "Y" means censored
        if "LESS_THAN_IND" in df.columns:
            lt_raw = df["LESS_THAN_IND"].astype(str).str.strip().str.upper()
            censored_mask = lt_raw.isin({"<", "Y"})
        else:
            censored_mask = pd.Series(False, index=df.index)

        # Also flag zero-concentration rows as censored
        censored_mask = censored_mask | (conc == 0.0)
        result["censored"] = censored_mask

        # 6. Detection limit
        dl = pd.to_numeric(df.get("DETECT_LMT", pd.Series(dtype=float)), errors="coerce")

        # 7. Unit conversion — apply per-row
        conc_ugl = pd.Series(0.0, index=df.index)
        dl_ugl = pd.Series(np.nan, index=df.index)
        for i in df.index:
            unit_str = str(unit_col.iloc[i] if i < len(unit_col) else "").upper()
            # Map common MO DNR unit strings
            if unit_str in ("NG/L", "PPT"):
                unit_str = "NG/L"
            elif unit_str in ("UG/L", ""):
                unit_str = "UG/L"
            conc_ugl.iloc[i] = convert_to_ugl(float(conc.iloc[i]), unit_str)
            if pd.notna(dl.iloc[i]):
                dl_ugl.iloc[i] = convert_to_ugl(float(dl.iloc[i]), unit_str)

        # Zero out censored concentrations
        conc_ugl[censored_mask] = 0.0

        result["concentration"] = conc_ugl
        result["detection_limit"] = dl_ugl
        result["unit"] = "ug/L"

        # 8. Coordinates — prefer geometry fields from ArcGIS, then explicit columns
        lat_col = "_latitude" if "_latitude" in df.columns else "LATITUDE"
        lon_col = "_longitude" if "_longitude" in df.columns else "LONGITUDE"
        result["latitude"] = pd.to_numeric(
            df.get(lat_col, pd.Series(dtype=float)), errors="coerce"
        )
        result["longitude"] = pd.to_numeric(
            df.get(lon_col, pd.Series(dtype=float)), errors="coerce"
        )

        # 9. Sample date
        if "COLLECT_DT" in df.columns:
            result["sample_date"] = pd.to_datetime(df["COLLECT_DT"].astype(str), errors="coerce")
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

        logger.info("Parsed %d MO DNR samples", len(result))
        return cast(pd.DataFrame, result)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate MO DNR data."""
        df = validate_schema(df, skip_coord_check=False, drop_invalid_rows=True)

        if "analyte" in df.columns:
            unique = set(df["analyte"].unique())
            expected = set(MO_DNR_ANALYTES)
            unexpected = unique - expected
            if unexpected:
                logger.warning("Unexpected MO DNR analytes: %s", sorted(unexpected))

        return df
