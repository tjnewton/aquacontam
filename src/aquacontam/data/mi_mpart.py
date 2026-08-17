"""Michigan MPART PFAS data source — PFAS compliance monitoring.

Downloads PFAS sampling results from the Michigan EGLE PFAS Open Data
ArcGIS REST API (Layer 3: PFAS Compliance Monitoring table) and maps
them to the standard water quality schema.

The data is in wide format (one column per analyte). Values are numeric
strings in PPT (ng/L) or "ND" for non-detect. System ID is WSSN
(Water Supply Serial Number), which is prefixed with "MI" and zero-padded
to form a 9-character PWSID.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from aquacontam._config import load_data_config
from aquacontam._constants import (
    MI_MPART_ANALYTES,
    PPT_TO_UGL,
)
from aquacontam.data._arcgis import fetch_arcgis_features
from aquacontam.data._utils import find_col, normalize_pwsid
from aquacontam.data.base import DataSource
from aquacontam.data.schema import validate_schema

logger = logging.getLogger(__name__)

# Map ArcGIS column names → standard analyte names
_WIDE_COL_MAP: dict[str, str] = {
    "HFPODA": "HFPO-DA",
    "PFBS": "PFBS",
    "PFHxA": "PFHxA",
    "PFHxS": "PFHxS",
    "PFNA": "PFNA",
    "PFOA": "PFOA",
    "PFOS": "PFOS",
}


class MiMpartSource(DataSource):
    """Michigan MPART PFAS data loader.

    Downloads sampling results from the Michigan EGLE PFAS Open Data
    ArcGIS REST API. Monitors 7+ PFAS compounds at public water supplies.

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
        self._config = load_data_config(config_path, source_name="mi_mpart")

    @property
    def name(self) -> str:
        return "mi_mpart"

    def download(self, *, force: bool = False, **kwargs: Any) -> list[Path]:
        """Download MI MPART data from ArcGIS REST API."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        dest = self.raw_dir / self._config["expected_files"][0]

        if dest.exists() and not force:
            logger.info("MI MPART data already exists: %s", dest)
            return [dest]

        if self._config.get("unavailable", False):
            reason = self._config.get("unavailable_reason", "source unavailable")
            raise RuntimeError(f"MI MPART download skipped: {reason}")

        base_url = self._config["arcgis_base_url"]
        layer_id = self._config.get("layer_id", 3)
        features = fetch_arcgis_features(base_url, layer_id)

        if not features:
            raise RuntimeError("No features returned from MI MPART ArcGIS API")

        df = pd.DataFrame(features)
        df.to_csv(dest, index=False)
        logger.info("Saved %d MI MPART records to %s", len(df), dest)
        return [dest]

    def parse(self, **kwargs: Any) -> pd.DataFrame:
        """Parse MI MPART CSV into standardized schema.

        The new EGLE ArcGIS data comes in wide format (one column per
        analyte) from Layer 3 (PFAS Compliance Monitoring table). Also
        supports the legacy long-format data (with Contaminant/Result
        columns) for backward compatibility with existing test fixtures.
        """
        csv_path = self.raw_dir / self._config["expected_files"][0]
        df = pd.read_csv(csv_path, dtype=str, low_memory=False)
        logger.info("Read %d rows from MI MPART", len(df))

        # Detect format: wide (PFOA/PFOS columns) vs long (Contaminant/Result)
        has_wide_cols = any(col in df.columns for col in _WIDE_COL_MAP)
        has_long_cols = any(
            col in df.columns for col in ["Contaminant", "Analyte", "Parameter", "analyte"]
        )

        if has_wide_cols and not has_long_cols:
            return self._parse_wide(df)
        return self._parse_long(df)

    def _parse_wide(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse wide-format data (one column per analyte).

        Used for the current EGLE ArcGIS Layer 3 data where each PFAS
        compound has its own column with values like "ND", "15.0", or null.
        """
        # Find WSSN or PWSID column
        wssn_col = find_col(df, ["WSSN", "PWSID", "PWSId", "SystemID"])
        date_col = find_col(df, ["SampleDate", "CollectionDate", "sample_date"], required=False)

        # Detect which analyte columns are present
        analyte_cols = {
            std_name: col_name
            for col_name, std_name in _WIDE_COL_MAP.items()
            if col_name in df.columns
        }

        rows: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            # PWSID from WSSN (integer → "MI" + zero-padded)
            raw_id = str(row.get(wssn_col, "")).strip()
            pwsid = normalize_pwsid(raw_id, "MI")

            # Sample date — may be epoch ms or date string
            sample_date = pd.NaT  # type: ignore[assignment]
            if date_col and pd.notna(row.get(date_col)):
                raw_date = str(row[date_col]).strip()
                # ArcGIS epoch milliseconds
                try:
                    epoch_ms = float(raw_date)
                    sample_date = pd.Timestamp(epoch_ms, unit="ms")  # type: ignore[assignment]
                except (ValueError, OverflowError):
                    sample_date = pd.to_datetime(raw_date, errors="coerce")  # type: ignore[assignment]

            for col_name, std_name in analyte_cols.items():
                raw_val = str(row.get(col_name, "")).strip()

                if raw_val in ("", "nan", "None"):
                    # No data for this analyte in this sample — skip
                    continue

                # Parse concentration
                is_nd = raw_val.upper() in ("ND", "U", "<", "NON-DETECT")
                if is_nd:
                    conc_ppt = 0.0
                    censored = True
                else:
                    conc_ppt = pd.to_numeric(raw_val, errors="coerce")
                    if pd.isna(conc_ppt):
                        continue  # Unparseable value — skip
                    conc_ppt = float(conc_ppt)
                    censored = conc_ppt == 0.0

                conc_ugl = conc_ppt * PPT_TO_UGL
                # Use concentration as DL proxy when > 0; NaN otherwise
                dl_ugl = conc_ugl if conc_ugl > 0 else float("nan")

                # Zero out censored concentrations
                if censored:
                    conc_ugl = 0.0

                rows.append(
                    {
                        "pwsid": pwsid,
                        "analyte": std_name,
                        "concentration": conc_ugl,
                        "unit": "ug/L",
                        "censored": censored,
                        "detection_limit": dl_ugl,
                        "sample_date": sample_date,
                        # No coordinates in Layer 3 table
                        "latitude": float("nan"),
                        "longitude": float("nan"),
                    }
                )

        result = (
            pd.DataFrame(rows)
            if rows
            else pd.DataFrame(
                columns=[
                    "pwsid",
                    "analyte",
                    "concentration",
                    "unit",
                    "censored",
                    "detection_limit",
                    "sample_date",
                    "latitude",
                    "longitude",
                ]
            )
        )
        result = result[result["analyte"].str.len() > 0]

        # Deterministic row ordering
        sort_cols = [c for c in ["pwsid", "analyte", "sample_date"] if c in result.columns]
        if sort_cols:
            result = result.sort_values(by=sort_cols).reset_index(drop=True)
        else:
            result = result.reset_index(drop=True)

        logger.info("Parsed %d MI MPART samples (wide format)", len(result))
        return cast(pd.DataFrame, result)

    def _parse_long(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse long-format data (one row per analyte per sample).

        Used for legacy CSV exports and test fixtures that have
        Contaminant/Result/Qualifier columns.
        """
        pwsid_col = find_col(df, ["PWSID", "PWSId", "pws_id", "SystemID"])
        analyte_col = find_col(df, ["Contaminant", "Analyte", "Parameter", "analyte"])
        conc_col = find_col(df, ["Result", "Concentration", "concentration", "ResultValue"])
        date_col = find_col(
            df, ["SampleDate", "CollectionDate", "sample_date", "SampleCollectionDate"]
        )
        qualifier_col = find_col(
            df, ["Qualifier", "ResultQualifier", "DetectionFlag", "qualifier"], required=False
        )

        result = pd.DataFrame(index=df.index)

        # 1. PWSID normalization
        result["pwsid"] = (
            df[pwsid_col].astype(str).apply(lambda x: normalize_pwsid(x, "MI"))
            if pwsid_col
            else ""
        )

        # 2. Analyte
        result["analyte"] = df[analyte_col].astype(str) if analyte_col else ""

        # 3. Concentration (numeric coerce, fill NaN with 0)
        conc = (
            pd.to_numeric(df[conc_col].astype(str), errors="coerce").fillna(0.0)
            if conc_col
            else pd.Series(0.0, index=df.index)
        )

        # 4. Censoring (qualifier check + zero-concentration check)
        if qualifier_col:
            qual_upper = df[qualifier_col].astype(str).str.strip().str.upper()
            censored_by_qual = (
                qual_upper.isin({"ND", "U", "<", "NON-DETECT"}) & df[qualifier_col].notna()
            )
        else:
            censored_by_qual = pd.Series(False, index=df.index)
        censored_by_zero = (conc == 0.0) & ~censored_by_qual
        censored_mask = censored_by_qual | censored_by_zero
        result["censored"] = censored_mask

        # 5. Detection limit
        dl = pd.Series(np.nan, index=df.index)
        dl[censored_mask & (conc > 0)] = conc[censored_mask & (conc > 0)]
        dl[~censored_mask] = conc[~censored_mask]

        # 6. Zero out censored concentrations
        conc = conc.copy()
        conc[censored_mask] = 0.0

        # 7. PPT → ug/L conversion
        result["concentration"] = conc * PPT_TO_UGL
        dl_ugl = dl * PPT_TO_UGL
        dl_ugl[(dl.isna()) | (dl <= 0)] = np.nan
        result["detection_limit"] = dl_ugl

        result["unit"] = "ug/L"

        # 8. Coordinates
        lat_col = "_latitude" if "_latitude" in df.columns else "Latitude"
        lon_col = "_longitude" if "_longitude" in df.columns else "Longitude"
        result["latitude"] = pd.to_numeric(
            df.get(lat_col, pd.Series(dtype=float)), errors="coerce"
        )
        result["longitude"] = pd.to_numeric(
            df.get(lon_col, pd.Series(dtype=float)), errors="coerce"
        )

        # 9. Sample date
        result["sample_date"] = (
            pd.to_datetime(df[date_col].astype(str), errors="coerce") if date_col else pd.NaT
        )
        # Drop rows with empty analyte
        result = result[result["analyte"].str.len() > 0]

        # Deterministic row ordering
        sort_cols = [c for c in ["pwsid", "analyte", "sample_date"] if c in result.columns]
        if sort_cols:
            result = result.sort_values(by=sort_cols).reset_index(drop=True)
        else:
            result = result.reset_index(drop=True)

        logger.info("Parsed %d MI MPART samples (long format)", len(result))
        return cast(pd.DataFrame, result)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate MI MPART data."""
        # Layer 3 table has no coordinates; skip coord check
        has_coords = df["latitude"].notna().any() if "latitude" in df.columns else False
        df = validate_schema(df, skip_coord_check=not has_coords, drop_invalid_rows=True)

        if "analyte" in df.columns:
            unique = set(df["analyte"].unique())
            expected = set(MI_MPART_ANALYTES)
            unexpected = unique - expected
            if unexpected:
                logger.warning("Unexpected MI MPART analytes: %s", sorted(unexpected))

        return df
