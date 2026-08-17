"""North Carolina DEQ PFAS data source.

Downloads PFAS sampling results from the NC DEQ Division of Water
Resources. Focus on GenX (HFPO-DA) from the Chemours Fayetteville
Works facility, plus PFOA, PFOS, PFNA, and PFHxS.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import pandas as pd

from aquacontam._config import load_data_config
from aquacontam._constants import (
    DOWNLOAD_TIMEOUT_DEFAULT,
    NC_DEQ_ANALYTES,
    PPT_TO_UGL,
)
from aquacontam.data._download import download_and_extract
from aquacontam.data._utils import find_col, normalize_pwsid
from aquacontam.data.base import DataSource
from aquacontam.data.schema import validate_schema

logger = logging.getLogger(__name__)


class NcDeqSource(DataSource):
    """North Carolina DEQ PFAS data loader.

    Downloads PFAS sampling results from the NC DEQ website. Data may
    be Excel (.xlsx) or CSV format. Covers 533+ small water systems
    (2023) and 228+ systems (2024), with a focus on GenX (HFPO-DA).

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
        self._config = load_data_config(config_path, source_name="nc_deq")

    @property
    def name(self) -> str:
        return "nc_deq"

    def download(self, *, force: bool = False, **kwargs: Any) -> list[Path]:
        """Download NC DEQ PFAS data.

        Raises
        ------
        RuntimeError
            If the source is marked unavailable in config (PDF-only).
        """
        if self._config.get("unavailable", False):
            reason = self._config.get("unavailable_reason", "source unavailable")
            raise RuntimeError(f"NC DEQ download skipped: {reason}")
        return download_and_extract(
            self._config["url"],
            self.raw_dir,
            self._config["expected_files"],
            force=force,
            timeout=DOWNLOAD_TIMEOUT_DEFAULT,
            progress_desc="NC DEQ download",
        )

    def parse(self, **kwargs: Any) -> pd.DataFrame:
        """Parse NC DEQ Excel/CSV into standardized schema."""
        expected_file = self._config["expected_files"][0]
        filepath = self.raw_dir / expected_file

        if filepath.suffix in (".xlsx", ".xls"):
            try:
                df = pd.read_excel(filepath, dtype=str)
            except ValueError:
                # File may actually be CSV despite .xlsx extension
                logger.info("Excel read failed, trying CSV for %s", filepath)
                df = pd.read_csv(filepath, dtype=str, low_memory=False)
        else:
            df = pd.read_csv(filepath, dtype=str, low_memory=False)
        logger.info("Read %d rows from NC DEQ", len(df))

        # Identify columns
        pwsid_col = find_col(df, ["PWSID", "PWS_ID", "SystemID", "System ID"])
        analyte_col = find_col(
            df, ["Analyte", "Contaminant", "Parameter", "Chemical"], required=False
        )
        conc_col = find_col(df, ["Result", "Concentration", "Value", "Level"], required=False)
        date_col = find_col(
            df, ["Sample Date", "SampleDate", "Collection Date", "Date"], required=False
        )
        lat_col = find_col(df, ["Latitude", "LAT", "Y"], required=False)
        lon_col = find_col(df, ["Longitude", "LON", "LONG", "X"], required=False)

        # Check if data is wide format (one column per analyte)
        pfas_cols = _detect_pfas_columns(df)

        if pfas_cols and not analyte_col:
            # Wide format → melt
            rows = []
            for _, row in df.iterrows():
                raw_pwsid = str(row.get(pwsid_col, "")).strip() if pwsid_col else ""
                pwsid = normalize_pwsid(raw_pwsid, "NC")

                lat = (
                    pd.to_numeric(str(row.get(lat_col, "")), errors="coerce")
                    if lat_col
                    else float("nan")
                )
                lon = (
                    pd.to_numeric(str(row.get(lon_col, "")), errors="coerce")
                    if lon_col
                    else float("nan")
                )
                sample_date = (
                    pd.to_datetime(str(row.get(date_col, "")), errors="coerce")
                    if date_col
                    else pd.NaT
                )

                for analyte_name, col_name in pfas_cols.items():
                    conc_raw = pd.to_numeric(str(row.get(col_name, "")), errors="coerce")
                    conc = float(conc_raw) if pd.notna(conc_raw) else 0.0
                    censored = pd.isna(conc_raw) or conc == 0.0

                    # NC data typically in PPT (ng/L)
                    conc_ugl = conc * PPT_TO_UGL
                    # Censored rows with zero concentration have no usable DL;
                    # NaN here causes intentional drop during schema validation.
                    dl_ugl = conc_ugl if conc_ugl > 0 else float("nan")

                    rows.append(
                        {
                            "pwsid": pwsid,
                            "analyte": analyte_name,
                            "concentration": conc_ugl,
                            "unit": "ug/L",
                            "censored": censored,
                            "detection_limit": dl_ugl,
                            "sample_date": sample_date,
                            "latitude": lat,
                            "longitude": lon,
                        }
                    )
            result = pd.DataFrame(rows)
        else:
            # Long format
            rows = []
            for _, row in df.iterrows():
                raw_pwsid = str(row.get(pwsid_col, "")).strip() if pwsid_col else ""
                pwsid = normalize_pwsid(raw_pwsid, "NC")

                analyte = str(row.get(analyte_col, "")).strip() if analyte_col else ""
                conc_raw = (
                    pd.to_numeric(str(row.get(conc_col, "")), errors="coerce") if conc_col else 0.0
                )
                conc = float(conc_raw) if pd.notna(conc_raw) else 0.0
                censored = pd.isna(conc_raw) or conc == 0.0

                conc_ugl = conc * PPT_TO_UGL
                # Censored rows with zero concentration have no usable DL;
                # NaN here causes intentional drop during schema validation.
                dl_ugl = conc_ugl if conc_ugl > 0 else float("nan")

                lat = (
                    pd.to_numeric(str(row.get(lat_col, "")), errors="coerce")
                    if lat_col
                    else float("nan")
                )
                lon = (
                    pd.to_numeric(str(row.get(lon_col, "")), errors="coerce")
                    if lon_col
                    else float("nan")
                )
                sample_date = (
                    pd.to_datetime(str(row.get(date_col, "")), errors="coerce")
                    if date_col
                    else pd.NaT
                )

                rows.append(
                    {
                        "pwsid": pwsid,
                        "analyte": analyte,
                        "concentration": conc_ugl,
                        "unit": "ug/L",
                        "censored": censored,
                        "detection_limit": dl_ugl,
                        "sample_date": sample_date,
                        "latitude": lat,
                        "longitude": lon,
                    }
                )
            result = pd.DataFrame(rows)

        result = result[result["analyte"].str.len() > 0]

        # Deterministic row ordering for reproducibility across filesystems
        sort_cols = [c for c in ["pwsid", "analyte", "sample_date"] if c in result.columns]
        if sort_cols:
            result = result.sort_values(by=sort_cols).reset_index(drop=True)
        else:
            result = result.reset_index(drop=True)

        logger.info("Parsed %d NC DEQ samples", len(result))
        return cast(pd.DataFrame, result)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate NC DEQ data."""
        # NC DEQ data often lacks geocoded coordinates
        df = validate_schema(df, skip_coord_check=True, drop_invalid_rows=True)

        if "analyte" in df.columns:
            unique = set(df["analyte"].unique())
            expected = set(NC_DEQ_ANALYTES)
            unexpected = unique - expected
            if unexpected:
                logger.warning("Unexpected NC DEQ analytes: %s", sorted(unexpected))

        return df


def _detect_pfas_columns(df: pd.DataFrame) -> dict[str, str]:
    """Detect PFAS analyte columns in the dataframe.

    Returns mapping of standardized analyte name → actual column name.
    """
    pfas_patterns = {
        "HFPO-DA": ["HFPO-DA", "GenX", "HFPODA", "hfpo"],
        "PFOA": ["PFOA", "pfoa"],
        "PFOS": ["PFOS", "pfos"],
        "PFNA": ["PFNA", "pfna"],
        "PFHxS": ["PFHxS", "pfhxs"],
    }
    found: dict[str, str] = {}
    for analyte, patterns in pfas_patterns.items():
        for pat in patterns:
            matches = [c for c in df.columns if pat.lower() in c.lower()]
            if matches:
                found[analyte] = matches[0]
                break
    return found
