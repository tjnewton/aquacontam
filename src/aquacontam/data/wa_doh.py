"""Washington State DOH PFAS drinking water testing data source.

Downloads PFAS testing results from the Washington State Department of
Health auto-published CSV. The dataset covers ~2,400 public water systems
with ~247K sample records.

WA DOH uses Water System IDs (WRIS numbers) prefixed with "WA" for PWSID
normalization. The CSV has Excel-escaped IDs (e.g., ``="60850"``).
Concentrations are in ng/L. Result column contains "None Detected" or
"Detected but lower than state action level (SAL)" etc.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from aquacontam._config import load_data_config
from aquacontam._constants import DOWNLOAD_TIMEOUT_DEFAULT, PPT_TO_UGL
from aquacontam.data._download import download_and_extract
from aquacontam.data._utils import normalize_pwsid
from aquacontam.data.base import DataSource
from aquacontam.data.schema import validate_schema

logger = logging.getLogger(__name__)

# Map parenthesized analyte names to standard names
_ANALYTE_RE = re.compile(r"\((\w+(?:-\w+)?)\)")
_ANALYTE_MAP: dict[str, str] = {
    "PFBS": "PFBS",
    "PFHpA": "PFHpA",
    "PFHxS": "PFHxS",
    "PFNA": "PFNA",
    "PFOS": "PFOS",
    "PFOA": "PFOA",
    "PFHxA": "PFHxA",
    "PFDA": "PFDA",
    "PFPeA": "PFPeA",
    "PFUnA": "PFUnA",
    "HFPO-DA": "HFPO-DA",
    "ADONA": "ADONA",
    "NEtFOSAA": "NEtFOSAA",
    "NMeFOSAA": "NMeFOSAA",
}

WA_DOH_ANALYTES: tuple[str, ...] = tuple(sorted(_ANALYTE_MAP.values()))
"""Washington DOH PFAS compounds."""


def _clean_excel_id(val: str) -> str:
    """Strip Excel escape wrapper from IDs like ``="60850"``."""
    val = val.strip()
    if val.startswith('="') and val.endswith('"'):
        return val[2:-1]
    return val


def _extract_analyte(measure: str) -> str:
    """Extract standard analyte name from WA DOH PFAS Measure column.

    Examples:
        "(PFBS) Perfluorobutanesulfonic acid" → "PFBS"
        "(HFPO-DA) Hexafluoropropylene oxide dimer acid" → "HFPO-DA"
    """
    m = _ANALYTE_RE.search(measure)
    if m:
        abbrev = m.group(1)
        return _ANALYTE_MAP.get(abbrev, abbrev)
    return measure.strip()


class WaDohSource(DataSource):
    """Washington State DOH PFAS data loader.

    Downloads PFAS testing results from the WA DOH auto-published CSV.
    Covers ~2,400 public water systems across Washington state
    (EPA Region 10 — test split).

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
        self._config = load_data_config(config_path, source_name="wa_doh")

    @property
    def name(self) -> str:
        return "wa_doh"

    def download(self, *, force: bool = False, **kwargs: Any) -> list[Path]:
        """Download WA DOH PFAS CSV."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        dest = self.raw_dir / self._config["expected_files"][0]

        if dest.exists() and not force:
            logger.info("WA DOH data already exists: %s", dest)
            return [dest]

        if self._config.get("unavailable", False):
            reason = self._config.get("unavailable_reason", "source unavailable")
            raise RuntimeError(f"WA DOH download skipped: {reason}")

        return download_and_extract(
            self._config["url"],
            self.raw_dir,
            self._config["expected_files"],
            force=force,
            timeout=DOWNLOAD_TIMEOUT_DEFAULT,
            progress_desc="WA DOH PFAS download",
        )

    def parse(self, **kwargs: Any) -> pd.DataFrame:
        """Parse WA DOH CSV into standardized schema."""
        csv_path = self.raw_dir / self._config["expected_files"][0]
        df = pd.read_csv(csv_path, dtype=str, low_memory=False)
        logger.info("Read %d rows from WA DOH", len(df))

        result = pd.DataFrame(index=df.index)

        # 1. PWSID — Water System ID with Excel escaping
        raw_id = df.get("Water System ID", pd.Series("", index=df.index)).astype(str)
        result["pwsid"] = raw_id.apply(lambda x: normalize_pwsid(_clean_excel_id(x), "WA"))

        # 2. Analyte — extract abbreviation from "PFAS Measure" column
        raw_measure = df.get("PFAS Measure", pd.Series("", index=df.index)).astype(str)
        result["analyte"] = raw_measure.apply(_extract_analyte)

        # 3. Result / concentration
        raw_result = df.get("Result", pd.Series("", index=df.index)).astype(str).str.strip()
        is_non_detect = raw_result.str.contains("None Detected", case=False, na=True)

        # Parse PFAS Level column (e.g., "2 ng/L", "None Detected")
        raw_level = df.get("PFAS Level", pd.Series("", index=df.index)).astype(str).str.strip()
        conc_ngl = pd.to_numeric(
            raw_level.str.replace(r"\s*ng/L\s*", "", regex=True),
            errors="coerce",
        )
        conc_ngl = conc_ngl.fillna(0.0)

        # 4. Censoring
        censored_mask = is_non_detect | (conc_ngl == 0.0)
        result["censored"] = censored_mask

        # 5. Unit conversion ng/L → ug/L
        conc_ugl = conc_ngl * PPT_TO_UGL
        conc_ugl[censored_mask] = 0.0
        result["concentration"] = conc_ugl

        # 6. Detection limit — use concentration as proxy for detected rows
        dl_ugl = pd.Series(np.nan, index=df.index)
        dl_ugl[~censored_mask] = conc_ugl[~censored_mask]
        result["detection_limit"] = dl_ugl

        result["unit"] = "ug/L"

        # 7. Coordinates — WA DOH CSV has no lat/lon columns
        result["latitude"] = np.nan
        result["longitude"] = np.nan

        # 8. Sample date
        raw_date = df.get("Testing Date", pd.Series("", index=df.index)).astype(str)
        result["sample_date"] = pd.to_datetime(raw_date, errors="coerce")

        # Drop rows with empty analyte
        result = result[result["analyte"].str.len() > 0]

        # Deterministic row ordering
        sort_cols = [c for c in ["pwsid", "analyte", "sample_date"] if c in result.columns]
        if sort_cols:
            result = result.sort_values(by=sort_cols).reset_index(drop=True)
        else:
            result = result.reset_index(drop=True)

        logger.info("Parsed %d WA DOH samples", len(result))
        return cast(pd.DataFrame, result)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate WA DOH data."""
        # WA DOH CSV has no coordinates
        df = validate_schema(df, skip_coord_check=True, drop_invalid_rows=True)

        if "analyte" in df.columns:
            unique = set(df["analyte"].unique())
            expected = set(WA_DOH_ANALYTES)
            unexpected = unique - expected
            if unexpected:
                logger.warning("Unexpected WA DOH analytes: %s", sorted(unexpected))

        return df
