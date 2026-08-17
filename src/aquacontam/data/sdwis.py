"""SDWIS heavy metal data source — lead and copper.

Downloads the Safe Drinking Water Information System (SDWIS) bulk
data from EPA ECHO and extracts Lead and Copper Rule (LCR) sample
results along with public water system metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import pandas as pd

from aquacontam._config import load_data_config
from aquacontam._constants import (
    DOWNLOAD_TIMEOUT_LARGE,
    HEAVY_METAL_ANALYTES,
    MGL_TO_UGL,
    SDWIS_CODE_TO_ANALYTE,
    SDWIS_CONTAMINANT_CODES,
)
from aquacontam.data._download import download_and_extract
from aquacontam.data._utils import find_col
from aquacontam.data.base import DataSource

logger = logging.getLogger(__name__)


class SDWISSource(DataSource):
    """SDWIS heavy metal data loader.

    Downloads the SDWIS bulk ZIP (~457 MB) from EPA ECHO containing
    Lead and Copper Rule sampling results and public water system
    metadata. Extracts lead (PB90) and copper (CU90) 90th-percentile results.

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
        self._config = load_data_config(config_path, source_name="sdwis")

    @property
    def name(self) -> str:
        return "sdwis"

    def download(self, *, force: bool = False, **kwargs: Any) -> list[Path]:
        """Download SDWIS bulk ZIP from EPA ECHO."""
        # Note: buffers ZIP in memory (~457MB). Streaming deferred to future PR.
        return download_and_extract(
            self._config["url"],
            self.raw_dir,
            self._config["expected_files"],
            force=force,
            timeout=DOWNLOAD_TIMEOUT_LARGE,
            progress_desc="SDWIS download",
        )

    def parse(self, *, zero_handling: str = "keep", **kwargs: Any) -> pd.DataFrame:
        """Parse SDWIS LCR samples and system metadata into standardized schema.

        Reads ``SDWA_LCR_SAMPLES.csv`` for lead/copper 90th percentile
        results, joins with ``SDWA_PUB_WATER_SYSTEMS.csv`` for system
        metadata, and maps contaminant codes to analyte names.

        Parameters
        ----------
        zero_handling : {"keep", "drop"}, default "keep"
            How to treat LCR rows with ``SAMPLE_MEASURE == 0`` (~24% of lead
            rows form a literal-0 spike, distinct from the positive continuum;
            EPA reports non-detect / below-detection 90th-percentile results
            as 0).
            - ``"keep"`` (default): retain them as genuine non-detects
              (concentration 0 -> clear non-exceedance). Appropriate for the
              action-level exceedance tasks, which need non-detect systems as
              negatives.
            - ``"drop"``: exclude ``SAMPLE_MEASURE <= 0``. Reproduces the
              originally-frozen 694,419-row canonical parse that the published
              numbers were computed on (it excluded these rows). See
              ``paper/revision_ledger.md`` for the root-cause record.
        """
        if zero_handling not in ("keep", "drop"):
            raise ValueError(f"zero_handling must be 'keep' or 'drop', got {zero_handling!r}")
        fmt = self._config.get("format", {})
        encoding = fmt.get("encoding", "latin-1")
        dtype_overrides = fmt.get("dtype_overrides", {})

        # Read LCR samples
        samples_path = self.raw_dir / self._config["expected_files"][0]
        logger.info("Reading SDWIS LCR samples from %s …", samples_path)
        samples = pd.read_csv(
            samples_path,
            encoding=encoding,
            dtype=dtype_overrides,
            low_memory=False,
        )
        logger.info("Read %d LCR sample rows", len(samples))

        # Read system metadata
        systems_path = self.raw_dir / self._config["expected_files"][1]
        logger.info("Reading SDWIS system metadata from %s …", systems_path)
        systems = pd.read_csv(
            systems_path,
            encoding=encoding,
            dtype={"PWSID": str},
            low_memory=False,
        )
        logger.info("Read %d system rows", len(systems))

        # Filter samples to target contaminant codes
        target_codes = set(SDWIS_CONTAMINANT_CODES.values())
        col_map = self._config.get("column_map", {})

        code_col = find_col(samples, ["CONTAMINANT_CODE", col_map.get("CONTAMINANT_CODE", "")])
        samples[code_col] = samples[code_col].astype(str).str.strip()
        samples = samples[samples[code_col].isin(target_codes)].copy()
        logger.info("Filtered to %d rows for target contaminant codes", len(samples))

        # Map contaminant codes → analyte names
        samples["analyte"] = samples[code_col].map(SDWIS_CODE_TO_ANALYTE)

        # Map column names
        pwsid_col = find_col(samples, ["PWSID"])
        conc_col = find_col(samples, ["SAMPLE_MEASURE", "RESULT"])
        sign_col = find_col(samples, ["RESULT_SIGN_CODE"], required=False)
        date_col = find_col(samples, ["SAMPLE_COLLECTION_DATE"], required=False)
        unit_col = find_col(samples, ["UNIT_OF_MEASURE"], required=False)

        # Coerce concentration
        samples["concentration"] = pd.to_numeric(samples[conc_col], errors="coerce").fillna(0.0)

        # SAMPLE_MEASURE == 0 rows are EPA non-detect / below-detection
        # 90th-percentile results (a literal-0 spike, ~24% of lead rows, distinct
        # from the positive continuum). "drop" reproduces the originally-frozen
        # canonical parse (which excluded them); "keep" (default) retains them as
        # non-detect non-exceedances. See parse() docstring / revision_ledger.
        if zero_handling == "drop":
            n_before = len(samples)
            samples = samples[samples["concentration"] > 0].copy()
            logger.info(
                "zero_handling='drop': removed %d non-positive SAMPLE_MEASURE rows (%d -> %d)",
                n_before - len(samples),
                n_before,
                len(samples),
            )

        # Censored: RESULT_SIGN_CODE == "<" (less than)
        if sign_col:
            samples["censored"] = samples[sign_col].fillna("").str.strip() == "<"
        else:
            samples["censored"] = False

        # Detection limit: for censored rows the reported "<X" value X *is* the
        # reporting limit, so use it. For non-censored (detected) rows the true
        # reporting limit is unknown and lies below the measurement, so leave it
        # NaN — copying the measured concentration would leak the result, because
        # the T4 lead/copper action-level target is derived from these same
        # concentrations (referee report M1). SDWIS LCR is ~100% non-censored, so
        # mean_detection_limit is then ~all-NaN for SDWIS systems and is dropped at
        # feature assembly (>50% NaN); UCMR/PFAS detection_limit comes from a real
        # MRL column and is unaffected.
        samples["detection_limit"] = float("nan")
        if sign_col:
            censored_mask = samples["censored"]
            samples.loc[censored_mask, "detection_limit"] = samples.loc[
                censored_mask, "concentration"
            ]
            samples.loc[censored_mask, "concentration"] = 0.0

        # Unit normalization: SDWIS uses mg/L for LCR → convert to ug/L
        if unit_col:
            mg_mask = samples[unit_col].fillna("").str.upper().str.contains("MG/L")
            samples.loc[mg_mask, "concentration"] = (
                samples.loc[mg_mask, "concentration"] * MGL_TO_UGL
            )
            dl_and_mg = mg_mask & samples["detection_limit"].notna()
            samples.loc[dl_and_mg, "detection_limit"] = (
                samples.loc[dl_and_mg, "detection_limit"] * MGL_TO_UGL
            )
        samples["unit"] = "ug/L"

        # Parse dates
        if date_col:
            samples["sample_date"] = pd.to_datetime(samples[date_col], errors="coerce")
        else:
            samples["sample_date"] = pd.NaT

        # Join with system metadata for coordinates and extras
        sys_col_map = self._config.get("system_column_map", {})
        sys_pwsid_col = find_col(systems, ["PWSID"])

        # Deduplicate systems to one row per PWSID
        systems = systems.drop_duplicates(subset=[sys_pwsid_col], keep="first")

        # Merge
        samples = samples.merge(
            systems[[sys_pwsid_col] + [c for c in systems.columns if c != sys_pwsid_col]],
            left_on=pwsid_col,
            right_on=sys_pwsid_col,
            how="left",
            suffixes=("", "_sys"),
        )

        # Coordinates from system metadata (often county centroids — low quality)
        lat_col = find_col(systems, ["GEO_LATITUDE", "LATITUDE"], required=False)
        lon_col = find_col(systems, ["GEO_LONGITUDE", "LONGITUDE"], required=False)

        if lat_col and lat_col in samples.columns:
            samples["latitude"] = pd.to_numeric(samples[lat_col], errors="coerce")
        else:
            samples["latitude"] = float("nan")

        if lon_col and lon_col in samples.columns:
            samples["longitude"] = pd.to_numeric(samples[lon_col], errors="coerce")
        else:
            samples["longitude"] = float("nan")

        # Flag low-quality coordinates
        samples["raw_coord_quality"] = "low"

        # Preserve ZIP_CODE for downstream geocoding
        zip_col = find_col(samples, ["ZIP_CODE", "ZIP_CODE_sys"], required=False)
        if zip_col:
            samples["raw_ZipCode"] = samples[zip_col].astype(str).str.strip()

        # Preserve extra metadata
        for raw_col, std_name in sys_col_map.items():
            if raw_col in samples.columns and std_name.startswith("raw_"):
                samples = samples.rename(columns={raw_col: std_name})

        # Rename PWSID
        samples = samples.rename(columns={pwsid_col: "pwsid"})

        # Select standard columns + raw_ extras
        standard_cols = [
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
        raw_cols = [c for c in samples.columns if c.startswith("raw_")]
        keep = [c for c in standard_cols if c in samples.columns] + raw_cols
        result = samples[keep]

        # Deterministic row ordering for reproducibility across filesystems
        sort_cols = [c for c in ["pwsid", "analyte", "sample_date"] if c in result.columns]
        if sort_cols:
            result = result.sort_values(by=sort_cols).reset_index(drop=True)
        else:
            result = result.reset_index(drop=True)

        logger.info("Parsed %d SDWIS samples", len(result))
        return cast(pd.DataFrame, result)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate SDWIS data."""
        return self._validate_standard(
            df, set(HEAVY_METAL_ANALYTES), skip_coord_check=True, check_duplicates=True
        )
