"""Water Quality Portal (WQP) PFAS data source.

Downloads PFAS sampling results from the USGS/EPA Water Quality Portal
REST API and maps them to the standard water quality schema. Covers
multiple states (TX, MO) and any other state with WQP submissions.
"""

from __future__ import annotations

import hashlib
import io
import logging
import time
import zipfile
from pathlib import Path
from typing import Any, cast

import pandas as pd
import requests

from aquacontam._config import load_data_config
from aquacontam._constants import DOWNLOAD_TIMEOUT_DEFAULT
from aquacontam.data._utils import convert_to_ugl
from aquacontam.data.base import DataSource
from aquacontam.data.schema import validate_schema

logger = logging.getLogger(__name__)

# PFAS characteristic names used in WQP queries
_WQP_PFAS_CHARACTERISTICS: tuple[str, ...] = (
    "Perfluorooctanoic acid",
    "Perfluorooctane sulfonic acid",
    "Perfluorononanoic acid",
    "Perfluorohexanesulfonic acid",
    "Perfluorobutanesulfonic acid",
    "Hexafluoropropylene oxide dimer acid",
    "Perfluoropentanoic acid",
    "Perfluorohexanoic acid",
    "Perfluorodecanoic acid",
    "Perfluoroundecanoic acid",
    "Perfluorobutanoic acid",
    "Perfluoroheptanoic acid",
    "Perfluoroheptanesulfonate",
)

# WQP API rejects requests with >3 characteristicName values (400 Bad Request).
# Batch characteristics into chunks of this size and merge results.
_WQP_CHARACTERISTIC_BATCH_SIZE = 3

# Map WQP characteristic names to standard analyte abbreviations
_CHARACTERISTIC_TO_ANALYTE: dict[str, str] = {
    "Perfluorooctanoic acid": "PFOA",
    "Perfluorooctane sulfonic acid": "PFOS",
    "Perfluorononanoic acid": "PFNA",
    "Perfluorohexanesulfonic acid": "PFHxS",
    "Perfluorobutanesulfonic acid": "PFBS",
    "Hexafluoropropylene oxide dimer acid": "HFPO-DA",
    "Perfluoropentanoic acid": "PFPeA",
    "Perfluorohexanoic acid": "PFHxA",
    "Perfluorodecanoic acid": "PFDA",
    "Perfluoroundecanoic acid": "PFUnA",
    "Perfluorobutanoic acid": "PFBA",
    "Perfluoroheptanoic acid": "PFHpA",
    "Perfluoroheptanesulfonate": "PFHpS",
}


class WqpSource(DataSource):
    """Water Quality Portal PFAS data loader.

    Downloads PFAS results from the USGS/EPA Water Quality Portal
    (waterqualitydata.us) REST API. Queries are filtered by state
    FIPS code and PFAS characteristic names.

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
        self._config = load_data_config(config_path, source_name="wqp")

    @property
    def name(self) -> str:
        return "wqp"

    def download(self, *, force: bool = False, **kwargs: Any) -> list[Path]:
        """Download WQP PFAS data for configured states.

        Iterates over all configured CONUS state codes, downloading results
        and station metadata for each. Per-state errors are caught and logged
        so that a single failing state does not abort the entire download.
        Results are written incrementally to avoid accumulating large
        DataFrames in memory.
        """
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        expected = self._config["expected_files"]
        results_dest = self.raw_dir / expected[0]
        stations_dest = self.raw_dir / expected[1]

        if results_dest.exists() and stations_dest.exists() and not force:
            logger.info("WQP data already exists: %s", results_dest)
            return [results_dest, stations_dest]

        if self._config.get("unavailable", False):
            reason = self._config.get("unavailable_reason", "source unavailable")
            raise RuntimeError(f"WQP download skipped: {reason}")

        states = self._config.get("states", ["US:48", "US:29"])
        result_url = self._config["url"]
        station_url = self._config["station_url"]
        timeout = self._config.get("download_timeout", DOWNLOAD_TIMEOUT_DEFAULT)
        delay = self._config.get("request_delay_seconds", 2)

        n_states = len(states)
        results_written = 0
        stations_written = 0
        failed_states: list[str] = []

        # Remove destination files to write fresh
        for dest in (results_dest, stations_dest):
            if dest.exists():
                dest.unlink()

        for idx, state_code in enumerate(states, 1):
            logger.info(
                "Downloading WQP data for state %s (%d/%d) ...",
                state_code,
                idx,
                n_states,
            )
            # WQP rejects >3 characteristicName values per request.
            # Batch characteristics and merge results per state.
            char_batches = [
                _WQP_PFAS_CHARACTERISTICS[i : i + _WQP_CHARACTERISTIC_BATCH_SIZE]
                for i in range(0, len(_WQP_PFAS_CHARACTERISTICS), _WQP_CHARACTERISTIC_BATCH_SIZE)
            ]
            succeeded_batches = 0
            failed_batch_chars: list[str] = []
            for batch in char_batches:
                try:
                    # Download results
                    params: list[tuple[str, str]] = [
                        ("statecode", state_code),
                        *[("characteristicName", c) for c in batch],
                        ("mimeType", "csv"),
                        ("zip", "yes"),
                        ("dataProfile", "resultPhysChem"),
                        ("sorted", "no"),
                    ]
                    resp = requests.get(result_url, params=params, timeout=timeout, stream=True)
                    resp.raise_for_status()
                    df_results = _read_wqp_zip(resp.content)
                    if df_results is not None and len(df_results) > 0:
                        write_header = not results_dest.exists()
                        df_results.to_csv(results_dest, mode="a", header=write_header, index=False)
                        results_written += len(df_results)
                    succeeded_batches += 1

                    # Download station metadata
                    station_params: list[tuple[str, str]] = [
                        ("statecode", state_code),
                        *[("characteristicName", c) for c in batch],
                        ("mimeType", "csv"),
                        ("zip", "yes"),
                        ("sorted", "no"),
                    ]
                    resp_st = requests.get(
                        station_url, params=station_params, timeout=timeout, stream=True
                    )
                    resp_st.raise_for_status()
                    df_stations = _read_wqp_zip(resp_st.content)
                    if df_stations is not None and len(df_stations) > 0:
                        write_header = not stations_dest.exists()
                        df_stations.to_csv(
                            stations_dest, mode="a", header=write_header, index=False
                        )
                        stations_written += len(df_stations)

                except (requests.RequestException, zipfile.BadZipFile) as exc:
                    logger.warning(
                        "Failed WQP batch for %s (%s): %s — skipping batch",
                        state_code,
                        ";".join(batch),
                        exc,
                    )
                    failed_batch_chars.extend(batch)
                    continue

            if succeeded_batches == 0:
                failed_states.append(state_code)
            else:
                logger.info("Downloaded %d result rows for %s", results_written, state_code)
                if failed_batch_chars:
                    logger.warning(
                        "Partial data for %s: %d/%d batches failed (missing characteristics: %s)",
                        state_code,
                        len(failed_batch_chars),
                        len(char_batches),
                        "; ".join(failed_batch_chars),
                    )

            # Rate limit between state queries
            if idx < n_states and delay > 0:
                time.sleep(delay)

        # Ensure files exist even if no data was retrieved
        if not results_dest.exists():
            pd.DataFrame().to_csv(results_dest, index=False)
        if not stations_dest.exists():
            pd.DataFrame().to_csv(stations_dest, index=False)

        logger.info(
            "WQP download complete: %d result rows, %d station rows from %d/%d states",
            results_written,
            stations_written,
            n_states - len(failed_states),
            n_states,
        )
        if failed_states:
            logger.warning("Failed states: %s", ", ".join(failed_states))

        return [results_dest, stations_dest]

    def parse(self, **kwargs: Any) -> pd.DataFrame:
        """Parse WQP CSV files into standardized schema.

        WQP results use ``CharacteristicName`` for analyte,
        ``ResultMeasureValue`` for concentration, and
        ``ResultDetectionConditionText`` for censoring status.
        Station coordinates come from a separate station file.
        """
        expected = self._config["expected_files"]
        results_path = self.raw_dir / expected[0]
        stations_path = self.raw_dir / expected[1]

        encoding = self._config.get("format", {}).get("encoding", "utf-8")
        try:
            df = pd.read_csv(results_path, dtype=str, encoding=encoding, low_memory=False)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        logger.info("Read %d rows from WQP results", len(df))

        if df.empty:
            return cast(
                pd.DataFrame,
                pd.DataFrame(
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
                ),
            )

        # Load station coordinates
        coords: dict[str, tuple[float, float]] = {}
        if stations_path.exists():
            stations = pd.read_csv(stations_path, dtype=str, encoding=encoding, low_memory=False)
            for _, row in stations.iterrows():
                site_id = str(row.get("MonitoringLocationIdentifier", "")).strip()
                lat = pd.to_numeric(row.get("LatitudeMeasure", ""), errors="coerce")
                lon = pd.to_numeric(row.get("LongitudeMeasure", ""), errors="coerce")
                if site_id and pd.notna(lat) and pd.notna(lon):
                    coords[site_id] = (float(lat), float(lon))

        rows = []
        for _, row in df.iterrows():
            # Site ID → synthetic PWSID
            site_id = str(row.get("MonitoringLocationIdentifier", "")).strip()
            pwsid = _site_id_to_pwsid(site_id)

            # Analyte
            char_name = str(row.get("CharacteristicName", "")).strip()
            analyte = _CHARACTERISTIC_TO_ANALYTE.get(char_name, char_name)

            # Concentration
            conc_raw = pd.to_numeric(row.get("ResultMeasureValue", ""), errors="coerce")
            conc = float(conc_raw) if pd.notna(conc_raw) else 0.0

            # Unit — WQP may have NaN units or non-liquid units (e.g. NG/G)
            unit_raw = str(row.get("ResultMeasure/MeasureUnitCode", "")).strip().upper()
            if unit_raw in ("", "NAN", "NONE"):
                unit_raw = ""
            try:
                conc = convert_to_ugl(conc, unit_raw) if unit_raw else conc
            except ValueError:
                # Non-liquid unit (e.g. NG/G for solid matrix) — skip row
                continue

            # Censoring
            detect_cond = str(row.get("ResultDetectionConditionText", "")).strip().lower()
            censored = detect_cond in ("not detected", "present below quantification limit")
            if censored:
                conc = 0.0

            # Detection limit
            dl_raw = pd.to_numeric(
                row.get("DetectionQuantitationLimitMeasure/MeasureValue", ""),
                errors="coerce",
            )
            dl_unit = (
                str(row.get("DetectionQuantitationLimitMeasure/MeasureUnitCode", ""))
                .strip()
                .upper()
            )
            dl = float("nan")
            if dl_unit in ("", "NAN", "NONE"):
                dl_unit = ""
            if pd.notna(dl_raw):
                try:
                    dl = convert_to_ugl(float(dl_raw), dl_unit) if dl_unit else float(dl_raw)
                except ValueError:
                    dl = float("nan")

            # Sample date
            sample_date = pd.to_datetime(row.get("ActivityStartDate", ""), errors="coerce")

            # Coordinates from station lookup
            lat, lon_val = coords.get(site_id, (float("nan"), float("nan")))

            rows.append(
                {
                    "pwsid": pwsid,
                    "analyte": analyte,
                    "concentration": conc,
                    "unit": "ug/L",
                    "censored": censored,
                    "detection_limit": dl,
                    "sample_date": sample_date,
                    "latitude": lat,
                    "longitude": lon_val,
                }
            )

        result = pd.DataFrame(rows)
        result = result[result["analyte"].str.len() > 0]

        # Deterministic row ordering
        sort_cols = [c for c in ["pwsid", "analyte", "sample_date"] if c in result.columns]
        if sort_cols:
            result = result.sort_values(by=sort_cols).reset_index(drop=True)
        else:
            result = result.reset_index(drop=True)

        logger.info("Parsed %d WQP samples", len(result))
        return cast(pd.DataFrame, result)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate WQP data.

        WQP monitoring locations are not public water systems, so most rows
        have synthetic ``WQP_`` IDs that don't conform to the 9-char PWSID
        format. We skip PWSID validation for these rows while still checking
        coordinates, concentrations, and other schema requirements.
        """
        return cast(
            pd.DataFrame,
            validate_schema(
                df,
                skip_coord_check=False,
                skip_pwsid_check=True,
                drop_invalid_rows=True,
            ),
        )


def _read_wqp_zip(content: bytes) -> pd.DataFrame | None:
    """Read a WQP ZIP response into a DataFrame."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                return None
            with zf.open(csv_names[0]) as f:
                return pd.read_csv(f, dtype=str, low_memory=False)
    except (zipfile.BadZipFile, KeyError):
        logger.warning("Failed to read WQP ZIP response")
        return None


def _site_id_to_pwsid(site_id: str) -> str:
    """Convert a WQP MonitoringLocationIdentifier to a PWSID-like string.

    WQP site IDs vary in format (e.g., ``"USGS-12345678"``,
    ``"21TXSWQ-12345"``). If the ID contains a recognizable PWSID
    (9-char state+number format), extract it. Otherwise generate a
    synthetic ID with ``WQP_`` prefix.
    """
    if not site_id:
        return "WQP_0000000"

    # Check if site_id itself is a valid PWSID
    cleaned = site_id.replace("-", "").strip()
    if len(cleaned) >= 9 and cleaned[:2].isalpha() and cleaned[2:9].isdigit():
        return cleaned[:9].upper()

    # Try extracting state prefix from common WQP patterns
    # e.g. "21TXSWQ-12345" -> TX prefix
    parts = site_id.split("-", 1)
    if len(parts) == 2:
        prefix_part = parts[0]
        # Look for 2-letter state code embedded in prefix
        for i in range(len(prefix_part) - 1):
            candidate = prefix_part[i : i + 2].upper()
            if candidate.isalpha():
                # Use a deterministic hash-based synthetic ID
                num = int(hashlib.md5(site_id.encode("utf-8")).hexdigest(), 16) % 9999999
                return f"WQP_{num:07d}"

    # Fallback: deterministic synthetic ID from content hash
    num = int(hashlib.md5(site_id.encode("utf-8")).hexdigest(), 16) % 9999999
    return f"WQP_{num:07d}"
