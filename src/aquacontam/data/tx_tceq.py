"""Texas TCEQ PFAS data source — placeholder (unavailable).

TCEQ PFAS drinking water data is not available via public API or download.
Texas data is partially available through the Water Quality Portal (WQP).
A direct data request can be made via openrecs@tceq.texas.gov.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import pandas as pd

from aquacontam._config import load_data_config
from aquacontam.data._utils import convert_to_ugl, find_col, normalize_pwsid
from aquacontam.data.base import DataSource
from aquacontam.data.schema import validate_schema

logger = logging.getLogger(__name__)


class TxTceqSource(DataSource):
    """Texas TCEQ PFAS data loader (unavailable — placeholder).

    TCEQ does not provide a public API or bulk download for PFAS data.
    This loader will raise ``RuntimeError`` on download. Texas PFAS data
    is partially available via the Water Quality Portal (``WqpSource``).

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
        self._config = load_data_config(config_path, source_name="tx_tceq")

    @property
    def name(self) -> str:
        return "tx_tceq"

    def download(self, *, force: bool = False, **kwargs: Any) -> list[Path]:
        """Raise RuntimeError — TCEQ data requires manual request."""
        reason = self._config.get(
            "unavailable_reason",
            "TCEQ PFAS data requires manual request via openrecs@tceq.texas.gov. "
            "No public API available. Texas PFAS data is partially available via WQP.",
        )
        raise RuntimeError(f"TX TCEQ download unavailable: {reason}")

    def parse(self, **kwargs: Any) -> pd.DataFrame:
        """Parse TX TCEQ CSV into standardized schema.

        Expects a manually obtained CSV with standard water quality columns.
        Uses flexible column lookup to handle varying column names from
        TCEQ data exports.
        """
        csv_path = self.raw_dir / self._config["expected_files"][0]
        if not csv_path.exists():
            raise FileNotFoundError(
                f"TX TCEQ data file not found: {csv_path}. "
                "This data source requires manual data request."
            )

        df = pd.read_csv(csv_path, dtype=str, low_memory=False)
        logger.info("Read %d rows from TX TCEQ", len(df))

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

        # Flexible column lookup — accommodates different TCEQ export formats
        pwsid_col = find_col(df, ["PWSID", "PWS_ID", "pws_id", "SystemID", "pwsid"], required=True)
        analyte_col = find_col(
            df,
            ["analyte", "ANALYTE", "ContaminantName", "ParameterName", "Analyte"],
            required=True,
        )
        conc_col = find_col(
            df,
            ["concentration", "RESULT", "SampleResult", "ResultValue", "Concentration"],
            required=True,
        )
        unit_col = find_col(df, ["unit", "UNIT", "Units", "ResultUnit", "UOM"], required=False)
        censored_col = find_col(
            df, ["censored", "CENSORED", "ResultSign", "Qualifier", "ND_Flag"], required=False
        )
        dl_col = find_col(
            df,
            ["detection_limit", "DETECTION_LIMIT", "DetectionLimit", "MRL", "MDL"],
            required=False,
        )
        date_col = find_col(
            df, ["sample_date", "SAMPLE_DATE", "SampleDate", "CollectionDate"], required=False
        )
        lat_col = find_col(df, ["latitude", "LATITUDE", "Latitude", "LAT"], required=False)
        lon_col = find_col(
            df, ["longitude", "LONGITUDE", "Longitude", "LON", "LONG"], required=False
        )

        # Build result DataFrame
        assert pwsid_col is not None  # guaranteed by required=True
        assert analyte_col is not None
        assert conc_col is not None

        result = pd.DataFrame(index=df.index)

        result["pwsid"] = df[pwsid_col].astype(str).apply(lambda x: normalize_pwsid(x, "TX"))
        result["analyte"] = df[analyte_col].astype(str).str.strip()

        # Concentration — parse and convert units
        conc = pd.to_numeric(df[conc_col], errors="coerce").fillna(0.0)
        if unit_col is not None:
            unit_raw = df[unit_col].astype(str).str.strip().str.upper()
            conc_ugl = pd.Series(0.0, index=df.index)
            for i in df.index:
                u = str(unit_raw.iloc[i]) if unit_raw.iloc[i] else "UG/L"
                conc_ugl.iloc[i] = convert_to_ugl(float(conc.iloc[i]), u)
            result["concentration"] = conc_ugl
        else:
            result["concentration"] = conc

        result["unit"] = "ug/L"

        # Censoring
        if censored_col is not None:
            result["censored"] = (
                df[censored_col].astype(str).str.upper().isin({"TRUE", "1", "YES", "<", "ND", "U"})
            )
        else:
            result["censored"] = False

        # Detection limit
        if dl_col is not None:
            result["detection_limit"] = pd.to_numeric(df[dl_col], errors="coerce")
        else:
            result["detection_limit"] = float("nan")

        # Sample date
        if date_col is not None:
            result["sample_date"] = pd.to_datetime(df[date_col], errors="coerce")
        else:
            result["sample_date"] = pd.NaT

        # Coordinates
        if lat_col is not None:
            result["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
        else:
            result["latitude"] = float("nan")
        if lon_col is not None:
            result["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
        else:
            result["longitude"] = float("nan")

        # Drop rows with empty analyte
        result = result[result["analyte"].str.len() > 0]

        # Deterministic row ordering
        sort_cols = [c for c in ["pwsid", "analyte", "sample_date"] if c in result.columns]
        if sort_cols:
            result = result.sort_values(by=sort_cols).reset_index(drop=True)
        else:
            result = result.reset_index(drop=True)

        logger.info("Parsed %d TX TCEQ samples", len(result))
        return cast(pd.DataFrame, result)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate TX TCEQ data."""
        return cast(
            pd.DataFrame, validate_schema(df, skip_coord_check=False, drop_invalid_rows=True)
        )
