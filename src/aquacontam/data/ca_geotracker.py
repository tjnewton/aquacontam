"""California GeoTracker (GAMA) PFAS groundwater data source.

Downloads PFAS results from the California GAMA (Groundwater Ambient
Monitoring & Assessment) program's public data portal and maps them to
the standard water quality schema.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from aquacontam._config import load_data_config
from aquacontam._constants import DOWNLOAD_TIMEOUT_DEFAULT, MGL_TO_UGL, PPT_TO_UGL
from aquacontam.data._download import download_and_extract
from aquacontam.data._utils import PGL_TO_UGL
from aquacontam.data.base import DataSource
from aquacontam.data.schema import validate_schema

logger = logging.getLogger(__name__)

# GAMA ``gm_chemical_vvl`` short codes -> canonical analyte names (matching the
# UCMR5/state-source vocabulary). Only confident matches are mapped; ambiguous or
# non-canonical codes (e.g. fluorotelomer carboxylic acids, PFHA, PFPA) pass
# through unchanged and are excluded from canonical-analyte tasks. PFOSA is
# intentionally NOT mapped to PFOS — perfluorooctane sulfonamide is a distinct
# compound.
_CA_ANALYTE_MAP: dict[str, str] = {
    "11ClPF3OUDS": "11Cl-PF3OUdS",
    "4:2FTS": "4:2 FTS",
    "6:2FTS": "6:2 FTS",
    "8:2FTS": "8:2 FTS",
    "9ClPF3ONS": "9Cl-PF3ONS",
    "HFPA-DA": "HFPO-DA",
    "NETFOSAA": "NEtFOSAA",
    "NMEFOSAA": "NMeFOSAA",
    "PFBSA": "PFBS",
    "PFDOA": "PFDoA",
    "PFHPA": "PFHpA",
    "PFHPSA": "PFHpS",
    "PFHXSA": "PFHxS",
    "PFPES": "PFPeS",
    # exact-match canonical codes (kept explicit for stability)
    "ADONA": "ADONA",
    "NFDHA": "NFDHA",
    "PFEESA": "PFEESA",
    "PFMBA": "PFMBA",
    "PFMPA": "PFMPA",
    "PFNA": "PFNA",
    "PFOA": "PFOA",
    "PFOS": "PFOS",
}


class CaGeoTrackerSource(DataSource):
    """California GAMA PFAS groundwater data loader.

    Downloads PFAS results from the California Open Data portal (GAMA
    program). The GAMA export keys samples by ``gm_well_id`` of the form
    ``<PWSID>_<facility>_<well>`` (e.g. ``CA0800552_002_002``); the
    9-character California PWSID prefix is extracted to group samples by
    public water system. Rows whose well id does not carry a valid CA PWSID
    prefix (non-PWS monitoring wells) are dropped, restricting the benchmark
    population to public water systems.

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
        self._config = load_data_config(config_path, source_name="ca_geotracker")

    @property
    def name(self) -> str:
        return "ca_geotracker"

    def download(self, *, force: bool = False, **kwargs: Any) -> list[Path]:
        """Download GAMA PFAS CSV from California Open Data."""
        return download_and_extract(
            self._config["url"],
            self.raw_dir,
            self._config["expected_files"],
            force=force,
            timeout=DOWNLOAD_TIMEOUT_DEFAULT,
            progress_desc="CA GeoTracker download",
        )

    @staticmethod
    def _col(df: pd.DataFrame, name: str) -> str | None:
        """Case-insensitive column lookup."""
        lut = {c.lower(): c for c in df.columns}
        return lut.get(name.lower())

    def parse(self, **kwargs: Any) -> pd.DataFrame:
        """Parse GAMA PFAS CSV into the standardized schema (vectorized).

        The public water system id is the 9-character ``CA#######`` prefix of
        ``gm_well_id``; rows without a valid CA PWSID prefix (non-PWS wells)
        are excluded. Concentrations are converted to ug/L, ``<`` results are
        flagged as non-detect (concentration 0), and records are deduplicated
        to one row per ``(pwsid, analyte, sample_date)`` keeping the maximum
        concentration so detections are not lost. Fully vectorized and
        deterministic.
        """
        csv_path = self.raw_dir / self._config["expected_files"][0]
        encoding = self._config.get("format", {}).get("encoding", "utf-8")
        df = pd.read_csv(csv_path, dtype=str, encoding=encoding, low_memory=False)
        logger.info("Read %d rows from CA GeoTracker", len(df))

        wid_col = self._col(df, "gm_well_id")
        if wid_col is None:
            raise ValueError("CA GeoTracker raw is missing the gm_well_id column")

        # PWSID = 9-char CA prefix (CA + 7 digits) of gm_well_id; drop non-PWS wells.
        pwsid = df[wid_col].astype(str).str.extract(r"^(CA\d{7})", expand=False)
        keep = pwsid.notna()
        n_dropped = int((~keep).sum())
        df = df[keep].copy()
        df["pwsid"] = pwsid[keep].to_numpy()
        if n_dropped:
            logger.info(
                "Dropped %d non-PWS-format GAMA rows; %d PWS rows remain", n_dropped, len(df)
            )

        vvl_col = self._col(df, "gm_chemical_vvl") or self._col(df, "gm_chemical_name")
        result_col = self._col(df, "gm_result")
        unit_col = self._col(df, "gm_chemical_units")
        rl_col = self._col(df, "gm_reporting_limit")
        mod_col = self._col(df, "gm_result_modifier")
        lat_col = self._col(df, "gm_latitude")
        lon_col = self._col(df, "gm_longitude")
        date_col = self._col(df, "gm_samp_collection_date")

        # Map GAMA short codes to canonical analyte names; unmapped codes pass
        # through (excluded downstream from canonical-analyte tasks).
        analyte = (
            df[vvl_col].astype(str).str.strip().map(lambda x: _CA_ANALYTE_MAP.get(x, x))
            if vvl_col
            else pd.Series("", index=df.index)
        )
        result = (
            pd.to_numeric(df[result_col], errors="coerce")
            if result_col
            else pd.Series(np.nan, index=df.index)
        )
        rl = (
            pd.to_numeric(df[rl_col], errors="coerce")
            if rl_col
            else pd.Series(np.nan, index=df.index)
        )

        # Vectorized unit -> ug/L factor.
        units = (
            df[unit_col].astype(str).str.strip().str.upper()
            if unit_col
            else pd.Series("", index=df.index)
        )
        factor = pd.Series(1.0, index=df.index)
        factor[units.isin(["PPT", "NG/L"])] = PPT_TO_UGL
        factor[units == "PG/L"] = PGL_TO_UGL
        factor[units == "MG/L"] = MGL_TO_UGL

        conc = result * factor
        dl = rl * factor

        # Censored: '<' modifier (non-detect) or missing result.
        censored = (
            df[mod_col].astype(str).str.strip().eq("<")
            if mod_col
            else pd.Series(False, index=df.index)
        )
        censored = censored | result.isna()
        conc = conc.where(~censored, 0.0).fillna(0.0)

        lat = (
            pd.to_numeric(df[lat_col], errors="coerce")
            if lat_col
            else pd.Series(np.nan, index=df.index)
        )
        lon = (
            pd.to_numeric(df[lon_col], errors="coerce")
            if lon_col
            else pd.Series(np.nan, index=df.index)
        )
        sample_date = (
            pd.to_datetime(df[date_col], errors="coerce")
            if date_col
            else pd.Series(pd.NaT, index=df.index)
        )

        out = pd.DataFrame(
            {
                "pwsid": df["pwsid"].to_numpy(),
                "analyte": analyte.to_numpy(),
                "concentration": conc.to_numpy(dtype=float),
                "unit": "ug/L",
                "censored": censored.to_numpy(dtype=bool),
                "detection_limit": dl.to_numpy(dtype=float),
                "sample_date": sample_date.to_numpy(),
                "latitude": lat.to_numpy(dtype=float),
                "longitude": lon.to_numpy(dtype=float),
            }
        )
        out = out[out["analyte"].str.len() > 0]

        # One row per (pwsid, analyte, sample_date), keeping the max concentration
        # so a detection is never dropped in favour of a co-dated non-detect.
        out = (
            out.sort_values(["pwsid", "analyte", "sample_date", "concentration"])
            .drop_duplicates(subset=["pwsid", "analyte", "sample_date"], keep="last")
            .sort_values(["pwsid", "analyte", "sample_date"])
            .reset_index(drop=True)
        )

        logger.info(
            "Parsed %d CA GeoTracker samples (%d systems)", len(out), out["pwsid"].nunique()
        )
        return cast(pd.DataFrame, out)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate CA GeoTracker data against the standard schema.

        All rows carry real CA PWSIDs (synthetic well ids are excluded in
        ``parse``), so standard schema validation applies directly.
        """
        return cast(
            pd.DataFrame, validate_schema(df, skip_coord_check=False, drop_invalid_rows=True)
        )
