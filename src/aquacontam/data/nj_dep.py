"""New Jersey DEP PFAS data source — waterviewer.nj.gov API.

Downloads water system inventory and facility coordinates from the NJ DEP
Drinking Water Viewer API. Chemical sample data is protected by reCAPTCHA
and cannot be downloaded automatically; however, manually-exported sample
files (Excel/CSV) are parsed if present.

NJ has the strictest MCLs: PFNA 13 ppt, PFOA 14 ppt, PFOS 13 ppt.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, cast

import pandas as pd
import requests

from aquacontam._config import load_data_config
from aquacontam._constants import (
    DOWNLOAD_TIMEOUT_DEFAULT,
    NJ_DEP_ANALYTE_CODES,
    NJ_DEP_ANALYTES,
    PPT_TO_UGL,
)
from aquacontam.data._utils import find_col, normalize_pwsid
from aquacontam.data.base import DataSource
from aquacontam.data.schema import validate_schema

logger = logging.getLogger(__name__)

# Composite/summary analyte codes that should not appear in sample-level data
_SUMMARY_CODES: frozenset[int] = frozenset({2800, 2830, 2840})

# Request delay between paginated API calls (seconds)
_REQUEST_DELAY: float = 0.25


def _create_session(base_url: str, *, timeout: int = 30) -> requests.Session:
    """Create an authenticated session for waterviewer.nj.gov.

    Parameters
    ----------
    base_url : str
        Base URL of the NJ Drinking Water Viewer (e.g. ``https://waterviewer.nj.gov``).
    timeout : int
        HTTP timeout for the initial GET request.

    Returns
    -------
    requests.Session
        Session with cookies and ``X-XSRF-TOKEN`` header set.
    """
    session = requests.Session()
    resp = session.get(base_url, timeout=timeout)
    resp.raise_for_status()

    xsrf = session.cookies.get("XSRF-TOKEN", "")
    if xsrf:
        session.headers["X-XSRF-TOKEN"] = xsrf
        logger.debug("Obtained XSRF token (%d chars)", len(xsrf))
    else:
        logger.warning("No XSRF-TOKEN cookie received from %s", base_url)

    return session


def _fetch_inventory(
    session: requests.Session,
    base_url: str,
    endpoint: str,
    page_size: int = 1000,
    *,
    timeout: int = DOWNLOAD_TIMEOUT_DEFAULT,
) -> list[dict[str, Any]]:
    """Download the full water system inventory via OData pagination.

    Parameters
    ----------
    session : requests.Session
        Authenticated session.
    base_url : str
        Base URL (e.g. ``https://waterviewer.nj.gov``).
    endpoint : str
        Inventory endpoint path (e.g. ``/sdwis/DashMain``).
    page_size : int
        Records per page (max 1000).
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    list[dict]
        List of water system records.
    """
    url = f"{base_url}{endpoint}"
    all_records: list[dict[str, Any]] = []
    offset = 0

    while True:
        params = {"skip": offset, "take": page_size}
        logger.info("Fetching NJ DEP inventory, offset %d ...", offset)
        resp = session.get(url, params=params, timeout=timeout)
        resp.raise_for_status()

        data = resp.json()
        batch = data.get("value", [])
        if not batch:
            break

        all_records.extend(batch)
        offset += len(batch)
        logger.debug("Fetched %d records (total %d)", len(batch), len(all_records))

        if len(batch) < page_size:
            break

        time.sleep(_REQUEST_DELAY)

    logger.info("Downloaded %d water system records from NJ DEP inventory", len(all_records))
    return all_records


def _fetch_facilities(
    session: requests.Session,
    base_url: str,
    endpoint: str,
    systems: list[dict[str, Any]],
    *,
    timeout: int = DOWNLOAD_TIMEOUT_DEFAULT,
) -> list[dict[str, Any]]:
    """Download facility coordinates for active systems.

    Parameters
    ----------
    session : requests.Session
        Authenticated session.
    base_url : str
        Base URL.
    endpoint : str
        Facility endpoint path (e.g. ``/sdwis/FacilityList``).
    systems : list[dict]
        Inventory records (must contain ``TINWSYS_IS_NUMBER`` and
        ``ACTIVITY_STATUS_CD``).
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    list[dict]
        Facility records with coordinates.
    """
    url = f"{base_url}{endpoint}"
    all_facilities: list[dict[str, Any]] = []

    active_systems = [
        s for s in systems if str(s.get("ACTIVITY_STATUS_CD", "")).strip().upper() == "A"
    ]
    logger.info("Fetching facilities for %d active systems ...", len(active_systems))

    for idx, sys_rec in enumerate(active_systems):
        is_num = sys_rec.get("TINWSYS_IS_NUMBER")
        pwsid = str(sys_rec.get("NUMBER0", "")).strip()
        if is_num is None:
            continue

        params: dict[str, str | int] = {
            "n0": pwsid,
            "$filter": f"(TINWSYS_IS_NUMBER eq {is_num} and TINWSYS_ST_CODE eq 'NJ')",
        }
        try:
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            facilities = data.get("value", [])
            for fac in facilities:
                fac["_PWSID"] = pwsid
            all_facilities.extend(facilities)
        except requests.RequestException as exc:
            logger.warning("Failed to fetch facilities for %s: %s", pwsid, exc)

        if (idx + 1) % 100 == 0:
            logger.info("  ... processed %d / %d systems", idx + 1, len(active_systems))

        time.sleep(_REQUEST_DELAY)

    logger.info("Downloaded %d facility records from NJ DEP", len(all_facilities))
    return all_facilities


class NjDepSource(DataSource):
    """New Jersey DEP PFAS data loader via waterviewer.nj.gov.

    Downloads the water system inventory and facility coordinates from
    the NJ Drinking Water Viewer API. Chemical sample data is
    reCAPTCHA-protected and must be provided manually (Excel/CSV export).

    If a manual export file is present (``nj_dep_pfas_export.xlsx`` in
    ``raw_dir``), ``parse()`` will process it into the standard schema.
    Otherwise it produces an inventory-only DataFrame (no sample data).

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
        self._config = load_data_config(config_path, source_name="nj_dep")

    @property
    def name(self) -> str:
        return "nj_dep"

    def download(self, *, force: bool = False, **kwargs: Any) -> list[Path]:
        """Download NJ DEP water system inventory and facility coordinates.

        The inventory and facility data are accessible without reCAPTCHA.
        Chemical sample data (SamplesSearchResults) is NOT downloaded
        because it requires a reCAPTCHA token.

        Returns
        -------
        list[Path]
            Paths to the downloaded CSV files.
        """
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        expected = self._config["expected_files"]
        inventory_dest = self.raw_dir / expected[0]
        facilities_dest = self.raw_dir / expected[1]

        if inventory_dest.exists() and facilities_dest.exists() and not force:
            logger.info("NJ DEP data already exists: %s", inventory_dest)
            return [inventory_dest, facilities_dest]

        base_url = self._config["base_url"]
        inv_endpoint = self._config["inventory_endpoint"]
        fac_endpoint = self._config["facility_endpoint"]
        page_size = self._config.get("page_size", 1000)
        timeout = self._config.get("download_timeout", DOWNLOAD_TIMEOUT_DEFAULT)

        # 1. Authenticate
        session = _create_session(base_url, timeout=timeout)

        # 2. Download inventory (paginated)
        systems = _fetch_inventory(
            session, base_url, inv_endpoint, page_size=page_size, timeout=timeout
        )
        if not systems:
            raise RuntimeError("No systems returned from NJ DEP inventory API")

        inv_df = pd.DataFrame(systems)
        inv_df.to_csv(inventory_dest, index=False)
        logger.info("Saved %d NJ DEP inventory records to %s", len(inv_df), inventory_dest)

        # 3. Download facility coordinates for active systems
        facilities = _fetch_facilities(session, base_url, fac_endpoint, systems, timeout=timeout)
        fac_df = pd.DataFrame(facilities) if facilities else pd.DataFrame()
        fac_df.to_csv(facilities_dest, index=False)
        logger.info("Saved %d NJ DEP facility records to %s", len(fac_df), facilities_dest)

        return [inventory_dest, facilities_dest]

    def parse(self, **kwargs: Any) -> pd.DataFrame:
        """Parse NJ DEP data into standardized schema.

        Checks for a manually-provided sample export first. If not found,
        falls back to the inventory data (no sample concentrations).

        Returns
        -------
        pd.DataFrame
            Standardized water quality DataFrame.
        """
        # Priority 1: manually-provided sample export
        manual_name = self._config.get("manual_export_file", "nj_dep_pfas_export.xlsx")
        manual_path = self.raw_dir / manual_name
        if manual_path.exists():
            logger.info("Parsing manual NJ DEP export: %s", manual_path)
            return self._parse_manual_export(manual_path)

        # Also check for CSV variant
        manual_csv = manual_path.with_suffix(".csv")
        if manual_csv.exists():
            logger.info("Parsing manual NJ DEP CSV export: %s", manual_csv)
            return self._parse_manual_export(manual_csv)

        # Priority 2: inventory data (no sample concentrations)
        logger.info("No manual export found; parsing inventory-only data")
        return self._parse_inventory()

    def _parse_manual_export(self, path: Path) -> pd.DataFrame:
        """Parse a manually-exported sample file from waterviewer.nj.gov.

        Supports both Excel (.xlsx) and CSV formats. Expected columns
        align with the SamplesSearchResults grid: Water System ID,
        Analyte Code, Analyte Name, Concentration, Detection Value, etc.
        """
        if path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path, dtype=str)
        else:
            df = pd.read_csv(path, dtype=str, low_memory=False)
        logger.info("Read %d rows from manual NJ DEP export", len(df))

        if df.empty:
            return _empty_schema_df()

        # Find key columns (waterviewer column names vary between API and manual export)
        pwsid_col = find_col(
            df,
            ["PWS_ID", "Water System ID", "NUMBER0", "PWSID", "WaterSystemID"],
            required=False,
        )
        analyte_col = find_col(
            df,
            ["ANALYTE_NAME", "Analyte Name", "AnalyteName", "ANALYTE", "Contaminant"],
            required=False,
        )
        analyte_code_col = find_col(
            df,
            ["ANALYTE_CODE", "Analyte Code", "AnalyteCode"],
            required=False,
        )
        conc_col = find_col(
            df,
            ["RESULT", "Concentration", "CONCENTRATION", "Result"],
            required=False,
        )
        detect_col = find_col(
            df,
            ["Detection Value", "DetectionValue", "DETECTION_LIMIT", "MRL"],
            required=False,
        )
        date_col = find_col(
            df,
            ["COLLECTION_DATE", "Collection Date", "CollectionDate", "SAMPLE_DATE", "SampleDate"],
            required=False,
        )
        unit_col = find_col(
            df,
            ["UOM", "Unit", "UNIT"],
            required=False,
        )
        nondetect_col = find_col(
            df,
            ["NON_DETECT", "LESS_THAN_IND"],
            required=False,
        )

        rows = []
        for _, row in df.iterrows():
            # PWSID
            raw_pwsid = str(row.get(pwsid_col, "")).strip() if pwsid_col else ""
            pwsid = normalize_pwsid(raw_pwsid, "NJ")

            # Analyte — prefer code mapping, fallback to name
            analyte = ""
            if analyte_code_col:
                code_raw = pd.to_numeric(row.get(analyte_code_col, ""), errors="coerce")
                if pd.notna(code_raw):
                    analyte = NJ_DEP_ANALYTE_CODES.get(int(code_raw), "")
            if not analyte and analyte_col:
                analyte = _normalize_nj_analyte(str(row.get(analyte_col, "")).strip())
            if not analyte:
                continue

            # Skip summary/composite codes
            if analyte in ("PFAS RULE", "TOTAL PFOA AND PFOS", "HAZARD INDEX PFAS"):
                continue

            # Concentration — API format may have "<2" for non-detects
            conc_str = str(row.get(conc_col, "")).strip() if conc_col else ""
            conc_str_clean = conc_str.lstrip("<").strip()
            conc_raw = pd.to_numeric(conc_str_clean, errors="coerce")
            conc = float(conc_raw) if pd.notna(conc_raw) else 0.0

            # Detection limit — from explicit column or extracted from "<DL" format
            dl_raw = (
                pd.to_numeric(row.get(detect_col, ""), errors="coerce") if detect_col else None
            )
            if pd.isna(dl_raw) and conc_str.startswith("<"):
                dl_raw = conc_raw  # "<2" means DL=2
            dl = float(dl_raw) if pd.notna(dl_raw) else float("nan")

            # Determine unit — API may provide UOM column
            unit_str = str(row.get(unit_col, "")).strip().upper() if unit_col else ""
            data_in_ugl = unit_str.startswith("UG/L") or unit_str.startswith("UG/L")

            # Convert PPT → ug/L only if data is in PPT (ng/L)
            if data_in_ugl:
                conc_ugl = conc
                dl_ugl = dl if pd.notna(dl) else float("nan")
            else:
                # NJ data is typically in PPT (ng/L) when no unit column
                conc_ugl = conc * PPT_TO_UGL
                dl_ugl = dl * PPT_TO_UGL if pd.notna(dl) else float("nan")

            # Censoring — check NON_DETECT/LESS_THAN_IND columns or "<" prefix
            censored = False
            if nondetect_col:
                nd_val = str(row.get(nondetect_col, "")).strip().upper()
                censored = nd_val in ("Y", "YES", "TRUE", "1")
            if not censored:
                censored = conc_str.startswith("<") or conc == 0.0 or pd.isna(conc_raw)
            if censored:
                conc_ugl = 0.0

            # Sample date
            sample_date = (
                pd.to_datetime(row.get(date_col, ""), errors="coerce") if date_col else pd.NaT
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
                    "latitude": float("nan"),
                    "longitude": float("nan"),
                }
            )

        result = pd.DataFrame(rows) if rows else _empty_schema_df()

        # Merge facility coordinates if available
        expected = self._config["expected_files"]
        facilities_path = self.raw_dir / expected[1]
        if facilities_path.exists() and not result.empty:
            result = _merge_facility_coords(result, facilities_path)

        result = _sort_and_reset(result)
        logger.info("Parsed %d NJ DEP samples from manual export", len(result))
        return cast(pd.DataFrame, result)

    def _parse_inventory(self) -> pd.DataFrame:
        """Parse inventory + facilities into an inventory-only DataFrame.

        Returns a DataFrame with system info and coordinates but no
        sample concentration data. Useful for spatial analysis even
        without chemical results.
        """
        expected = self._config["expected_files"]
        inv_path = self.raw_dir / expected[0]

        if not inv_path.exists():
            logger.warning("NJ DEP inventory file not found: %s", inv_path)
            return _empty_schema_df()

        try:
            inv_df = pd.read_csv(inv_path, dtype=str, low_memory=False)
        except pd.errors.EmptyDataError:
            inv_df = pd.DataFrame()
        logger.info("Read %d inventory records from NJ DEP", len(inv_df))

        if inv_df.empty:
            return _empty_schema_df()

        # Load facility coordinates
        fac_path = self.raw_dir / expected[1]
        coords: dict[str, tuple[float, float]] = {}
        if fac_path.exists():
            fac_df = pd.read_csv(fac_path, dtype=str, low_memory=False)
            for _, row in fac_df.iterrows():
                pwsid_raw = str(row.get("_PWSID", "")).strip()
                lat = pd.to_numeric(row.get("LATITUDE_MEASURE", ""), errors="coerce")
                lon = pd.to_numeric(row.get("LONGITUDE_MEASURE", ""), errors="coerce")
                if pwsid_raw and pd.notna(lat) and pd.notna(lon):
                    pwsid = normalize_pwsid(pwsid_raw, "NJ")
                    if pwsid not in coords:
                        coords[pwsid] = (float(lat), float(lon))

        # Build inventory-only rows (one row per system, no analyte data)
        rows = []
        for _, row in inv_df.iterrows():
            raw_pwsid = str(row.get("NUMBER0", "")).strip()
            pwsid = normalize_pwsid(raw_pwsid, "NJ")
            lat, lon = coords.get(pwsid, (float("nan"), float("nan")))

            rows.append(
                {
                    "pwsid": pwsid,
                    "analyte": "",
                    "concentration": float("nan"),
                    "unit": "ug/L",
                    "censored": True,
                    "detection_limit": float("nan"),
                    "sample_date": pd.NaT,
                    "latitude": lat,
                    "longitude": lon,
                }
            )

        result = pd.DataFrame(rows) if rows else _empty_schema_df()
        result = _sort_and_reset(result)
        logger.info("Parsed %d NJ DEP inventory records (no sample data)", len(result))
        return cast(pd.DataFrame, result)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate NJ DEP data."""
        df = validate_schema(df, skip_coord_check=True, drop_invalid_rows=True)

        if "analyte" in df.columns:
            non_empty = df[df["analyte"].str.len() > 0]
            if len(non_empty) > 0:
                unique = set(non_empty["analyte"].unique())
                expected = set(NJ_DEP_ANALYTES)
                unexpected = unique - expected
                if unexpected:
                    logger.warning("Unexpected NJ DEP analytes: %s", sorted(unexpected))

        return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _normalize_nj_analyte(name: str) -> str:
    """Normalize NJ DEP analyte names to standard short codes.

    Handles common variations in naming from the waterviewer export.
    """
    # Build reverse lookup from NJ_DEP_ANALYTE_CODES values
    upper = name.upper().strip()

    # Direct match in known constants
    for _code, short_name in NJ_DEP_ANALYTE_CODES.items():
        if upper == short_name.upper():
            return short_name

    # Common long-name patterns
    _LONG_NAME_MAP: dict[str, str] = {
        "PERFLUOROOCTANOIC ACID": "PFOA",
        "PERFLUOROOCTANE SULFONIC ACID": "PFOS",
        "PERFLUORONONANOIC ACID": "PFNA",
        "PERFLUOROHEXANESULFONIC ACID": "PFHxS",
        "PERFLUOROBUTANESULFONIC ACID": "PFBS",
        "PERFLUOROBUTANOIC ACID": "PFBA",
        "PERFLUORODECANOIC ACID": "PFDA",
        "PERFLUORODODECANOIC ACID": "PFDoA",
        "PERFLUOROHEPTANOIC ACID": "PFHpA",
        "PERFLUOROHEPTANESULFONIC ACID": "PFHpS",
        "PERFLUOROHEXANOIC ACID": "PFHxA",
        "PERFLUOROPENTANOIC ACID": "PFPeA",
        "PERFLUOROPENTANESULFONIC ACID": "PFPeS",
        "PERFLUOROTETRADECANOIC ACID": "PFTA",
        "PERFLUOROTRIDECANOIC ACID": "PFTrDA",
        "PERFLUOROUNDECANOIC ACID": "PFUnA",
        "HEXAFLUOROPROPYLENE OXIDE DIMER ACID": "HFPO-DA",
    }
    for long_name, short in _LONG_NAME_MAP.items():
        if upper.startswith(long_name):
            return short

    # Parenthesized abbreviation extraction: "Perfluoro... (PFOA)" -> PFOA
    if "(" in name and ")" in name:
        abbrev = name[name.rindex("(") + 1 : name.rindex(")")].strip()
        if abbrev:
            return _normalize_nj_analyte(abbrev)

    return name


def _merge_facility_coords(df: pd.DataFrame, facilities_path: Path) -> pd.DataFrame:
    """Merge facility lat/lon into a sample DataFrame."""
    try:
        fac_df = pd.read_csv(facilities_path, dtype=str, low_memory=False)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return df

    # Build a lookup of first valid coordinates per normalized PWSID
    fac_df = fac_df.copy()
    fac_df["_pwsid_raw"] = fac_df["_PWSID"].astype(str).str.strip()
    fac_df["_lat"] = pd.to_numeric(fac_df["LATITUDE_MEASURE"], errors="coerce")
    fac_df["_lon"] = pd.to_numeric(fac_df["LONGITUDE_MEASURE"], errors="coerce")
    fac_valid = fac_df[
        (fac_df["_pwsid_raw"] != "") & fac_df["_lat"].notna() & fac_df["_lon"].notna()
    ].copy()
    fac_valid["_pwsid"] = fac_valid["_pwsid_raw"].apply(lambda x: normalize_pwsid(x, "NJ"))
    # Keep first occurrence per PWSID (matches original dict-based dedup)
    coord_lookup = fac_valid.drop_duplicates(subset="_pwsid", keep="first").set_index("_pwsid")[
        ["_lat", "_lon"]
    ]

    if not coord_lookup.empty:
        missing_mask = df["latitude"].isna()
        if missing_mask.any():
            matched = df.loc[missing_mask, "pwsid"].map(coord_lookup["_lat"])
            fill_mask = missing_mask & matched.notna()
            df.loc[fill_mask, "latitude"] = df.loc[fill_mask, "pwsid"].map(coord_lookup["_lat"])
            df.loc[fill_mask, "longitude"] = df.loc[fill_mask, "pwsid"].map(coord_lookup["_lon"])

    return df


def _empty_schema_df() -> pd.DataFrame:
    """Return an empty DataFrame with the standard schema columns."""
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


def _sort_and_reset(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by pwsid/analyte/sample_date and reset index."""
    if df.empty:
        return df
    sort_cols = [c for c in ["pwsid", "analyte", "sample_date"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df
