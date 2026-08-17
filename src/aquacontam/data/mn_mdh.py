"""Minnesota Dept. of Health (MDH) PFAS data source — manual bulk export.

Parses the MDH public-water-system PFAS monitoring bulk export into the
standardized water quality schema. There is no public API or bulk-download
endpoint; the data is provided as an Excel workbook on request
(``health.drinkingwater@state.mn.us``), so :meth:`MnMdhSource.download` verifies
a manually-staged file rather than fetching (mirrors :mod:`aquacontam.data.nj_dep`).

Three source quirks are handled in :meth:`MnMdhSource.parse`:

* **ANALYTE_GROUP replication** — each measurement is repeated across the
  ``PFAS_533`` / ``PFC_EXPANDED`` / ``PFAS_Regulated`` panels. Rows are collapsed
  to one per ``(PWS_ID, SAMPLE_ID, ANALYTE, COLLECTION_DATE)``.
* **Mixed units** — concentrations arrive in ng/L or ug/L; both are harmonized
  to ug/L per row.
* **Left-censoring by substitution** — non-detects are reported as
  ``RESULT == REPORTING_LIMIT`` (~79% of rows); these and any below-RL estimates
  are marked censored with concentration ``0.0``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from aquacontam._config import load_data_config
from aquacontam._constants import MN_MDH_ANALYTES, PPT_TO_UGL
from aquacontam.data._utils import normalize_pwsid
from aquacontam.data.base import DataSource
from aquacontam.data.schema import validate_schema

logger = logging.getLogger(__name__)

# Map the exact MDH analyte strings (full chemical name + parenthetical
# abbreviation) to the project's canonical short codes. A full exact-match dict
# is used rather than parenthesis regex extraction because two source strings
# have truncated/unclosed parens ("...(11CI-PF3", "...(9CI-PF3O") and several
# abbreviations contain non-word characters (":" and spaces) a ``\(\w+\)``
# pattern cannot capture. Seven canonical names differ from the source
# abbreviation (e.g. FPMPA -> PFMPA, PFDoDA -> PFDoA, PFUDA -> PFUnA, the
# n:n FTS spacing, and the CI -> Cl casing).
_MN_MDH_ANALYTE_MAP: dict[str, str] = {
    "11-chloroeicosafluoro-3-oxandecane-1-sulfonic acid (11CI-PF3": "11Cl-PF3OUdS",
    "9-Chlorohexadecafluoro-3-oxanonane-1-sulfonic acid (9CI-PF3O": "9Cl-PF3ONS",
    "1H,1H,2H,2H-Perfluorodecane sulfonic acid (8:2FTS)": "8:2 FTS",
    "1H,1H,2H,2H-Perfluorohexane sulfonic acid (4:2FTS)": "4:2 FTS",
    "1H,1H,2H,2H-Perfluorooctane sulfonic acid (6:2 FtS)": "6:2 FTS",
    "4,8-Dioxa-3H-perfluorononanoic acid (ADONA)": "ADONA",
    "Hexafluoropropylene oxide dimer acide (HFPO-DA)": "HFPO-DA",
    "N-ethyl perfluorooctanesulfonaminoacetic acid (NEtFOSAA)": "NEtFOSAA",
    "Nonafluoro-3, 6-dioxaheptanoic acid (NFDHA)": "NFDHA",
    "Perfluoro(2-ethoxyethane)sulfonic acid (PFEESA)": "PFEESA",
    "Perfluoro-3-methoxypropanoic acid (FPMPA)": "PFMPA",
    "Perfluoro-4-methoxybutanoic acid (PFMBA)": "PFMBA",
    "Perfluorobutanesulfonate (PFBS)": "PFBS",
    "Perfluorobutanoic acid (PFBA)": "PFBA",
    "Perfluorodecanoic acid (PFDA)": "PFDA",
    "Perfluorododecanoic acid (PFDoDA)": "PFDoA",
    "Perfluoroheptanoic acid (PFHpA)": "PFHpA",
    "Perfluoroheptasulfonate (PFHpS)": "PFHpS",
    "Perfluorohexanesulfonate (PFHxS)": "PFHxS",
    "Perfluorohexanoic acid (PFHxA)": "PFHxA",
    "Perfluorononanoic acid (PFNA)": "PFNA",
    "Perfluorooctane sulfonamide (FOSA)": "FOSA",
    "Perfluorooctanesulfonate (PFOS)": "PFOS",
    "Perfluorooctanoic acid (PFOA)": "PFOA",
    "Perfluoropentanoic acid (PFPeA)": "PFPeA",
    "Perfluoropentasulfonate (PFPeS)": "PFPeS",
    "Perfluoroundecanoic acid (PFUDA)": "PFUnA",
}

# Per-row multiplier from the source unit to ug/L: ng/L (== PPT) scales by
# PPT_TO_UGL (0.001); ug/L passes through unchanged.
_UNIT_FACTORS: dict[str, float] = {"NG/L": PPT_TO_UGL, "UG/L": 1.0}

#: Keys that collapse the ANALYTE_GROUP replication to one row per measurement.
_DEDUP_KEYS = ["PWS_ID", "SAMPLE_ID", "ANALYTE", "COLLECTION_DATE"]


def _normalize_mn_mdh_analyte(name: str) -> str:
    """Map an exact MDH analyte string to its canonical short code.

    Unmapped names are returned stripped (so :meth:`MnMdhSource.validate` can
    surface them) rather than silently dropped.
    """
    return _MN_MDH_ANALYTE_MAP.get(name.strip(), name.strip())


class MnMdhSource(DataSource):
    """Minnesota MDH public-water-system PFAS monitoring loader.

    The MDH bulk export is a single-sheet Excel workbook keyed by a 7-digit
    integer ``PWS_ID`` (normalized to the federal ``"MN" + 7 digit`` form), with
    one row per analyte per sample replicated across regulatory panels. The
    loader collapses that replication, harmonizes mixed ng/L and ug/L units to
    ug/L, and represents substitution-at-the-reporting-limit non-detects as
    left-censored rows with concentration ``0.0``.

    Parameters
    ----------
    raw_dir : Path
        Directory containing the manually-staged raw export.
    interim_dir : Path
        Directory for intermediate artifacts.
    processed_dir : Path
        Directory for the final Parquet output.
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
        self._config = load_data_config(config_path, source_name="mn_mdh")

    @property
    def name(self) -> str:
        return "mn_mdh"

    def _resolve_raw_file(self) -> Path:
        """Locate the staged MDH export.

        Tries each configured ``expected_files`` entry, then the original
        emailed ``source_file`` name. Raises :class:`FileNotFoundError` (a caught
        type in :func:`aquacontam.pipeline.download.download_data`, so the
        pipeline degrades gracefully) with staging instructions if absent.
        """
        candidates: list[str] = list(self._config.get("expected_files") or [])
        source_file = self._config.get("source_file")
        if source_file:
            candidates.append(source_file)
        for filename in candidates:
            path = self.raw_dir / filename
            if path.exists():
                return path
        target = candidates[0] if candidates else "mn_mdh_pfas.xlsx"
        raise FileNotFoundError(
            f"MN MDH export not found. Place the emailed MDH PFAS workbook at "
            f"{self.raw_dir / target}. There is no public API/bulk download; "
            "request a bulk export from health.drinkingwater@state.mn.us."
        )

    def download(self, *, force: bool = False, **kwargs: Any) -> list[Path]:
        """Verify the manually-staged MDH export is present (no remote fetch)."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        return [self._resolve_raw_file()]

    def parse(self, **kwargs: Any) -> pd.DataFrame:
        """Parse the MDH export into the standardized water quality schema."""
        path = self._resolve_raw_file()
        df = pd.read_excel(path, sheet_name=0)
        logger.info("Read %d rows from MN MDH export %s", len(df), path.name)

        # --- 1. Per-row unit factor (vectorized); fail loud on an unmapped unit.
        unit_norm = df["UNIT_MEASURE_CODE"].astype(str).str.strip().str.upper()
        factor = unit_norm.map(_UNIT_FACTORS)
        if factor.isna().any():
            bad = sorted(unit_norm[factor.isna()].unique())
            raise ValueError(f"Unrecognized MN MDH unit(s): {bad}")

        result_val = pd.to_numeric(df["RESULT"], errors="coerce")
        reporting_limit = pd.to_numeric(df["REPORTING_LIMIT"], errors="coerce")

        work = pd.DataFrame(
            {
                "PWS_ID": df["PWS_ID"],
                "SAMPLE_ID": df["SAMPLE_ID"].astype(str),
                "ANALYTE": df["ANALYTE"].astype(str),
                "COLLECTION_DATE": pd.to_datetime(df["COLLECTION_DATE"], errors="coerce"),
                "_result": result_val,
                "_rl": reporting_limit,
                "_factor": factor.astype(float),
                "_norm_rl": reporting_limit * factor.astype(float),
            }
        )

        # --- 2. Collapse ANALYTE_GROUP replication to one row per measurement.
        # Sort by the ug/L-normalized reporting limit ascending so the
        # most-sensitive (lowest-RL) reading wins for the ~63 ng/L-vs-ug/L value
        # pairs AND the ~17 groups whose redundant rows disagree on
        # detect/non-detect; deterministic regardless of input row order.
        before = len(work)
        work = (
            work.sort_values([*_DEDUP_KEYS, "_norm_rl"], kind="mergesort")
            .drop_duplicates(subset=_DEDUP_KEYS, keep="first")
            .reset_index(drop=True)
        )
        logger.info("Collapsed ANALYTE_GROUP replication: %d -> %d rows", before, len(work))

        # --- 3. Build the standardized output.
        out = pd.DataFrame(index=work.index)
        out["pwsid"] = (
            work["PWS_ID"].astype("int64").astype(str).map(lambda raw: normalize_pwsid(raw, "MN"))
        )
        out["analyte"] = work["ANALYTE"].map(_normalize_mn_mdh_analyte)

        # detect <=> RESULT strictly above the reporting limit. Everything at or
        # below RL (the ~79% ND-at-RL substitution + ~9.5% below-RL estimates)
        # is left-censored with concentration 0.0 — matching how ucmr5/mo_dnr
        # set ``censored``/``concentration`` so the downstream ``any_detected``
        # target is consistent across sources.
        censored = (work["_result"] <= work["_rl"]).to_numpy()
        out["censored"] = censored
        out["concentration"] = np.where(
            censored, 0.0, (work["_result"] * work["_factor"]).to_numpy()
        ).astype(float)
        out["detection_limit"] = (work["_rl"] * work["_factor"]).astype(float)
        out["unit"] = "ug/L"
        out["sample_date"] = work["COLLECTION_DATE"]
        # No coordinates in the source; backfilled by pwsid in preprocessing.
        out["latitude"] = np.nan
        out["longitude"] = np.nan

        out = out[out["analyte"].str.len() > 0]
        out = out.sort_values(["pwsid", "analyte", "sample_date"]).reset_index(drop=True)
        logger.info("Parsed %d MN MDH samples", len(out))
        return cast(pd.DataFrame, out)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate against the standard schema (the source has no coordinates)."""
        df = validate_schema(df, skip_coord_check=True, drop_invalid_rows=True)

        if "analyte" in df.columns:
            unexpected = set(df["analyte"].unique()) - set(MN_MDH_ANALYTES)
            if unexpected:
                logger.warning("Unexpected MN MDH analytes: %s", sorted(unexpected))

        return df
