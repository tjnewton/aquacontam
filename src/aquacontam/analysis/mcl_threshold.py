"""MCL-threshold exceedance analysis.

Supplementary analysis comparing detection-based (T1) targets with
EPA Maximum Contaminant Level (MCL) exceedance targets. The EPA MCLs
for PFAS took effect April 2024 and represent the regulatory-relevant
threshold for monitoring prioritization.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import pandas as pd

logger = logging.getLogger(__name__)

#: Per-analyte exceedance thresholds in ug/L (EPA, effective April 2024). PFOS/PFOA/PFHxS/
#: HFPO-DA are individual MCLs; PFBS 2.0 is its Hazard-Index Health-Based Water Concentration
#: (NOT an individual MCL). PFNA (individual MCL 0.010) is omitted here for source-coverage
#: parity. Values are frozen: changing them would alter results/paper_frozen/mcl_exceedance.
DEFAULT_MCL_THRESHOLDS: dict[str, float] = {
    "PFOS": 0.004,
    "PFOA": 0.004,
    "PFHxS": 0.010,
    "HFPO-DA": 0.010,
    "PFBS": 2.0,  # Hazard-Index HBWC, not an individual MCL (see note above)
}


def compute_mcl_exceedance(
    wq_df: pd.DataFrame,
    mcl_thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute per-system MCL exceedance for regulated PFAS analytes.

    For each system (PWSID), determines whether any sample of a regulated
    analyte exceeds the EPA MCL threshold. Systems with only censored
    (non-detect) samples are classified as non-exceedant.

    Parameters
    ----------
    wq_df : pd.DataFrame
        Merged water quality data with columns: ``pwsid``, ``analyte``,
        ``concentration``, ``censored``.
    mcl_thresholds : dict[str, float] or None
        Mapping analyte name → MCL in ug/L. Defaults to EPA 2024 MCLs.

    Returns
    -------
    pd.DataFrame
        One row per PWSID with columns:
        - ``pwsid``: system identifier
        - ``mcl_exceedance``: 1 if any regulated analyte exceeds MCL, 0 otherwise
        - ``max_pfas_ratio``: maximum concentration/MCL ratio across analytes
        - ``n_analytes_tested``: number of regulated analytes with data
        - ``n_analytes_exceeding``: number exceeding MCL
    """
    if mcl_thresholds is None:
        mcl_thresholds = DEFAULT_MCL_THRESHOLDS

    required = {"pwsid", "analyte", "concentration", "censored"}
    if not required.issubset(wq_df.columns):
        missing = required - set(wq_df.columns)
        raise ValueError(f"Missing columns: {missing}")

    # Filter to regulated analytes only
    regulated = wq_df[wq_df["analyte"].isin(mcl_thresholds)]
    if regulated.empty:
        logger.warning("No regulated analytes found in data")
        return cast(
            pd.DataFrame,
            pd.DataFrame(columns=["pwsid", "mcl_exceedance", "max_pfas_ratio"]),
        )

    # For each system+analyte, find max concentration
    detected = regulated[~regulated["censored"]]

    rows: list[dict[str, Any]] = []
    for pwsid, group in regulated.groupby("pwsid"):
        det_group = detected[detected["pwsid"] == pwsid]
        n_tested = group["analyte"].nunique()
        n_exceeding = 0
        max_ratio = 0.0

        for analyte, mcl in mcl_thresholds.items():
            analyte_data = det_group[det_group["analyte"] == analyte]
            if analyte_data.empty:
                continue
            max_conc = analyte_data["concentration"].max()
            ratio = max_conc / mcl if mcl > 0 else 0.0
            max_ratio = max(max_ratio, ratio)
            if max_conc >= mcl:
                n_exceeding += 1

        rows.append(
            {
                "pwsid": pwsid,
                "mcl_exceedance": int(n_exceeding > 0),
                "max_pfas_ratio": max_ratio,
                "n_analytes_tested": n_tested,
                "n_analytes_exceeding": n_exceeding,
            }
        )

    return cast(pd.DataFrame, pd.DataFrame(rows))


def mcl_summary_statistics(
    mcl_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute summary statistics for MCL exceedance analysis.

    Parameters
    ----------
    mcl_df : pd.DataFrame
        Output from ``compute_mcl_exceedance``.

    Returns
    -------
    dict
        Summary including exceedance rate, class balance, and ratio stats.
    """
    if mcl_df.empty:
        return {"n_systems": 0, "exceedance_rate": 0.0}

    n = len(mcl_df)
    n_exceed = mcl_df["mcl_exceedance"].sum()

    return {
        "n_systems": n,
        "n_exceeding": int(n_exceed),
        "n_non_exceeding": int(n - n_exceed),
        "exceedance_rate": float(n_exceed / n),
        "class_balance": f"{n_exceed}:{n - n_exceed}",
        "mean_max_ratio": float(mcl_df["max_pfas_ratio"].mean()),
        "median_max_ratio": float(mcl_df["max_pfas_ratio"].median()),
        "p95_max_ratio": float(mcl_df["max_pfas_ratio"].quantile(0.95)),
    }
