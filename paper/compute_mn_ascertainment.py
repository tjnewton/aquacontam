"""Minnesota MDH ascertainment corroboration.

The MN MDH bulk export provides far denser PFAS monitoring than the federal UCMR
cycles for the same Minnesota public water systems. This script quantifies how
much of that extra sampling turns into *additional detections*: among MN systems
sampled by BOTH UCMR (federal) and MDH (state), how many that UCMR recorded as a
PFOS non-detect are recorded as detected once MDH's denser longitudinal sampling
is included. A non-trivial flip rate is direct, source-independent corroboration
of the paper's thesis that measured detection tracks monitoring intensity, not
only the underlying environment.

Detection follows the pipeline convention: a system is "detected" for an analyte
if ANY of its samples is uncensored (``~censored``). Inputs are local processed
parquets (``data/processed/{ucmr5,ucmr3,mn_mdh}.parquet``); no model is run and
no frozen number is modified. Output: ``results/mn_ascertainment.json`` (picked
up by the glob-freeze).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
PROCESSED = REPO / "data" / "processed"
OUT_PATH = REPO / "results" / "mn_ascertainment.json"

#: EPA-regulated PFAS (2024 NPDWR) for the "any regulated" detection view.
REGULATED: tuple[str, ...] = ("PFOS", "PFOA", "PFHxS", "PFNA", "PFBS", "HFPO-DA")

_COLS = ["pwsid", "analyte", "censored"]


def _load_mn(source_stems: list[str]) -> pd.DataFrame:
    """Concatenate the given processed sources, restricted to MN PWSIDs."""
    frames: list[pd.DataFrame] = []
    for stem in source_stems:
        path = PROCESSED / f"{stem}.parquet"
        if not path.exists():
            logger.warning("Missing %s — skipping", path)
            continue
        df = pd.read_parquet(path, columns=_COLS)
        df = df[df["pwsid"].astype(str).str.startswith("MN")]
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=_COLS)
    return pd.concat(frames, ignore_index=True)


def _system_detection(df: pd.DataFrame, analytes: tuple[str, ...] | None) -> pd.Series:
    """Per-system detection (any uncensored sample) over the given analytes."""
    sub = df if analytes is None else df[df["analyte"].isin(analytes)]
    if sub.empty:
        return pd.Series(dtype=bool)
    return sub.groupby("pwsid")["censored"].apply(lambda s: bool((~s).any()))


def _ascertainment(
    ucmr: pd.DataFrame, mdh: pd.DataFrame, analytes: tuple[str, ...], label: str
) -> dict[str, Any]:
    """Flip statistics on the UCMR∩MDH overlap for one analyte set."""
    u_det = _system_detection(ucmr, analytes)
    m_det = _system_detection(mdh, analytes)
    overlap = u_det.index.intersection(m_det.index)
    u = u_det.loc[overlap]
    m = m_det.loc[overlap]
    n_ucmr_nondetect = int((~u).sum())
    nd_to_detect = int(((~u) & m).sum())  # UCMR non-detect -> MDH detect
    return {
        "analyte_set": label,
        "n_overlap_systems": len(overlap),
        "n_ucmr_detected": int(u.sum()),
        "n_mdh_detected": int(m.sum()),
        "n_ucmr_nondetect": n_ucmr_nondetect,
        "n_flip_nd_to_detect": nd_to_detect,
        "flip_rate_of_ucmr_nondetect": (
            round(nd_to_detect / n_ucmr_nondetect, 4) if n_ucmr_nondetect else None
        ),
        # systems UCMR called detected but MDH did not (expected small; reported for honesty)
        "n_mdh_nondetect_where_ucmr_detected": int((u & (~m)).sum()),
    }


def _sampling_intensity(
    ucmr: pd.DataFrame, mdh: pd.DataFrame, analyte: str = "PFOS"
) -> dict[str, Any]:
    """Median PFOS samples per system for each source (the density contrast)."""

    def median_samples(df: pd.DataFrame) -> float | None:
        sub = df[df["analyte"] == analyte]
        if sub.empty:
            return None
        return float(sub.groupby("pwsid").size().median())

    return {
        "ucmr_median_pfos_samples_per_system": median_samples(ucmr),
        "mdh_median_pfos_samples_per_system": median_samples(mdh),
    }


def compute() -> dict[str, Any]:
    """Compute the MN ascertainment corroboration statistics."""
    ucmr = _load_mn(["ucmr5", "ucmr3"])
    mdh = _load_mn(["mn_mdh"])
    return {
        "description": (
            "Among MN systems sampled by both UCMR (federal) and MDH (state), the rate "
            "at which UCMR-recorded PFOS non-detects become detected once MDH's denser "
            "sampling is included. Detection = any uncensored sample (pipeline convention)."
        ),
        "n_mn_systems_ucmr": int(ucmr["pwsid"].nunique()),
        "n_mn_systems_mdh": int(mdh["pwsid"].nunique()),
        "pfos": _ascertainment(ucmr, mdh, ("PFOS",), "PFOS"),
        "regulated6": _ascertainment(ucmr, mdh, REGULATED, "regulated6"),
        "sampling_intensity_pfos": _sampling_intensity(ucmr, mdh),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = compute()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote %s", OUT_PATH)
    p = result["pfos"]
    rate = p["flip_rate_of_ucmr_nondetect"]
    logger.info(
        "PFOS: %d/%d UCMR-nondetect MN systems flip to MDH-detect (rate %s)",
        p["n_flip_nd_to_detect"],
        p["n_ucmr_nondetect"],
        f"{rate:.3f}" if rate is not None else "n/a",
    )


if __name__ == "__main__":
    main()
