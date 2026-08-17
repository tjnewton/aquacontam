"""M3: areal-apportionment robustness of environmental-justice burden ratios.

Referee M3: every burden ratio attributes demographics from the *single nearest*
block-group centroid (``sjoin_nearest``), so one block group stands in for a
multi-block-group service area -- a textbook ecological-inference / modifiable-
areal-unit problem that could bias the high/low demographic split in unknown
directions. This script tests whether the geographically held-out test-region
detection-burden ratios are stable when each system's demographics are instead
**population-weighted, areally apportioned** over every block group whose
polygon intersects a buffer (service-area proxy) around the system.

Method
------
* Test-system set, PFOS detection targets, EPA regions and the single-nearest
  demographics are obtained from the *exact* pipeline path used to produce the
  frozen ``equity_analysis.json`` (``aggregate_to_system_level`` +
  ``assemble_with_split_imputation``), so the single-nearest **baseline** burden
  ratios reproduce the frozen values as a gate.
* Apportioned demographics: for each system, buffer its point by ``R`` km
  (EPSG:5070), intersect with TIGER 2023 block-group polygons, and take the
  population-weighted mean of each EJScreen demographic share, where each block
  group's weight is ``ACSTOTPOP * (area of BG within buffer / BG area)``.
* Burden ratios under apportioned demographics are compared to the baseline at
  R = 5 km (primary) and R = 10 km (sensitivity).

Inputs are all local: canonical ``merged_wq``; ``data/raw/tiger_bg/*.zip``
(TIGER 2023 block-group polygons, EPA regions 8/9/10 states); the raw EJScreen
2023 block-group CSV (population + demographic shares). No model and no frozen
number is modified. Output: ``results/paper_frozen/areal_apportionment.json``
and ``paper/tables/table_supp_areal_apportionment.{csv,md}``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
TIGER_DIR = REPO / "data" / "raw" / "tiger_bg"
EJSCREEN_CSV = REPO / "data" / "raw" / "EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.csv"
CRS_DISTANCE = "EPSG:5070"

# Paper demographic column -> EJScreen raw share column.
GROUP_TO_EJ = {
    "pct_people_of_color": "PEOPCOLORPCT",
    "pct_low_income": "LOWINCPCT",
    "pct_limited_english": "LINGISOPCT",
    "pct_less_hs_education": "LESSHSPCT",
}
GROUP_COLS = tuple(GROUP_TO_EJ)
BUFFERS_KM = (5.0, 10.0)
GATE_TOL = 0.05  # baseline burden ratio must reproduce frozen within this


def _discover_downloaded(data_path: Path) -> dict[str, Path]:
    downloaded: dict[str, Path] = {}
    for sub in ("raw", "interim", "processed"):
        d = data_path / sub
        if not d.exists():
            continue
        for p in d.glob("*.parquet"):
            downloaded.setdefault(p.stem, p)
    for p in (data_path / "raw").glob("*.shp"):
        downloaded[p.stem] = p
    for p in (data_path / "raw").rglob("*.tif"):
        if not p.name.startswith("._"):
            downloaded.setdefault(p.stem, p)
    if "us_aquifers" in downloaded:
        downloaded["usgs_aquifers"] = downloaded["us_aquifers"]
    for stem in list(downloaded):
        if "nlcd" in stem.lower():
            downloaded.setdefault("nlcd", downloaded[stem])
    return downloaded


def _prep_test_systems() -> tuple[Any, Any, Any, Any]:
    """Reproduce the frozen equity test set: (wq_df, idx, y_test, demo_single)."""
    import pandas as pd

    import aquacontam.benchmark  # noqa: F401  (register tasks)
    from aquacontam.features.assembly import aggregate_to_system_level, drop_leakage_columns
    from aquacontam.pipeline.assembly import assemble_with_split_imputation
    from aquacontam.pipeline.features import extract_features

    data_path = REPO / "data"
    wq_df = pd.read_parquet(data_path / "interim" / "merged_wq.parquet")
    downloaded = _discover_downloaded(data_path)
    feature_dfs = extract_features(data_path, wq_df, downloaded, skip_large=False)

    sys_targets = aggregate_to_system_level(wq_df, "PFOS", target="detected")
    sys_targets = drop_leakage_columns(sys_targets)
    X, y, _, regions = assemble_with_split_imputation(sys_targets, feature_dfs, wq_df=wq_df)
    test_mask = regions.isin((8, 9, 10))
    idx = X.loc[test_mask].index
    y_test = y.loc[idx]

    demo_df = next(
        df
        for df in feature_dfs
        if isinstance(df, pd.DataFrame) and "pct_people_of_color" in df.columns
    )
    demo_single = demo_df.reindex(idx)
    return wq_df, idx, y_test, demo_single


def _burden_ratios(y_test: Any, demo: Any) -> dict[str, float]:
    """Burden ratio per demographic dimension via the paper's 80/50 split."""
    import numpy as np

    from aquacontam.analysis.equity import analyze_equity

    dummy = np.zeros(len(y_test), dtype=int)
    out: dict[str, float] = {}
    for col in GROUP_COLS:
        if col not in demo.columns:
            continue
        rep = analyze_equity(y_test, dummy, None, demo, col)
        out[col] = float(rep.burden_ratio)
    return out


def _load_block_groups() -> Any:
    """TIGER 2023 block-group polygons joined to EJScreen pop + shares (EPSG:5070)."""
    import geopandas as gpd
    import pandas as pd

    frames = []
    for zp in sorted(TIGER_DIR.glob("tl_2023_*_bg.zip")):
        g = gpd.read_file(f"zip://{zp}")
        geoid_col = next((c for c in ("GEOID", "GEOID20", "GEOID10") if c in g.columns), None)
        if geoid_col is None:
            continue
        g = g[[geoid_col, "geometry"]].rename(columns={geoid_col: "GEOID"})
        frames.append(g)
    bgs = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    bgs["GEOID"] = bgs["GEOID"].astype(str).str.zfill(12)

    usecols = ["ID", "ACSTOTPOP", *GROUP_TO_EJ.values()]
    ej = pd.read_csv(
        EJSCREEN_CSV, usecols=usecols, dtype={"ID": str}, low_memory=False, encoding="latin-1"
    )
    ej["ID"] = ej["ID"].astype(str).str.zfill(12)
    for c in ("ACSTOTPOP", *GROUP_TO_EJ.values()):
        ej[c] = pd.to_numeric(ej[c], errors="coerce")
    bgs = bgs.merge(ej, left_on="GEOID", right_on="ID", how="inner")
    bgs = bgs.to_crs(CRS_DISTANCE)
    bgs = bgs[bgs.geometry.notna() & (bgs["ACSTOTPOP"].fillna(0) > 0)].copy()
    bgs["bg_area"] = bgs.geometry.area
    bgs = bgs[bgs["bg_area"] > 0].copy()
    return bgs


def _apportion(wq_df: Any, idx: Any, bgs: Any, radius_m: float) -> Any:
    """Population-area-weighted demographic shares per system within ``radius_m``."""
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    coords = (
        wq_df[["pwsid", "latitude", "longitude"]]
        .dropna(subset=["latitude", "longitude"])
        .drop_duplicates("pwsid")
        .set_index("pwsid")
        .reindex(idx)
        .dropna(subset=["latitude", "longitude"])
    )
    pts = gpd.GeoDataFrame(
        coords.reset_index(),
        geometry=gpd.points_from_xy(coords["longitude"], coords["latitude"]),
        crs="EPSG:4326",
    ).to_crs(CRS_DISTANCE)
    pts["geometry"] = pts.geometry.buffer(radius_m)

    pairs = gpd.sjoin(pts[["pwsid", "geometry"]], bgs, predicate="intersects", how="inner")
    if pairs.empty:
        return pd.DataFrame(columns=list(GROUP_COLS))
    # Intersection area between each buffer and its matched block group.
    buf_geom = pts.set_index("pwsid").geometry
    left_geom = buf_geom.loc[pairs["pwsid"]].reset_index(drop=True)
    right_geom = bgs.geometry.loc[pairs["index_right"]].reset_index(drop=True)
    inter_area = left_geom.intersection(right_geom, align=False).area.to_numpy()
    weight = pairs["ACSTOTPOP"].to_numpy() * (inter_area / pairs["bg_area"].to_numpy())
    pairs = pairs.assign(_w=weight)

    out: dict[str, Any] = {}
    for col, ejc in GROUP_TO_EJ.items():
        num = pairs.groupby("pwsid").apply(
            lambda g, ejc=ejc: np.nansum(g["_w"] * g[ejc]), include_groups=False
        )
        den = pairs.groupby("pwsid")["_w"].sum()
        out[col] = (num / den.replace(0, np.nan)).rename(col)
    return pd.DataFrame(out).reindex(idx)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    wq_df, idx, y_test, demo_single = _prep_test_systems()
    logger.info("Test systems: %d", len(idx))

    baseline = _burden_ratios(y_test, demo_single)
    logger.info("Baseline (single-nearest) burden ratios: %s", baseline)

    # Gate: baseline POC burden ratio must reproduce frozen equity_analysis.json.
    frozen = {e["group"]: e for e in json.loads((FROZEN / "equity_analysis.json").read_text())}
    gate: dict[str, Any] = {}
    for col in GROUP_COLS:
        if col in frozen and col in baseline:
            dev = abs(baseline[col] - float(frozen[col]["burden_ratio"]))
            gate[col] = {
                "baseline": baseline[col],
                "frozen": float(frozen[col]["burden_ratio"]),
                "abs_dev": dev,
            }
    max_dev = max((g["abs_dev"] for g in gate.values()), default=float("nan"))
    gate_pass = max_dev <= GATE_TOL
    logger.info("Baseline-vs-frozen max dev=%.4f gate_pass=%s", max_dev, gate_pass)

    bgs = _load_block_groups()
    logger.info("Block groups loaded: %d", len(bgs))

    apportioned: dict[str, dict[str, float]] = {}
    for r_km in BUFFERS_KM:
        demo_app = _apportion(wq_df, idx, bgs, r_km * 1000.0)
        ratios = _burden_ratios(y_test, demo_app)
        # restrict y to systems with an apportioned value, per dimension handled
        # inside analyze_equity (NaNs -> "unknown" group, excluded from high/low)
        apportioned[f"{r_km:.0f}km"] = ratios
        logger.info("Apportioned R=%.0fkm burden ratios: %s", r_km, ratios)

    import math

    def _clean(obj: Any) -> Any:
        if isinstance(obj, float):
            return None if (math.isinf(obj) or math.isnan(obj)) else obj
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        return obj

    # pct_limited_english has a degenerate single-nearest baseline (n_low = 0 in
    # the held-out test set), so its burden ratio is undefined; flag it.
    degenerate = [c for c in GROUP_COLS if baseline.get(c) in (float("inf"), float("nan"))]

    out = {
        "baseline_single_nearest": _clean(baseline),
        "apportioned": _clean(apportioned),
        "reproduction_gate": _clean(
            {
                "per_group": gate,
                "max_abs_dev": max_dev,
                "tolerance": GATE_TOL,
                "passed": gate_pass,
            }
        ),
        "_meta": {
            "n_test_systems": len(idx),
            "buffers_km": list(BUFFERS_KM),
            "degenerate_baseline_dimensions": degenerate,
            "method": "population-weighted areal apportionment: weight = ACSTOTPOP * "
            "(BG area within buffer / BG area); demographic shares from EJScreen 2023; "
            "block-group polygons from TIGER 2023 (EPSG:5070).",
            "scope": "geographically held-out test set (EPA regions 8/9/10).",
        },
    }
    (FROZEN / "areal_apportionment.json").write_text(json.dumps(out, indent=2, allow_nan=False))
    logger.info("Wrote %s", FROZEN / "areal_apportionment.json")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
