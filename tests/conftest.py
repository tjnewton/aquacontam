"""Shared test fixtures for AquaContam."""

from __future__ import annotations

import os

# Set the deterministic cuBLAS workspace BEFORE anything imports torch or creates
# a CUDA context. On a GPU box the first torch model fit (e.g. in
# test_benchmark_tuning) initializes the process-global cuBLAS handle; if
# CUBLAS_WORKSPACE_CONFIG is not already set at that moment, a later
# torch.use_deterministic_algorithms(True, warn_only=True) can only warn, and the
# TestReproducibilityTorch same-seed fits diverge (nondeterministic GEMM) — a
# full-suite-only failure. Setting it here, at import time, guarantees the handle
# is always created with a deterministic workspace regardless of test order.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from aquacontam._constants import CRS_STORAGE

# ---------------------------------------------------------------------------
# Global state cleanup: clear module-level caches between tests.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_module_caches():
    """Clear spatial KD-tree cache after each test to prevent collisions.

    Python can reuse object IDs for garbage-collected GeoDataFrames, causing
    stale cache hits when two different GeoDataFrames share the same
    ``(id, len)`` cache key.
    """
    yield
    from aquacontam.geo.distance import clear_tree_cache

    clear_tree_cache()


@pytest.fixture()
def tmp_data_dirs(tmp_path: Path) -> dict[str, Path]:
    """Create temporary raw/interim/processed directories."""
    dirs = {}
    for name in ("raw", "interim", "processed"):
        d = tmp_path / name
        d.mkdir()
        dirs[name] = d
    return dirs


@pytest.fixture()
def make_systems_gdf():
    """Factory fixture for creating water system point GeoDataFrames."""

    def _make(
        pwsids: list[str],
        lons: list[float],
        lats: list[float],
    ) -> gpd.GeoDataFrame:
        geometry = [Point(lon, lat) for lon, lat in zip(lons, lats, strict=True)]
        return gpd.GeoDataFrame(
            {"pwsid": pwsids},
            geometry=geometry,
            crs=CRS_STORAGE,
        )

    return _make


@pytest.fixture()
def sample_water_quality_df() -> pd.DataFrame:
    """Minimal water quality DataFrame for testing."""
    return pd.DataFrame(
        {
            "pwsid": ["CA0101001", "TX0200002", "MI0300003"],
            "analyte": ["PFOS", "PFOA", "PFHxS"],
            "concentration": [12.5, 0.0, 8.3],
            "unit": ["ng/L", "ng/L", "ng/L"],
            "censored": [False, True, False],
            "detection_limit": [2.0, 4.0, 2.0],
            "sample_date": pd.to_datetime(["2023-01-15", "2023-02-20", "2023-03-10"]),
            "latitude": [34.05, 30.27, 42.33],
            "longitude": [-118.24, -97.74, -83.05],
        }
    )


@pytest.fixture()
def synthetic_wq_df() -> pd.DataFrame:
    """Synthetic water quality DataFrame with 100 systems across 10 EPA regions.

    Two analytes: PFOS (~80% censored), PFOA (~50% censored).
    Each system has 3 samples per analyte.
    """
    rng = np.random.RandomState(42)

    # Build systems across all 10 EPA regions
    state_coords: dict[str, tuple[float, float]] = {
        "CT": (41.6, -72.7),
        "NJ": (40.2, -74.7),
        "PA": (40.3, -76.9),
        "GA": (33.7, -84.4),
        "IL": (40.6, -89.4),
        "TX": (31.0, -97.0),
        "KS": (38.5, -98.8),
        "CO": (39.5, -105.0),
        "CA": (36.8, -119.4),
        "WA": (47.4, -120.5),
    }
    # 10 systems per state (10 states x 10 = 100 systems)
    pwsids: list[str] = []
    lats: list[float] = []
    lons: list[float] = []
    for state, (lat, lon) in state_coords.items():
        for i in range(10):
            pwsid = f"{state}{i + 1:07d}"
            pwsids.append(pwsid)
            lats.append(lat + rng.normal(0, 0.5))
            lons.append(lon + rng.normal(0, 0.5))

    rows = []
    for j, pwsid in enumerate(pwsids):
        for analyte, censor_rate in [("PFOS", 0.80), ("PFOA", 0.50)]:
            for sample_idx in range(3):
                is_censored = rng.random() < censor_rate
                dl = rng.uniform(1.0, 5.0)
                conc = 0.0 if is_censored else dl + rng.exponential(10.0)
                rows.append(
                    {
                        "pwsid": pwsid,
                        "analyte": analyte,
                        "concentration": conc,
                        "unit": "ug/L",
                        "censored": is_censored,
                        "detection_limit": dl,
                        "sample_date": pd.Timestamp("2023-01-15")
                        + pd.Timedelta(days=sample_idx * 30),
                        "latitude": lats[j],
                        "longitude": lons[j],
                    }
                )

    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_feature_dfs() -> list[pd.DataFrame]:
    """Synthetic feature DataFrames matching the 100-system synthetic_wq_df.

    Returns proximity, land use, and aquifer features indexed by pwsid.
    """
    rng = np.random.RandomState(42)

    state_coords = {
        "CT": (41.6, -72.7),
        "NJ": (40.2, -74.7),
        "PA": (40.3, -76.9),
        "GA": (33.7, -84.4),
        "IL": (40.6, -89.4),
        "TX": (31.0, -97.0),
        "KS": (38.5, -98.8),
        "CO": (39.5, -105.0),
        "CA": (36.8, -119.4),
        "WA": (47.4, -120.5),
    }
    pwsids = []
    for state in state_coords:
        for i in range(10):
            pwsids.append(f"{state}{i + 1:07d}")

    # Proximity features
    proximity_df = pd.DataFrame(
        {
            "nearest_industrial_km": rng.exponential(5.0, 100),
            "nearest_military_km": rng.exponential(20.0, 100),
            "nearest_wwtp_km": rng.exponential(3.0, 100),
            "count_industrial_5km": rng.poisson(2, 100),
            "count_wwtp_10km": rng.poisson(5, 100),
        },
        index=pwsids,
    )
    proximity_df.index.name = "pwsid"

    # Land use features
    land_use_df = pd.DataFrame(
        {
            "frac_developed_1km": rng.beta(2, 5, 100),
            "frac_agriculture_1km": rng.beta(1, 3, 100),
            "frac_forest_1km": rng.beta(1, 2, 100),
            "frac_developed_5km": rng.beta(2, 5, 100),
            "frac_agriculture_5km": rng.beta(1, 3, 100),
        },
        index=pwsids,
    )
    land_use_df.index.name = "pwsid"

    # Aquifer features (with some NaN to test imputation)
    aquifer_types = rng.choice(
        ["sandstone", "carbonate", "unconsolidated", "igneous", np.nan],
        100,
        p=[0.3, 0.2, 0.3, 0.1, 0.1],
    )
    aquifer_df = pd.DataFrame(
        {
            "aquifer_type": aquifer_types,
            "aquifer_depth_m": rng.uniform(10, 500, 100),
        },
        index=pwsids,
    )
    aquifer_df.index.name = "pwsid"

    return [proximity_df, land_use_df, aquifer_df]


@pytest.fixture()
def synthetic_feature_dfs_with_demographics(
    synthetic_feature_dfs: list[pd.DataFrame],
) -> list[pd.DataFrame]:
    """Extend synthetic_feature_dfs with a demographics DataFrame.

    Appends a DataFrame with ``pct_people_of_color``, ``pct_low_income``,
    and ``pct_less_hs_education`` columns (floats 0-1) indexed by pwsid.
    """
    rng = np.random.RandomState(99)
    pwsids = synthetic_feature_dfs[0].index.tolist()

    demo_df = pd.DataFrame(
        {
            "pct_people_of_color": rng.beta(2, 5, len(pwsids)),
            "pct_low_income": rng.beta(2, 5, len(pwsids)),
            "pct_less_hs_education": rng.beta(1, 8, len(pwsids)),
        },
        index=pwsids,
    )
    demo_df.index.name = "pwsid"

    return [*synthetic_feature_dfs, demo_df]


@pytest.fixture()
def synthetic_multi_pfas_df() -> pd.DataFrame:
    """Synthetic water quality DataFrame with 50 systems and 5 PFAS analytes.

    Varying censoring rates per analyte for multilabel (T3) testing.
    """
    rng = np.random.RandomState(123)

    state_coords: dict[str, tuple[float, float]] = {
        "CT": (41.6, -72.7),
        "NJ": (40.2, -74.7),
        "PA": (40.3, -76.9),
        "GA": (33.7, -84.4),
        "IL": (40.6, -89.4),
        "TX": (31.0, -97.0),
        "KS": (38.5, -98.8),
        "CO": (39.5, -105.0),
        "CA": (36.8, -119.4),
        "WA": (47.4, -120.5),
    }
    # 5 systems per state (10 states x 5 = 50 systems)
    pwsids: list[str] = []
    lats: list[float] = []
    lons: list[float] = []
    for state, (lat, lon) in state_coords.items():
        for i in range(5):
            pwsids.append(f"{state}{i + 1:07d}")
            lats.append(lat + rng.normal(0, 0.3))
            lons.append(lon + rng.normal(0, 0.3))

    analyte_censor_rates = {
        "PFOS": 0.60,
        "PFOA": 0.40,
        "PFBS": 0.85,
        "PFHxS": 0.70,
        "HFPO-DA": 0.90,
    }

    rows = []
    for j, pwsid in enumerate(pwsids):
        for analyte, censor_rate in analyte_censor_rates.items():
            for sample_idx in range(3):
                is_censored = rng.random() < censor_rate
                dl = rng.uniform(1.0, 5.0)
                conc = 0.0 if is_censored else dl + rng.exponential(10.0)
                rows.append(
                    {
                        "pwsid": pwsid,
                        "analyte": analyte,
                        "concentration": conc,
                        "unit": "ug/L",
                        "censored": is_censored,
                        "detection_limit": dl,
                        "sample_date": pd.Timestamp("2023-01-15")
                        + pd.Timedelta(days=sample_idx * 30),
                        "latitude": lats[j],
                        "longitude": lons[j],
                    }
                )

    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_ucmr3_df() -> pd.DataFrame:
    """Synthetic UCMR3-era water quality DataFrame with 40 systems.

    6 shared PFAS analytes, date range 2013-2015.
    """
    rng = np.random.RandomState(200)

    state_coords: dict[str, tuple[float, float]] = {
        "CT": (41.6, -72.7),
        "NJ": (40.2, -74.7),
        "PA": (40.3, -76.9),
        "GA": (33.7, -84.4),
        "IL": (40.6, -89.4),
        "TX": (31.0, -97.0),
        "KS": (38.5, -98.8),
        "CO": (39.5, -105.0),
        "CA": (36.8, -119.4),
        "WA": (47.4, -120.5),
    }
    pwsids: list[str] = []
    lats: list[float] = []
    lons: list[float] = []
    for state, (lat, lon) in state_coords.items():
        for i in range(4):
            pwsids.append(f"{state}{i + 1:07d}")
            lats.append(lat + rng.normal(0, 0.3))
            lons.append(lon + rng.normal(0, 0.3))

    analytes = ["PFOS", "PFOA", "PFNA", "PFHxS", "PFBS", "PFHpA"]
    rows = []
    for j, pwsid in enumerate(pwsids):
        for analyte in analytes:
            censor_rate = 0.75
            for _sample_idx in range(2):
                is_censored = rng.random() < censor_rate
                dl = rng.uniform(10.0, 50.0)
                conc = 0.0 if is_censored else dl + rng.exponential(20.0)
                rows.append(
                    {
                        "pwsid": pwsid,
                        "analyte": analyte,
                        "concentration": conc,
                        "unit": "ng/L",
                        "censored": is_censored,
                        "detection_limit": dl,
                        "sample_date": pd.Timestamp("2013-06-01")
                        + pd.Timedelta(days=rng.randint(0, 730)),
                        "latitude": lats[j],
                        "longitude": lons[j],
                    }
                )

    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_ucmr5_df() -> pd.DataFrame:
    """Synthetic UCMR5-era water quality DataFrame with 60 systems.

    30 systems overlap with synthetic_ucmr3_df (same PWS IDs),
    30 are new. 6 shared PFAS analytes, date range 2023-2025.
    """
    rng = np.random.RandomState(300)

    state_coords: dict[str, tuple[float, float]] = {
        "CT": (41.6, -72.7),
        "NJ": (40.2, -74.7),
        "PA": (40.3, -76.9),
        "GA": (33.7, -84.4),
        "IL": (40.6, -89.4),
        "TX": (31.0, -97.0),
        "KS": (38.5, -98.8),
        "CO": (39.5, -105.0),
        "CA": (36.8, -119.4),
        "WA": (47.4, -120.5),
    }
    # First 3 per state overlap with UCMR3 (same IDs), next 3 are new
    pwsids: list[str] = []
    lats: list[float] = []
    lons: list[float] = []
    for state, (lat, lon) in state_coords.items():
        for i in range(6):
            pwsids.append(f"{state}{i + 1:07d}")
            lats.append(lat + rng.normal(0, 0.3))
            lons.append(lon + rng.normal(0, 0.3))

    analytes = ["PFOS", "PFOA", "PFNA", "PFHxS", "PFBS", "PFHpA"]
    rows = []
    for j, pwsid in enumerate(pwsids):
        for analyte in analytes:
            censor_rate = 0.70
            for _sample_idx in range(3):
                is_censored = rng.random() < censor_rate
                dl = rng.uniform(2.0, 10.0)
                conc = 0.0 if is_censored else dl + rng.exponential(15.0)
                rows.append(
                    {
                        "pwsid": pwsid,
                        "analyte": analyte,
                        "concentration": conc,
                        "unit": "ng/L",
                        "censored": is_censored,
                        "detection_limit": dl,
                        "sample_date": pd.Timestamp("2023-01-01")
                        + pd.Timedelta(days=rng.randint(0, 730)),
                        "latitude": lats[j],
                        "longitude": lons[j],
                    }
                )

    return pd.DataFrame(rows)
