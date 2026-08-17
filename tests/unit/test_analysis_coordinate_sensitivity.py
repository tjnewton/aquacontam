"""Tests for coordinate perturbation sensitivity analysis."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from aquacontam._constants import CRS_STORAGE
from aquacontam.analysis.coordinate_sensitivity import (
    PerturbationResult,
    coordinate_sensitivity_summary,
    feature_shortcut_perturbation,
    perturb_coordinates,
)


def _make_test_gdf(n: int = 50) -> gpd.GeoDataFrame:
    """Create a test GeoDataFrame with point geometries in EPSG:4326."""
    rng = np.random.RandomState(42)
    lats = rng.uniform(30, 48, n)
    lons = rng.uniform(-120, -70, n)
    geom = [Point(lon, lat) for lon, lat in zip(lons, lats, strict=True)]
    return gpd.GeoDataFrame(
        {"pwsid": [f"SYS{i:07d}" for i in range(n)]},
        geometry=geom,
        crs=CRS_STORAGE,
    ).set_index("pwsid")


class TestPerturbCoordinates:
    def test_zero_magnitude_preserves_coords(self):
        """No perturbation when magnitude is 0."""
        gdf = _make_test_gdf()
        result = perturb_coordinates(gdf, 0.0)
        assert result.crs.to_epsg() == 4326
        # Coordinates should be identical
        orig_x = gdf.geometry.x.to_numpy()
        result_x = result.geometry.x.to_numpy()
        np.testing.assert_array_almost_equal(orig_x, result_x, decimal=6)

    def test_perturbation_scales_with_magnitude(self):
        """Mean displacement should increase with magnitude."""
        gdf = _make_test_gdf()
        displacements = []
        for mag in (1.0, 5.0, 20.0):
            result = perturb_coordinates(gdf, mag, seed=42)
            # Compute displacement in meters (convert to EPSG:5070)
            from aquacontam.geo.crs import to_conus_albers

            orig_proj = to_conus_albers(gdf)
            result_proj = to_conus_albers(result)
            dx = result_proj.geometry.x.to_numpy() - orig_proj.geometry.x.to_numpy()
            dy = result_proj.geometry.y.to_numpy() - orig_proj.geometry.y.to_numpy()
            mean_disp = float(np.mean(np.sqrt(dx**2 + dy**2))) / 1000.0
            displacements.append(mean_disp)
        # Each should be larger than the previous
        assert displacements[0] < displacements[1] < displacements[2]

    def test_different_seeds_differ(self):
        """Different seeds should produce different perturbations."""
        gdf = _make_test_gdf(n=20)
        r1 = perturb_coordinates(gdf, 5.0, seed=1)
        r2 = perturb_coordinates(gdf, 5.0, seed=2)
        # Coordinates should differ
        assert not np.allclose(r1.geometry.x.to_numpy(), r2.geometry.x.to_numpy(), atol=1e-6)

    def test_same_seed_reproducible(self):
        """Same seed should produce identical results."""
        gdf = _make_test_gdf(n=20)
        r1 = perturb_coordinates(gdf, 5.0, seed=42)
        r2 = perturb_coordinates(gdf, 5.0, seed=42)
        np.testing.assert_array_almost_equal(
            r1.geometry.x.to_numpy(), r2.geometry.x.to_numpy(), decimal=6
        )

    def test_crs_preserved(self):
        """Output should always be EPSG:4326."""
        gdf = _make_test_gdf()
        result = perturb_coordinates(gdf, 10.0)
        assert result.crs.to_epsg() == 4326

    def test_empty_gdf(self):
        """Empty GeoDataFrame should return empty."""
        gdf = gpd.GeoDataFrame(
            {"pwsid": pd.Series(dtype=str)},
            geometry=[],
            crs=CRS_STORAGE,
        ).set_index("pwsid")
        result = perturb_coordinates(gdf, 5.0)
        assert result.empty

    def test_negative_magnitude_raises(self):
        """Negative magnitude should raise ValueError."""
        gdf = _make_test_gdf(n=5)
        with pytest.raises(ValueError, match="magnitude_km must be >= 0"):
            perturb_coordinates(gdf, -1.0)

    def test_no_crs_raises(self):
        """GeoDataFrame without CRS should raise ValueError."""
        gdf = _make_test_gdf(n=5)
        gdf = gdf.set_crs(None, allow_override=True)
        with pytest.raises(ValueError, match="no CRS set"):
            perturb_coordinates(gdf, 5.0)


class TestFeatureShortcutPerturbation:
    def _make_feature_dfs(self):
        rng = np.random.RandomState(42)
        n = 20
        pwsids = [f"SYS{i:07d}" for i in range(n)]
        spatial_df = pd.DataFrame(
            {
                "nearest_industrial_km": rng.exponential(5.0, n),
                "count_industrial_5km": rng.poisson(3, n).astype(float),
                "pct_people_of_color": rng.beta(2, 5, n),
            },
            index=pwsids,
        )
        spatial_df.index.name = "pwsid"

        non_spatial_df = pd.DataFrame(
            {"population_served": rng.uniform(100, 100000, n)},
            index=pwsids,
        )
        non_spatial_df.index.name = "pwsid"
        return [spatial_df, non_spatial_df]

    def test_spatial_only_perturbed(self):
        """Only columns with spatial prefixes should change."""
        dfs = self._make_feature_dfs()
        result = feature_shortcut_perturbation(dfs, 5.0, seed=42)
        # Spatial features should differ
        assert not np.allclose(
            dfs[0]["nearest_industrial_km"].to_numpy(),
            result[0]["nearest_industrial_km"].to_numpy(),
        )
        # Non-spatial features should be identical
        np.testing.assert_array_equal(
            dfs[1]["population_served"].to_numpy(),
            result[1]["population_served"].to_numpy(),
        )

    def test_zero_magnitude_no_noise(self):
        """Zero magnitude should return identical copies."""
        dfs = self._make_feature_dfs()
        result = feature_shortcut_perturbation(dfs, 0.0)
        for orig, pert in zip(dfs, result, strict=True):
            pd.testing.assert_frame_equal(orig, pert)

    def test_returns_copies(self):
        """Should return new DataFrames, not modify originals."""
        dfs = self._make_feature_dfs()
        orig_vals = dfs[0]["nearest_industrial_km"].to_numpy().copy()
        feature_shortcut_perturbation(dfs, 10.0)
        np.testing.assert_array_equal(dfs[0]["nearest_industrial_km"].to_numpy(), orig_vals)


class TestCoordinateSensitivitySummary:
    def test_aggregation(self):
        """Summary should correctly aggregate mean/std per magnitude."""
        results = [
            PerturbationResult(magnitude_km=0.0, seed=42, metrics={"auroc": 0.80, "auprc": 0.50}),
            PerturbationResult(magnitude_km=0.0, seed=43, metrics={"auroc": 0.82, "auprc": 0.52}),
            PerturbationResult(magnitude_km=5.0, seed=42, metrics={"auroc": 0.78, "auprc": 0.48}),
            PerturbationResult(magnitude_km=5.0, seed=43, metrics={"auroc": 0.76, "auprc": 0.46}),
        ]
        summary = coordinate_sensitivity_summary(results)
        assert len(summary) == 2
        assert set(summary.columns) >= {
            "magnitude_km",
            "auroc_mean",
            "auroc_std",
            "delta_auroc",
        }
        # Baseline (magnitude=0) delta should be 0
        baseline_row = summary[summary["magnitude_km"] == 0.0].iloc[0]
        assert abs(baseline_row["delta_auroc"]) < 1e-10
        # Magnitude 5 should have negative delta
        mag5_row = summary[summary["magnitude_km"] == 5.0].iloc[0]
        assert mag5_row["delta_auroc"] < 0

    def test_empty_results(self):
        """Empty input should return empty DataFrame with correct columns."""
        summary = coordinate_sensitivity_summary([])
        assert summary.empty
        assert "magnitude_km" in summary.columns
        assert "auroc_mean" in summary.columns

    def test_single_seed_std_zero(self):
        """With one seed per magnitude, std should be 0."""
        results = [
            PerturbationResult(magnitude_km=0.0, seed=42, metrics={"auroc": 0.80, "auprc": 0.50}),
            PerturbationResult(magnitude_km=10.0, seed=42, metrics={"auroc": 0.75, "auprc": 0.45}),
        ]
        summary = coordinate_sensitivity_summary(results)
        for _, row in summary.iterrows():
            assert row["auroc_std"] == 0.0
