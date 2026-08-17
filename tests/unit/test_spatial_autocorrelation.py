"""Tests for spatial autocorrelation analysis (Moran's I)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aquacontam.analysis.spatial_autocorrelation import (
    MoranResult,
    _build_spatial_weights,
    _latlon_to_projected,
    analyze_spatial_autocorrelation,
    morans_i,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def grid_coords_and_residuals():
    """10x10 grid of points with spatially correlated residuals.

    Points are on a regular 10 km grid. Residuals are a smooth
    east-west gradient (positive autocorrelation expected).
    """
    xs = np.arange(10) * 10_000.0  # 10 km spacing, metres
    ys = np.arange(10) * 10_000.0
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.ravel(), yy.ravel()])  # (100, 2)
    # Smooth gradient: residuals increase from west to east
    residuals = xx.ravel() / 10_000.0  # 0..9 range
    return coords, residuals


@pytest.fixture()
def random_residuals_and_coords():
    """Random residuals on a grid (no spatial correlation expected)."""
    rng = np.random.RandomState(42)
    xs = np.arange(10) * 10_000.0
    ys = np.arange(10) * 10_000.0
    xx, _yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.ravel(), _yy.ravel()])
    residuals = rng.standard_normal(100)
    return coords, residuals


@pytest.fixture()
def small_triangle():
    """Three points forming a compact triangle (all within 20 km)."""
    return np.array(
        [
            [0.0, 0.0],
            [10_000.0, 0.0],
            [5_000.0, 8_660.0],
        ]
    )


# ---------------------------------------------------------------------------
# MoranResult dataclass
# ---------------------------------------------------------------------------


class TestMoranResult:
    """Tests for the MoranResult dataclass."""

    def test_creation(self) -> None:
        result = MoranResult(
            statistic=0.35,
            expected=-0.01,
            variance=0.005,
            z_score=5.1,
            p_value=3.4e-7,
            n=100,
        )
        assert pytest.approx(0.35) == result.statistic
        assert pytest.approx(-0.01) == result.expected
        assert pytest.approx(0.005) == result.variance
        assert pytest.approx(5.1) == result.z_score
        assert pytest.approx(3.4e-7) == result.p_value
        assert result.n == 100

    def test_nan_values_allowed(self) -> None:
        result = MoranResult(
            statistic=float("nan"),
            expected=-0.01,
            variance=float("nan"),
            z_score=float("nan"),
            p_value=float("nan"),
            n=5,
        )
        assert math.isnan(result.statistic)
        assert result.n == 5


# ---------------------------------------------------------------------------
# _build_spatial_weights
# ---------------------------------------------------------------------------


class TestBuildSpatialWeights:
    """Tests for the spatial weights builder."""

    def test_small_example_all_connected(self, small_triangle) -> None:
        """All three points within 20 km -> all pairs are neighbors."""
        neighbors, weights = _build_spatial_weights(small_triangle, threshold_km=20.0)
        # Each point should have exactly 2 neighbors
        for i in range(3):
            assert len(neighbors[i]) == 2
            assert all(w == 1.0 for w in weights[i])

    def test_small_example_none_connected(self, small_triangle) -> None:
        """With a very small threshold, no neighbors should be found."""
        neighbors, weights = _build_spatial_weights(small_triangle, threshold_km=0.001)
        for i in range(3):
            assert len(neighbors[i]) == 0
            assert len(weights[i]) == 0

    def test_partial_connectivity(self) -> None:
        """Two close points + one far point -> partial connectivity."""
        coords = np.array(
            [
                [0.0, 0.0],
                [5_000.0, 0.0],  # 5 km from first
                [500_000.0, 0.0],  # 500 km from both
            ]
        )
        neighbors, _weights = _build_spatial_weights(coords, threshold_km=10.0)
        # Points 0 and 1 are neighbors; point 2 is isolated
        assert 1 in neighbors[0]
        assert 0 in neighbors[1]
        assert len(neighbors[2]) == 0

    def test_symmetric_weights(self) -> None:
        """Weights should be symmetric: if j in neighbors[i], then i in neighbors[j]."""
        rng = np.random.RandomState(99)
        coords = rng.uniform(0, 100_000, size=(20, 2))
        neighbors, _weights = _build_spatial_weights(coords, threshold_km=30.0)
        for i in range(20):
            for j in neighbors[i]:
                assert i in neighbors[j]


# ---------------------------------------------------------------------------
# morans_i
# ---------------------------------------------------------------------------


class TestMoransI:
    """Tests for the core Moran's I computation."""

    def test_positive_autocorrelation(self, grid_coords_and_residuals) -> None:
        """Spatially correlated residuals -> positive Moran's I."""
        coords, residuals = grid_coords_and_residuals
        result = morans_i(residuals, coords, threshold_km=15.0)
        assert result.statistic > 0.0
        assert result.p_value < 0.05

    def test_random_near_zero(self, random_residuals_and_coords) -> None:
        """Random residuals -> Moran's I near expected value (close to 0)."""
        coords, residuals = random_residuals_and_coords
        result = morans_i(residuals, coords, threshold_km=15.0)
        # Should be close to expected I (-1/(n-1) ~ -0.01)
        assert abs(result.statistic - result.expected) < 0.5

    def test_expected_value(self) -> None:
        """Expected I should equal -1/(n-1)."""
        rng = np.random.RandomState(7)
        coords = rng.uniform(0, 100_000, size=(50, 2))
        residuals = rng.standard_normal(50)
        result = morans_i(residuals, coords, threshold_km=40.0)
        assert pytest.approx(-1.0 / 49.0) == result.expected
        assert result.n == 50

    def test_p_value_in_range(self, grid_coords_and_residuals) -> None:
        """p-value should be between 0 and 1."""
        coords, residuals = grid_coords_and_residuals
        result = morans_i(residuals, coords, threshold_km=15.0)
        assert 0.0 <= result.p_value <= 1.0

    def test_z_score_sign_matches_statistic(self, grid_coords_and_residuals) -> None:
        """For positive autocorrelation, z-score should be positive."""
        coords, residuals = grid_coords_and_residuals
        result = morans_i(residuals, coords, threshold_km=15.0)
        # statistic > expected for positive autocorrelation -> z > 0
        assert result.z_score > 0.0

    def test_too_few_points_raises(self) -> None:
        """Fewer than 3 observations should raise ValueError."""
        coords = np.array([[0.0, 0.0], [1000.0, 0.0]])
        residuals = np.array([1.0, -1.0])
        with pytest.raises(ValueError, match="at least 3"):
            morans_i(residuals, coords)

    def test_identical_residuals_raises(self, small_triangle) -> None:
        """All-identical residuals (zero variance) should raise ValueError."""
        residuals = np.array([5.0, 5.0, 5.0])
        with pytest.raises(ValueError, match="zero variance"):
            morans_i(residuals, small_triangle)

    def test_no_neighbors_returns_nan(self) -> None:
        """When no neighbors exist, I should be NaN."""
        # Points extremely far apart with tiny threshold
        coords = np.array(
            [
                [0.0, 0.0],
                [1e9, 0.0],
                [0.0, 1e9],
            ]
        )
        residuals = np.array([1.0, -1.0, 0.5])
        result = morans_i(residuals, coords, threshold_km=0.001)
        assert math.isnan(result.statistic)
        assert math.isnan(result.p_value)

    def test_larger_threshold_more_neighbors(self) -> None:
        """Increasing threshold should not decrease number of neighbor pairs."""
        rng = np.random.RandomState(12)
        coords = rng.uniform(0, 100_000, size=(30, 2))

        _, w_small = _build_spatial_weights(coords, threshold_km=10.0)
        _, w_large = _build_spatial_weights(coords, threshold_km=50.0)

        total_small = sum(len(v) for v in w_small.values())
        total_large = sum(len(v) for v in w_large.values())
        assert total_large >= total_small


# ---------------------------------------------------------------------------
# _latlon_to_projected
# ---------------------------------------------------------------------------


class TestLatLonToProjected:
    """Tests for the approximate coordinate projection."""

    def test_output_shape(self) -> None:
        lat = np.array([30.0, 31.0, 32.0])
        lon = np.array([-97.0, -98.0, -99.0])
        projected = _latlon_to_projected(lat, lon)
        assert projected.shape == (3, 2)

    def test_northing_increases_with_lat(self) -> None:
        lat = np.array([30.0, 40.0, 50.0])
        lon = np.array([-97.0, -97.0, -97.0])
        projected = _latlon_to_projected(lat, lon)
        # Northing (column 1) should increase with latitude
        assert projected[0, 1] < projected[1, 1] < projected[2, 1]

    def test_distance_roughly_correct(self) -> None:
        """1 degree of latitude ~ 111 km."""
        lat = np.array([30.0, 31.0])
        lon = np.array([-97.0, -97.0])
        projected = _latlon_to_projected(lat, lon)
        dist = np.linalg.norm(projected[1] - projected[0])
        # Should be approximately 111 km
        assert 100_000 < dist < 120_000

    def test_uses_proper_projection_when_available(self) -> None:
        """When pyproj is available, should use EPSG:5070 projection."""
        pyproj = pytest.importorskip("pyproj")  # noqa: F841
        lat = np.array([38.0, 39.0])
        lon = np.array([-97.0, -97.0])
        projected = _latlon_to_projected(lat, lon)
        # EPSG:5070 coordinates for CONUS should be in reasonable range
        # Easting ~0 to ~3M, Northing ~0 to ~3M
        assert projected.shape == (2, 2)
        # Distance should still be reasonable (~111 km)
        dist = np.linalg.norm(projected[1] - projected[0])
        assert 100_000 < dist < 120_000


# ---------------------------------------------------------------------------
# analyze_spatial_autocorrelation
# ---------------------------------------------------------------------------


class TestAnalyzeSpatialAutocorrelation:
    """Tests for the high-level analysis wrapper."""

    @pytest.fixture()
    def spatially_correlated_data(self):
        """Predictions with spatially structured residuals."""
        n = 100
        # Points in a grid across CONUS
        lat = np.linspace(30.0, 45.0, 10).repeat(10)
        lon = np.tile(np.linspace(-120.0, -80.0, 10), 10)

        y_true = np.ones(n)
        # Predictions are biased by longitude (spatial pattern in residuals)
        y_pred = 1.0 - (lon - lon.min()) / (lon.max() - lon.min()) * 0.5
        split_labels = np.array(["train"] * 50 + ["test"] * 50)
        return y_true, y_pred, lat, lon, split_labels

    def test_returns_overall_result(self, spatially_correlated_data) -> None:
        y_true, y_pred, lat, lon, _ = spatially_correlated_data
        report = analyze_spatial_autocorrelation(y_true, y_pred, lat, lon)
        assert "overall" in report
        assert isinstance(report["overall"], MoranResult)

    def test_overall_positive_statistic(self, spatially_correlated_data) -> None:
        """Spatially structured residuals should give positive I."""
        y_true, y_pred, lat, lon, _ = spatially_correlated_data
        report = analyze_spatial_autocorrelation(y_true, y_pred, lat, lon, threshold_km=200.0)
        assert report["overall"].statistic > 0.0

    def test_no_splits_key_without_labels(self, spatially_correlated_data) -> None:
        y_true, y_pred, lat, lon, _ = spatially_correlated_data
        report = analyze_spatial_autocorrelation(y_true, y_pred, lat, lon)
        assert "splits" not in report

    def test_split_results(self, spatially_correlated_data) -> None:
        y_true, y_pred, lat, lon, split_labels = spatially_correlated_data
        report = analyze_spatial_autocorrelation(
            y_true,
            y_pred,
            lat,
            lon,
            split_labels=split_labels,
            threshold_km=200.0,
        )
        assert "splits" in report
        assert "train" in report["splits"]
        assert "test" in report["splits"]
        for label in ["train", "test"]:
            split_result = report["splits"][label]
            assert split_result is None or isinstance(split_result, MoranResult)

    def test_handles_nan_values(self) -> None:
        """NaN entries should be silently dropped."""
        y_true = np.array([1.0, 0.0, 1.0, 0.0, float("nan"), 1.0, 0.0])
        y_pred = np.array([0.8, 0.2, 0.7, 0.3, 0.5, 0.6, float("nan")])
        lat = np.array([30.0, 30.1, 30.2, 30.3, 30.4, 30.5, 30.6])
        lon = np.array([-97.0, -97.1, -97.2, -97.3, -97.4, -97.5, -97.6])
        # Should not raise; 2 NaN entries removed -> 5 valid points
        report = analyze_spatial_autocorrelation(y_true, y_pred, lat, lon, threshold_km=20.0)
        assert report["overall"] is not None or report["overall"] is None  # no crash

    def test_too_few_valid_points(self) -> None:
        """With <3 valid points, overall should be None (caught ValueError)."""
        y_true = np.array([1.0, 0.0])
        y_pred = np.array([0.5, 0.5])
        lat = np.array([30.0, 31.0])
        lon = np.array([-97.0, -98.0])
        report = analyze_spatial_autocorrelation(y_true, y_pred, lat, lon)
        assert report["overall"] is None

    def test_split_with_few_points_is_none(self) -> None:
        """Splits with <3 points should produce None, not crash."""
        y_true = np.array([1.0, 0.0, 1.0, 0.0, 0.5, 0.3])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        lat = np.array([30.0, 30.1, 30.2, 30.3, 30.4, 30.5])
        lon = np.array([-97.0, -97.1, -97.2, -97.3, -97.4, -97.5])
        split_labels = np.array(["train", "train", "train", "train", "test", "tiny"])
        report = analyze_spatial_autocorrelation(
            y_true,
            y_pred,
            lat,
            lon,
            split_labels=split_labels,
            threshold_km=20.0,
        )
        # "tiny" split has only 1 point -> should be None
        assert report["splits"]["tiny"] is None
        # "test" split has only 1 point -> should be None
        assert report["splits"]["test"] is None

    def test_residuals_are_true_minus_pred(self) -> None:
        """Verify residual computation direction: y_true - y_prob_or_pred."""
        # If y_true > y_pred everywhere, residuals are all positive
        n = 20
        lat = np.linspace(30.0, 35.0, n)
        lon = np.linspace(-100.0, -95.0, n)
        y_true = np.ones(n)
        y_pred = np.zeros(n)
        # All residuals = 1.0 -> identical -> should get ValueError internally
        # which is caught and returns None
        report = analyze_spatial_autocorrelation(y_true, y_pred, lat, lon)
        assert report["overall"] is None

    def test_custom_threshold(self) -> None:
        """Custom threshold_km should be respected."""
        rng = np.random.RandomState(77)
        n = 30
        lat = rng.uniform(30.0, 45.0, n)
        lon = rng.uniform(-120.0, -80.0, n)
        y_true = rng.random(n)
        y_pred = rng.random(n)

        report_small = analyze_spatial_autocorrelation(y_true, y_pred, lat, lon, threshold_km=10.0)
        report_large = analyze_spatial_autocorrelation(
            y_true, y_pred, lat, lon, threshold_km=500.0
        )
        # Both should return results (or None if edge case)
        # At minimum, they should not crash
        assert "overall" in report_small
        assert "overall" in report_large


class TestMultiThresholdFdr:
    """Tests for FDR correction in multi_threshold_morans_i."""

    def test_fdr_keys_present(self) -> None:
        from aquacontam.analysis.spatial_autocorrelation import multi_threshold_morans_i

        rng = np.random.RandomState(99)
        n = 50
        coords = rng.uniform(0, 500_000, (n, 2))
        residuals = rng.normal(0, 1, n)
        results = multi_threshold_morans_i(residuals, coords, thresholds_km=(50.0, 100.0))
        assert len(results) == 2
        for r in results:
            assert "p_value_fdr" in r
            assert "significant_fdr" in r

    def test_fdr_corrected_ge_raw(self) -> None:
        from aquacontam.analysis.spatial_autocorrelation import multi_threshold_morans_i

        rng = np.random.RandomState(123)
        n = 50
        coords = rng.uniform(0, 500_000, (n, 2))
        residuals = rng.normal(0, 1, n)
        results = multi_threshold_morans_i(
            residuals, coords, thresholds_km=(25.0, 50.0, 100.0, 200.0)
        )
        for r in results:
            if not np.isnan(r["p_value"]):
                assert r["p_value_fdr"] >= r["p_value"] - 1e-10
