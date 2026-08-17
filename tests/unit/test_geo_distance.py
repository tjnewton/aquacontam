"""Tests for geo.distance — spatial distance calculations."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from aquacontam._constants import CRS_STORAGE
from aquacontam.geo.distance import (
    build_spatial_index,
    count_within_radius,
    kernel_density_at_points,
    nearest_distances,
    query_k_nearest,
)


def _make_targets_gdf(
    lons: list[float],
    lats: list[float],
) -> gpd.GeoDataFrame:
    """Helper: create a target facilities GeoDataFrame."""
    geometry = [Point(lon, lat) for lon, lat in zip(lons, lats, strict=True)]
    return gpd.GeoDataFrame(
        {"id": range(len(lons))},
        geometry=geometry,
        crs=CRS_STORAGE,
    )


class TestBuildSpatialIndex:
    def test_builds_kdtree(self) -> None:
        targets = _make_targets_gdf([-97.0, -98.0], [30.0, 31.0])
        tree = build_spatial_index(targets)
        assert tree.n == 2

    def test_single_point(self) -> None:
        targets = _make_targets_gdf([-97.0], [30.0])
        tree = build_spatial_index(targets)
        assert tree.n == 1


class TestNearestDistances:
    def test_same_location_zero_distance(self, make_systems_gdf) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.0], [30.0])
        targets = _make_targets_gdf([-97.0], [30.0])
        dist = nearest_distances(systems, targets)
        assert abs(dist.iloc[0]) < 1.0  # < 1 meter

    def test_known_distance_approximate(self, make_systems_gdf) -> None:
        """Austin TX to Dallas TX is ~300 km."""
        systems = make_systems_gdf(["TX0000001"], [-97.74], [30.27])
        targets = _make_targets_gdf([-96.80], [32.78])
        dist = nearest_distances(systems, targets)
        # Should be roughly 270-310 km in Albers projection
        assert 250_000 < dist.iloc[0] < 350_000

    def test_empty_targets_returns_inf(self, make_systems_gdf) -> None:
        systems = make_systems_gdf(["CA0000001"], [-118.24], [34.05])
        targets = _make_targets_gdf([], [])
        dist = nearest_distances(systems, targets)
        assert dist.iloc[0] == np.inf

    def test_multiple_systems(self, make_systems_gdf) -> None:
        systems = make_systems_gdf(
            ["TX0000001", "CA0000001"],
            [-97.74, -118.24],
            [30.27, 34.05],
        )
        # Target near TX, far from CA
        targets = _make_targets_gdf([-97.74], [30.27])
        dist = nearest_distances(systems, targets)
        assert dist.iloc[0] < dist.iloc[1]

    def test_returns_series_indexed_by_pwsid(self, make_systems_gdf) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.74], [30.27])
        targets = _make_targets_gdf([-96.80], [32.78])
        dist = nearest_distances(systems, targets)
        assert dist.index[0] == "TX0000001"
        assert dist.name == "dist_nearest"


class TestCountWithinRadius:
    def test_single_target_within_radius(self, make_systems_gdf) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.74], [30.27])
        # Target ~500m away (very close)
        targets = _make_targets_gdf([-97.735], [30.27])
        count = count_within_radius(systems, targets, 5000.0)
        assert count.iloc[0] >= 1

    def test_no_targets_within_radius(self, make_systems_gdf) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.74], [30.27])
        # Target very far away (California)
        targets = _make_targets_gdf([-118.24], [34.05])
        count = count_within_radius(systems, targets, 1000.0)
        assert count.iloc[0] == 0

    def test_empty_targets_returns_zero(self, make_systems_gdf) -> None:
        systems = make_systems_gdf(["CA0000001"], [-118.24], [34.05])
        targets = _make_targets_gdf([], [])
        count = count_within_radius(systems, targets, 5000.0)
        assert count.iloc[0] == 0

    def test_counts_monotonically_increase_with_radius(self, make_systems_gdf) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.74], [30.27])
        # Multiple targets at varying distances
        targets = _make_targets_gdf(
            [-97.735, -97.72, -97.65, -97.5],
            [30.27, 30.27, 30.27, 30.27],
        )
        c1 = count_within_radius(systems, targets, 1000.0).iloc[0]
        c5 = count_within_radius(systems, targets, 5000.0).iloc[0]
        c10 = count_within_radius(systems, targets, 50000.0).iloc[0]
        assert c1 <= c5 <= c10


class TestQueryKNearest:
    def test_shape(self, make_systems_gdf) -> None:
        systems = make_systems_gdf(["TX0000001", "CA0000001"], [-97.74, -118.24], [30.27, 34.05])
        targets = _make_targets_gdf([-97.74, -97.72, -97.70], [30.27, 30.27, 30.27])
        dists, indices = query_k_nearest(systems, targets, k=2)
        assert dists.shape == (2, 2)
        assert indices.shape == (2, 2)

    def test_nearest_is_first(self, make_systems_gdf) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.74], [30.27])
        targets = _make_targets_gdf([-97.74, -97.0], [30.27, 30.27])
        dists, _ = query_k_nearest(systems, targets, k=2)
        assert dists[0, 0] <= dists[0, 1]

    def test_empty_targets(self, make_systems_gdf) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.74], [30.27])
        targets = _make_targets_gdf([], [])
        dists, indices = query_k_nearest(systems, targets, k=3)
        assert dists.shape == (1, 3)
        assert np.all(np.isinf(dists))
        assert np.all(indices == -1)

    def test_k_larger_than_targets(self, make_systems_gdf) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.74], [30.27])
        targets = _make_targets_gdf([-97.74], [30.27])
        dists, _indices = query_k_nearest(systems, targets, k=5)
        assert dists.shape == (1, 5)
        assert not np.isinf(dists[0, 0])
        assert np.all(np.isinf(dists[0, 1:]))


class TestKernelDensity:
    def test_density_nonnegative(self, make_systems_gdf) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.74], [30.27])
        targets = _make_targets_gdf([-97.74, -97.72], [30.27, 30.27])
        density = kernel_density_at_points(systems, targets, 5000.0)
        assert density.iloc[0] >= 0

    def test_closer_targets_higher_density(self, make_systems_gdf) -> None:
        """Systems closer to targets should have higher density."""
        systems = make_systems_gdf(["TX0000001", "CA0000001"], [-97.74, -118.24], [30.27, 34.05])
        # Targets near TX
        targets = _make_targets_gdf([-97.74, -97.72, -97.70], [30.27, 30.27, 30.27])
        density = kernel_density_at_points(systems, targets, 50000.0)
        assert density.iloc[0] > density.iloc[1]

    def test_empty_targets_zero_density(self, make_systems_gdf) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.74], [30.27])
        targets = _make_targets_gdf([], [])
        density = kernel_density_at_points(systems, targets, 5000.0)
        assert density.iloc[0] == 0.0
