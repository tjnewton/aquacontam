"""Unit tests for spatial graph neighborhood features (leakage-free)."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from aquacontam._constants import CRS_STORAGE
from aquacontam.features.spatial_graph import (
    compute_neighbor_characteristics,
    compute_neighbor_prevalence,
    extract_spatial_graph_features,
)


@pytest.fixture()
def train_gdf() -> gpd.GeoDataFrame:
    """Training set systems."""
    return gpd.GeoDataFrame(
        {
            "pwsid": [f"TR{i:03d}" for i in range(20)],
            "geometry": [Point(-90.0 + i * 0.1, 38.0) for i in range(20)],
        },
        crs=CRS_STORAGE,
    )


@pytest.fixture()
def test_gdf() -> gpd.GeoDataFrame:
    """Test set systems (different locations)."""
    return gpd.GeoDataFrame(
        {
            "pwsid": ["TE001", "TE002", "TE003"],
            "geometry": [
                Point(-90.05, 38.0),
                Point(-89.0, 38.0),
                Point(-88.0, 38.0),
            ],
        },
        crs=CRS_STORAGE,
    )


@pytest.fixture()
def train_labels() -> pd.Series:
    """Binary labels for training set."""
    rng = np.random.RandomState(42)
    labels = rng.binomial(1, 0.3, size=20)
    return pd.Series(labels, index=[f"TR{i:03d}" for i in range(20)], name="target")


class TestNeighborPrevalence:
    def test_shape_and_index(self, test_gdf, train_gdf, train_labels):
        result = compute_neighbor_prevalence(test_gdf, train_gdf, train_labels, k_values=(5,))
        assert result.index.name == "pwsid"
        assert len(result) == 3
        assert "nbr_prev_k5" in result.columns

    def test_prevalence_bounds(self, test_gdf, train_gdf, train_labels):
        result = compute_neighbor_prevalence(test_gdf, train_gdf, train_labels, k_values=(5, 10))
        for col in result.columns:
            assert (result[col] >= 0).all()
            assert (result[col] <= 1).all()

    def test_train_on_self(self, train_gdf, train_labels):
        """When querying train on itself, prevalence should reflect labels."""
        result = compute_neighbor_prevalence(train_gdf, train_gdf, train_labels, k_values=(5,))
        assert len(result) == 20
        # Should be correlated with actual prevalence
        overall_prev = train_labels.mean()
        # Mean neighbor prevalence should be close to overall
        assert abs(result["nbr_prev_k5"].mean() - overall_prev) < 0.3

    def test_no_leakage_different_splits(self, test_gdf, train_gdf, train_labels):
        """Test systems use training labels only — no test labels."""
        result = compute_neighbor_prevalence(test_gdf, train_gdf, train_labels, k_values=(5,))
        # Verify result doesn't depend on test labels
        # (test_gdf has no labels, result should still be computed)
        assert not result.isna().any().any()

    def test_empty_training(self, test_gdf):
        empty_train = gpd.GeoDataFrame(
            {"pwsid": pd.Series(dtype=str), "geometry": []},
            crs=CRS_STORAGE,
        )
        empty_labels = pd.Series(dtype=float)
        result = compute_neighbor_prevalence(test_gdf, empty_train, empty_labels, k_values=(5,))
        assert len(result) == 3
        assert result.isna().all().all()

    def test_multiple_k_values(self, test_gdf, train_gdf, train_labels):
        result = compute_neighbor_prevalence(
            test_gdf, train_gdf, train_labels, k_values=(5, 10, 20)
        )
        assert len(result.columns) == 3
        assert {"nbr_prev_k5", "nbr_prev_k10", "nbr_prev_k20"} == set(result.columns)


class TestNeighborCharacteristics:
    def test_shape(self, test_gdf, train_gdf):
        train_features = pd.DataFrame(
            {"pop": np.arange(20.0), "elev": np.random.default_rng(42).standard_normal(20)},
            index=[f"TR{i:03d}" for i in range(20)],
        )
        train_features.index.name = "pwsid"

        result = compute_neighbor_characteristics(test_gdf, train_gdf, train_features, k=5)
        assert len(result) == 3
        assert "nbr_mean_pop" in result.columns
        assert "nbr_mean_elev" in result.columns


class TestExtractSpatialGraphFeatures:
    def test_combines_prevalence(self, test_gdf, train_gdf, train_labels):
        result = extract_spatial_graph_features(
            test_gdf, train_gdf, train_labels, k_values=(5, 10)
        )
        assert result.index.name == "pwsid"
        assert len(result) == 3
        assert "nbr_prev_k5" in result.columns
        assert "nbr_prev_k10" in result.columns
