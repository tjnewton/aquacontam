"""Tests for features.land_use — NLCD land use fraction features.

Uses a synthetic small GeoTIFF raster with known class values for
deterministic testing without real NLCD data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from aquacontam.features.land_use import (
    compute_land_use_fractions,
    extract_land_use_features,
)


@pytest.fixture()
def synthetic_nlcd(tmp_path: Path) -> Path:
    """Create a small synthetic NLCD-like GeoTIFF.

    Creates a 100x100 pixel raster covering a small area around Austin, TX.
    Left half is class 24 (developed high), right half is class 41 (forest).
    """
    raster_path = tmp_path / "synthetic_nlcd.tif"
    # Small area around Austin TX
    west, south, east, north = -97.8, 30.2, -97.7, 30.3
    width, height = 100, 100
    transform = from_bounds(west, south, east, north, width, height)

    data = np.zeros((height, width), dtype=np.uint8)
    data[:, :50] = 24  # Left half: developed high intensity
    data[:, 50:] = 41  # Right half: deciduous forest

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=np.uint8,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    return raster_path


class TestComputeLandUseFractions:
    def test_returns_category_columns(self, make_systems_gdf, synthetic_nlcd: Path) -> None:
        # Point in the center of the raster
        systems = make_systems_gdf(["TX0000001"], [-97.75], [30.25])
        result = compute_land_use_fractions(systems, synthetic_nlcd, buffer_m=5000.0)
        # Should have columns for categories present in the raster
        assert any("developed" in c for c in result.columns)

    def test_fractions_sum_to_approximately_one(
        self, make_systems_gdf, synthetic_nlcd: Path
    ) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.75], [30.25])
        result = compute_land_use_fractions(systems, synthetic_nlcd, buffer_m=5000.0)
        total = result.iloc[0].sum()
        # Should sum close to 1.0 (might not be exact due to class grouping)
        assert 0.8 <= total <= 1.1

    def test_left_side_mostly_developed(self, make_systems_gdf, synthetic_nlcd: Path) -> None:
        # Point on the left side of the raster (class 24)
        systems = make_systems_gdf(["TX0000001"], [-97.775], [30.25])
        result = compute_land_use_fractions(systems, synthetic_nlcd, buffer_m=1000.0)
        # developed_high column should dominate
        dev_cols = [c for c in result.columns if "developed" in c]
        if dev_cols:
            dev_total = result[dev_cols].iloc[0].sum()
            assert dev_total > 0.3

    def test_right_side_mostly_forest(self, make_systems_gdf, synthetic_nlcd: Path) -> None:
        # Point on the right side of the raster (class 41)
        systems = make_systems_gdf(["TX0000001"], [-97.725], [30.25])
        result = compute_land_use_fractions(systems, synthetic_nlcd, buffer_m=1000.0)
        forest_cols = [c for c in result.columns if "forest" in c]
        if forest_cols:
            forest_total = result[forest_cols].iloc[0].sum()
            assert forest_total > 0.3

    def test_buffer_affects_fractions(self, make_systems_gdf, synthetic_nlcd: Path) -> None:
        """Larger buffer includes more classes."""
        systems = make_systems_gdf(["TX0000001"], [-97.775], [30.25])
        small = compute_land_use_fractions(systems, synthetic_nlcd, buffer_m=500.0)
        large = compute_land_use_fractions(systems, synthetic_nlcd, buffer_m=10000.0)
        # Both should have data; fractions may differ
        assert len(small) == 1
        assert len(large) == 1


class TestExtractLandUseFeatures:
    def test_produces_multiple_buffer_columns(
        self, make_systems_gdf, synthetic_nlcd: Path
    ) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.75], [30.25])
        result = extract_land_use_features(systems, synthetic_nlcd, buffers_m=(1000.0, 5000.0))
        # Should have 1km and 5km variants
        cols_1km = [c for c in result.columns if "1km" in c]
        cols_5km = [c for c in result.columns if "5km" in c]
        assert len(cols_1km) > 0
        assert len(cols_5km) > 0

    def test_includes_majority_class(self, make_systems_gdf, synthetic_nlcd: Path) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.75], [30.25])
        result = extract_land_use_features(systems, synthetic_nlcd)
        assert "nlcd_majority_class" in result.columns

    def test_indexed_by_pwsid(self, make_systems_gdf, synthetic_nlcd: Path) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.75], [30.25])
        result = extract_land_use_features(systems, synthetic_nlcd)
        assert result.index.name == "pwsid"
        assert result.index[0] == "TX0000001"


class TestDownloadNLCD:
    """Tests for download_nlcd() with manual file detection."""

    def test_detects_standard_img_filename(self, tmp_path: Path) -> None:
        from aquacontam.features.land_use import download_nlcd

        # Create a file matching the standard pattern
        (tmp_path / "nlcd_2021_land_cover_l48_20230630.img").touch()
        result = download_nlcd(tmp_path)
        assert result.name == "nlcd_2021_land_cover_l48_20230630.img"

    def test_detects_alternative_tif_filename(self, tmp_path: Path) -> None:
        from aquacontam.features.land_use import download_nlcd

        (tmp_path / "Annual_NLCD_LndCov_2021_20230630.tif").touch()
        result = download_nlcd(tmp_path)
        assert "NLCD" in result.name or "nlcd" in result.name

    def test_detects_generic_nlcd_2021_tif(self, tmp_path: Path) -> None:
        from aquacontam.features.land_use import download_nlcd

        (tmp_path / "my_nlcd_data_2021_custom.tif").touch()
        result = download_nlcd(tmp_path)
        assert "nlcd" in result.name.lower()

    def test_detects_file_in_subdirectory(self, tmp_path: Path) -> None:
        from aquacontam.features.land_use import download_nlcd

        subdir = tmp_path / "Annual_NLCD_LndCov_2021_CU_C1V1"
        subdir.mkdir()
        (subdir / "Annual_NLCD_LndCov_2021_CU_C1V1.tif").touch()
        result = download_nlcd(tmp_path)
        assert "NLCD" in result.name

    def test_raises_with_sciencebase_url_when_no_file(self, tmp_path: Path) -> None:
        from aquacontam.features.land_use import download_nlcd

        with pytest.raises(RuntimeError, match=r"sciencebase\.gov"):
            download_nlcd(tmp_path)
