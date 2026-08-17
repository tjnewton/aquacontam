"""Tests for EJScreen data loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aquacontam._constants import EJSCREEN_DEMOGRAPHIC_COLS, EJSCREEN_EJ_INDEX_COLS


def _write_mock_bg_gazetteer(path: Path, n_rows: int = 10) -> None:
    """Create a mock Census block group gazetteer for testing."""
    rows = []
    for i in range(n_rows):
        rows.append(
            {
                "GEOID": f"36001000{i:03d}",
                "INTPTLAT": 40.0 + i * 0.1,
                "INTPTLONG": -74.0 + i * 0.1,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False)


def _write_mock_ejscreen_csv(
    path: Path, n_rows: int = 10, *, include_coords: bool = False
) -> None:
    """Create a mock EJScreen CSV file for testing.

    Parameters
    ----------
    path : Path
        Output CSV path.
    n_rows : int
        Number of rows to generate.
    include_coords : bool
        If True, include LATITUDE/LONGITUDE columns (original EPA format).
        If False, omit them (Zenodo 2023 format).
    """
    rows = []
    for i in range(n_rows):
        row: dict = {
            "ID": f"36001000{i:03d}",
            "PEOPCOLORPCT": 0.3 + i * 0.01,
            "LOWINCPCT": 0.2 + i * 0.01,
            "LINGISOPCT": 0.05 + i * 0.005,
            "LESSHSPCT": 0.1 + i * 0.01,
            "UNDER5PCT": 0.06,
            "OVER64PCT": 0.15,
            "P_PWDIS": 50 + i,
            "P_PNPL": 30 + i,
            "P_PRMP": 20 + i,
            "P_PTSDF": 40 + i,
            "DEMOGIDX_5": 0.5 + i * 0.01,
        }
        if include_coords:
            row["LATITUDE"] = 40.0 + i * 0.1
            row["LONGITUDE"] = -74.0 + i * 0.1
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


class TestLoadEJScreen:
    """Tests for load_ejscreen()."""

    def test_loads_with_valid_data(self, tmp_path: Path) -> None:
        from aquacontam.data.ejscreen import load_ejscreen

        csv_path = tmp_path / "EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.csv"
        _write_mock_ejscreen_csv(csv_path)
        _write_mock_bg_gazetteer(tmp_path / "2020_Gaz_bg_national.txt")
        gdf = load_ejscreen(tmp_path)
        assert len(gdf) > 0
        assert "block_group_id" in gdf.columns

    def test_has_crs(self, tmp_path: Path) -> None:
        from aquacontam.data.ejscreen import load_ejscreen

        csv_path = tmp_path / "EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.csv"
        _write_mock_ejscreen_csv(csv_path)
        _write_mock_bg_gazetteer(tmp_path / "2020_Gaz_bg_national.txt")
        gdf = load_ejscreen(tmp_path)
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 4326

    def test_demographic_columns_present(self, tmp_path: Path) -> None:
        from aquacontam.data.ejscreen import load_ejscreen

        csv_path = tmp_path / "EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.csv"
        _write_mock_ejscreen_csv(csv_path)
        _write_mock_bg_gazetteer(tmp_path / "2020_Gaz_bg_national.txt")
        gdf = load_ejscreen(tmp_path)
        for col in EJSCREEN_DEMOGRAPHIC_COLS:
            assert col in gdf.columns, f"Missing column: {col}"

    def test_ej_index_columns_present(self, tmp_path: Path) -> None:
        from aquacontam.data.ejscreen import load_ejscreen

        csv_path = tmp_path / "EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.csv"
        _write_mock_ejscreen_csv(csv_path)
        _write_mock_bg_gazetteer(tmp_path / "2020_Gaz_bg_national.txt")
        gdf = load_ejscreen(tmp_path)
        # At least some EJ index columns should be present
        found = [c for c in EJSCREEN_EJ_INDEX_COLS if c in gdf.columns]
        assert len(found) > 0

    def test_filters_outside_conus(self, tmp_path: Path) -> None:
        from aquacontam.data.ejscreen import load_ejscreen

        csv_path = tmp_path / "EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.csv"
        # Write gazetteer with one Alaska and one CONUS centroid
        gaz = pd.DataFrame(
            {
                "GEOID": ["020010001", "360010002"],
                "INTPTLAT": [64.0, 40.0],
                "INTPTLONG": [-153.0, -74.0],
            }
        )
        gaz.to_csv(tmp_path / "2020_Gaz_bg_national.txt", sep="\t", index=False)

        rows = [
            {
                "ID": "020010001",
                "PEOPCOLORPCT": 0.3,
                "LOWINCPCT": 0.2,
                "LINGISOPCT": 0.05,
                "LESSHSPCT": 0.1,
                "UNDER5PCT": 0.06,
                "OVER64PCT": 0.15,
                "P_PWDIS": 50,
                "P_PNPL": 30,
                "P_PRMP": 20,
                "P_PTSDF": 40,
                "DEMOGIDX_5": 0.5,
            },
            {
                "ID": "360010002",
                "PEOPCOLORPCT": 0.4,
                "LOWINCPCT": 0.3,
                "LINGISOPCT": 0.06,
                "LESSHSPCT": 0.12,
                "UNDER5PCT": 0.07,
                "OVER64PCT": 0.16,
                "P_PWDIS": 55,
                "P_PNPL": 35,
                "P_PRMP": 25,
                "P_PTSDF": 45,
                "DEMOGIDX_5": 0.55,
            },
        ]
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        gdf = load_ejscreen(tmp_path)
        # Only the CONUS row should remain
        assert len(gdf) == 1

    def test_file_not_found(self, tmp_path: Path) -> None:
        from aquacontam.data.ejscreen import load_ejscreen

        with pytest.raises(FileNotFoundError):
            load_ejscreen(tmp_path)

    def test_handles_missing_coordinates_via_gazetteer(self, tmp_path: Path) -> None:
        """Block groups without gazetteer matches should be dropped."""
        from aquacontam.data.ejscreen import load_ejscreen

        csv_path = tmp_path / "EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.csv"
        # Gazetteer only has the second block group
        gaz = pd.DataFrame(
            {
                "GEOID": ["360010002"],
                "INTPTLAT": [40.0],
                "INTPTLONG": [-74.0],
            }
        )
        gaz.to_csv(tmp_path / "2020_Gaz_bg_national.txt", sep="\t", index=False)

        rows = [
            {
                "ID": "360010001",
                "PEOPCOLORPCT": 0.3,
                "LOWINCPCT": 0.2,
                "LINGISOPCT": 0.05,
                "LESSHSPCT": 0.1,
                "UNDER5PCT": 0.06,
                "OVER64PCT": 0.15,
                "P_PWDIS": 50,
                "P_PNPL": 30,
                "P_PRMP": 20,
                "P_PTSDF": 40,
                "DEMOGIDX_5": 0.5,
            },
            {
                "ID": "360010002",
                "PEOPCOLORPCT": 0.4,
                "LOWINCPCT": 0.3,
                "LINGISOPCT": 0.06,
                "LESSHSPCT": 0.12,
                "UNDER5PCT": 0.07,
                "OVER64PCT": 0.16,
                "P_PWDIS": 55,
                "P_PNPL": 35,
                "P_PRMP": 25,
                "P_PTSDF": 45,
                "DEMOGIDX_5": 0.55,
            },
        ]
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        gdf = load_ejscreen(tmp_path)
        # Only the row with a gazetteer match should remain
        assert len(gdf) == 1

    def test_numeric_coercion(self, tmp_path: Path) -> None:
        from aquacontam.data.ejscreen import load_ejscreen

        csv_path = tmp_path / "EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.csv"
        _write_mock_ejscreen_csv(csv_path)
        _write_mock_bg_gazetteer(tmp_path / "2020_Gaz_bg_national.txt")
        gdf = load_ejscreen(tmp_path)
        # Demographic columns should be numeric
        assert pd.api.types.is_numeric_dtype(gdf["pct_people_of_color"])
        assert pd.api.types.is_numeric_dtype(gdf["pct_low_income"])

    def test_custom_columns(self, tmp_path: Path) -> None:
        from aquacontam.data.ejscreen import load_ejscreen

        csv_path = tmp_path / "EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.csv"
        _write_mock_ejscreen_csv(csv_path)
        _write_mock_bg_gazetteer(tmp_path / "2020_Gaz_bg_national.txt")
        gdf = load_ejscreen(tmp_path, columns=["pct_people_of_color", "pct_low_income"])
        assert "pct_people_of_color" in gdf.columns
        assert "pct_low_income" in gdf.columns

    def test_loads_with_inline_coordinates(self, tmp_path: Path) -> None:
        """CSV with LATITUDE/LONGITUDE columns should use them directly."""
        from aquacontam.data.ejscreen import load_ejscreen

        csv_path = tmp_path / "EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.csv"
        _write_mock_ejscreen_csv(csv_path, include_coords=True)
        gdf = load_ejscreen(tmp_path)
        assert len(gdf) > 0
        assert "block_group_id" in gdf.columns

    def test_gdb_centroid_fallback(self, tmp_path: Path) -> None:
        """GDB polygon centroids used when CSV lacks coords and no gazetteer."""
        pytest.importorskip("fiona")  # GDB read/write needs the fiona OGR backend

        import geopandas as gpd
        from shapely.geometry import box

        from aquacontam.data.ejscreen import _try_load_gdb_centroids

        # Create a mock GDB with block group polygons
        gdb_dir = tmp_path / "EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.gdb"
        gdf = gpd.GeoDataFrame(
            {"ID": ["360010001", "360010002"]},
            geometry=[box(-74.1, 39.9, -73.9, 40.1), box(-73.1, 40.9, -72.9, 41.1)],
            crs="EPSG:4326",
        )
        gdf.to_file(gdb_dir, driver="OpenFileGDB")

        result = _try_load_gdb_centroids(tmp_path)
        assert result is not None
        assert len(result) == 2
        assert "latitude" in result.columns
        assert "longitude" in result.columns
        # Centroids should be roughly at the center of the boxes
        assert abs(result.loc["360010001", "latitude"] - 40.0) < 0.2
        assert abs(result.loc["360010001", "longitude"] - (-74.0)) < 0.2

    def test_gdb_centroids_none_when_missing(self, tmp_path: Path) -> None:
        """Returns None when no GDB file exists."""
        from aquacontam.data.ejscreen import _try_load_gdb_centroids

        result = _try_load_gdb_centroids(tmp_path)
        assert result is None

    def test_gdb_centroids_bypass_fail_fast(self, tmp_path: Path) -> None:
        """GDB presence should bypass the fail-fast check for missing coords."""
        from aquacontam.data.ejscreen import find_ejscreen_gdb

        csv_path = tmp_path / "EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.csv"
        _write_mock_ejscreen_csv(csv_path, include_coords=False)
        # Create a mock GDB zip so find_ejscreen_gdb returns non-None
        gdb_zip = tmp_path / "EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.gdb.zip"
        gdb_zip.touch()
        # Verify GDB is found
        assert find_ejscreen_gdb(tmp_path) is not None


class TestFindEJScreenGDB:
    """Tests for find_ejscreen_gdb() helper."""

    def test_finds_gdb_zip(self, tmp_path: Path) -> None:
        from aquacontam.data.ejscreen import find_ejscreen_gdb

        gdb_zip = tmp_path / "EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.gdb.zip"
        gdb_zip.touch()
        result = find_ejscreen_gdb(tmp_path)
        assert result is not None
        assert result == gdb_zip

    def test_finds_gdb_directory(self, tmp_path: Path) -> None:
        from aquacontam.data.ejscreen import find_ejscreen_gdb

        gdb_dir = tmp_path / "EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.gdb"
        gdb_dir.mkdir()
        result = find_ejscreen_gdb(tmp_path)
        assert result is not None
        assert result == gdb_dir

    def test_finds_nested_gdb(self, tmp_path: Path) -> None:
        from aquacontam.data.ejscreen import find_ejscreen_gdb

        nested = tmp_path / "subdir" / "EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.gdb.zip"
        nested.parent.mkdir(parents=True)
        nested.touch()
        result = find_ejscreen_gdb(tmp_path)
        assert result is not None
        assert result == nested

    def test_returns_none_when_no_gdb(self, tmp_path: Path) -> None:
        from aquacontam.data.ejscreen import find_ejscreen_gdb

        result = find_ejscreen_gdb(tmp_path)
        assert result is None
