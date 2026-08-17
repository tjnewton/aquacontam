"""Tests for Michigan MPART data source."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aquacontam._constants import PPT_TO_UGL
from aquacontam.data._utils import find_col
from aquacontam.data.mi_mpart import MiMpartSource


@pytest.fixture()
def mi_source(tmp_data_dirs: dict[str, Path]) -> MiMpartSource:
    return MiMpartSource(
        raw_dir=tmp_data_dirs["raw"],
        interim_dir=tmp_data_dirs["interim"],
        processed_dir=tmp_data_dirs["processed"],
    )


@pytest.fixture()
def sample_mi_csv(tmp_data_dirs: dict[str, Path]) -> Path:
    """Create a sample MI MPART CSV file (legacy long format)."""
    data = pd.DataFrame(
        {
            "PWSID": ["MI0000001", "MI0000001", "MI0000002"],
            "Contaminant": ["PFOS", "PFOA", "HFPO-DA"],
            "Result": [15.0, 0.0, 25.0],
            "Qualifier": ["", "ND", ""],
            "SampleDate": ["2023-01-15", "2023-02-20", "2023-03-10"],
            "_latitude": [42.33, 42.33, 43.01],
            "_longitude": [-83.05, -83.05, -84.22],
        }
    )
    path = tmp_data_dirs["raw"] / "mi_mpart_results.csv"
    data.to_csv(path, index=False)
    return path


@pytest.fixture()
def sample_mi_wide_csv(tmp_data_dirs: dict[str, Path]) -> Path:
    """Create a sample MI MPART CSV in wide format (new ArcGIS Layer 3)."""
    data = pd.DataFrame(
        {
            "WSSN": [1545, 1545, 3018],
            "SystemName": ["MAIN STREET APTS", "MAIN STREET APTS", "HARING TWP"],
            "SampleDate": [
                "1765929600000",
                "1765929600000",
                "1765411200000",
            ],
            "PFOS": ["2", "ND", "5.7"],
            "PFOA": ["ND", "3", "ND"],
            "HFPODA": ["ND", "ND", "ND"],
            "PFHxS": ["ND", "ND", "ND"],
            "PFNA": ["ND", "ND", "ND"],
            "PFHxA": ["ND", "ND", "ND"],
            "PFBS": ["ND", "ND", "ND"],
        }
    )
    path = tmp_data_dirs["raw"] / "mi_mpart_results.csv"
    data.to_csv(path, index=False)
    return path


class TestMiMpartSource:
    """Tests for MiMpartSource (legacy long format)."""

    def test_name_property(self, mi_source: MiMpartSource) -> None:
        assert mi_source.name == "mi_mpart"

    def test_config_loaded(self, mi_source: MiMpartSource) -> None:
        assert "arcgis_base_url" in mi_source._config
        assert "expected_files" in mi_source._config

    def test_parse_column_mapping(self, mi_source: MiMpartSource, sample_mi_csv: Path) -> None:
        df = mi_source.parse()
        assert "pwsid" in df.columns
        assert "analyte" in df.columns
        assert "concentration" in df.columns
        assert "unit" in df.columns
        assert len(df) == 3

    def test_ppt_to_ugl_conversion(self, mi_source: MiMpartSource, sample_mi_csv: Path) -> None:
        df = mi_source.parse()
        # First row: 15.0 PPT → 0.015 ug/L
        pfos_row = df[df["analyte"] == "PFOS"].iloc[0]
        assert abs(pfos_row["concentration"] - 15.0 * PPT_TO_UGL) < 1e-9

    def test_non_detect_censoring(self, mi_source: MiMpartSource, sample_mi_csv: Path) -> None:
        df = mi_source.parse()
        nd_row = df[(df["analyte"] == "PFOA") & (df["pwsid"] == "MI0000001")].iloc[0]
        assert bool(nd_row["censored"]) is True
        assert nd_row["concentration"] == 0.0

    def test_detected_not_censored(self, mi_source: MiMpartSource, sample_mi_csv: Path) -> None:
        df = mi_source.parse()
        det_row = df[df["analyte"] == "PFOS"].iloc[0]
        assert bool(det_row["censored"]) is False

    def test_coordinates_preserved(self, mi_source: MiMpartSource, sample_mi_csv: Path) -> None:
        df = mi_source.parse()
        assert df["latitude"].notna().all()
        assert df["longitude"].notna().all()

    def test_validate_warns_unexpected_analyte(
        self, mi_source: MiMpartSource, caplog: pytest.LogCaptureFixture
    ) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["MI0000001"],
                "analyte": ["UNKNOWN"],
                "concentration": [1.0],
                "unit": ["ug/L"],
                "censored": [False],
                "detection_limit": [0.5],
                "sample_date": pd.to_datetime(["2023-01-15"]),
                "latitude": [42.33],
                "longitude": [-83.05],
            }
        )
        with caplog.at_level("WARNING"):
            mi_source.validate(df)
        assert "Unexpected MI MPART analytes" in caplog.text

    def test_to_parquet(self, mi_source: MiMpartSource, sample_mi_csv: Path) -> None:
        df = mi_source.parse()
        # Skip validation (coordinates may not be in CONUS for test data)
        path = mi_source.to_parquet(df)
        assert path.exists()
        reloaded = pd.read_parquet(path)
        assert len(reloaded) == len(df)

    def test_pwsid_zero_padded(
        self, mi_source: MiMpartSource, tmp_data_dirs: dict[str, Path]
    ) -> None:
        """Short PWSIDs should be zero-padded, not space-padded."""
        data = pd.DataFrame(
            {
                "PWSID": ["123", "MI45"],
                "Contaminant": ["PFOS", "PFOA"],
                "Result": [15.0, 10.0],
                "Qualifier": ["", ""],
                "SampleDate": ["2023-01-15", "2023-02-20"],
                "_latitude": [42.33, 43.01],
                "_longitude": [-83.05, -84.22],
            }
        )
        path = tmp_data_dirs["raw"] / "mi_mpart_results.csv"
        data.to_csv(path, index=False)

        df = mi_source.parse()
        pwsids = df["pwsid"].unique().tolist()
        assert "MI0000123" in pwsids
        assert "MI0000045" in pwsids
        for p in pwsids:
            assert " " not in p

    def test_download_skip_existing(self, mi_source: MiMpartSource, sample_mi_csv: Path) -> None:
        """Download should skip if file already exists."""
        result = mi_source.download(force=False)
        assert len(result) == 1
        assert result[0].exists()


class TestMiMpartWideFormat:
    """Tests for MiMpartSource with new ArcGIS Layer 3 wide format."""

    def test_parse_wide_format(self, mi_source: MiMpartSource, sample_mi_wide_csv: Path) -> None:
        """Wide-format CSV should be melted to long format."""
        df = mi_source.parse()
        assert "pwsid" in df.columns
        assert "analyte" in df.columns
        assert "concentration" in df.columns
        assert "unit" in df.columns
        # 3 rows x 7 analytes, but ND rows also count
        assert len(df) > 0

    def test_wide_format_ppt_conversion(
        self, mi_source: MiMpartSource, sample_mi_wide_csv: Path
    ) -> None:
        """Detected values should be converted from PPT to ug/L."""
        df = mi_source.parse()
        pfos_det = df[(df["analyte"] == "PFOS") & (~df["censored"])]
        assert len(pfos_det) > 0
        # WSSN 1545: PFOS = 2 PPT → 0.002 ug/L
        row = pfos_det.iloc[0]
        assert abs(row["concentration"] - 2.0 * PPT_TO_UGL) < 1e-9

    def test_wide_format_nd_censored(
        self, mi_source: MiMpartSource, sample_mi_wide_csv: Path
    ) -> None:
        """ND values should be censored with zero concentration."""
        df = mi_source.parse()
        nd_rows = df[df["censored"]]
        assert len(nd_rows) > 0
        assert (nd_rows["concentration"] == 0.0).all()

    def test_wide_format_wssn_to_pwsid(
        self, mi_source: MiMpartSource, sample_mi_wide_csv: Path
    ) -> None:
        """WSSN integer should be converted to MI-prefixed PWSID."""
        df = mi_source.parse()
        pwsids = df["pwsid"].unique()
        for p in pwsids:
            assert p.startswith("MI")
            assert len(p) == 9
        # WSSN 1545 → MI0001545
        assert "MI0001545" in pwsids

    def test_wide_format_epoch_date(
        self, mi_source: MiMpartSource, sample_mi_wide_csv: Path
    ) -> None:
        """Epoch millisecond dates should be parsed correctly."""
        df = mi_source.parse()
        assert df["sample_date"].notna().all()

    def test_wide_format_no_coordinates(
        self, mi_source: MiMpartSource, sample_mi_wide_csv: Path
    ) -> None:
        """Layer 3 table has no coordinates; lat/lon should be NaN."""
        df = mi_source.parse()
        assert df["latitude"].isna().all()
        assert df["longitude"].isna().all()

    def test_validate_skips_coord_check(
        self, mi_source: MiMpartSource, sample_mi_wide_csv: Path
    ) -> None:
        """Validation should skip coord check when no coordinates present."""
        df = mi_source.parse()
        # Should not raise even though lat/lon are NaN
        result = mi_source.validate(df)
        assert len(result) > 0


class TestFindColumn:
    """Tests for the _find_column helper."""

    def test_finds_first_match(self) -> None:
        df = pd.DataFrame({"A": [1], "B": [2]})
        assert find_col(df, ["C", "A", "B"]) == "A"

    def test_raises_when_required_missing(self) -> None:
        df = pd.DataFrame({"A": [1]})
        with pytest.raises(ValueError, match="None of"):
            find_col(df, ["X", "Y"])

    def test_returns_none_when_optional(self) -> None:
        df = pd.DataFrame({"A": [1]})
        assert find_col(df, ["X"], required=False) is None
