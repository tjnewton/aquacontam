"""Tests for California GeoTracker (GAMA) data source."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aquacontam._constants import PPT_TO_UGL
from aquacontam.data.ca_geotracker import CaGeoTrackerSource


@pytest.fixture()
def ca_source(tmp_data_dirs: dict[str, Path]) -> CaGeoTrackerSource:
    return CaGeoTrackerSource(
        raw_dir=tmp_data_dirs["raw"],
        interim_dir=tmp_data_dirs["interim"],
        processed_dir=tmp_data_dirs["processed"],
    )


@pytest.fixture()
def sample_ca_csv(tmp_data_dirs: dict[str, Path]) -> Path:
    """Create a sample CA GAMA PFAS CSV in the real ``gm_*`` schema.

    gm_well_id encodes the PWSID as a 9-char prefix (``CA#######_...``); rows
    whose well id is not a CA PWSID (monitoring wells like ``AMD-2/MP4`` or
    ``T...``) must be dropped.
    """
    data = pd.DataFrame(
        {
            "gm_well_id": [
                "CA0101001_001_001",  # PWS, PFOS detected
                "CA0101001_001_001",  # PWS, PFOA non-detect ("<")
                "CA0202002_003_001",  # PWS, PFHxS detected
                "AMD-2/MP4",  # non-PWS monitoring well -> dropped
                "T10000013_MW3",  # non-PWS well -> dropped
            ],
            # gm_chemical_vvl short codes; PFHXSA must map to canonical PFHxS
            "gm_chemical_vvl": ["PFOS", "PFOA", "PFHXSA", "PFBSA", "PFNA"],
            "gm_result": ["12500", "4.0", "8.3", "5.0", "10.0"],
            "gm_result_modifier": ["=", "<", "=", "=", "="],
            "gm_chemical_units": ["NG/L", "NG/L", "NG/L", "NG/L", "NG/L"],
            "gm_reporting_limit": ["2.0", "4.0", "2.0", "1.0", "2.0"],
            "gm_samp_collection_date": [
                "2023-01-15",
                "2023-02-20",
                "2023-03-10",
                "2023-04-01",
                "2023-05-01",
            ],
            "gm_latitude": ["34.05", "34.05", "36.77", "35.0", "37.0"],
            "gm_longitude": ["-118.24", "-118.24", "-119.42", "-120.0", "-121.0"],
        }
    )
    path = tmp_data_dirs["raw"] / "ca_geotracker_pfas.csv"
    data.to_csv(path, index=False)
    return path


class TestCaGeoTrackerSource:
    """Tests for CaGeoTrackerSource (vectorized PWS-scoped parser)."""

    def test_name_property(self, ca_source: CaGeoTrackerSource) -> None:
        assert ca_source.name == "ca_geotracker"

    def test_config_loaded(self, ca_source: CaGeoTrackerSource) -> None:
        assert "url" in ca_source._config
        assert "expected_files" in ca_source._config

    def test_parse_column_mapping(
        self, ca_source: CaGeoTrackerSource, sample_ca_csv: Path
    ) -> None:
        df = ca_source.parse()
        for c in ("pwsid", "analyte", "concentration", "censored", "detection_limit"):
            assert c in df.columns
        # 3 PWS rows kept; 2 non-PWS wells dropped
        assert len(df) == 3

    def test_pwsid_extracted_from_well_id(
        self, ca_source: CaGeoTrackerSource, sample_ca_csv: Path
    ) -> None:
        df = ca_source.parse()
        pfos_row = df[df["analyte"] == "PFOS"].iloc[0]
        assert pfos_row["pwsid"] == "CA0101001"
        assert set(df["pwsid"]) == {"CA0101001", "CA0202002"}

    def test_non_pws_wells_dropped(
        self, ca_source: CaGeoTrackerSource, sample_ca_csv: Path
    ) -> None:
        """Wells whose id is not a CA PWSID prefix must be excluded."""
        df = ca_source.parse()
        assert not df["pwsid"].str.contains("AMD").any()
        assert "PFBS" not in set(df["analyte"])  # AMD-2/MP4 row dropped
        assert "PFNA" not in set(df["analyte"])  # T... row dropped

    def test_unit_conversion_ngl(self, ca_source: CaGeoTrackerSource, sample_ca_csv: Path) -> None:
        """NG/L values are converted to ug/L."""
        df = ca_source.parse()
        pfhxs_row = df[df["analyte"] == "PFHxS"].iloc[0]
        assert abs(pfhxs_row["concentration"] - 8.3 * PPT_TO_UGL) < 1e-12

    def test_detected_value_converted(
        self, ca_source: CaGeoTrackerSource, sample_ca_csv: Path
    ) -> None:
        df = ca_source.parse()
        pfos_row = df[df["analyte"] == "PFOS"].iloc[0]
        assert abs(pfos_row["concentration"] - 12500 * PPT_TO_UGL) < 1e-9  # 12.5 ug/L

    def test_analyte_code_mapping(
        self, ca_source: CaGeoTrackerSource, sample_ca_csv: Path
    ) -> None:
        """GAMA vvl codes map to canonical names (PFHXSA -> PFHxS)."""
        df = ca_source.parse()
        assert "PFHxS" in set(df["analyte"])
        assert "PFHXSA" not in set(df["analyte"])

    def test_censoring_modifier(self, ca_source: CaGeoTrackerSource, sample_ca_csv: Path) -> None:
        """gm_result_modifier '<' marks a non-detect (concentration 0)."""
        df = ca_source.parse()
        pfoa_row = df[df["analyte"] == "PFOA"].iloc[0]
        assert bool(pfoa_row["censored"]) is True
        assert pfoa_row["concentration"] == 0.0

    def test_coordinates_present(self, ca_source: CaGeoTrackerSource, sample_ca_csv: Path) -> None:
        df = ca_source.parse()
        assert df["latitude"].notna().all()
        assert df["longitude"].notna().all()

    def test_all_unit_is_ugl(self, ca_source: CaGeoTrackerSource, sample_ca_csv: Path) -> None:
        df = ca_source.parse()
        assert (df["unit"] == "ug/L").all()

    def test_to_parquet(self, ca_source: CaGeoTrackerSource, sample_ca_csv: Path) -> None:
        df = ca_source.parse()
        path = ca_source.to_parquet(df)
        assert path.exists()
        reloaded = pd.read_parquet(path)
        assert len(reloaded) == len(df)

    def test_deterministic(self, ca_source: CaGeoTrackerSource, sample_ca_csv: Path) -> None:
        """Parse is deterministic regardless of row order."""
        df1 = ca_source.parse()
        raw = pd.read_csv(sample_ca_csv, dtype=str)
        raw.iloc[::-1].to_csv(sample_ca_csv, index=False)
        df2 = ca_source.parse()
        assert df1.equals(df2)

    def test_dedup_keeps_max_concentration(
        self, ca_source: CaGeoTrackerSource, tmp_data_dirs: dict[str, Path]
    ) -> None:
        """Duplicate (pwsid, analyte, date) collapses to the max concentration."""
        data = pd.DataFrame(
            {
                "gm_well_id": ["CA0303003_001_001", "CA0303003_002_001"],
                "gm_chemical_vvl": ["PFOS", "PFOS"],
                "gm_result": ["5000", "10000"],  # 5.0 vs 10.0 ug/L, same system+date
                "gm_result_modifier": ["=", "="],
                "gm_chemical_units": ["NG/L", "NG/L"],
                "gm_reporting_limit": ["2.0", "2.0"],
                "gm_samp_collection_date": ["2023-06-01", "2023-06-01"],
                "gm_latitude": ["38.0", "38.0"],
                "gm_longitude": ["-121.0", "-121.0"],
            }
        )
        path = tmp_data_dirs["raw"] / "ca_geotracker_pfas.csv"
        data.to_csv(path, index=False)
        df = ca_source.parse()
        assert len(df) == 1
        assert abs(df.iloc[0]["concentration"] - 10.0) < 1e-9
