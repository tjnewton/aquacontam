"""Tests for SDWIS system characteristic features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aquacontam.features.system_characteristics import extract_system_characteristics


@pytest.fixture()
def sdwis_systems_csv(tmp_path: Path) -> Path:
    """Create a mock SDWA_PUB_WATER_SYSTEMS.csv."""
    data = {
        "PWSID": ["CA0101001", "CA0101002", "NY0201001", "TX0601001"],
        "PWS_NAME": ["City Water A", "City Water B", "Town Water C", "Rural Water D"],
        "GW_SW_CODE": ["GW", "SW", "GW", "GU"],
        "POPULATION_SERVED_COUNT": [5000, 150000, 3300, 25],
        "PWS_TYPE_CODE": ["CWS", "CWS", "NTNCWS", "TNCWS"],
        "OWNER_TYPE_CODE": ["L", "L", "P", "F"],
        "GEO_LATITUDE": [34.0, 34.1, 40.7, 30.3],
        "GEO_LONGITUDE": [-118.0, -118.1, -74.0, -97.7],
    }
    df = pd.DataFrame(data)

    # Write to expected location
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    df.to_csv(raw_dir / "SDWA_PUB_WATER_SYSTEMS.csv", index=False)

    # Also create the LCR samples file (empty, just needed for config)
    pd.DataFrame(columns=["PWSID"]).to_csv(raw_dir / "SDWA_LCR_SAMPLES.csv", index=False)

    return raw_dir


class TestExtractSystemCharacteristics:
    """Tests for extract_system_characteristics."""

    def test_extracts_all_features(self, sdwis_systems_csv: Path) -> None:
        result = extract_system_characteristics(sdwis_systems_csv)
        assert result.index.name == "pwsid"
        assert len(result) == 4
        assert "source_water_type" in result.columns
        assert "population_served" in result.columns
        assert "log_population_served" in result.columns
        assert "system_type" in result.columns
        assert "owner_type" in result.columns

    def test_source_water_type_values(self, sdwis_systems_csv: Path) -> None:
        result = extract_system_characteristics(sdwis_systems_csv)
        assert result.loc["CA0101001", "source_water_type"] == "GW"
        assert result.loc["CA0101002", "source_water_type"] == "SW"
        assert result.loc["TX0601001", "source_water_type"] == "GU"

    def test_population_served(self, sdwis_systems_csv: Path) -> None:
        result = extract_system_characteristics(sdwis_systems_csv)
        assert result.loc["CA0101001", "population_served"] == 5000
        assert result.loc["CA0101002", "population_served"] == 150000

    def test_log_population(self, sdwis_systems_csv: Path) -> None:
        result = extract_system_characteristics(sdwis_systems_csv)
        log_pop = result.loc["CA0101001", "log_population_served"]
        assert log_pop == pytest.approx(np.log1p(5000))

    def test_system_type(self, sdwis_systems_csv: Path) -> None:
        result = extract_system_characteristics(sdwis_systems_csv)
        assert result.loc["CA0101001", "system_type"] == "CWS"
        assert result.loc["NY0201001", "system_type"] == "NTNCWS"

    def test_owner_type(self, sdwis_systems_csv: Path) -> None:
        result = extract_system_characteristics(sdwis_systems_csv)
        assert result.loc["CA0101001", "owner_type"] == "L"
        assert result.loc["NY0201001", "owner_type"] == "P"
        assert result.loc["TX0601001", "owner_type"] == "F"

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = extract_system_characteristics(tmp_path)
        assert result.empty or len(result) == 0

    def test_pwsid_index(self, sdwis_systems_csv: Path) -> None:
        result = extract_system_characteristics(sdwis_systems_csv)
        assert "CA0101001" in result.index
        assert "TX0601001" in result.index
