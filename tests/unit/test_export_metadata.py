"""Tests for dataset metadata generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from aquacontam.export.metadata import (
    _analyte_info,
    generate_dataset_card,
    generate_feature_metadata,
    write_metadata_files,
)


@pytest.fixture()
def feature_dfs() -> dict[str, pd.DataFrame]:
    """Sample feature DataFrames for metadata tests."""
    return {
        "proximity": pd.DataFrame(
            {"dist_industrial": [100.0, 200.0], "count_wwtp_5km": [1, 3]},
            index=pd.Index(["A", "B"], name="pwsid"),
        ),
        "land_use": pd.DataFrame(
            {"pct_developed": [0.3, 0.7], "pct_agriculture": [0.5, 0.1]},
            index=pd.Index(["A", "B"], name="pwsid"),
        ),
    }


class TestGenerateFeatureMetadata:
    """Tests for generate_feature_metadata."""

    def test_returns_list_of_dicts(self, feature_dfs: dict[str, pd.DataFrame]) -> None:
        result = generate_feature_metadata(feature_dfs)
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_correct_count(self, feature_dfs: dict[str, pd.DataFrame]) -> None:
        result = generate_feature_metadata(feature_dfs)
        assert len(result) == 4  # 2 + 2 columns

    def test_includes_source(self, feature_dfs: dict[str, pd.DataFrame]) -> None:
        result = generate_feature_metadata(feature_dfs)
        sources = {r["source"] for r in result}
        assert sources == {"proximity", "land_use"}

    def test_includes_dtype(self, feature_dfs: dict[str, pd.DataFrame]) -> None:
        result = generate_feature_metadata(feature_dfs)
        assert all("dtype" in r for r in result)

    def test_custom_descriptions(self, feature_dfs: dict[str, pd.DataFrame]) -> None:
        desc = {"dist_industrial": "Distance to nearest industrial facility"}
        result = generate_feature_metadata(feature_dfs, descriptions=desc)
        industrial = next(r for r in result if r["name"] == "dist_industrial")
        assert industrial["description"] == desc["dist_industrial"]

    def test_n_non_null_counted(self, feature_dfs: dict[str, pd.DataFrame]) -> None:
        result = generate_feature_metadata(feature_dfs)
        assert all(r["n_non_null"] == 2 for r in result)


class TestAnalyteInfo:
    """Tests for _analyte_info."""

    def test_returns_list(self) -> None:
        result = _analyte_info()
        assert isinstance(result, list)
        assert len(result) > 30  # 30 UCMR5 + extras

    def test_no_duplicates(self) -> None:
        result = _analyte_info()
        names = [r["name"] for r in result]
        assert len(names) == len(set(names))

    def test_includes_heavy_metals(self) -> None:
        result = _analyte_info()
        names = {r["name"] for r in result}
        assert "lead" in names
        assert "copper" in names


class TestGenerateDatasetCard:
    """Tests for generate_dataset_card."""

    def test_has_required_keys(self) -> None:
        card = generate_dataset_card()
        assert "name" in card
        assert "version" in card
        assert "tasks" in card
        assert "license" in card

    def test_tasks_present(self) -> None:
        card = generate_dataset_card()
        assert "T1" in card["tasks"]
        assert "T2" in card["tasks"]
        assert "T4" in card["tasks"]

    def test_version_override(self) -> None:
        card = generate_dataset_card(version="2.0")
        assert card["version"] == "2.0"

    def test_extra_metadata(self) -> None:
        card = generate_dataset_card(extra={"custom": "value"})
        assert card["custom"] == "value"

    def test_license_and_citation_reflect_the_dual_split(self) -> None:
        """The card ships inside the Zenodo deposit; its license must match the split."""
        card = generate_dataset_card()
        assert "CC BY 4.0" in card["license"]
        assert "Apache License 2.0" in card["license"]
        assert "MIT" not in card["license"]
        assert card["citation"].startswith("Newton, T. J. (2026).")
        assert card["version"] == "4.0.0"


class TestWriteMetadataFiles:
    """Tests for write_metadata_files."""

    def test_writes_analyte_info(self, tmp_path: Path) -> None:
        write_metadata_files(tmp_path)
        assert (tmp_path / "analyte_info.json").exists()
        with open(tmp_path / "analyte_info.json") as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_writes_dataset_card(self, tmp_path: Path) -> None:
        write_metadata_files(tmp_path)
        assert (tmp_path / "dataset_card.json").exists()

    def test_writes_feature_columns_when_provided(
        self, tmp_path: Path, feature_dfs: dict[str, pd.DataFrame]
    ) -> None:
        paths = write_metadata_files(tmp_path, feature_dfs=feature_dfs)
        assert (tmp_path / "feature_columns.json").exists()
        assert len(paths) == 3

    def test_no_feature_columns_without_features(self, tmp_path: Path) -> None:
        paths = write_metadata_files(tmp_path)
        assert not (tmp_path / "feature_columns.json").exists()
        assert len(paths) == 2
