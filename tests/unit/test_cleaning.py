"""Tests for data cleaning pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from aquacontam._constants import MGL_TO_UGL, PPT_TO_UGL
from aquacontam.preprocessing.cleaning import (
    deduplicate_samples,
    harmonize_units,
    merge_datasets,
)


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Sample water quality DataFrame with duplicates."""
    return pd.DataFrame(
        {
            "pwsid": ["CA0101001", "CA0101001", "CA0101001", "TX0200002"],
            "analyte": ["PFOS", "PFOS", "PFOA", "PFOS"],
            "concentration": [12.5, 8.0, 5.0, 3.0],
            "unit": ["ug/L", "ug/L", "ug/L", "ug/L"],
            "censored": [False, False, False, True],
            "detection_limit": [2.0, 2.0, 1.0, 2.0],
            "sample_date": pd.to_datetime(
                ["2023-01-15", "2023-01-15", "2023-01-15", "2023-02-20"]
            ),
            "latitude": [34.05, 34.05, 34.05, 30.27],
            "longitude": [-118.24, -118.24, -118.24, -97.74],
        }
    )


class TestDeduplicateSamples:
    """Tests for deduplicate_samples."""

    def test_max_strategy(self, sample_df: pd.DataFrame) -> None:
        result = deduplicate_samples(sample_df, strategy="max")
        # CA PFOS has 2 dups (12.5 and 8.0), keep max=12.5
        ca_pfos = result[(result["pwsid"] == "CA0101001") & (result["analyte"] == "PFOS")]
        assert len(ca_pfos) == 1
        assert ca_pfos.iloc[0]["concentration"] == 12.5

    def test_mean_strategy(self, sample_df: pd.DataFrame) -> None:
        result = deduplicate_samples(sample_df, strategy="mean")
        ca_pfos = result[(result["pwsid"] == "CA0101001") & (result["analyte"] == "PFOS")]
        assert len(ca_pfos) == 1
        assert abs(ca_pfos.iloc[0]["concentration"] - 10.25) < 0.01  # (12.5+8.0)/2

    def test_first_strategy(self, sample_df: pd.DataFrame) -> None:
        result = deduplicate_samples(sample_df, strategy="first")
        ca_pfos = result[(result["pwsid"] == "CA0101001") & (result["analyte"] == "PFOS")]
        assert len(ca_pfos) == 1

    def test_no_duplicates_unchanged(self) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["CA0101001"],
                "analyte": ["PFOS"],
                "concentration": [12.5],
                "sample_date": pd.to_datetime(["2023-01-15"]),
            }
        )
        result = deduplicate_samples(df)
        assert len(result) == 1

    def test_invalid_strategy_raises(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="strategy"):
            deduplicate_samples(sample_df, strategy="invalid")

    def test_missing_keys_raises(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Missing key"):
            deduplicate_samples(sample_df, keys=("nonexistent",))

    def test_missing_concentration_raises(self) -> None:
        df = pd.DataFrame({"pwsid": ["A"], "analyte": ["B"], "sample_date": ["2023-01-01"]})
        with pytest.raises(ValueError, match="concentration"):
            deduplicate_samples(df)

    def test_non_dup_rows_preserved(self, sample_df: pd.DataFrame) -> None:
        result = deduplicate_samples(sample_df, strategy="max")
        tx_rows = result[result["pwsid"] == "TX0200002"]
        assert len(tx_rows) == 1


class TestHarmonizeUnits:
    """Tests for harmonize_units."""

    def test_ppt_to_ugl(self) -> None:
        df = pd.DataFrame(
            {
                "concentration": [100.0],
                "unit": ["ng/L"],
                "detection_limit": [10.0],
            }
        )
        result = harmonize_units(df)
        assert abs(result.iloc[0]["concentration"] - 100.0 * PPT_TO_UGL) < 1e-9
        assert result.iloc[0]["unit"] == "ug/L"
        assert result.iloc[0]["original_unit"] == "ng/L"

    def test_mgl_to_ugl(self) -> None:
        df = pd.DataFrame(
            {
                "concentration": [0.015],
                "unit": ["mg/L"],
                "detection_limit": [0.001],
            }
        )
        result = harmonize_units(df)
        assert abs(result.iloc[0]["concentration"] - 0.015 * MGL_TO_UGL) < 0.01

    def test_ugl_unchanged(self) -> None:
        df = pd.DataFrame(
            {
                "concentration": [15.0],
                "unit": ["ug/L"],
                "detection_limit": [1.0],
            }
        )
        result = harmonize_units(df)
        assert result.iloc[0]["concentration"] == 15.0

    def test_ppb_treated_as_ugl(self) -> None:
        df = pd.DataFrame(
            {
                "concentration": [15.0],
                "unit": ["PPB"],
            }
        )
        result = harmonize_units(df)
        assert result.iloc[0]["concentration"] == 15.0

    def test_original_unit_preserved(self) -> None:
        df = pd.DataFrame(
            {
                "concentration": [1.0],
                "unit": ["PPT"],
            }
        )
        result = harmonize_units(df)
        assert result.iloc[0]["original_unit"] == "PPT"
        assert result.iloc[0]["unit"] == "ug/L"

    def test_unknown_unit_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        df = pd.DataFrame(
            {
                "concentration": [1.0],
                "unit": ["unknown_unit"],
            }
        )
        with caplog.at_level("WARNING"):
            harmonize_units(df)
        assert "unrecognized unit" in caplog.text

    def test_missing_unit_column_raises(self) -> None:
        df = pd.DataFrame({"concentration": [1.0]})
        with pytest.raises(ValueError, match="unit"):
            harmonize_units(df)

    def test_missing_concentration_column_raises(self) -> None:
        df = pd.DataFrame({"unit": ["ug/L"]})
        with pytest.raises(ValueError, match="concentration"):
            harmonize_units(df)

    def test_unknown_unit_preserves_original(self) -> None:
        """Unknown units should keep their original unit string, not be overwritten."""
        df = pd.DataFrame(
            {
                "concentration": [1.0],
                "unit": ["foo_unit"],
            }
        )
        result = harmonize_units(df)
        assert result.iloc[0]["unit"] == "foo_unit"
        assert result.iloc[0]["original_unit"] == "foo_unit"

    def test_detection_limit_converted(self) -> None:
        df = pd.DataFrame(
            {
                "concentration": [10.0],
                "unit": ["ng/L"],
                "detection_limit": [5.0],
            }
        )
        result = harmonize_units(df)
        assert abs(result.iloc[0]["detection_limit"] - 5.0 * PPT_TO_UGL) < 1e-9


class TestMergeDatasets:
    """Tests for merge_datasets."""

    def test_basic_merge(self) -> None:
        df1 = pd.DataFrame(
            {
                "pwsid": ["CA0101001"],
                "analyte": ["PFOS"],
                "source": ["ucmr5"],
                "latitude": [34.05],
                "longitude": [-118.24],
            }
        )
        df2 = pd.DataFrame(
            {
                "pwsid": ["TX0200002"],
                "analyte": ["PFOA"],
                "source": ["ucmr3"],
                "latitude": [30.27],
                "longitude": [-97.74],
            }
        )
        result = merge_datasets(df1, df2)
        assert len(result) == 2
        assert set(result["source"]) == {"ucmr5", "ucmr3"}

    def test_auto_source_column(self) -> None:
        """Missing source column should be auto-generated."""
        df1 = pd.DataFrame({"pwsid": ["A"], "latitude": [1.0], "longitude": [2.0]})
        df2 = pd.DataFrame({"pwsid": ["B"], "latitude": [3.0], "longitude": [4.0]})
        result = merge_datasets(df1, df2)
        assert "source" in result.columns
        assert result.iloc[0]["source"] == "dataset_0"
        assert result.iloc[1]["source"] == "dataset_1"

    def test_has_native_coordinates_flag(self) -> None:
        df1 = pd.DataFrame({"latitude": [34.0], "longitude": [-118.0]})
        df2 = pd.DataFrame({"latitude": [float("nan")], "longitude": [float("nan")]})
        result = merge_datasets(df1, df2)
        assert "has_native_coordinates" in result.columns
        assert bool(result.iloc[0]["has_native_coordinates"]) is True
        assert bool(result.iloc[1]["has_native_coordinates"]) is False

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            merge_datasets()

    def test_single_dataset(self) -> None:
        df = pd.DataFrame(
            {"pwsid": ["A"], "source": ["test"], "latitude": [1.0], "longitude": [2.0]}
        )
        result = merge_datasets(df)
        assert len(result) == 1
