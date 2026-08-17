"""Tests for data schema definitions and validation."""

from __future__ import annotations

import pandas as pd

from aquacontam._constants import REQUIRED_COLUMNS

# String columns may report as "object" (pandas <3) or "str" (pandas >=3).
_STRING_DTYPES = {"object", "str", "string"}

EXPECTED_DTYPES: dict[str, str | set[str]] = {
    "pwsid": _STRING_DTYPES,
    "analyte": _STRING_DTYPES,
    "concentration": "float64",
    "unit": _STRING_DTYPES,
    "censored": "bool",
    "detection_limit": "float64",
}


class TestWaterQualitySchema:
    """Verify that the standard water quality schema is well-defined."""

    def test_required_columns_are_defined(self) -> None:
        assert len(REQUIRED_COLUMNS) == 9

    def test_sample_df_has_required_columns(self, sample_water_quality_df: pd.DataFrame) -> None:
        missing = set(REQUIRED_COLUMNS) - set(sample_water_quality_df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_sample_df_dtypes(self, sample_water_quality_df: pd.DataFrame) -> None:
        for col, expected in EXPECTED_DTYPES.items():
            actual = str(sample_water_quality_df[col].dtype)
            if isinstance(expected, set):
                assert actual in expected, f"{col}: expected one of {expected}, got {actual}"
            else:
                assert actual == expected, f"{col}: expected {expected}, got {actual}"

    def test_pwsid_is_string_not_int(self, sample_water_quality_df: pd.DataFrame) -> None:
        """PWSID must remain a string — leading zeros are meaningful."""
        assert str(sample_water_quality_df["pwsid"].dtype) in _STRING_DTYPES
        for val in sample_water_quality_df["pwsid"]:
            assert isinstance(val, str)
            assert len(val) == 9, f"PWSID should be 9 chars, got {len(val)}: {val}"

    def test_censored_column_is_bool(self, sample_water_quality_df: pd.DataFrame) -> None:
        assert sample_water_quality_df["censored"].dtype == "bool"

    def test_concentrations_non_negative(self, sample_water_quality_df: pd.DataFrame) -> None:
        assert (sample_water_quality_df["concentration"] >= 0).all()
