"""Tests for the schema validation logic in data/schema.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquacontam.data.schema import (
    validate_concentrations,
    validate_coordinates,
    validate_pwsid,
    validate_schema,
)


def _make_valid_df(n: int = 3) -> pd.DataFrame:
    """Create a small valid water quality DataFrame."""
    return pd.DataFrame(
        {
            "pwsid": ["CA0101001", "TX0200002", "MI0300003"][:n],
            "analyte": ["PFOS", "PFOA", "PFHxS"][:n],
            "concentration": [12.5, 0.0, 8.3][:n],
            "unit": ["ug/L", "ug/L", "ug/L"][:n],
            "censored": [False, True, False][:n],
            "detection_limit": [2.0, 4.0, 2.0][:n],
            "sample_date": pd.to_datetime(["2023-01-15", "2023-02-20", "2023-03-10"][:n]),
            "latitude": [34.05, 30.27, 42.33][:n],
            "longitude": [-118.24, -97.74, -83.05][:n],
        }
    )


class TestValidateSchema:
    """Tests for the main validate_schema entry point."""

    def test_valid_df_passes(self) -> None:
        df = _make_valid_df()
        result = validate_schema(df)
        assert len(result) == 3

    def test_missing_columns_raises(self) -> None:
        df = _make_valid_df().drop(columns=["pwsid", "analyte"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df, require_all_columns=True)

    def test_missing_columns_warns_when_not_required(self) -> None:
        df = _make_valid_df().drop(columns=["latitude", "longitude"])
        result = validate_schema(df, require_all_columns=False, skip_coord_check=True)
        assert len(result) == 3

    def test_dtype_coercion(self) -> None:
        df = _make_valid_df()
        # Make concentration a string to test coercion
        df["concentration"] = df["concentration"].astype(str)
        result = validate_schema(df, coerce_dtypes=True)
        assert result["concentration"].dtype == np.float64

    def test_drop_invalid_rows(self) -> None:
        df = _make_valid_df()
        # Make one PWSID invalid (too short)
        df.loc[0, "pwsid"] = "BAD"
        result = validate_schema(df, drop_invalid_rows=True)
        assert len(result) == 2

    def test_skip_coord_check(self) -> None:
        df = _make_valid_df()
        df["latitude"] = float("nan")
        df["longitude"] = float("nan")
        result = validate_schema(df, skip_coord_check=True)
        assert len(result) == 3

    def test_coord_check_flags_out_of_bounds(self) -> None:
        df = _make_valid_df()
        df.loc[0, "latitude"] = 90.0  # North Pole — outside CONUS
        result = validate_schema(df, drop_invalid_rows=True)
        assert len(result) == 2


class TestValidatePWSID:
    """Tests for PWSID validation."""

    def test_valid_pwsids(self) -> None:
        series = pd.Series(["CA0101001", "TX0200002"])
        mask = validate_pwsid(series)
        assert mask.all()

    def test_invalid_length(self) -> None:
        series = pd.Series(["SHORT", "CA0101001"])
        mask = validate_pwsid(series)
        assert not mask.iloc[0]
        assert mask.iloc[1]

    def test_non_string(self) -> None:
        series = pd.Series([101001, "CA0101001"])
        mask = validate_pwsid(series)
        assert not mask.iloc[0]
        assert mask.iloc[1]


class TestValidateConcentrations:
    """Tests for concentration validation."""

    def test_valid_concentrations(self) -> None:
        df = _make_valid_df()
        mask = validate_concentrations(df)
        assert mask.all()

    def test_negative_concentration(self) -> None:
        df = _make_valid_df()
        df.loc[0, "concentration"] = -1.0
        mask = validate_concentrations(df)
        assert not mask.iloc[0]

    def test_zero_detection_limit_censored(self) -> None:
        """Zero DL on a censored row is invalid (must know the limit)."""
        df = _make_valid_df()
        # Row 1 is censored — set its DL to 0
        df.loc[1, "detection_limit"] = 0.0
        mask = validate_concentrations(df)
        assert not mask.iloc[1]

    def test_zero_detection_limit_uncensored(self) -> None:
        """Zero DL on an uncensored (detected) row is valid."""
        df = _make_valid_df()
        # Row 0 is not censored — zero DL is acceptable
        df.loc[0, "detection_limit"] = 0.0
        mask = validate_concentrations(df)
        assert mask.iloc[0]

    def test_censored_above_dl(self) -> None:
        df = _make_valid_df()
        # Row 1 is censored with DL=4.0; set concentration above DL
        df.loc[1, "concentration"] = 5.0
        mask = validate_concentrations(df)
        assert not mask.iloc[1]


class TestValidateCoordinates:
    """Tests for coordinate validation."""

    def test_valid_conus_coords(self) -> None:
        df = _make_valid_df()
        mask = validate_coordinates(df)
        assert mask.all()

    def test_nan_coords_are_invalid(self) -> None:
        df = _make_valid_df()
        df.loc[0, "latitude"] = float("nan")
        mask = validate_coordinates(df)
        assert not mask.iloc[0]

    def test_outside_conus(self) -> None:
        df = _make_valid_df()
        df.loc[0, "latitude"] = 65.0  # Alaska
        df.loc[0, "longitude"] = -150.0
        mask = validate_coordinates(df)
        assert not mask.iloc[0]


class TestEdgeCases:
    """Edge case tests for schema validation."""

    def test_empty_dataframe_returns_empty(self) -> None:
        df = _make_valid_df()
        df = df.iloc[:0]  # keep schema/dtypes, zero rows
        result = validate_schema(df, drop_invalid_rows=True)
        assert len(result) == 0

    def test_all_censored_zero_concentration(self) -> None:
        df = _make_valid_df()
        df["censored"] = True
        df["concentration"] = 0.0
        result = validate_schema(df, drop_invalid_rows=False)
        assert len(result) == 3

    def test_empty_dataframe_missing_columns_raises(self) -> None:
        df = pd.DataFrame()  # no columns at all
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df, require_all_columns=True)

    def test_coords_outside_conus_dropped(self) -> None:
        df = _make_valid_df()
        df.loc[0, "latitude"] = 90.0  # North Pole
        df.loc[0, "longitude"] = 0.0
        result = validate_schema(df, skip_coord_check=False, drop_invalid_rows=True)
        assert len(result) == 2
        assert "CA0101001" not in result["pwsid"].to_numpy()
