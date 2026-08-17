"""Tests for shared data utility functions."""

from __future__ import annotations

import pandas as pd
import pytest

from aquacontam._constants import MGL_TO_UGL, PPT_TO_UGL
from aquacontam.data._utils import PGL_TO_UGL, convert_to_ugl, find_col, normalize_pwsid


class TestFindCol:
    """Tests for find_col."""

    def test_finds_first_match(self) -> None:
        df = pd.DataFrame({"A": [1], "B": [2]})
        assert find_col(df, ["C", "A", "B"]) == "A"

    def test_returns_first_candidate_when_multiple_exist(self) -> None:
        df = pd.DataFrame({"A": [1], "B": [2]})
        assert find_col(df, ["A", "B"]) == "A"

    def test_raises_when_required_missing(self) -> None:
        df = pd.DataFrame({"A": [1]})
        with pytest.raises(ValueError, match="None of"):
            find_col(df, ["X", "Y"])

    def test_returns_none_when_optional(self) -> None:
        df = pd.DataFrame({"A": [1]})
        assert find_col(df, ["X"], required=False) is None

    def test_required_true_by_default(self) -> None:
        df = pd.DataFrame({"A": [1]})
        with pytest.raises(ValueError):
            find_col(df, ["X"])


class TestNormalizePwsid:
    """Tests for normalize_pwsid."""

    def test_already_9_chars_noop(self) -> None:
        assert normalize_pwsid("NJ0000045", "NJ") == "NJ0000045"

    def test_short_with_prefix(self) -> None:
        assert normalize_pwsid("NJ45", "NJ") == "NJ0000045"

    def test_short_without_prefix(self) -> None:
        assert normalize_pwsid("123", "NJ") == "NJ0000123"

    def test_empty_string(self) -> None:
        assert normalize_pwsid("", "NJ") == "NJ0000000"

    def test_case_insensitive_prefix(self) -> None:
        assert normalize_pwsid("nj45", "NJ") == "NJ0000045"

    def test_mi_prefix(self) -> None:
        assert normalize_pwsid("MI45", "MI") == "MI0000045"
        assert normalize_pwsid("123", "MI") == "MI0000123"

    def test_nc_prefix(self) -> None:
        assert normalize_pwsid("NC100", "NC") == "NC0000100"

    def test_long_value_unchanged(self) -> None:
        assert normalize_pwsid("NJ12345678", "NJ") == "NJ12345678"

    def test_whitespace_stripped(self) -> None:
        assert normalize_pwsid("  NJ45  ", "NJ") == "NJ0000045"


class TestConvertToUgl:
    """Tests for convert_to_ugl."""

    def test_ugl_noop(self) -> None:
        assert convert_to_ugl(5.0, "UG/L") == 5.0

    def test_empty_unit_noop(self) -> None:
        assert convert_to_ugl(5.0, "") == 5.0

    def test_ppt(self) -> None:
        assert convert_to_ugl(1000.0, "PPT") == pytest.approx(1000.0 * PPT_TO_UGL)

    def test_ngl(self) -> None:
        assert convert_to_ugl(1000.0, "NG/L") == pytest.approx(1000.0 * PPT_TO_UGL)

    def test_pgl(self) -> None:
        assert convert_to_ugl(1000.0, "PG/L") == pytest.approx(1000.0 * PGL_TO_UGL)

    def test_mgl(self) -> None:
        assert convert_to_ugl(1.0, "MG/L") == pytest.approx(1.0 * MGL_TO_UGL)

    def test_case_insensitive(self) -> None:
        assert convert_to_ugl(5.0, "ug/l") == 5.0
        assert convert_to_ugl(1.0, "mg/l") == pytest.approx(MGL_TO_UGL)

    def test_whitespace_stripped(self) -> None:
        assert convert_to_ugl(5.0, "  UG/L  ") == 5.0

    def test_unknown_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown unit"):
            convert_to_ugl(5.0, "GALLONS")
