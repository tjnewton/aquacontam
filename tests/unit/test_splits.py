"""Tests for geographic stratification splits."""

from __future__ import annotations

import pandas as pd
import pytest

from aquacontam._constants import EPA_REGIONS, STATE_TO_EPA_REGION
from aquacontam.preprocessing.splits import (
    assign_epa_region,
    geographic_split,
    leave_one_region_out,
    random_split,
    split_summary,
)


@pytest.fixture()
def multi_state_df() -> pd.DataFrame:
    """DataFrame with systems across multiple EPA regions."""
    return pd.DataFrame(
        {
            "pwsid": [
                "MA0000001",  # Region 1
                "NY0000002",  # Region 2
                "PA0000003",  # Region 3
                "FL0000004",  # Region 4
                "IL0000005",  # Region 5
                "TX0000006",  # Region 6
                "KS0000007",  # Region 7
                "CO0000008",  # Region 8
                "CA0000009",  # Region 9
                "WA0000010",  # Region 10
            ],
            "analyte": ["PFOS"] * 10,
            "concentration": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "censored": [False] * 5 + [True] * 5,
        }
    )


class TestAssignEpaRegion:
    """Tests for assign_epa_region."""

    def test_basic_assignment(self, multi_state_df: pd.DataFrame) -> None:
        result = assign_epa_region(multi_state_df)
        assert "epa_region" in result.columns
        assert result.loc[0, "epa_region"] == 1  # MA → Region 1
        assert result.loc[8, "epa_region"] == 9  # CA → Region 9

    def test_all_states_mapped(self) -> None:
        """Every state in EPA_REGIONS should map correctly."""
        for region, states in EPA_REGIONS.items():
            for state in states:
                assert STATE_TO_EPA_REGION[state] == region

    def test_pwsid_column_required(self) -> None:
        df = pd.DataFrame({"not_pwsid": ["XX0000001"]})
        with pytest.raises(ValueError, match="pwsid"):
            assign_epa_region(df)

    def test_unknown_state_code_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        df = pd.DataFrame({"pwsid": ["ZZ0000001"]})
        with caplog.at_level("WARNING"):
            result = assign_epa_region(df)
        assert "could not be assigned" in caplog.text
        assert result["epa_region"].isna().all()

    def test_fallback_to_raw_region(self) -> None:
        """Should use raw_Region as fallback for unknown state codes."""
        df = pd.DataFrame(
            {
                "pwsid": ["XX0000001"],
                "raw_Region": [5],
            }
        )
        result = assign_epa_region(df)
        assert result.loc[0, "epa_region"] == 5

    def test_does_not_modify_original(self, multi_state_df: pd.DataFrame) -> None:
        original_cols = list(multi_state_df.columns)
        assign_epa_region(multi_state_df)
        assert list(multi_state_df.columns) == original_cols

    def test_wqp_systems_get_region_via_coordinates(self) -> None:
        """WQP synthetic IDs (prefix 'WQ') should get EPA region from lat/lon fallback."""
        df = pd.DataFrame(
            {
                "pwsid": ["WQP_1234567", "WQP_9876543"],
                "latitude": [30.27, 47.61],  # Austin TX (Region 6), Seattle WA (Region 10)
                "longitude": [-97.74, -122.33],
            }
        )
        result = assign_epa_region(df)
        # WQ prefix is not in STATE_TO_EPA_REGION, so coordinate fallback should fire
        assert result.loc[0, "epa_region"] == 6  # TX → Region 6
        assert result.loc[1, "epa_region"] == 10  # WA → Region 10


class TestGeographicSplit:
    """Tests for geographic_split."""

    def test_default_split(self, multi_state_df: pd.DataFrame) -> None:
        df = assign_epa_region(multi_state_df)
        train, val, test = geographic_split(df)
        # Train: regions 1,3,4,5,6 → MA,PA,FL,IL,TX = 5
        assert len(train) == 5
        # Val: regions 2,7 → NY,KS = 2
        assert len(val) == 2
        # Test: regions 8,9,10 → CO,CA,WA = 3
        assert len(test) == 3

    def test_no_data_leakage(self, multi_state_df: pd.DataFrame) -> None:
        """Train, val, test should have non-overlapping regions."""
        df = assign_epa_region(multi_state_df)
        train, val, test = geographic_split(df)

        train_regions = set(train["epa_region"].unique())
        val_regions = set(val["epa_region"].unique())
        test_regions = set(test["epa_region"].unique())

        assert train_regions & val_regions == set()
        assert train_regions & test_regions == set()
        assert val_regions & test_regions == set()

    def test_total_preserved(self, multi_state_df: pd.DataFrame) -> None:
        """Total rows should equal sum of splits."""
        df = assign_epa_region(multi_state_df)
        train, val, test = geographic_split(df)
        assert len(train) + len(val) + len(test) == len(df)

    def test_overlapping_regions_raises(self, multi_state_df: pd.DataFrame) -> None:
        df = assign_epa_region(multi_state_df)
        with pytest.raises(ValueError, match="Overlapping"):
            geographic_split(df, train_regions=(1, 2), val_regions=(2, 3))

    def test_missing_column_raises(self, multi_state_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="not found"):
            geographic_split(multi_state_df, region_column="nonexistent")

    def test_custom_regions(self, multi_state_df: pd.DataFrame) -> None:
        df = assign_epa_region(multi_state_df)
        train, val, test = geographic_split(
            df,
            train_regions=(1, 2, 3),
            val_regions=(4, 5),
            test_regions=(6, 7, 8, 9, 10),
        )
        assert len(train) == 3
        assert len(val) == 2
        assert len(test) == 5


class TestSplitSummary:
    """Tests for split_summary."""

    def test_summary_structure(self, multi_state_df: pd.DataFrame) -> None:
        df = assign_epa_region(multi_state_df)
        train, val, test = geographic_split(df)
        summary = split_summary(train, val, test)
        assert "total_samples" in summary
        assert "train" in summary
        assert "val" in summary
        assert "test" in summary

    def test_summary_sizes_correct(self, multi_state_df: pd.DataFrame) -> None:
        df = assign_epa_region(multi_state_df)
        train, val, test = geographic_split(df)
        summary = split_summary(train, val, test)
        assert summary["total_samples"] == 10
        assert summary["train"]["n_samples"] == 5
        assert summary["val"]["n_samples"] == 2
        assert summary["test"]["n_samples"] == 3

    def test_summary_censoring_rate(self, multi_state_df: pd.DataFrame) -> None:
        df = assign_epa_region(multi_state_df)
        train, val, test = geographic_split(df)
        summary = split_summary(train, val, test)
        # All censoring rates should be between 0 and 1
        for name in ("train", "val", "test"):
            rate = summary[name]["censoring_rate"]
            assert 0.0 <= rate <= 1.0


class TestRandomSplit:
    """Tests for random_split."""

    def test_basic_split(self, multi_state_df: pd.DataFrame) -> None:
        train, val, test = random_split(multi_state_df)
        assert len(train) + len(val) + len(test) == len(multi_state_df)

    def test_fractions(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        train, val, test = random_split(df, train_frac=0.60, val_frac=0.15)
        assert len(train) == 60
        assert len(val) == 15
        assert len(test) == 25

    def test_reproducibility(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        t1, v1, te1 = random_split(df, seed=42)
        t2, v2, te2 = random_split(df, seed=42)
        assert list(t1["x"]) == list(t2["x"])
        assert list(v1["x"]) == list(v2["x"])
        assert list(te1["x"]) == list(te2["x"])

    def test_different_seeds(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        t1, _, _ = random_split(df, seed=42)
        t2, _, _ = random_split(df, seed=99)
        assert list(t1["x"]) != list(t2["x"])

    def test_invalid_fractions_raises(self) -> None:
        df = pd.DataFrame({"x": range(10)})
        with pytest.raises(ValueError, match=r"<= 1\.0"):
            random_split(df, train_frac=0.8, val_frac=0.3)


class TestLeaveOneRegionOut:
    """Tests for leave_one_region_out LORO-CV."""

    def test_produces_folds(self, multi_state_df: pd.DataFrame) -> None:
        df = assign_epa_region(multi_state_df)
        folds = leave_one_region_out(df)
        assert len(folds) == 10  # one fold per EPA region

    def test_no_data_leakage(self, multi_state_df: pd.DataFrame) -> None:
        """Test region in test set never appears in train or val."""
        df = assign_epa_region(multi_state_df)
        folds = leave_one_region_out(df)
        for train, val, test, test_region in folds:
            if len(test) > 0:
                test_regions = set(test["epa_region"].unique())
                assert test_regions == {test_region}
            if len(train) > 0:
                assert test_region not in set(train["epa_region"].unique())
            if len(val) > 0:
                assert test_region not in set(val["epa_region"].unique())

    def test_all_data_used(self, multi_state_df: pd.DataFrame) -> None:
        """Each sample appears in exactly one test fold."""
        df = assign_epa_region(multi_state_df)
        folds = leave_one_region_out(df)
        total_test = sum(len(test) for _, _, test, _ in folds)
        assert total_test == len(df)

    def test_train_val_non_overlapping(self, multi_state_df: pd.DataFrame) -> None:
        df = assign_epa_region(multi_state_df)
        folds = leave_one_region_out(df)
        for train, val, _test, _region in folds:
            if len(train) > 0 and len(val) > 0:
                train_regions = set(train["epa_region"].unique())
                val_regions = set(val["epa_region"].unique())
                assert train_regions & val_regions == set()

    def test_missing_column_raises(self, multi_state_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="not found"):
            leave_one_region_out(multi_state_df, region_column="nonexistent")

    def test_reproducibility(self, multi_state_df: pd.DataFrame) -> None:
        df = assign_epa_region(multi_state_df)
        folds1 = leave_one_region_out(df, seed=42)
        folds2 = leave_one_region_out(df, seed=42)
        for (t1, v1, te1, r1), (t2, v2, te2, r2) in zip(folds1, folds2, strict=True):
            assert r1 == r2
            assert len(t1) == len(t2)
            assert len(v1) == len(v2)
            assert len(te1) == len(te2)


class TestBackwardCompatShim:
    """Verify imports from the old aquacontam.splits path still work."""

    def test_old_import_path(self) -> None:
        from aquacontam.splits import (
            assign_epa_region as _a,
        )
        from aquacontam.splits import (
            geographic_split as _g,
        )
        from aquacontam.splits import (
            split_summary as _s,
        )

        assert callable(_a)
        assert callable(_g)
        assert callable(_s)
