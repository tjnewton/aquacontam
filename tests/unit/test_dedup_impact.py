"""Tests for deduplication impact on system-level aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquacontam.features.assembly import aggregate_to_system_level


def _make_wq_with_duplicates(n_systems: int = 20, n_dupes_per: int = 3) -> pd.DataFrame:
    """Create water quality DataFrame with intentional duplicate samples."""
    rng = np.random.RandomState(42)
    rows = []
    for i in range(n_systems):
        pwsid = f"SYS{i:07d}"
        is_positive = rng.random() < 0.3
        # Original samples
        for j in range(3):
            dl = rng.uniform(1.0, 5.0)
            conc = dl + rng.exponential(5.0) if is_positive else 0.0
            rows.append(
                {
                    "pwsid": pwsid,
                    "analyte": "PFOS",
                    "concentration": conc,
                    "censored": not is_positive,
                    "detection_limit": dl,
                    "sample_date": pd.Timestamp("2023-01-15") + pd.Timedelta(days=j * 30),
                }
            )
        # Duplicate samples (same date as first sample)
        for _ in range(n_dupes_per):
            dl = rng.uniform(1.0, 5.0)
            conc = dl + rng.exponential(5.0) if is_positive else 0.0
            rows.append(
                {
                    "pwsid": pwsid,
                    "analyte": "PFOS",
                    "concentration": conc,
                    "censored": not is_positive,
                    "detection_limit": dl,
                    "sample_date": pd.Timestamp("2023-01-15"),
                }
            )
    return pd.DataFrame(rows)


class TestDedupImpactOnAggregation:
    """Tests for deduplicate_strategy parameter in aggregate_to_system_level."""

    def test_none_preserves_backward_compat(self):
        """Default None should give identical results to previous behavior."""
        df = _make_wq_with_duplicates()
        result_none = aggregate_to_system_level(df, "PFOS", deduplicate_strategy=None)
        result_default = aggregate_to_system_level(df, "PFOS")
        pd.testing.assert_frame_equal(result_none, result_default)

    def test_dedup_reduces_n_samples(self):
        """Deduplication should reduce n_samples compared to no dedup."""
        df = _make_wq_with_duplicates(n_systems=20, n_dupes_per=3)
        result_no_dedup = aggregate_to_system_level(df, "PFOS", deduplicate_strategy=None)
        result_dedup = aggregate_to_system_level(df, "PFOS", deduplicate_strategy="max")
        assert result_dedup["n_samples"].sum() < result_no_dedup["n_samples"].sum()

    def test_dedup_preserves_system_count(self):
        """Deduplication should not change the number of systems."""
        df = _make_wq_with_duplicates()
        result_no_dedup = aggregate_to_system_level(df, "PFOS")
        result_dedup = aggregate_to_system_level(df, "PFOS", deduplicate_strategy="max")
        assert len(result_dedup) == len(result_no_dedup)

    def test_dedup_preserves_target(self):
        """Target labels should be identical (max aggregation preserves detections)."""
        df = _make_wq_with_duplicates()
        result_no_dedup = aggregate_to_system_level(df, "PFOS")
        result_dedup = aggregate_to_system_level(df, "PFOS", deduplicate_strategy="max")
        # Sort to align, then compare targets
        common = result_no_dedup.index.intersection(result_dedup.index)
        pd.testing.assert_series_equal(
            result_no_dedup.loc[common, "target"],
            result_dedup.loc[common, "target"],
        )

    def test_strategies_all_valid(self):
        """All three valid strategies should run without error."""
        df = _make_wq_with_duplicates(n_systems=5)
        for strategy in ("max", "mean", "first"):
            result = aggregate_to_system_level(df, "PFOS", deduplicate_strategy=strategy)
            assert not result.empty

    def test_invalid_strategy_raises(self):
        """Invalid strategy should raise ValueError."""
        df = _make_wq_with_duplicates(n_systems=5)
        with pytest.raises(ValueError, match="deduplicate_strategy must be one of"):
            aggregate_to_system_level(df, "PFOS", deduplicate_strategy="invalid")

    def test_no_duplicates_is_noop(self):
        """If data has no duplicates, result should match no-dedup."""
        rng = np.random.RandomState(99)
        rows = []
        for i in range(10):
            pwsid = f"SYS{i:07d}"
            for j in range(3):
                rows.append(
                    {
                        "pwsid": pwsid,
                        "analyte": "PFOS",
                        "concentration": rng.uniform(0, 10),
                        "censored": rng.random() < 0.5,
                        "detection_limit": 2.0,
                        "sample_date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=j * 30),
                    }
                )
        df = pd.DataFrame(rows)
        result_none = aggregate_to_system_level(df, "PFOS")
        result_dedup = aggregate_to_system_level(df, "PFOS", deduplicate_strategy="max")
        pd.testing.assert_frame_equal(result_none, result_dedup)
