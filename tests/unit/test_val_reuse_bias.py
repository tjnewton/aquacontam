"""Tests for aquacontam.analysis.val_reuse_bias."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquacontam.analysis.val_reuse_bias import (
    _evaluate_per_region,
    _make_internal_val_split,
    run_val_reuse_experiment,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_data(
    n_per_region: int = 40,
    n_features: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create synthetic data with 10 EPA regions."""
    rng = np.random.RandomState(seed)
    regions_list = []
    ids = []
    for r in range(1, 11):
        regions_list.extend([r] * n_per_region)
        ids.extend([f"R{r:02d}{i:04d}" for i in range(n_per_region)])
    n = len(ids)
    X = pd.DataFrame(
        rng.randn(n, n_features),
        index=ids,
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    # Target correlated with first feature + noise
    prob = 1 / (1 + np.exp(-(X["feat_0"].to_numpy() + 0.5 * rng.randn(n))))
    y = pd.Series((prob > 0.5).astype(int), index=ids, name="target")
    regions = pd.Series(regions_list, index=ids, name="epa_region")
    return X, y, regions


# ---------------------------------------------------------------------------
# TestMakeInternalValSplit
# ---------------------------------------------------------------------------


class TestMakeInternalValSplit:
    """Tests for _make_internal_val_split."""

    def test_correct_fraction(self):
        X, y, _ = _make_synthetic_data()
        X_tr, _y_tr, X_vl, _y_vl = _make_internal_val_split(X, y, 0.15, seed=42)
        expected_n_val = max(1, round(len(X) * 0.15))
        assert len(X_vl) == expected_n_val
        assert len(X_tr) == len(X) - expected_n_val

    def test_disjoint_indices(self):
        X, y, _ = _make_synthetic_data()
        X_tr, _, X_vl, _ = _make_internal_val_split(X, y, 0.2, seed=123)
        assert len(X_tr.index.intersection(X_vl.index)) == 0

    def test_union_equals_original(self):
        X, y, _ = _make_synthetic_data()
        X_tr, _y_tr, X_vl, _y_vl = _make_internal_val_split(X, y, 0.15, seed=42)
        combined = pd.concat([X_tr, X_vl]).sort_index()
        assert combined.shape == X.sort_index().shape

    def test_reproducible_with_seed(self):
        X, y, _ = _make_synthetic_data()
        _, _, X_vl_a, _ = _make_internal_val_split(X, y, 0.15, seed=99)
        _, _, X_vl_b, _ = _make_internal_val_split(X, y, 0.15, seed=99)
        pd.testing.assert_frame_equal(X_vl_a, X_vl_b)

    def test_different_seeds_differ(self):
        X, y, _ = _make_synthetic_data()
        _, _, X_vl_a, _ = _make_internal_val_split(X, y, 0.15, seed=42)
        _, _, X_vl_b, _ = _make_internal_val_split(X, y, 0.15, seed=99)
        assert not X_vl_a.index.equals(X_vl_b.index)


# ---------------------------------------------------------------------------
# TestEvaluatePerRegion
# ---------------------------------------------------------------------------


class TestEvaluatePerRegion:
    """Tests for _evaluate_per_region."""

    def test_computes_per_region_auroc(self):
        rng = np.random.RandomState(42)
        n = 100
        ids = [f"S{i:04d}" for i in range(n)]
        y_true = pd.Series(rng.randint(0, 2, n), index=ids)
        y_prob = np.clip(y_true.to_numpy() + rng.randn(n) * 0.3, 0, 1)
        regions = pd.Series([8] * 50 + [9] * 50, index=ids)

        result = _evaluate_per_region(y_true, y_prob, regions, (8, 9))
        assert set(result.keys()) == {8, 9}
        # Probabilities correlate with labels, so AUROC should be high
        assert result[8] > 0.7
        assert result[9] > 0.7

    def test_single_class_region_returns_nan(self):
        ids = [f"S{i:04d}" for i in range(20)]
        y_true = pd.Series([1] * 10 + [0, 1] * 5, index=ids)
        y_prob = np.random.RandomState(42).rand(20)
        regions = pd.Series([8] * 10 + [9] * 10, index=ids)
        # Region 8 is all-positive
        result = _evaluate_per_region(y_true, y_prob, regions, (8, 9))
        assert np.isnan(result[8])
        assert not np.isnan(result[9])

    def test_missing_region_returns_nan(self):
        ids = [f"S{i:04d}" for i in range(20)]
        y_true = pd.Series([0, 1] * 10, index=ids)
        y_prob = np.random.RandomState(42).rand(20)
        regions = pd.Series([8] * 20, index=ids)
        result = _evaluate_per_region(y_true, y_prob, regions, (8, 99))
        assert not np.isnan(result[8])
        assert np.isnan(result[99])


# ---------------------------------------------------------------------------
# TestRunValReuseExperiment
# ---------------------------------------------------------------------------


class TestRunValReuseExperiment:
    """Tests for run_val_reuse_experiment."""

    @pytest.fixture()
    def synthetic_data(self):
        return _make_synthetic_data(n_per_region=30, n_features=5, seed=42)

    @pytest.fixture()
    def small_config(self):
        return {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.3,
            "early_stopping_rounds": 3,
            "eval_metric": "logloss",
        }

    def test_returns_three_conditions(self, synthetic_data, small_config):
        pytest.importorskip("xgboost")
        X, y, regions = synthetic_data
        result = run_val_reuse_experiment(
            X,
            y,
            regions,
            model_config=small_config,
            seeds=[42],
        )
        assert "conditions" in result
        assert set(result["conditions"].keys()) == {
            "triple_use",
            "internal_val",
            "no_es",
        }

    def test_condition_has_expected_keys(self, synthetic_data, small_config):
        pytest.importorskip("xgboost")
        X, y, regions = synthetic_data
        result = run_val_reuse_experiment(
            X,
            y,
            regions,
            model_config=small_config,
            seeds=[42],
        )
        for cond in result["conditions"].values():
            assert "mean_auroc" in cond
            assert "std_auroc" in cond
            assert "per_region" in cond
            assert "auroc_per_seed" in cond
            assert "n_train" in cond
            assert "n_test" in cond

    def test_per_region_breakdown(self, synthetic_data, small_config):
        pytest.importorskip("xgboost")
        X, y, regions = synthetic_data
        result = run_val_reuse_experiment(
            X,
            y,
            regions,
            model_config=small_config,
            seeds=[42],
        )
        for cond in result["conditions"].values():
            for r in ("8", "9", "10"):
                assert r in cond["per_region"]
                assert "mean_auroc" in cond["per_region"][r]

    def test_decomposition_arithmetic(self, synthetic_data, small_config):
        pytest.importorskip("xgboost")
        X, y, regions = synthetic_data
        result = run_val_reuse_experiment(
            X,
            y,
            regions,
            model_config=small_config,
            seeds=[42, 123],
        )
        d = result["decomposition"]
        # val_reuse_effect = A - C, val_identity = A - B, es_benefit = B - C
        # So val_identity + es_benefit should equal val_reuse_effect
        assert abs(d["val_identity_effect"] + d["es_benefit"] - d["val_reuse_effect"]) < 0.01

    def test_multi_seed_length(self, synthetic_data, small_config):
        pytest.importorskip("xgboost")
        X, y, regions = synthetic_data
        seeds = [42, 123, 456]
        result = run_val_reuse_experiment(
            X,
            y,
            regions,
            model_config=small_config,
            seeds=seeds,
        )
        for cond in result["conditions"].values():
            assert len(cond["auroc_per_seed"]) == len(seeds)

    def test_loro_comparison_populated(self, synthetic_data, small_config):
        pytest.importorskip("xgboost")
        X, y, regions = synthetic_data
        loro = {8: 0.65, 9: 0.70, 10: 0.95}
        result = run_val_reuse_experiment(
            X,
            y,
            regions,
            model_config=small_config,
            seeds=[42],
            loro_per_region=loro,
        )
        prc = result["per_region_comparison"]
        assert set(prc.keys()) == {"8", "9", "10"}
        for r_str in prc:
            assert "holdout_auroc" in prc[r_str]
            assert "loro_auroc" in prc[r_str]
            assert "delta" in prc[r_str]

    def test_loro_comparison_empty_without_input(self, synthetic_data, small_config):
        pytest.importorskip("xgboost")
        X, y, regions = synthetic_data
        result = run_val_reuse_experiment(
            X,
            y,
            regions,
            model_config=small_config,
            seeds=[42],
        )
        assert result["per_region_comparison"] == {}

    def test_methodology_note_present(self, synthetic_data, small_config):
        pytest.importorskip("xgboost")
        X, y, regions = synthetic_data
        result = run_val_reuse_experiment(
            X,
            y,
            regions,
            model_config=small_config,
            seeds=[42],
        )
        assert "methodology_note" in result
        assert "threshold-independent" in result["methodology_note"]

    def test_auroc_values_in_range(self, synthetic_data, small_config):
        pytest.importorskip("xgboost")
        X, y, regions = synthetic_data
        result = run_val_reuse_experiment(
            X,
            y,
            regions,
            model_config=small_config,
            seeds=[42],
        )
        for cond in result["conditions"].values():
            assert 0.0 <= cond["mean_auroc"] <= 1.0
            for a in cond["auroc_per_seed"]:
                assert 0.0 <= a <= 1.0
