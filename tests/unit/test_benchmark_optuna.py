"""Unit tests for Optuna Bayesian HPO."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

optuna = pytest.importorskip("optuna")

from aquacontam.benchmark.optuna_hpo import (  # noqa: E402
    SEARCH_SPACES,
    _suggest_param,
    optuna_search,
)
from aquacontam.models.random_forest import RandomForestClassifier  # noqa: E402


@pytest.fixture()
def classification_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Small synthetic train/val split."""
    rng = np.random.RandomState(42)
    n = 80
    X = pd.DataFrame(
        {
            "f1": rng.randn(n),
            "f2": rng.randn(n),
            "f3": rng.randn(n),
        }
    )
    y = pd.Series((rng.randn(n) > 0).astype(int), name="target")
    return X.iloc[:60], y.iloc[:60], X.iloc[60:], y.iloc[60:]


class TestSearchSpaces:
    """Tests for predefined search spaces."""

    def test_search_spaces_defined(self) -> None:
        assert "xgboost_classifier" in SEARCH_SPACES
        assert "random_forest_classifier" in SEARCH_SPACES

    def test_search_space_keys_are_strings(self) -> None:
        for _space_name, space in SEARCH_SPACES.items():
            for param_name, spec in space.items():
                assert isinstance(param_name, str)
                assert isinstance(spec, tuple)
                assert len(spec) >= 2

    def test_lightgbm_and_catboost_spaces(self) -> None:
        assert "lightgbm_classifier" in SEARCH_SPACES
        assert "catboost_classifier" in SEARCH_SPACES


class TestSuggestParam:
    """Tests for _suggest_param helper."""

    def test_int_param(self) -> None:
        trial = optuna.trial.FixedTrial({"depth": 6})
        val = _suggest_param(trial, "depth", ("int", "3", "10"))
        assert val == 6
        assert isinstance(val, int)

    def test_float_param(self) -> None:
        trial = optuna.trial.FixedTrial({"subsample": 0.8})
        val = _suggest_param(trial, "subsample", ("float", "0.5", "1.0"))
        assert val == pytest.approx(0.8)

    def test_float_log_param(self) -> None:
        trial = optuna.trial.FixedTrial({"lr": 0.05})
        val = _suggest_param(trial, "lr", ("float_log", "0.001", "0.3"))
        assert val == pytest.approx(0.05)

    def test_categorical_param(self) -> None:
        trial = optuna.trial.FixedTrial({"feat": "sqrt"})
        val = _suggest_param(trial, "feat", ("categorical", "sqrt", "log2"))
        assert val == "sqrt"

    def test_categorical_none(self) -> None:
        trial = optuna.trial.FixedTrial({"depth": None})
        val = _suggest_param(trial, "depth", ("categorical", "10", "None"))
        assert val is None

    def test_unknown_type_raises(self) -> None:
        trial = optuna.trial.FixedTrial({})
        with pytest.raises(ValueError, match="Unknown type spec"):
            _suggest_param(trial, "x", ("unknown", "1", "2"))


class TestOptunaSearch:
    """Tests for optuna_search function."""

    def test_search_returns_best_config(
        self,
        classification_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    ) -> None:
        X_train, y_train, X_val, y_val = classification_data
        best_config, study = optuna_search(
            RandomForestClassifier,
            X_train,
            y_train,
            X_val,
            y_val,
            n_trials=5,
            metric="accuracy",
            seed=42,
        )
        assert isinstance(best_config, dict)
        assert "random_state" in best_config
        assert study.best_value > 0.0
        assert len(study.trials) == 5

    def test_custom_search_space(
        self,
        classification_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    ) -> None:
        X_train, y_train, X_val, y_val = classification_data
        custom_space = {
            "n_estimators": ("int", "50", "200"),
            "max_depth": ("categorical", "5", "10"),
        }
        best_config, study = optuna_search(
            RandomForestClassifier,
            X_train,
            y_train,
            X_val,
            y_val,
            search_space=custom_space,
            n_trials=3,
            metric="accuracy",
            seed=42,
        )
        assert "n_estimators" in best_config
        assert study.best_value > 0.0

    def test_with_base_config(
        self,
        classification_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    ) -> None:
        X_train, y_train, X_val, y_val = classification_data
        base = {"n_jobs": 1, "class_weight": "balanced"}
        best_config, _study = optuna_search(
            RandomForestClassifier,
            X_train,
            y_train,
            X_val,
            y_val,
            base_config=base,
            n_trials=3,
            metric="accuracy",
            seed=42,
        )
        assert best_config["n_jobs"] == 1
        assert best_config["class_weight"] == "balanced"
