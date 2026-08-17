"""Tests for Optuna Bayesian hyperparameter optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquacontam.benchmark.optuna_hpo import (
    PARAM_TRANSFORMS,
    SEARCH_SPACES,
    _apply_param_transforms,
)


class TestSearchSpaces:
    def test_mlp_in_search_spaces(self) -> None:
        assert "mlp_classifier" in SEARCH_SPACES

    def test_cnn1d_in_search_spaces(self) -> None:
        assert "cnn1d_classifier" in SEARCH_SPACES

    def test_tree_models_still_present(self) -> None:
        for key in [
            "xgboost_classifier",
            "random_forest_classifier",
            "lightgbm_classifier",
            "catboost_classifier",
        ]:
            assert key in SEARCH_SPACES


class TestParamTransforms:
    def test_mlp_hidden_sizes_transform(self) -> None:
        config: dict = {"hidden_sizes": "medium_3", "learning_rate": 0.001}
        result = _apply_param_transforms(config, "mlp_classifier")
        assert result["hidden_sizes"] == [128, 64, 32]
        # learning_rate should be unchanged
        assert result["learning_rate"] == 0.001

    def test_cnn1d_n_filters_transform(self) -> None:
        config: dict = {"n_filters": "large_3", "dropout": 0.3}
        result = _apply_param_transforms(config, "cnn1d_classifier")
        assert result["n_filters"] == [64, 128, 64]
        assert result["dropout"] == 0.3

    def test_no_transform_for_tree_models(self) -> None:
        config: dict = {"max_depth": 6, "learning_rate": 0.05}
        result = _apply_param_transforms(config, "xgboost_classifier")
        assert result == {"max_depth": 6, "learning_rate": 0.05}

    def test_unknown_key_left_unchanged(self) -> None:
        config: dict = {"hidden_sizes": "nonexistent_key", "learning_rate": 0.01}
        result = _apply_param_transforms(config, "mlp_classifier")
        # Key not in mapping → left as-is
        assert result["hidden_sizes"] == "nonexistent_key"

    def test_all_mlp_keys_valid(self) -> None:
        """Every categorical choice in the MLP search space has a transform."""
        mlp_space = SEARCH_SPACES["mlp_classifier"]
        hidden_spec = mlp_space["hidden_sizes"]
        assert hidden_spec[0] == "categorical"
        mlp_transforms = PARAM_TRANSFORMS["mlp_classifier"]["hidden_sizes"]
        for key in hidden_spec[1:]:
            assert key in mlp_transforms, f"Missing transform for {key}"

    def test_all_cnn1d_keys_valid(self) -> None:
        """Every categorical choice in the CNN1D search space has a transform."""
        cnn_space = SEARCH_SPACES["cnn1d_classifier"]
        filters_spec = cnn_space["n_filters"]
        assert filters_spec[0] == "categorical"
        cnn_transforms = PARAM_TRANSFORMS["cnn1d_classifier"]["n_filters"]
        for key in filters_spec[1:]:
            assert key in cnn_transforms, f"Missing transform for {key}"


class TestOptunaSearchIntegration:
    def test_optuna_search_mlp(self) -> None:
        """Run optuna_search with n_trials=2 on tiny data, verify list output."""
        optuna = pytest.importorskip("optuna")  # noqa: F841
        torch = pytest.importorskip("torch")  # noqa: F841
        from aquacontam.benchmark.optuna_hpo import optuna_search
        from aquacontam.models.mlp import MLPClassifier

        rng = np.random.RandomState(42)
        n = 80
        X_train = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})
        y_train = pd.Series((X_train["f1"] > 0).astype(int))
        X_val = pd.DataFrame({"f1": rng.normal(0, 1, 20), "f2": rng.normal(0, 1, 20)})
        y_val = pd.Series((X_val["f1"] > 0).astype(int))

        best_config, study = optuna_search(
            MLPClassifier,
            X_train,
            y_train,
            X_val,
            y_val,
            n_trials=2,
            metric="accuracy",
            base_config={"epochs": 3, "patience": 2, "random_state": 42},
        )
        # hidden_sizes should be a list, not a string key
        assert isinstance(best_config["hidden_sizes"], list)
        assert all(isinstance(v, int) for v in best_config["hidden_sizes"])
        assert len(study.trials) == 2
