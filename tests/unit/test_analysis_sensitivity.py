"""Tests for hyperparameter and detection-limit sensitivity analysis."""

from __future__ import annotations

from aquacontam.analysis.sensitivity import (
    detection_limit_sensitivity,
    generate_config_variants,
    hyperparameter_sensitivity,
)
from aquacontam.models.xgboost import XGBoostClassifier


class TestGenerateConfigVariants:
    def test_basic_grid(self):
        base = {"n_estimators": 100, "random_state": 42}
        grid = {"max_depth": [4, 6], "learning_rate": [0.01, 0.1]}
        configs = generate_config_variants(base, grid)
        assert len(configs) == 4
        assert all("max_depth" in c for c in configs)
        assert all(c["random_state"] == 42 for c in configs)

    def test_max_configs(self):
        base = {}
        grid = {"a": [1, 2, 3], "b": [4, 5, 6]}
        configs = generate_config_variants(base, grid, max_configs=3)
        assert len(configs) == 3


class TestHyperparameterSensitivity:
    def test_basic_sensitivity(self, classification_data):
        X, y = classification_data
        X_train, X_val = X[:70], X[70:]
        y_train, y_val = y[:70], y[70:]

        configs = [
            {"n_estimators": 10, "max_depth": 3, "random_state": 42},
            {"n_estimators": 10, "max_depth": 6, "random_state": 42},
        ]
        result = hyperparameter_sensitivity(
            XGBoostClassifier, configs, X_train, y_train, X_val, y_val, metric="auroc"
        )
        assert "per_config_results" in result
        assert "summary" in result
        assert result["summary"]["n_configs"] == 2


class TestDetectionLimitSensitivity:
    def test_basic_sensitivity(self, classification_data):
        X, y = classification_data
        X_train, X_val = X[:70], X[70:]
        y_train, y_val = y[:70], y[70:]

        result = detection_limit_sensitivity(
            XGBoostClassifier,
            X_train,
            y_train,
            X_val,
            y_val,
            fillna_values=[0.0, 0.004, 0.01],
            metric="auroc",
            config={"n_estimators": 10, "max_depth": 3, "random_state": 42},
        )
        assert "per_value_results" in result
        assert "summary" in result
        assert result["summary"]["n_values"] == 3
        assert result["summary"]["n_successful"] == 3

    def test_default_fillna_values(self, classification_data):
        X, y = classification_data
        X_train, X_val = X[:70], X[70:]
        y_train, y_val = y[:70], y[70:]

        result = detection_limit_sensitivity(
            XGBoostClassifier,
            X_train,
            y_train,
            X_val,
            y_val,
            config={"n_estimators": 10, "max_depth": 3, "random_state": 42},
        )
        # Default has 6 fill values
        assert result["summary"]["n_values"] == 6
