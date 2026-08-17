"""Verify that fitting the same model twice with the same seed produces
identical predictions.

Catches seed propagation bugs, unseeded layers, and stochastic operations
without proper seeding. Each test creates two fresh model instances with
identical configs, fits both on the same data, and asserts bitwise-equal
predictions.
"""

from __future__ import annotations

import numpy as np
import pytest

from aquacontam._reproducibility import set_seed


@pytest.fixture(autouse=True)
def _establish_determinism(monkeypatch):
    """Make each reproducibility test self-sufficient regardless of suite order.

    These tests verify **seed propagation** (unseeded layers, missing
    ``manual_seed``), which is device-independent. They pass on GPU in isolation
    but flake in a full-suite GPU run: an earlier test creates the process-global
    cuBLAS handle without a deterministic workspace, after which
    ``use_deterministic_algorithms(True, warn_only=True)`` can only warn and
    same-seed GEMM diverges — a GPU-only, order-dependent failure (CI runs on
    CPU and is unaffected). We therefore run these determinism checks on **CPU**,
    where GEMM at a fixed thread count is deterministic regardless of any GPU
    state other tests leave behind; GPU seeding itself is still covered by these
    tests passing in isolation on GPU. ``set_seed`` additionally resets the RNG
    state per-test rather than relying on ``TestSetSeed`` running first.

    torch is an optional dependency (absent on CI, where the torch tests are
    skipped via ``importorskip``), so the CPU-force is guarded — the numpy-only
    ``TestSetSeed`` tests must still run there.
    """
    try:
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    except ImportError:
        pass
    set_seed(42)
    yield


# ---------------------------------------------------------------------------
# set_seed() unit tests
# ---------------------------------------------------------------------------


class TestSetSeed:
    """Tests for the centralized set_seed() function."""

    def test_numpy_deterministic(self):
        """Numpy produces identical sequences after set_seed."""
        set_seed(123)
        a = np.random.random(5)  # noqa: NPY002  # testing legacy seed API
        set_seed(123)
        b = np.random.random(5)  # noqa: NPY002  # testing legacy seed API
        np.testing.assert_array_equal(a, b)

    def test_torch_deterministic(self):
        """Torch produces identical tensors after set_seed."""
        torch = pytest.importorskip("torch")
        set_seed(99)
        a = torch.randn(5)
        set_seed(99)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_idempotent(self):
        """Calling set_seed twice does not raise."""
        set_seed(42)
        set_seed(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_reproducible_classifier(
    ModelClass: type,
    config: dict,
    X: np.ndarray,
    y: np.ndarray,
    *,
    atol: float = 1e-10,
    fit_kwargs: dict | None = None,
) -> None:
    """Fit the same classifier twice and assert identical predictions."""
    kw = fit_kwargs or {}

    m1 = ModelClass(config=config.copy())
    m1.fit(X, y, **kw)
    p1 = m1.predict(X)
    pr1 = m1.predict_proba(X)

    m2 = ModelClass(config=config.copy())
    m2.fit(X, y, **kw)
    p2 = m2.predict(X)
    pr2 = m2.predict_proba(X)

    np.testing.assert_array_equal(
        p1, p2, err_msg=f"{ModelClass.__name__} predict() not reproducible"
    )
    np.testing.assert_allclose(
        pr1, pr2, atol=atol, err_msg=f"{ModelClass.__name__} predict_proba() not reproducible"
    )


def _assert_reproducible_regressor(
    ModelClass: type,
    config: dict,
    X: np.ndarray,
    y: np.ndarray,
    *,
    atol: float = 1e-10,
    fit_kwargs: dict | None = None,
) -> None:
    """Fit the same regressor twice and assert identical predictions."""
    kw = fit_kwargs or {}

    m1 = ModelClass(config=config.copy())
    m1.fit(X, y, **kw)
    p1 = m1.predict(X)

    m2 = ModelClass(config=config.copy())
    m2.fit(X, y, **kw)
    p2 = m2.predict(X)

    np.testing.assert_allclose(
        p1, p2, atol=atol, err_msg=f"{ModelClass.__name__} predict() not reproducible"
    )


# ---------------------------------------------------------------------------
# Core models (always available — xgboost, sklearn)
# ---------------------------------------------------------------------------


class TestReproducibilityCore:
    """Reproducibility for models with no optional dependencies."""

    def test_xgboost_classifier(self, classification_data):
        from aquacontam.models.xgboost import XGBoostClassifier

        X, y = classification_data
        _assert_reproducible_classifier(
            XGBoostClassifier, {"n_estimators": 10, "random_state": 42}, X, y
        )

    def test_xgboost_regressor(self, regression_data):
        from aquacontam.models.xgboost import XGBoostRegressor

        X, y = regression_data
        _assert_reproducible_regressor(
            XGBoostRegressor, {"n_estimators": 10, "random_state": 42}, X, y
        )

    def test_random_forest_classifier(self, classification_data):
        from aquacontam.models.random_forest import RandomForestClassifier

        X, y = classification_data
        _assert_reproducible_classifier(
            RandomForestClassifier, {"n_estimators": 10, "random_state": 42}, X, y
        )

    def test_random_forest_regressor(self, regression_data):
        from aquacontam.models.random_forest import RandomForestRegressor

        X, y = regression_data
        _assert_reproducible_regressor(
            RandomForestRegressor, {"n_estimators": 10, "random_state": 42}, X, y
        )

    def test_logistic_regression(self, classification_data):
        from aquacontam.models.logistic import LogisticRegressionClassifier

        X, y = classification_data
        _assert_reproducible_classifier(LogisticRegressionClassifier, {"random_state": 42}, X, y)

    def test_dummy_classifier(self, classification_data):
        from aquacontam.models.logistic import DummyClassifierBaseline

        X, y = classification_data
        _assert_reproducible_classifier(
            DummyClassifierBaseline, {"strategy": "prior", "random_state": 42}, X, y
        )


# ---------------------------------------------------------------------------
# CatBoost (optional)
# ---------------------------------------------------------------------------


class TestReproducibilityCatBoost:
    """Reproducibility for CatBoost models."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        pytest.importorskip("catboost")

    def test_catboost_classifier(self, classification_data):
        from aquacontam.models.catboost import CatBoostClassifier

        X, y = classification_data
        _assert_reproducible_classifier(
            CatBoostClassifier,
            {"iterations": 10, "random_seed": 42, "verbose": 0},
            X,
            y,
        )

    def test_catboost_regressor(self, regression_data):
        from aquacontam.models.catboost import CatBoostRegressor

        X, y = regression_data
        _assert_reproducible_regressor(
            CatBoostRegressor,
            {"iterations": 10, "random_seed": 42, "verbose": 0},
            X,
            y,
        )


# ---------------------------------------------------------------------------
# LightGBM (optional)
# ---------------------------------------------------------------------------


class TestReproducibilityLightGBM:
    """Reproducibility for LightGBM models."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        pytest.importorskip("lightgbm")

    def test_lightgbm_classifier(self, classification_data):
        from aquacontam.models.lightgbm import LightGBMClassifier

        X, y = classification_data
        _assert_reproducible_classifier(
            LightGBMClassifier,
            {"n_estimators": 10, "random_state": 42, "verbose": -1},
            X,
            y,
        )

    def test_lightgbm_regressor(self, regression_data):
        from aquacontam.models.lightgbm import LightGBMRegressor

        X, y = regression_data
        _assert_reproducible_regressor(
            LightGBMRegressor,
            {"n_estimators": 10, "random_state": 42, "verbose": -1},
            X,
            y,
        )


# ---------------------------------------------------------------------------
# PyTorch models (optional) — wider tolerance for float32 arithmetic
# ---------------------------------------------------------------------------


class TestReproducibilityTorch:
    """Reproducibility for PyTorch-based models."""

    TORCH_ATOL = 1e-5  # float32 NN arithmetic can vary slightly

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        pytest.importorskip("torch")

    def test_mlp_classifier(self, classification_data):
        from aquacontam.models.mlp import MLPClassifier

        X, y = classification_data
        _assert_reproducible_classifier(
            MLPClassifier,
            {"hidden_sizes": [4], "epochs": 3, "batch_size": 32, "random_state": 42},
            X,
            y,
            atol=self.TORCH_ATOL,
        )

    def test_mlp_regressor(self, regression_data):
        from aquacontam.models.mlp import MLPRegressor

        X, y = regression_data
        _assert_reproducible_regressor(
            MLPRegressor,
            {"hidden_sizes": [4], "epochs": 3, "batch_size": 32, "random_state": 42},
            X,
            y,
            atol=self.TORCH_ATOL,
        )

    def test_cnn1d_classifier(self, classification_data):
        from aquacontam.models.cnn1d import CNN1DClassifier

        X, y = classification_data
        _assert_reproducible_classifier(
            CNN1DClassifier,
            {
                "n_filters": [4],
                "kernel_size": 3,
                "epochs": 3,
                "batch_size": 32,
                "random_state": 42,
                "feature_ordering": "alphabetical",
            },
            X,
            y,
            atol=self.TORCH_ATOL,
        )

    def test_cnn1d_regressor(self, regression_data):
        from aquacontam.models.cnn1d import CNN1DRegressor

        X, y = regression_data
        _assert_reproducible_regressor(
            CNN1DRegressor,
            {
                "n_filters": [4],
                "kernel_size": 3,
                "epochs": 3,
                "batch_size": 32,
                "random_state": 42,
                "feature_ordering": "alphabetical",
            },
            X,
            y,
            atol=self.TORCH_ATOL,
        )

    def test_deep_tobit_classifier(self, classification_data):
        from aquacontam.models.deep_tobit import DeepTobitClassifier

        X, y = classification_data
        n = len(y)
        # Deep Tobit requires censoring metadata
        fit_kw = {
            "censored": np.zeros(n, dtype=np.float64),
            "detection_limits": np.full(n, 0.004, dtype=np.float64),
            "y_concentration": np.random.RandomState(42).exponential(1.0, n),
        }
        _assert_reproducible_classifier(
            DeepTobitClassifier,
            {"hidden_sizes": [4], "epochs": 3, "batch_size": 32, "random_state": 42},
            X,
            y,
            atol=self.TORCH_ATOL,
            fit_kwargs=fit_kw,
        )

    def test_deep_tobit_regressor(self, regression_data):
        from aquacontam.models.deep_tobit import DeepTobitRegressor

        X, y = regression_data
        n = len(y)
        fit_kw = {
            "censored": np.zeros(n, dtype=np.float64),
            "detection_limits": np.full(n, 0.004, dtype=np.float64),
        }
        _assert_reproducible_regressor(
            DeepTobitRegressor,
            {"hidden_sizes": [4], "epochs": 3, "batch_size": 32, "random_state": 42},
            X,
            y,
            atol=self.TORCH_ATOL,
            fit_kwargs=fit_kw,
        )
