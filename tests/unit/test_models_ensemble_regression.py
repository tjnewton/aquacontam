"""Tests for AveragingEnsembleRegressor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquacontam.models.base import BaseModel
from aquacontam.models.ensemble import AveragingEnsembleRegressor

# ---------------------------------------------------------------------------
# Stub sub-models for isolated testing
# ---------------------------------------------------------------------------


class _StubRegressor(BaseModel):
    """Minimal regressor that predicts a constant offset from the mean."""

    def __init__(self, offset: float = 0.0) -> None:
        super().__init__({})
        self._offset = offset

    @property
    def name(self) -> str:
        return f"stub_{self._offset}"

    def fit(self, X_train, y_train, **kwargs):
        self._mean = float(np.mean(y_train))
        self._n_features = X_train.shape[1]
        self._model = True

    def predict(self, X, **kwargs):
        return np.full(X.shape[0], self._mean + self._offset)

    def predict_proba(self, X, **kwargs):
        return np.full((X.shape[0], 2), 0.5)

    def feature_importances(self) -> pd.Series:
        imp = np.random.default_rng(42).random(self._n_features)
        imp /= imp.sum()
        return pd.Series(imp, index=[f"f{i}" for i in range(self._n_features)])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_data():
    """Zero-inflated synthetic regression data."""
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)])
    # 70% zeros, 30% positive values
    y = np.where(rng.random(n) < 0.7, 0.0, rng.exponential(0.5, n))
    return X, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAveragingEnsembleRegressor:
    def test_fit_predict(self, synthetic_data):
        X, y = synthetic_data
        model = AveragingEnsembleRegressor(
            models=[_StubRegressor(0.0), _StubRegressor(1.0)],
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(np.isfinite(preds))

    def test_prediction_is_average(self, synthetic_data):
        X, y = synthetic_data
        m1 = _StubRegressor(0.0)
        m2 = _StubRegressor(2.0)
        ensemble = AveragingEnsembleRegressor(models=[m1, m2])
        ensemble.fit(X, y)

        p1 = m1.predict(X)
        p2 = m2.predict(X)
        expected = (p1 + p2) / 2.0
        actual = ensemble.predict(X)
        np.testing.assert_allclose(actual, expected)

    def test_weighted_prediction(self, synthetic_data):
        X, y = synthetic_data
        m1 = _StubRegressor(0.0)
        m2 = _StubRegressor(4.0)
        ensemble = AveragingEnsembleRegressor(
            models=[m1, m2],
            weights=[0.75, 0.25],
        )
        ensemble.fit(X, y)

        p1 = m1.predict(X)
        p2 = m2.predict(X)
        expected = 0.75 * p1 + 0.25 * p2
        actual = ensemble.predict(X)
        np.testing.assert_allclose(actual, expected)

    def test_requires_censoring_metadata(self):
        model = AveragingEnsembleRegressor(models=[_StubRegressor()])
        assert model.requires_censoring_metadata is True

    def test_name(self):
        model = AveragingEnsembleRegressor(models=[_StubRegressor()])
        assert model.name == "hurdle_aft_ensemble"

    def test_feature_importances(self, synthetic_data):
        X, y = synthetic_data
        model = AveragingEnsembleRegressor(
            models=[_StubRegressor(0.0), _StubRegressor(1.0)],
        )
        model.fit(X, y)
        imp = model.feature_importances()
        assert isinstance(imp, pd.Series)
        assert len(imp) == X.shape[1]
        assert imp.name == "importance"

    def test_fit_raises_without_models(self, synthetic_data):
        X, y = synthetic_data
        model = AveragingEnsembleRegressor(models=[])
        with pytest.raises(RuntimeError, match="at least 1 sub-model"):
            model.fit(X, y)

    def test_predict_before_fit_raises(self, synthetic_data):
        X, _y = synthetic_data
        model = AveragingEnsembleRegressor(models=[_StubRegressor()])
        with pytest.raises(RuntimeError):
            model.predict(X)
