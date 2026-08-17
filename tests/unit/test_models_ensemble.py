"""Unit tests for ensemble models (Voting + Stacking)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

sklearn = pytest.importorskip("sklearn")

from aquacontam.models.ensemble import (  # noqa: E402
    StackingEnsembleClassifier,
    VotingEnsembleClassifier,
    _build_base_estimators,
)


@pytest.fixture()
def classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Small synthetic binary classification dataset."""
    rng = np.random.RandomState(42)
    n = 100
    X = pd.DataFrame(
        {
            "f1": rng.randn(n),
            "f2": rng.randn(n),
            "f3": rng.randn(n),
        }
    )
    y = pd.Series((rng.randn(n) > 0).astype(int), name="target")
    return X, y


class TestBuildBaseEstimators:
    """Tests for _build_base_estimators helper."""

    def test_random_forest_always_available(self) -> None:
        estimators = _build_base_estimators(["random_forest"], seed=42)
        assert len(estimators) == 1
        assert estimators[0][0] == "random_forest"

    def test_xgboost_available(self) -> None:
        xgb = pytest.importorskip("xgboost")  # noqa: F841
        estimators = _build_base_estimators(["xgboost"], seed=42)
        assert len(estimators) == 1
        assert estimators[0][0] == "xgboost"

    def test_unknown_model_skipped(self) -> None:
        estimators = _build_base_estimators(["unknown_model"], seed=42)
        assert len(estimators) == 0

    def test_subset_models(self) -> None:
        estimators = _build_base_estimators(["random_forest"], seed=42)
        assert len(estimators) == 1


class TestVotingEnsembleClassifier:
    """Tests for VotingEnsembleClassifier."""

    def test_name(self) -> None:
        model = VotingEnsembleClassifier(config={})
        assert model.name == "voting_ensemble"

    def test_fit_predict(
        self,
        classification_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = classification_data
        xgb = pytest.importorskip("xgboost")  # noqa: F841
        model = VotingEnsembleClassifier(
            config={
                "base_models": ["xgboost", "random_forest"],
                "voting": "soft",
                "random_state": 42,
            }
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba(
        self,
        classification_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = classification_data
        xgb = pytest.importorskip("xgboost")  # noqa: F841
        model = VotingEnsembleClassifier(
            config={
                "base_models": ["xgboost", "random_forest"],
                "voting": "soft",
                "random_state": 42,
            }
        )
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (len(X), 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_evaluate(
        self,
        classification_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = classification_data
        xgb = pytest.importorskip("xgboost")  # noqa: F841
        model = VotingEnsembleClassifier(
            config={
                "base_models": ["xgboost", "random_forest"],
                "voting": "soft",
                "random_state": 42,
            }
        )
        model.fit(X, y)
        metrics = model.evaluate(X, y, metrics=["accuracy", "auroc"])
        assert "accuracy" in metrics
        assert "auroc" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_too_few_models_raises(self) -> None:
        model = VotingEnsembleClassifier(
            config={"base_models": ["random_forest"], "random_state": 42}
        )
        X = pd.DataFrame({"f1": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        with pytest.raises(RuntimeError, match="requires >= 2"):
            model.fit(X, y)

    def test_not_fitted_raises(self) -> None:
        model = VotingEnsembleClassifier(config={})
        with pytest.raises(RuntimeError, match="has not been fitted"):
            model.predict(pd.DataFrame({"f1": [1]}))

    def test_feature_importances(
        self,
        classification_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = classification_data
        xgb = pytest.importorskip("xgboost")  # noqa: F841
        model = VotingEnsembleClassifier(
            config={
                "base_models": ["xgboost", "random_forest"],
                "voting": "soft",
                "random_state": 42,
            }
        )
        model.fit(X, y)
        importances = model.feature_importances()
        assert isinstance(importances, pd.Series)
        assert len(importances) == 3


class TestStackingEnsembleClassifier:
    """Tests for StackingEnsembleClassifier."""

    def test_name(self) -> None:
        model = StackingEnsembleClassifier(config={})
        assert model.name == "stacking_ensemble"

    def test_fit_predict(
        self,
        classification_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = classification_data
        xgb = pytest.importorskip("xgboost")  # noqa: F841
        model = StackingEnsembleClassifier(
            config={
                "base_models": ["xgboost", "random_forest"],
                "cv": 3,
                "random_state": 42,
            }
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba(
        self,
        classification_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = classification_data
        xgb = pytest.importorskip("xgboost")  # noqa: F841
        model = StackingEnsembleClassifier(
            config={
                "base_models": ["xgboost", "random_forest"],
                "cv": 3,
                "random_state": 42,
            }
        )
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (len(X), 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_evaluate(
        self,
        classification_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = classification_data
        xgb = pytest.importorskip("xgboost")  # noqa: F841
        model = StackingEnsembleClassifier(
            config={
                "base_models": ["xgboost", "random_forest"],
                "cv": 3,
                "random_state": 42,
            }
        )
        model.fit(X, y)
        metrics = model.evaluate(X, y, metrics=["accuracy"])
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_too_few_models_raises(self) -> None:
        model = StackingEnsembleClassifier(
            config={"base_models": ["random_forest"], "random_state": 42}
        )
        X = pd.DataFrame({"f1": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        with pytest.raises(RuntimeError, match="requires >= 2"):
            model.fit(X, y)

    def test_feature_importances(
        self,
        classification_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        X, y = classification_data
        xgb = pytest.importorskip("xgboost")  # noqa: F841
        model = StackingEnsembleClassifier(
            config={
                "base_models": ["xgboost", "random_forest"],
                "cv": 3,
                "random_state": 42,
            }
        )
        model.fit(X, y)
        importances = model.feature_importances()
        assert isinstance(importances, pd.Series)
        assert len(importances) == 3
