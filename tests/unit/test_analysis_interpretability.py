"""Tests for SHAP and permutation importance analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")

from aquacontam.analysis.interpretability import (
    _infer_explainer_method,
    _metric_to_scorer,
    _SklearnAdapter,
    compute_shap_values,
    force_plot_data,
    permutation_importance,
    summary_plot_data,
)
from aquacontam.models.xgboost import XGBoostClassifier


class TestInferExplainerMethod:
    """Tests for _infer_explainer_method."""

    def test_xgboost_detected_as_tree(self, classification_data):
        X, y = classification_data
        model = XGBoostClassifier(config={"n_estimators": 5, "random_state": 42})
        model.fit(X, y)
        assert _infer_explainer_method(model._model) == "tree"

    def test_random_forest_detected_as_tree(self):
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        assert _infer_explainer_method(clf) == "tree"

    def test_logistic_regression_detected_as_linear(self):
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression()
        assert _infer_explainer_method(clf) == "linear"

    def test_unknown_model_falls_back_to_kernel(self):
        class CustomModel:
            pass

        assert _infer_explainer_method(CustomModel()) == "kernel"


class TestSHAP:
    """Tests for compute_shap_values and related plot data functions."""

    def _fit_model(self, classification_data):
        X, y = classification_data
        model = XGBoostClassifier(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        return model, X, y

    def test_compute_shap_tree(self, classification_data):
        model, X, _y = self._fit_model(classification_data)
        sv = compute_shap_values(model, X, method="tree")
        assert isinstance(sv, pd.DataFrame)
        assert sv.shape == X.shape

    def test_compute_shap_auto(self, classification_data):
        model, X, _y = self._fit_model(classification_data)
        sv = compute_shap_values(model, X, method="auto")
        assert sv.shape == X.shape

    def test_shap_preserves_index_and_columns(self, classification_data):
        model, X, _y = self._fit_model(classification_data)
        sv = compute_shap_values(model, X)
        assert list(sv.columns) == list(X.columns)
        assert list(sv.index) == list(X.index)

    def test_invalid_method_raises(self, classification_data):
        model, X, _y = self._fit_model(classification_data)
        with pytest.raises(ValueError, match="Unknown SHAP method"):
            compute_shap_values(model, X, method="bogus")

    def test_unfitted_model_raises(self):
        model = XGBoostClassifier(config={})
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((10, 3)), columns=["a", "b", "c"])
        with pytest.raises(RuntimeError, match="has not been fitted"):
            compute_shap_values(model, X)

    def test_dummy_classifier_warns(self, classification_data, caplog):
        """SHAP on DummyClassifier should log a warning about uninformative values."""
        from aquacontam.models.logistic import DummyClassifierBaseline

        X, y = classification_data
        model = DummyClassifierBaseline(config={})
        model.fit(X, y)
        with caplog.at_level("WARNING"):
            sv = compute_shap_values(model, X)
        assert "trivial model" in caplog.text or "uninformative" in caplog.text
        # SHAP values should still be returned (all near-zero)
        assert sv.shape == X.shape

    def test_summary_plot_data(self, classification_data):
        model, X, _y = self._fit_model(classification_data)
        sv = compute_shap_values(model, X)
        result = summary_plot_data(sv, X, top_n=3)
        assert "feature_names" in result
        assert len(result["feature_names"]) == 3
        assert "mean_abs_shap" in result
        assert len(result["mean_abs_shap"]) == 3
        assert "shap_values" in result
        assert len(result["shap_values"]) == 3
        assert "feature_values" in result
        assert len(result["feature_values"]) == 3

    def test_summary_plot_data_sorted_descending(self, classification_data):
        model, X, _y = self._fit_model(classification_data)
        sv = compute_shap_values(model, X)
        result = summary_plot_data(sv, X, top_n=5)
        shap_vals = result["mean_abs_shap"]
        assert shap_vals == sorted(shap_vals, reverse=True)

    def test_summary_plot_top_n_capped(self, classification_data):
        """top_n larger than number of features returns all features."""
        model, X, _y = self._fit_model(classification_data)
        sv = compute_shap_values(model, X)
        result = summary_plot_data(sv, X, top_n=100)
        assert len(result["feature_names"]) == X.shape[1]

    def test_force_plot_data(self, classification_data):
        model, X, _y = self._fit_model(classification_data)
        sv = compute_shap_values(model, X)
        result = force_plot_data(sv, X, idx=0)
        assert "base_value" in result
        assert isinstance(result["base_value"], float)
        assert "shap_values" in result
        assert isinstance(result["shap_values"], dict)
        assert set(result["shap_values"].keys()) == set(X.columns)
        assert "feature_values" in result
        assert isinstance(result["feature_values"], dict)
        assert set(result["feature_values"].keys()) == set(X.columns)


class TestSklearnAdapter:
    """Tests for the _SklearnAdapter wrapper."""

    def test_predict(self, classification_data):
        X, y = classification_data
        model = XGBoostClassifier(config={"n_estimators": 5, "random_state": 42})
        model.fit(X, y)
        adapter = _SklearnAdapter(model)
        preds = adapter.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(X)

    def test_predict_proba(self, classification_data):
        X, y = classification_data
        model = XGBoostClassifier(config={"n_estimators": 5, "random_state": 42})
        model.fit(X, y)
        adapter = _SklearnAdapter(model)
        probas = adapter.predict_proba(X)
        assert probas.shape == (len(X), 2)

    def test_classes(self, classification_data):
        X, y = classification_data
        model = XGBoostClassifier(config={"n_estimators": 5, "random_state": 42})
        model.fit(X, y)
        adapter = _SklearnAdapter(model)
        assert list(adapter.classes_) == [0, 1]

    def test_is_fitted(self, classification_data):
        X, y = classification_data
        model = XGBoostClassifier(config={"n_estimators": 5, "random_state": 42})
        model.fit(X, y)
        adapter = _SklearnAdapter(model)
        assert adapter.__sklearn_is_fitted__() is True


class TestMetricToScorer:
    """Tests for _metric_to_scorer."""

    def test_auprc(self):
        scorer = _metric_to_scorer("auprc")
        assert callable(scorer)

    def test_auroc(self):
        scorer = _metric_to_scorer("auroc")
        assert callable(scorer)

    def test_accuracy(self):
        assert _metric_to_scorer("accuracy") == "accuracy"

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            _metric_to_scorer("f1")


class TestPermutationImportance:
    """Tests for permutation_importance."""

    def test_permutation_importance(self, classification_data):
        X, y = classification_data
        model = XGBoostClassifier(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        result = permutation_importance(model, X, y, metric="accuracy", n_repeats=3)
        assert isinstance(result, pd.DataFrame)
        assert "importance_mean" in result.columns
        assert "importance_std" in result.columns
        assert len(result) == X.shape[1]

    def test_permutation_importance_index_matches_features(self, classification_data):
        X, y = classification_data
        model = XGBoostClassifier(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        result = permutation_importance(model, X, y, metric="accuracy", n_repeats=3)
        assert list(result.index) == list(X.columns)

    def test_unfitted_model_raises(self):
        model = XGBoostClassifier(config={})
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((10, 3)), columns=["a", "b", "c"])
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        with pytest.raises(RuntimeError, match="has not been fitted"):
            permutation_importance(model, X, y)
