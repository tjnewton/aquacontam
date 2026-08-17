"""Tests for T3 — multi-PFAS profile prediction (multilabel classification)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquacontam.benchmark.metrics import compute_multilabel_metrics
from aquacontam.benchmark.multilabel import (
    EvalSetMultiOutputClassifier,
    _build_inner_estimator,
    _supports_eval_set,
    aggregate_to_multilabel_target,
    run_t3,
)
from aquacontam.benchmark.registry import TaskResult, get_task
from aquacontam.models.random_forest import RandomForestClassifier


class TestMultilabelMetrics:
    """Tests for compute_multilabel_metrics."""

    def test_perfect_predictions(self) -> None:
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
        y_pred = y_true.copy()
        y_prob = y_true.astype(float)
        result = compute_multilabel_metrics(y_true, y_pred, y_prob)
        assert result["subset_accuracy"] == 1.0
        assert result["hamming_loss"] == 0.0
        assert result["macro_f1"] == 1.0
        assert result["micro_f1"] == 1.0

    def test_no_probs_returns_nan_for_auc(self) -> None:
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = y_true.copy()
        result = compute_multilabel_metrics(y_true, y_pred, y_prob=None)
        assert np.isnan(result["macro_auroc"])
        assert np.isnan(result["micro_auroc"])
        assert np.isnan(result["macro_auprc"])

    def test_single_class_label_handled(self) -> None:
        # Second label is always 0 — AUROC undefined for it
        y_true = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])
        y_pred = y_true.copy()
        y_prob = np.array([[0.9, 0.1], [0.2, 0.1], [0.8, 0.05], [0.3, 0.05]])
        result = compute_multilabel_metrics(y_true, y_pred, y_prob, label_names=["A", "B"])
        # Per-label: B should be NaN (single class)
        assert np.isnan(result["auroc_B"])
        # A should be computable
        assert not np.isnan(result["auroc_A"])

    def test_hamming_range(self) -> None:
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_pred = np.array([[0, 1, 0], [1, 0, 1]])  # All wrong
        result = compute_multilabel_metrics(y_true, y_pred)
        assert 0.0 <= result["hamming_loss"] <= 1.0
        assert result["hamming_loss"] == 1.0

    def test_per_label_keys(self) -> None:
        y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        y_pred = y_true.copy()
        y_prob = y_true.astype(float)
        result = compute_multilabel_metrics(y_true, y_pred, y_prob, label_names=["PFOS", "PFOA"])
        assert "auroc_PFOS" in result
        assert "auprc_PFOS" in result
        assert "auroc_PFOA" in result
        assert "auprc_PFOA" in result

    def test_all_zeros_predictions(self) -> None:
        y_true = np.array([[1, 1], [0, 1], [1, 0]])
        y_pred = np.zeros_like(y_true)
        result = compute_multilabel_metrics(y_true, y_pred)
        assert result["subset_accuracy"] < 1.0
        assert "hamming_loss" in result

    def test_custom_metrics_subset(self) -> None:
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = y_true.copy()
        result = compute_multilabel_metrics(
            y_true, y_pred, metrics=["subset_accuracy", "hamming_loss"]
        )
        assert set(result.keys()) == {"subset_accuracy", "hamming_loss"}


class TestAggregateToMultilabelTarget:
    """Tests for aggregate_to_multilabel_target."""

    def test_basic_aggregation(self, synthetic_multi_pfas_df: pd.DataFrame) -> None:
        analytes = ("PFOS", "PFOA", "PFBS")
        result = aggregate_to_multilabel_target(synthetic_multi_pfas_df, analytes)
        assert "detected_PFOS" in result.columns
        assert "detected_PFOA" in result.columns
        assert "detected_PFBS" in result.columns
        assert result.index.name == "pwsid"
        assert len(result) > 0

    def test_missing_analyte_zero(self) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["A", "A", "B", "B"],
                "analyte": ["PFOS", "PFOS", "PFOS", "PFOS"],
                "censored": [False, True, True, True],
            }
        )
        result = aggregate_to_multilabel_target(df, ("PFOS", "PFOA"))
        # System B has no PFOA samples → should be 0
        assert result.loc["B", "detected_PFOA"] == 0

    def test_detection_logic(self) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["A", "A", "A"],
                "analyte": ["PFOS", "PFOS", "PFOS"],
                "censored": [True, True, False],  # One detect → detected=1
            }
        )
        result = aggregate_to_multilabel_target(df, ("PFOS",))
        assert result.loc["A", "detected_PFOS"] == 1

    def test_all_censored(self) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["A", "A"],
                "analyte": ["PFOS", "PFOS"],
                "censored": [True, True],
            }
        )
        result = aggregate_to_multilabel_target(df, ("PFOS",))
        assert result.loc["A", "detected_PFOS"] == 0

    def test_empty_input(self) -> None:
        df = pd.DataFrame(columns=["pwsid", "analyte", "censored"])
        result = aggregate_to_multilabel_target(df, ("PFOS",))
        assert len(result) == 0
        assert "detected_PFOS" in result.columns

    def test_empty_analytes_raises(self) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["A"],
                "analyte": ["PFOS"],
                "censored": [False],
            }
        )
        with pytest.raises(ValueError, match="non-empty"):
            aggregate_to_multilabel_target(df, ())

    def test_detection_rate_column(self) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["A", "A", "A", "A"],
                "analyte": ["PFOS", "PFOS", "PFOS", "PFOS"],
                "censored": [True, False, True, False],
            }
        )
        result = aggregate_to_multilabel_target(df, ("PFOS",))
        assert result.loc["A", "detection_rate_PFOS"] == pytest.approx(0.5)

    def test_n_samples_column(self, synthetic_multi_pfas_df: pd.DataFrame) -> None:
        result = aggregate_to_multilabel_target(synthetic_multi_pfas_df, ("PFOS", "PFOA"))
        # Each system should have samples
        assert (result["n_samples"] > 0).all()


class TestT3Task:
    """Tests for T3 task registration and execution."""

    def test_t3_registered(self) -> None:
        info, _ = get_task("T3")
        assert info.task_type == "multilabel_classification"
        assert info.primary_metric == "macro_auprc"

    def test_t3_runs(self, synthetic_multi_pfas_df: pd.DataFrame) -> None:
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t3(
            model=model,
            data=synthetic_multi_pfas_df,
            analytes=("PFOS", "PFOA"),
        )
        assert isinstance(result, TaskResult)
        assert result.task_name == "T3"

    def test_t3_has_multilabel_metrics(self, synthetic_multi_pfas_df: pd.DataFrame) -> None:
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t3(
            model=model,
            data=synthetic_multi_pfas_df,
            analytes=("PFOS", "PFOA"),
        )
        assert result.metrics, "Expected non-empty metrics"
        assert "macro_f1" in result.metrics or "subset_accuracy" in result.metrics

    def test_t3_has_split_metrics(self, synthetic_multi_pfas_df: pd.DataFrame) -> None:
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t3(
            model=model,
            data=synthetic_multi_pfas_df,
            analytes=("PFOS", "PFOA"),
        )
        assert "train" in result.split_metrics

    def test_t3_per_label_breakdown(self, synthetic_multi_pfas_df: pd.DataFrame) -> None:
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t3(
            model=model,
            data=synthetic_multi_pfas_df,
            analytes=("PFOS", "PFOA"),
        )
        # Per-label keys in train metrics
        train_m = result.split_metrics.get("train", {})
        if train_m:
            assert "auroc_PFOS" in train_m or "auprc_PFOS" in train_m

    def test_t3_custom_analytes(self, synthetic_multi_pfas_df: pd.DataFrame) -> None:
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t3(
            model=model,
            data=synthetic_multi_pfas_df,
            analytes=("PFBS", "HFPO-DA"),
        )
        assert result.metadata["analytes"] == ["PFBS", "HFPO-DA"]

    def test_t3_metadata(self, synthetic_multi_pfas_df: pd.DataFrame) -> None:
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t3(
            model=model,
            data=synthetic_multi_pfas_df,
            analytes=("PFOS", "PFOA"),
        )
        assert "n_train" in result.metadata
        assert "n_labels" in result.metadata
        assert result.metadata["n_labels"] == 2

    def test_t3_nonexistent_analytes(self, synthetic_multi_pfas_df: pd.DataFrame) -> None:
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t3(
            model=model,
            data=synthetic_multi_pfas_df,
            analytes=("nonexistent",),
        )
        # Should still run but may have empty/degenerate results
        assert isinstance(result, TaskResult)


class TestBuildInnerEstimator:
    """Tests for _build_inner_estimator model handler dispatch."""

    def _make_stub_model(self, name: str) -> RandomForestClassifier:
        """Create a stub model with a given name for dispatch testing."""

        class StubModel(RandomForestClassifier):
            @property  # type: ignore[override]
            def name(self) -> str:
                return name

        return StubModel(config={"n_estimators": 10, "random_state": 42})

    def test_dummy_returns_dummy_classifier(self) -> None:
        from sklearn.dummy import DummyClassifier

        model = self._make_stub_model("dummy_classifier")
        estimator = _build_inner_estimator(model)
        assert isinstance(estimator, DummyClassifier)

    def test_xgboost_returns_xgb(self) -> None:
        xgb = pytest.importorskip("xgboost")
        model = self._make_stub_model("xgboost_classifier")
        estimator = _build_inner_estimator(model)
        assert isinstance(estimator, xgb.XGBClassifier)

    def test_lightgbm_returns_lgbm(self) -> None:
        lgb = pytest.importorskip("lightgbm")
        model = self._make_stub_model("lightgbm_classifier")
        estimator = _build_inner_estimator(model)
        assert isinstance(estimator, lgb.LGBMClassifier)

    def test_catboost_returns_catboost(self) -> None:
        cb = pytest.importorskip("catboost")
        model = self._make_stub_model("catboost_classifier")
        estimator = _build_inner_estimator(model)
        assert isinstance(estimator, cb.CatBoostClassifier)

    def test_logistic_returns_logistic_regression(self) -> None:
        from sklearn.linear_model import LogisticRegression

        model = self._make_stub_model("logistic_regression")
        estimator = _build_inner_estimator(model)
        assert isinstance(estimator, LogisticRegression)

    def test_gnn_returns_gradient_boosting_proxy(self) -> None:
        from sklearn.ensemble import GradientBoostingClassifier

        model = self._make_stub_model("gnn_gcn_classifier")
        estimator = _build_inner_estimator(model)
        assert isinstance(estimator, GradientBoostingClassifier)

    def test_rf_returns_rf(self) -> None:
        from sklearn.ensemble import RandomForestClassifier as SkRF

        model = self._make_stub_model("random_forest_classifier")
        estimator = _build_inner_estimator(model)
        assert isinstance(estimator, SkRF)

    def test_mlp_returns_sklearn_mlp(self) -> None:
        from sklearn.neural_network import MLPClassifier

        model = self._make_stub_model("mlp_classifier")
        estimator = _build_inner_estimator(model)
        assert isinstance(estimator, MLPClassifier)

    def test_cnn_returns_sklearn_mlp_proxy(self) -> None:
        from sklearn.neural_network import MLPClassifier

        model = self._make_stub_model("cnn1d_classifier")
        estimator = _build_inner_estimator(model)
        assert isinstance(estimator, MLPClassifier)

    def test_lightgbm_no_early_stopping_rounds_in_params(self) -> None:
        """LightGBM estimator must not have early_stopping_rounds (popped for wrapper)."""
        pytest.importorskip("lightgbm")
        model = self._make_stub_model("lightgbm_classifier")
        model.config["early_stopping_rounds"] = 50
        estimator = _build_inner_estimator(model)
        params = estimator.get_params()
        assert params.get("early_stopping_rounds") is None

    def test_catboost_keeps_early_stopping_rounds(self) -> None:
        """CatBoost estimator keeps early_stopping_rounds (used by CatBoost natively)."""
        pytest.importorskip("catboost")
        model = self._make_stub_model("catboost_classifier")
        model.config["early_stopping_rounds"] = 50
        estimator = _build_inner_estimator(model)
        init_params = estimator.get_params()
        assert init_params.get("early_stopping_rounds") == 50

    def test_all_handlers_produce_distinct_estimator_types(self) -> None:
        """Verify no two unrelated model types map to the same estimator."""
        names = [
            "dummy_classifier",
            "xgboost_classifier",
            "random_forest_classifier",
            "logistic_regression",
        ]
        # lightgbm_classifier and gnn_gcn_classifier are tested in dedicated
        # tests with proper importorskip guards (optional deps).
        estimator_types = set()
        for n in names:
            est = _build_inner_estimator(self._make_stub_model(n))
            estimator_types.add(type(est).__name__)
        assert len(estimator_types) == len(names)


class TestEvalSetMultiOutputClassifier:
    """Tests for the EvalSetMultiOutputClassifier wrapper."""

    def test_supports_eval_set_xgboost(self) -> None:
        xgb = pytest.importorskip("xgboost")
        est = xgb.XGBClassifier(n_estimators=10)
        assert _supports_eval_set(est) is True

    def test_supports_eval_set_lightgbm(self) -> None:
        lgb = pytest.importorskip("lightgbm")
        est = lgb.LGBMClassifier(n_estimators=10, verbose=-1)
        assert _supports_eval_set(est) is True

    def test_supports_eval_set_catboost(self) -> None:
        cb = pytest.importorskip("catboost")
        est = cb.CatBoostClassifier(iterations=10, verbose=0)
        assert _supports_eval_set(est) is True

    def test_supports_eval_set_rf_false(self) -> None:
        from sklearn.ensemble import RandomForestClassifier as SkRF

        est = SkRF(n_estimators=10)
        assert _supports_eval_set(est) is False

    def test_supports_eval_set_dummy_false(self) -> None:
        from sklearn.dummy import DummyClassifier

        est = DummyClassifier()
        assert _supports_eval_set(est) is False

    def test_eval_set_forwarded_to_xgboost(self) -> None:
        xgb = pytest.importorskip("xgboost")
        rng = np.random.RandomState(42)
        n = 100
        X = rng.rand(n, 5)
        Y = rng.randint(0, 2, (n, 2))
        X_val = rng.rand(30, 5)
        Y_val = rng.randint(0, 2, (30, 2))

        inner = xgb.XGBClassifier(n_estimators=50, early_stopping_rounds=5, random_state=42)
        clf = EvalSetMultiOutputClassifier(inner, early_stopping_rounds=5)
        clf.fit(X, Y, eval_set=(X_val, Y_val))

        # Each sub-estimator should have best_iteration set (early stopping used)
        for est in clf.estimators_:
            assert hasattr(est, "best_iteration")
            # best_iteration should be <= n_estimators (early stopping triggered or ran all)
            assert est.best_iteration <= 50

    def test_no_eval_set_for_sklearn_estimator(self) -> None:
        """RF trains without error when eval_set is given (ignored for sklearn)."""
        from sklearn.ensemble import RandomForestClassifier as SkRF

        rng = np.random.RandomState(42)
        X = rng.rand(50, 3)
        Y = rng.randint(0, 2, (50, 2))
        X_val = rng.rand(15, 3)
        Y_val = rng.randint(0, 2, (15, 2))

        clf = EvalSetMultiOutputClassifier(SkRF(n_estimators=10, random_state=42))
        clf.fit(X, Y, eval_set=(X_val, Y_val))
        preds = clf.predict(X)
        assert preds.shape == (50, 2)

    def test_no_eval_set_kwarg(self) -> None:
        """eval_set=None falls back to plain fit."""
        from sklearn.ensemble import RandomForestClassifier as SkRF

        rng = np.random.RandomState(42)
        X = rng.rand(50, 3)
        Y = rng.randint(0, 2, (50, 2))

        clf = EvalSetMultiOutputClassifier(SkRF(n_estimators=10, random_state=42))
        clf.fit(X, Y, eval_set=None)
        preds = clf.predict(X)
        assert preds.shape == (50, 2)

    def test_predict_shape(self) -> None:
        from sklearn.ensemble import RandomForestClassifier as SkRF

        rng = np.random.RandomState(42)
        X = rng.rand(60, 4)
        Y = rng.randint(0, 2, (60, 3))

        clf = EvalSetMultiOutputClassifier(SkRF(n_estimators=10, random_state=42))
        clf.fit(X, Y)
        preds = clf.predict(X)
        assert preds.shape == (60, 3)

    def test_predict_proba_returns_list(self) -> None:
        from sklearn.ensemble import RandomForestClassifier as SkRF

        rng = np.random.RandomState(42)
        X = rng.rand(50, 3)
        Y = rng.randint(0, 2, (50, 2))

        clf = EvalSetMultiOutputClassifier(SkRF(n_estimators=10, random_state=42))
        clf.fit(X, Y)
        proba = clf.predict_proba(X)
        assert isinstance(proba, list)
        assert len(proba) == 2
        assert proba[0].shape == (50, 2)
