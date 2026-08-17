"""Tests for calibration and conformal prediction modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquacontam.models.xgboost import XGBoostClassifier


class TestCalibration:
    """Tests for post-hoc probability calibration."""

    def _fit_base_model(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> tuple[XGBoostClassifier, pd.DataFrame, pd.Series]:
        X, y = classification_data
        model = XGBoostClassifier(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        return model, X, y

    def test_isotonic_calibration(self, classification_data: tuple) -> None:
        from aquacontam.calibration.methods import calibrate

        model, X, y = self._fit_base_model(classification_data)
        cal_model = calibrate(model, X, y, method="isotonic")
        probs = cal_model.predict_proba(X)
        assert probs.shape == (len(X), 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=0.01)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_platt_calibration(self, classification_data: tuple) -> None:
        from aquacontam.calibration.methods import calibrate

        model, X, y = self._fit_base_model(classification_data)
        cal_model = calibrate(model, X, y, method="platt")
        probs = cal_model.predict_proba(X)
        assert probs.shape == (len(X), 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=0.01)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_temperature_calibration(self, classification_data: tuple) -> None:
        from aquacontam.calibration.methods import calibrate

        model, X, y = self._fit_base_model(classification_data)
        cal_model = calibrate(model, X, y, method="temperature")
        probs = cal_model.predict_proba(X)
        assert probs.shape == (len(X), 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=0.01)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_calibrated_model_predict(self, classification_data: tuple) -> None:
        from aquacontam.calibration.methods import calibrate

        model, X, y = self._fit_base_model(classification_data)
        cal_model = calibrate(model, X, y, method="isotonic")
        preds = cal_model.predict(X)
        raw_preds = model.predict(X)
        # predict() should delegate to the wrapped model
        np.testing.assert_array_equal(preds, raw_preds)

    def test_calibrated_model_name(self, classification_data: tuple) -> None:
        from aquacontam.calibration.methods import calibrate

        model, X, y = self._fit_base_model(classification_data)
        cal_model = calibrate(model, X, y, method="isotonic")
        assert cal_model.name == "xgboost_classifier_calibrated_isotonic"

        cal_model_temp = calibrate(model, X, y, method="temperature")
        assert cal_model_temp.name == "xgboost_classifier_calibrated_temperature"

    def test_calibrated_model_fit_raises(self, classification_data: tuple) -> None:
        from aquacontam.calibration.methods import calibrate

        model, X, y = self._fit_base_model(classification_data)
        cal_model = calibrate(model, X, y, method="isotonic")
        with pytest.raises(RuntimeError, match="already fitted"):
            cal_model.fit(X, y)

    def test_invalid_method_raises(self, classification_data: tuple) -> None:
        from aquacontam.calibration.methods import calibrate

        model, X, y = self._fit_base_model(classification_data)
        with pytest.raises(ValueError, match="method must be one of"):
            calibrate(model, X, y, method="bogus")

    def test_reliability_diagram_data(self, classification_data: tuple) -> None:
        from aquacontam.calibration.methods import reliability_diagram_data

        model, X, y = self._fit_base_model(classification_data)
        probs = model.predict_proba(X)[:, 1]
        result = reliability_diagram_data(np.asarray(y), probs)
        assert "ece" in result
        assert "fraction_of_positives" in result
        assert "mean_predicted_value" in result
        assert "n_bins" in result
        assert result["ece"] >= 0
        assert result["n_bins"] == 10
        assert len(result["fraction_of_positives"]) == len(result["mean_predicted_value"])


class TestConformalPrediction:
    """Tests for split conformal prediction."""

    def _fit_base_model(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> tuple[XGBoostClassifier, pd.DataFrame, pd.Series]:
        X, y = classification_data
        model = XGBoostClassifier(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        return model, X, y

    def test_conformal_coverage(self, classification_data: tuple) -> None:
        from aquacontam.calibration.conformal import split_conformal

        model, X, y = self._fit_base_model(classification_data)
        cc = split_conformal(model, X, y, alpha=0.1)
        stats = cc.coverage_and_set_size(X, y)
        # Coverage should be >= 1 - alpha (with some tolerance for finite samples)
        assert stats["coverage"] >= 0.85  # 1 - 0.1 - tolerance

    def test_predict_sets_shape(self, classification_data: tuple) -> None:
        from aquacontam.calibration.conformal import split_conformal

        model, X, y = self._fit_base_model(classification_data)
        cc = split_conformal(model, X, y, alpha=0.1)
        sets = cc.predict_sets(X)
        assert sets.shape == (len(X), 2)
        assert sets.dtype == bool

    def test_alpha_extreme_values(self, classification_data: tuple) -> None:
        from aquacontam.calibration.conformal import split_conformal

        model, X, y = self._fit_base_model(classification_data)
        # alpha=0.01 -> should include almost everything
        cc = split_conformal(model, X, y, alpha=0.01)
        stats = cc.coverage_and_set_size(X, y)
        assert stats["coverage"] >= 0.95

    def test_conformal_stats_keys(self, classification_data: tuple) -> None:
        from aquacontam.calibration.conformal import split_conformal

        model, X, y = self._fit_base_model(classification_data)
        cc = split_conformal(model, X, y, alpha=0.1)
        stats = cc.coverage_and_set_size(X, y)
        expected_keys = {"coverage", "avg_set_size", "singleton_frac", "both_classes_frac"}
        assert set(stats.keys()) == expected_keys
        # Set sizes should be between 0 and 2
        assert 0 <= stats["avg_set_size"] <= 2.0
        assert 0 <= stats["singleton_frac"] <= 1.0
        assert 0 <= stats["both_classes_frac"] <= 1.0

    def test_uncalibrated_predict_sets_raises(self, classification_data: tuple) -> None:
        from aquacontam.calibration.conformal import ConformalClassifier

        model, X, _y = self._fit_base_model(classification_data)
        cc = ConformalClassifier(model, alpha=0.1)
        with pytest.raises(RuntimeError, match="not been calibrated"):
            cc.predict_sets(X)

    def test_conformal_via_convenience(self, classification_data: tuple) -> None:
        from aquacontam.calibration import split_conformal

        model, X, y = self._fit_base_model(classification_data)
        cc = split_conformal(model, X, y, alpha=0.2)
        stats = cc.coverage_and_set_size(X, y)
        assert stats["coverage"] >= 0.7  # 1 - 0.2 - tolerance


class TestGroupConformalPrediction:
    """Tests for group-conditional conformal prediction."""

    def _fit_base_model(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> tuple[XGBoostClassifier, pd.DataFrame, pd.Series]:
        X, y = classification_data
        model = XGBoostClassifier(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        return model, X, y

    def test_group_conformal_coverage(self, classification_data: tuple) -> None:
        from aquacontam.calibration.conformal import GroupConformalClassifier

        model, X, y = self._fit_base_model(classification_data)
        # Assign groups (3 groups of ~33 each)
        groups = np.array([i % 3 for i in range(len(y))])
        gcc = GroupConformalClassifier(model, alpha=0.1)
        gcc.calibrate(X, y, groups)
        stats = gcc.coverage_and_set_size(X, y, groups)
        assert stats["coverage"] >= 0.80  # 1 - 0.1 - tolerance
        assert "per_group" in stats
        assert len(stats["per_group"]) == 3

    def test_group_conformal_predict_sets_shape(self, classification_data: tuple) -> None:
        from aquacontam.calibration.conformal import GroupConformalClassifier

        model, X, y = self._fit_base_model(classification_data)
        groups = np.array([i % 2 for i in range(len(y))])
        gcc = GroupConformalClassifier(model, alpha=0.1)
        gcc.calibrate(X, y, groups)
        sets = gcc.predict_sets(X, groups)
        assert sets.shape == (len(X), 2)
        assert sets.dtype == bool

    def test_group_conformal_uncalibrated_raises(self, classification_data: tuple) -> None:
        from aquacontam.calibration.conformal import GroupConformalClassifier

        model, X, _y = self._fit_base_model(classification_data)
        gcc = GroupConformalClassifier(model, alpha=0.1)
        groups = np.zeros(len(X), dtype=int)
        with pytest.raises(RuntimeError, match="not been calibrated"):
            gcc.predict_sets(X, groups)

    def test_group_conformal_small_group_fallback(self, classification_data: tuple) -> None:
        """Small groups should fall back to marginal threshold with warning."""
        from aquacontam.calibration.conformal import GroupConformalClassifier

        model, X, y = self._fit_base_model(classification_data)
        # Create groups: group 0 has most data, group 99 has only 3 samples
        groups = np.zeros(len(y), dtype=int)
        groups[:3] = 99  # only 3 samples — below n_min=10 for alpha=0.1
        gcc = GroupConformalClassifier(model, alpha=0.1)
        gcc.calibrate(X, y, groups)
        # Group 99 should have fallen back to the marginal threshold
        assert gcc._group_thresholds[99] == gcc._fallback_threshold

    def test_uncalibrated_group_not_reliable(self, classification_data: tuple) -> None:
        """A group present only at evaluation time (never calibrated) must report
        reliable_guarantee=False -- it falls back to the marginal threshold and has
        no per-group guarantee. Regression test: in the pipeline, calibration used
        disjoint validation regions {2,7} while evaluation used test regions {8,9,10},
        so every test region fell back to the marginal threshold yet was previously
        (incorrectly) marked reliable.
        """
        from aquacontam.calibration.conformal import GroupConformalClassifier

        model, X, y = self._fit_base_model(classification_data)
        cal_groups = np.zeros(len(y), dtype=int)  # all calibration data in group 0
        gcc = GroupConformalClassifier(model, alpha=0.1)
        gcc.calibrate(X, y, cal_groups)
        # Evaluate on a DISJOINT group (7) never seen during calibration
        eval_groups = np.full(len(y), 7, dtype=int)
        stats = gcc.coverage_and_set_size(X, y, eval_groups)
        pg = stats["per_group"]["7"]
        assert pg["n_calibration"] == 0
        assert pg["reliable_guarantee"] is False
        # The genuinely calibrated group remains reliable
        stats0 = gcc.coverage_and_set_size(X, y, cal_groups)
        assert stats0["per_group"]["0"]["reliable_guarantee"] is True


class TestConformalCalibrationSizeWarning:
    """Tests for small calibration set size warnings."""

    def _fit_base_model(
        self, classification_data: tuple[pd.DataFrame, pd.Series]
    ) -> tuple[XGBoostClassifier, pd.DataFrame, pd.Series]:
        X, y = classification_data
        model = XGBoostClassifier(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        return model, X, y

    def test_small_calibration_set_warns(
        self, classification_data: tuple, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Calibrating with fewer samples than ceil(1/alpha) should warn."""
        from aquacontam.calibration.conformal import ConformalClassifier

        model, X, y = self._fit_base_model(classification_data)
        cc = ConformalClassifier(model, alpha=0.1)
        # Use only 5 samples (n_min = ceil(1/0.1) = 10)
        with caplog.at_level("WARNING", logger="aquacontam.calibration.conformal"):
            cc.calibrate(X.iloc[:5], y.iloc[:5])
        assert cc._threshold is not None
        assert "Calibration set size 5 < minimum 10" in caplog.text

    def test_adequate_calibration_set_no_issue(self, classification_data: tuple) -> None:
        """Calibrating with enough samples should work without issues."""
        from aquacontam.calibration.conformal import ConformalClassifier

        model, X, y = self._fit_base_model(classification_data)
        cc = ConformalClassifier(model, alpha=0.1)
        # Use all samples (should be well above n_min=10)
        cc.calibrate(X, y)
        assert cc._threshold is not None
