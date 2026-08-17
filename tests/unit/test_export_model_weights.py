"""Tests for model weight export and import (Zenodo release artifacts)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from aquacontam.export.model_weights import (
    export_model_weights,
    load_model_weights,
    verify_model_integrity,
)
from aquacontam.models.logistic import LogisticRegressionClassifier
from aquacontam.models.random_forest import RandomForestClassifier


@pytest.fixture()
def synth_data() -> tuple[pd.DataFrame, np.ndarray]:
    """50-sample synthetic classification dataset."""
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    return X_df, y


@pytest.fixture()
def fitted_rf(synth_data: tuple[pd.DataFrame, np.ndarray]) -> RandomForestClassifier:
    """A fitted Random Forest classifier (no scaler)."""
    X, y = synth_data
    model = RandomForestClassifier({"n_estimators": 10, "random_state": 42})
    model.fit(X, y)
    return model


@pytest.fixture()
def fitted_lr(synth_data: tuple[pd.DataFrame, np.ndarray]) -> LogisticRegressionClassifier:
    """A fitted Logistic Regression classifier (has scaler)."""
    X, y = synth_data
    model = LogisticRegressionClassifier({"random_state": 42})
    model.fit(X, y)
    return model


class TestExportModelWeights:
    """Tests for export_model_weights."""

    def test_creates_joblib_and_metadata(self, fitted_rf: RandomForestClassifier, tmp_path: Path):
        """Export produces both .joblib and _metadata.json files."""
        path = export_model_weights(fitted_rf, "T1", tmp_path)
        assert path.exists()
        assert path.suffix == ".joblib"
        meta_path = tmp_path / f"T1_{fitted_rf.name}_metadata.json"
        assert meta_path.exists()

    def test_metadata_contents(self, fitted_rf: RandomForestClassifier, tmp_path: Path):
        """Metadata JSON has correct task, model_name, config, and sha256."""
        export_model_weights(fitted_rf, "T1", tmp_path, metrics={"auroc": 0.92})
        meta_path = tmp_path / f"T1_{fitted_rf.name}_metadata.json"
        meta = json.loads(meta_path.read_text())
        assert meta["task"] == "T1"
        assert meta["model_name"] == fitted_rf.name
        assert "config" in meta
        assert meta["test_metrics"]["auroc"] == 0.92
        assert "sha256" in meta
        assert len(meta["sha256"]) == 64  # SHA-256 hex digest length

    def test_metadata_feature_names_explicit(
        self, fitted_rf: RandomForestClassifier, tmp_path: Path
    ):
        """Explicit feature_names are saved in metadata."""
        names = ["a", "b", "c", "d", "e"]
        export_model_weights(fitted_rf, "T1", tmp_path, feature_names=names)
        meta = json.loads((tmp_path / f"T1_{fitted_rf.name}_metadata.json").read_text())
        assert meta["feature_names"] == names


class TestRoundtrip:
    """Tests for export + load roundtrip."""

    def test_rf_roundtrip_predictions(
        self,
        fitted_rf: RandomForestClassifier,
        synth_data: tuple[pd.DataFrame, np.ndarray],
        tmp_path: Path,
    ):
        """RF predictions are identical after export+load roundtrip."""
        X, _ = synth_data
        original_preds = fitted_rf.predict(X)
        original_proba = fitted_rf.predict_proba(X)

        path = export_model_weights(fitted_rf, "T1", tmp_path)
        loaded = load_model_weights(RandomForestClassifier, path, config=fitted_rf.config)

        np.testing.assert_array_equal(loaded.predict(X), original_preds)
        np.testing.assert_array_equal(loaded.predict_proba(X), original_proba)

    def test_logistic_with_scaler_roundtrip(
        self,
        fitted_lr: LogisticRegressionClassifier,
        synth_data: tuple[pd.DataFrame, np.ndarray],
        tmp_path: Path,
    ):
        """LogisticRegression scaler survives export+load roundtrip."""
        X, _ = synth_data
        original_preds = fitted_lr.predict(X)

        path = export_model_weights(fitted_lr, "T1", tmp_path)
        loaded = load_model_weights(LogisticRegressionClassifier, path, config=fitted_lr.config)

        assert loaded._scaler is not None
        np.testing.assert_array_equal(loaded.predict(X), original_preds)

    def test_feature_names_restored(self, fitted_rf: RandomForestClassifier, tmp_path: Path):
        """feature_names_in_ attribute is restored from metadata."""
        names = ["a", "b", "c", "d", "e"]
        path = export_model_weights(fitted_rf, "T1", tmp_path, feature_names=names)
        loaded = load_model_weights(RandomForestClassifier, path)
        assert hasattr(loaded, "feature_names_in_")
        assert loaded.feature_names_in_ == names


class TestLegacyFormat:
    """Test backwards compatibility with raw sklearn model exports."""

    def test_load_legacy_raw_model(
        self,
        fitted_rf: RandomForestClassifier,
        synth_data: tuple[pd.DataFrame, np.ndarray],
        tmp_path: Path,
    ):
        """Models saved as raw sklearn objects (not dict) can still be loaded."""
        import joblib

        # Simulate legacy format: joblib.dump(sklearn_model) directly
        legacy_path = tmp_path / "T1_legacy.joblib"
        joblib.dump(fitted_rf._model, legacy_path)

        loaded = load_model_weights(RandomForestClassifier, legacy_path, config=fitted_rf.config)

        X, _ = synth_data
        np.testing.assert_array_equal(loaded.predict(X), fitted_rf.predict(X))


class TestIntegrity:
    """Tests for SHA-256 integrity verification."""

    def test_sha256_stored_in_metadata(self, fitted_rf: RandomForestClassifier, tmp_path: Path):
        """Export stores SHA-256 hash in metadata JSON."""
        export_model_weights(fitted_rf, "T1", tmp_path)
        meta_path = tmp_path / f"T1_{fitted_rf.name}_metadata.json"
        meta = json.loads(meta_path.read_text())
        assert "sha256" in meta
        assert isinstance(meta["sha256"], str)
        assert len(meta["sha256"]) == 64

    def test_load_verifies_integrity(
        self,
        fitted_rf: RandomForestClassifier,
        synth_data: tuple[pd.DataFrame, np.ndarray],
        tmp_path: Path,
    ):
        """Load succeeds when hash matches (roundtrip is clean)."""
        X, _ = synth_data
        path = export_model_weights(fitted_rf, "T1", tmp_path)
        loaded = load_model_weights(RandomForestClassifier, path, config=fitted_rf.config)
        np.testing.assert_array_equal(loaded.predict(X), fitted_rf.predict(X))

    def test_load_warns_on_tampered_file(
        self, fitted_rf: RandomForestClassifier, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        """Load warns when model file has been modified after export."""
        import logging

        path = export_model_weights(fitted_rf, "T1", tmp_path)

        # Tamper with the model file by appending bytes
        with open(path, "ab") as f:
            f.write(b"tampered")

        # Should not raise in non-strict mode, but should log a warning
        with caplog.at_level(logging.WARNING):
            loaded = load_model_weights(RandomForestClassifier, path, config=fitted_rf.config)

        assert loaded is not None
        assert "integrity check FAILED" in caplog.text

    def test_load_legacy_no_hash_succeeds(self, fitted_rf: RandomForestClassifier, tmp_path: Path):
        """Legacy metadata without sha256 field loads without error."""
        import joblib

        model_path = tmp_path / "T1_legacy.joblib"
        joblib.dump({"model": fitted_rf._model}, model_path)

        # Write metadata WITHOUT sha256
        meta = {"task": "T1", "model_name": fitted_rf.name, "config": {}}
        meta_path = tmp_path / "T1_legacy_metadata.json"
        meta_path.write_text(json.dumps(meta))

        loaded = load_model_weights(RandomForestClassifier, model_path, config=fitted_rf.config)
        assert loaded is not None

    def test_verify_model_integrity_all_ok(
        self, fitted_rf: RandomForestClassifier, tmp_path: Path
    ):
        """verify_model_integrity returns True for all valid models."""
        export_model_weights(fitted_rf, "T1", tmp_path)
        results = verify_model_integrity(tmp_path)
        assert len(results) == 1
        assert all(results.values())

    def test_verify_model_integrity_tampered(
        self, fitted_rf: RandomForestClassifier, tmp_path: Path
    ):
        """verify_model_integrity detects tampered files."""
        path = export_model_weights(fitted_rf, "T1", tmp_path)

        # Tamper
        with open(path, "ab") as f:
            f.write(b"tampered")

        results = verify_model_integrity(tmp_path)
        assert len(results) == 1
        assert not all(results.values())

    def test_verify_model_integrity_missing_metadata(self, tmp_path: Path):
        """verify_model_integrity returns False for models without metadata."""
        import joblib

        orphan = tmp_path / "orphan.joblib"
        joblib.dump({"model": None}, orphan)

        results = verify_model_integrity(tmp_path)
        assert results["orphan.joblib"] is False
