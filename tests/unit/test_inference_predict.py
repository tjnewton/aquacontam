"""Tests for grid prediction pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from aquacontam.inference.predict import (
    _assign_grid_epa_region,
    _predict_for_grid,
    generate_grid_predictions,
)


@pytest.fixture()
def mock_classifier() -> MagicMock:
    """Fitted classification model mock."""
    model = MagicMock()
    model.name = "mock_clf"
    model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2], [0.1, 0.9]])
    model.predict.return_value = np.array([1, 0, 1])
    return model


@pytest.fixture()
def mock_regressor() -> MagicMock:
    """Fitted regression model mock."""
    model = MagicMock()
    model.name = "mock_reg"
    model.predict.return_value = np.array([2.0, 5.0, 8.0])
    return model


@pytest.fixture()
def grid_df() -> pd.DataFrame:
    """Small grid DataFrame."""
    return pd.DataFrame(
        {
            "grid_id": ["G0000001", "G0000002", "G0000003"],
            "latitude": [35.0, 40.0, 45.0],
            "longitude": [-90.0, -75.0, -120.0],
        }
    )


@pytest.fixture()
def feature_df() -> pd.DataFrame:
    """Feature matrix matching grid size."""
    rng = np.random.RandomState(42)
    return pd.DataFrame(rng.randn(3, 5), columns=[f"f{i}" for i in range(5)])


class TestAssignGridEpaRegion:
    """Tests for _assign_grid_epa_region."""

    def test_returns_series_of_ints(self) -> None:
        lats = pd.Series([35.0, 40.0, 45.0])
        lons = pd.Series([-90.0, -75.0, -120.0])
        result = _assign_grid_epa_region(lats, lons)
        assert isinstance(result, pd.Series)
        assert result.dtype == int

    def test_correct_length(self) -> None:
        lats = np.array([30.0, 40.0])
        lons = np.array([-80.0, -100.0])
        result = _assign_grid_epa_region(lats, lons)
        assert len(result) == 2

    def test_boston_area_region(self) -> None:
        # Boston (~42.36, ~-71.0) — heuristic assigns Region 2
        result = _assign_grid_epa_region(np.array([42.36]), np.array([-71.0]))
        assert result.iloc[0] == 2

    def test_pacific_default(self) -> None:
        # LA area
        result = _assign_grid_epa_region(np.array([34.0]), np.array([-118.0]))
        assert result.iloc[0] == 9


class TestPredictForGrid:
    """Tests for _predict_for_grid."""

    def test_classification_returns_scores_and_preds(
        self, mock_classifier: MagicMock, feature_df: pd.DataFrame
    ) -> None:
        scores, preds = _predict_for_grid(mock_classifier, feature_df, "classification")
        assert len(scores) == 3
        assert len(preds) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert set(preds).issubset({0.0, 1.0})

    def test_regression_normalizes_scores(
        self, mock_regressor: MagicMock, feature_df: pd.DataFrame
    ) -> None:
        scores, preds = _predict_for_grid(mock_regressor, feature_df, "regression")
        assert len(scores) == 3
        assert float(scores.min()) == pytest.approx(0.0)
        assert float(scores.max()) == pytest.approx(1.0)
        # Raw predictions preserved
        np.testing.assert_array_almost_equal(preds, [2.0, 5.0, 8.0])

    def test_regression_constant_predictions(self, feature_df: pd.DataFrame) -> None:
        model = MagicMock()
        model.name = "const"
        model.predict.return_value = np.array([5.0, 5.0, 5.0])
        scores, _ = _predict_for_grid(model, feature_df, "regression")
        np.testing.assert_array_equal(scores, [0.5, 0.5, 0.5])


class TestGenerateGridPredictions:
    """Tests for generate_grid_predictions."""

    def test_writes_parquet_files(
        self,
        tmp_path: Path,
        mock_classifier: MagicMock,
        grid_df: pd.DataFrame,
        feature_df: pd.DataFrame,
    ) -> None:
        paths = generate_grid_predictions(
            models={"T1": mock_classifier},
            feature_df=feature_df,
            grid=grid_df,
            output_dir=tmp_path / "preds",
            task_analytes={"T1": ("PFOS",)},
        )
        assert len(paths) == 1
        assert paths[0].exists()

    def test_parquet_schema(
        self,
        tmp_path: Path,
        mock_classifier: MagicMock,
        grid_df: pd.DataFrame,
        feature_df: pd.DataFrame,
    ) -> None:
        generate_grid_predictions(
            models={"T1": mock_classifier},
            feature_df=feature_df,
            grid=grid_df,
            output_dir=tmp_path / "preds",
            task_analytes={"T1": ("PFOS",)},
        )
        result = pd.read_parquet(
            tmp_path / "preds" / "task=T1" / "analyte=PFOS" / "predictions.parquet"
        )
        expected_cols = {
            "grid_id",
            "latitude",
            "longitude",
            "task",
            "analyte",
            "model_name",
            "risk_score",
            "prediction",
            "epa_region",
        }
        assert set(result.columns) == expected_cols
        assert len(result) == 3

    def test_multiple_tasks_and_analytes(
        self,
        tmp_path: Path,
        mock_classifier: MagicMock,
        mock_regressor: MagicMock,
        grid_df: pd.DataFrame,
        feature_df: pd.DataFrame,
    ) -> None:
        paths = generate_grid_predictions(
            models={"T1": mock_classifier, "T2": mock_regressor},
            feature_df=feature_df,
            grid=grid_df,
            output_dir=tmp_path / "preds",
            task_analytes={"T1": ("PFOS", "PFOA"), "T2": ("PFOS",)},
        )
        assert len(paths) == 3

    def test_risk_scores_in_range(
        self,
        tmp_path: Path,
        mock_classifier: MagicMock,
        grid_df: pd.DataFrame,
        feature_df: pd.DataFrame,
    ) -> None:
        generate_grid_predictions(
            models={"T1": mock_classifier},
            feature_df=feature_df,
            grid=grid_df,
            output_dir=tmp_path / "preds",
            task_analytes={"T1": ("PFOS",)},
        )
        result = pd.read_parquet(
            tmp_path / "preds" / "task=T1" / "analyte=PFOS" / "predictions.parquet"
        )
        assert (result["risk_score"] >= 0).all()
        assert (result["risk_score"] <= 1).all()

    def test_missing_lat_lon_raises(
        self,
        tmp_path: Path,
        mock_classifier: MagicMock,
        feature_df: pd.DataFrame,
    ) -> None:
        bad_grid = pd.DataFrame({"grid_id": ["G1", "G2", "G3"]})
        with pytest.raises(ValueError, match="latitude/longitude"):
            generate_grid_predictions(
                models={"T1": mock_classifier},
                feature_df=feature_df,
                grid=bad_grid,
                output_dir=tmp_path / "preds",
            )

    def test_default_analytes_used(
        self,
        tmp_path: Path,
        mock_classifier: MagicMock,
        grid_df: pd.DataFrame,
        feature_df: pd.DataFrame,
    ) -> None:
        paths = generate_grid_predictions(
            models={"T1": mock_classifier},
            feature_df=feature_df,
            grid=grid_df,
            output_dir=tmp_path / "preds",
        )
        # T1 has 5 default analytes
        assert len(paths) == 5
