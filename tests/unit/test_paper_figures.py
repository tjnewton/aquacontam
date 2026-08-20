"""Tests for paper figure generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

import paper.generate_figures as gf
from paper.generate_figures import (
    MissingFigureDataError,
    fig_benchmark_results,
    fig_causal_comparison,
    fig_data_distribution,
    fig_dataset_overview,
    fig_equity_analysis,
    fig_feature_importance,
    fig_group_conformal,
    fig_national_risk_map,
    fig_reliability_diagram,
    fig_temporal_prediction,
    fig_transfer_learning,
)

from aquacontam.benchmark.metrics import calibration_curve_data


@pytest.fixture()
def allow_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt in to placeholder panels (canonical runs loud-fail instead)."""
    monkeypatch.setattr(gf, "ALLOW_PLACEHOLDERS", True)


@pytest.fixture()
def synthetic_wq_for_fig() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "pwsid": [f"SYS{i:05d}" for i in range(50)],
            "analyte": rng.choice(["PFOS", "PFOA"], 50),
            "concentration": rng.exponential(10.0, 50),
            "censored": rng.random(50) < 0.7,
            "latitude": rng.uniform(25, 48, 50),
            "longitude": rng.uniform(-125, -67, 50),
        }
    )


class TestFigures:
    """Test each figure function produces valid output."""

    def test_fig1_dataset_overview(
        self, tmp_path: Path, synthetic_wq_for_fig: pd.DataFrame
    ) -> None:
        path = fig_dataset_overview(synthetic_wq_for_fig, tmp_path)
        assert path.exists()
        assert path.suffix == ".pdf"
        assert path.with_suffix(".png").exists()

    def test_fig2_data_distribution(
        self, tmp_path: Path, synthetic_wq_for_fig: pd.DataFrame
    ) -> None:
        path = fig_data_distribution(synthetic_wq_for_fig, tmp_path)
        assert path.exists()

    def test_fig3_benchmark_results_with_data(self, tmp_path: Path) -> None:
        results = [
            {"task": "T1", "model": "xgboost", "metrics": {"auprc": 0.85}},
            {"task": "T1", "model": "rf", "metrics": {"auprc": 0.80}},
        ]
        path = fig_benchmark_results(results, tmp_path)
        assert path.exists()

    def test_fig3_benchmark_results_empty(self, tmp_path: Path, allow_placeholders: None) -> None:
        path = fig_benchmark_results([], tmp_path)
        assert path.exists()

    def test_missing_data_raises_by_default(self, tmp_path: Path) -> None:
        """Canonical regeneration must never silently ship a placeholder."""
        with pytest.raises(MissingFigureDataError):
            fig_benchmark_results([], tmp_path)
        with pytest.raises(MissingFigureDataError):
            fig_feature_importance(None, tmp_path)
        with pytest.raises(MissingFigureDataError):
            fig_reliability_diagram(None, tmp_path)

    def test_fig4_feature_importance(self, tmp_path: Path) -> None:
        imp = pd.Series(
            [0.15, 0.10, 0.08],
            index=["feat_a", "feat_b", "feat_c"],
            name="importance",
        )
        path = fig_feature_importance(imp, tmp_path)
        assert path.exists()

    def test_fig4_feature_importance_none(self, tmp_path: Path, allow_placeholders: None) -> None:
        path = fig_feature_importance(None, tmp_path)
        assert path.exists()

    def test_fig5_national_risk_map(self, tmp_path: Path, allow_placeholders: None) -> None:
        path = fig_national_risk_map(None, tmp_path)
        assert path.exists()

    def test_fig5_national_risk_map_with_data(self, tmp_path: Path) -> None:
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            {
                "latitude": rng.uniform(25, 48, 100),
                "longitude": rng.uniform(-125, -67, 100),
                "risk_score": rng.random(100),
            }
        )
        path = fig_national_risk_map(df, tmp_path)
        assert path.exists()

    def test_fig6_equity_analysis(self, tmp_path: Path, allow_placeholders: None) -> None:
        path = fig_equity_analysis(None, tmp_path)
        assert path.exists()

    def test_fig7_transfer_learning(self, tmp_path: Path, allow_placeholders: None) -> None:
        path = fig_transfer_learning(None, tmp_path)
        assert path.exists()

    def test_fig8_temporal_prediction(self, tmp_path: Path, allow_placeholders: None) -> None:
        path = fig_temporal_prediction(None, tmp_path)
        assert path.exists()

    def test_fig_ext8_causal_comparison_with_data(self, tmp_path: Path) -> None:
        causal_data = [
            {
                "feature": "pct_people_of_color",
                "causal_effect": 0.153,
                "std_error": 0.014,
                "p_value": 0.0,
                "shap_importance": 0.011,
                "ratio": 14.4,
            },
            {
                "feature": "n_samples",
                "causal_effect": 0.05,
                "std_error": 0.01,
                "p_value": 0.001,
                "shap_importance": 0.314,
                "ratio": 0.16,
            },
        ]
        path = fig_causal_comparison(causal_data, tmp_path)
        assert path.exists()
        assert path.suffix == ".pdf"
        assert path.with_suffix(".png").exists()

    def test_fig_ext8_causal_comparison_none(
        self, tmp_path: Path, allow_placeholders: None
    ) -> None:
        path = fig_causal_comparison(None, tmp_path)
        assert path.exists()

    def test_fig_ext9_group_conformal_with_data(self, tmp_path: Path) -> None:
        conformal_data = [
            {
                "alpha": 0.05,
                "target_coverage": 0.95,
                "coverage": 0.945,
                "avg_set_size": 1.403,
                "per_region": {
                    "8": {"coverage": 0.980, "avg_set_size": 1.384, "n": 489},
                    "9": {"coverage": 0.911, "avg_set_size": 1.373, "n": 1004},
                    "10": {"coverage": 0.978, "avg_set_size": 1.482, "n": 492},
                },
            }
        ]
        path = fig_group_conformal(conformal_data, tmp_path)
        assert path.exists()
        assert path.suffix == ".pdf"
        assert path.with_suffix(".png").exists()

    def test_fig_ext9_group_conformal_none(self, tmp_path: Path, allow_placeholders: None) -> None:
        path = fig_group_conformal(None, tmp_path)
        assert path.exists()

    def test_all_figures_produce_pdfs(
        self, tmp_path: Path, synthetic_wq_for_fig: pd.DataFrame, allow_placeholders: None
    ) -> None:
        """All 10 main+ext figures produce output files."""
        fig_dataset_overview(synthetic_wq_for_fig, tmp_path)
        fig_data_distribution(synthetic_wq_for_fig, tmp_path)
        fig_benchmark_results([], tmp_path)
        fig_feature_importance(None, tmp_path)
        fig_national_risk_map(None, tmp_path)
        fig_equity_analysis(None, tmp_path)
        fig_transfer_learning(None, tmp_path)
        fig_temporal_prediction(None, tmp_path)
        fig_causal_comparison(None, tmp_path)
        fig_group_conformal(None, tmp_path)

        pdfs = list(tmp_path.glob("*.pdf"))
        assert len(pdfs) == 10


# Keys ``calibration_curve_data`` emits; ``fig_reliability_diagram`` must read these.
_EXPECTED_CURVE_KEYS = {"mean_predicted", "fraction_positive", "bin_counts"}


class TestReliabilityDiagram:
    """Calibration figure data-contract guards (Supplementary Fig. 5).

    A key-name mismatch between ``calibration_curve_data`` (writer) and
    ``fig_reliability_diagram`` (reader) once silently emptied the figure; these
    tests fail loudly if the curve-dict keys ever drift apart again.
    """

    def test_calibration_curve_data_keys(self) -> None:
        rng = np.random.RandomState(0)
        out = calibration_curve_data(rng.randint(0, 2, size=200), rng.random(200))
        assert set(out.keys()) == _EXPECTED_CURVE_KEYS
        assert len(out["mean_predicted"]) == len(out["fraction_positive"]) > 1

    def test_reliability_diagram_plots_model_curve(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from matplotlib.axes import Axes

        rng = np.random.RandomState(1)
        entry = {
            "task": "T1",
            "model": "xgboost_classifier",
            "ece": 0.05,
            "calibration_curve": calibration_curve_data(
                rng.randint(0, 2, size=300), rng.random(300)
            ),
        }
        labels: list[str] = []
        original = Axes.plot

        def spy(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            labels.append(str(kwargs.get("label", "")))
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Axes, "plot", spy)
        fig_reliability_diagram([entry], tmp_path)

        model_curves = [lbl for lbl in labels if lbl and lbl != "Perfect"]
        assert model_curves, "no model curve plotted -- calibration key-mismatch regression"
        assert (tmp_path / "fig_reliability_diagram.png").exists()

    def test_reliability_diagram_skips_dummy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from matplotlib.axes import Axes

        rng = np.random.RandomState(2)
        curve = calibration_curve_data(rng.randint(0, 2, size=300), rng.random(300))
        entries = [
            {"task": "T1", "model": "dummy_classifier", "ece": 0.02, "calibration_curve": curve},
            {"task": "T1", "model": "xgboost_classifier", "ece": 0.05, "calibration_curve": curve},
        ]
        labels: list[str] = []
        original = Axes.plot

        def spy(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            labels.append(str(kwargs.get("label", "")))
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Axes, "plot", spy)
        fig_reliability_diagram(entries, tmp_path)
        assert not any("dummy" in lbl.lower() for lbl in labels)
