"""Tests for paper table generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("tabulate")

from paper.generate_tables import (
    table_benchmark_results,
    table_causal_deconfounding,
    table_dataset_summary,
    table_equity_summary,
    table_ext_feature_catalog,
    table_ext_per_analyte_results,
    table_feature_importance,
    table_group_conformal,
    table_regional_performance,
    table_tuning_comparison,
)


class TestTables:
    """Test each table function produces valid output files."""

    def test_table1_dataset_summary(self, tmp_path: Path) -> None:
        path = table_dataset_summary(tmp_path)
        assert path.exists()
        assert Path(str(path).replace(".csv", ".tex")).exists()
        assert Path(str(path).replace(".csv", ".md")).exists()

    def test_table2_benchmark_results_with_data(self, tmp_path: Path) -> None:
        results = [
            {"task": "T1", "model": "xgboost", "metrics": {"auroc": 0.9, "auprc": 0.85}},
            {"task": "T2", "model": "rf", "metrics": {"rmse": 1.5, "mae": 1.0}},
        ]
        path = table_benchmark_results(results, tmp_path)
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 2

    def test_table2_benchmark_results_empty(self, tmp_path: Path) -> None:
        path = table_benchmark_results([], tmp_path)
        assert path.exists()

    def test_table_supp_feature_importance(self, tmp_path: Path) -> None:
        imp = pd.Series(
            [0.15, 0.10, 0.08],
            index=["feat_a", "feat_b", "feat_c"],
            name="importance",
        )
        path = table_feature_importance(imp, tmp_path)
        assert path.exists()

    def test_table_supp_feature_importance_none(self, tmp_path: Path) -> None:
        path = table_feature_importance(None, tmp_path)
        assert path.exists()

    def test_table_ext4_equity_summary_placeholder(self, tmp_path: Path) -> None:
        path = table_equity_summary(None, tmp_path)
        assert path.exists()
        df = pd.read_csv(path)
        assert "Demographic Group" in df.columns
        assert "AUROC (High)" in df.columns
        assert "AUROC (Low)" in df.columns
        assert "p-value" in df.columns
        assert len(df) == 4  # 4 demographic groups

    def test_table_ext4_equity_summary_with_data(self, tmp_path: Path) -> None:
        equity_data = [
            {
                "group": "pct_people_of_color",
                "burden_ratio": 1.25,
                "prediction_ratio": 1.10,
                "n_high": 500,
                "n_low": 1000,
                "group_metrics": {
                    "high": {"auroc": 0.85, "f1": 0.7},
                    "low": {"auroc": 0.90, "f1": 0.75},
                },
                "p_value": 0.023,
            },
            {
                "group": "pct_low_income",
                "burden_ratio": 1.15,
                "prediction_ratio": 1.05,
                "n_high": 400,
                "n_low": 900,
                "group_metrics": {},
                "p_value": None,
            },
        ]
        path = table_equity_summary(equity_data, tmp_path)
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 2
        assert df.loc[0, "AUROC (High)"] == "0.850"
        assert df.loc[0, "p-value"] == "0.0230"

    def test_table_regional_performance_placeholder(self, tmp_path: Path) -> None:
        path = table_regional_performance([], tmp_path)
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 10  # 10 EPA regions
        assert set(df.columns) == {"Task", "Model", "Region", "AUROC", "AUPRC", "N"}

    def test_table_regional_performance_with_data(self, tmp_path: Path) -> None:
        results = [
            {
                "task": "T1",
                "model": "xgboost",
                "metrics": {"auroc": 0.9},
                "metadata": {
                    "region_metrics": {
                        "1": {"auroc": 0.92, "auprc": 0.88, "n_samples": 300},
                        "5": {"auroc": 0.87, "auprc": 0.80, "n_samples": 250},
                    },
                },
            },
            {
                "task": "T2",
                "model": "xgboost",
                "metrics": {"rmse": 1.5},
                "metadata": {},
            },
        ]
        path = table_regional_performance(results, tmp_path)
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 2  # only T1 (classification), T2 skipped
        assert df.loc[0, "Model"] == "XGBoost"

    def test_table_ext1_feature_catalog(self, tmp_path: Path) -> None:
        path = table_ext_feature_catalog(tmp_path)
        assert path.exists()
        df = pd.read_csv(path)
        assert "Feature" in df.columns

    def test_table_tuning_comparison_placeholder(self, tmp_path: Path) -> None:
        path = table_tuning_comparison(tmp_path)
        assert path.exists()
        content = path.read_text()
        assert "Tuning Comparison" in content

    def test_table_tuning_comparison_with_data(self, tmp_path: Path) -> None:
        import json

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        tuning_data = [
            {
                "model": "xgboost_classifier",
                "n_configs_tried": 18,
                "best_val_auprc": 0.25,
                "best_config": {"max_depth": 6},
                "tuned_test_metrics": {"auroc": 0.70, "auprc": 0.22},
                "default_test_metrics": {"auroc": 0.68, "auprc": 0.21},
                "delta_auroc": 0.02,
                "delta_auprc": 0.01,
            }
        ]
        (results_dir / "tuning_comparison.json").write_text(json.dumps(tuning_data))
        path = table_tuning_comparison(tmp_path, results_dir=results_dir)
        assert path.exists()
        content = path.read_text()
        assert "0.220" in content or "0.22" in content

    def test_table_causal_deconfounding_with_data(self, tmp_path: Path) -> None:
        import json

        results_dir = tmp_path / "results"
        results_dir.mkdir()
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
        auroc_data = {
            "original_auroc": 0.672,
            "deconfounded_auroc": 0.571,
            "n_features": 127,
        }
        (results_dir / "causal_deconfounding.json").write_text(json.dumps(causal_data))
        (results_dir / "deconfounded_auroc.json").write_text(json.dumps(auroc_data))
        path = table_causal_deconfounding(tmp_path, results_dir=results_dir)
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 2
        assert "pct_people_of_color" in df["Feature"].to_numpy()

    def test_table_causal_deconfounding_placeholder(self, tmp_path: Path) -> None:
        path = table_causal_deconfounding(tmp_path, results_dir=tmp_path / "empty")
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 5  # placeholder rows

    def test_table_group_conformal_with_data(self, tmp_path: Path) -> None:
        import json

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        conformal_data = [
            {
                "task": "T1",
                "model": "xgboost_classifier",
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
        (results_dir / "group_conformal_results.json").write_text(json.dumps(conformal_data))
        path = table_group_conformal(tmp_path, results_dir=results_dir)
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 1
        assert "Region 8" in df.columns

    def test_table_group_conformal_placeholder(self, tmp_path: Path) -> None:
        path = table_group_conformal(tmp_path, results_dir=tmp_path / "empty")
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 1

    def test_table_ext2_per_analyte(self, tmp_path: Path) -> None:
        results = [
            {
                "task": "T1",
                "model": "xgb",
                "metrics": {"auroc": 0.9},
                "metadata": {"analyte": "PFOS"},
            }
        ]
        path = table_ext_per_analyte_results(results, tmp_path)
        assert path.exists()


class TestProvenanceFreeLookup:
    """Lookup order for the environment-only ("without provenance") results.

    The frozen-snapshot name ``results_provenance_free.json`` must win over the
    legacy ``results_no_monitoring.json`` / ``../results_no_monitoring/results.json``
    fallbacks, and the decomposition table must render its values (not dashes).
    """

    @staticmethod
    def _entry(task: str, auroc: float, auprc: float) -> dict:
        return {
            "task": task,
            "model": "xgboost_classifier",
            "metrics": {"auroc": auroc, "auprc": auprc},
        }

    def _results_dir(self, tmp_path: Path, *, pf: bool, legacy: bool) -> Path:
        import json

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        full = [self._entry("T1", 0.859, 0.731), self._entry("T4", 0.962, 0.900)]
        (results_dir / "results.json").write_text(json.dumps(full), encoding="utf-8")
        if pf:
            pf_entries = [self._entry("T1", 0.820, 0.750), self._entry("T4", 0.545, 0.253)]
            (results_dir / "results_provenance_free.json").write_text(
                json.dumps(pf_entries), encoding="utf-8"
            )
        if legacy:
            legacy_entries = [self._entry("T1", 0.637, 0.400), self._entry("T4", 0.700, 0.300)]
            (results_dir / "results_no_monitoring.json").write_text(
                json.dumps(legacy_entries), encoding="utf-8"
            )
        return results_dir

    def test_monitoring_invariance_prefers_provenance_free(self, tmp_path: Path) -> None:
        from paper.generate_tables import table_monitoring_invariance

        results_dir = self._results_dir(tmp_path, pf=True, legacy=True)
        out = tmp_path / "tables"
        out.mkdir()
        path = table_monitoring_invariance(out, results_dir)
        df = pd.read_csv(path)
        pf_rows = df[df["Approach"].str.startswith("Provenance-free")]
        assert len(pf_rows) == 2
        t1 = pf_rows[pf_rows["Task"] == "T1"].iloc[0]
        assert t1["AUROC"] == "0.820"  # frozen PF value, not legacy 0.637
        assert t1["AUPRC"] == "0.750"
        assert "—" not in set(pf_rows["AUROC"])

    def test_monitoring_invariance_legacy_fallback(self, tmp_path: Path) -> None:
        from paper.generate_tables import table_monitoring_invariance

        results_dir = self._results_dir(tmp_path, pf=False, legacy=True)
        out = tmp_path / "tables"
        out.mkdir()
        path = table_monitoring_invariance(out, results_dir)
        df = pd.read_csv(path)
        t1 = df[df["Approach"].str.startswith("Provenance-free") & (df["Task"] == "T1")].iloc[0]
        assert t1["AUROC"] == "0.637"  # legacy back-compat

    def test_monitoring_comparison_uses_provenance_free(self, tmp_path: Path) -> None:
        from paper.generate_tables import table_monitoring_comparison

        results_dir = self._results_dir(tmp_path, pf=True, legacy=True)
        out = tmp_path / "tables"
        out.mkdir()
        table_monitoring_comparison(out, results_dir)
        df = pd.read_csv(out / "table_supp_monitoring_comparison.csv")
        t1 = df[df["Task"] == "T1"].iloc[0]
        assert float(t1["AUROC (with)"]) == pytest.approx(0.859)
        assert float(t1["AUROC (without)"]) == pytest.approx(0.820)  # PF wins over legacy


# The power table must fail loud on a missing/partial input, never
# emit a placeholder or zero-filled row (that is how a wrong power table shipped).


def test_table_power_analysis_fails_loud_on_missing_input(tmp_path: Path) -> None:
    from paper.generate_tables import table_power_analysis

    with pytest.raises(FileNotFoundError):
        table_power_analysis(tmp_path, results_dir=tmp_path)  # no power_analysis.json


def test_table_power_analysis_fails_loud_on_missing_field(tmp_path: Path) -> None:
    import json as _json

    from paper.generate_tables import table_power_analysis

    (tmp_path / "power_analysis.json").write_text(
        _json.dumps({"core_tests": {"delong_auroc": {"effect_size": 0.007, "alpha": 0.05}}})
    )  # 'power' field missing
    with pytest.raises(ValueError, match="missing"):
        table_power_analysis(tmp_path, results_dir=tmp_path)
