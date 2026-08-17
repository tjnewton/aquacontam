"""Tests for paper/verify_paper.py — paper consistency verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("click")

from paper.verify_paper import (
    check_figure_files,
    check_metrics,
    check_sequential,
    check_stale_frozen_files,
    check_supplementary_metrics,
    check_table2_rows,
    check_table3_invariance_rows,
    check_table_files,
    check_text_metric_reconciliation,
    extract_metrics_from_results,
    parse_figure_refs,
    parse_table_refs,
    verify_paper,
)


def _write_table2(tables_dir: Path, body_rows: str) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "table2_benchmark_condensed.md").write_text(
        "| Task | Model | AUROC | AUPRC |\n|---|---|---|---|\n" + body_rows,
        encoding="utf-8",
    )


class TestCheckTable2Rows:
    """Per-model Table 2 row validation against results.json."""

    def test_matching_rows_no_error(self, tmp_path: Path) -> None:
        results = [
            {
                "task": "T1",
                "model": "random_forest_classifier",
                "metrics": {"auroc": 0.8714, "auprc": 0.7896},
            },
            {
                "task": "T1",
                "model": "cnn1d_classifier",
                "metrics": {"auroc": 0.8686, "auprc": 0.7740},
            },
        ]
        (tmp_path / "results.json").write_text(json.dumps(results), encoding="utf-8")
        _write_table2(
            tmp_path / "tables",
            "| T1 | Random Forest | 0.871 (0.856-0.886) | 0.790 |\n| T1 | CNN1D | 0.869 | 0.774 |\n",
        )
        errors = check_table2_rows(tmp_path / "tables", tmp_path / "results.json")
        assert errors == []

    def test_stale_value_is_caught(self, tmp_path: Path) -> None:
        # Table claims Stacking AUROC 0.872 but results.json has 0.830.
        results = [
            {
                "task": "T1",
                "model": "stacking_ensemble",
                "metrics": {"auroc": 0.8300, "auprc": 0.7069},
            },
        ]
        (tmp_path / "results.json").write_text(json.dumps(results), encoding="utf-8")
        _write_table2(tmp_path / "tables", "| T1 | Stacking | 0.872 | 0.787 |\n")
        errors = check_table2_rows(tmp_path / "tables", tmp_path / "results.json")
        assert any("Stacking" in e and "AUROC" in e for e in errors)

    def test_missing_entry_is_caught(self, tmp_path: Path) -> None:
        (tmp_path / "results.json").write_text(json.dumps([]), encoding="utf-8")
        _write_table2(tmp_path / "tables", "| T1 | Random Forest | 0.871 | 0.790 |\n")
        errors = check_table2_rows(tmp_path / "tables", tmp_path / "results.json")
        assert any("no matching results.json entry" in e for e in errors)

    def test_missing_files_no_error(self, tmp_path: Path) -> None:
        # Absent files are handled as warnings elsewhere, not hard errors here.
        assert check_table2_rows(tmp_path / "tables", tmp_path / "results.json") == []


class TestParseFigureRefs:
    def test_parses_main_figures(self) -> None:
        text = "As shown in Fig. 1, the data... See also Fig. 3 and Figure 2."
        main, ext = parse_figure_refs(text)
        assert main == {1, 2, 3}
        assert ext == set()

    def test_parses_extended_figures(self) -> None:
        text = "Extended Data Fig. 1 shows... Extended Data Figure 2 confirms..."
        main, ext = parse_figure_refs(text)
        assert ext == {1, 2}
        # Extended refs should not appear in main
        assert main == set()

    def test_mixed_main_and_extended(self) -> None:
        text = "Fig. 1 and Extended Data Fig. 1 show complementary views."
        main, ext = parse_figure_refs(text)
        assert main == {1}
        assert ext == {1}


class TestParseTableRefs:
    def test_parses_main_tables(self) -> None:
        text = "Table 1 summarizes the data. See Table 2 for details."
        main, ext = parse_table_refs(text)
        assert main == {1, 2}
        assert ext == set()

    def test_parses_extended_tables(self) -> None:
        text = "Extended Data Table 1 lists all features."
        main, ext = parse_table_refs(text)
        assert ext == {1}
        assert main == set()


class TestCheckSequential:
    def test_no_gaps(self) -> None:
        assert check_sequential({1, 2, 3}, "Fig.") == []

    def test_detects_gap(self) -> None:
        warnings = check_sequential({1, 3}, "Fig.")
        assert len(warnings) == 1
        assert "Fig. 2" in warnings[0]

    def test_empty_set(self) -> None:
        assert check_sequential(set(), "Fig.") == []


class TestCheckFigureFiles:
    def test_finds_existing_figures(self, tmp_path: Path) -> None:
        (tmp_path / "fig1_overview.pdf").touch()
        (tmp_path / "fig2_data.png").touch()
        warnings = check_figure_files({1, 2}, tmp_path, "main")
        assert warnings == []

    def test_reports_missing_figures(self, tmp_path: Path) -> None:
        (tmp_path / "fig1_overview.pdf").touch()
        warnings = check_figure_files({1, 2}, tmp_path, "main")
        assert len(warnings) == 1
        assert "Fig. 2" in warnings[0]

    def test_extended_figures(self, tmp_path: Path) -> None:
        (tmp_path / "fig_ext1_correlation.pdf").touch()
        warnings = check_figure_files({1, 2}, tmp_path, "ext")
        assert len(warnings) == 1
        assert "Extended Data Fig. 2" in warnings[0]


class TestCheckTableFiles:
    def test_finds_existing_tables(self, tmp_path: Path) -> None:
        (tmp_path / "table1_summary.csv").touch()
        warnings = check_table_files({1}, tmp_path, "main")
        assert warnings == []

    def test_reports_missing_tables(self, tmp_path: Path) -> None:
        warnings = check_table_files({1}, tmp_path, "main")
        assert len(warnings) == 1
        assert "Table 1" in warnings[0]


class TestExtractMetrics:
    def test_independent_auroc_auprc_tracking(self, tmp_path: Path) -> None:
        """AUROC and AUPRC should be tracked independently across models."""
        results = [
            # MLP has best AUROC and best AUPRC
            {"task": "T1", "model": "mlp", "metrics": {"auroc": 0.833, "auprc": 0.684}},
            # TabPFN has lower AUROC and lower AUPRC
            {"task": "T1", "model": "tabpfn", "metrics": {"auroc": 0.792, "auprc": 0.548}},
        ]
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(results))
        metrics = extract_metrics_from_results(results_path)
        assert abs(metrics["T1"]["auroc"] - 0.833) < 0.001  # MLP
        assert abs(metrics["T1"]["auprc"] - 0.684) < 0.001  # MLP


class TestCheckMetrics:
    def test_matching_metrics(self, tmp_path: Path) -> None:
        results = [
            {"task": "T1", "model": "lightgbm", "metrics": {"auroc": 0.853, "auprc": 0.767}},
            {"task": "T1", "model": "xgboost", "metrics": {"auroc": 0.852, "auprc": 0.764}},
            {"task": "T2", "model": "lightgbm", "metrics": {"rmse": 0.157}},
            {"task": "T4", "model": "voting", "metrics": {"auroc": 0.704, "auprc": 0.403}},
        ]
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(results))
        warnings = check_metrics(results_path)
        assert warnings == []

    def test_detects_metric_mismatch(self, tmp_path: Path) -> None:
        results = [
            {"task": "T1", "model": "xgb", "metrics": {"auroc": 0.600, "auprc": 0.100}},
        ]
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(results))
        warnings = check_metrics(results_path)
        assert any("T1 auroc" in w for w in warnings)

    def test_missing_results_file(self, tmp_path: Path) -> None:
        warnings = check_metrics(tmp_path / "nonexistent.json")
        assert any("No results found" in w for w in warnings)


class TestVerifyPaper:
    def test_minimal_valid_skeleton(self, tmp_path: Path) -> None:
        # Create minimal skeleton
        skeleton = tmp_path / "skeleton.md"
        skeleton.write_text("Fig. 1 shows the overview. Table 1 has details.")

        # Create paper directory structure
        paper_dir = tmp_path / "paper"
        (paper_dir / "figures").mkdir(parents=True)
        (paper_dir / "tables").mkdir(parents=True)
        (paper_dir / "figures" / "fig1_overview.pdf").touch()
        (paper_dir / "tables" / "table1_summary.csv").touch()

        # results.json must exist or the hard results-dir gate fires.
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        (out_dir / "results.json").write_text("[]")

        import os

        orig = os.getcwd()
        try:
            os.chdir(tmp_path)
            errors, _warnings = verify_paper(skeleton, out_dir)
        finally:
            os.chdir(orig)
        assert len(errors) == 0

    def test_missing_results_json_is_hard_error(self, tmp_path: Path) -> None:
        skeleton = tmp_path / "skeleton.md"
        skeleton.write_text("Fig. 1 shows the overview. Table 1 has details.")
        (tmp_path / "paper" / "figures").mkdir(parents=True)
        (tmp_path / "paper" / "tables").mkdir(parents=True)
        errors, _warnings = verify_paper(skeleton, tmp_path / "no_results")
        assert any("results.json not found" in e for e in errors)

    def test_missing_skeleton(self, tmp_path: Path) -> None:
        errors, _warnings = verify_paper(tmp_path / "nonexistent.md")
        assert len(errors) == 1
        assert "not found" in errors[0]


class TestCheckSupplementaryMetrics:
    """Tests for check_supplementary_metrics cross-validation."""

    def test_detects_stale_deep_tobit_auroc(self, tmp_path: Path) -> None:
        supp = tmp_path / "supplementary_information.md"
        supp.write_text("the model's weaker T1 AUROC (0.571; Table 2)")
        results = [
            {"task": "T1", "model": "deep_tobit_classifier", "metrics": {"auroc": 0.249}},
        ]
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(results))
        warnings = check_supplementary_metrics(supp, results_path)
        assert len(warnings) == 1
        assert "S13" in warnings[0]
        assert "0.571" in warnings[0]
        assert "0.249" in warnings[0]

    def test_no_warning_when_values_match(self, tmp_path: Path) -> None:
        supp = tmp_path / "supplementary_information.md"
        supp.write_text("the model's weaker T1 AUROC (0.249; Table 2)")
        results = [
            {"task": "T1", "model": "deep_tobit_classifier", "metrics": {"auroc": 0.249}},
        ]
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(results))
        warnings = check_supplementary_metrics(supp, results_path)
        assert warnings == []

    def test_missing_supplementary_file(self, tmp_path: Path) -> None:
        results_path = tmp_path / "results.json"
        results_path.write_text("[]")
        warnings = check_supplementary_metrics(tmp_path / "missing.md", results_path)
        assert len(warnings) == 1
        assert "not found" in warnings[0]


def _write_invariance_csv(tables_dir: Path, t1_auroc: str, t1_auprc: str) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        "Approach,Task,AUROC,AUPRC,Notes",
        "Full model (XGBoost),T1,0.859,0.731,All features",
        f"Provenance-free (XGBoost),T1,{t1_auroc},{t1_auprc},Provenance features dropped (environment-only)",
        "Provenance-free (XGBoost),T4,0.545,0.253,Provenance features dropped (environment-only)",
        "ICP,T1,0.777,0.579,Adversarially invariant",
        "DML adjusted,T1,0.727,—,Post-hoc adjusted association",
    ]
    (tables_dir / "table_supp11_monitoring_invariance.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def _write_pf_results(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "task": "T1",
            "model": "xgboost_classifier",
            "metrics": {"auroc": 0.8203, "auprc": 0.7496},
        },
        {
            "task": "T4",
            "model": "xgboost_classifier",
            "metrics": {"auroc": 0.5445, "auprc": 0.2525},
        },
    ]
    (results_dir / "results_provenance_free.json").write_text(
        json.dumps(entries), encoding="utf-8"
    )


class TestCheckMonitoringInvarianceRows:
    """Decomposition-table provenance-free rows vs the frozen env-only run."""

    def test_matching_rows_no_error(self, tmp_path: Path) -> None:
        from paper.verify_paper import check_monitoring_invariance_rows

        _write_pf_results(tmp_path)
        _write_invariance_csv(tmp_path / "tables", "0.820", "0.750")
        errors = check_monitoring_invariance_rows(tmp_path / "tables", tmp_path)
        assert errors == []

    def test_mismatched_cell_is_caught(self, tmp_path: Path) -> None:
        from paper.verify_paper import check_monitoring_invariance_rows

        _write_pf_results(tmp_path)
        _write_invariance_csv(tmp_path / "tables", "0.700", "0.750")
        errors = check_monitoring_invariance_rows(tmp_path / "tables", tmp_path)
        assert any("T1" in e and "AUROC" in e for e in errors)

    def test_dash_cells_with_pf_results_is_caught(self, tmp_path: Path) -> None:
        # The exact regression: snapshot frozen before the provenance-free run
        # existed -> the decomposition table rendered em-dashes.
        from paper.verify_paper import check_monitoring_invariance_rows

        _write_pf_results(tmp_path)
        _write_invariance_csv(tmp_path / "tables", "—", "—")
        errors = check_monitoring_invariance_rows(tmp_path / "tables", tmp_path)
        assert any("regenerate tables" in e for e in errors)

    def test_missing_files_no_error(self, tmp_path: Path) -> None:
        from paper.verify_paper import check_monitoring_invariance_rows

        assert check_monitoring_invariance_rows(tmp_path / "tables", tmp_path) == []


# ---------------------------------------------------------------------------
# Ratchet checks added for the Nature Water final-gate revision
# ---------------------------------------------------------------------------


def _write_results_json(
    results_dir: Path, *, icp_t4_auroc: float = 0.607, xgb_t4_auroc: float = 0.677
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    entries = [
        {"task": "T1", "model": "xgboost_classifier", "metrics": {"auroc": 0.859, "auprc": 0.731}},
        {
            "task": "T4",
            "model": "xgboost_classifier",
            "metrics": {"auroc": xgb_t4_auroc, "auprc": 0.417},
        },
        {"task": "T1", "model": "icp_classifier", "metrics": {"auroc": 0.717, "auprc": 0.536}},
        {
            "task": "T4",
            "model": "icp_classifier",
            "metrics": {"auroc": icp_t4_auroc, "auprc": 0.297},
        },
    ]
    (results_dir / "results.json").write_text(json.dumps(entries))


def _write_table3(tables_dir: Path, icp_t4_auroc: str, icp_t4_auprc: str) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Approach | Task | AUROC | AUPRC | Notes |",
        "|---|---|---|---|---|",
        "| Full model (XGBoost) | T1 | 0.859 | 0.731 | All features |",
        f"| ICP | T4 | {icp_t4_auroc} | {icp_t4_auprc} | inv |",
    ]
    (tables_dir / "table3_monitoring_invariance.md").write_text("\n".join(lines), encoding="utf-8")


class TestCheckTable3InvarianceRows:
    def test_stale_icp_t4_is_caught(self, tmp_path: Path) -> None:
        _write_results_json(tmp_path, icp_t4_auroc=0.607)
        _write_table3(tmp_path / "tables", "0.623", "0.323")  # stale icp_diagnostics value
        errors = check_table3_invariance_rows(tmp_path / "tables", tmp_path)
        assert any("ICP" in e and "T4" in e for e in errors)

    def test_matching_icp_t4_no_error(self, tmp_path: Path) -> None:
        _write_results_json(tmp_path, icp_t4_auroc=0.607)
        _write_table3(tmp_path / "tables", "0.607", "0.297")
        assert check_table3_invariance_rows(tmp_path / "tables", tmp_path) == []

    def test_missing_table_no_error(self, tmp_path: Path) -> None:
        _write_results_json(tmp_path)
        assert check_table3_invariance_rows(tmp_path / "tables", tmp_path) == []


class TestCheckStaleFrozenFiles:
    def test_stale_t4_file_warns(self, tmp_path: Path) -> None:
        _write_results_json(tmp_path, xgb_t4_auroc=0.677)
        (tmp_path / "spatial_block_bootstrap.json").write_text(
            json.dumps({"T4_xgboost": {"weighted_mean_auroc": 0.9684}})
        )
        warnings = check_stale_frozen_files(tmp_path)
        assert any("spatial_block_bootstrap.json" in w for w in warnings)

    def test_canonical_value_no_warning(self, tmp_path: Path) -> None:
        _write_results_json(tmp_path, xgb_t4_auroc=0.677)
        (tmp_path / "some_T4_metrics.json").write_text(
            json.dumps({"T4_xgboost": {"weighted_mean_auroc": 0.677}})
        )
        assert check_stale_frozen_files(tmp_path) == []


class TestCheckTextMetricReconciliation:
    def test_stale_icp_in_prose_is_caught(self, tmp_path: Path) -> None:
        _write_results_json(tmp_path, icp_t4_auroc=0.607)
        prose = "the monitoring-invariant ICP model 0.623. Lead sampling is, however,"
        errors = check_text_metric_reconciliation(prose, tmp_path)
        assert any("0.623" in e for e in errors)

    def test_matching_icp_in_prose_no_error(self, tmp_path: Path) -> None:
        _write_results_json(tmp_path, icp_t4_auroc=0.607)
        prose = "the monitoring-invariant ICP model 0.607. Lead sampling is, however,"
        assert check_text_metric_reconciliation(prose, tmp_path) == []

    def test_historical_literal_not_flagged(self, tmp_path: Path) -> None:
        _write_results_json(tmp_path, icp_t4_auroc=0.607)
        prose = "appearing to reach AUROC 0.962 ... corrected skill 0.677."
        assert check_text_metric_reconciliation(prose, tmp_path) == []

    def test_m4_fnr_ece_reconciliation(self, tmp_path: Path) -> None:
        _write_results_json(tmp_path)
        (tmp_path / "group_error_calibration.json").write_text(
            json.dumps(
                {
                    "error_rates": {
                        "pct_people_of_color": {
                            "high": {"fnr": 0.253, "ece": 0.177},
                            "low": {"fnr": 0.156, "ece": 0.059},
                        }
                    }
                }
            )
        )
        good = (
            "false-negative rate 0.25 vs 0.16 and is more poorly calibrated "
            "(expected calibration error 0.18 vs 0.06)"
        )
        assert check_text_metric_reconciliation(good, tmp_path) == []
        bad = "false-negative rate 0.40 vs 0.16 for high-share systems"
        assert any("0.40" in e for e in check_text_metric_reconciliation(bad, tmp_path))
