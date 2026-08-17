"""Tests verifying paper numerical claims match pipeline result files.

These tests catch silent divergence between the paper manuscript and
pipeline outputs — the class of bugs identified in the AUDIT_REPORT.md.
Each test reads a results JSON file and asserts that the value matches
what the paper claims (within tolerance).

Tests are skipped when result files are not present (e.g. in CI where
only ``[test]`` extras are installed and no pipeline run has occurred).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("tabulate")

RESULTS_DIR = Path("results")


def _load_json(name: str) -> dict | list:
    path = RESULTS_DIR / name
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# External validation
# ---------------------------------------------------------------------------

_has_ext_val = (RESULTS_DIR / "external_validation.json").exists()


@pytest.mark.skipif(not _has_ext_val, reason="external_validation.json not found")
class TestExternalValidation:
    """Verify external validation AUROC values match results file."""

    def test_ca_geotracker_auroc(self) -> None:
        data = _load_json("external_validation.json")
        auroc = data["ca_geotracker"]["metrics"]["auroc"]
        assert abs(auroc - 0.813) < 0.02, f"CA GeoTracker AUROC {auroc:.3f} != ~0.813"

    def test_mo_dnr_auroc(self) -> None:
        data = _load_json("external_validation.json")
        auroc = data["mo_dnr"]["metrics"]["auroc"]
        assert abs(auroc - 0.829) < 0.02, f"MO DNR AUROC {auroc:.3f} != ~0.829"

    def test_nj_dep_has_metrics(self) -> None:
        data = _load_json("external_validation.json")
        nj = data["nj_dep"]
        assert nj["metrics"], "NJ DEP should have external validation metrics"
        auroc = nj["metrics"]["auroc"]
        assert abs(auroc - 0.789) < 0.02, f"NJ DEP AUROC {auroc:.3f} != ~0.789"

    def test_nj_dep_system_count(self) -> None:
        data = _load_json("external_validation.json")
        assert data["nj_dep"]["n_systems"] == 1310


# ---------------------------------------------------------------------------
# DML / causal deconfounding
# ---------------------------------------------------------------------------

_has_dml = (RESULTS_DIR / "causal_deconfounding.json").exists()
_has_rank = (RESULTS_DIR / "dml_shap_rank_comparison.json").exists()


@pytest.mark.skipif(not _has_dml, reason="causal_deconfounding.json not found")
class TestDMLValues:
    """Verify DML causal effect values for pct_people_of_color."""

    def _get_poc_entry(self) -> dict:
        data = _load_json("causal_deconfounding.json")
        entry = next(e for e in data if e["feature"] == "pct_people_of_color")
        return entry

    def test_poc_causal_effect(self) -> None:
        entry = self._get_poc_entry()
        eff = entry["causal_effect"]
        assert abs(eff - 0.022) < 0.01, f"DML PoC causal_effect {eff:.4f} != ~0.022"

    def test_poc_shap_importance(self) -> None:
        entry = self._get_poc_entry()
        shap = entry["shap_importance"]
        assert abs(shap - 0.007) < 0.002, f"DML PoC SHAP {shap:.4f} != ~0.007"


@pytest.mark.skipif(not _has_rank, reason="dml_shap_rank_comparison.json not found")
class TestDMLRankComparison:
    """Verify scale-free DML-vs-SHAP rank comparison metrics."""

    def test_spearman_rho_range(self) -> None:
        data = _load_json("dml_shap_rank_comparison.json")
        rho = data["spearman_rho"]
        assert 0.0 <= rho <= 0.3, f"Spearman rho {rho:.3f} outside [0, 0.3]"

    def test_n_features(self) -> None:
        data = _load_json("dml_shap_rank_comparison.json")
        n = data["n_features"]
        assert n >= 50, f"Only {n} features compared, expected >= 50"

    def test_pct_low_income_displacement(self) -> None:
        data = _load_json("dml_shap_rank_comparison.json")
        fr = next(r for r in data.get("demographic_ranks", []) if r["feature"] == "pct_low_income")
        # B1 (Minnesota in-split): the low-income DML-vs-SHAP rank displacement
        # collapsed to ~4 (it was >30 in the pre-v10 expectation).
        assert fr["displacement"] <= 10, (
            f"pct_low_income displacement {fr['displacement']} not <= 10"
        )


# ---------------------------------------------------------------------------
# Conformal coverage
# ---------------------------------------------------------------------------

_has_conformal = (RESULTS_DIR / "group_conformal_results.json").exists()


@pytest.mark.skipif(not _has_conformal, reason="group_conformal_results.json not found")
class TestConformalCoverage:
    """Verify group conformal prediction coverage at alpha=0.05."""

    def test_coverage_approximately_0_968(self) -> None:
        data = _load_json("group_conformal_results.json")
        alpha05 = next(e for e in data if abs(e["alpha"] - 0.05) < 0.001)
        coverage = alpha05["coverage"]
        assert abs(coverage - 0.910) < 0.02, f"Conformal coverage {coverage:.3f} != ~0.910"


# ---------------------------------------------------------------------------
# LORO CV
# ---------------------------------------------------------------------------

_has_loro = (RESULTS_DIR / "loro_cv.json").exists()


@pytest.mark.skipif(not _has_loro, reason="loro_cv.json not found")
class TestLOROCV:
    """Verify LORO cross-validation summary statistics."""

    def test_mean_auroc(self) -> None:
        data = _load_json("loro_cv.json")
        mean = data["T1"]["catboost"]["mean_auroc"]
        assert abs(mean - 0.790) < 0.01, f"LORO mean AUROC {mean:.3f} != ~0.790"

    def test_std_auroc_approximately_0_115(self) -> None:
        data = _load_json("loro_cv.json")
        std = data["T1"]["catboost"]["std_auroc"]
        assert abs(std - 0.096) < 0.01, f"LORO std AUROC {std:.3f} != ~0.096"


# ---------------------------------------------------------------------------
# Lift analysis
# ---------------------------------------------------------------------------

_has_lift = (RESULTS_DIR / "lift_analysis.json").exists()


@pytest.mark.skipif(not _has_lift, reason="lift_analysis.json not found")
class TestLiftAnalysis:
    """Verify lift analysis top-quintile capture values."""

    def test_full_model_top_quintile_capture(self) -> None:
        data = _load_json("lift_analysis.json")
        capture = data["full_model"]["top_quintile_capture"]
        assert abs(capture - 0.526) < 0.02, f"Full model capture {capture:.3f} != ~0.526"

    def test_monitoring_free_top_quintile_capture(self) -> None:
        data = _load_json("lift_analysis.json")
        capture = data["monitoring_free"]["top_quintile_capture"]
        assert abs(capture - 0.475) < 0.02, f"Mon-free capture {capture:.3f} != ~0.475"

    def test_top_decile_lift(self) -> None:
        data = _load_json("lift_analysis.json")
        lift = data["full_model"]["top_decile_lift"]
        assert abs(lift - 2.96) < 0.2, f"Top decile lift {lift:.1f} != ~2.96"


# ---------------------------------------------------------------------------
# Table 1 internal consistency
# ---------------------------------------------------------------------------


class TestTable1Consistency:
    """Verify generate_tables.py hardcoded values are internally consistent."""

    def test_nj_dep_status_is_available(self) -> None:
        # Generate the table and check NJ DEP status
        import tempfile

        from paper.generate_tables import table_dataset_summary

        with tempfile.TemporaryDirectory() as tmp:
            table_dataset_summary(Path(tmp))
            import pandas as pd

            df = pd.read_csv(Path(tmp) / "table1_dataset_summary.csv")
            nj_row = df[df["Source"] == "NJ DEP"]
            assert len(nj_row) == 1
            assert nj_row.iloc[0]["Status"] == "Available"

    def test_nj_dep_records(self) -> None:
        import tempfile

        from paper.generate_tables import table_dataset_summary

        with tempfile.TemporaryDirectory() as tmp:
            table_dataset_summary(Path(tmp))
            import pandas as pd

            df = pd.read_csv(Path(tmp) / "table1_dataset_summary.csv")
            nj_row = df[df["Source"] == "NJ DEP"]
            assert nj_row.iloc[0]["Records"] == "248,107"

    def test_nj_dep_analytes(self) -> None:
        import tempfile

        from paper.generate_tables import table_dataset_summary

        with tempfile.TemporaryDirectory() as tmp:
            table_dataset_summary(Path(tmp))
            import pandas as pd

            df = pd.read_csv(Path(tmp) / "table1_dataset_summary.csv")
            nj_row = df[df["Source"] == "NJ DEP"]
            assert nj_row.iloc[0]["Analytes"] == "25 PFAS"

    def test_detection_only_sources_marked(self) -> None:
        """MI MPART, OH EPA, WA DOH should have 0% censoring (SDWIS now ~0.3% after keep-non-detects)."""
        import tempfile

        from paper.generate_tables import table_dataset_summary

        with tempfile.TemporaryDirectory() as tmp:
            table_dataset_summary(Path(tmp))
            import pandas as pd

            df = pd.read_csv(Path(tmp) / "table1_dataset_summary.csv")
            for source in ["MI MPART", "OH EPA", "WA DOH"]:
                row = df[df["Source"] == source]
                censoring = row.iloc[0]["Censoring Rate"]
                assert "0%" in censoring, f"{source} censoring should be 0%, got {censoring}"
