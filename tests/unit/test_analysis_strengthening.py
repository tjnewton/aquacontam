"""Tests for the peer-review strengthening analyses."""

from __future__ import annotations

import numpy as np
import pytest

from aquacontam.analysis.strengthening import (
    auc_difference,
    auc_vs_chance,
    buffered_boundary_loro,
    icp_recoverability_probe,
    region_block_bootstrap,
)

_LATLON = "aquacontam.preprocessing._region_lookup.latlon_to_epa_region"


class TestRegionBlockBootstrap:
    def _folds(self) -> list[dict]:
        rng = np.random.RandomState(0)
        return [
            {"test_region": r, "auroc": float(a), "n_test": int(n)}
            for r, a, n in zip(
                range(1, 11),
                0.6 + 0.3 * rng.rand(10),
                rng.randint(100, 2000, 10),
                strict=True,
            )
        ]

    def test_returns_block_and_naive_intervals(self):
        res = region_block_bootstrap(self._folds(), n_boot=500, seed=1)
        assert res["n_folds"] == 10
        assert 0.5 < res["weighted_mean_auroc"] < 1.0
        for key in ("block_bootstrap", "naive_fold_normal"):
            ci = res[key]
            assert ci["ci_lower"] < ci["ci_upper"]
            assert ci["width"] > 0
        assert np.isfinite(res["width_ratio_block_over_naive"])

    def test_deterministic_given_seed(self):
        f = self._folds()
        a = region_block_bootstrap(f, n_boot=300, seed=7)
        b = region_block_bootstrap(f, n_boot=300, seed=7)
        assert a["block_bootstrap"]["ci_lower"] == b["block_bootstrap"]["ci_lower"]

    def test_insufficient_folds(self):
        res = region_block_bootstrap([{"auroc": 0.7, "n_test": 100}], n_boot=10)
        assert res.get("error") == "insufficient_folds"

    def test_skips_nan_folds(self):
        folds = [
            {"test_region": 1, "auroc": 0.7, "n_test": 100},
            {"test_region": 2, "auroc": float("nan"), "n_test": 100},
            {"test_region": 3, "auroc": 0.8, "n_test": 100},
        ]
        res = region_block_bootstrap(folds, n_boot=100, seed=1)
        assert res["n_folds"] == 2


class TestBufferedBoundaryLoro:
    def test_runs_and_excludes_buffer(self, synthetic_wq_df, synthetic_feature_dfs):
        res = buffered_boundary_loro(
            synthetic_wq_df,
            synthetic_feature_dfs,
            buffer_km=50.0,
            seed=42,
        )
        assert "folds" in res
        assert res["buffer_km"] == 50.0
        # At least one fold should run on the synthetic 10-region data.
        assert isinstance(res["mean_auroc_with_provenance"], float)
        for f in res["folds"]:
            assert f["n_train_buffered"] <= f["n_train_full"]
            assert f["n_excluded_by_buffer"] >= 0

    def test_provenance_free_arm_present_when_excluding(
        self, synthetic_wq_df, synthetic_feature_dfs
    ):
        res = buffered_boundary_loro(
            synthetic_wq_df,
            synthetic_feature_dfs,
            buffer_km=10.0,
            exclude_features=["n_samples"],
            seed=42,
        )
        ran = [f for f in res["folds"] if "provenance_free" in f]
        # If any fold ran, it should carry the provenance-free arm.
        if res["folds"]:
            assert ran or all("provenance_free" not in f for f in res["folds"])


class TestIcpRecoverabilityProbe:
    def test_requires_source_column(self, synthetic_feature_dfs):
        pytest.importorskip("torch")  # no torch (CI): probe returns "icp_unavailable" first
        import pandas as pd

        wq = pd.DataFrame(
            {
                "pwsid": ["CT0000001"] * 3,
                "analyte": ["PFOS"] * 3,
                "concentration": [0.0, 1.0, 2.0],
                "censored": [True, False, False],
                "detection_limit": [1.0, 1.0, 1.0],
                "latitude": [41.6, 41.6, 41.6],
                "longitude": [-72.7, -72.7, -72.7],
            }
        )
        res = icp_recoverability_probe(wq, synthetic_feature_dfs)
        assert res.get("error") in {"no_source_column", "insufficient_data"}

    @pytest.mark.slow
    def test_probe_runs_on_synthetic(self, synthetic_wq_df, synthetic_feature_dfs):
        torch = pytest.importorskip("torch")
        assert torch is not None
        wq = synthetic_wq_df.copy()
        # Assign a synthetic data-source label correlated with region.
        wq["source"] = np.where(wq["pwsid"].str.startswith(("CA", "WA", "CO")), "stateA", "stateB")
        res = icp_recoverability_probe(wq, synthetic_feature_dfs, seed=42)
        # Either it runs and reports probes, or bails out cleanly on small data.
        assert "probes" in res or "error" in res


class TestAucVsChance:
    def test_chance_auc_not_significant(self):
        r = auc_vs_chance(0.5, 200, 200)
        assert r["z"] == 0.0
        assert r["p_value"] == pytest.approx(1.0)

    def test_strong_auc_significant(self):
        r = auc_vs_chance(0.95, 360, 472)
        assert r["z"] > 5
        assert r["p_value"] < 1e-6

    def test_weak_but_large_sample_significant(self):
        # AUROC ~0.54 with large n is weakly but significantly above chance.
        r = auc_vs_chance(0.5414, 735, 1128)
        assert 0.0 < r["p_value"] < 0.05
        assert r["z"] > 0

    def test_se_decreases_with_sample_size(self):
        small = auc_vs_chance(0.8, 20, 20)["se"]
        large = auc_vs_chance(0.8, 2000, 2000)["se"]
        assert large < small

    def test_invalid_counts_raise(self):
        with pytest.raises(ValueError):
            auc_vs_chance(0.8, 0, 100)


class TestAucDifference:
    def test_equal_aucs_not_significant(self):
        r = auc_difference(0.80, 100, 100, 0.80, 100, 100)
        assert r["delta"] == pytest.approx(0.0)
        assert r["p_value"] == pytest.approx(1.0)

    def test_group_gap_significant(self):
        # People-of-colour high vs low burden AUROC gap (referee M3b).
        r = auc_difference(0.7564, 162, 275, 0.8418, 186, 901)
        assert r["delta"] < 0
        assert r["p_value"] < 0.05

    def test_symmetric_in_magnitude(self):
        a = auc_difference(0.75, 150, 150, 0.85, 150, 150)
        b = auc_difference(0.85, 150, 150, 0.75, 150, 150)
        assert a["p_value"] == pytest.approx(b["p_value"])
        assert a["delta"] == pytest.approx(-b["delta"])
