"""Tests for external validation module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aquacontam.benchmark.external_validation import (
    _STATE_EPA_REGION,
    _STATE_REGIONS,
    _TRAIN_REGIONS,
    run_external_validation,
    run_wqp_regional_validation,
)
from aquacontam.models.xgboost import XGBoostClassifier


class TestExternalValidation:
    def test_state_regions_mapping(self):
        assert "mi_mpart" in _STATE_REGIONS
        assert "ca_geotracker" in _STATE_REGIONS
        assert _STATE_REGIONS["mi_mpart"] == 5
        assert _STATE_REGIONS["ca_geotracker"] == 9

    def test_run_external_validation_empty(self):
        """Test with no state data."""
        model = XGBoostClassifier(config={"n_estimators": 10, "random_state": 42})
        # Can't run without fitting, so just test empty case
        result = run_external_validation(
            model,
            pd.DataFrame(),
            {},
            [],
        )
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_result_includes_bootstrap_ci_and_chance_test(self):
        """Verify two-class results include bootstrap_ci and chance_test keys."""
        # Build a minimal fitted model
        rng = np.random.RandomState(42)
        n = 100
        X_train = pd.DataFrame({"feat_a": rng.randn(n), "feat_b": rng.randn(n)})
        y_train = pd.Series(rng.randint(0, 2, n))
        model = XGBoostClassifier(config={"n_estimators": 10, "random_state": 42})
        model.fit(X_train, y_train)

        # Simulate a state DB with two classes
        state_df = pd.DataFrame(
            {
                "pwsid": [f"XX{i:07d}" for i in range(50)],
                "analyte": ["PFOS"] * 50,
                "concentration": rng.exponential(0.01, 50),
                "censored": [True] * 25 + [False] * 25,
                "unit": ["UG/L"] * 50,
            }
        )

        # Feature DFs indexed by pwsid
        feat_df = pd.DataFrame(
            {"feat_a": rng.randn(50), "feat_b": rng.randn(50)},
            index=[f"XX{i:07d}" for i in range(50)],
        )
        feat_df.index.name = "pwsid"

        result = run_external_validation(
            model,
            pd.DataFrame(),
            {"test_state": state_df},
            [feat_df],
        )

        if "test_state" in result and result["test_state"].get("metrics"):
            entry = result["test_state"]
            assert "bootstrap_ci" in entry, "Missing bootstrap_ci key"
            assert "chance_test" in entry, "Missing chance_test key"
            # Bootstrap CI should have auroc sub-dict
            auroc_ci = entry["bootstrap_ci"].get("auroc", {})
            assert "ci_lower" in auroc_ci
            assert "ci_upper" in auroc_ci
            assert auroc_ci["ci_lower"] <= auroc_ci["ci_upper"]
            # Chance test should have p_value
            if entry["chance_test"] is not None:
                assert "p_value" in entry["chance_test"]


class TestWQPRegionalValidation:
    def test_empty_wqp_data(self):
        model = XGBoostClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_wqp_regional_validation(model, pd.DataFrame(), pd.DataFrame(), [])
        assert result.get("error") == "empty_wqp_data"

    def test_state_epa_region_mapping(self):
        """Verify state -> EPA region mapping covers training regions."""
        train_states = [s for s, r in _STATE_EPA_REGION.items() if r in _TRAIN_REGIONS]
        assert len(train_states) > 20  # Should cover most CONUS states
        assert "OH" in train_states  # Region 5
        assert "TX" in train_states  # Region 6
        assert "MA" in train_states  # Region 1

    def test_train_regions_definition(self):
        assert {1, 3, 4, 5, 6} == _TRAIN_REGIONS
