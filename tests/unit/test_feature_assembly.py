"""Tests for feature matrix assembly pipeline."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from aquacontam.features.assembly import (
    PROVENANCE_FREE_EXCLUDE,
    aggregate_to_system_level,
    assemble_feature_matrix,
    build_system_geodataframe,
    prepare_train_val_test,
    resolve_excluded_columns,
)


class TestAggregateToSystemLevel:
    """Tests for aggregate_to_system_level."""

    def test_basic_aggregation(self) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["SYS001", "SYS001", "SYS002", "SYS002"],
                "analyte": ["PFOS", "PFOS", "PFOS", "PFOS"],
                "concentration": [10.0, 0.0, 0.0, 0.0],
                "censored": [False, True, True, True],
                "detection_limit": [2.0, 2.0, 3.0, 3.0],
            }
        )
        result = aggregate_to_system_level(df, "PFOS")

        assert len(result) == 2
        assert bool(result.loc["SYS001", "any_detected"]) is True
        assert bool(result.loc["SYS002", "any_detected"]) is False
        assert result.loc["SYS001", "n_samples"] == 2
        assert result.loc["SYS001", "detection_rate"] == pytest.approx(0.5)

    def test_detected_target(self) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["A", "A", "B"],
                "analyte": ["PFOS", "PFOS", "PFOS"],
                "concentration": [5.0, 0.0, 0.0],
                "censored": [False, True, True],
                "detection_limit": [1.0, 1.0, 1.0],
            }
        )
        result = aggregate_to_system_level(df, "PFOS", target="detected")
        assert result.loc["A", "target"] == 1
        assert result.loc["B", "target"] == 0

    def test_max_concentration_target(self) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["A", "A"],
                "analyte": ["PFOS", "PFOS"],
                "concentration": [5.0, 10.0],
                "censored": [False, False],
                "detection_limit": [1.0, 1.0],
            }
        )
        result = aggregate_to_system_level(df, "PFOS", target="max_concentration")
        assert result.loc["A", "target"] == pytest.approx(10.0)

    def test_analyte_filter(self) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["A", "A"],
                "analyte": ["PFOS", "PFOA"],
                "concentration": [10.0, 5.0],
                "censored": [False, False],
                "detection_limit": [1.0, 1.0],
            }
        )
        result = aggregate_to_system_level(df, "PFOS")
        assert len(result) == 1

    def test_empty_analyte(self) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["A"],
                "analyte": ["PFOS"],
                "concentration": [1.0],
                "censored": [False],
                "detection_limit": [0.5],
            }
        )
        result = aggregate_to_system_level(df, "nonexistent")
        assert result.empty

    def test_invalid_target(self) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["A"],
                "analyte": ["PFOS"],
                "concentration": [1.0],
                "censored": [False],
                "detection_limit": [0.5],
            }
        )
        with pytest.raises(ValueError, match="target must be one of"):
            aggregate_to_system_level(df, "PFOS", target="invalid")

    def test_missing_columns(self) -> None:
        df = pd.DataFrame({"pwsid": ["A"]})
        with pytest.raises(ValueError, match="Missing columns"):
            aggregate_to_system_level(df, "PFOS")

    def test_all_censored_system(self) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["A", "A", "A"],
                "analyte": ["PFOS", "PFOS", "PFOS"],
                "concentration": [0.0, 0.0, 0.0],
                "censored": [True, True, True],
                "detection_limit": [2.0, 2.0, 2.0],
            }
        )
        result = aggregate_to_system_level(df, "PFOS")
        assert result.loc["A", "detection_rate"] == pytest.approx(0.0)
        assert bool(result.loc["A", "any_detected"]) is False

    def test_action_level_target_lead(self) -> None:
        """Action level target: lead >= 15 µg/L."""
        df = pd.DataFrame(
            {
                "pwsid": ["A", "A", "B", "B"],
                "analyte": ["lead", "lead", "lead", "lead"],
                "concentration": [20.0, 10.0, 5.0, 3.0],
                "censored": [False, False, False, False],
                "detection_limit": [0.5, 0.5, 0.5, 0.5],
            }
        )
        result = aggregate_to_system_level(df, "lead", target="action_level")
        # System A: max=20 >= 15 → 1
        assert result.loc["A", "target"] == 1
        # System B: max=5 < 15 → 0
        assert result.loc["B", "target"] == 0

    def test_action_level_target_copper(self) -> None:
        """Action level target: copper >= 1300 µg/L."""
        df = pd.DataFrame(
            {
                "pwsid": ["A", "B"],
                "analyte": ["copper", "copper"],
                "concentration": [1500.0, 800.0],
                "censored": [False, False],
                "detection_limit": [1.0, 1.0],
            }
        )
        result = aggregate_to_system_level(df, "copper", target="action_level")
        assert result.loc["A", "target"] == 1
        assert result.loc["B", "target"] == 0

    def test_action_level_unknown_analyte_raises(self) -> None:
        """Action level target for analyte without defined level raises ValueError."""
        df = pd.DataFrame(
            {
                "pwsid": ["A"],
                "analyte": ["PFOS"],
                "concentration": [10.0],
                "censored": [False],
                "detection_limit": [1.0],
            }
        )
        with pytest.raises(ValueError, match="No action level defined"):
            aggregate_to_system_level(df, "PFOS", target="action_level")


class TestBuildSystemGeoDataFrame:
    """Tests for build_system_geodataframe."""

    def test_basic_geodataframe(self) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["A", "A", "B"],
                "latitude": [40.0, 40.1, 35.0],
                "longitude": [-74.0, -74.1, -118.0],
            }
        )
        gdf = build_system_geodataframe(df)
        assert len(gdf) == 2
        assert gdf.crs.to_epsg() == 4326
        assert "A" in gdf.index

    def test_filters_missing_coords(self) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["A", "B"],
                "latitude": [40.0, np.nan],
                "longitude": [-74.0, np.nan],
            }
        )
        gdf = build_system_geodataframe(df)
        assert len(gdf) == 1

    def test_empty_input(self) -> None:
        df = pd.DataFrame(
            {
                "pwsid": pd.Series(dtype=str),
                "latitude": pd.Series(dtype=float),
                "longitude": pd.Series(dtype=float),
            }
        )
        gdf = build_system_geodataframe(df)
        assert gdf.empty

    def test_all_nan_coords(self) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["A", "B"],
                "latitude": [np.nan, np.nan],
                "longitude": [np.nan, np.nan],
            }
        )
        gdf = build_system_geodataframe(df)
        assert gdf.empty

    def test_missing_columns(self) -> None:
        df = pd.DataFrame({"pwsid": ["A"]})
        with pytest.raises(ValueError, match="Missing columns"):
            build_system_geodataframe(df)


class TestAssembleFeatureMatrix:
    """Tests for assemble_feature_matrix."""

    def test_basic_assembly(self) -> None:
        targets = pd.DataFrame(
            {
                "target": [1, 0, 1],
                "n_samples": [5, 3, 4],
            },
            index=["A", "B", "C"],
        )
        targets.index.name = "pwsid"

        features = pd.DataFrame(
            {
                "feat1": [0.5, 0.3, 0.8],
                "feat2": [1.0, 2.0, 3.0],
            },
            index=["A", "B", "C"],
        )
        features.index.name = "pwsid"

        X, y, _stats = assemble_feature_matrix(targets, features)
        assert len(X) == 3
        assert len(y) == 3
        assert "feat1" in X.columns
        assert "feat2" in X.columns

    def test_imputation(self) -> None:
        targets = pd.DataFrame(
            {
                "target": [1, 0],
            },
            index=["A", "B"],
        )
        targets.index.name = "pwsid"

        features = pd.DataFrame(
            {
                "feat1": [0.5, np.nan],
            },
            index=["A", "B"],
        )
        features.index.name = "pwsid"

        X, _y, stats = assemble_feature_matrix(targets, features)
        assert not X.isna().any().any()
        assert "feat1" in stats

    def test_impute_stats_reuse(self) -> None:
        targets = pd.DataFrame({"target": [1, 0, 1]}, index=["A", "B", "C"])
        targets.index.name = "pwsid"

        # 1/3 NaN (33%) — below 50% threshold, so column is kept
        features = pd.DataFrame({"feat1": [1.0, np.nan, 2.0]}, index=["A", "B", "C"])
        features.index.name = "pwsid"

        pre_stats = {"feat1": 99.0}
        X, _y, _stats = assemble_feature_matrix(targets, features, impute_stats=pre_stats)
        assert X.loc["B", "feat1"] == pytest.approx(99.0)

    def test_drop_high_na_columns(self) -> None:
        targets = pd.DataFrame({"target": [1, 0, 1]}, index=["A", "B", "C"])
        targets.index.name = "pwsid"

        features = pd.DataFrame(
            {
                "good_feat": [1.0, 2.0, 3.0],
                "bad_feat": [np.nan, np.nan, 1.0],  # 66% NaN > 50% threshold
            },
            index=["A", "B", "C"],
        )
        features.index.name = "pwsid"

        X, _y, _stats = assemble_feature_matrix(targets, features)
        assert "good_feat" in X.columns
        assert "bad_feat" not in X.columns

    def test_categorical_encoding(self) -> None:
        targets = pd.DataFrame({"target": [1, 0, 1]}, index=["A", "B", "C"])
        targets.index.name = "pwsid"

        features = pd.DataFrame(
            {
                "aquifer_type": ["sand", "gravel", "sand"],
            },
            index=["A", "B", "C"],
        )
        features.index.name = "pwsid"

        X, _y, _stats = assemble_feature_matrix(
            targets, features, categorical_columns=["aquifer_type"]
        )
        assert any("aquifer_type" in c for c in X.columns)
        assert "aquifer_type" not in X.columns  # Original gone

    def test_multiple_feature_dfs(self) -> None:
        targets = pd.DataFrame({"target": [1, 0]}, index=["A", "B"])
        targets.index.name = "pwsid"

        f1 = pd.DataFrame({"feat1": [1.0, 2.0]}, index=["A", "B"])
        f1.index.name = "pwsid"
        f2 = pd.DataFrame({"feat2": [3.0, 4.0]}, index=["A", "B"])
        f2.index.name = "pwsid"

        X, _y, _stats = assemble_feature_matrix(targets, f1, f2)
        assert "feat1" in X.columns
        assert "feat2" in X.columns

    def test_mismatched_index_left_join(self) -> None:
        targets = pd.DataFrame({"target": [1, 0]}, index=["A", "B"])
        targets.index.name = "pwsid"

        # Feature only for system A
        features = pd.DataFrame({"feat1": [5.0]}, index=["A"])
        features.index.name = "pwsid"

        X, _y, _stats = assemble_feature_matrix(targets, features)
        assert len(X) == 2
        assert not X.isna().any().any()  # Imputed

    def test_column_alignment_across_splits(self) -> None:
        """Train and val/test get same columns even with different categories."""
        train_targets = pd.DataFrame({"target": [1, 0, 1]}, index=["A", "B", "C"])
        train_targets.index.name = "pwsid"
        train_feats = pd.DataFrame(
            {"cat": ["sand", "gravel", "sand"], "num": [1.0, 2.0, 3.0]},
            index=["A", "B", "C"],
        )
        train_feats.index.name = "pwsid"

        X_train, _, stats = assemble_feature_matrix(
            train_targets, train_feats, categorical_columns=["cat"]
        )

        # Val has a category ("clay") not seen in train, and is missing "gravel"
        val_targets = pd.DataFrame({"target": [0, 1]}, index=["D", "E"])
        val_targets.index.name = "pwsid"
        val_feats = pd.DataFrame(
            {"cat": ["sand", "clay"], "num": [4.0, 5.0]},
            index=["D", "E"],
        )
        val_feats.index.name = "pwsid"

        X_val, _, _ = assemble_feature_matrix(
            val_targets, val_feats, categorical_columns=["cat"], impute_stats=stats
        )

        # Columns must match exactly
        assert list(X_val.columns) == list(X_train.columns)
        # "clay" column from val should NOT appear (not in train)
        assert not any("clay" in c for c in X_val.columns)
        # "gravel" column from train should appear (filled with 0)
        gravel_cols = [c for c in X_val.columns if "gravel" in c]
        assert len(gravel_cols) == 1
        assert (X_val[gravel_cols[0]] == 0.0).all()

    def test_non_numeric_column_auto_encoded(self) -> None:
        """String column not in categorical_columns is auto one-hot encoded."""
        targets = pd.DataFrame({"target": [1, 0]}, index=["A", "B"])
        targets.index.name = "pwsid"

        features = pd.DataFrame({"text_col": ["foo", "bar"]}, index=["A", "B"])
        features.index.name = "pwsid"

        X, _y, _stats = assemble_feature_matrix(targets, features)
        assert "text_col" not in X.columns
        assert any("text_col" in c for c in X.columns)

    def test_non_numeric_column_ok_when_in_categorical(self) -> None:
        """String column in categorical_columns is encoded correctly."""
        targets = pd.DataFrame({"target": [1, 0, 1]}, index=["A", "B", "C"])
        targets.index.name = "pwsid"

        features = pd.DataFrame({"aquifer": ["sand", "gravel", "sand"]}, index=["A", "B", "C"])
        features.index.name = "pwsid"

        X, _y, _stats = assemble_feature_matrix(targets, features, categorical_columns=["aquifer"])
        assert "aquifer" not in X.columns
        assert any("aquifer" in c for c in X.columns)

    def test_duplicate_columns_raise_error(self) -> None:
        """Overlapping columns between feature DataFrames raises ValueError."""
        targets = pd.DataFrame({"target": [1, 0]}, index=["A", "B"])
        targets.index.name = "pwsid"

        f1 = pd.DataFrame({"shared": [1.0, 2.0]}, index=["A", "B"])
        f1.index.name = "pwsid"
        f2 = pd.DataFrame({"shared": [3.0, 4.0]}, index=["A", "B"])
        f2.index.name = "pwsid"

        with pytest.raises(ValueError, match="Duplicate columns in feature DataFrames"):
            assemble_feature_matrix(targets, f1, f2)

    def test_duplicate_column_with_system_targets(self) -> None:
        """Feature column overlapping with system_targets raises ValueError."""
        targets = pd.DataFrame({"target": [1, 0], "n_samples": [5, 3]}, index=["A", "B"])
        targets.index.name = "pwsid"

        features = pd.DataFrame({"n_samples": [10.0, 20.0]}, index=["A", "B"])
        features.index.name = "pwsid"

        with pytest.raises(ValueError, match="Duplicate columns in feature DataFrames"):
            assemble_feature_matrix(targets, features)

    def test_high_na_cols_propagated_from_train(self) -> None:
        """High-NaN columns determined by train are also dropped from val/test."""
        train_targets = pd.DataFrame({"target": [1, 0, 1]}, index=["A", "B", "C"])
        train_targets.index.name = "pwsid"
        train_feats = pd.DataFrame(
            {
                "good": [1.0, 2.0, 3.0],
                "bad": [np.nan, np.nan, 1.0],  # 66% NaN > 50%
            },
            index=["A", "B", "C"],
        )
        train_feats.index.name = "pwsid"

        X_train, _, stats = assemble_feature_matrix(train_targets, train_feats)
        assert "bad" not in X_train.columns

        # Val has the "bad" column fully populated — should still be dropped
        val_targets = pd.DataFrame({"target": [0, 1]}, index=["D", "E"])
        val_targets.index.name = "pwsid"
        val_feats = pd.DataFrame(
            {"good": [4.0, 5.0], "bad": [6.0, 7.0]},
            index=["D", "E"],
        )
        val_feats.index.name = "pwsid"

        X_val, _, _ = assemble_feature_matrix(val_targets, val_feats, impute_stats=stats)
        assert "bad" not in X_val.columns
        assert list(X_val.columns) == list(X_train.columns)


class TestPrepareTrainValTest:
    """Tests for prepare_train_val_test."""

    def test_basic_split(self, synthetic_wq_df: pd.DataFrame) -> None:
        result = prepare_train_val_test(synthetic_wq_df, "PFOS")
        assert "train" in result
        assert "val" in result
        assert "test" in result
        for _split_name, (X, y) in result.items():
            assert len(X) == len(y)

    def test_returns_features_and_targets(self, synthetic_wq_df: pd.DataFrame) -> None:
        result = prepare_train_val_test(synthetic_wq_df, "PFOS")
        X_train, y_train = result["train"]
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)

    def test_no_data_leakage(self, synthetic_wq_df: pd.DataFrame) -> None:
        result = prepare_train_val_test(synthetic_wq_df, "PFOS")
        train_idx = set(result["train"][0].index)
        val_idx = set(result["val"][0].index)
        test_idx = set(result["test"][0].index)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)

    def test_no_target_leakage_columns(self, synthetic_wq_df: pd.DataFrame) -> None:
        """Leakage columns (any_detected, detection_rate, max_concentration) must
        never appear as features — they are derived from monitoring results and
        would perfectly predict the target."""
        from aquacontam.features.assembly import _LEAKAGE_COLUMNS

        result = prepare_train_val_test(synthetic_wq_df, "PFOS")
        for split_name, (X, _y) in result.items():
            if X.empty:
                continue
            leaked = set(X.columns) & set(_LEAKAGE_COLUMNS)
            assert not leaked, f"Leakage columns {leaked} found in {split_name} features"

    def test_warns_when_leakage_columns_present(self, caplog: pytest.LogCaptureFixture) -> None:
        targets = pd.DataFrame(
            {"target": [1, 0], "n_samples": [5, 3], "max_concentration": [9.0, 0.0]},
            index=pd.Index(["A", "B"], name="pwsid"),
        )
        features = pd.DataFrame({"feat1": [0.5, 0.3]}, index=pd.Index(["A", "B"], name="pwsid"))
        with caplog.at_level(logging.WARNING, logger="aquacontam.features.assembly"):
            assemble_feature_matrix(targets, features)
        assert any("target-leakage" in r.getMessage() for r in caplog.records)

    def test_no_warning_when_clean(self, caplog: pytest.LogCaptureFixture) -> None:
        targets = pd.DataFrame(
            {"target": [1, 0], "n_samples": [5, 3]},
            index=pd.Index(["A", "B"], name="pwsid"),
        )
        features = pd.DataFrame({"feat1": [0.5, 0.3]}, index=pd.Index(["A", "B"], name="pwsid"))
        with caplog.at_level(logging.WARNING, logger="aquacontam.features.assembly"):
            assemble_feature_matrix(targets, features)
        assert not any("target-leakage" in r.getMessage() for r in caplog.records)

    def test_drop_leakage_columns_is_exported(self) -> None:
        from aquacontam.features import drop_leakage_columns as exported

        df = pd.DataFrame(
            {"target": [1], "any_detected": [1], "detection_rate": [1.0], "feat1": [0.2]},
            index=pd.Index(["A"], name="pwsid"),
        )
        cleaned = exported(df)
        assert "any_detected" not in cleaned.columns
        assert "detection_rate" not in cleaned.columns
        assert "feat1" in cleaned.columns


_PROV_COLS = [
    "n_samples",
    "mean_detection_limit",
    "population_served",
    "log_population_served",
    "owner_type_nan",
    "source_water_type_nan",
    "system_type_nan",
    "nlcd_majority_class_nan",
    # genuine signal that must be RETAINED:
    "source_water_type_GW",
    "source_water_type_SW",
    "system_type_CWS",
    "system_type_NTNCWS",
    "owner_type_L",
    "pct_wetland_5km",
    "dist_nearest_industrial",
    "pct_people_of_color",
]


class TestProvenanceExclusion:
    """Tests for glob-aware feature exclusion (provenance-free honest model)."""

    def test_exact_match_backward_compatible(self) -> None:
        # Plain name lists (no globs) still match exactly, as before.
        dropped = resolve_excluded_columns(_PROV_COLS, ["n_samples", "mean_detection_limit"])
        assert dropped == ["n_samples", "mean_detection_limit"]

    def test_glob_matches_missingness_only(self) -> None:
        # "*_nan" matches every one-hot missingness indicator and nothing else.
        dropped = set(resolve_excluded_columns(_PROV_COLS, ["*_nan"]))
        assert dropped == {
            "owner_type_nan",
            "source_water_type_nan",
            "system_type_nan",
            "nlcd_majority_class_nan",
        }

    def test_provenance_free_drops_provenance_keeps_signal(self) -> None:
        dropped = set(resolve_excluded_columns(_PROV_COLS, PROVENANCE_FREE_EXCLUDE))
        # provenance / monitoring / missingness removed
        assert {
            "n_samples",
            "mean_detection_limit",
            "population_served",
            "log_population_served",
            "owner_type_nan",
            "source_water_type_nan",
            "system_type_nan",
            "nlcd_majority_class_nan",
        } <= dropped
        # genuine hydrology / exposure / environmental signal RETAINED
        kept = set(_PROV_COLS) - dropped
        assert {
            "source_water_type_GW",
            "source_water_type_SW",
            "system_type_CWS",
            "system_type_NTNCWS",
            "pct_wetland_5km",
            "dist_nearest_industrial",
            "pct_people_of_color",
        } <= kept

    def test_result_preserves_column_order(self) -> None:
        dropped = resolve_excluded_columns(_PROV_COLS, ["*_nan", "n_samples"])
        # returned in original column order
        assert dropped == [c for c in _PROV_COLS if c in dropped]
