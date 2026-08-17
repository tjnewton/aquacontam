"""Tests for temporal shift diagnosis functions."""

from __future__ import annotations


class TestDiagnoseTemporalShift:
    def test_basic_diagnosis(self, synthetic_ucmr3_df, synthetic_ucmr5_df):
        from aquacontam.benchmark.temporal import diagnose_temporal_shift

        result = diagnose_temporal_shift(synthetic_ucmr3_df, synthetic_ucmr5_df)
        assert "per_analyte" in result
        assert "system_overlap" in result
        assert len(result["per_analyte"]) > 0

    def test_per_analyte_fields(self, synthetic_ucmr3_df, synthetic_ucmr5_df):
        from aquacontam.benchmark.temporal import diagnose_temporal_shift

        result = diagnose_temporal_shift(synthetic_ucmr3_df, synthetic_ucmr5_df)
        for entry in result["per_analyte"]:
            assert "analyte" in entry
            assert "ucmr3_detection_rate" in entry or "ucmr5_detection_rate" in entry

    def test_system_overlap(self, synthetic_ucmr3_df, synthetic_ucmr5_df):
        from aquacontam.benchmark.temporal import diagnose_temporal_shift

        result = diagnose_temporal_shift(synthetic_ucmr3_df, synthetic_ucmr5_df)
        overlap = result["system_overlap"]
        assert overlap["ucmr3_systems"] > 0
        assert overlap["ucmr5_systems"] > 0
        assert overlap["persistent"] >= 0

    def test_with_feature_dfs(self, synthetic_ucmr3_df, synthetic_ucmr5_df, synthetic_feature_dfs):
        from aquacontam.benchmark.temporal import diagnose_temporal_shift

        result = diagnose_temporal_shift(
            synthetic_ucmr3_df, synthetic_ucmr5_df, feature_dfs=synthetic_feature_dfs
        )
        assert "feature_shift" in result
