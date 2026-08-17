"""Tests for leaderboard ranking and formatting."""

from __future__ import annotations

import pandas as pd
import pytest

from aquacontam.leaderboard.ranking import format_leaderboard, rank_submissions


def _sub(name: str, results: list[dict]) -> dict:
    """Helper to build a minimal submission."""
    return {
        "schema_version": "1.0",
        "model_name": name,
        "results": results,
    }


class TestRankSubmissions:
    """Tests for rank_submissions."""

    def test_single_submission(self) -> None:
        subs = [_sub("A", [{"task": "T1", "analyte": "PFOS", "metrics": {"auprc": 0.9}}])]
        df = rank_submissions(subs)
        assert len(df) == 1
        assert df.iloc[0]["rank"] == 1
        assert df.iloc[0]["overall_rank"] == 1

    def test_two_models_ranked(self) -> None:
        subs = [
            _sub("A", [{"task": "T1", "analyte": "PFOS", "metrics": {"auprc": 0.9}}]),
            _sub("B", [{"task": "T1", "analyte": "PFOS", "metrics": {"auprc": 0.8}}]),
        ]
        df = rank_submissions(subs)
        a_row = df[df["model_name"] == "A"].iloc[0]
        b_row = df[df["model_name"] == "B"].iloc[0]
        # Higher auprc = better = rank 1
        assert a_row["rank"] == 1
        assert b_row["rank"] == 2

    def test_regression_lower_is_better(self) -> None:
        subs = [
            _sub("A", [{"task": "T2", "analyte": "PFOS", "metrics": {"rmse": 2.0}}]),
            _sub("B", [{"task": "T2", "analyte": "PFOS", "metrics": {"rmse": 5.0}}]),
        ]
        df = rank_submissions(subs)
        a_row = df[df["model_name"] == "A"].iloc[0]
        b_row = df[df["model_name"] == "B"].iloc[0]
        # Lower RMSE = better = rank 1
        assert a_row["rank"] == 1
        assert b_row["rank"] == 2

    def test_overall_rank_mean_of_task_ranks(self) -> None:
        subs = [
            _sub(
                "A",
                [
                    {"task": "T1", "analyte": "PFOS", "metrics": {"auprc": 0.9}},
                    {"task": "T4", "analyte": "lead", "metrics": {"auprc": 0.7}},
                ],
            ),
            _sub(
                "B",
                [
                    {"task": "T1", "analyte": "PFOS", "metrics": {"auprc": 0.8}},
                    {"task": "T4", "analyte": "lead", "metrics": {"auprc": 0.9}},
                ],
            ),
        ]
        df = rank_submissions(subs)
        # Both models: mean rank = (1+2)/2 = 1.5 → tied
        a_overall = df[df["model_name"] == "A"]["overall_rank"].iloc[0]
        b_overall = df[df["model_name"] == "B"]["overall_rank"].iloc[0]
        assert a_overall == b_overall

    def test_empty_submissions(self) -> None:
        df = rank_submissions([])
        assert df.empty

    def test_missing_metric_skipped(self) -> None:
        subs = [_sub("A", [{"task": "T1", "analyte": "PFOS", "metrics": {"f1": 0.7}}])]
        df = rank_submissions(subs)
        assert df.empty  # auprc not present, so skipped

    def test_returns_correct_columns(self) -> None:
        subs = [_sub("A", [{"task": "T1", "analyte": "PFOS", "metrics": {"auprc": 0.9}}])]
        df = rank_submissions(subs)
        expected = {
            "model_name",
            "task",
            "analyte",
            "primary_metric",
            "value",
            "rank",
            "overall_rank",
        }
        assert set(df.columns) == expected

    def test_custom_primary_metrics(self) -> None:
        subs = [
            _sub("A", [{"task": "T1", "analyte": "PFOS", "metrics": {"f1": 0.9, "auprc": 0.5}}]),
        ]
        df = rank_submissions(subs, primary_metrics={"T1": "f1"})
        assert df.iloc[0]["value"] == pytest.approx(0.9)


class TestFormatLeaderboard:
    """Tests for format_leaderboard."""

    def test_markdown_output(self) -> None:
        subs = [_sub("A", [{"task": "T1", "analyte": "PFOS", "metrics": {"auprc": 0.9}}])]
        df = rank_submissions(subs)
        result = format_leaderboard(df, fmt="markdown")
        assert "| Rank |" in result
        assert "A" in result

    def test_json_output(self) -> None:
        subs = [_sub("A", [{"task": "T1", "analyte": "PFOS", "metrics": {"auprc": 0.9}}])]
        df = rank_submissions(subs)
        result = format_leaderboard(df, fmt="json")
        assert '"model_name"' in result

    def test_empty_returns_message(self) -> None:
        df = pd.DataFrame(
            columns=[
                "model_name",
                "task",
                "analyte",
                "primary_metric",
                "value",
                "rank",
                "overall_rank",
            ]
        )
        result = format_leaderboard(df)
        assert "No submissions" in result
