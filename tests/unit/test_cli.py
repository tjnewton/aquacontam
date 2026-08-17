"""Tests for the AquaContam CLI."""

from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from aquacontam.__main__ import cli
from aquacontam._constants import SUBMISSION_SCHEMA_VERSION


@pytest.fixture()
def runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


class TestCLIBasic:
    """Tests for basic CLI commands."""

    def test_cli_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "AquaContam" in result.output

    def test_cli_version(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0


class TestWebappCommand:
    """Tests for the webapp subcommand."""

    def test_webapp_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["webapp", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output
        assert "--host" in result.output
        assert "--debug" in result.output

    def test_webapp_missing_module_degrades_gracefully(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Public-release path: an absent webapp module yields a clean message."""
        real_import = builtins.__import__

        def _blocked(name: str, *args: object, **kwargs: object) -> object:
            if name.startswith("aquacontam.webapp"):
                raise ModuleNotFoundError(f"No module named '{name}'", name=name)
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.delitem(sys.modules, "aquacontam.webapp", raising=False)
        monkeypatch.setattr(builtins, "__import__", _blocked)
        result = runner.invoke(cli, ["webapp"])
        assert result.exit_code != 0
        assert "not included in this distribution" in result.output


class TestBenchmarkCommand:
    """Tests for the benchmark subcommand."""

    def test_benchmark_no_task_lists_tasks(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["benchmark"])
        assert result.exit_code == 0
        assert "Registered tasks" in result.output

    def test_benchmark_with_task_errors_no_data(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["benchmark", "--task", "T1"])
        assert result.exit_code != 0
        assert "Hint" in result.output or "Error" in result.output


class TestExportCommand:
    """Tests for the export subcommand."""

    def test_export_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["export", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output

    def test_export_default(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["export"])
        assert result.exit_code == 0
        assert "Version" in result.output


class TestLeaderboardCommand:
    """Tests for the leaderboard subcommands."""

    def test_leaderboard_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["leaderboard", "--help"])
        assert result.exit_code == 0
        assert "validate" in result.output
        assert "rank" in result.output

    def test_leaderboard_validate_missing_file(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["leaderboard", "validate", "nonexistent.json"])
        assert result.exit_code != 0

    def test_leaderboard_validate_valid_file(self, runner: CliRunner, tmp_path: Path) -> None:
        submission = {
            "schema_version": SUBMISSION_SCHEMA_VERSION,
            "model_name": "TestModel",
            "results": [
                {
                    "task": "T1",
                    "metrics": {"auroc": 0.85, "auprc": 0.80, "f1": 0.75},
                }
            ],
        }
        path = tmp_path / "submission.json"
        path.write_text(json.dumps(submission))
        result = runner.invoke(cli, ["leaderboard", "validate", str(path)])
        assert result.exit_code == 0
        assert "Valid submission" in result.output

    def test_leaderboard_rank(self, runner: CliRunner, tmp_path: Path) -> None:
        submission = {
            "schema_version": SUBMISSION_SCHEMA_VERSION,
            "model_name": "TestModel",
            "results": [
                {
                    "task": "T1",
                    "metrics": {"auroc": 0.85, "auprc": 0.80, "f1": 0.75},
                }
            ],
        }
        sub_path = tmp_path / "sub1.json"
        sub_path.write_text(json.dumps(submission))
        result = runner.invoke(cli, ["leaderboard", "rank", str(tmp_path)])
        assert result.exit_code == 0
