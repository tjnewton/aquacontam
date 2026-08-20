"""Tests for the AquaContam CLI."""

from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from aquacontam.__main__ import cli, main
from aquacontam._constants import SUBMISSION_SCHEMA_VERSION

_HAS_WEBAPP = importlib.util.find_spec("aquacontam.webapp") is not None


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
    """Tests for the optional webapp subcommand (registered only when present)."""

    def test_registration_probe_matches_spec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_webapp_available() mirrors find_spec; falsy spec means no registration."""
        from aquacontam import __main__ as main_mod

        assert main_mod._webapp_available() is _HAS_WEBAPP
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        assert main_mod._webapp_available() is False

    @pytest.mark.skipif(not _HAS_WEBAPP, reason="webapp module not distributed in this tree")
    def test_webapp_help_when_present(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["webapp", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output
        assert "--host" in result.output
        assert "--debug" in result.output

    @pytest.mark.skipif(not _HAS_WEBAPP, reason="webapp module not distributed in this tree")
    def test_webapp_missing_module_degrades_gracefully(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A registered command whose module breaks at call time yields a clean message."""
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

    @pytest.mark.skipif(_HAS_WEBAPP, reason="webapp module present in this tree")
    def test_webapp_hidden_when_absent(self, runner: CliRunner) -> None:
        """Public-release path: the command is not registered at all."""
        result = runner.invoke(cli, ["webapp"])
        assert result.exit_code == 2
        assert "No such command" in result.output
        help_result = runner.invoke(cli, ["--help"])
        assert help_result.exit_code == 0
        assert "webapp" not in help_result.output


def test_main_entry_point_exits_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() runs the CLI in standalone mode and exits 0 on success."""
    monkeypatch.setattr(sys, "argv", ["aquacontam", "--version"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0


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
