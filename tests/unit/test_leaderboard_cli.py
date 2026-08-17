"""Tests for leaderboard CLI commands and document generation."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from aquacontam.__main__ import cli
from aquacontam.leaderboard.ranking import generate_leaderboard_document


def _make_submission(model_name: str = "TestModel", task: str = "T1") -> dict:
    return {
        "schema_version": "1.0",
        "model_name": model_name,
        "results": [
            {
                "task": task,
                "analyte": "PFOS",
                "metrics": {"auroc": 0.90, "auprc": 0.85, "f1": 0.80},
            }
        ],
    }


class TestLeaderboardValidate:
    """Tests for 'leaderboard validate' CLI command."""

    def test_valid_submission(self, tmp_path: Path) -> None:
        path = tmp_path / "test.json"
        path.write_text(json.dumps(_make_submission()))

        runner = CliRunner()
        result = runner.invoke(cli, ["leaderboard", "validate", str(path)])
        assert result.exit_code == 0
        assert "Valid submission" in result.output

    def test_invalid_submission(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"schema_version": "1.0"}))

        runner = CliRunner()
        result = runner.invoke(cli, ["leaderboard", "validate", str(path)])
        assert result.exit_code == 1

    def test_missing_file(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["leaderboard", "validate", "/nonexistent/file.json"])
        assert result.exit_code != 0

    def test_valid_with_optional_fields(self, tmp_path: Path) -> None:
        sub = _make_submission()
        sub["author"] = "Test Author"
        sub["date"] = "2026-01-01"
        sub["hardware"] = "CPU only"
        path = tmp_path / "test.json"
        path.write_text(json.dumps(sub))

        runner = CliRunner()
        result = runner.invoke(cli, ["leaderboard", "validate", str(path)])
        assert result.exit_code == 0

    def test_valid_t3_submission(self, tmp_path: Path) -> None:
        sub = {
            "schema_version": "1.0",
            "model_name": "T3Model",
            "results": [
                {
                    "task": "T3",
                    "metrics": {
                        "macro_auroc": 0.85,
                        "macro_auprc": 0.75,
                        "macro_f1": 0.70,
                    },
                }
            ],
        }
        path = tmp_path / "t3.json"
        path.write_text(json.dumps(sub))

        runner = CliRunner()
        result = runner.invoke(cli, ["leaderboard", "validate", str(path)])
        assert result.exit_code == 0

    def test_valid_t7_submission(self, tmp_path: Path) -> None:
        sub = {
            "schema_version": "1.0",
            "model_name": "T7Model",
            "results": [
                {
                    "task": "T7",
                    "analyte": "PFOS",
                    "metrics": {"auroc": 0.80, "auprc": 0.70, "f1": 0.65},
                }
            ],
        }
        path = tmp_path / "t7.json"
        path.write_text(json.dumps(sub))

        runner = CliRunner()
        result = runner.invoke(cli, ["leaderboard", "validate", str(path)])
        assert result.exit_code == 0


class TestLeaderboardRank:
    """Tests for 'leaderboard rank' CLI command."""

    def test_rank_single_submission(self, tmp_path: Path) -> None:
        (tmp_path / "model.json").write_text(json.dumps(_make_submission()))

        runner = CliRunner()
        result = runner.invoke(cli, ["leaderboard", "rank", str(tmp_path)])
        assert result.exit_code == 0
        assert "TestModel" in result.output

    def test_rank_multiple_submissions(self, tmp_path: Path) -> None:
        (tmp_path / "model_a.json").write_text(json.dumps(_make_submission("ModelA")))
        (tmp_path / "model_b.json").write_text(json.dumps(_make_submission("ModelB")))

        runner = CliRunner()
        result = runner.invoke(cli, ["leaderboard", "rank", str(tmp_path)])
        assert result.exit_code == 0
        assert "ModelA" in result.output
        assert "ModelB" in result.output

    def test_rank_empty_dir(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["leaderboard", "rank", str(tmp_path)])
        assert result.exit_code == 0
        assert "No valid submissions" in result.output

    def test_rank_skips_non_json(self, tmp_path: Path) -> None:
        (tmp_path / "readme.txt").write_text("not a submission")
        (tmp_path / "model.json").write_text(json.dumps(_make_submission()))

        runner = CliRunner()
        result = runner.invoke(cli, ["leaderboard", "rank", str(tmp_path)])
        assert result.exit_code == 0
        assert "TestModel" in result.output


class TestLeaderboardPublish:
    """Tests for 'leaderboard publish' CLI command."""

    def test_publish_creates_file(self, tmp_path: Path) -> None:
        (tmp_path / "model.json").write_text(json.dumps(_make_submission()))
        output = tmp_path / "LEADERBOARD.md"

        runner = CliRunner()
        result = runner.invoke(cli, ["leaderboard", "publish", str(tmp_path), "-o", str(output)])
        assert result.exit_code == 0
        assert output.exists()
        content = output.read_text()
        assert "Leaderboard" in content
        assert "TestModel" in content

    def test_publish_default_output(self, tmp_path: Path) -> None:
        (tmp_path / "model.json").write_text(json.dumps(_make_submission()))

        runner = CliRunner()
        result = runner.invoke(
            cli, ["leaderboard", "publish", str(tmp_path), "-o", str(tmp_path / "out.md")]
        )
        assert result.exit_code == 0


class TestGenerateLeaderboardDocument:
    """Tests for generate_leaderboard_document function."""

    def test_basic_document(self, tmp_path: Path) -> None:
        (tmp_path / "model.json").write_text(json.dumps(_make_submission()))
        doc = generate_leaderboard_document(tmp_path)
        assert "AquaContam Benchmark Leaderboard" in doc
        assert "TestModel" in doc
        assert "Citation" in doc

    def test_empty_directory(self, tmp_path: Path) -> None:
        doc = generate_leaderboard_document(tmp_path)
        assert "No valid submissions" in doc

    def test_json_format(self, tmp_path: Path) -> None:
        (tmp_path / "model.json").write_text(json.dumps(_make_submission()))
        doc = generate_leaderboard_document(tmp_path, fmt="json")
        parsed = json.loads(doc)
        assert isinstance(parsed, dict)
        assert "title" in parsed
        assert "submissions" in parsed
        assert isinstance(parsed["rankings"], list)
        assert parsed["submissions"] == 1

    def test_invalid_file_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "bad.json").write_text("not valid json {{{")
        (tmp_path / "good.json").write_text(json.dumps(_make_submission()))
        doc = generate_leaderboard_document(tmp_path)
        assert "TestModel" in doc
