"""Tests for leaderboard submission schema validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aquacontam.leaderboard.schema import (
    SUBMISSION_SCHEMA_VERSION,
    load_submission,
    validate_submission,
)


def _valid_submission() -> dict:
    """Minimal valid submission."""
    return {
        "schema_version": SUBMISSION_SCHEMA_VERSION,
        "model_name": "TestModel",
        "results": [
            {
                "task": "T1",
                "analyte": "PFOS",
                "metrics": {"auroc": 0.9, "auprc": 0.8, "f1": 0.7},
            }
        ],
    }


class TestValidateSubmission:
    """Tests for validate_submission."""

    def test_valid_submission_no_errors(self) -> None:
        errors = validate_submission(_valid_submission())
        assert errors == []

    def test_missing_schema_version(self) -> None:
        sub = _valid_submission()
        del sub["schema_version"]
        errors = validate_submission(sub)
        assert any("schema_version" in e for e in errors)

    def test_missing_model_name(self) -> None:
        sub = _valid_submission()
        del sub["model_name"]
        errors = validate_submission(sub)
        assert any("model_name" in e for e in errors)

    def test_missing_results(self) -> None:
        sub = _valid_submission()
        del sub["results"]
        errors = validate_submission(sub)
        assert any("results" in e for e in errors)

    def test_wrong_schema_version(self) -> None:
        sub = _valid_submission()
        sub["schema_version"] = "99.0"
        errors = validate_submission(sub)
        assert any("schema_version" in e for e in errors)

    def test_empty_model_name(self) -> None:
        sub = _valid_submission()
        sub["model_name"] = ""
        errors = validate_submission(sub)
        assert any("model_name" in e for e in errors)

    def test_empty_results_list(self) -> None:
        sub = _valid_submission()
        sub["results"] = []
        errors = validate_submission(sub)
        assert any("non-empty" in e for e in errors)

    def test_unknown_task(self) -> None:
        sub = _valid_submission()
        sub["results"][0]["task"] = "T99"
        errors = validate_submission(sub)
        assert any("unknown task" in e for e in errors)

    def test_missing_task_in_result(self) -> None:
        sub = _valid_submission()
        del sub["results"][0]["task"]
        errors = validate_submission(sub)
        assert any("task" in e for e in errors)

    def test_non_numeric_metric(self) -> None:
        sub = _valid_submission()
        sub["results"][0]["metrics"]["auroc"] = "high"
        errors = validate_submission(sub)
        assert any("number" in e for e in errors)

    def test_multiple_results(self) -> None:
        sub = _valid_submission()
        sub["results"].append({"task": "T4", "analyte": "lead", "metrics": {"auroc": 0.75}})
        errors = validate_submission(sub)
        assert errors == []

    def test_optional_fields_allowed(self) -> None:
        sub = _valid_submission()
        sub["author"] = "Test"
        sub["institution"] = "Uni"
        sub["paper_url"] = None
        sub["code_url"] = "https://example.com"
        errors = validate_submission(sub)
        assert errors == []


class TestLoadSubmission:
    """Tests for load_submission."""

    def test_loads_valid_file(self, tmp_path: Path) -> None:
        path = tmp_path / "sub.json"
        with open(path, "w") as f:
            json.dump(_valid_submission(), f)
        result = load_submission(path)
        assert result["model_name"] == "TestModel"

    def test_raises_on_invalid(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        with open(path, "w") as f:
            json.dump({"model_name": "X"}, f)
        with pytest.raises(ValueError, match="errors"):
            load_submission(path)

    def test_raises_on_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_submission("/nonexistent/path.json")

    def test_raises_on_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{invalid json")
        with pytest.raises(json.JSONDecodeError):
            load_submission(path)

    def test_loads_template(self) -> None:
        template_path = Path("configs/submission_template.json")
        if template_path.exists():
            result = load_submission(template_path)
            assert result["model_name"] == "ExampleModel"
