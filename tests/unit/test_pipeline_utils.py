"""Tests for pipeline utility functions, checksum, and strict mode."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from aquacontam.pipeline._strict import is_strict, set_strict
from aquacontam.pipeline.checksum import generate_results_checksums, verify_results_checksums
from aquacontam.pipeline.utils import (
    _grid_size,
    _is_nan,
    clean_caches,
    collect_software_versions,
    print_results_summary,
    save_software_versions,
)

# ---------------------------------------------------------------------------
# _strict.py
# ---------------------------------------------------------------------------


class TestStrict:
    """Tests for strict mode flag."""

    @pytest.fixture(autouse=True)
    def _reset_strict(self):
        """Reset strict mode to False around each test."""
        set_strict(False)
        yield
        set_strict(False)

    def test_default_is_false(self) -> None:
        assert is_strict() is False

    def test_set_strict_true(self) -> None:
        set_strict(True)
        assert is_strict() is True

    def test_set_strict_false(self) -> None:
        set_strict(True)
        assert is_strict() is True
        set_strict(False)
        assert is_strict() is False


# ---------------------------------------------------------------------------
# utils.py — clean_caches
# ---------------------------------------------------------------------------


class TestCleanCaches:
    """Tests for clean_caches()."""

    def test_deletes_merged_wq(self, tmp_path: Path) -> None:
        merged = tmp_path / "interim" / "merged_wq.parquet"
        merged.parent.mkdir(parents=True)
        merged.write_bytes(b"dummy")
        assert merged.exists()

        clean_caches(tmp_path)
        assert not merged.exists()

    def test_deletes_feature_caches(self, tmp_path: Path) -> None:
        features_dir = tmp_path / "interim" / "features"
        features_dir.mkdir(parents=True)
        f1 = features_dir / "proximity.parquet"
        f2 = features_dir / "land_use.parquet"
        f1.write_bytes(b"a")
        f2.write_bytes(b"b")

        clean_caches(tmp_path)
        assert not f1.exists()
        assert not f2.exists()

    def test_noop_if_missing(self, tmp_path: Path) -> None:
        # Neither interim/ nor features/ exist — should not raise
        clean_caches(tmp_path)


# ---------------------------------------------------------------------------
# utils.py — collect_software_versions
# ---------------------------------------------------------------------------


class TestCollectSoftwareVersions:
    """Tests for collect_software_versions()."""

    def test_contains_python_key(self) -> None:
        versions = collect_software_versions()
        assert "python" in versions

    def test_contains_numpy_key(self) -> None:
        versions = collect_software_versions()
        assert "numpy" in versions

    def test_returns_dict_of_strings(self) -> None:
        versions = collect_software_versions()
        assert isinstance(versions, dict)
        for key, value in versions.items():
            assert isinstance(key, str), f"key {key!r} is not a str"
            assert isinstance(value, str), f"value for {key!r} is not a str"


# ---------------------------------------------------------------------------
# utils.py — save_software_versions
# ---------------------------------------------------------------------------


class TestSaveSoftwareVersions:
    """Tests for save_software_versions()."""

    def test_writes_json_file(self, tmp_path: Path) -> None:
        save_software_versions(tmp_path)
        versions_file = tmp_path / "software_versions.json"
        assert versions_file.exists()

        data = json.loads(versions_file.read_text())
        assert "python" in data

    def test_returns_versions_dict(self, tmp_path: Path) -> None:
        result = save_software_versions(tmp_path)
        assert isinstance(result, dict)
        assert "python" in result


# ---------------------------------------------------------------------------
# utils.py — print_results_summary
# ---------------------------------------------------------------------------


class TestPrintResultsSummary:
    """Tests for print_results_summary()."""

    def test_prints_table_header(self, capsys: pytest.CaptureFixture[str]) -> None:
        results: list[dict[str, Any]] = [
            {"task": "T1", "model": "XGBoost", "metrics": {"auroc": 0.85}},
        ]
        print_results_summary(results)
        captured = capsys.readouterr().out
        assert "RESULTS SUMMARY" in captured

    def test_empty_results(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_results_summary([])
        captured = capsys.readouterr().out
        assert "No results to summarize." in captured

    def test_results_with_metrics(self, capsys: pytest.CaptureFixture[str]) -> None:
        results: list[dict[str, Any]] = [
            {"task": "T1", "model": "RF", "metrics": {"auroc": 0.9123}},
            {"task": "T2", "model": "XGB", "metrics": {"rmse": 1.2345}},
        ]
        print_results_summary(results)
        captured = capsys.readouterr().out
        assert "0.9123" in captured
        assert "1.2345" in captured
        assert "RF" in captured
        assert "XGB" in captured


# ---------------------------------------------------------------------------
# utils.py — _grid_size
# ---------------------------------------------------------------------------


class TestGridSize:
    """Tests for _grid_size()."""

    def test_two_params(self) -> None:
        assert _grid_size({"a": [1, 2], "b": [3, 4, 5]}) == 6

    def test_empty_dict(self) -> None:
        assert _grid_size({}) == 0

    def test_single_param(self) -> None:
        assert _grid_size({"x": [1, 2, 3]}) == 3

    def test_one_value_per_param(self) -> None:
        assert _grid_size({"a": [1], "b": [2], "c": [3]}) == 1


# ---------------------------------------------------------------------------
# utils.py — _is_nan
# ---------------------------------------------------------------------------


class TestIsNan:
    """Tests for _is_nan()."""

    def test_float_nan(self) -> None:
        assert _is_nan(float("nan")) is True

    def test_math_nan(self) -> None:
        assert _is_nan(math.nan) is True

    def test_integer(self) -> None:
        assert _is_nan(42) is False

    def test_string(self) -> None:
        assert _is_nan("foo") is False

    def test_none(self) -> None:
        assert _is_nan(None) is False

    def test_zero(self) -> None:
        assert _is_nan(0) is False


# ---------------------------------------------------------------------------
# checksum.py — generate_results_checksums
# ---------------------------------------------------------------------------


class TestGenerateResultsChecksums:
    """Tests for generate_results_checksums()."""

    def test_generates_checksum_file(self, tmp_path: Path) -> None:
        # Create two JSON result files
        (tmp_path / "t1_results.json").write_text('{"auroc": 0.85}')
        (tmp_path / "t2_results.json").write_text('{"rmse": 1.23}')

        checksum_path = generate_results_checksums(tmp_path)

        assert checksum_path.exists()
        assert checksum_path.name == "checksums.sha256"
        lines = [line for line in checksum_path.read_text().strip().splitlines() if line.strip()]
        assert len(lines) == 2
        # Each line should have the format: <sha256>  <filename>
        for line in lines:
            parts = line.split("  ", 1)
            assert len(parts) == 2
            assert len(parts[0]) == 64  # SHA-256 hex digest length

    def test_empty_dir(self, tmp_path: Path) -> None:
        # No JSON files — should not error
        checksum_path = generate_results_checksums(tmp_path)
        assert checksum_path.exists()
        # Only a trailing newline expected
        content = checksum_path.read_text().strip()
        assert content == ""


# ---------------------------------------------------------------------------
# checksum.py — verify_results_checksums
# ---------------------------------------------------------------------------


class TestVerifyResultsChecksums:
    """Tests for verify_results_checksums()."""

    def test_valid_checksums_pass(self, tmp_path: Path) -> None:
        (tmp_path / "results.json").write_text('{"ok": true}')
        generate_results_checksums(tmp_path)

        assert verify_results_checksums(tmp_path) is True

    def test_tampered_file_fails(self, tmp_path: Path) -> None:
        results_file = tmp_path / "results.json"
        results_file.write_text('{"ok": true}')
        generate_results_checksums(tmp_path)

        # Tamper with the file
        results_file.write_text('{"ok": false}')

        assert verify_results_checksums(tmp_path) is False

    def test_missing_checksum_file(self, tmp_path: Path) -> None:
        # No checksums.sha256 exists
        assert verify_results_checksums(tmp_path) is False

    def test_missing_referenced_file(self, tmp_path: Path) -> None:
        results_file = tmp_path / "results.json"
        results_file.write_text('{"data": 1}')
        generate_results_checksums(tmp_path)

        # Delete the referenced file after checksum generation
        results_file.unlink()

        assert verify_results_checksums(tmp_path) is False
