"""Tests for paper/freeze_results.py — the tracked frozen result snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import paper.freeze_results as freeze_results
import pytest
from paper.freeze_results import _PF_RENAME_MAP, _SLIM_ARRAY_FIELDS, freeze, slim_results


def _write_results_json(path: Path, auroc: float = 0.8) -> None:
    entries = [
        {
            "task": "T1",
            "model": "xgboost_classifier",
            "metrics": {"auroc": auroc, "auprc": auroc - 0.1},
            "metadata": {
                "y_true": [0, 1],
                "y_prob": [0.1, 0.9],
                "latitudes": [40.0],
                "longitudes": [-90.0],
                "split_labels": ["test"],
                "n_test": 2,
            },
        }
    ]
    path.write_text(json.dumps(entries), encoding="utf-8")


def _make_source_dir(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    _write_results_json(source / "results.json")
    (source / "equity_analysis.json").write_text(
        json.dumps({"regime": "canonical"}), encoding="utf-8"
    )
    return source


def _make_pf_dir(tmp_path: Path) -> Path:
    pf = tmp_path / "pf"
    pf.mkdir()
    _write_results_json(pf / "results.json", auroc=0.62)
    for name in _PF_RENAME_MAP:
        if name != "results.json":
            (pf / name).write_text(json.dumps({"regime": "provenance_free", "src": name}))
    return pf


class TestSlimResults:
    def test_slim_results_drops_array_fields(self, tmp_path: Path) -> None:
        src = tmp_path / "results.json"
        _write_results_json(src)
        dst = tmp_path / "slim.json"
        removed = slim_results(src, dst)
        assert removed == len(_SLIM_ARRAY_FIELDS)
        slimmed = json.loads(dst.read_text())
        md = slimmed[0]["metadata"]
        assert not any(f in md for f in _SLIM_ARRAY_FIELDS)
        assert md["n_test"] == 2  # scalar metadata retained


class TestFreeze:
    def test_freeze_skips_bak_and_optuna_files(self, tmp_path: Path) -> None:
        source = _make_source_dir(tmp_path)
        (source / "results.json.pre-rerun.bak").write_text("{}")
        (source / "optuna_tuning.json").write_text("[]")
        frozen = tmp_path / "frozen"
        freeze(source, frozen)
        assert not (frozen / "results.json.pre-rerun.bak").exists()
        assert not (frozen / "optuna_tuning.json").exists()
        assert (frozen / "results.json").exists()

    def test_freeze_renames_provenance_free_files(self, tmp_path: Path) -> None:
        source = _make_source_dir(tmp_path)
        pf = _make_pf_dir(tmp_path)
        frozen = tmp_path / "frozen"
        summary = freeze(source, frozen, pf_source=pf)

        # All rename targets exist; the PF results.json was slimmed.
        for target in _PF_RENAME_MAP.values():
            assert (frozen / target).exists(), target
        pf_results = json.loads((frozen / "results_provenance_free.json").read_text())
        assert "y_true" not in pf_results[0]["metadata"]
        assert pf_results[0]["metrics"]["auroc"] == 0.62

        # Same-named PF files must NOT overwrite the canonical-regime copies.
        canonical = json.loads((frozen / "equity_analysis.json").read_text())
        assert canonical["regime"] == "canonical"
        pf_equity = json.loads((frozen / "equity_analysis_provenance_free.json").read_text())
        assert pf_equity["regime"] == "provenance_free"

        # Checksums cover the renamed files.
        checksums = (frozen / "checksums.sha256").read_text()
        assert "results_provenance_free.json" in checksums
        assert sorted(summary["pf_files"]) == sorted(_PF_RENAME_MAP.values())

    def test_freeze_pf_missing_file_raises(self, tmp_path: Path) -> None:
        source = _make_source_dir(tmp_path)
        pf = _make_pf_dir(tmp_path)
        (pf / "shap_T1.json").unlink()
        with pytest.raises(SystemExit, match=r"reproduce\.py --shap --provenance-free"):
            freeze(source, tmp_path / "frozen", pf_source=pf)

    def test_freeze_pf_source_absent_is_noop_when_already_frozen(self, tmp_path: Path) -> None:
        source = _make_source_dir(tmp_path)
        pf = _make_pf_dir(tmp_path)
        frozen = tmp_path / "frozen"
        freeze(source, frozen, pf_source=pf)
        # Re-freeze on a machine without the PF run: targets already frozen -> no-op.
        summary = freeze(source, frozen, pf_source=tmp_path / "does_not_exist")
        assert summary["pf_files"] == []
        assert (frozen / "results_provenance_free.json").exists()

    def test_freeze_pf_source_absent_and_not_frozen_raises(self, tmp_path: Path) -> None:
        source = _make_source_dir(tmp_path)
        with pytest.raises(SystemExit, match="Provenance-free source not found"):
            freeze(source, tmp_path / "frozen", pf_source=tmp_path / "does_not_exist")

    def test_freeze_extra_file_copied(self, tmp_path: Path) -> None:
        source = _make_source_dir(tmp_path)
        extra = tmp_path / "t6_arsenic.json"
        extra.write_text(json.dumps({"task": "T6"}), encoding="utf-8")
        frozen = tmp_path / "frozen"
        summary = freeze(source, frozen, extras=[extra])
        assert (frozen / "t6_arsenic.json").exists()
        assert summary["extras"] == ["t6_arsenic.json"]
        assert "t6_arsenic.json" in (frozen / "checksums.sha256").read_text()

    def test_freeze_extra_missing_raises(self, tmp_path: Path) -> None:
        source = _make_source_dir(tmp_path)
        with pytest.raises(SystemExit, match="Extra result file not found"):
            freeze(source, tmp_path / "frozen", extras=[tmp_path / "missing.json"])

    def test_freeze_size_guard_applies_to_pf_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = _make_source_dir(tmp_path)
        pf = _make_pf_dir(tmp_path)
        monkeypatch.setattr(freeze_results, "_MAX_FILE_MB", 1e-6)
        with pytest.raises(SystemExit, match="exceed"):
            freeze(source, tmp_path / "frozen", pf_source=pf)

    def test_checksums_are_lf_only(self, tmp_path: Path) -> None:
        """The `sha256sum -c` reviewer command breaks on CRLF; the manifest must be LF.

        Escape replay: write_text would translate \n to CRLF on Windows; _write_checksums
        forces LF via write_bytes so the documented reviewer command stays portable.
        """
        source = _make_source_dir(tmp_path)
        frozen = tmp_path / "frozen"
        freeze(source, frozen)
        raw = (frozen / "checksums.sha256").read_bytes()
        assert b"\r\n" not in raw, "checksums.sha256 must be LF-only for sha256sum -c on Unix"
        assert raw.endswith(b"\n")
