"""Tests for the review-figure regeneration wired into the reproducibility path.

The heavy ICP re-fit is not exercised here; these pin the pipeline-safety contract: a
missing-input run degrades gracefully (never breaks a long pipeline run), the results dir
is parameterized, and the frozen archive is never written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import paper.regen_review_figures as rrf  # noqa: E402


def test_regenerate_skips_gracefully_when_inputs_missing(tmp_path, monkeypatch):
    """Both results_dir AND the frozen fallback empty → return False, no ICP re-fit.

    monkeypatch the frozen fallback to an empty dir so the skip path is reached without
    triggering the heavy fit (the fallback normally supplies the frozen inputs).
    """
    empty_frozen = tmp_path / "empty_frozen"
    empty_frozen.mkdir()
    monkeypatch.setattr(rrf, "FROZEN_DIR", empty_frozen)
    figures = tmp_path / "figures"
    ok = rrf.regenerate(results_dir=tmp_path / "empty_results", figures_dir=figures)
    assert ok is False
    assert not figures.exists() or not any(figures.iterdir())


def test_read_results_json_prefers_results_dir_then_frozen(tmp_path, monkeypatch):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "x.json").write_text('{"src": "frozen"}', encoding="utf-8")
    monkeypatch.setattr(rrf, "FROZEN_DIR", frozen)
    # absent in results_dir -> falls back to frozen
    assert rrf._read_results_json(tmp_path / "run", "x.json") == {"src": "frozen"}
    # present in results_dir -> preferred over frozen
    run = tmp_path / "run"
    run.mkdir()
    (run / "x.json").write_text('{"src": "run"}', encoding="utf-8")
    assert rrf._read_results_json(run, "x.json") == {"src": "run"}
    # absent in both -> None (skip signal)
    assert rrf._read_results_json(tmp_path / "nope", "absent.json") is None


def test_build_icp_data_reads_results_dir(tmp_path):
    (tmp_path / "icp_diagnostics.json").write_text(
        json.dumps({"results": [{"task": "T1", "metrics": {"auroc": 0.73}}]}), encoding="utf-8"
    )
    data = rrf.build_icp_data([{"epoch": 0, "task_loss": 0.4}], tmp_path)
    assert data["results"] == [{"task": "T1", "metrics": {"auroc": 0.73}}]
    assert data["history"] == {"T1": [{"epoch": 0, "task_loss": 0.4}]}
