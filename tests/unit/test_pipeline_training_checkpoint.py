"""Tests for per-task checkpointing in training.py."""

from __future__ import annotations

import json
from pathlib import Path

from aquacontam.pipeline.training import _load_completed_tasks, _save_task_checkpoint


def test_save_task_checkpoint_creates_file(tmp_path: Path):
    """_save_task_checkpoint writes a JSON file under checkpoints/."""
    results = [
        {"task": "T1", "model": "xgboost", "metrics": {"auroc": 0.85}},
    ]
    _save_task_checkpoint("T1", results, tmp_path)
    ckpt = tmp_path / "checkpoints" / "T1_results.json"
    assert ckpt.exists()
    loaded = json.loads(ckpt.read_text())
    assert loaded == results


def test_load_completed_tasks_empty_dir(tmp_path: Path):
    """_load_completed_tasks returns empty when no checkpoints exist."""
    completed, results = _load_completed_tasks(tmp_path)
    assert completed == set()
    assert results == []


def test_load_completed_tasks_finds_existing(tmp_path: Path):
    """_load_completed_tasks discovers previously saved checkpoints."""
    r1 = [{"task": "T1", "model": "rf", "metrics": {"auroc": 0.8}}]
    r2 = [{"task": "T4", "model": "xgb", "metrics": {"auroc": 0.9}}]
    _save_task_checkpoint("T1", r1, tmp_path)
    _save_task_checkpoint("T4", r2, tmp_path)

    completed, results = _load_completed_tasks(tmp_path)
    assert completed == {"T1", "T4"}
    assert len(results) == 2


def test_roundtrip_results_with_numpy_types(tmp_path: Path):
    """Results containing numpy floats survive JSON roundtrip via default=str."""
    import numpy as np

    results = [
        {
            "task": "T1",
            "model": "rf",
            "metrics": {"auroc": np.float64(0.873)},
            "metadata": {"n_samples": np.int64(1000)},
        },
    ]
    _save_task_checkpoint("T1", results, tmp_path)
    completed, loaded = _load_completed_tasks(tmp_path)
    assert "T1" in completed
    # Values survive as strings (via default=str), but the checkpoint exists
    assert len(loaded) == 1
    assert loaded[0]["task"] == "T1"


def test_corrupt_checkpoint_skipped(tmp_path: Path):
    """A corrupt checkpoint file is skipped gracefully."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "T1_results.json").write_text("NOT VALID JSON{{{")

    completed, results = _load_completed_tasks(tmp_path)
    assert "T1" not in completed
    assert results == []
