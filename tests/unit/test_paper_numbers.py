"""Tests for the single-source-of-truth paper-number generator.

``paper/paper_numbers.py`` renders every headline metric in the manuscript prose
from the frozen result snapshot via idempotent ``<!--pn:ID-->VALUE<!--/pn-->``
markers. These tests cover the registry (every id resolves), the sync/check
round-trip (TP: a correct marker passes; TN: a wrong value is caught), marker
stripping, and the live gate.

The frozen snapshot (``results/paper_frozen/``) is committed, so the frozen-based
loaders resolve in CI. The ``ds_*`` dataset-total loaders read the gitignored
``data/interim/merged_wq.parquet`` and are therefore guarded with a skip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")

from paper.paper_numbers import (
    REGISTRY,
    REPO,
    TABLE_ONLY,
    check_paper_numbers,
    expected,
    strip_markers,
    sync,
)

_MERGED = REPO / "data" / "interim" / "merged_wq.parquet"
_FROZEN = REPO / "results" / "paper_frozen" / "results.json"
_needs_data = pytest.mark.skipif(
    not _MERGED.exists(), reason="merged_wq.parquet (gitignored) not present"
)
_needs_frozen = pytest.mark.skipif(not _FROZEN.exists(), reason="frozen snapshot not present")

# ids whose loader reads the gitignored parquet rather than the frozen JSON.
_DATA_IDS = {mid for mid in REGISTRY if mid.startswith("ds_")}


@_needs_frozen
def test_frozen_ids_resolve_to_nonempty_strings() -> None:
    """Every non-dataset registry id resolves from frozen to a formatted string."""
    for mid in sorted(set(REGISTRY) - _DATA_IDS):
        value = expected(mid)
        assert isinstance(value, str) and value, f"{mid} resolved to {value!r}"


@_needs_frozen
def test_table_only_ids_are_registered_and_resolve() -> None:
    """TABLE_ONLY ids are a subset of the registry and still resolve."""
    assert set(REGISTRY) >= TABLE_ONLY
    for mid in TABLE_ONLY - _DATA_IDS:
        assert expected(mid)


@_needs_data
def test_dataset_ids_resolve() -> None:
    for mid in _DATA_IDS:
        assert expected(mid)


def test_unknown_marker_id_raises() -> None:
    with pytest.raises(KeyError):
        expected("definitely_not_a_real_marker_id")


@_needs_frozen
def test_check_passes_on_correct_marker(tmp_path: Path) -> None:
    """TP: a marker already holding the frozen value yields no value-mismatch."""
    val = expected("t1_full_auroc")
    f = tmp_path / "doc.md"
    f.write_text(f"AUROC <!--pn:t1_full_auroc-->{val}<!--/pn--> here.", encoding="utf-8")
    problems = sync(write=False, files=[f])
    # Completeness will flag the other (unmarked) ids, but no value-mismatch for ours.
    assert not [p for p in problems if "t1_full_auroc" in p and "but frozen" in p]


@_needs_frozen
def test_check_detects_wrong_value(tmp_path: Path) -> None:
    """TN: a marker holding a wrong value is reported as a mismatch."""
    f = tmp_path / "doc.md"
    f.write_text("AUROC <!--pn:t1_full_auroc-->9.999<!--/pn--> here.", encoding="utf-8")
    problems = sync(write=False, files=[f])
    assert any("t1_full_auroc" in p and "9.999" in p for p in problems)


@_needs_frozen
def test_write_corrects_wrong_value(tmp_path: Path) -> None:
    """--write replaces a wrong marker value with the frozen-derived value."""
    f = tmp_path / "doc.md"
    f.write_text("AUROC <!--pn:t1_full_auroc-->0.001<!--/pn--> end.", encoding="utf-8")
    sync(write=True, files=[f])
    text = f.read_text(encoding="utf-8")
    assert f"<!--pn:t1_full_auroc-->{expected('t1_full_auroc')}<!--/pn-->" in text


def test_unknown_marker_in_file_is_reported(tmp_path: Path) -> None:
    """TN: a marker referencing an unknown id is surfaced as a problem."""
    f = tmp_path / "doc.md"
    f.write_text("x <!--pn:bogus_id-->1.0<!--/pn--> y", encoding="utf-8")
    problems = sync(write=False, files=[f])
    assert any("bogus_id" in p for p in problems)


def test_strip_markers_keeps_value_drops_comments() -> None:
    raw = "AUROC <!--pn:t1_full_auroc-->0.852<!--/pn--> and <!--pn:x-->1.2<!--/pn-->."
    assert strip_markers(raw) == "AUROC 0.852 and 1.2."


def test_strip_markers_noop_without_markers() -> None:
    assert strip_markers("plain text 0.852") == "plain text 0.852"


@_needs_data
def test_live_paper_gate_passes() -> None:
    """Integration: the committed paper prose matches frozen (the wired gate)."""
    if not (Path(__file__).resolve().parents[2] / "paper" / "skeleton.md").exists():
        pytest.skip("manuscript sources not distributed")
    assert check_paper_numbers() == []
