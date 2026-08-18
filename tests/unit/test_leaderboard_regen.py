"""LEADERBOARD.md is a generated artifact regenerated from the frozen archive.

These guard the M6 fix: regeneration is deterministic (byte-stable, so the diff-match gate is
meaningful), the debunked pre-leakage-fix T4 value cannot reappear, and the committed file
equals a fresh regeneration from ``results/paper_frozen/``.
"""

from __future__ import annotations

from paper import regenerate_leaderboard as rl


def test_regeneration_is_deterministic():
    assert rl.render() == rl.render()


def test_debunked_t4_leakage_value_absent():
    # 0.8999 was the pre-leakage-fix T4 artefact the paper's T4 narrative debunks.
    assert "0.8999" not in rl.render()


def test_committed_leaderboard_matches_frozen_regeneration():
    committed = rl.OUT.read_text(encoding="utf-8")
    assert rl.render() == committed, "LEADERBOARD.md drifted from a frozen regeneration"
