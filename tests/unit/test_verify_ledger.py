"""Tests for paper/verify_ledger.py — machine-verified terminal states."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("click")

from paper.verify_ledger import parse_machine_state, verify_ledger

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "results" / "paper_frozen"


def _write_ledger(tmp_path: Path, issues: list[dict]) -> Path:
    block = "<!-- ledger-machine-state -->\n```json\n" + json.dumps({"issues": issues}) + "\n```\n"
    p = tmp_path / "revision_ledger.md"
    p.write_text("# ledger\n\n" + block, encoding="utf-8")
    return p


def _write_multiblock_ledger(tmp_path: Path, *issue_lists: list[dict]) -> Path:
    """Write a ledger with one ``ledger-machine-state`` block per issue list."""
    blocks = [
        "<!-- ledger-machine-state -->\n```json\n" + json.dumps({"issues": issues}) + "\n```\n"
        for issues in issue_lists
    ]
    p = tmp_path / "revision_ledger.md"
    p.write_text("# ledger\n\n" + "\n\n## next block\n\n".join(blocks), encoding="utf-8")
    return p


def _write_decisions(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "DECISIONS.md"
    p.write_text(body, encoding="utf-8")
    return p


def _write_response_bank(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "RESPONSE_BANK.md"
    p.write_text(body, encoding="utf-8")
    return p


class TestParseMachineState:
    def test_parses_block(self, tmp_path: Path) -> None:
        led = _write_ledger(tmp_path, [{"id": "X", "terminal_state": "DECIDED"}])
        issues = parse_machine_state(led)
        assert issues == [{"id": "X", "terminal_state": "DECIDED"}]

    def test_missing_block_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "revision_ledger.md"
        p.write_text("# ledger with no machine-state block\n")
        with pytest.raises(ValueError):
            parse_machine_state(p)

    def test_multiple_blocks_are_unioned(self, tmp_path: Path) -> None:
        # The R5+R6 case: two blocks, both must be read. Reading only the first
        # (the historical bug) would hide the R6 rows from G7.
        led = _write_multiblock_ledger(
            tmp_path,
            [{"id": "R5-M1", "terminal_state": "PROSE_FIX"}],
            [{"id": "R6-R5-1", "terminal_state": "OPEN"}],
        )
        issues = parse_machine_state(led)
        ids = {i["id"] for i in issues}
        assert ids == {"R5-M1", "R6-R5-1"}

    def test_duplicate_id_across_blocks_raises(self, tmp_path: Path) -> None:
        led = _write_multiblock_ledger(
            tmp_path,
            [{"id": "DUP", "terminal_state": "PROSE_FIX"}],
            [{"id": "DUP", "terminal_state": "OPEN"}],
        )
        with pytest.raises(ValueError, match="Duplicate issue id"):
            parse_machine_state(led)

    def test_open_row_in_second_block_is_not_hidden(self, tmp_path: Path) -> None:
        # End-to-end: a terminal first block cannot mask an OPEN second block.
        led = _write_multiblock_ledger(
            tmp_path,
            [{"id": "R5-M1", "terminal_state": "REFUTED", "rationale": "done"}],
            [{"id": "R6-R5-1", "terminal_state": "OPEN"}],
        )
        errors, _ = verify_ledger(led, _write_decisions(tmp_path, "#\n"), results_dir=FROZEN)
        assert any("R6-R5-1: OPEN" in e for e in errors)


class TestObjective:
    def test_real_check_and_passing_verify_paper(self, tmp_path: Path) -> None:
        led = _write_ledger(
            tmp_path,
            [
                {
                    "id": "M5a",
                    "terminal_state": "OBJECTIVE",
                    "verify_check": "check_table3_invariance_rows",
                }
            ],
        )
        dec = _write_decisions(tmp_path, "# decisions\n")
        errors, _ = verify_ledger(led, dec, results_dir=FROZEN)
        assert errors == []

    def test_nonexistent_check_is_error(self, tmp_path: Path) -> None:
        led = _write_ledger(
            tmp_path,
            [{"id": "M5a", "terminal_state": "OBJECTIVE", "verify_check": "no_such_check"}],
        )
        dec = _write_decisions(tmp_path, "# decisions\n")
        errors, _ = verify_ledger(led, dec, results_dir=FROZEN)
        assert any("not callable" in e for e in errors)

    def test_verify_paper_errors_fail_objective(self, tmp_path: Path) -> None:
        # results_dir without results.json -> verify_paper hard-errors -> OBJECTIVE fails.
        led = _write_ledger(
            tmp_path,
            [
                {
                    "id": "M5a",
                    "terminal_state": "OBJECTIVE",
                    "verify_check": "check_table3_invariance_rows",
                }
            ],
        )
        dec = _write_decisions(tmp_path, "# decisions\n")
        errors, _ = verify_ledger(led, dec, results_dir=tmp_path / "no_results")
        assert any("verify_paper reports" in e for e in errors)


class TestAcknowledged:
    def test_present_phrase_ok(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.md"
        doc.write_text("... the honest estimate of the transportable signal ...")
        led = _write_ledger(
            tmp_path,
            [
                {
                    "id": "m",
                    "terminal_state": "ACKNOWLEDGED",
                    "manuscript_ref": {
                        "file": str(doc),
                        "phrase": "honest estimate of the transportable signal",
                    },
                }
            ],
        )
        dec = _write_decisions(tmp_path, "# decisions\n")
        errors, _ = verify_ledger(led, dec, results_dir=FROZEN)
        assert errors == []

    def test_absent_phrase_is_error(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.md"
        doc.write_text("nothing relevant here")
        led = _write_ledger(
            tmp_path,
            [
                {
                    "id": "m",
                    "terminal_state": "ACKNOWLEDGED",
                    "manuscript_ref": {
                        "file": str(doc),
                        "phrase": "a limitation that is not present",
                    },
                }
            ],
        )
        dec = _write_decisions(tmp_path, "# decisions\n")
        errors, _ = verify_ledger(led, dec, results_dir=FROZEN)
        assert any("phrase not found" in e for e in errors)

    def test_phrase_match_is_whitespace_insensitive(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.md"
        doc.write_text("a modeled\nestimate predating UCMR5)")  # line-wrapped
        led = _write_ledger(
            tmp_path,
            [
                {
                    "id": "m",
                    "terminal_state": "ACKNOWLEDGED",
                    "manuscript_ref": {
                        "file": str(doc),
                        "phrase": "a modeled estimate predating UCMR5",
                    },
                }
            ],
        )
        dec = _write_decisions(tmp_path, "# decisions\n")
        errors, _ = verify_ledger(led, dec, results_dir=FROZEN)
        assert errors == []


class TestDecided:
    def test_ratified_with_section_ok(self, tmp_path: Path) -> None:
        led = _write_ledger(
            tmp_path,
            [{"id": "M1", "terminal_state": "DECIDED", "ratified": True, "rationale": "reorder"}],
        )
        dec = _write_decisions(tmp_path, "### M1\nCall and rationale.\n")
        errors, warnings = verify_ledger(led, dec, results_dir=FROZEN)
        assert errors == []
        assert warnings == []

    def test_not_ratified_is_error(self, tmp_path: Path) -> None:
        led = _write_ledger(
            tmp_path,
            [{"id": "M1", "terminal_state": "DECIDED", "ratified": False, "rationale": "reorder"}],
        )
        dec = _write_decisions(tmp_path, "### M1\n")
        errors, _ = verify_ledger(led, dec, results_dir=FROZEN)
        assert any("not ratified" in e for e in errors)

    def test_empty_rationale_is_error(self, tmp_path: Path) -> None:
        led = _write_ledger(
            tmp_path,
            [{"id": "M1", "terminal_state": "DECIDED", "ratified": True, "rationale": "  "}],
        )
        dec = _write_decisions(tmp_path, "### M1\n")
        errors, _ = verify_ledger(led, dec, results_dir=FROZEN)
        assert any("empty rationale" in e for e in errors)

    def test_missing_decisions_section_warns(self, tmp_path: Path) -> None:
        led = _write_ledger(
            tmp_path,
            [{"id": "M1", "terminal_state": "DECIDED", "ratified": True, "rationale": "reorder"}],
        )
        dec = _write_decisions(tmp_path, "# decisions with no section\n")
        errors, warnings = verify_ledger(led, dec, results_dir=FROZEN)
        assert errors == []
        assert any("no '### M1' section" in w for w in warnings)

    def test_decision_ref_points_at_shared_decision(self, tmp_path: Path) -> None:
        # R6 case: a per-referee row (R6-R1-2) cites a shared decision (R6-D1).
        led = _write_ledger(
            tmp_path,
            [
                {
                    "id": "R6-R1-2",
                    "terminal_state": "DECIDED",
                    "ratified": True,
                    "rationale": "lead with LORO",
                    "decision_ref": "R6-D1",
                }
            ],
        )
        dec = _write_decisions(tmp_path, "### R6-D1\nLead T1 with LORO.\n")
        errors, warnings = verify_ledger(led, dec, results_dir=FROZEN)
        assert errors == []
        assert warnings == []

    def test_decision_ref_absent_section_warns(self, tmp_path: Path) -> None:
        led = _write_ledger(
            tmp_path,
            [
                {
                    "id": "R6-R1-2",
                    "terminal_state": "DECIDED",
                    "ratified": True,
                    "rationale": "lead with LORO",
                    "decision_ref": "R6-D1",
                }
            ],
        )
        dec = _write_decisions(tmp_path, "### R6-OTHER\n")
        _errors, warnings = verify_ledger(led, dec, results_dir=FROZEN)
        assert any("no '### R6-D1' section" in w for w in warnings)


def test_unknown_state_is_error(tmp_path: Path) -> None:
    led = _write_ledger(tmp_path, [{"id": "Z", "terminal_state": "BOGUS"}])
    dec = _write_decisions(tmp_path, "# d\n")
    errors, _ = verify_ledger(led, dec, results_dir=FROZEN)
    assert any("unknown terminal_state" in e for e in errors)


# ---------------------------------------------------------------------------
# R5 work-list states (FINAL_FIX_CONTRACT_v3) — the ratchet + lane-typed terminals.
# Escape-replay: an OPEN row is red; flipping to a valid terminal state is green.
# ---------------------------------------------------------------------------


class TestOpenRatchet:
    def test_open_row_is_error(self, tmp_path: Path) -> None:
        # The ratchet: an OPEN R5 row makes verify_ledger (hence G7) red.
        led = _write_ledger(tmp_path, [{"id": "R5-M1", "terminal_state": "OPEN"}])
        dec = _write_decisions(tmp_path, "# d\n")
        errors, _ = verify_ledger(led, dec, results_dir=FROZEN)
        assert any("R5-M1: OPEN" in e for e in errors)

    def test_flipping_open_to_terminal_clears_the_error(self, tmp_path: Path) -> None:
        doc = tmp_path / "skeleton.md"
        doc.write_text("... reframed to an upper-bound estimate ...")
        led = _write_ledger(
            tmp_path,
            [
                {
                    "id": "R5-M1",
                    "terminal_state": "PROSE_FIX",
                    "witness": {"file": str(doc), "phrase": "an upper-bound estimate"},
                }
            ],
        )
        dec = _write_decisions(tmp_path, "# d\n")
        errors, _ = verify_ledger(led, dec, results_dir=FROZEN)
        assert errors == []


class TestProseFix:
    def test_present_phrase_ok(self, tmp_path: Path) -> None:
        doc = tmp_path / "skeleton.md"
        doc.write_text("gaps are directionally present but not individually significant")
        led = _write_ledger(
            tmp_path,
            [
                {
                    "id": "R5-m7",
                    "terminal_state": "PROSE_FIX",
                    "witness": {"file": str(doc), "phrase": "not individually significant"},
                }
            ],
        )
        errors, _ = verify_ledger(led, _write_decisions(tmp_path, "#\n"), results_dir=FROZEN)
        assert errors == []

    def test_absent_phrase_is_error(self, tmp_path: Path) -> None:
        doc = tmp_path / "skeleton.md"
        doc.write_text("nothing relevant")
        led = _write_ledger(
            tmp_path,
            [
                {
                    "id": "R5-m7",
                    "terminal_state": "PROSE_FIX",
                    "witness": {"file": str(doc), "phrase": "a corrected sentence not present"},
                }
            ],
        )
        errors, _ = verify_ledger(led, _write_decisions(tmp_path, "#\n"), results_dir=FROZEN)
        assert any("PROSE_FIX phrase not found" in e for e in errors)

    def test_empty_phrase_is_error(self, tmp_path: Path) -> None:
        doc = tmp_path / "skeleton.md"
        doc.write_text("x")
        led = _write_ledger(
            tmp_path,
            [
                {
                    "id": "R5-m7",
                    "terminal_state": "PROSE_FIX",
                    "witness": {"file": str(doc), "phrase": " "},
                }
            ],
        )
        errors, _ = verify_ledger(led, _write_decisions(tmp_path, "#\n"), results_dir=FROZEN)
        assert any("empty witness phrase" in e for e in errors)


class TestCompute:
    def test_existing_artifact_ok(self, tmp_path: Path) -> None:
        art = tmp_path / "group_gap_tests.json"
        art.write_text("{}")
        led = _write_ledger(
            tmp_path,
            [{"id": "R5-M1", "terminal_state": "COMPUTE", "artifact": "group_gap_tests.json"}],
        )
        errors, _ = verify_ledger(led, _write_decisions(tmp_path, "#\n"), results_dir=tmp_path)
        assert errors == []

    def test_missing_artifact_is_error(self, tmp_path: Path) -> None:
        led = _write_ledger(
            tmp_path,
            [{"id": "R5-M1", "terminal_state": "COMPUTE", "artifact": "does_not_exist.json"}],
        )
        errors, _ = verify_ledger(led, _write_decisions(tmp_path, "#\n"), results_dir=tmp_path)
        assert any("COMPUTE artifact missing" in e for e in errors)


class TestReDerive:
    def test_existing_artifacts_ok(self, tmp_path: Path) -> None:
        (tmp_path / "a.xlsx").write_text("x")
        (tmp_path / "b.xlsx").write_text("y")
        led = _write_ledger(
            tmp_path,
            [
                {
                    "id": "R6-R5-1",
                    "terminal_state": "RE_DERIVE",
                    "artifacts": [str(tmp_path / "a.xlsx"), str(tmp_path / "b.xlsx")],
                }
            ],
        )
        errors, _ = verify_ledger(led, _write_decisions(tmp_path, "#\n"), results_dir=FROZEN)
        assert errors == []

    def test_single_artifact_field_ok(self, tmp_path: Path) -> None:
        (tmp_path / "a.xlsx").write_text("x")
        led = _write_ledger(
            tmp_path,
            [
                {
                    "id": "R6-R5-1",
                    "terminal_state": "RE_DERIVE",
                    "artifact": str(tmp_path / "a.xlsx"),
                }
            ],
        )
        errors, _ = verify_ledger(led, _write_decisions(tmp_path, "#\n"), results_dir=FROZEN)
        assert errors == []

    def test_missing_artifact_is_error(self, tmp_path: Path) -> None:
        led = _write_ledger(
            tmp_path,
            [
                {
                    "id": "R6-R5-1",
                    "terminal_state": "RE_DERIVE",
                    "artifacts": [str(tmp_path / "gone.xlsx")],
                }
            ],
        )
        errors, _ = verify_ledger(led, _write_decisions(tmp_path, "#\n"), results_dir=FROZEN)
        assert any("RE_DERIVE artifact missing" in e for e in errors)

    def test_no_artifacts_is_error(self, tmp_path: Path) -> None:
        led = _write_ledger(tmp_path, [{"id": "R6-R5-1", "terminal_state": "RE_DERIVE"}])
        errors, _ = verify_ledger(led, _write_decisions(tmp_path, "#\n"), results_dir=FROZEN)
        assert any("RE_DERIVE names no artifacts" in e for e in errors)


class TestBanked:
    def test_heading_present_ok(self, tmp_path: Path) -> None:
        led = _write_ledger(tmp_path, [{"id": "R6-R1-4", "terminal_state": "BANKED"}])
        bank = _write_response_bank(tmp_path, "## R6\n\n### R6-R1-4\nBanked because ...\n")
        errors, _ = verify_ledger(
            led,
            _write_decisions(tmp_path, "#\n"),
            results_dir=FROZEN,
            response_bank_path=bank,
        )
        assert errors == []

    def test_heading_absent_is_error(self, tmp_path: Path) -> None:
        led = _write_ledger(tmp_path, [{"id": "R6-R1-4", "terminal_state": "BANKED"}])
        bank = _write_response_bank(tmp_path, "## R6\n\n### R6-OTHER\nnope\n")
        errors, _ = verify_ledger(
            led,
            _write_decisions(tmp_path, "#\n"),
            results_dir=FROZEN,
            response_bank_path=bank,
        )
        assert any("BANKED but no '### R6-R1-4' heading" in e for e in errors)


class TestRefuted:
    def test_rationale_ok(self, tmp_path: Path) -> None:
        led = _write_ledger(
            tmp_path,
            [
                {
                    "id": "R5-m1",
                    "terminal_state": "REFUTED",
                    "rationale": "0.947 already fixed to 0.910",
                }
            ],
        )
        errors, _ = verify_ledger(led, _write_decisions(tmp_path, "#\n"), results_dir=FROZEN)
        assert errors == []

    def test_empty_rationale_is_error(self, tmp_path: Path) -> None:
        led = _write_ledger(
            tmp_path,
            [{"id": "R5-m1", "terminal_state": "REFUTED", "rationale": "  "}],
        )
        errors, _ = verify_ledger(led, _write_decisions(tmp_path, "#\n"), results_dir=FROZEN)
        assert any("REFUTED has an empty rationale" in e for e in errors)
