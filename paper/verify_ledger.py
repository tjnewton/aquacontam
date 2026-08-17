#!/usr/bin/env python
"""Machine-verify the terminal state of every authoritative revision issue.

Reads the JSON ``ledger-machine-state`` block from ``paper/revision_ledger.md``
and, for each authoritative-set issue, verifies that its declared terminal state
actually holds:

* ``OBJECTIVE``    -> names a ``verify_paper`` check that exists (is callable) AND
  the full ``verify_paper`` run on ``results/paper_frozen`` reports 0 errors.
* ``ACKNOWLEDGED`` -> cites a manuscript file + phrase that is literally present
  (whitespace-normalized), i.e. the owned limitation actually appears in the text.
* ``DECIDED``      -> has a non-empty rationale, a ``ratified: true`` flag, and a
  matching ``### <id>`` section in ``paper/DECISIONS.md``.

This is the decidable "every item is in a terminal state" gate. Usage::

    python paper/verify_ledger.py
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "paper" / "revision_ledger.md"
DECISIONS = REPO / "paper" / "DECISIONS.md"
RESPONSE_BANK = REPO / "paper" / "RESPONSE_BANK.md"
FROZEN = REPO / "results" / "paper_frozen"

# Allow ``import paper.verify_paper`` when run directly as a script.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_BLOCK_RE = re.compile(r"<!--\s*ledger-machine-state\s*-->\s*```json\s*(.*?)```", re.DOTALL)


def _normalize(text: str) -> str:
    """Collapse all whitespace runs to single spaces (survives line-wrapping)."""
    return re.sub(r"\s+", " ", text)


def parse_machine_state(ledger_path: Path) -> list[dict]:
    """Extract and union the ``issues`` from EVERY ledger machine-state block.

    A ledger can carry more than one ``ledger-machine-state`` block (e.g. an R5
    block and an R6 block appended in a later pass). ALL blocks are read and
    their issues unioned; reading only the first (the historical bug) would hide
    a later block of OPEN rows from G7. Fail-closed on zero blocks or on any
    duplicate issue id across blocks — a repeated id would let one block's
    terminal state silently mask another's OPEN row.
    """
    text = ledger_path.read_text(encoding="utf-8")
    blocks = list(_BLOCK_RE.finditer(text))
    if not blocks:
        raise ValueError(f"No ledger-machine-state JSON block found in {ledger_path}")
    issues: list[dict] = []
    seen: set[str] = set()
    for m in blocks:
        for issue in json.loads(m.group(1)).get("issues", []):
            iid = issue.get("id", "<no-id>")
            if iid in seen:
                raise ValueError(f"Duplicate issue id across ledger machine-state blocks: {iid!r}")
            seen.add(iid)
            issues.append(issue)
    return issues


def verify_ledger(
    ledger_path: Path = LEDGER,
    decisions_path: Path = DECISIONS,
    *,
    verify_module: str = "paper.verify_paper",
    results_dir: Path = FROZEN,
    response_bank_path: Path = RESPONSE_BANK,
) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)`` for the ledger terminal-state verification."""
    errors: list[str] = []
    warnings: list[str] = []
    issues = parse_machine_state(ledger_path)

    vp = importlib.import_module(verify_module)
    vp_errors, _vp_warnings = vp.verify_paper(results_dir=results_dir)
    overall_ok = len(vp_errors) == 0

    decisions_text = decisions_path.read_text(encoding="utf-8") if decisions_path.exists() else ""
    response_bank_text = (
        response_bank_path.read_text(encoding="utf-8") if response_bank_path.exists() else ""
    )

    for issue in issues:
        iid = issue.get("id", "<no-id>")
        state = issue.get("terminal_state")
        if state == "OBJECTIVE":
            check = issue.get("verify_check", "")
            if not (check and callable(getattr(vp, check, None))):
                errors.append(
                    f"{iid}: OBJECTIVE names verify_check '{check}' which is not callable "
                    f"in {verify_module}"
                )
            elif not overall_ok:
                errors.append(
                    f"{iid}: OBJECTIVE but verify_paper reports {len(vp_errors)} error(s)"
                )
        elif state == "ACKNOWLEDGED":
            ref = issue.get("manuscript_ref", {})
            f = Path(ref.get("file", ""))
            if not f.is_absolute():
                f = REPO / f
            phrase = ref.get("phrase", "")
            if not f.exists():
                errors.append(f"{iid}: ACKNOWLEDGED cites missing file {ref.get('file')}")
            elif _normalize(phrase) not in _normalize(f.read_text(encoding="utf-8")):
                errors.append(
                    f"{iid}: ACKNOWLEDGED phrase not found in {ref.get('file')}: '{phrase}'"
                )
        elif state == "DECIDED":
            if not issue.get("rationale", "").strip():
                errors.append(f"{iid}: DECIDED has an empty rationale")
            if issue.get("ratified") is not True:
                errors.append(f"{iid}: DECIDED is not ratified")
            # A row may cite a shared decision id (e.g. R6-R1-2 -> R6-D1) via
            # decision_ref; absent, the witness section id is the row id itself.
            dec_id = issue.get("decision_ref", iid)
            if f"### {dec_id}" not in decisions_text:
                warnings.append(f"{iid}: no '### {dec_id}' section in DECISIONS.md")
        # --- R5 work-list states (FINAL_FIX_CONTRACT_v3) --------------------
        elif state == "OPEN":
            # The ratchet: an undispositioned row is a hard error, so G7 stays red
            # (calibrated as EXPECTED_RED={G7}) until every work-list item is terminal.
            section = issue.get("section", "")
            label = f"{section} work-list item" if section else "work-list item"
            errors.append(f"{iid}: OPEN — undispositioned {label}")
        elif state == "PROSE_FIX":
            # Witness = the corrected sentence is present in the named surface (the
            # marker-gated numbers inside it are separately enforced by G1/G3).
            ref = issue.get("witness", {})
            f = Path(ref.get("file", ""))
            if not f.is_absolute():
                f = REPO / f
            phrase = ref.get("phrase", "")
            if not phrase.strip():
                errors.append(f"{iid}: PROSE_FIX has an empty witness phrase")
            elif not f.exists():
                errors.append(f"{iid}: PROSE_FIX cites missing file {ref.get('file')}")
            elif _normalize(phrase) not in _normalize(f.read_text(encoding="utf-8")):
                errors.append(
                    f"{iid}: PROSE_FIX phrase not found in {ref.get('file')}: '{phrase}'"
                )
        elif state == "COMPUTE":
            # Witness = the frozen artifact the compute produced exists (its input-hash
            # stamp is separately enforced by G8; its reader-facing value by G1).
            art = issue.get("artifact", "")
            ap = Path(art)
            if not ap.is_absolute():
                ap = results_dir / art
            if not art.strip():
                errors.append(f"{iid}: COMPUTE names no artifact")
            elif not ap.exists():
                errors.append(f"{iid}: COMPUTE artifact missing: {art}")
        elif state == "REFUTED":
            # Terminal with no manuscript change: the finding is wrong; the evidence
            # of its wrongness is the rationale (mirrored into RESPONSE_BANK.md).
            if not issue.get("rationale", "").strip():
                errors.append(f"{iid}: REFUTED has an empty rationale")
        elif state == "RE_DERIVE":
            # Witness = the regenerated artifact(s) exist on disk (repo-relative).
            # For B1 (source data) this is the freshly-generated per-figure xlsx set;
            # the generator-vs-frozen content equality is separately enforced by G14.
            arts = issue.get("artifacts") or ([issue["artifact"]] if issue.get("artifact") else [])
            if not arts:
                errors.append(f"{iid}: RE_DERIVE names no artifacts")
            for art in arts:
                ap = Path(art)
                if not ap.is_absolute():
                    ap = REPO / art
                if not ap.exists():
                    errors.append(f"{iid}: RE_DERIVE artifact missing: {art}")
        elif state == "BANKED":
            # Terminal "not worth a manuscript edit": the witness is a response-letter
            # paragraph — a '### <id>' heading in RESPONSE_BANK.md must exist.
            if f"### {iid}" not in response_bank_text:
                errors.append(f"{iid}: BANKED but no '### {iid}' heading in RESPONSE_BANK.md")
        else:
            errors.append(f"{iid}: unknown terminal_state {state!r}")
    return errors, warnings


def _main() -> None:
    errors, warnings = verify_ledger()
    for w in warnings:
        print(f"  ⚠ {w}")
    if errors:
        print(f"\n✗ verify_ledger: {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"\n✓ verify_ledger: all issues in a verified terminal state ({len(warnings)} warnings)")
    sys.exit(0)


if __name__ == "__main__":
    _main()
