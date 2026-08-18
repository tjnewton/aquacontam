#!/usr/bin/env python
"""The single deterministic exit oracle for the paper's reader-facing consistency (v2).

Runs gate clauses **G0-G14** and prints a pass/fail table plus the failing-clause count
**F**. This exit code -- never a review opinion -- is the definition of "done".
In the distributed benchmark tree (manuscript sources not included), the
manuscript-consistency clauses report a skip and every benchmark-artifact clause
enforces in full; partial absence of the manuscript set is treated as damage, not
distribution, and fails loud.

    G0  no sticky red flag                           paper/GATE_RED.flag absent
    G1  paper_numbers.py --check                     markers == frozen (+ completeness)
    G2  verify_paper.py                              0 hard errors; warning multiset ==
                                                     paper/allowed_warnings.txt (fail-closed)
    G3  number_audit.py --check                      no ungated reader-facing derived number
    G4  generate_tables regen == committed tables    (temp-dir diff; tree not mutated)
    G5  pytest (paper unit tests)                     green
    G6  results/paper_frozen unchanged               archive byte-identical to HEAD
    G7  verify_ledger.py                              every work-list item has a disposition
    G8  DERIVATIONS.tsv input-hash stamps            derived artifacts not stale vs sources
    G9  non-manuscript surfaces                      no conflicting concept renderings
                                                     (README, .zenodo.json, CHANGELOG latest)
    G10 CONCEPT_REGISTRY.tsv                         every concept ACTIVE + no manuscript
                                                     conflicts (fail-closed while pending)
    G11 frozen checksums                             sha256 of every frozen JSON matches
                                                     checksums.sha256 (both directions)
    G12 DOCX freshness                               embedded source-manifest hash ==
                                                     recomputed hash of current sources
    G13 CLAIMS.tsv                                   every pinned load-bearing claim sentence
                                                     present (normalized) + evidence holds
    G14 source data                                  paper/source_data/*.xlsx regenerate from
                                                     frozen (file-set + per-cell equality) and
                                                     carry per-figure sentinel values from frozen

Staged CI (calibrated red): while ``paper/EXPECTED_RED.json`` exists, ``--check`` exits 0
iff the failing-clause set EXACTLY equals the expected set — Phase 0's decidable exit
against the old archive. Deleting that file (Phase 2) flips the same command to hard
true-green, which is the only state that prints the ``GATE GREEN sha=<HEAD> dirty=<n>``
line consumed by /goal.

Usage::

    python paper/final_gate.py --check            # gate: exit 0 iff green (or calibrated-red match)
    python paper/final_gate.py --fast             # Stop-hook subset: G0+G1+G3+G6+G8, seconds
    python paper/final_gate.py --clear-red-flag   # delete GATE_RED.flag iff everything else is green

Each subprocess clause is isolated so a crash in one cannot mask another. ``run_clauses``
is exposed (and dependency-injectable) so ``tests/unit/test_final_gate.py`` can gate the
gater: inject a failing clause and assert the driver reports F>0 / exits nonzero (a green
stub must not yield a false pass).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

try:
    from paper import gate_lib
except ImportError:  # pragma: no cover - script-dir-on-path fallback
    import gate_lib

REPO = Path(__file__).resolve().parents[1]
FROZEN_REL = "results/paper_frozen"
ALLOWED_WARNINGS = REPO / "paper" / "allowed_warnings.txt"

#: The six reader-facing manuscript files (mirrors paper_numbers.PAPER_FILES), resolved
#: against an injectable root so fixture tests can exercise G10 hermetically.
_MANUSCRIPT_NAMES = (
    "skeleton.md",
    "supplementary_information.md",
    "extended_data.md",
    "cover_letter.md",
    "reporting_summary.md",
    "plain_language_primer.md",
)


class ClauseResult:
    def __init__(self, name: str, ok: bool, summary: str):
        self.name = name
        self.ok = ok
        self.summary = summary


def _run(cmd: list[str]) -> tuple[int, str]:
    """Run a subprocess from the repo root with PYTHONUTF8=1; return (exit_code, output)."""
    import os

    env = dict(os.environ, PYTHONUTF8="1")
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


# ---------------------------------------------------------------------------
# Clauses
# ---------------------------------------------------------------------------


def g0_sticky_flag(flag: Path | None = None) -> ClauseResult:
    """Sticky red: the Stop hook writes paper/GATE_RED.flag on block-cap exhaustion."""
    f = flag or gate_lib.GATE_RED_FLAG
    if f.exists():
        head = f.read_text(encoding="utf-8").splitlines()
        detail = head[0][:70] if head else ""
        return ClauseResult(
            "G0 sticky_flag", False, f"GATE_RED.flag present ({detail}); use --clear-red-flag"
        )
    return ClauseResult("G0 sticky_flag", True, "no sticky red flag")


#: Manuscript sources withheld from the distributed benchmark tree. All-or-nothing:
#: the skip below fires only when EVERY source is absent; partial absence is damage,
#: not distribution, and the affected clauses then fail loud.
_MANUSCRIPT_SOURCES = (
    "skeleton.md",
    "supplementary_information.md",
    "extended_data.md",
    "reporting_summary.md",
    "plain_language_primer.md",
)

_MANUSCRIPT_SKIP = "manuscript sources not distributed -- skipped"


def _manuscript_absent(root: Path | None = None) -> bool:
    """True only when every manuscript source is absent (the distributed tree)."""
    paper = (root or REPO) / "paper"
    return not any((paper / name).exists() for name in _MANUSCRIPT_SOURCES)


def g1_paper_numbers() -> ClauseResult:
    if _manuscript_absent():
        return ClauseResult("G1 paper_numbers", True, _MANUSCRIPT_SKIP)
    rc, out = _run([sys.executable, "paper/paper_numbers.py", "--check"])
    last = out.strip().splitlines()[-1] if out.strip() else ""
    return ClauseResult("G1 paper_numbers", rc == 0, last)


def g2_verify_paper(
    allowed_warnings: Path | None = None, _result: tuple[int, str] | None = None
) -> ClauseResult:
    """0 hard errors AND the emitted warning multiset EXACTLY equals the frozen manifest.

    Fail-closed: a missing manifest fails (the v1 fail-open let an absent file skip the
    warning comparison entirely). Matching is exact on whitespace-normalized strings with
    counts — the v1 substring rule let one broad manifest entry bless whole warning
    families. ``_result`` injects a (rc, output) pair for hermetic tests.
    """
    aw = allowed_warnings or ALLOWED_WARNINGS
    if _result is None and _manuscript_absent() and not aw.exists():
        return ClauseResult("G2 verify_paper", True, _MANUSCRIPT_SKIP)
    rc, out = (
        _result
        if _result is not None
        else _run([sys.executable, "paper/verify_paper.py", "--results-dir", FROZEN_REL])
    )
    emitted = [
        " ".join(ln.strip()[2:].split()) for ln in out.splitlines() if ln.strip().startswith("- ")
    ]
    if not aw.exists():
        return ClauseResult("G2 verify_paper", False, "allowed_warnings.txt ABSENT (fail-closed)")
    allowed = [
        " ".join(ln.split())
        for ln in aw.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    extra = Counter(emitted) - Counter(allowed)
    missing = Counter(allowed) - Counter(emitted)
    ok = rc == 0 and not extra and not missing
    summ = (
        f"0 errors; warning multiset == manifest ({len(allowed)})"
        if ok
        else (
            f"exit={rc}; +{sum(extra.values())} unmanifested / "
            f"-{sum(missing.values())} manifested-but-absent warning(s)"
        )
    )
    return ClauseResult("G2 verify_paper", ok, summ)


def g3_number_audit() -> ClauseResult:
    rc, out = _run([sys.executable, "paper/number_audit.py", "--check"])
    last = next((ln for ln in out.splitlines() if "number audit" in ln), "")
    return ClauseResult("G3 number_audit", rc == 0, last.strip())


def g4_tables_match() -> ClauseResult:
    """Regenerate tables to a TEMP dir from the frozen archive and diff vs committed (R12)."""
    import tempfile

    committed = REPO / "paper" / "tables"
    if not committed.exists():
        return ClauseResult("G4 tables_match", False, "paper/tables/ missing")
    with tempfile.TemporaryDirectory() as td:
        rc, tbl_out = _run(
            [sys.executable, "paper/generate_tables.py", "--results", FROZEN_REL, "--output", td]
        )
        if rc != 0:
            # Surface the real cause (e.g. a missing optional dep like tabulate for
            # df.to_markdown) instead of an opaque exit code — the last non-blank output line.
            tail = next((ln for ln in reversed(tbl_out.splitlines()) if ln.strip()), "")
            return ClauseResult(
                "G4 tables_match", False, f"generate_tables exit={rc}: {tail.strip()[:160]}"
            )

        # Every regenerated file must match its committed counterpart, comparing content
        # with line endings normalized to LF — the tables are text, and a Windows-committed
        # (CRLF) table must still match a Linux-regenerated (LF) one in CI. Not content.
        def _same(a: Path, b: Path) -> bool:
            return a.read_bytes().replace(b"\r\n", b"\n") == b.read_bytes().replace(b"\r\n", b"\n")

        diffs = []
        for f in Path(td).iterdir():
            comm = committed / f.name
            if not comm.exists() or not _same(f, comm):
                diffs.append(f.name)
        # Prose must be UNMUTATED by regeneration (H7). Ignore CR-at-EOL differences: a
        # [TBD]-injection that re-emits the file with the platform line ending (LF on Linux
        # CI vs a CRLF-committed manuscript) is not a content mutation. `git diff
        # --ignore-cr-at-eol --quiet` exits nonzero only on a real (non-EOL) change.
        prose_rc, _ = _run(
            [
                "git",
                "diff",
                "--ignore-cr-at-eol",
                "--quiet",
                "--",
                "paper/skeleton.md",
                "paper/extended_data.md",
            ]
        )
        prose_dirty = prose_rc != 0
        # LEADERBOARD.md is a generated artifact too — regenerated from frozen (M6).
        lb_rc, _ = _run([sys.executable, "paper/regenerate_leaderboard.py", "--check"])
        lb_ok = lb_rc == 0
        ok = not diffs and not prose_dirty and lb_ok
        summ = (
            "tables + leaderboard match; prose unmutated"
            if ok
            else (
                f"{len(diffs)} table diff(s)"
                + ("; PROSE MUTATED" if prose_dirty else "")
                + ("; LEADERBOARD drift" if not lb_ok else "")
            )
        )
        return ClauseResult("G4 generated_match", ok, summ)


def g5_pytest() -> ClauseResult:
    # paper unit tests (NOT test_final_gate.py -> would recurse). test_number_audit optional.
    tests = [
        "tests/unit/test_paper_numbers.py",
        "tests/unit/test_paper_verify.py",
        "tests/unit/test_paper_data_consistency.py",
    ]
    for extra in ("tests/unit/test_number_audit.py",):
        if (REPO / extra).exists():
            tests.append(extra)
    rc, out = _run([sys.executable, "-m", "pytest", *tests, "-q", "--no-header"])
    last = next((ln for ln in reversed(out.splitlines()) if ln.strip()), "")
    return ClauseResult("G5 pytest", rc == 0, last.strip()[:80])


def g6_archive_unchanged() -> ClauseResult:
    rc, out = _run(["git", "status", "--porcelain", FROZEN_REL])
    ok = rc == 0 and not out.strip()
    return ClauseResult("G6 archive_unchanged", ok, "clean" if ok else "ARCHIVE MODIFIED")


def g7_verify_ledger() -> ClauseResult:
    ledger_set = [
        REPO / "paper" / "revision_ledger.md",
        REPO / "paper" / "DECISIONS.md",
        REPO / "paper" / "RESPONSE_BANK.md",
    ]
    missing = [p.name for p in ledger_set if not p.exists()]
    if len(missing) == len(ledger_set):
        return ClauseResult(
            "G7 verify_ledger", True, "internal revision ledgers not distributed -- skipped"
        )
    if missing:
        return ClauseResult("G7 verify_ledger", False, f"ledger set incomplete: {missing}")
    ledger = REPO / "paper" / "verify_ledger.py"
    if not ledger.exists():
        return ClauseResult("G7 verify_ledger", False, "verify_ledger.py missing")
    rc, _ = _run([sys.executable, "paper/verify_ledger.py"])
    return ClauseResult("G7 verify_ledger", rc == 0, f"exit={rc}")


def g8_derivations(derivations: Path | None = None, root: Path | None = None) -> ClauseResult:
    """Input-hash stamps: every G8 row's recorded input hash matches the CURRENT input.

    Catches the historical escape class (a derived frozen artifact older than its own
    source, e.g. group_error_calibration.json vs the re-frozen equity_analysis.json).
    PENDING stamps fail (fail-closed until the Phase 1 canonical re-freeze writes them);
    scope=external inputs verify when present locally and are skipped-with-note on CI.
    """
    r = root or REPO
    try:
        rows = gate_lib.parse_derivations(derivations)
    except (OSError, ValueError) as e:
        return ClauseResult("G8 derivations", False, f"DERIVATIONS.tsv unreadable: {e}")
    g8_rows = [x for x in rows if x.check == "G8"]
    problems: list[str] = []
    skipped = 0
    if not g8_rows:
        problems.append("no G8 rows in DERIVATIONS.tsv (fail-closed)")
    for row in g8_rows:
        artifact, inp = r / row.artifact, r / row.input
        if not artifact.exists():
            problems.append(f"MISSING artifact: {row.artifact}")
            continue
        if row.sha256 in ("", "-", "PENDING"):
            problems.append(f"PENDING stamp: {row.artifact} <- {row.input}")
            continue
        if not inp.exists():
            if row.scope == "external":
                skipped += 1
                continue
            problems.append(f"MISSING repo input: {row.artifact} <- {row.input}")
            continue
        if gate_lib.sha256_file(inp) != row.sha256:
            problems.append(f"STALE derivation: {row.artifact} <- {row.input} (hash mismatch)")
    ok = not problems
    summ = (
        f"all stamps match ({len(g8_rows)} rows"
        + (f", {skipped} external skipped" if skipped else "")
        + ")"
        if ok
        else f"{len(problems)} stamp problem(s); first: {problems[0][:70]}"
    )
    return ClauseResult("G8 derivations", ok, summ)


def g9_surfaces(root: Path | None = None, registry: Path | None = None) -> ClauseResult:
    """Non-manuscript surfaces carry no CONFLICTING concept renderings.

    Scans README.md and .zenodo.json in full, and CHANGELOG.md's LATEST release section
    only (older entries are legitimate history). Not default-deny — these surfaces are
    full of legitimate config/version literals; only the registry's known-bad aliases
    fail.
    """
    r = root or REPO
    try:
        rows = gate_lib.parse_concept_registry(registry)
    except (OSError, ValueError) as e:
        return ClauseResult("G9 surfaces", False, f"CONCEPT_REGISTRY.tsv unreadable: {e}")
    surfaces: list[tuple[str, str]] = []
    for name in ("README.md", ".zenodo.json"):
        p = r / name
        if p.exists():
            surfaces.append((name, p.read_text(encoding="utf-8")))
    cl = r / "CHANGELOG.md"
    if cl.exists():
        surfaces.append(
            (
                "CHANGELOG.md[latest release]",
                gate_lib.changelog_latest_release_section(cl.read_text(encoding="utf-8")),
            )
        )
    problems = gate_lib.scan_conflicts(surfaces, rows)
    ok = not problems
    summ = (
        f"no conflicts on {len(surfaces)} surface(s)"
        if ok
        else f"{len(problems)} conflict(s); first: {problems[0][:70]}"
    )
    return ClauseResult("G9 surfaces", ok, summ)


def g10_concepts(root: Path | None = None, registry: Path | None = None) -> ClauseResult:
    """Every concept row ACTIVE, and no conflicting renderings on manuscript surfaces.

    Fail-closed while rows are PENDING_PHASE2 / PENDING_DECIDE: canonical values do not
    exist until the Phase 1 re-freeze, and contested concepts need a user-ratified
    decision batch before activation (see CONCEPT_REGISTRY.tsv).
    """
    r = root or REPO
    try:
        rows = gate_lib.parse_concept_registry(registry)
    except (OSError, ValueError) as e:
        return ClauseResult("G10 concepts", False, f"CONCEPT_REGISTRY.tsv unreadable: {e}")
    problems = [
        f"concept '{x.concept}' status={x.status} (not ACTIVE)"
        for x in rows
        if x.status != "ACTIVE"
    ]
    active = [x for x in rows if x.status == "ACTIVE"]
    if active:
        surfaces = []
        for name in _MANUSCRIPT_NAMES:
            p = r / "paper" / name
            if p.exists():
                surfaces.append((name, p.read_text(encoding="utf-8")))
        problems.extend(gate_lib.scan_conflicts(surfaces, active))
    ok = not problems
    summ = (
        f"all {len(rows)} concepts ACTIVE; no manuscript conflicts"
        if ok
        else f"{len(problems)} problem(s); first: {problems[0][:70]}"
    )
    return ClauseResult("G10 concepts", ok, summ)


def g11_checksums(frozen: Path | None = None) -> ClauseResult:
    """Every frozen JSON hashes to its checksums.sha256 entry, and vice versa.

    This is the reviewer command the paper itself documents (sha256sum -c); it must pass.
    """
    fz = frozen or (REPO / FROZEN_REL)
    cs = fz / "checksums.sha256"
    if not cs.exists():
        return ClauseResult("G11 checksums", False, "checksums.sha256 missing")
    listed: dict[str, str] = {}
    for ln in cs.read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        parts = ln.split(None, 1)
        if len(parts) != 2:
            return ClauseResult("G11 checksums", False, f"malformed line: {ln[:50]!r}")
        listed[parts[1].strip().lstrip("*")] = parts[0].strip()
    problems: list[str] = []
    for name, want in listed.items():
        p = fz / name
        if not p.exists():
            problems.append(f"{name}: listed but missing")
        elif gate_lib.sha256_file(p) != want:
            problems.append(f"{name}: hash mismatch")
    for p in sorted(fz.glob("*.json")):
        if p.name not in listed:
            problems.append(f"{p.name}: frozen JSON not in checksums.sha256")
    ok = not problems
    summ = (
        f"{len(listed)} checksums verify"
        if ok
        else f"{len(problems)} problem(s); first: {problems[0][:60]}"
    )
    return ClauseResult("G11 checksums", ok, summ)


def g12_docx(derivations: Path | None = None, root: Path | None = None) -> ClauseResult:
    """Every tracked DOCX embeds the source-manifest hash of its CURRENT sources.

    Content-addressed (never SHA==HEAD: committing a regenerated DOCX always advances
    HEAD past any embedded commit SHA, so that predicate could never pass). A missing or
    mismatched embed means the user would upload a stale DOCX — the v1 blind spot.
    """
    r = root or REPO
    try:
        rows = gate_lib.parse_derivations(derivations)
    except (OSError, ValueError) as e:
        return ClauseResult("G12 docx", False, f"DERIVATIONS.tsv unreadable: {e}")
    artifacts = sorted({x.artifact for x in rows if x.check == "G12"})
    if not artifacts:
        return ClauseResult("G12 docx", False, "no G12 rows in DERIVATIONS.tsv (fail-closed)")
    if _manuscript_absent(r) and not any((r / a).exists() for a in artifacts):
        return ClauseResult("G12 docx", True, _MANUSCRIPT_SKIP)
    problems: list[str] = []
    for a in artifacts:
        p = r / a
        if not p.exists():
            problems.append(f"{a}: missing")
            continue
        prov = gate_lib.read_docx_provenance(p)
        if prov is None:
            problems.append(f"{a}: no embedded provenance (regenerate via convert_to_docx.py)")
            continue
        missing = gate_lib.manifest_missing(a, rows, r)
        if missing:
            problems.append(f"{a}: manifest input(s) missing: {', '.join(missing[:3])}")
            continue
        if prov["source_sha256"] != gate_lib.manifest_hash(a, rows, r):
            problems.append(f"{a}: STALE (embedded source hash != current sources)")
    ok = not problems
    summ = (
        f"{len(artifacts)} DOCX fresh vs sources"
        if ok
        else f"{len(problems)} problem(s); first: {problems[0][:70]}"
    )
    return ClauseResult("G12 docx", ok, summ)


def g13_claims(
    claims: Path | None = None,
    root: Path | None = None,
    frozen: Path | None = None,
    decisions: Path | None = None,
) -> ClauseResult:
    """Every pinned load-bearing claim sentence is present in its file, and its evidence holds.

    The claims-layer ratchet (v3): CLAIMS.tsv pins the abstract/discussion
    sentences that carry the paper's conclusions. Presence is checked after normalization —
    ``<!--...-->`` markers stripped, whitespace/EOL collapsed — so a legitimate marker re-point
    never trips it, but a reworded conclusion does. SIGNIFICANT / SUGGESTIVE_NS rows additionally
    require their frozen evidence artifact to exist; DECIDED rows cite a DECISIONS.md id.
    Fail-closed on no rows. Pinned sentences must live in hand-written prose, not in
    generator-written regions (tables/extended_data), so a table regen cannot silently rewrite one.
    """
    r = root or REPO
    fz = frozen or (r / FROZEN_REL)
    dec = decisions or (r / "paper" / "DECISIONS.md")
    claims_path = claims or gate_lib.CLAIMS_TSV
    if _manuscript_absent(r) and not claims_path.exists():
        return ClauseResult("G13 claims", True, _MANUSCRIPT_SKIP)
    try:
        rows = gate_lib.parse_claims(claims)
    except (OSError, ValueError) as e:
        return ClauseResult("G13 claims", False, f"CLAIMS.tsv unreadable: {e}")
    if not rows:
        return ClauseResult("G13 claims", False, "no claims pinned (fail-closed)")
    dec_text = dec.read_text(encoding="utf-8") if dec.exists() else ""
    cache: dict[str, str | None] = {}
    problems: list[str] = []
    for row in rows:
        if row.file not in cache:
            fp = r / row.file
            cache[row.file] = (
                gate_lib.normalize_claim_text(fp.read_text(encoding="utf-8"))
                if fp.exists()
                else None
            )
        norm = cache[row.file]
        if norm is None:
            problems.append(f"{row.id}: file missing {row.file}")
            continue
        if gate_lib.normalize_claim_text(row.sentence) not in norm:
            problems.append(f"{row.id}: pinned sentence not found in {row.file}")
            continue
        if row.strength in ("SIGNIFICANT", "SUGGESTIVE_NS") and not (fz / row.evidence).exists():
            problems.append(f"{row.id}: {row.strength} evidence missing ({row.evidence})")
        elif row.strength == "DECIDED" and dec.exists() and f"### {row.evidence}" not in dec_text:
            problems.append(
                f"{row.id}: DECIDED cites '### {row.evidence}' absent from DECISIONS.md"
            )
    ok = not problems
    summ = (
        f"{len(rows)} claims pinned + present"
        if ok
        else f"{len(problems)} problem(s); first: {problems[0][:70]}"
    )
    return ClauseResult("G13 claims", ok, summ)


def _xlsx_cells(path: Path) -> dict[str, list]:
    """Return {sheet_name: [list of non-empty cell values]} for content comparison.

    Content only — xlsx are zip archives whose bytes are nondeterministic (timestamps,
    ordering), so a byte compare is meaningless; the figure numbers live in the cells.
    """
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out: dict[str, list] = {}
    for ws in wb.worksheets:
        out[ws.title] = [v for row in ws.iter_rows(values_only=True) for v in row if v is not None]
    wb.close()
    return out


def _has_value(cells: list, target: float, tol: float = 5e-4) -> bool:
    return any(isinstance(v, (int, float)) and abs(float(v) - target) < tol for v in cells)


def g14_source_data(root: Path | None = None, frozen: Path | None = None) -> ClauseResult:
    """Per-figure Source Data regenerate from frozen and carry their sentinel values.

    Closes a historical blind spot: no prior clause regenerated ``paper/source_data/*.xlsx``, so a
    stale generator shipped Source Data that did not match Figs 2 and 4 while the gate stayed
    green. G14 (a) regenerates to a temp dir from the frozen archive (generator nonzero => red);
    (b) requires committed and regenerated FILE SETS to be equal (a lingering Fig5 or a dropped
    figure => red); (c) requires per-sheet cell-value equality via openpyxl (content only); and
    (d) asserts per-figure SENTINEL values read straight from the frozen JSON — not from the
    generator — so a generator that regenerates its own mistake into both files still fails.
    """
    import tempfile

    r = root or REPO
    fz = frozen or (r / FROZEN_REL)
    committed = r / "paper" / "source_data"
    if not committed.exists():
        return ClauseResult("G14 source_data", False, "paper/source_data/ missing")
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        return ClauseResult("G14 source_data", False, "openpyxl not installed (fail-closed)")

    with tempfile.TemporaryDirectory() as td:
        rc, out = _run(
            [
                sys.executable,
                "paper/generate_source_data.py",
                "--results",
                str(fz),
                "--output",
                td,
            ]
        )
        if rc != 0:
            tail = next((ln for ln in reversed(out.splitlines()) if ln.strip()), "")
            return ClauseResult(
                "G14 source_data", False, f"generate exit={rc}: {tail.strip()[:120]}"
            )

        committed_files = {p.name for p in committed.glob("*.xlsx")}
        regen_files = {p.name for p in Path(td).glob("*.xlsx")}
        if committed_files != regen_files:
            missing = regen_files - committed_files
            extra = committed_files - regen_files
            return ClauseResult(
                "G14 source_data",
                False,
                f"file-set mismatch: missing {sorted(missing)} / extra {sorted(extra)}",
            )

        for name in sorted(committed_files):
            try:
                a = _xlsx_cells(committed / name)
                b = _xlsx_cells(Path(td) / name)
            except (OSError, ValueError) as e:
                return ClauseResult("G14 source_data", False, f"{name} unreadable: {e}")
            if a != b:
                return ClauseResult(
                    "G14 source_data", False, f"{name} content differs from a frozen regeneration"
                )

        # Per-figure sentinels read from the FROZEN JSON, not the generator.
        try:
            mi = json.loads((fz / "monitoring_inequity.json").read_text(encoding="utf-8"))
            sc = json.loads((fz / "split_comparison.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            return ClauseResult("G14 source_data", False, f"frozen sentinel input unreadable: {e}")
        fig4 = _xlsx_cells(committed / "Fig4_source_data.xlsx")
        fig4_cells = [v for vals in fig4.values() for v in vals]
        if not _has_value(fig4_cells, mi["pct_people_of_color"]["monitoring_ratio"], 1e-4):
            return ClauseResult(
                "G14 source_data", False, "Fig4 missing monitoring_inequity sentinel (1.847)"
            )
        fig2 = _xlsx_cells(committed / "Fig2_source_data.xlsx")
        fig2_cells = [v for vals in fig2.values() for v in vals]
        simple = sc["simple"]
        for label, target in (
            ("random AUPRC 0.744", simple["random_split_metrics"]["auprc"]),
            ("geographic AUPRC 0.532", simple["geographic_split_metrics"]["auprc"]),
        ):
            if not _has_value(fig2_cells, target):
                return ClauseResult("G14 source_data", False, f"Fig2 missing sentinel ({label})")

    return ClauseResult(
        "G14 source_data",
        True,
        f"{len(committed_files)} Source Data files match frozen + sentinels",
    )


#: Ordered clauses. Injectable for the gate-the-gater test.
CLAUSES: tuple[Callable[[], ClauseResult], ...] = (
    g0_sticky_flag,
    g1_paper_numbers,
    g2_verify_paper,
    g3_number_audit,
    g4_tables_match,
    g5_pytest,
    g6_archive_unchanged,
    g7_verify_ledger,
    g8_derivations,
    g9_surfaces,
    g10_concepts,
    g11_checksums,
    g12_docx,
    g13_claims,
    g14_source_data,
)

#: Stop-hook fast subset: seconds, not minutes (consumed by the repo's local stop hook).
FAST_CLAUSES: tuple[Callable[[], ClauseResult], ...] = (
    g0_sticky_flag,
    g1_paper_numbers,
    g3_number_audit,
    g6_archive_unchanged,
    g8_derivations,
)


def run_clauses(clauses: tuple[Callable[[], ClauseResult], ...] = CLAUSES) -> list[ClauseResult]:
    return [c() for c in clauses]


def f_count(results: list[ClauseResult]) -> int:
    """F: the number of failing clauses. F == 0 is the definition of done."""
    return sum(1 for r in results if not r.ok)


def _load_expected() -> set[str] | None:
    """The calibrated-red expectation, or None for true-green mode (file absent)."""
    if not gate_lib.EXPECTED_RED_JSON.exists():
        return None
    try:
        data = json.loads(gate_lib.EXPECTED_RED_JSON.read_text(encoding="utf-8"))
        return set(data["expected_failing"])
    except (ValueError, KeyError, TypeError):
        return {"<EXPECTED_RED_JSON unreadable — fix or delete it>"}


def _git_head_and_dirty() -> tuple[str, int]:
    rc, head = _run(["git", "rev-parse", "HEAD"])
    rc2, porcelain = _run(["git", "status", "--porcelain"])
    sha = head.strip() if rc == 0 else "unknown"
    dirty = len([ln for ln in porcelain.splitlines() if ln.strip()]) if rc2 == 0 else -1
    return sha, dirty


def _evaluate(results: list[ClauseResult], expected: set[str] | None) -> int:
    """Print the verdict and return the exit code (shared by --check/--fast)."""
    failing = [r for r in results if not r.ok]
    failing_keys = {r.name.split()[0] for r in failing}
    ran_keys = {r.name.split()[0] for r in results}
    f = f_count(results)
    print(f"\nF = {f} failing clause(s): {sorted(failing_keys)}")
    if expected is not None:
        exp_here = expected & ran_keys
        if failing_keys == exp_here:
            print(
                f"✓ CALIBRATED-RED match: failing == expected {sorted(exp_here)} "
                "(true-green requires deleting paper/EXPECTED_RED.json after Phase 2)"
            )
            return 0
        print(
            f"✗ CALIBRATED-RED MISMATCH: failing={sorted(failing_keys)} "
            f"expected={sorted(exp_here)} "
            f"(unexpected={sorted(failing_keys - exp_here)}, "
            f"missing={sorted(exp_here - failing_keys)})"
        )
        return 1
    if f == 0:
        sha, dirty = _git_head_and_dirty()
        print("✓ FINAL GATE GREEN — done.")
        print(f"GATE GREEN sha={sha} dirty={dirty}")
        return 0
    print("✗ final gate red — resolve ≥1 clause and re-run (convergence loop, cap 3).")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="Gate mode (default): exit 0 iff green.")
    ap.add_argument(
        "--fast",
        action="store_true",
        help="Stop-hook fast subset (G0+G1+G3+G6+G8; seconds).",
    )
    ap.add_argument(
        "--clear-red-flag",
        action="store_true",
        help="Delete paper/GATE_RED.flag iff every other clause is green; else refuse.",
    )
    args = ap.parse_args()

    if args.clear_red_flag:
        others = tuple(c for c in CLAUSES if c is not g0_sticky_flag)
        results = run_clauses(others)
        print("Clear-red-flag preflight — G1..G12:")
        for r in results:
            print(f"  [{'PASS' if r.ok else 'FAIL'}] {r.name:24s} {r.summary}")
        if _evaluate(results, _load_expected()) == 0:
            gate_lib.GATE_RED_FLAG.unlink(missing_ok=True)
            print("GATE_RED.flag cleared.")
            return 0
        print("REFUSED: resolve the red clauses above first; the flag stays.")
        return 1

    clauses = FAST_CLAUSES if args.fast else CLAUSES
    results = run_clauses(clauses)
    label = "G0/G1/G3/G6/G8 (fast)" if args.fast else "G0..G14"
    print(f"Final gate — {label}:")
    for r in results:
        print(f"  [{'PASS' if r.ok else 'FAIL'}] {r.name:24s} {r.summary}")
    return _evaluate(results, _load_expected())


if __name__ == "__main__":
    sys.exit(main())
