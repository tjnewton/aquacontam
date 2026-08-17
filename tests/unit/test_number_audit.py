"""Escape-replay tests for the number audit + gate_lib (FINAL_FIX_CONTRACT_v2, Phase 0).

Each replay reproduces an ACTUAL historical escape at clause level against planted
fixtures — not an easy synthetic variant. Red-on-defect must hold now; the paired
green-on-clean fixture asserts the state the Phase 2 re-sync produces. This file is part
of the gate's own G5 subset, so it must stay hermetic and fast (no subprocesses).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper"))

import gate_lib  # noqa: E402
import number_audit as na  # noqa: E402
import paper_numbers as pn  # noqa: E402

# ---------------------------------------------------------------------------
# Escape 1 (de novo M2): embedded Extended Data AFTER "## References" evaded the audit
# because the references zone was unbounded — every post-References numeral
# auto-classified "citation". Byte-level replay of skeleton.md:1266's stale cell.
# ---------------------------------------------------------------------------

_ED_DEFECT = """# Title

Body prose with a gated value <!--pn:mo_ext_auroc-->0.829<!--/pn--> here.

## References

1. Foo, A. & Bar, B. Water things. *J. Water* 12, 345-360 (2020).

## Extended Data

### Extended Data Table 5: external validation

| State DB | EPA Region | Split Role | n Systems | AUROC | AUPRC | Detection Rate |
|----------|-----------|------------|-----------|-------|-------|----------------|
| MO DNR | R7 | val | 1,208 | 0.872 | 0.408 | 5.2% |
"""

_ED_CLEAN = _ED_DEFECT.replace(
    "| MO DNR | R7 | val | 1,208 | 0.872 | 0.408 | 5.2% |",
    "| MO DNR | R7 | val | <!--pn:mo_ext_n-->1,208<!--/pn--> | "
    "<!--pn:mo_ext_auroc-->0.829<!--/pn--> | <!--pn:mo_ext_auprc-->0.281<!--/pn--> | "
    "<!--pn:mo_ext_det-->0.052<!--/pn-->% |",
)


def _audit_text(tmp_path: Path, text: str) -> na.FileResult:
    p = tmp_path / "skeleton.md"
    p.write_text(text, encoding="utf-8")
    return na.audit_file(p, whitelist={}, scoped_ranges={})


def test_embedded_ed_after_references_is_audited(tmp_path):
    """Red-on-defect: the stale unmarked 0.872 table cell must FAIL, not 'citation'."""
    res = _audit_text(tmp_path, _ED_DEFECT)
    failed_values = {t.canon for t in res.failed}
    assert "0.872" in failed_values, "the M2 escape cell must be caught"
    assert "0.408" in failed_values
    assert "1208" in failed_values  # canon form strips the thousands separator


def test_references_zone_still_whitelists_bibliography(tmp_path):
    """The bounded zone keeps genuine bibliography numerals citation-classified."""
    res = _audit_text(tmp_path, _ED_DEFECT)
    wl_values = {t.canon for t, _cat in res.whitelisted}
    # volume 12, pages 345-360, year 2020 live INSIDE the references zone
    assert {"12", "345", "360"} <= wl_values
    failed_lines = {t.line for t in res.failed}
    refs_line = _ED_DEFECT.splitlines().index("## References") + 1
    ed_line = _ED_DEFECT.splitlines().index("## Extended Data") + 1
    assert not any(refs_line < ln < ed_line for ln in failed_lines)


def test_embedded_ed_green_when_gated(tmp_path):
    """Green-on-clean: the same row fully pn-gated passes (Phase 2 target state)."""
    res = _audit_text(tmp_path, _ED_CLEAN)
    assert not res.failed, [t.value for t in res.failed]
    gated_ids = {t.marker_id for t in res.gated}
    assert {"mo_ext_auroc", "mo_ext_auprc", "mo_ext_n"} <= gated_ids


# ---------------------------------------------------------------------------
# Escape 2 (G9 rule): a table-row DECIMAL must never be predicate-whitelisted — the
# historical-value shortcut blessed the debunked 0.962 inside an SI table cell.
# ---------------------------------------------------------------------------


def test_table_row_decimal_never_predicate_whitelisted(tmp_path):
    text = "| model | AUROC |\n|---|---|\n| legacy pre-fix | 0.962 |\n"
    res = _audit_text(tmp_path, text)
    assert "0.962" in {t.canon for t in res.failed}


def test_historical_decimal_in_prose_still_whitelisted(tmp_path):
    text = "The debunked pre-fix value 0.962 is cited deliberately here.\n"
    res = _audit_text(tmp_path, text)
    assert "0.962" in {t.canon for t, _cat in res.whitelisted}
    assert not res.failed


def test_license_version_predicate_covers_apache(tmp_path):
    """'Apache License 2.0' is a fixed license identifier, not a manuscript quantity."""
    text = (
        "The code is available under the Apache License 2.0 and the data under "
        "CC BY 4.0 where source terms permit.\n"
    )
    res = _audit_text(tmp_path, text)
    assert not res.failed, [t.value for t in res.failed]
    wl_values = {t.canon for t, _cat in res.whitelisted}
    assert {"2.0", "4.0"} <= wl_values


def test_bare_two_point_zero_without_license_context_still_fails(tmp_path):
    """The Apache extension must not bless a bare 2.0 outside a license phrase."""
    res = _audit_text(tmp_path, "The improvement was 2.0 across models.\n")
    assert "2.0" in {t.canon for t in res.failed}


# ---------------------------------------------------------------------------
# Escape 3 (de novo M7 / 95,223<->91,061 class): the abstract count is registry-gated,
# so a re-plant is caught by G1 mechanics (covered in test_paper_numbers). Regression:
# the ids stay present and resolve to the frozen snapshot values.
# ---------------------------------------------------------------------------


def test_dataset_scale_ids_present_and_resolve():
    assert pn.expected("ds_systems") == "95,223"
    assert pn.expected("ds_geocoded") == "87,450"
    assert pn.expected("mo_ext_auroc") == "0.829"


# ---------------------------------------------------------------------------
# gate_lib: DERIVATIONS.tsv / manifests / provenance string / CHANGELOG slicing
# ---------------------------------------------------------------------------


def test_parse_derivations_strict_field_count(tmp_path):
    p = tmp_path / "DERIVATIONS.tsv"
    p.write_text("# comment\na\tb\tc\td\trepo\tG8\n", encoding="utf-8")
    rows = gate_lib.parse_derivations(p)
    assert rows[0].artifact == "a" and rows[0].check == "G8"
    p.write_text("a\tb\tc\n", encoding="utf-8")
    with pytest.raises(ValueError, match="6 tab-separated"):
        gate_lib.parse_derivations(p)


def test_repo_derivations_tsv_parses_and_covers_the_escape_class():
    rows = gate_lib.parse_derivations()
    g8 = {(r.artifact, r.input) for r in rows if r.check == "G8"}
    # the actual M1 escape edge must be stamped
    assert (
        "results/paper_frozen/group_error_calibration.json",
        "results/paper_frozen/equity_analysis.json",
    ) in g8
    assert (
        "results/paper_frozen/inference_strengthening.json",
        "results/paper_frozen/equity_analysis.json",
    ) in g8
    g12_artifacts = {r.artifact for r in rows if r.check == "G12"}
    assert {
        "paper/manuscript.docx",
        "paper/extended_data.docx",
        "paper/supplementary_information.docx",
        "paper/manuscript_review.docx",
        "paper/plain_language_primer.docx",
    } <= g12_artifacts


def test_manifest_hash_changes_with_source_content(tmp_path):
    (tmp_path / "paper").mkdir()
    src = tmp_path / "paper" / "skeleton.md"
    src.write_text("v1", encoding="utf-8")
    d = tmp_path / "DERIVATIONS.tsv"
    d.write_text(
        "paper/m.docx\tpaper/convert_to_docx.py\tpaper/skeleton.md\t-\trepo\tG12\n",
        encoding="utf-8",
    )
    rows = gate_lib.parse_derivations(d)
    h1 = gate_lib.manifest_hash("paper/m.docx", rows, tmp_path)
    src.write_text("v2", encoding="utf-8")
    h2 = gate_lib.manifest_hash("paper/m.docx", rows, tmp_path)
    assert h1 != h2


def test_provenance_string_roundtrip():
    s = gate_lib.provenance_string("ab" * 32, "deadbeef123", "2026-07-01T00:00:00Z")
    m = gate_lib._PROV_RE.search(f"<dc:description>{s}</dc:description>")
    assert m and m.group(1) == "ab" * 32 and m.group(2) == "deadbeef123"


def test_changelog_latest_release_section_slicing():
    text = (
        "# Changelog\n\n## [Unreleased]\n- pending 10,746\n\n"
        "## [3.0.0] - 2026-03-03\n- expanded dataset to 10,746 systems\n\n"
        "## [2.0.0] - 2026-02-27\n- old stuff 10,746 systems\n"
    )
    sec = gate_lib.changelog_latest_release_section(text)
    assert sec.startswith("## [3.0.0]")
    assert "expanded dataset" in sec
    assert "old stuff" not in sec and "pending" not in sec


def test_scan_conflicts_reports_line_numbers():
    rows = gate_lib.parse_concept_registry()
    hits = gate_lib.scan_conflicts(
        [("CHANGELOG.md[latest release]", "line1\n- expanded dataset to 10,746 systems\n")],
        rows,
    )
    assert any("10,746 systems" in h and ":2:" in h for h in hits)


def test_repo_concept_registry_parses():
    rows = gate_lib.parse_concept_registry()
    names = {r.concept for r in rows}
    assert {"sdwis_rows", "model_family_count", "geocoded_count"} <= names
    assert all(r.status in ("ACTIVE", "PENDING_PHASE2", "PENDING_DECIDE") for r in rows)


# ---------------------------------------------------------------------------
# stamp_derivations: writes the G8 stamps the gate then enforces — must be exact
# ---------------------------------------------------------------------------


def test_stamp_derivations_writes_current_input_hashes(tmp_path):
    import stamp_derivations as sd

    inp = tmp_path / "equity.json"
    inp.write_text('{"v": 1}', encoding="utf-8")
    tsv = tmp_path / "DERIVATIONS.tsv"
    tsv.write_text(
        "# comment preserved\n"
        "frozen/gec.json\tgen.py\tequity.json\tPENDING\trepo\tG8\n"
        "paper/m.docx\tconv.py\tskeleton.md\t-\trepo\tG12\n",
        encoding="utf-8",
    )
    assert sd.stamp(derivations=tsv, root=tmp_path, write=True) == 0
    out = tsv.read_text(encoding="utf-8")
    assert "# comment preserved" in out
    assert gate_lib.sha256_file(inp) in out  # G8 row stamped
    assert "\t-\trepo\tG12" in out  # non-G8 rows untouched


def test_stamp_derivations_refuses_missing_input(tmp_path):
    import stamp_derivations as sd

    tsv = tmp_path / "DERIVATIONS.tsv"
    tsv.write_text("frozen/a.json\tgen.py\tabsent.json\tPENDING\trepo\tG8\n", encoding="utf-8")
    assert sd.stamp(derivations=tsv, root=tmp_path, write=True) == 1
    # nothing written on problems: the PENDING stamp (and G8 red) must survive
    assert "PENDING" in tsv.read_text(encoding="utf-8")
