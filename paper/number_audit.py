#!/usr/bin/env python
"""Comprehensive reader-facing NUMBER AUDIT (default-deny).

Every numeric literal in the reader-facing files must land in exactly one bucket:

* **GATED**       — inside a ``<!--pn:ID-->VALUE<!--/pn-->`` marker whose ``ID`` is a  # noqa: RUF001
  registered ``paper_numbers`` id (its VALUE is then machine-checked against the frozen
  archive by ``paper_numbers.py --check`` / gate clause G1).
* **WHITELISTED** — a structural literal (year, citation index, figure/table/section/region  # noqa: RUF001
  reference, regulatory constant, list enumerator) matched by a typed, position-scoped
  predicate or an explicit ``number_whitelist.tsv`` entry.
* **SCOPED-OUT**  — a documented ``non-frozen-backed-config`` literal (hyperparameter grid  # noqa: RUF001
  size, software version, ICP config): no frozen source exists, low drift risk. Explicit
  per-position entries in ``number_whitelist.tsv`` (category ``config``).

Anything else FAILS (default-deny). This is gate clause **G3**: green ==> no ungated
reader-facing DERIVED number remains.

Completeness of the enumeration is proven by **two orthogonal, stdlib-only extractors**
(a regex pass over the raw text with comment-masking, and a hand-rolled character scanner
over the marker-stripped, structure-stripped text). Their canonical value multisets must
match; a mismatch is reported (H4). Residual (documented, out of scope): numbers embedded in
figure images / binary assets are not text and are not audited here.

Usage::

    python paper/number_audit.py --check     # gate: exit 0 iff nothing FAILS
    python paper/number_audit.py --report     # full per-bucket inventory (never fails)
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter
from pathlib import Path

try:  # dual import: `python paper/number_audit.py` (script) vs `paper.number_audit` (tests)
    from paper import paper_numbers as pn
except ImportError:  # pragma: no cover - script-dir-on-path fallback
    import paper_numbers as pn

REPO = pn.REPO
WHITELIST_TSV = pn.PAPER / "number_whitelist.tsv"
#: SCOPED-OUT config/version ranges (file<TAB>start<TAB>end<TAB>reason): non-frozen-backed
#: numbers (hyperparameter grids, software versions) that are documented-not-gated. Reviewed
#: once (adversarially) so no derived metric hides inside a range.
SCOPED_RANGES_TSV = pn.PAPER / "number_scoped_ranges.tsv"

#: Files audited. MUST be a subset of paper_numbers' scanned set (asserted by a test, H0),
#: so every marker placed in an audited file is also value-checked by G1.
AUDIT_FILES: tuple[Path, ...] = tuple(pn._default_files())

# --------------------------------------------------------------------------
# Tokenization
# --------------------------------------------------------------------------
#: thousands-grouped | decimal(+exp) | plain-int(+exp). Order matters (grouped first).
_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
#: any HTML comment (masked before tokenizing so pn-marker WRAPPERS / ids never tokenize;
#: the marker VALUE lives BETWEEN two comments so it survives masking).
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
#: fenced / inline code (values there are illustrative, not manuscript claims).
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
#: identifier codes that are NOT quantities — masked before tokenizing so their digit groups
#: never register as numeric literals: ORCID, DOI, dataset/model names with embedded digits
#: (UCMR5, UCMR3, CNN1D, 1D), EPSG codes, PWSIDs (CA0101001), CC0/3M/F1, region-id "3M".
_IDENT_RE = re.compile(
    r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dxX]\b|\b10\.\d{3,}/\S+|"
    r"\b[A-Za-z]{2,}\d[A-Za-z0-9]*\b|\bEPSG:\d+\b|\b[A-Z]{2}\d{7,}\b|\bCC0\b|\b3M\b|\bF\d\b|"
    r"\b\dD\b|\b[Ll]\d\b|Latin-\d|R\^\d"  # 1D / 2D / 3D; L1/L2 reg; Latin-1 encoding; R^2
)
#: markdown superscript references (^1^, ^12,13^) — affiliations / footnote markers.
_SUPERSCRIPT_RE = re.compile(r"\^\d+(?:[,\-]\d+)*\^")
#: markdown link / image target ("](figures/fig_ext8_data.png)") — a filename/URL, not a
#: manuscript quantity; its embedded digits (ext8, 8859) must not tokenize.
_LINK_RE = re.compile(r"\]\([^)]*\)")

#: Finite spelled-cardinal lexicon actually used as quantities (verified against the corpus).
_SPELLED = {
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "billion",
    "ninety-five",
    "thirty-eight",
    "forty-seven",
}
_SPELLED_RE = re.compile(r"(?i)\b(" + "|".join(sorted(_SPELLED, key=len, reverse=True)) + r")\b")

_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|?\s*$")


def _normalize(text: str) -> str:
    """NFKC + fold unicode minus / en / em dashes to ASCII so tokens don't fracture."""
    text = unicodedata.normalize("NFKC", text)
    return text.translate({0x2212: "-", 0x2013: "-", 0x2014: "-", 0x2010: "-", 0x00A0: " "})


def _canon(tok: str) -> str:
    """Canonical numeric string for A-vs-B comparison (drop thousands separators)."""
    return tok.replace(",", "")


def _mask(text: str, pattern: re.Pattern[str]) -> str:
    """Replace every ``pattern`` match with equal-length spaces (positions preserved)."""
    return pattern.sub(lambda m: " " * (m.end() - m.start()), text)


# --------------------------------------------------------------------------
# Whitelist (typed, position-scoped)
# --------------------------------------------------------------------------
_CATEGORIES = {"year", "citation", "structural_ref", "regulatory_constant", "enumerator", "config"}


def load_whitelist() -> dict[tuple[str, int, str], tuple[str, str]]:
    """(filename, line, canonical_value) -> (category, note). Missing file -> empty."""
    wl: dict[tuple[str, int, str], tuple[str, str]] = {}
    if not WHITELIST_TSV.exists():
        return wl
    for raw in WHITELIST_TSV.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        fname, lineno, value, category = parts[0], parts[1], parts[2], parts[3]
        note = parts[4] if len(parts) > 4 else ""
        if category not in _CATEGORIES:
            continue
        try:
            wl[(fname, int(lineno), _canon(value))] = (category, note)
        except ValueError:
            continue
    return wl


def load_scoped_ranges() -> dict[str, list[tuple[int, int, str]]]:
    """filename -> [(start_line, end_line, reason)] SCOPED-OUT config ranges."""
    ranges: dict[str, list[tuple[int, int, str]]] = {}
    if not SCOPED_RANGES_TSV.exists():
        return ranges
    for raw in SCOPED_RANGES_TSV.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        try:
            fname, start, end = parts[0], int(parts[1]), int(parts[2])
        except ValueError:
            continue
        ranges.setdefault(fname, []).append((start, end, parts[3] if len(parts) > 3 else "config"))
    return ranges


def _scoped_reason(
    ranges: dict[str, list[tuple[int, int, str]]], fname: str, line: int
) -> str | None:
    for start, end, reason in ranges.get(fname, []):
        if start <= line <= end:
            return reason
    return None


# --------------------------------------------------------------------------
# Auto-classification predicates (structural literals; conservative / default-deny)
# --------------------------------------------------------------------------
#: fixed regulatory / statistical constants (value, adjacent-keyword). Reported (derived)
#: p-values / effects are NOT in this set, so they still FAIL -> must be gated.
_REG_CONSTANTS = {
    "4", "10", "70", "2000", "1", "15",  # PFAS MCLs / 1 ng/L modeled level / 15 µg/L Pb action
    "2.0",                        # PFBS/PFHxS HBWC (µg/L); Hazard Index unit
    "0.05", "0.01", "0.001", "0.10", "0.20", "0.90", "0.95",  # significance / coverage levels
    "1.0", "1.5", "2.5", "3.0",   # Rosenbaum Gamma grid (2.0 already above)
    "0.5", "0.500",               # chance-level AUROC
}  # fmt: skip
_REG_KEYWORD_RE = re.compile(
    r"(?i)(ng/?l|µg/?l|ug/?l|ppb|ppt|mcl|hazard index|\balpha\b|α|gamma|Γ|"  # noqa: RUF001
    r"rosenbaum|significance|p\s*[<=]|p-value|p_?fdr|nominal|chance|coin\s*flip|perfect|"
    r"target|coverage)"
)
_STRUCT_KEYWORD_RE = re.compile(
    r"(?i)(figures?|fig\.?|tables?|sections?|panels?|appendix|supplementary(\s+(figure|table))?|"
    r"extended\s+data(\s+(figure|table))?|ed\s+(fig|table)|equations?|eq\.?|references?|ref\.?|"
    r"epa\s+region|region|regions|task|tasks|\bT\d?|\bS\d?|\bR\d?|steps?|phase|round|box|"
    r"stages?|ranks?|items?)[-\s]*$"
)
_ENUM_RE = re.compile(r"^\s*(?:\(?\d+[.)]|\d+\))\s")  # "1.", "2)", "(3)"
#: deliberately-cited historical / debunked values (the pre-leakage-fix artefacts the paper
#: contrasts against corrected numbers); mirrors verify_paper._INTENTIONAL_HISTORICAL_LITERALS.
_INTENTIONAL_HISTORICAL = {"0.962", "0.902"}


_MAGNITUDE_AFTER_RE = re.compile(r"(?i)\s*(percent|%|-?fold|times|-?point)")
#: confidence / coverage / significance keywords that make 90/95/99 a fixed statistical level.
_CI_KEYWORD_RE = re.compile(
    r"(?i)(\bci\b|confidence|coverage|nominal|interval|bootstrap|credible|significance)"
)
#: study-design descriptors: an integer naming a count of the experimental setup, not a metric.
_DESIGN_AFTER_RE = re.compile(
    r"(?i)^\s*-?\s*(fold|km\b|s\.?d\.?|seeds?|models?|model\s+families|families|"
    r"feature[-\s]sets?|features?|nearest|epa\s+regions?|regions?|sources?|states?|tasks?|"
    r"analytes?|folds?|classes?|compounds?|pfas|steps?|categories|panels?|dimensions?|"
    r"permutations?|perms?|bootstraps?|iterations?|resamples?|rounds?|draws?|blocks?|"
    r"buffers?|systems?|samples?|records?|contaminants?|databases?|train|validation|test|"
    r"holdout|matched|significant|wells?|characters?|sites?|configurations?|leaves|estimators?|"
    r"possible|neighbou?rs?|permutations?|conus|federal\s+sites?|ratio|split|"
    r"remaining\s+regions?|item)\b"
)
#: seed context ("seed 42", "seed-42", "seed=42", "seeds (42, 123)") — a fixed config value.
_SEED_BEFORE_RE = re.compile(r"(?i)seeds?[-\s=]*[(:]?\s*$")
#: software version ("Python >= 3.10", "v3.4") — a config version, not a metric.
_VERSION_BEFORE_RE = re.compile(r"(?i)(python|>=|version|>|v|numpy|pandas|torch|optuna)\s*$")
#: population / scale multiplier ("200 million Americans", "4+ million", "1.9M records") — scale.
_MULTIPLIER_AFTER_RE = re.compile(r"(?i)^\s*[+]?\s*(million|billion|thousand|M\b|B\b|K\b)")
#: ordinal ("90th percentile", "80th") — a fixed percentile cut, not a metric.
_ORDINAL_AFTER_RE = re.compile(r"(?i)^\s*(st|nd|rd|th)\b")
#: value immediately followed by CFR / FR ("40 CFR 141.86") — a statute citation.
_CFR_AFTER_RE = re.compile(r"(?i)^\s*(cfr|fr|c\.f\.r\.)\b")
#: inline enumerator ("that: (1) two ... (2) ...") — a list index in parentheses.
_PAREN_ENUM_RE = re.compile(r"\(\s*$")
#: tilde-approximated count ("~150,000 systems", "~97% negative") — an approximate scale
#: figure, never a precise frozen metric (a real metric is not written with a leading ~).
_APPROX_BEFORE_RE = re.compile(r"~\s*$")
#: count-approximation words ("more than 150,000", "at least 30") — applied only to integers
#: NOT followed by "%" (so a "roughly 40%" derived delta is still forced to gate).
_COUNT_APPROX_BEFORE_RE = re.compile(
    r"(?i)(more than|fewer than|at least|at most|over|under|roughly|approximately|about|nearly)\s*$"
)
#: spelled-cardinal magnitude (percent/fold/times) that must be GATED; "N points" is a rounded
#: descriptor ("about four points below"), not a precise frozen metric, so it is NOT included.
_SPELLED_MAGNITUDE_RE = re.compile(r"(?i)\s*(percent|%|-?fold|times)")
#: sample-size context ("n = 14,405", "n_high = 618") — a design count, not a metric.
_N_BEFORE_RE = re.compile(r"(?i)\bn(?:_[a-z]+)?\s*=\s*$")
#: CFR / Federal Register / statute citation ("40 CFR 141.86", "NN FR NNNNN", "§ N").
_CFR_BEFORE_RE = re.compile(r"(?i)(\bcfr\b|\bfr\b|c\.f\.r\.|§)\s*$")
#: license version ("CC BY 4.0", "CC BY-SA 3.0", "GPLv3", "Apache License 2.0"): a fixed
#: identifier of a public license, not a manuscript quantity. Audited prose writes the
#: spelled form "Apache License 2.0" (never hyphenated "Apache-2.0", whose hyphen defeats
#: the identifier mask).
_LICENSE_BEFORE_RE = re.compile(
    r"(?i)\bCC[- ]BY(?:[- ](?:SA|NC|ND))?[- ]$|\bGPLv$|\bApache License,?\s+(?:Version\s+)?$"
)


def auto_category(
    tok: Token, ctx_before: str, ctx_after: str, line_text: str, in_references: bool
) -> str | None:
    """Return a structural category if a typed predicate matches at this position, else None."""
    v = tok.canon
    # G9 rule: reader-facing TABLE-ROW DECIMALS are never predicate-whitelistable. A decimal
    # in a table cell is a reported quantity; it must be GATED, explicitly whitelisted
    # (number_whitelist.tsv), or in a reviewed scoped range — the loose "citation"/historical
    # shortcuts below must not bless it. (Typed positional predicates further down — year,
    # CI level, n=, CFR — are unaffected: they either exclude decimals or carry context.)
    table_decimal = tok.on_table and "." in v
    # references section: every number is bibliographic (volume / page / index / year).
    if in_references and not table_decimal:
        return "citation"
    # deliberately-cited historical / debunked value (pre-fix artefact vs corrected number).
    if v in _INTENTIONAL_HISTORICAL and not table_decimal:
        return "citation"
    # spelled cardinal: a study-design count ("sixteen model families", "seven tasks").
    # BUT if it quantifies a derived magnitude (percent / fold / -point) it must be GATED.
    if tok.kind == "spelled":
        # a spelled CI level ("Ninety-five percent confidence") is structural, not a metric.
        if v in ("ninety-five", "ninety-nine", "ninety", "ninety-eight"):
            return "regulatory_constant"
        return None if _SPELLED_MAGNITUDE_RE.match(ctx_after) else "structural_ref"
    is_int = v.isdigit()
    # year: 4-digit 1990..2099
    if is_int and len(v) == 4 and 1990 <= int(v) <= 2099:
        return "year"
    # confidence / coverage / significance level (95% CI, 90% coverage target): a fixed level
    if (
        is_int
        and v in ("80", "90", "95", "99")
        and ctx_after.lstrip().startswith("%")
        and _CI_KEYWORD_RE.search(ctx_before + " " + ctx_after)
    ):
        return "regulatory_constant"
    # study-design descriptor: an integer naming an experimental-setup count (5-km, 10 regions)
    if is_int and _DESIGN_AFTER_RE.match(ctx_after):
        return "structural_ref"
    # population / scale multiplier (200 million, 4 million) — approximate scale figure
    if is_int and _MULTIPLIER_AFTER_RE.match(ctx_after):
        return "structural_ref"
    # sample size (n = N, n_high = N) — a design count, not a metric
    if _N_BEFORE_RE.search(ctx_before):
        return "structural_ref"
    # CFR / Federal Register / statute citation (40 CFR 141.86)
    if _CFR_BEFORE_RE.search(ctx_before):
        return "citation"
    # public-license version ("CC BY 4.0") — a fixed identifier, not a quantity
    if _LICENSE_BEFORE_RE.search(ctx_before):
        return "regulatory_constant"
    # random seed (seed 42) — a fixed config value
    if _SEED_BEFORE_RE.search(ctx_before):
        return "config"
    # software version (Python >= 3.10) — a config version
    if _VERSION_BEFORE_RE.search(ctx_before):
        return "config"
    # ordinal percentile (90th, 80th percentile) — a fixed percentile cut
    if is_int and _ORDINAL_AFTER_RE.match(ctx_after):
        return "structural_ref"
    # statute citation where the number precedes CFR/FR (40 CFR 141.86)
    if _CFR_AFTER_RE.match(ctx_after):
        return "citation"
    # inline parenthesized enumerator ("that: (1) two ... (2) ...")
    if _PAREN_ENUM_RE.search(ctx_before):
        return "enumerator"
    # tilde-approximated value (~150,000 systems, ~97% negative, ~0.73 illustrative AUROC):
    # a leading ~ explicitly marks the number approximate, so it is never a precise frozen
    # metric — applies to decimals too (an author does not write a gated headline with a ~).
    if _APPROX_BEFORE_RE.search(ctx_before):
        return "structural_ref"
    # count-approximation ("more than 150,000", "at least 30") — integer, not a percentage
    if (
        is_int
        and not ctx_after.lstrip().startswith("%")
        and _COUNT_APPROX_BEFORE_RE.search(ctx_before)
    ):
        return "structural_ref"
    # EPA-region / split enumeration list ("regions 1,3,4,5,6"; "test: 8,9,10")
    if (
        is_int
        and int(v) <= 10
        and re.search(r"(?i)\b(region|train|validation|test|split|holdout)", line_text)
        and (
            ctx_before.rstrip().endswith(",")
            or ctx_after.lstrip().startswith(",")
            or ctx_before.rstrip().endswith("and")
        )
    ):
        return "structural_ref"
    # "top N" / "top N%" ranking-or-fraction descriptor ("top 15 pairs", "top 20% of risk").
    if is_int and re.search(r"(?i)\btop\s*$", ctx_before):
        return "structural_ref"
    # distance / unit enumeration ("uniform noise (1, 5, 10 km)") — design magnitudes, not
    # metrics; the trailing member is caught by _DESIGN_AFTER_RE, the earlier list members here.
    if (
        is_int
        and re.search(r"(?i)\b(km|kg|ng/l|ug/l)\b", line_text)
        and (ctx_after.lstrip().startswith(",") or ctx_before.rstrip().endswith("("))
    ):
        return "structural_ref"
    # numeric range with a unit ("5-25 km", "25-200 km buffers", "5/10/25 km") — design
    # magnitudes; the range's leading member (the trailing one is caught by _DESIGN_AFTER_RE).
    if is_int and re.match(
        r"(?i)^\s*[‒–—―/-]+\s*\d[\d,.]*\s*"  # noqa: RUF001
        r"(km|regions?|systems?|features?|analytes?|pfas|nm|kb|neighbou?rs?)\b",
        ctx_after,
    ):
        return "structural_ref"
    # slash-separated unit/ratio list ("counts within 5/10/25 km", "60/15/25 ratio") where the
    # line names the unit or a split ratio.
    if (
        is_int
        and (re.match(r"^\s*/\s*\d", ctx_after) or re.search(r"\d\s*/\s*$", ctx_before))
        and re.search(r"(?i)\b(km|ratio|split|regions?)\b", line_text)
    ):
        return "structural_ref"
    # "N of M <design-noun>" enumeration ("5 of 10 EPA regions", "3 of 5 folds")
    if is_int and re.match(
        r"(?i)^\s*of\s+\d+\s+(epa\s+)?"
        r"(regions?|folds?|features?|systems?|tasks?|categories|sources?|states?|analytes?|models?)",
        ctx_after,
    ):
        return "structural_ref"
    # bare small integer standing alone in a table cell (Region / rank id column): a label,
    # not a metric. Metrics are decimals (is_int is False for them), so this cannot swallow one.
    if (
        tok.on_table
        and is_int
        and int(v) <= 20
        and ctx_before.rstrip().endswith("|")
        and ctx_after.lstrip().startswith("|")
    ):
        return "structural_ref"
    # enumerator: markdown ordered-list index at line start
    if _ENUM_RE.match(line_text) and line_text.lstrip().startswith(v):
        return "enumerator"
    # structural reference: immediately follows a Figure/Table/Section/Region/... keyword
    if _STRUCT_KEYWORD_RE.search(ctx_before):
        return "structural_ref"
    # regulatory / statistical constant: a fixed value adjacent to a unit/keyword
    if v in _REG_CONSTANTS and _REG_KEYWORD_RE.search(ctx_before + " " + ctx_after):
        return "regulatory_constant"
    return None


# --------------------------------------------------------------------------
# Extractor A: regex over comment-masked raw text, bucketed by marker value spans
# --------------------------------------------------------------------------
class Token:
    __slots__ = ("canon", "in_marker", "kind", "line", "marker_id", "on_table", "start", "value")

    def __init__(self, value, canon, start, line, in_marker, marker_id, on_table, kind):
        self.value = value
        self.canon = canon
        self.start = start
        self.line = line
        self.in_marker = in_marker
        self.marker_id = marker_id
        self.on_table = on_table
        self.kind = kind  # "digit" | "spelled"


def _line_starts(text: str) -> list[int]:
    starts, pos = [0], text.find("\n")
    while pos != -1:
        starts.append(pos + 1)
        pos = text.find("\n", pos + 1)
    return starts


def _line_of(pos: int, starts: list[int]) -> int:
    import bisect

    return bisect.bisect_right(starts, pos)  # 1-indexed


def extract_a(text: str) -> list[Token]:
    """Tokens (digit + spelled) from raw text, with GATED / table classification."""
    text = _normalize(text)
    value_spans = [(m.start(2), m.end(2), m.group(1)) for m in pn._MARKER.finditer(text)]
    masked = _mask(text, _COMMENT_RE)  # kills wrapper ids + stray comments; keeps values
    masked = _mask(masked, _FENCE_RE)
    masked = _mask(masked, _INLINE_CODE_RE)
    masked = _mask(masked, _LINK_RE)
    masked = _mask(masked, _IDENT_RE)
    masked = _mask(masked, _SUPERSCRIPT_RE)
    starts = _line_starts(text)
    lines = text.split("\n")

    def marker_at(pos: int) -> str | None:
        for s, e, mid in value_spans:
            if s <= pos < e:
                return mid
        return None

    def on_table_row(lineno: int) -> bool:
        return bool(_TABLE_ROW_RE.match(lines[lineno - 1])) if 1 <= lineno <= len(lines) else False

    toks: list[Token] = []
    for m in _NUM_RE.finditer(masked):
        pos = m.start()
        lineno = _line_of(pos, starts)
        mid = marker_at(pos)
        toks.append(
            Token(
                m.group(),
                _canon(m.group()),
                pos,
                lineno,
                mid is not None,
                mid,
                on_table_row(lineno),
                "digit",
            )
        )
    for m in _SPELLED_RE.finditer(masked):
        pos = m.start()
        lineno = _line_of(pos, starts)
        mid = marker_at(pos)
        toks.append(
            Token(
                m.group().lower(),
                m.group().lower(),
                pos,
                lineno,
                mid is not None,
                mid,
                on_table_row(lineno),
                "spelled",
            )
        )
    return toks


# --------------------------------------------------------------------------
# Extractor B: independent hand-rolled scanner over stripped text (completeness proof)
# --------------------------------------------------------------------------
def _scan_number(text: str, i: int) -> tuple[str, int]:
    """Consume one number at position ``i`` by hand (mirrors _NUM_RE's grammar).

    A comma is a thousands separator ONLY when followed by exactly three digits, so
    region lists like ``1,3,4,5,6`` split into single integers (not merged), matching
    extractor A. Returns (token, next_index).
    """
    n = len(text)
    j = i
    while j < n and text[j].isdigit():
        j += 1
    # thousands groups: (,ddd) where the group is exactly 3 digits (not 2, not 4+)
    while (
        j + 3 < n
        and text[j] == ","
        and text[j + 1].isdigit()
        and text[j + 2].isdigit()
        and text[j + 3].isdigit()
        and not (j + 4 < n and text[j + 4].isdigit())
    ):
        j += 4
    # decimal part
    if j + 1 < n and text[j] == "." and text[j + 1].isdigit():
        j += 1
        while j < n and text[j].isdigit():
            j += 1
    # exponent
    if (
        j + 1 < n
        and text[j] in "eE"
        and (
            text[j + 1].isdigit() or (text[j + 1] in "+-" and j + 2 < n and text[j + 2].isdigit())
        )
    ):
        j += 1
        if text[j] in "+-":
            j += 1
        while j < n and text[j].isdigit():
            j += 1
    return text[i:j], j


def extract_b(text: str) -> Counter[str]:
    """Independent digit+spelled multiset from marker-stripped, structure-stripped text."""
    text = _normalize(text)
    text = pn.strip_markers(text)  # wrappers gone, VALUES kept (as a reader sees)
    text = _mask(text, _COMMENT_RE)
    text = _mask(text, _FENCE_RE)
    text = _mask(text, _INLINE_CODE_RE)
    text = _mask(text, _LINK_RE)
    text = _mask(text, _IDENT_RE)
    text = _mask(text, _SUPERSCRIPT_RE)

    counts: Counter[str] = Counter()
    n = len(text)
    i = 0
    while i < n:
        if not text[i].isdigit():
            i += 1
            continue
        tok, i = _scan_number(text, i)
        counts[_canon(tok)] += 1
    for m in _SPELLED_RE.finditer(text):
        counts[m.group().lower()] += 1
    return counts


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------
class FileResult:
    def __init__(self, name: str):
        self.name = name
        self.gated: list[Token] = []
        self.whitelisted: list[tuple[Token, str]] = []
        self.scoped_out: list[tuple[Token, str]] = []
        self.failed: list[Token] = []
        self.completeness_gap: list[str] = []
        self.unknown_marker: list[Token] = []


def audit_file(
    path: Path,
    whitelist: dict[tuple[str, int, str], tuple[str, str]],
    scoped_ranges: dict[str, list[tuple[int, int, str]]] | None = None,
) -> FileResult:
    scoped_ranges = scoped_ranges or {}
    text = path.read_text(encoding="utf-8")
    fname = path.name
    res = FileResult(fname)
    norm = _normalize(text)
    lines = norm.split("\n")
    toks = extract_a(text)

    # Everything from the "References"/"Bibliography" heading to the NEXT heading (or EOF)
    # is bibliographic. The zone is BOUNDED: content after a later heading (e.g. the embedded
    # "## Extended Data" section that follows "## References" in skeleton.md) is manuscript
    # prose again and MUST be audited. The previously unbounded zone was the M2/G9 escape:
    # every numeral in skeleton.md's embedded Extended Data tables (incl. the stale, ungated
    # MO DNR 0.872 at skeleton.md:1266) auto-classified as "citation" and evaded the gate.
    refs_start = len(lines) + 1
    refs_end = len(lines) + 1
    for i, ln in enumerate(lines, start=1):
        if re.match(r"^#+\s+(supplementary\s+)?(references|bibliography)\b", ln, re.IGNORECASE):
            refs_start = i
            for j in range(i + 1, len(lines) + 1):
                if re.match(r"^#+\s+", lines[j - 1]):
                    refs_end = j
                    break
            break

    for t in toks:
        line_text = lines[t.line - 1] if 1 <= t.line <= len(lines) else ""
        cb = norm[max(0, t.start - 40) : t.start]  # 40 chars before
        ca = norm[t.start + len(t.value) : t.start + len(t.value) + 24]  # 24 chars after

        if t.in_marker:
            if t.marker_id in pn.REGISTRY:
                res.gated.append(t)
            else:
                res.unknown_marker.append(t)  # defense-in-depth vs G1
            continue

        key = (fname, t.line, t.canon)
        if key in whitelist:
            category, note = whitelist[key]
            if category == "config":
                res.scoped_out.append((t, note or "config"))
            else:
                res.whitelisted.append((t, category))
            continue

        cat = auto_category(t, cb, ca, line_text, refs_start <= t.line < refs_end)
        if cat is not None:
            (res.scoped_out if cat == "config" else res.whitelisted).append((t, cat))
            continue

        sr = _scoped_reason(scoped_ranges, fname, t.line)
        if sr is not None:
            res.scoped_out.append((t, sr))
            continue

        res.failed.append(t)

    # completeness: A's visible digit+spelled multiset must equal B's
    a_counts: Counter[str] = Counter(t.canon for t in toks)
    b_counts = extract_b(text)
    if a_counts != b_counts:
        only_a = a_counts - b_counts
        only_b = b_counts - a_counts
        if only_a:
            res.completeness_gap.append(
                f"A-only tokens (regex found, scanner missed): {dict(only_a)}"
            )
        if only_b:
            res.completeness_gap.append(
                f"B-only tokens (scanner found, regex missed): {dict(only_b)}"
            )
    return res


def audit() -> tuple[list[FileResult], list[str]]:
    whitelist = load_whitelist()
    scoped_ranges = load_scoped_ranges()
    results = [audit_file(p, whitelist, scoped_ranges) for p in AUDIT_FILES if p.exists()]
    problems: list[str] = []
    for r in results:
        for t in r.failed:
            problems.append(
                f"{r.name}:{t.line}: ungated numeric literal {t.value!r} "
                f"({'table cell' if t.on_table else 'prose'}) — gate or whitelist it"
            )
        for t in r.unknown_marker:
            problems.append(
                f"{r.name}:{t.line}: marker id {t.marker_id!r} not in paper_numbers.REGISTRY"
            )
        problems.extend(f"{r.name}: completeness — {g}" for g in r.completeness_gap)
    return results, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="Gate: exit 1 if any literal FAILS.")
    g.add_argument(
        "--report", action="store_true", help="Full per-bucket inventory (never fails)."
    )
    args = ap.parse_args()

    results, problems = audit()
    tot_gated = sum(len(r.gated) for r in results)
    tot_wl = sum(len(r.whitelisted) for r in results)
    tot_so = sum(len(r.scoped_out) for r in results)
    tot_fail = sum(len(r.failed) for r in results)

    if args.report or not args.check:
        print("Number audit — per-file buckets (gated / whitelisted / scoped-out / FAIL):")
        for r in results:
            print(
                f"  {r.name:34s} "
                f"gated={len(r.gated):4d} wl={len(r.whitelisted):4d} "
                f"scoped={len(r.scoped_out):3d} FAIL={len(r.failed):4d}"
                + (
                    f"  [+{len(r.completeness_gap)} completeness gap]"
                    if r.completeness_gap
                    else ""
                )
                + (f"  [+{len(r.unknown_marker)} unknown-marker]" if r.unknown_marker else "")
            )
        print(
            f"  {'TOTAL':34s} gated={tot_gated:4d} wl={tot_wl:4d} scoped={tot_so:3d} FAIL={tot_fail:4d}"
        )

    if problems:
        print(f"\n✗ number audit: {len(problems)} problem(s)")
        for p in problems[:80]:
            print(f"  - {p}")
        if len(problems) > 80:
            print(f"  ... and {len(problems) - 80} more")
        return 1 if args.check else 0
    if args.check:
        print(
            f"✓ number audit: every literal GATED/WHITELISTED/SCOPED-OUT "
            f"(gated={tot_gated}, wl={tot_wl}, scoped={tot_so})"
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
