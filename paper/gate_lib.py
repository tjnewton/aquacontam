"""Shared helpers for the expanded final gate (G8-G12) and the DOCX provenance embed.

stdlib-only, so every consumer — ``final_gate.py`` clauses, ``convert_to_docx.py``,
tests, and the CI gate job — can import it without optional dependencies. python-docx is
needed only to WRITE the embed (``convert_to_docx.py``); reading it back uses ``zipfile``
on ``docProps/core.xml``.

Data files (both under ``paper/``):

``DERIVATIONS.tsv``
    artifact -> generator -> input -> sha256_at_derivation -> scope -> check.
    check=G8 rows are input-hash stamps (``PENDING`` = not yet stamped = FAIL);
    check=G12 rows define each DOCX's source manifest (hash embedded in the DOCX).

``CONCEPT_REGISTRY.tsv``
    One row per reader-facing dataset-scale concept: authority + allowed renderings +
    CONFLICTING rendering regexes + activation status. G9 scans non-manuscript surfaces
    for conflicts; G10 requires every row ACTIVE and scans manuscript surfaces.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"
DERIVATIONS_TSV = PAPER / "DERIVATIONS.tsv"
CONCEPT_REGISTRY_TSV = PAPER / "CONCEPT_REGISTRY.tsv"
EXPECTED_RED_JSON = PAPER / "EXPECTED_RED.json"
GATE_RED_FLAG = PAPER / "GATE_RED.flag"

# ---------------------------------------------------------------------------
# DERIVATIONS.tsv
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DerivationRow:
    artifact: str
    generator: str
    input: str
    sha256: str
    scope: str  # repo | external
    check: str  # G8 | G12 | G4 | G11 | PROV


def parse_derivations(path: Path | None = None) -> list[DerivationRow]:
    """Parse DERIVATIONS.tsv (tab-separated, '#' comments); strict on field count."""
    p = path or DERIVATIONS_TSV
    rows: list[DerivationRow] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) != 6:
            raise ValueError(
                f"DERIVATIONS.tsv: expected 6 tab-separated fields, got {len(parts)}: {raw!r}"
            )
        rows.append(DerivationRow(*(x.strip() for x in parts)))
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# DOCX source manifests (G12)
# ---------------------------------------------------------------------------


def manifest_files(
    artifact: str, rows: list[DerivationRow] | None = None, root: Path | None = None
) -> list[Path]:
    """All source files in `artifact`'s G12 manifest (directories expand recursively)."""
    root = root or REPO
    out: set[Path] = set()
    for r in rows if rows is not None else parse_derivations():
        if r.check != "G12" or r.artifact != artifact:
            continue
        p = root / r.input
        if p.is_dir():
            out.update(q for q in p.rglob("*") if q.is_file())
        elif p.exists():
            out.add(p)
        else:
            # A named-but-missing manifest input must poison the hash rather than
            # silently vanish; callers surface it via manifest_missing().
            out.add(p)
    return sorted(out, key=lambda q: q.relative_to(root).as_posix())


def manifest_missing(
    artifact: str, rows: list[DerivationRow] | None = None, root: Path | None = None
) -> list[str]:
    root = root or REPO
    return [
        f.relative_to(root).as_posix()
        for f in manifest_files(artifact, rows, root)
        if not f.exists()
    ]


# Text sources are hashed with line endings normalized to LF so a DOCX generated on one
# platform (Windows, CRLF) verifies against the same sources checked out on another (Linux,
# LF). Binary sources (figures) are hashed raw. Line endings are not content.
_TEXT_SUFFIXES = frozenset(
    {".md", ".csv", ".tex", ".txt", ".tsv", ".json", ".yaml", ".yml", ".bib", ".rst"}
)


def _manifest_bytes(f: Path) -> bytes:
    """File bytes for hashing: LF-normalized for text sources, raw for binary."""
    if not f.exists():
        return b"<MISSING>"
    data = f.read_bytes()
    if f.suffix.lower() in _TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return data


def manifest_hash(
    artifact: str, rows: list[DerivationRow] | None = None, root: Path | None = None
) -> str:
    """SHA-256 over (relpath NUL bytes NUL) of every manifest file, sorted by relpath.

    Commit-order-independent by construction: it depends only on source CONTENT, so a
    freshly generated DOCX stays verifiable after any number of unrelated commits
    (embedded-SHA==HEAD could never pass with git-tracked DOCX). Text bytes are
    LF-normalized so the hash is platform-independent.
    """
    root = root or REPO
    h = hashlib.sha256()
    for f in manifest_files(artifact, rows, root):
        h.update(f.relative_to(root).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(_manifest_bytes(f))
        h.update(b"\0")
    return h.hexdigest()


PROVENANCE_PREFIX = "aquacontam-provenance"
_PROV_RE = re.compile(
    PROVENANCE_PREFIX + r'\s+source_sha256=([0-9a-f]{64})\s+git=(\S+?)\s+generated=([^\s<"]+)'
)


def provenance_string(source_sha256: str, git_sha: str, generated: str) -> str:
    """The string convert_to_docx embeds in docProps/core.xml (dc:description).

    ``source_sha256`` (the manifest hash) is the GATED value; ``git``/``generated`` are
    display-only provenance for humans checking which upload they hold.
    """
    return f"{PROVENANCE_PREFIX} source_sha256={source_sha256} git={git_sha} generated={generated}"


def read_docx_provenance(docx_path: Path) -> dict[str, str] | None:
    """Extract the embedded provenance from a .docx (stdlib zipfile; no python-docx)."""
    try:
        with zipfile.ZipFile(docx_path) as z:
            core = z.read("docProps/core.xml").decode("utf-8", errors="replace")
    except (KeyError, OSError, zipfile.BadZipFile):
        return None
    m = _PROV_RE.search(core)
    if not m:
        return None
    return {"source_sha256": m.group(1), "git": m.group(2), "generated": m.group(3)}


# ---------------------------------------------------------------------------
# CONCEPT_REGISTRY.tsv (G9 / G10)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConceptRow:
    concept: str
    authority_type: str
    authority_path: str
    allowed_renderings: str
    conflicting: tuple[re.Pattern[str], ...]
    status: str  # ACTIVE | PENDING_PHASE2 | PENDING_DECIDE
    notes: str


_VALID_STATUSES = ("ACTIVE", "PENDING_PHASE2", "PENDING_DECIDE")


def parse_concept_registry(path: Path | None = None) -> list[ConceptRow]:
    """Parse CONCEPT_REGISTRY.tsv; `conflicting_renderings` is ';'-separated regexes."""
    p = path or CONCEPT_REGISTRY_TSV
    rows: list[ConceptRow] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) != 7:
            raise ValueError(
                f"CONCEPT_REGISTRY.tsv: expected 7 tab-separated fields, got {len(parts)}: {raw!r}"
            )
        concept, atype, apath, allowed, conflicting, status, notes = (x.strip() for x in parts)
        if status not in _VALID_STATUSES:
            raise ValueError(f"CONCEPT_REGISTRY.tsv: bad status {status!r} for {concept!r}")
        # Case-INSENSITIVE: a concept alias must be caught regardless of capitalization
        # (a review pass found "Sixteen model families" in Methods escape a case-sensitive
        # "\bsixteen" pattern — a real G10 gap).
        pats = tuple(
            re.compile(pat, re.IGNORECASE) for pat in conflicting.split(";") if pat.strip()
        )
        rows.append(ConceptRow(concept, atype, apath, allowed, pats, status, notes))
    return rows


def changelog_latest_release_section(text: str) -> str:
    """The FIRST versioned release section (skips [Unreleased]); '' if none.

    Older release entries are legitimate historical records — only the latest release
    describing the dataset with stale values is a defect (de novo §9-ii).
    """
    headings = [
        m
        for m in re.finditer(r"^## \[([^\]]+)\].*$", text, re.MULTILINE)
        if m.group(1).lower() != "unreleased"
    ]
    if not headings:
        return ""
    start = headings[0].start()
    nxt = text.find("\n## ", headings[0].end())
    return text[start : nxt if nxt != -1 else len(text)]


def scan_conflicts(surfaces: list[tuple[str, str]], rows: list[ConceptRow]) -> list[str]:
    """Scan (name, text) surfaces for any concept's conflicting renderings."""
    problems: list[str] = []
    for name, text in surfaces:
        for row in rows:
            for pat in row.conflicting:
                for m in pat.finditer(text):
                    line = text.count("\n", 0, m.start()) + 1
                    problems.append(
                        f"{name}:{line}: {m.group(0)!r} conflicts with concept '{row.concept}'"
                    )
    return problems


# ---------------------------------------------------------------------------
# CLAIMS.tsv (G13 — the claims-layer gate)
# ---------------------------------------------------------------------------

CLAIMS_TSV = PAPER / "CLAIMS.tsv"

#: A pinned load-bearing claim sentence and the strength class that governs it.
#: SIGNIFICANT / SUGGESTIVE_NS require a frozen evidence artifact; DECIDED cites a
#: DECISIONS.md id; DESCRIPTIVE needs only that the sentence is present.
_VALID_CLAIM_STRENGTHS = ("SIGNIFICANT", "SUGGESTIVE_NS", "DESCRIPTIVE", "DECIDED")
_MARKER_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


@dataclass(frozen=True)
class ClaimRow:
    id: str
    file: str
    sentence: str
    evidence: str
    strength: str


def normalize_claim_text(text: str) -> str:
    """Strip ``<!--...-->`` marker comments and collapse whitespace (EOL-agnostic).

    Both are mandatory: without stripping markers, every legitimate ``<!--pn:id-->``
    re-point would spuriously trip G13; without collapsing whitespace, line-wrapping would.
    """
    return re.sub(r"\s+", " ", _MARKER_COMMENT_RE.sub("", text)).strip()


def parse_claims(path: Path | None = None) -> list[ClaimRow]:
    """Parse CLAIMS.tsv (5 tab-separated fields, '#' comments); strict on field count."""
    p = path or CLAIMS_TSV
    rows: list[ClaimRow] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) != 5:
            raise ValueError(
                f"CLAIMS.tsv: expected 5 tab-separated fields, got {len(parts)}: {raw!r}"
            )
        cid, f, sentence, evidence, strength = (x.strip() for x in parts)
        if strength not in _VALID_CLAIM_STRENGTHS:
            raise ValueError(f"CLAIMS.tsv: bad strength {strength!r} for {cid!r}")
        rows.append(ClaimRow(cid, f, sentence, evidence, strength))
    return rows


# ---------------------------------------------------------------------------
# Degenerate-cell detector
# ---------------------------------------------------------------------------


def flag_degenerate_metric_cells(
    model_metrics: dict[str, dict[str, float]], keys: tuple[str, ...]
) -> list[str]:
    """Flag metric cells a table generator must render as FAILED, not plausible numbers.

    Two T3 failure modes this guards against: (a) two or more models emitting a
    **byte-identical** metric tuple (a silent non-convergence fallback masquerading as a
    real result — e.g. five T3 models all at micro-AUROC 0.753 / AUPRC 0.321), and (b) a
    **NaN** in a reported metric (e.g. macro_auroc NaN for every T3 model). Returns a list
    of human-readable problems; empty means every cell is a genuine, distinct number.

    Pure/stdlib so the generator (to relabel), a test, and a future gate clause can all
    share one definition of "degenerate".
    """
    import math

    problems: list[str] = []
    # (a) NaN cells
    for model, metrics in model_metrics.items():
        for k in keys:
            v = metrics.get(k)
            if isinstance(v, float) and math.isnan(v):
                problems.append(f"{model}: {k} is NaN (non-converged, must be flagged not shown)")
    # (b) byte-identical metric tuples across >=2 models
    seen: dict[tuple[float, ...], list[str]] = {}
    for model, metrics in model_metrics.items():
        tup = tuple(metrics.get(k, float("nan")) for k in keys)
        if any(isinstance(x, float) and math.isnan(x) for x in tup):
            continue  # NaN rows already flagged above
        seen.setdefault(tup, []).append(model)
    for tup, models in seen.items():
        if len(models) >= 2:
            problems.append(
                f"models {sorted(models)} share byte-identical {keys} = {tup} "
                "(degenerate; flag as non-converged)"
            )
    return problems


def degenerate_metric_models(
    model_metrics: dict[str, dict[str, float]], keys: tuple[str, ...]
) -> set[str]:
    """Return the set of models whose metric cells are degenerate over ``keys``.

    Rendering-side companion to :func:`flag_degenerate_metric_cells` (which returns
    human-readable problem strings for a gate/test): a model is degenerate if any of its
    ``keys`` is NaN, or if its ``keys`` tuple is byte-identical to another model's. The
    table generator relabels exactly these models "non-converged" so a silent
    non-convergence fallback never renders as a plausible number. One shared definition.
    """
    import math

    bad: set[str] = set()
    seen: dict[tuple[float, ...], list[str]] = {}
    for model, metrics in model_metrics.items():
        tup = tuple(metrics.get(k, float("nan")) for k in keys)
        if any(isinstance(x, float) and math.isnan(x) for x in tup):
            bad.add(model)
            continue
        seen.setdefault(tup, []).append(model)
    for _tup, models in seen.items():
        if len(models) >= 2:
            bad.update(models)
    return bad
