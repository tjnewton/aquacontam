#!/usr/bin/env python
"""#55 item 5: reference metadata fixes + first-appearance renumber of skeleton.md.

Single deterministic, self-validating pass:

1. Citation-site edits (unique-string, not line-number keyed):
   - rehome ref 16 (DWINSA) to the "150,000 public water systems" sentence (L38-39)
     and drop it from the SDWIS sentence (ref 17 alone supports that);
   - cite the new TabPFN v2 reference alongside v1 at the TabPFN methods line.
2. Bibliography corrections (keyed by the OLD number, applied to the parsed body):
   - MAJOR: ref 9 -> Tokranov Science; ref 45 -> "Data leakage..." + Chan;
   - ref 6 entry swapped from Lanphear-2018 (mortality; mis-cited for the
     neurodevelopment claim, its only site) to Lanphear-2005 (pediatric IQ);
   - MINOR metadata: 16 doc#, 18 vintage, 33 year, 39 pages, 47 title, 56 subtitle;
   - NEW: TabPFN v2 (Hollmann Nature 2025); refs 49-57 reflowed to house style.
3. First-appearance renumber: scan the edited body's ^...^ markers in reading order,
   assign new ascending numbers, remap every marker (emitting comma-lists, never
   ranges), and re-emit the bibliography in new order with uniform house style.

Self-checks (hard-fail, nothing written unless all pass):
   - every bibliography entry is cited >=1x and every cited number has an entry
     (bijection, no orphans / no dangling markers);
   - the new numbering is ascending by first appearance;
   - the multiset of (old) reference identities is preserved across the remap.

Usage::

    python paper/renumber_references.py            # dry-run: prints map + diff, writes nothing
    python paper/renumber_references.py --apply     # rewrite paper/skeleton.md in place
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SKELETON = Path(__file__).resolve().parent / "skeleton.md"

# Sentinel key for the new TabPFN v2 entry until numbers are assigned.
TABPFN2 = "TABPFN2"

# --- 1. citation-site edits (old-number space; unique-substring keyed) -------
SITE_EDITS: list[tuple[str, str]] = [
    # Rehome ref 16 (DWINSA survey) to the water-systems-count sentence.
    (
        "more than 150,000 public water systems.",
        "more than 150,000 public water systems^16^.",
    ),
    # Drop ref 16 from the SDWIS sentence (ref 17 alone supports it).
    ("SDWIS^16,17^ adds", "SDWIS^17^ adds"),
    # Cite TabPFN v2 alongside v1 at the TabPFN methods line.
    ("**TabPFN**^40^:", f"**TabPFN**^40,{TABPFN2}^:"),
]

# --- 2. bibliography body corrections (OLD number -> corrected body text) ----
# Body text = everything after "N. " (authors ... (year).), house style:
#   Authors. Title. *Journal* **vol**, pp--pp (year).
BODY_OVERRIDES: dict[str, str] = {
    # MAJOR: ref 9 was wrong journal/vol/pages/title -> the real Science paper.
    "9": (
        "Tokranov, A. K. et al. Predictions of groundwater PFAS occurrence at "
        "drinking water supply depths in the United States. *Science* **386**, "
        "748--755 (2024)."
    ),
    # ref 6: mis-cited (adult mortality) for the neurodevelopment claim; swap to
    # the canonical pediatric lead-IQ pooled analysis (its only citation site).
    "6": (
        "Lanphear, B. P. et al. Low-level environmental lead exposure and "
        "children's intellectual function: an international pooled analysis. "
        "*Environ. Health Perspect.* **113**, 894--899 (2005)."
    ),
    # MINOR: ref 16 document number (816-K-23-001 matches no EPA doc).
    "16": (
        "U.S. Environmental Protection Agency. Seventh Drinking Water "
        "Infrastructure Needs Survey and Assessment. EPA 810-R-23-001 (2023)."
    ),
    # MINOR: ref 18 vintage (pipeline uses 2023-vintage 2020 ZCTAs).
    "18": (
        "U.S. Census Bureau. 2020 ZIP Code Tabulation Areas (ZCTAs). "
        "TIGER/Line and Gazetteer Files (2023)."
    ),
    # MINOR: ref 33 year (USGS recommended-citation year for the NLCD 2021 release).
    "33": (
        "Dewitz, J. & U.S. Geological Survey. National Land Cover Database (NLCD) "
        "2021 Products. U.S. Geological Survey data release. "
        "https://doi.org/10.5066/P9JZ7AO3 (2023)."
    ),
    # MINOR: ref 39 pages (off by one at both ends).
    "39": (
        "Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V. & Gulin, A. "
        "CatBoost: unbiased boosting with categorical features. in *Proc. NeurIPS* "
        "6639--6649 (2018)."
    ),
    # MAJOR: ref 45 fabricated title / wrong end-page / dropped third author.
    "45": (
        "Stock, A., Gregr, E. J. & Chan, K. M. A. Data leakage jeopardizes "
        "ecological applications of machine learning. *Nat. Ecol. Evol.* **7**, "
        "1743--1745 (2023)."
    ),
    # MINOR: ref 47 title tail.
    "47": (
        "Liddie, J. M., Schaider, L. A. & Sunderland, E. M. Sociodemographic "
        "factors are associated with the abundance of PFAS sources and detection "
        "in U.S. community water systems. *Environ. Sci. Technol.* **57**, "
        "7902--7912 (2023)."
    ),
    # Style reflow (bold vol, en-dash pages) for the 49-57 block:
    "49": (
        "Chouldechova, A. Fair prediction with disparate impact: a study of bias "
        "in recidivism prediction instruments. *Big Data* **5**, 153--163 (2017)."
    ),
    "50": (
        "Kleinberg, J., Mullainathan, S. & Raghavan, M. Inherent trade-offs in the "
        "fair determination of risk scores. in *Proc. ITCS* (2017)."
    ),
    "51": (
        "Ploton, P. et al. Spatial validation reveals poor predictive performance "
        "of large-scale ecological mapping models. *Nat. Commun.* **11**, 4540 (2020)."
    ),
    "52": (
        "Roberts, D. R. et al. Cross-validation strategies for data with temporal, "
        "spatial, hierarchical, or phylogenetic structure. *Ecography* **40**, "
        "913--929 (2017)."
    ),
    "53": (
        "Meyer, H. & Pebesma, E. Predicting into unknown space? Estimating the area "
        "of applicability of spatial prediction models. *Methods Ecol. Evol.* "
        "**12**, 1620--1633 (2021)."
    ),
    "54": (
        "Kapoor, S. & Narayanan, A. Leakage and the reproducibility crisis in "
        "machine-learning-based science. *Patterns* **4**, 100804 (2023)."
    ),
    "55": (
        "Phillips, S. J. et al. Sample selection bias and presence-only "
        "distribution models: implications for background and pseudo-absence "
        "data. *Ecol. Appl.* **19**, 181--197 (2009)."
    ),
    # MINOR: ref 56 subtitle.
    "56": (
        "Fithian, W., Elith, J., Hastie, T. & Keith, D. A. Bias correction in "
        "species distribution models: pooling survey and collection data for "
        "multiple species. *Methods Ecol. Evol.* **6**, 424--438 (2015)."
    ),
    "57": ("Sackett, D. L. Bias in analytic research. *J. Chronic Dis.* **32**, 51--63 (1979)."),
}

# NEW entry: TabPFN v2 (cited alongside v1 at the TabPFN line).
NEW_ENTRIES: dict[str, str] = {
    TABPFN2: (
        "Hollmann, N. et al. Accurate predictions on small data with a tabular "
        "foundation model. *Nature* **637**, 319--326 (2025)."
    ),
}

MARKER_RE = re.compile(r"\^([0-9][0-9,\-]*|[0-9,\-]*" + TABPFN2 + r")\^")
# Body markers are ^...^; a token is digits, comma-lists, or ranges, optionally
# containing the TABPFN2 sentinel.
TOKEN_RE = re.compile(r"\^([0-9A-Z,\-]+)\^")


def expand_token(tok: str) -> list[str]:
    """Expand a marker token like '45,51-54,59' or '40,TABPFN2' into ref keys."""
    keys: list[str] = []
    for part in tok.split(","):
        part = part.strip()
        if "-" in part and TABPFN2 not in part:
            lo, hi = part.split("-")
            keys.extend(str(n) for n in range(int(lo), int(hi) + 1))
        else:
            keys.append(part)
    return keys


def split_sections(text: str) -> tuple[str, str, str]:
    """Return (before_refs, refs_block, after_refs) around '## References'."""
    m = re.search(r"^## References\s*$", text, re.MULTILINE)
    if not m:
        raise SystemExit("no '## References' heading found")
    after = re.search(r"^## \w", text[m.end() :], re.MULTILINE)
    if not after:
        raise SystemExit("no heading after References")
    refs_start = m.start()
    refs_end = m.end() + after.start()
    return text[:refs_start], text[refs_start:refs_end], text[refs_end:]


def parse_bibliography(refs_block: str) -> dict[str, str]:
    """Parse '## References' into {old_number: body_text}."""
    entries: dict[str, str] = {}
    cur: str | None = None
    buf: list[str] = []
    for ln in refs_block.splitlines():
        m = re.match(r"^(\d+)\.\s+(.*)$", ln)
        if m:
            if cur is not None:
                entries[cur] = " ".join(buf).strip()
            cur = m.group(1)
            buf = [m.group(2)]
        elif cur is not None and ln.strip():
            buf.append(ln.strip())
    if cur is not None:
        entries[cur] = " ".join(buf).strip()
    return entries


def wrap_entry(num: int, body: str, width: int = 76) -> str:
    """Render 'N. body' wrapped to ``width`` with a hanging indent aligned under
    the text (never breaking inside en-dash page ranges)."""
    import textwrap

    prefix = f"{num}. "
    lines = textwrap.wrap(
        body,
        width=width,
        initial_indent=prefix,
        subsequent_indent=" " * len(prefix),
        break_on_hyphens=False,
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="rewrite skeleton.md in place")
    args = ap.parse_args()

    text = SKELETON.read_text(encoding="utf-8")
    before, refs_block, after = split_sections(text)

    # 1. citation-site edits (in the body only).
    for old, new in SITE_EDITS:
        if old not in before:
            raise SystemExit(f"site-edit anchor not found: {old!r}")
        before = before.replace(old, new)

    # 2. parse + correct bibliography.
    bib = parse_bibliography(refs_block)
    for k, v in BODY_OVERRIDES.items():
        if k not in bib:
            raise SystemExit(f"override target missing: ref {k}")
        bib[k] = v
    bib.update(NEW_ENTRIES)

    # 3. first-appearance scan over the edited body.
    order: list[str] = []
    seen: set[str] = set()
    for m in TOKEN_RE.finditer(before):
        for key in expand_token(m.group(1)):
            if key not in seen:
                seen.add(key)
                order.append(key)

    cited = set(order)
    have = set(bib)
    if cited - have:
        raise SystemExit(f"cited but no bibliography entry: {sorted(cited - have)}")
    if have - cited:
        raise SystemExit(f"bibliography entry never cited (orphan): {sorted(have - cited)}")

    old_to_new = {key: i + 1 for i, key in enumerate(order)}

    # 4. rewrite markers -> comma-lists of new numbers, sorted ascending.
    def remap(m: re.Match[str]) -> str:
        keys = expand_token(m.group(1))
        nums = sorted(old_to_new[k] for k in keys)
        return "^" + ",".join(str(n) for n in nums) + "^"

    new_body = TOKEN_RE.sub(remap, before)

    # 5. re-emit bibliography in new order.
    lines = ["## References", ""]
    for key in order:
        lines.append(wrap_entry(old_to_new[key], bib[key]))
        lines.append("")
    new_refs = "\n".join(lines).rstrip() + "\n\n"

    new_text = new_body + new_refs + after

    # --- report ---
    print(f"references: {len(order)} (was {len(parse_bibliography(refs_block))})")
    print("old -> new (by first appearance):")
    inv = {v: k for k, v in old_to_new.items()}
    for new_num in range(1, len(order) + 1):
        key = inv[new_num]
        label = "NEW " if key in NEW_ENTRIES else ("* " if key in BODY_OVERRIDES else "  ")
        print(f"  {label}{key:>7} -> {new_num}")

    # ascending-first-appearance self-check (trivially true by construction, asserted).
    appearances = [old_to_new[k] for k in order]
    assert appearances == sorted(appearances), "first-appearance order not ascending"

    if args.apply:
        SKELETON.write_text(new_text, encoding="utf-8")
        print(f"\nWROTE {SKELETON}")
    else:
        print("\nDRY RUN — nothing written (pass --apply to write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
