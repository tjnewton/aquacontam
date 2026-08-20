# `paper/` — verification gate, frozen-archive tooling, and method record

This directory is **not the paper**. The manuscript sources (article text, extended
data, supplementary information, and their DOCX builds) are not distributed with this
repository; they accompany the journal submission. What lives here is the machinery
that makes the benchmark's results verifiable, plus the method record for the derived
artifacts.

## The gate

`final_gate.py` runs deterministic clauses G0–G14:

```bash
PYTHONUTF8=1 python paper/final_gate.py --check
```

In this distribution the benchmark clauses enforce:

- **G4** regenerates every table in `paper/tables/` and the leaderboard
  (`LEADERBOARD.md`) from the frozen archive and compares them to the committed copies.
- **G6 / G11** pin the frozen archive: `results/paper_frozen/` must be byte-clean
  against HEAD and match `checksums.sha256` in both directions.
- **G8** re-hashes every stamped derivation input recorded in `DERIVATIONS.tsv`.
- **G9 / G10** check the concept registry (`CONCEPT_REGISTRY.tsv`): rendering
  consistency on the shipped prose surfaces, and that every concept is active.
- **G5** runs the paper test files; **G14** regenerates the per-figure Source Data
  workbooks (`paper/source_data/`) and checks sentinel values read directly from the
  frozen JSONs.

The manuscript-consistency clauses (G1, G2, G3, G12, G13) report
`manuscript sources not distributed -- skipped` here and enforce where the manuscript
lives; G7 skips likewise for the internal revision ledgers. A skip fires only when the
entire manuscript set is absent — partial absence fails loud.

## The archive

`results/paper_frozen/` (at the repository root) is the frozen, checksummed copy of
record for the reported results. Nothing in this repository mutates it.
`freeze_results.py` documents how it was produced and is the only writer of its
checksum manifest.

## The method record

- `compute_*.py`, `splice_*.py`, `build_predictions_slim.py` — the scripts behind the
  post-freeze derived artifacts. The derivation map (artifact ← generator ← inputs)
  lives in `DERIVATIONS.tsv`, and G8 verifies the stamped input hashes.
- `generate_tables.py`, `generate_source_data.py`, `regenerate_leaderboard.py` — the
  generators that G4 and G14 run.
- `generate_figures.py`, `generate_figures_geo.py`, `style.mplstyle` — produce the
  article's figures from the frozen archive plus the pipeline's interim data. The
  figures themselves are not committed here; they accompany the journal article.
- `paper_numbers.py`, `verify_paper.py`, `number_audit.py` — the
  manuscript-consistency checkers behind G1/G2/G3. They ship so the skipped clauses
  remain inspectable, and `paper_numbers.py`'s marker registry is exercised against
  the frozen archive by the shipped tests.
- `frozen_snapshots/` — the dataset census that the concept registry declares as its
  authority. `assets/` — the state-boundary GeoJSON (hash-pinned by G8) used by the
  geographic-flag checks.

See the root README's "Repository provenance" section for how this fits the
distribution as a whole.
