#!/usr/bin/env python
"""Freeze the result JSONs that back the paper into a tracked snapshot.

``results.json`` embeds large per-entry arrays (``y_true``/``y_prob``/
``latitudes``/``longitudes``/``split_labels``, ~85 MB across all entries) that
GitHub would reject (>100 MB) and that the tables/figures do not need. This tool
slims those arrays out, copies the small ancillary analysis JSONs verbatim into a
tracked ``results/paper_frozen/`` snapshot, writes a SHA-256 manifest plus a
``MANIFEST.md`` describing the regime, and enforces a per-file size guard -- so
the committed Table 2 (and all derived numbers) are reproducible from
version-controlled inputs.

Two result regimes coexist in the snapshot: the canonical with-provenance run
(``--source``) keeps its original filenames, and the environment-only
``--provenance-free`` run (``--provenance-free-source``) is folded in under
``*_provenance_free.json`` names via an explicit allowlist -- the honest
"without provenance" column of the monitoring-invariance decomposition.

Usage::

    python paper/freeze_results.py --source results_reframe --frozen results/paper_frozen \\
        --provenance-free-source results_provenance_free --extra results/t6_arsenic.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

#: Large per-entry arrays in results.json metadata that tables/figures don't need.
_SLIM_ARRAY_FIELDS = ("y_true", "y_prob", "latitudes", "longitudes", "split_labels")

#: Filename substrings never copied into the frozen snapshot (backups, HPO dumps).
_SKIP_SUBSTRINGS = (".bak", "optuna_tuning", "checksums.sha256")

#: results.json variants that must be slimmed (rather than copied verbatim).
_SLIM_FILES = ("results.json", "results_random_split.json")

#: Hard per-file size guard for the tracked snapshot (MB).
_MAX_FILE_MB = 5.0

#: Provenance-free (environment-only) run files folded into the snapshot under
#: renamed targets so both regimes coexist in one tracked directory. This is an
#: explicit allowlist, NOT a glob: the provenance-free dir also holds same-named
#: files (monitoring_inequity.json, feature_importance.json, ...) that must not
#: overwrite the canonical-regime copies.
_PF_RENAME_MAP: dict[str, str] = {
    "results.json": "results_provenance_free.json",  # slimmed
    "loro_cv.json": "loro_cv_provenance_free.json",
    "multi_seed_stability.json": "multi_seed_stability_provenance_free.json",
    "equity_analysis.json": "equity_analysis_provenance_free.json",
    "bootstrap_ci.json": "bootstrap_ci_provenance_free.json",
    "shap_T1.json": "shap_T1_provenance_free.json",
}

#: Hint shown when a provenance-free input is missing.
_PF_HINTS = {
    "shap_T1.json": (
        " (generate it with: python scripts/reproduce.py --shap --provenance-free"
        " --output-dir results_provenance_free)"
    ),
}


def slim_results(src: Path, dst: Path) -> int:
    """Write ``src`` to ``dst`` with the large per-entry arrays removed.

    Returns the number of array fields dropped.
    """
    data = json.loads(src.read_text(encoding="utf-8"))
    removed = 0
    for entry in data:
        md = entry.get("metadata", {})
        for field in _SLIM_ARRAY_FIELDS:
            if field in md:
                del md[field]
                removed += 1
    dst.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return removed


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _write_checksums(frozen: Path, lines: list[str]) -> None:
    """Write checksums.sha256 with LF-only line endings.

    ``write_text`` translates ``\\n`` to CRLF on Windows, which breaks the documented
    ``sha256sum -c`` reviewer command on Unix (each filename gets a trailing ``\\r``). G11
    is CRLF-tolerant (Python parsing), but the bash command a reviewer runs is not; forcing
    LF keeps the reproducibility apparatus portable.
    """
    text = "\n".join(lines) + "\n"
    (frozen / "checksums.sha256").write_bytes(text.encode("utf-8"))


def _freeze_provenance_free(pf_source: Path, frozen: Path) -> list[str]:
    """Fold the environment-only run into ``frozen`` via the rename allowlist."""
    if not pf_source.exists():
        already = [t for t in _PF_RENAME_MAP.values() if (frozen / t).exists()]
        if len(already) == len(_PF_RENAME_MAP):
            print(
                f"  provenance-free source {pf_source} absent; keeping the "
                f"{len(already)} already-frozen *_provenance_free.json files"
            )
            return []
        raise SystemExit(
            f"Provenance-free source not found: {pf_source} (and the frozen snapshot "
            "is missing some *_provenance_free.json targets). Run the env-only "
            "pipeline (scripts/reproduce.py --all --provenance-free --output-dir "
            f"{pf_source}) or pass --provenance-free-source ''."
        )

    pf_files: list[str] = []
    for src_name, dst_name in _PF_RENAME_MAP.items():
        src = pf_source / src_name
        if not src.exists():
            raise SystemExit(f"Provenance-free input missing: {src}{_PF_HINTS.get(src_name, '')}")
        out = frozen / dst_name
        if src_name in _SLIM_FILES:
            slim_results(src, out)
        else:
            shutil.copy2(src, out)
        pf_files.append(dst_name)
    return pf_files


def freeze(
    source: Path,
    frozen: Path,
    *,
    pf_source: Path | None = None,
    extras: list[Path] | None = None,
) -> dict[str, object]:
    """Build the frozen snapshot. Returns a summary dict."""
    frozen.mkdir(parents=True, exist_ok=True)

    slimmed: list[tuple[str, int]] = []
    copied: list[str] = []
    for p in sorted(source.glob("*.json")):
        if any(s in p.name for s in _SKIP_SUBSTRINGS):
            continue
        out = frozen / p.name
        if p.name in _SLIM_FILES:
            slimmed.append((p.name, slim_results(p, out)))
        else:
            shutil.copy2(p, out)
            copied.append(p.name)

    pf_files: list[str] = []
    if pf_source is not None:
        pf_files = _freeze_provenance_free(pf_source, frozen)

    extra_files: list[str] = []
    for extra in extras or []:
        if not extra.exists():
            raise SystemExit(f"Extra result file not found: {extra}")
        shutil.copy2(extra, frozen / extra.name)
        extra_files.append(extra.name)

    # Size guard BEFORE writing the manifest.
    oversize = [
        (f.name, round(f.stat().st_size / 1e6, 2))
        for f in frozen.glob("*")
        if f.is_file() and f.stat().st_size > _MAX_FILE_MB * 1e6
    ]
    if oversize:
        raise SystemExit(
            f"Frozen files exceed {_MAX_FILE_MB} MB (slim or exclude them): {oversize}"
        )

    # SHA-256 manifest in the same format as data/checksums.sha256.
    json_files = sorted(f for f in frozen.glob("*.json"))
    lines = [f"{_sha256(f)}  {f.name}" for f in json_files]
    _write_checksums(frozen, lines)

    return {
        "slimmed": slimmed,
        "copied": copied,
        "pf_files": pf_files,
        "extras": extra_files,
        "n_files": len(json_files),
    }


def write_manifest(
    frozen: Path,
    source: Path,
    summary: dict[str, object],
    *,
    pf_source: Path | None = None,
    extras: list[Path] | None = None,
) -> None:
    pf_section = ""
    if pf_source is not None:
        renames = "\n".join(f"  - `{k}` -> `{v}`" for k, v in _PF_RENAME_MAP.items())
        pf_section = f"""
## Provenance-free (environment-only) regime

`*_provenance_free.json` files come from the `scripts/reproduce.py --all
--provenance-free` run in `{pf_source}/` -- the honest environmental-signal
model. That run excludes exactly the 10 provenance/monitoring features in
`PROVENANCE_FREE_EXCLUDE` (`src/aquacontam/features/assembly.py`): `n_samples`,
`mean_detection_limit`, `population_served`, `log_population_served`, and the
six `*_nan` one-hot missingness indicators (data-source proxies). Hydrology
(`source_water_type_*`) and `system_type_*` values are retained. Rename map:

{renames}

`results_provenance_free.json` is slimmed like `results.json`.
"""

    extras_section = ""
    if extras:
        extra_lines = "\n".join(f"  - `{e.name}` (from `{e}`)" for e in extras)
        extras_section = f"""
## Extra files

Result files produced outside the `--source` run directory, copied verbatim:

{extra_lines}

`t6_arsenic.json` is the T6 arsenic public-supply->domestic transfer result
(`scripts/reproduce.py --t6-arsenic`); T6 uses its own env-only NGA feature
cache and is unaffected by the provenance-free flag.
"""

    manifest = f"""# AquaContam frozen result snapshot

This directory holds the exact result JSONs that back the paper's tables, figures,
and reported numbers, version-controlled so reviewers can reproduce Table 2 et al.

- **Regime:** default-configuration, fixed seed 42, geographic split
  (train EPA regions 1,3,4,5,6 / val 2,7 / test 8,9,10).
- **Source run:** regenerated by `scripts/reproduce.py --all` into `{source}/`.
- **Slimming:** `results.json` (and `results_random_split.json`, if present) have the
  large per-entry arrays {list(_SLIM_ARRAY_FIELDS)} removed (~85 MB) -- the tables and
  Fig. 2 do not need them. The Extended-Data ROC/PR and national-risk-map figures
  need those arrays; re-run them locally from `{source}/` if exact reproduction is
  required.
- **Files:** {summary["n_files"]} JSONs (each < {_MAX_FILE_MB} MB; guard enforced).
- **Integrity:** `checksums.sha256` lists SHA-256 of every JSON here.
- **Software:** see `software_versions.json` in this directory.
{pf_section}{extras_section}
## Reviewer verification

```bash
# regenerate the tables from the frozen snapshot and confirm they match the committed ones
python paper/generate_tables.py --results results/paper_frozen --output /tmp/tables_check
python paper/verify_paper.py --results-dir results/paper_frozen   # hard-fails on any Table 2 mismatch
# verify file integrity
cd results/paper_frozen && sha256sum -c checksums.sha256
```
"""
    (frozen / "MANIFEST.md").write_text(manifest, encoding="utf-8")


def freeze_only(source: Path, frozen: Path, names: list[str]) -> None:
    """Surgically replace only the named JSONs in the frozen snapshot.

    Copies each named file from ``source`` into ``frozen`` (slimming the
    ``_SLIM_FILES``), then recomputes ``checksums.sha256`` over the files
    already present in ``frozen`` — so the other snapshot files can never be
    silently replaced by drifted copies from ``source``. ``MANIFEST.md`` is
    left untouched; record single-file provenance there manually.
    """
    if not frozen.exists():
        raise SystemExit(f"Frozen directory not found: {frozen}")
    for name in names:
        src = source / name
        if not src.exists():
            raise SystemExit(f"--only input missing: {src}")
        out = frozen / name
        if name in _SLIM_FILES:
            slim_results(src, out)
        else:
            shutil.copy2(src, out)
        if out.stat().st_size > _MAX_FILE_MB * 1e6:
            raise SystemExit(f"{name} exceeds {_MAX_FILE_MB} MB after freeze; slim or exclude it")

    json_files = sorted(f for f in frozen.glob("*.json"))
    lines = [f"{_sha256(f)}  {f.name}" for f in json_files]
    _write_checksums(frozen, lines)
    print(
        f"Replaced {len(names)} file(s) in {frozen}; checksums recomputed over {len(json_files)} JSONs"
    )


def rehash_only(frozen: Path) -> None:
    """Recompute ``checksums.sha256`` over the frozen dir without copying anything.

    Needed because the post-freeze generators (``compute_group_error_calibration`` /
    ``compute_inference_strengthening`` / ``compute_areal_apportionment``) write directly
    INTO the frozen dir, invalidating the manifest the full freeze wrote — the checksum
    re-hash must be the LAST step of the canonical DAG. Checksums stay written only by
    this script (never hand-edited).
    """
    if not frozen.exists():
        raise SystemExit(f"Frozen directory not found: {frozen}")
    json_files = sorted(f for f in frozen.glob("*.json"))
    lines = [f"{_sha256(f)}  {f.name}" for f in json_files]
    _write_checksums(frozen, lines)
    print(f"Recomputed checksums.sha256 over {len(json_files)} JSONs in {frozen}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="results_reframe", help="Source results directory.")
    ap.add_argument(
        "--frozen", default="results/paper_frozen", help="Tracked frozen-snapshot directory."
    )
    ap.add_argument(
        "--provenance-free-source",
        default="results_provenance_free",
        help="Environment-only (--provenance-free) run directory; pass '' to skip.",
    )
    ap.add_argument(
        "--extra",
        action="append",
        default=None,
        help=("Extra result JSON copied verbatim (repeatable). Default: results/t6_arsenic.json"),
    )
    ap.add_argument(
        "--only",
        default=None,
        help=(
            "Comma-separated JSON filename(s) to surgically replace in --frozen from "
            "--source, recomputing checksums from the frozen dir. Skips the full freeze "
            "and leaves MANIFEST.md untouched."
        ),
    )
    ap.add_argument(
        "--rehash-only",
        action="store_true",
        help=(
            "Recompute checksums.sha256 over --frozen and exit (no copying). The final "
            "DAG step after the direct-to-frozen compute_* generators."
        ),
    )
    args = ap.parse_args()

    if args.rehash_only:
        rehash_only(Path(args.frozen))
        return

    source, frozen = Path(args.source), Path(args.frozen)
    if not source.exists():
        raise SystemExit(f"Source directory not found: {source}")

    if args.only:
        freeze_only(source, frozen, [n.strip() for n in args.only.split(",") if n.strip()])
        return

    pf_source = Path(args.provenance_free_source) if args.provenance_free_source else None
    extras = [Path(e) for e in (args.extra or ["results/t6_arsenic.json"])]

    summary = freeze(source, frozen, pf_source=pf_source, extras=extras)
    write_manifest(frozen, source, summary, pf_source=pf_source, extras=extras)

    print(f"Froze {summary['n_files']} JSONs -> {frozen}")
    for name, n in summary["slimmed"]:  # type: ignore[misc]
        print(f"  slimmed {name} (-{n} array fields)")
    print(f"  copied {len(summary['copied'])} ancillary JSONs")  # type: ignore[arg-type]
    if summary["pf_files"]:
        print(f"  folded {len(summary['pf_files'])} provenance-free files")  # type: ignore[arg-type]
    if summary["extras"]:
        print(f"  copied {len(summary['extras'])} extra files")  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
