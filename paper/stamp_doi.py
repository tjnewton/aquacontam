#!/usr/bin/env python
"""Stamp the minted Zenodo DOI into the manuscript, then regenerate DOCX + re-run the gate.

The final human-boundary machine step of the submission runbook (R7). Before submission the
Data/Code-Availability statements carry ``(DOI: [to be inserted before submission])``
placeholders next to ``<!-- TODO: insert Zenodo DOI before submission -->`` comments. Once the
maintainer mints the Zenodo DOI, run:

    PYTHONUTF8=1 python paper/stamp_doi.py 10.5281/zenodo.XXXXXXX --apply

which (1) replaces every placeholder with the DOI (removing the TODO comment), (2) regenerates
all DOCX, and (3) runs the deterministic gate. Dry-run (no --apply) prints the planned edits.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGETS = [
    REPO / "paper" / "skeleton.md",
    REPO / "paper" / "reporting_summary.md",
    REPO / "paper" / "cover_letter.md",
    REPO / "LICENSE-DATA",
]
TODO_RE = re.compile(r"<!--\s*TODO: insert Zenodo DOI before submission\s*-->")
PLACEHOLDER = "[to be inserted before submission]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("doi", help="the minted Zenodo DOI, e.g. 10.5281/zenodo.1234567")
    ap.add_argument("--apply", action="store_true", help="write changes + regen DOCX + gate")
    args = ap.parse_args()

    doi = args.doi.strip()
    if not re.match(r"^10\.\d{4,9}/", doi):
        print(f"! DOI does not look like a DOI: {doi!r}")
        return 2

    total = 0
    for path in TARGETS:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        n = text.count(PLACEHOLDER)
        if n == 0:
            continue
        new = TODO_RE.sub("", text).replace(PLACEHOLDER, doi)
        total += n
        print(f"  {path.relative_to(REPO)}: {n} placeholder(s) -> {doi}")
        if args.apply:
            path.write_text(new, encoding="utf-8")

    if total == 0:
        print("No placeholders found (already stamped?). Nothing to do.")
        return 0

    if not args.apply:
        print(f"\nDRY RUN — {total} placeholder(s) would be stamped (pass --apply).")
        return 0

    print(f"\nStamped {total} placeholder(s). Regenerating DOCX + running the gate...")
    env = {"PYTHONUTF8": "1"}
    for cmd in (
        [sys.executable, "paper/convert_to_docx.py", "--all-docs"],
        [sys.executable, "paper/convert_to_docx.py", "--review"],
    ):
        rc = subprocess.run(cmd, cwd=REPO, env={**_env(), **env}).returncode
        if rc != 0:
            print(f"! DOCX regen failed: {cmd}")
            return rc
    print("\nRunning final gate (commit the DOI stamp first for a dirty=0 GATE GREEN):")
    rc = subprocess.run(
        [sys.executable, "paper/final_gate.py", "--check"], cwd=REPO, env={**_env(), **env}
    ).returncode
    return rc


def _env() -> dict[str, str]:
    import os

    return dict(os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
