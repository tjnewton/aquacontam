#!/usr/bin/env python
"""Write the G8 input-hash stamps into paper/DERIVATIONS.tsv.

Run as the LAST step of the canonical re-freeze DAG, immediately after the
direct-to-frozen generators and the checksum re-hash: every G8 row's input is hashed AT
THIS MOMENT, when the artifacts were just derived from these exact inputs. The gate
(final_gate.py G8) then fails if any input ever changes out from under its derived
artifact — the M1 stale-derivation class becomes impossible to ship silently.

Fails (exit 1) if any G8 input is missing: stamping a derivation whose input is absent
would be a lie.

Usage::

    python paper/stamp_derivations.py            # stamp all G8 rows in place
    python paper/stamp_derivations.py --check    # report what would change; never writes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from paper import gate_lib
except ImportError:  # pragma: no cover - script-dir-on-path fallback
    import gate_lib


def stamp(derivations: Path | None = None, root: Path | None = None, write: bool = True) -> int:
    """Rewrite the sha256 column of every G8 row; returns the number of problems."""
    path = derivations or gate_lib.DERIVATIONS_TSV
    root = root or gate_lib.REPO
    problems: list[str] = []
    out_lines: list[str] = []
    n_stamped = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            out_lines.append(raw)
            continue
        parts = raw.split("\t")
        if len(parts) != 6:
            problems.append(f"malformed row (expected 6 fields): {raw!r}")
            out_lines.append(raw)
            continue
        artifact, generator, inp, _sha, scope, check = (x.strip() for x in parts)
        if check != "G8":
            out_lines.append(raw)
            continue
        inp_path = root / inp
        if not inp_path.exists():
            problems.append(f"G8 input missing (cannot stamp): {artifact} <- {inp}")
            out_lines.append(raw)
            continue
        new_sha = gate_lib.sha256_file(inp_path)
        out_lines.append("\t".join((artifact, generator, inp, new_sha, scope, check)))
        n_stamped += 1
    for p in problems:
        print(f"  ! {p}")
    if problems:
        print(f"✗ stamp_derivations: {len(problems)} problem(s); nothing written")
        return len(problems)
    if write:
        path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"✓ stamped {n_stamped} G8 row(s) in {path}")
    else:
        print(f"✓ would stamp {n_stamped} G8 row(s) (check mode)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="Dry run; never writes.")
    args = ap.parse_args()
    return 1 if stamp(write=not args.check) else 0


if __name__ == "__main__":
    sys.exit(main())
