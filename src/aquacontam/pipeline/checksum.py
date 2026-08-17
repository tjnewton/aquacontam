"""Pipeline result checksum generation and verification."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_results_checksums(output_dir: Path) -> Path:
    """Generate SHA-256 checksums for all JSON result files.

    Parameters
    ----------
    output_dir : Path
        Results directory containing pipeline output JSON files.

    Returns
    -------
    Path
        Path to the generated checksums file.
    """
    import hashlib

    checksum_lines: list[str] = []
    for json_file in sorted(output_dir.glob("*.json")):
        sha256 = hashlib.sha256(json_file.read_bytes()).hexdigest()
        checksum_lines.append(f"{sha256}  {json_file.name}")

    out_path = output_dir / "checksums.sha256"
    out_path.write_text("\n".join(checksum_lines) + "\n")
    logger.info("Results checksums saved to %s (%d files)", out_path, len(checksum_lines))
    return out_path


def verify_results_checksums(output_dir: Path) -> bool:
    """Verify JSON result files against stored checksums.

    Parameters
    ----------
    output_dir : Path
        Results directory containing checksums.sha256.

    Returns
    -------
    bool
        True if all checksums match, False otherwise.
    """
    import hashlib

    checksum_path = output_dir / "checksums.sha256"
    if not checksum_path.exists():
        logger.error("No checksums.sha256 found in %s", output_dir)
        return False

    ok = True
    for line in checksum_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        expected_hash, filename = line.split("  ", 1)
        filepath = output_dir / filename
        if not filepath.exists():
            logger.error("MISSING: %s", filename)
            ok = False
            continue
        actual_hash = hashlib.sha256(filepath.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            logger.error(
                "MISMATCH: %s (expected %s, got %s)", filename, expected_hash, actual_hash
            )
            ok = False
        else:
            logger.info("OK: %s", filename)

    return ok
