"""Shared streaming download and ZIP extraction utility.

Consolidates the download-extract-verify pattern used by UCMR, FRS,
NLCD, and USGS aquifer loaders into a single reusable function.
"""

from __future__ import annotations

import hashlib
import io
import logging
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from aquacontam._constants import DOWNLOAD_CHUNK_SIZE, DOWNLOAD_TIMEOUT_DEFAULT

logger = logging.getLogger(__name__)

# Browser-like headers to avoid WAF blocks (e.g. NJ DEP Imperva/Incapsula).
_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def download_and_extract(
    url: str,
    raw_dir: Path,
    expected_files: list[str],
    *,
    force: bool = False,
    timeout: int = DOWNLOAD_TIMEOUT_DEFAULT,
    progress_desc: str = "Downloading",
) -> list[Path]:
    """Download a file (optionally ZIP) and extract/verify expected files.

    Parameters
    ----------
    url : str
        URL to download.
    raw_dir : Path
        Directory to save/extract files into.
    expected_files : list[str]
        Filenames expected after extraction.
    force : bool
        If ``True``, re-download even if files already exist.
    timeout : int
        HTTP request timeout in seconds.
    progress_desc : str
        Label for the ``tqdm`` progress bar.

    Returns
    -------
    list[Path]
        Paths to the expected files.

    Raises
    ------
    FileNotFoundError
        If expected files are missing after extraction.
    ValueError
        If a ZIP member would extract outside the target directory
        (path traversal protection).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    dest_paths = [raw_dir / f for f in expected_files]
    existing = [p for p in dest_paths if p.exists()]
    if len(existing) == len(expected_files) and not force:
        logger.info("Files already exist, skipping download: %s", existing)
        return dest_paths

    logger.info("Downloading %s …", url)
    resp = requests.get(url, stream=True, timeout=timeout, headers=_DEFAULT_HEADERS)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    buf = io.BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, desc=progress_desc) as pbar:
        for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
            buf.write(chunk)
            pbar.update(len(chunk))

    buf.seek(0)

    # Detect ZIP by URL or content
    url_path = url.split("?")[0]  # Strip query params for extension check
    is_zip = url_path.endswith(".zip") or zipfile.is_zipfile(buf)
    buf.seek(0)

    if is_zip:
        with zipfile.ZipFile(buf) as zf:
            # Path traversal protection
            for member in zf.namelist():
                member_path = (raw_dir / member).resolve()
                if not member_path.is_relative_to(raw_dir.resolve()):
                    raise ValueError(
                        f"Zip member {member!r} would extract outside target directory"
                    )
            zf.extractall(raw_dir)
            logger.info("Extracted %d files to %s", len(zf.namelist()), raw_dir)
    else:
        # Plain file download — write to first expected filename
        dest_paths[0].write_bytes(buf.read())

    missing = [p for p in dest_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Expected files not found after extraction: {missing}")

    return dest_paths


def _load_checksums(project_root: Path | None = None) -> dict[str, str]:
    """Load expected checksums from ``data/checksums.sha256``.

    Returns
    -------
    dict[str, str]
        Mapping of relative path → expected SHA-256 hex digest.
    """
    if project_root is None:
        # Walk up from this file to find project root (contains data/ dir)
        project_root = Path(__file__).resolve().parents[3]
    checksum_file = project_root / "data" / "checksums.sha256"
    if not checksum_file.exists():
        return {}

    checksums: dict[str, str] = {}
    for line in checksum_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            checksums[parts[1].strip()] = parts[0].strip()
    return checksums


def verify_checksum(path: Path, project_root: Path | None = None) -> bool:
    """Verify a file's SHA-256 checksum against ``data/checksums.sha256``.

    Parameters
    ----------
    path : Path
        Absolute path to the file to verify.
    project_root : Path, optional
        Project root directory. Auto-detected if not provided.

    Returns
    -------
    bool
        ``True`` if checksum matches or no expected checksum exists.
        ``False`` if checksum does not match.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[3]

    checksums = _load_checksums(project_root)
    try:
        rel_path = str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        logger.debug("File %s is outside project root, skipping checksum", path)
        return True

    expected = checksums.get(rel_path)
    if expected is None:
        logger.debug("No checksum entry for %s", rel_path)
        return True

    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    actual = sha.hexdigest()

    if actual != expected:
        logger.warning(
            "Checksum mismatch for %s: expected %s, got %s",
            rel_path,
            expected[:16] + "…",
            actual[:16] + "…",
        )
        return False

    logger.debug("Checksum OK: %s", rel_path)
    return True
