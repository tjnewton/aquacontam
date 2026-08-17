"""Tests for data._download — shared download and extraction utility."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aquacontam.data._download import download_and_extract


def _make_zip_bytes(filenames: list[str], contents: bytes = b"test data") -> bytes:
    """Create an in-memory ZIP with the given filenames."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in filenames:
            zf.writestr(name, contents)
    return buf.getvalue()


def _mock_response(content: bytes, status_code: int = 200) -> MagicMock:
    """Create a mock requests.Response for streaming downloads."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {"content-length": str(len(content))}
    resp.iter_content.return_value = [content]
    resp.raise_for_status.return_value = None
    return resp


class TestDownloadAndExtract:
    def test_zip_download_and_extraction(self, tmp_path: Path) -> None:
        """Normal ZIP download extracts expected files."""
        zip_bytes = _make_zip_bytes(["data.csv"])
        mock_resp = _mock_response(zip_bytes)

        with patch("aquacontam.data._download.requests.get", return_value=mock_resp):
            paths = download_and_extract(
                "https://example.com/data.zip",
                tmp_path,
                ["data.csv"],
            )

        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].name == "data.csv"
        assert paths[0].read_text() == "test data"

    def test_skip_download_when_files_exist(self, tmp_path: Path) -> None:
        """Existing files are not re-downloaded when force=False."""
        (tmp_path / "data.csv").write_text("existing")

        with patch("aquacontam.data._download.requests.get") as mock_get:
            paths = download_and_extract(
                "https://example.com/data.zip",
                tmp_path,
                ["data.csv"],
                force=False,
            )

        mock_get.assert_not_called()
        assert paths[0].read_text() == "existing"

    def test_force_redownload(self, tmp_path: Path) -> None:
        """force=True re-downloads even when files exist."""
        (tmp_path / "data.csv").write_text("old")
        zip_bytes = _make_zip_bytes(["data.csv"], b"new data")
        mock_resp = _mock_response(zip_bytes)

        with patch("aquacontam.data._download.requests.get", return_value=mock_resp):
            paths = download_and_extract(
                "https://example.com/data.zip",
                tmp_path,
                ["data.csv"],
                force=True,
            )

        assert paths[0].read_text() == "new data"

    def test_path_traversal_attack_prevented(self, tmp_path: Path) -> None:
        """ZIP with path traversal member raises ValueError."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../../../etc/passwd", "malicious")
        zip_bytes = buf.getvalue()
        mock_resp = _mock_response(zip_bytes)

        with (
            patch("aquacontam.data._download.requests.get", return_value=mock_resp),
            pytest.raises(ValueError, match="outside target directory"),
        ):
            download_and_extract(
                "https://example.com/evil.zip",
                tmp_path,
                ["etc/passwd"],
            )

    def test_missing_expected_file_raises(self, tmp_path: Path) -> None:
        """FileNotFoundError if expected file not in ZIP."""
        zip_bytes = _make_zip_bytes(["wrong_file.csv"])
        mock_resp = _mock_response(zip_bytes)

        with (
            patch("aquacontam.data._download.requests.get", return_value=mock_resp),
            pytest.raises(FileNotFoundError, match="Expected files not found"),
        ):
            download_and_extract(
                "https://example.com/data.zip",
                tmp_path,
                ["expected.csv"],
            )

    def test_plain_file_download(self, tmp_path: Path) -> None:
        """Non-ZIP URL writes directly to first expected filename."""
        mock_resp = _mock_response(b"plain content")

        with patch("aquacontam.data._download.requests.get", return_value=mock_resp):
            paths = download_and_extract(
                "https://example.com/data.csv",
                tmp_path,
                ["data.csv"],
            )

        assert paths[0].read_text() == "plain content"

    def test_creates_raw_dir_if_missing(self, tmp_path: Path) -> None:
        """raw_dir is created if it doesn't exist."""
        raw_dir = tmp_path / "nested" / "raw"
        zip_bytes = _make_zip_bytes(["data.csv"])
        mock_resp = _mock_response(zip_bytes)

        with patch("aquacontam.data._download.requests.get", return_value=mock_resp):
            paths = download_and_extract(
                "https://example.com/data.zip",
                raw_dir,
                ["data.csv"],
            )

        assert raw_dir.exists()
        assert paths[0].exists()

    def test_multiple_expected_files(self, tmp_path: Path) -> None:
        """Handles ZIP with multiple expected files."""
        zip_bytes = _make_zip_bytes(["a.csv", "b.csv"])
        mock_resp = _mock_response(zip_bytes)

        with patch("aquacontam.data._download.requests.get", return_value=mock_resp):
            paths = download_and_extract(
                "https://example.com/data.zip",
                tmp_path,
                ["a.csv", "b.csv"],
            )

        assert len(paths) == 2
        assert all(p.exists() for p in paths)
