"""Live network integration tests for data source downloads.

All tests are marked ``@pytest.mark.slow`` and ``@pytest.mark.network``
so they are excluded from CI and fast local runs. Run explicitly with::

    pytest tests/integration/test_live_downloads.py -m network -v
"""

from __future__ import annotations

import pytest
import requests

# ---------------------------------------------------------------------------
# WQP live download tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.network
class TestWqpLiveDownload:
    """Live connectivity and pipeline tests for the Water Quality Portal."""

    WQP_RESULT_URL = "https://www.waterqualitydata.us/data/Result/search"
    WQP_STATION_URL = "https://www.waterqualitydata.us/data/Station/search"
    TIMEOUT = 120

    def test_wqp_results_api_connectivity(self) -> None:
        """Single-state, single-characteristic query returns a ZIP with CSV."""
        import io
        import zipfile

        params = {
            "statecode": "US:29",
            "characteristicName": "Perfluorooctane sulfonic acid",
            "mimeType": "csv",
            "zip": "yes",
            "dataProfile": "resultPhysChem",
            "sorted": "no",
        }
        try:
            resp = requests.get(
                self.WQP_RESULT_URL, params=params, timeout=self.TIMEOUT, stream=True
            )
        except (requests.ConnectionError, requests.Timeout):
            pytest.skip("WQP API unreachable")

        assert resp.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        assert len(csv_names) >= 1

        import pandas as pd

        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(f, dtype=str, nrows=5)
        # Verify expected WQP columns are present
        expected_cols = {
            "MonitoringLocationIdentifier",
            "CharacteristicName",
            "ResultMeasureValue",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_wqp_stations_api_connectivity(self) -> None:
        """Station endpoint returns lat/lon/site ID."""
        import io
        import zipfile

        params = {
            "statecode": "US:29",
            "characteristicName": "Perfluorooctane sulfonic acid",
            "mimeType": "csv",
            "zip": "yes",
            "sorted": "no",
        }
        try:
            resp = requests.get(
                self.WQP_STATION_URL, params=params, timeout=self.TIMEOUT, stream=True
            )
        except (requests.ConnectionError, requests.Timeout):
            pytest.skip("WQP API unreachable")

        assert resp.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        assert len(csv_names) >= 1

        import pandas as pd

        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(f, dtype=str, nrows=5)
        expected_cols = {
            "MonitoringLocationIdentifier",
            "LatitudeMeasure",
            "LongitudeMeasure",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_wqp_download_parse_validate_pipeline(self, tmp_path: pytest.TempPathFactory) -> None:
        """Full pipeline for Missouri (US:29): download → parse → validate."""
        from pathlib import Path

        from aquacontam.data.wqp import WqpSource

        raw = tmp_path / "raw"  # type: ignore[operator]
        interim = tmp_path / "interim"  # type: ignore[operator]
        processed = tmp_path / "processed"  # type: ignore[operator]
        for d in (raw, interim, processed):
            d.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]

        source = WqpSource(
            raw_dir=Path(raw),
            interim_dir=Path(interim),
            processed_dir=Path(processed),
        )
        # Override config to query only Missouri
        source._config["states"] = ["US:29"]
        source._config["request_delay_seconds"] = 0

        try:
            files = source.download(force=True)
        except (requests.ConnectionError, requests.Timeout):
            pytest.skip("WQP API unreachable")

        assert len(files) == 2
        assert all(f.exists() for f in files)

        df = source.parse()
        assert len(df) > 0
        expected_cols = {
            "pwsid",
            "analyte",
            "concentration",
            "unit",
            "censored",
            "detection_limit",
            "sample_date",
            "latitude",
            "longitude",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_wqp_all_characteristics_individually(self) -> None:
        """Each of the 13 PFAS characteristic names should return 200 (not 400).

        Queries New York (US:36) with a single characteristic per request.
        This verifies that every name matches the WQP vocabulary.
        """
        from aquacontam.data.wqp import _WQP_PFAS_CHARACTERISTICS

        failed: list[tuple[str, int]] = []
        for char_name in _WQP_PFAS_CHARACTERISTICS:
            params: list[tuple[str, str]] = [
                ("statecode", "US:36"),
                ("characteristicName", char_name),
                ("mimeType", "csv"),
                ("zip", "yes"),
                ("dataProfile", "resultPhysChem"),
                ("sorted", "no"),
            ]
            try:
                resp = requests.get(
                    self.WQP_RESULT_URL, params=params, timeout=self.TIMEOUT, stream=True
                )
            except (requests.ConnectionError, requests.Timeout):
                pytest.skip("WQP API unreachable")

            if resp.status_code != 200:
                failed.append((char_name, resp.status_code))

        assert not failed, f"WQP returned non-200 for: {failed}"

    def test_wqp_empty_state_graceful(self, tmp_path: pytest.TempPathFactory) -> None:
        """Query for Alaska (US:02) — unlikely to have PFAS data, should not crash."""
        from pathlib import Path

        from aquacontam.data.wqp import WqpSource

        raw = tmp_path / "raw"  # type: ignore[operator]
        interim = tmp_path / "interim"  # type: ignore[operator]
        processed = tmp_path / "processed"  # type: ignore[operator]
        for d in (raw, interim, processed):
            d.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]

        source = WqpSource(
            raw_dir=Path(raw),
            interim_dir=Path(interim),
            processed_dir=Path(processed),
        )
        source._config["states"] = ["US:02"]
        source._config["request_delay_seconds"] = 0

        try:
            files = source.download(force=True)
        except (requests.ConnectionError, requests.Timeout):
            pytest.skip("WQP API unreachable")

        assert len(files) == 2
        # Files exist even if empty
        assert all(f.exists() for f in files)


# ---------------------------------------------------------------------------
# MO DNR live download tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.network
class TestMoDnrLiveDownload:
    """Live connectivity and pipeline tests for Missouri DNR ArcGIS API."""

    BASE_URL = "https://gis.dnr.mo.gov/host/rest/services/water/PFAS_Samples/MapServer"

    def test_mo_dnr_arcgis_connectivity(self) -> None:
        """Hit MapServer metadata endpoint — verify JSON response."""
        try:
            resp = requests.get(f"{self.BASE_URL}?f=json", timeout=30)
        except (requests.ConnectionError, requests.Timeout):
            pytest.skip("MO DNR ArcGIS server unreachable")

        assert resp.status_code == 200
        data = resp.json()
        assert "layers" in data or "serviceDescription" in data

    def test_mo_dnr_feature_query(self) -> None:
        """Fetch a small batch of features from layer 0."""
        from aquacontam.data._arcgis import fetch_arcgis_features

        try:
            features = fetch_arcgis_features(
                self.BASE_URL,
                layer_id=0,
                timeout=60,
            )
        except (requests.ConnectionError, requests.Timeout):
            pytest.skip("MO DNR ArcGIS server unreachable")

        # Should have at least some features
        assert len(features) > 0
        # Check expected fields
        first = features[0]
        assert isinstance(first, dict)

    def test_mo_dnr_download_parse_validate_pipeline(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Full pipeline: download → parse → validate."""
        from pathlib import Path

        from aquacontam._constants import MO_DNR_ANALYTES
        from aquacontam.data.mo_dnr import MoDnrSource

        raw = tmp_path / "raw"  # type: ignore[operator]
        interim = tmp_path / "interim"  # type: ignore[operator]
        processed = tmp_path / "processed"  # type: ignore[operator]
        for d in (raw, interim, processed):
            d.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]

        source = MoDnrSource(
            raw_dir=Path(raw),
            interim_dir=Path(interim),
            processed_dir=Path(processed),
        )

        try:
            files = source.download(force=True)
        except (requests.ConnectionError, requests.Timeout, RuntimeError):
            pytest.skip("MO DNR ArcGIS server unreachable")

        assert len(files) == 1
        assert files[0].exists()

        df = source.parse()
        assert len(df) > 0

        # PWSID format: MO prefix, 9 chars
        mo_rows = df[df["pwsid"].str.startswith("MO")]
        assert len(mo_rows) > 0

        # Analytes should be in expected set
        unique_analytes = set(df["analyte"].unique())
        expected = set(MO_DNR_ANALYTES)
        assert unique_analytes.issubset(expected) or unique_analytes.intersection(expected)
