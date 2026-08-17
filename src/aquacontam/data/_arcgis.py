"""Shared ArcGIS REST API utilities for data source modules."""

from __future__ import annotations

import logging
from typing import Any

import requests
from tenacity import RetryCallState, Retrying, stop_after_attempt, wait_exponential

from aquacontam._constants import DOWNLOAD_TIMEOUT_DEFAULT

logger = logging.getLogger(__name__)

# ArcGIS REST API pagination limit
_PAGE_SIZE = 2000

# HTTP status codes that are transient and worth retrying
_RETRYABLE_STATUS_CODES = {429, 502, 503}


class _RetryableHTTPError(Exception):
    """Raised internally to trigger tenacity retry on transient HTTP status codes."""

    def __init__(self, response: requests.Response) -> None:
        self.response = response
        super().__init__(f"HTTP {response.status_code}")


def _log_retry(retry_state: RetryCallState) -> None:
    """Log retry attempts for transient HTTP errors."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None  # type: ignore[union-attr]
    status = getattr(getattr(exc, "response", None), "status_code", "?")
    logger.warning(
        "HTTP %s — retrying in %ds (attempt %d)",
        status,
        retry_state.next_action.sleep if retry_state.next_action else 0,  # type: ignore[union-attr]
        retry_state.attempt_number,
    )


def _request_with_retry(
    url: str,
    params: dict[str, str | int],
    *,
    timeout: int,
    verify: bool,
    max_retries: int = 3,
) -> requests.Response:
    """Issue a GET request with exponential backoff for transient HTTP errors.

    Parameters
    ----------
    url : str
        Request URL.
    params : dict
        Query parameters.
    timeout : int
        HTTP request timeout in seconds.
    verify : bool
        Whether to verify SSL certificates.
    max_retries : int
        Maximum number of retries for transient errors (429, 502, 503).
        Delays are 2, 4, 8, … seconds (exponential backoff).

    Returns
    -------
    requests.Response
        The successful HTTP response.

    Raises
    ------
    requests.HTTPError
        If all retries are exhausted or a non-retryable error occurs.
    """
    for attempt in Retrying(
        stop=stop_after_attempt(1 + max_retries),
        wait=wait_exponential(multiplier=2, min=2, max=16),
        reraise=True,
        before_sleep=_log_retry,
    ):
        with attempt:
            resp = requests.get(url, params=params, timeout=timeout, verify=verify)
            if resp.status_code in _RETRYABLE_STATUS_CODES:
                raise _RetryableHTTPError(resp)
            resp.raise_for_status()
            return resp
    # Unreachable — Retrying raises after exhaustion — but keeps mypy happy
    raise requests.HTTPError("All retries exhausted")  # pragma: no cover


def fetch_arcgis_features(
    base_url: str,
    layer_id: int = 0,
    *,
    timeout: int = DOWNLOAD_TIMEOUT_DEFAULT,
    verify_ssl: bool = True,
    max_retries: int = 3,
    out_sr: int | None = 4326,
) -> list[dict[str, Any]]:
    """Paginate through ArcGIS REST API query results.

    Parameters
    ----------
    base_url : str
        ArcGIS MapServer base URL.
    layer_id : int
        Layer index within the MapServer.
    timeout : int
        HTTP request timeout in seconds.
    verify_ssl : bool
        Whether to verify SSL certificates. Default ``True``.
        When ``True``, SSL errors trigger an automatic retry without
        verification (with a warning). Pass ``False`` to skip
        verification entirely.
    max_retries : int
        Maximum number of retries for transient HTTP errors (429, 502, 503).
        Default ``3`` with exponential backoff (2s, 4s, 8s).
    out_sr : int | None
        Output spatial reference WKID. Default ``4326`` (WGS84) so that
        geometry coordinates are returned as longitude/latitude in degrees.
        Pass ``None`` to use the server's native spatial reference.

    Returns
    -------
    list[dict]
        List of feature attribute dicts with optional geometry.
    """
    query_url = f"{base_url}/{layer_id}/query"
    all_features: list[dict[str, Any]] = []
    offset = 0

    while True:
        params: dict[str, str | int] = {
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "true",
            "f": "json",
            "resultRecordCount": _PAGE_SIZE,
            "resultOffset": offset,
        }
        if out_sr is not None:
            params["outSR"] = out_sr
        logger.info("Querying ArcGIS layer %d, offset %d …", layer_id, offset)
        try:
            resp = _request_with_retry(
                query_url,
                params,
                timeout=timeout,
                verify=verify_ssl,
                max_retries=max_retries,
            )
        except requests.exceptions.SSLError:
            if not verify_ssl:
                raise
            logger.warning(
                "SSL verification failed for %s; retrying without verification",
                query_url,
            )
            resp = _request_with_retry(
                query_url,
                params,
                timeout=timeout,
                verify=False,
                max_retries=max_retries,
            )
        data = resp.json()

        features = data.get("features", [])
        if not features:
            break

        for feat in features:
            row = dict(feat.get("attributes", {}))
            geom = feat.get("geometry")
            if geom:
                row["_longitude"] = geom.get("x")
                row["_latitude"] = geom.get("y")
            all_features.append(row)

        offset += len(features)
        if not data.get("exceededTransferLimit", False):
            break

    logger.info("Fetched %d total features from ArcGIS", len(all_features))
    return all_features
