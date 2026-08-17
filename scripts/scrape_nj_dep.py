#!/usr/bin/env python3
"""Download NJ DEP PFAS sample data from waterviewer.nj.gov.

NJ DEP drinking-water chemical-sample results are public records under the New
Jersey Open Public Records Act. This script launches a headed Chromium browser
and reads the same public WaterViewer pages a person would, using only the
session's own XSRF token from those pages — no account, no credentials, and no
CAPTCHA solving or access circumvention. It then paginates the public
SamplesSearchResults OData endpoint from within that ordinary browser session.

Usage
-----
    python scripts/scrape_nj_dep.py
    python scripts/scrape_nj_dep.py --output data/raw/nj_dep_pfas_export.csv
    python scripts/scrape_nj_dep.py --resume          # continue from checkpoint
    python scripts/scrape_nj_dep.py --max-records 100  # small test run
    python scripts/scrape_nj_dep.py --all-analytes     # download all OC, not just PFAS

Requirements
------------
    pip install playwright click tqdm
    playwright install chromium
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger(__name__)

_BASE_URL = "https://waterviewer.nj.gov"
_SAMPLES_ENDPOINT = "/sdwis/SamplesSearchResults"
_CHECKPOINT_FILE = ".nj_dep_scrape_checkpoint.json"
_DEFAULT_OUTPUT = "data/raw/nj_dep_pfas_export.csv"
_DEFAULT_PAGE_SIZE = 5000
_DEFAULT_DELAY = 0.5

# PFAS analyte codes from NJ DEP (individual compounds, excluding summary codes)
_PFAS_CODES: frozenset[int] = frozenset(
    {
        2801,
        2802,
        2803,
        2804,
        2805,
        2806,
        2807,
        2808,
        2809,
        2810,
        2811,
        2812,
        2813,
        2814,
        2816,
        2819,
        2820,
        2821,
        2822,
        2823,
        2824,
        2825,
        2826,
        2828,
        2829,
    }
)

# Summary/composite codes to include in output (for informational value)
_SUMMARY_CODES: frozenset[int] = frozenset({2800, 2830, 2840})

# Date ranges — API limits to 5 years per query
_DATE_RANGES: list[tuple[str, str]] = [
    ("2016-03-17", "2021-03-16"),
    ("2021-03-17", "2026-03-16"),
]

# JavaScript template for browser-side fetch with XSRF token
_FETCH_JS = """
async (args) => {
    const [xsrf, url] = args;
    try {
        const resp = await fetch(url, {
            headers: {
                "Accept": "application/json",
                "X-XSRF-TOKEN": xsrf,
                "X-Requested-With": "XMLHttpRequest"
            },
            credentials: "same-origin"
        });
        if (!resp.ok) {
            return {"error": resp.status, "statusText": resp.statusText, "value": []};
        }
        return await resp.json();
    } catch (e) {
        return {"error": -1, "statusText": e.message, "value": []};
    }
}
"""


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load scrape checkpoint from JSON file.

    Returns
    -------
    dict
        Checkpoint data with keys ``offset``, ``records_so_far``, and
        ``date_range_idx``, or empty dict if no checkpoint exists.
    """
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_checkpoint(path: Path, offset: int, records_so_far: int, date_range_idx: int) -> None:
    """Save scrape checkpoint to JSON file."""
    data = {
        "offset": offset,
        "records_so_far": records_so_far,
        "date_range_idx": date_range_idx,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def records_to_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write API records to CSV.

    Parameters
    ----------
    records : list[dict]
        Records from the SamplesSearchResults API.
    output_path : Path
        Destination CSV file path.
    """
    import pandas as pd

    if not records:
        logger.warning("No records to write")
        return

    df = pd.DataFrame(records)

    # Log schema on first write for discovery
    logger.info("Output columns (%d): %s", len(df.columns), list(df.columns))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d records to %s", len(df), output_path)


# ---------------------------------------------------------------------------
# Browser-based pagination
# ---------------------------------------------------------------------------


def _build_api_url(
    begin_date: str,
    end_date: str,
    offset: int,
    page_size: int,
) -> str:
    """Build the OData API URL for SamplesSearchResults.

    Parameters
    ----------
    begin_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.
    offset : int
        Pagination offset ($skip).
    page_size : int
        Records per page ($top).

    Returns
    -------
    str
        Relative API URL with query parameters.
    """
    return (
        f"{_SAMPLES_ENDPOINT}"
        f"?BeginDate={begin_date}"
        f"&EndDate={end_date}"
        f"&WaterSystemStatus=A&WaterSystemStatus=I"
        f"&cmp=Y&sampleDelay=0"
        f"&AnalyteType=OC+"
        f"&$skip={offset}&$top={page_size}&$count=true"
    )


def _fetch_page_in_browser(
    page: Any,
    xsrf_token: str,
    url: str,
    *,
    max_retries: int = 3,
) -> tuple[list[dict[str, Any]], int, bool]:
    """Fetch one page of sample results via browser-side fetch().

    Parameters
    ----------
    page : playwright.sync_api.Page
        Active browser page.
    xsrf_token : str
        XSRF-TOKEN cookie value.
    url : str
        Relative API URL with query parameters.
    max_retries : int
        Retries for transient errors.

    Returns
    -------
    tuple[list[dict], int, bool]
        ``(records, total_count, has_more)``.
    """
    for attempt in range(1 + max_retries):
        try:
            data = page.evaluate(_FETCH_JS, [xsrf_token, url])
        except Exception as exc:
            if attempt < max_retries:
                delay = 2 ** (attempt + 1)
                logger.warning("Browser fetch error: %s — retrying in %ds", exc, delay)
                page.wait_for_timeout(delay * 1000)
                continue
            raise

        if "error" in data and data["error"] != -1:
            status = data["error"]
            if status in (429, 500, 502, 503) and attempt < max_retries:
                delay = 2 ** (attempt + 1)
                logger.warning(
                    "HTTP %d — retrying in %ds (attempt %d/%d)",
                    status,
                    delay,
                    attempt + 1,
                    max_retries,
                )
                page.wait_for_timeout(delay * 1000)
                continue
            if status >= 400:
                raise RuntimeError(
                    f"API returned HTTP {status}: {data.get('statusText', 'unknown')}"
                )

        batch = data.get("value", [])
        total_count = data.get("@odata.count", -1)
        page_size_from_url = int(url.split("$top=")[1].split("&")[0])
        has_more = len(batch) >= page_size_from_url
        return batch, total_count, has_more

    return [], -1, False  # pragma: no cover


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--output",
    "-o",
    default=_DEFAULT_OUTPUT,
    type=click.Path(),
    help="Output CSV path.",
)
@click.option("--page-size", default=_DEFAULT_PAGE_SIZE, help="Records per API page.")
@click.option("--delay", default=_DEFAULT_DELAY, help="Delay between pages (seconds).")
@click.option("--resume", is_flag=True, help="Resume from last checkpoint.")
@click.option("--max-records", default=0, help="Stop after N records (0 = unlimited).")
@click.option("--all-analytes", is_flag=True, help="Keep all OC analytes, not just PFAS.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def main(
    output: str,
    page_size: int,
    delay: float,
    resume: bool,
    max_records: int,
    all_analytes: bool,
    verbose: bool,
) -> None:
    """Download NJ DEP PFAS sample data from waterviewer.nj.gov."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        click.echo(
            "ERROR: Playwright is not installed.\n"
            "  pip install playwright\n"
            "  playwright install chromium",
            err=True,
        )
        sys.exit(1)

    output_path = Path(output)
    checkpoint_path = output_path.parent / _CHECKPOINT_FILE

    # Resume from checkpoint?
    start_offset = 0
    start_date_idx = 0
    existing_records: list[dict[str, Any]] = []
    if resume:
        ckpt = load_checkpoint(checkpoint_path)
        if ckpt:
            start_offset = ckpt.get("offset", 0)
            start_date_idx = ckpt.get("date_range_idx", 0)
            click.echo(
                f"Resuming from offset {start_offset}, date range {start_date_idx} "
                f"({ckpt.get('records_so_far', 0)} PFAS records previously)"
            )
            if output_path.exists():
                import pandas as pd

                existing_df = pd.read_csv(output_path, dtype=str, low_memory=False)
                existing_records = existing_df.to_dict("records")
                click.echo(f"Loaded {len(existing_records)} existing records from {output_path}")
        else:
            click.echo("No checkpoint found, starting fresh.")

    # --- Launch browser and scrape ---
    click.echo("\nLaunching browser...")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        context = browser.new_context()
        pg = context.new_page()

        pg.goto(_BASE_URL, timeout=60_000)
        pg.wait_for_load_state("networkidle")
        pg.wait_for_timeout(3_000)

        # Extract XSRF token
        cookies = context.cookies()
        xsrf = next((c["value"] for c in cookies if c["name"] == "XSRF-TOKEN"), "")
        if not xsrf:
            click.echo("ERROR: No XSRF-TOKEN cookie found.", err=True)
            browser.close()
            sys.exit(1)

        click.echo("XSRF token acquired. Starting paginated download...\n")

        pfas_codes = _PFAS_CODES | _SUMMARY_CODES
        all_records = list(existing_records)
        total_api_records = 0
        hit_max = False

        try:
            from tqdm import tqdm

            pbar: Any = tqdm(
                desc="PFAS records" if not all_analytes else "Records",
                unit="rec",
                initial=len(all_records),
            )
        except ImportError:
            pbar = None

        try:
            for dr_idx, (begin_date, end_date) in enumerate(_DATE_RANGES):
                if dr_idx < start_date_idx:
                    continue

                offset = start_offset if dr_idx == start_date_idx else 0
                click.echo(f"Date range: {begin_date} to {end_date}")

                while True:
                    url = _build_api_url(begin_date, end_date, offset, page_size)
                    batch, total_count, has_more = _fetch_page_in_browser(pg, xsrf, url)

                    if total_count >= 0 and offset == 0:
                        click.echo(f"  Total OC records in range: {total_count:,}")

                    if not batch:
                        break

                    # Log schema on first page
                    if total_api_records == 0:
                        logger.info("API columns: %s", list(batch[0].keys()))

                    total_api_records += len(batch)

                    # Filter for PFAS if not keeping all
                    if all_analytes:
                        pfas_batch = batch
                    else:
                        pfas_batch = [
                            r
                            for r in batch
                            if r.get("ANALYTE_CODE") in pfas_codes
                            or (
                                isinstance(r.get("ANALYTE_CODE"), str)
                                and r["ANALYTE_CODE"].strip().isdigit()
                                and int(r["ANALYTE_CODE"].strip()) in pfas_codes
                            )
                        ]

                    all_records.extend(pfas_batch)
                    offset += len(batch)

                    if pbar is not None:
                        pbar.update(len(pfas_batch))

                    # Checkpoint every 10 pages
                    page_num = offset // page_size
                    if page_num % 10 == 0:
                        save_checkpoint(checkpoint_path, offset, len(all_records), dr_idx)

                    if max_records > 0 and len(all_records) >= max_records:
                        click.echo(f"\nReached max records limit ({max_records}).")
                        hit_max = True
                        break

                    if not has_more:
                        break

                    pg.wait_for_timeout(int(delay * 1000))

                if hit_max:
                    break

                # Reset for next date range
                start_offset = 0

                # Intermediate save between date ranges
                if all_records:
                    records_to_csv(all_records, output_path)
                    click.echo(
                        f"  Intermediate save: {len(all_records):,} records to {output_path}"
                    )

        except KeyboardInterrupt:
            click.echo("\nInterrupted! Saving checkpoint...")
            save_checkpoint(checkpoint_path, offset, len(all_records), dr_idx)
            if all_records:
                records_to_csv(all_records, output_path)
                click.echo(f"Saved {len(all_records):,} records before exit.")
            click.echo("Re-run with --resume to continue")
        except Exception:
            logger.exception("Scraper error — saving partial results")
            save_checkpoint(checkpoint_path, offset, len(all_records), dr_idx)
            if all_records:
                records_to_csv(all_records, output_path)
                click.echo(f"Saved {len(all_records):,} records before crash.")
            raise
        finally:
            if pbar is not None:
                pbar.close()

        browser.close()

    # --- Save results ---
    new_records = len(all_records) - len(existing_records)
    click.echo(
        f"\nTotal API records scanned: {total_api_records:,}"
        f"\nPFAS records: {len(all_records):,} ({new_records:,} new)"
    )

    if all_records:
        records_to_csv(all_records, output_path)
        click.echo(f"Saved to {output_path}")

        if checkpoint_path.exists() and not hit_max:
            checkpoint_path.unlink()
            click.echo("Checkpoint cleaned up.")
    else:
        click.echo("No records downloaded.")


if __name__ == "__main__":
    main()
