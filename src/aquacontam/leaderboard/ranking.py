"""Leaderboard ranking and formatting.

Ranks submissions by per-task primary metrics, computes overall ranks,
and formats results as markdown tables or structured JSON.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd

from aquacontam.leaderboard.schema import validate_submission

logger = logging.getLogger(__name__)

# Primary metric per task (higher is better, except RMSE/MAE)
_PRIMARY_METRICS: dict[str, str] = {
    "T1": "auprc",
    "T2": "rmse",
    "T3": "macro_auprc",
    "T4": "auprc",
    "T5": "auprc",
    "T6": "auprc",
    "T7": "auprc",
}

# Metrics where lower is better
_LOWER_IS_BETTER: frozenset[str] = frozenset({"rmse", "mae", "median_ae", "hamming_loss"})


def rank_submissions(
    submissions: list[dict[str, Any]],
    *,
    primary_metrics: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Rank multiple submissions across tasks.

    Parameters
    ----------
    submissions : list[dict]
        List of validated submission dicts (see ``schema.py``).
    primary_metrics : dict[str, str], optional
        Override primary metric per task.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``model_name``, ``task``, ``analyte``,
        ``primary_metric``, ``value``, ``rank``, ``overall_rank``.
    """
    if primary_metrics is None:
        primary_metrics = dict(_PRIMARY_METRICS)

    rows: list[dict[str, Any]] = []
    for sub in submissions:
        model_name = sub["model_name"]
        for result in sub["results"]:
            task = result["task"]
            analyte = result.get("analyte", "all")
            metric_name = primary_metrics.get(task, "auprc")
            metric_val = result["metrics"].get(metric_name)

            if metric_val is None:
                continue

            rows.append(
                {
                    "model_name": model_name,
                    "task": task,
                    "analyte": analyte,
                    "primary_metric": metric_name,
                    "value": float(metric_val),
                }
            )

    if not rows:
        return cast(
            pd.DataFrame,
            pd.DataFrame(
                columns=[
                    "model_name",
                    "task",
                    "analyte",
                    "primary_metric",
                    "value",
                    "rank",
                    "overall_rank",
                ]
            ),
        )

    df = pd.DataFrame(rows)

    # Rank within each (task, analyte) group
    ranks = []
    for _, group in df.groupby(["task", "analyte"], observed=True):
        metric = group["primary_metric"].iloc[0]
        ascending = metric in _LOWER_IS_BETTER
        ranks.append(group["value"].rank(ascending=ascending, method="min").astype(int))
    df["rank"] = pd.concat(ranks)

    # Overall rank = mean rank across all (task, analyte) entries per model
    mean_ranks = df.groupby("model_name")["rank"].mean()
    overall = mean_ranks.rank(method="min").astype(int)
    df["overall_rank"] = df["model_name"].map(overall)

    # model_name is the final tiebreaker so byte-identical metric ties (e.g. several models
    # sharing an ensemble base) sort deterministically across pandas/numpy versions and hash
    # seeds; kind="stable" avoids the default unstable quicksort reordering equal keys.
    ranked: pd.DataFrame = df.sort_values(
        ["overall_rank", "task", "analyte", "model_name"], kind="stable"
    ).reset_index(drop=True)
    return ranked


def format_leaderboard(
    rankings: pd.DataFrame,
    *,
    fmt: str = "markdown",
) -> str:
    """Format ranked submissions as a readable table.

    Parameters
    ----------
    rankings : pd.DataFrame
        Output from ``rank_submissions()``.
    fmt : str
        Output format: ``"markdown"`` or ``"json"``.

    Returns
    -------
    str
        Formatted leaderboard string.
    """
    if rankings.empty:
        return "No submissions to rank."

    if fmt == "json":
        return rankings.to_json(orient="records", indent=2)

    # Markdown table — one row per model with overall rank
    summary = (
        rankings.groupby("model_name")
        .agg(
            overall_rank=("overall_rank", "first"),
            n_entries=("task", "count"),
            mean_rank=("rank", "mean"),
        )
        .reset_index()
        .sort_values(["overall_rank", "model_name"], kind="stable")
    )

    lines = [
        "| Rank | Model | Entries | Mean Rank |",
        "|------|-------|---------|-----------|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['overall_rank']} | {row['model_name']} "
            f"| {row['n_entries']} | {row['mean_rank']:.2f} |"
        )

    # Per-task breakdown
    lines.append("")
    lines.append("### Per-Task Rankings")
    lines.append("")

    for task in sorted(rankings["task"].unique()):
        task_df = rankings[rankings["task"] == task].sort_values(
            ["rank", "model_name"], kind="stable"
        )
        metric = task_df["primary_metric"].iloc[0]
        lines.append(f"**{task}** (metric: {metric})")
        lines.append("")
        lines.append("| Rank | Model | Analyte | Value |")
        lines.append("|------|-------|---------|-------|")
        for _, row in task_df.iterrows():
            lines.append(
                f"| {row['rank']} | {row['model_name']} | {row['analyte']} | {row['value']:.4f} |"
            )
        lines.append("")

    return "\n".join(lines)


def generate_leaderboard_document(
    submissions_dir: str | Path,
    *,
    fmt: str = "markdown",
    title: str = "AquaContam Benchmark Leaderboard",
    timestamp: str | None = None,
) -> str:
    """Generate a full leaderboard document from a directory of submission JSONs.

    Parameters
    ----------
    submissions_dir : str | Path
        Directory containing ``.json`` submission files.
    fmt : str
        Output format: ``"markdown"`` or ``"json"``.
    title : str
        Document title.

    Returns
    -------
    str
        Full leaderboard document.
    """
    submissions_dir = Path(submissions_dir)
    submissions: list[dict[str, Any]] = []
    errors: dict[str, list[str]] = {}

    for json_path in sorted(submissions_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text())
            errs = validate_submission(data)
            if errs:
                errors[json_path.name] = errs
                logger.warning("Skipping invalid submission %s: %s", json_path.name, errs)
                continue
            submissions.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            errors[json_path.name] = [str(exc)]
            logger.warning("Failed to load %s: %s", json_path.name, exc)

    if not submissions:
        if fmt == "json":
            return json.dumps({"title": title, "submissions": 0, "rankings": []}, indent=2)
        return f"# {title}\n\nNo valid submissions found.\n"

    rankings = rank_submissions(submissions)
    table = format_leaderboard(rankings, fmt=fmt)

    if fmt == "json":
        try:
            parsed = json.loads(table)
            rankings_list = parsed if isinstance(parsed, list) else []
        except (TypeError, ValueError):
            rankings_list = []
        return json.dumps(
            {"title": title, "submissions": len(submissions), "rankings": rankings_list},
            indent=2,
        )

    # Build full markdown document. A caller may pin ``timestamp`` for deterministic,
    # diff-matchable regeneration (e.g. regenerating LEADERBOARD.md from the frozen archive).
    ts = timestamp or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# {title}",
        "",
        f"*Last updated: {ts}*",
        "",
        f"**{len(submissions)} submissions** from "
        f"{len({s['model_name'] for s in submissions})} models",
        "",
        "## Overall Rankings",
        "",
        table,
        "",
        "## How to Submit",
        "",
        "See [`submissions/README.md`](submissions/README.md) for instructions.",
        "",
        "## Citation",
        "",
        "If you use this benchmark, please cite:",
        "",
        "```bibtex",
        "@article{newton2026aquacontam,",
        "  title={AquaContam: machine-learning models of drinking-water contamination "
        "learn who is monitored as much as where contamination occurs},",
        "  author={Newton, Tyler J.},",
        "  journal={Nature Water},",
        "  year={2026}",
        "}",
        "```",
        "",
    ]

    return "\n".join(lines)
