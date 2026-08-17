"""Leaderboard submission schema and validation.

Defines the JSON schema for benchmark submissions and provides
validation/loading utilities.

Example submission::

    {
        "schema_version": "1.0",
        "model_name": "MyModel",
        "author": "Jane Doe",
        "institution": "University of Example",
        "paper_url": null,
        "code_url": "https://github.com/...",
        "results": [
            {
                "task": "T1",
                "analyte": "PFOS",
                "metrics": {"auroc": 0.92, "auprc": 0.85, "f1": 0.78}
            }
        ]
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from aquacontam._constants import SUBMISSION_SCHEMA_VERSION

logger = logging.getLogger(__name__)

# Valid task names and their expected metric keys
_VALID_TASKS: dict[str, list[str]] = {
    "T1": ["auroc", "auprc", "f1"],
    "T2": ["rmse", "mae", "r2"],
    "T3": ["macro_auroc", "macro_auprc", "macro_f1", "hamming_loss", "subset_accuracy"],
    "T4": ["auroc", "auprc", "f1"],
    "T5": ["auroc", "auprc", "f1"],
    "T6": ["auroc", "auprc", "f1"],
    "T7": ["auroc", "auprc", "f1"],
}

_REQUIRED_FIELDS = ("schema_version", "model_name", "results")
_OPTIONAL_FIELDS = (
    "author",
    "institution",
    "paper_url",
    "code_url",
    "date",
    "dataset_version",
    "hardware",
    "software_versions",
)
_RESULT_REQUIRED = ("task", "metrics")


def validate_submission(submission: dict[str, Any]) -> list[str]:
    """Validate a leaderboard submission dictionary.

    Parameters
    ----------
    submission : dict
        Submission data to validate.

    Returns
    -------
    list[str]
        List of validation error messages. Empty if valid.
    """
    errors: list[str] = []

    # Check required top-level fields
    for field in _REQUIRED_FIELDS:
        if field not in submission:
            errors.append(f"Missing required field: {field!r}")

    if errors:
        return errors  # Can't validate further without required fields

    # Schema version
    if submission["schema_version"] != SUBMISSION_SCHEMA_VERSION:
        errors.append(
            f"Unsupported schema_version: {submission['schema_version']!r} "
            f"(expected {SUBMISSION_SCHEMA_VERSION!r})"
        )

    # Model name must be non-empty string
    if not isinstance(submission["model_name"], str) or not submission["model_name"].strip():
        errors.append("model_name must be a non-empty string")

    # Results must be a non-empty list
    results = submission["results"]
    if not isinstance(results, list) or len(results) == 0:
        errors.append("results must be a non-empty list")
        return errors

    # Validate each result entry
    for i, result in enumerate(results):
        prefix = f"results[{i}]"

        if not isinstance(result, dict):
            errors.append(f"{prefix}: must be a dict")
            continue

        for field in _RESULT_REQUIRED:
            if field not in result:
                errors.append(f"{prefix}: missing required field {field!r}")

        if "task" in result:
            task = result["task"]
            if task not in _VALID_TASKS:
                errors.append(f"{prefix}: unknown task {task!r}")

        if "metrics" in result:
            metrics = result["metrics"]
            if not isinstance(metrics, dict):
                errors.append(f"{prefix}: metrics must be a dict")
            else:
                for key, value in metrics.items():
                    if not isinstance(value, (int, float)):
                        errors.append(f"{prefix}.metrics.{key}: must be a number")

    return errors


def load_submission(path: str | Path) -> dict[str, Any]:
    """Load and validate a submission from a JSON file.

    Parameters
    ----------
    path : str | Path
        Path to the JSON submission file.

    Returns
    -------
    dict[str, Any]
        Validated submission data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the submission fails validation.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    path = Path(path)
    with open(path) as f:
        data: dict[str, Any] = json.load(f)

    errors = validate_submission(data)
    if errors:
        msg = f"Invalid submission ({len(errors)} errors):\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise ValueError(msg)

    logger.info("Loaded valid submission from %s: %s", path, data.get("model_name", "unknown"))
    return data
