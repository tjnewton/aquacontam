"""Configuration loading for AquaContam data sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "data.yaml"
_DEFAULT_EXPERIMENT_PATH = Path(__file__).resolve().parents[2] / "configs" / "experiment.yaml"


def load_data_config(
    config_path: Path | str | None = None,
    source_name: str | None = None,
) -> dict[str, Any]:
    """Load the data ingestion YAML configuration.

    Parameters
    ----------
    config_path : Path | str | None
        Path to the YAML config file. Defaults to ``configs/data.yaml``
        relative to the repository root.
    source_name : str | None
        If provided, return only the section for this source
        (e.g. ``"ucmr5"``). Otherwise return the full config dict.

    Returns
    -------
    dict[str, Any]
        The full config or a single source section.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    KeyError
        If *source_name* is not found in the config.
    """
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        config: dict[str, Any] = dict(yaml.safe_load(fh))

    if source_name is not None:
        if source_name not in config:
            raise KeyError(
                f"Source '{source_name}' not found in config. "
                f"Available: {[k for k in config if k != 'defaults']}"
            )
        return dict(config[source_name])

    return config


def load_experiment_config(
    config_path: Path | str | None = None,
    section: str | None = None,
) -> dict[str, Any]:
    """Load the experiment YAML configuration.

    Parameters
    ----------
    config_path : Path | str | None
        Path to the YAML config file. Defaults to ``configs/experiment.yaml``
        relative to the repository root.
    section : str | None
        If provided, return only this top-level section (e.g. ``"models"``).
        Otherwise return the full config dict.

    Returns
    -------
    dict[str, Any]
        The full config or a single section.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    KeyError
        If *section* is not found in the config.
    """
    path = Path(config_path) if config_path is not None else _DEFAULT_EXPERIMENT_PATH

    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")

    with open(path, encoding="utf-8") as fh:
        config: dict[str, Any] = dict(yaml.safe_load(fh))

    if section is not None:
        if section not in config:
            raise KeyError(
                f"Section '{section}' not found in experiment config. "
                f"Available: {list(config.keys())}"
            )
        return dict(config[section])

    return config
