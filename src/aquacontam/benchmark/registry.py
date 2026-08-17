"""Benchmark task registry — decorator-based task registration and dispatch.

Tasks register themselves via ``@register_task(...)`` and can be
discovered/run uniformly by the benchmark runner.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskInfo:
    """Metadata for a benchmark task.

    Parameters
    ----------
    name : str
        Short task identifier (e.g., ``"T1"``).
    description : str
        Human-readable description.
    task_type : str
        ``"classification"`` or ``"regression"``.
    primary_metric : str
        Main metric for leaderboard ranking.
    analytes : tuple[str, ...]
        Default analytes for this task.
    """

    name: str
    description: str
    task_type: str
    primary_metric: str
    analytes: tuple[str, ...]


@dataclass
class TaskResult:
    """Result of running a benchmark task.

    Parameters
    ----------
    task_name : str
        Name of the task that was run.
    model_name : str
        Name of the model that was evaluated.
    metrics : dict[str, float]
        Aggregated metrics across splits.
    split_metrics : dict[str, dict[str, float]]
        Per-split metrics (train/val/test).
    metadata : dict[str, Any]
        Additional info (analyte, n_samples, etc.).
    """

    task_name: str
    model_name: str
    metrics: dict[str, float]
    split_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# Module-level registry
_REGISTRY: dict[str, tuple[TaskInfo, Callable[..., TaskResult]]] = {}


def register_task(
    name: str,
    description: str,
    task_type: str,
    primary_metric: str,
    analytes: tuple[str, ...] | list[str],
) -> Callable[[Callable[..., TaskResult]], Callable[..., TaskResult]]:
    """Decorator to register a benchmark task function.

    Parameters
    ----------
    name : str
        Unique task identifier.
    description : str
        Human-readable description.
    task_type : str
        ``"classification"`` or ``"regression"``.
    primary_metric : str
        Primary metric for ranking.
    analytes : tuple or list of str
        Default analytes for this task.

    Returns
    -------
    Callable
        Decorator that registers the function and returns it unchanged.

    Raises
    ------
    ValueError
        If a task with the same name is already registered.
    """

    def decorator(func: Callable[..., TaskResult]) -> Callable[..., TaskResult]:
        if name in _REGISTRY:
            raise ValueError(f"Task {name!r} is already registered")

        info = TaskInfo(
            name=name,
            description=description,
            task_type=task_type,
            primary_metric=primary_metric,
            analytes=tuple(analytes),
        )
        _REGISTRY[name] = (info, func)
        logger.debug("Registered benchmark task: %s", name)
        return func

    return decorator


def get_task(name: str) -> tuple[TaskInfo, Callable[..., TaskResult]]:
    """Retrieve a registered task by name.

    Parameters
    ----------
    name : str
        Task identifier.

    Returns
    -------
    tuple[TaskInfo, Callable[..., TaskResult]]
        Task metadata and runner function.

    Raises
    ------
    KeyError
        If the task is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown task {name!r}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_tasks() -> list[TaskInfo]:
    """List all registered tasks.

    Returns
    -------
    list[TaskInfo]
        Sorted by task name.
    """
    return [info for info, _ in sorted(_REGISTRY.values(), key=lambda t: t[0].name)]


def run_task(name: str, **kwargs: Any) -> TaskResult:
    """Run a single benchmark task.

    Parameters
    ----------
    name : str
        Task identifier.
    **kwargs
        Arguments passed to the task function.

    Returns
    -------
    TaskResult
        Results from the task run.
    """
    info, func = get_task(name)
    logger.info("Running task %s: %s", info.name, info.description)
    return func(**kwargs)


def run_all_tasks(**kwargs: Any) -> list[TaskResult]:
    """Run all registered benchmark tasks.

    Parameters
    ----------
    **kwargs
        Arguments passed to each task function.

    Returns
    -------
    list[TaskResult]
        Results from all tasks.
    """
    results = []
    for name in sorted(_REGISTRY.keys()):
        try:
            result = run_task(name, **kwargs)
            results.append(result)
        except Exception as exc:
            logger.error("Task %s failed", name, exc_info=True)
            results.append(
                TaskResult(
                    task_name=name,
                    model_name="unknown",
                    metrics={},
                    metadata={"error": str(exc)},
                )
            )
    return results


def clear_registry() -> None:
    """Clear all registered tasks (for testing)."""
    _REGISTRY.clear()
