"""Per-trial subprocess isolation for Optuna HPO with wall-clock timeouts.

Wraps a single model fit + evaluate in a `spawn`-context subprocess so that a
wedged CUDA kernel (which cannot be interrupted from Python) can be killed by
the parent. The parent enforces a wall-clock timeout via
``process.join(timeout=N)`` followed by ``process.kill()``; on Windows this
maps to ``TerminateProcess`` which forcibly reclaims the GPU.

Top-level imports must NOT include ``torch`` or ``torch_geometric`` — those
libraries initialize CUDA at first import and read ``CUDA_VISIBLE_DEVICES``
then. The child function ``_run_in_child`` sets ``os.environ`` first, then
lazy-imports the model class.
"""

from __future__ import annotations

import importlib
import logging
import multiprocessing as mp
import os
import pickle
import queue
import subprocess
import sys
import time
import warnings
from typing import Any, Literal, NamedTuple

logger = logging.getLogger(__name__)


# Per-model wall-clock timeouts. Cap = >= 3x measured T4 walltime per trial,
# except tabpfn (no T4 data yet — generous 90 min). gnn_* models are capped
# at 20 min on the assumption that any gnn trial exceeding that is wedged.
DEFAULT_TIMEOUT_S: float = 1800.0  # 30 min — fits all models seen so far

PER_MODEL_TIMEOUTS: dict[str, float] = {
    "tabpfn_classifier": 5400.0,
    "zi_tobit_classifier": 3600.0,
    "deep_tobit_classifier": 3600.0,
    "cnn1d_classifier": 2700.0,
    "gnn_gcn_classifier": 1200.0,
    "gnn_sage_classifier": 1200.0,
}

# Consecutive-timeout counter per model_key. Reset on any non-timeout result.
# After ``MAX_CONSECUTIVE_TIMEOUTS`` in a row on the same model, the runner
# raises so the sweep halts rather than silently churning through a wedged GPU.
MAX_CONSECUTIVE_TIMEOUTS: int = 3
_consecutive_timeouts: dict[str, int] = {}


class TrialResult(NamedTuple):
    """Outcome of one trial."""

    value: float | None
    status: Literal["ok", "timeout", "error", "crash"]
    message: str


def get_timeout(model_key: str) -> float:
    """Return the wall-clock timeout (seconds) for a model key."""
    return PER_MODEL_TIMEOUTS.get(model_key, DEFAULT_TIMEOUT_S)


def reset_timeout_counter(model_key: str | None = None) -> None:
    """Reset the consecutive-timeout counter (all models if ``model_key`` is None)."""
    if model_key is None:
        _consecutive_timeouts.clear()
    else:
        _consecutive_timeouts.pop(model_key, None)


def _probe_gpu_post_kill(gpu_id: int | None) -> None:
    """Log a warning if GPU VRAM is still held after a process kill.

    Uses ``nvidia-smi`` directly so the parent doesn't need a torch CUDA
    context of its own. Best-effort; failures to run nvidia-smi are silent.
    """
    if gpu_id is None:
        return
    try:
        time.sleep(2.0)
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                f"--id={gpu_id}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0:
            held_mib = int(out.stdout.strip().splitlines()[0])
            if held_mib > 1024:
                logger.warning(
                    "GPU %d still holding %d MiB after subprocess kill — possible CUDA context leak",
                    gpu_id,
                    held_mib,
                )
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass


def _run_in_child(
    model_module: str,
    model_class_name: str,
    config: dict[str, Any],
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    fit_kwargs: dict[str, Any],
    metric: str,
    task_type: str,
    env: dict[str, str],
    result_queue: Any,
) -> None:
    """Child entrypoint: instantiate model, fit, evaluate, send result via Queue.

    Order is load-bearing:
        1. ``os.environ.update(env)`` MUST run before any torch import.
        2. ``warnings.filterwarnings("ignore")`` silences the CuBLAS /
           torch_geometric warning storm that would otherwise re-emit per
           child process.
        3. Lazy-import the model class.
    """
    os.environ.update(env)
    warnings.filterwarnings("ignore")

    try:
        module = importlib.import_module(model_module)
        model_cls = getattr(module, model_class_name)
    except (ImportError, AttributeError) as exc:
        result_queue.put((None, "error", f"{type(exc).__name__}: {exc}"))
        sys.exit(0)

    try:
        import numpy as np

        model = model_cls(config=config)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val, **fit_kwargs)
        metrics = model.evaluate(X_val, y_val, metrics=[metric], task_type=task_type)
        value = metrics.get(metric)
        if value is None or not np.isfinite(value):
            result_queue.put((None, "error", f"metric {metric!r} was None or non-finite"))
        else:
            result_queue.put((float(value), "ok", ""))
    except BaseException as exc:
        # Includes SystemExit, KeyboardInterrupt, MemoryError, library-specific
        # errors. The parent always frees its end of the queue, so this is safe.
        result_queue.put((None, "error", f"{type(exc).__name__}: {exc}"))
    finally:
        sys.exit(0)


def run_trial_with_timeout(
    model_module: str,
    model_class_name: str,
    config: dict[str, Any],
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    fit_kwargs: dict[str, Any],
    metric: str,
    task_type: str,
    timeout_s: float,
    env: dict[str, str] | None = None,
    gpu_id: int | None = None,
    model_key: str = "",
) -> TrialResult:
    """Run one HPO trial in an isolated subprocess with a wall-clock timeout.

    Parameters
    ----------
    model_module, model_class_name : str
        Dotted module path and class name (e.g. ``("aquacontam.models.gnn",
        "GNNClassifier")``). Passed by string so the parent doesn't need to
        import the model class (which may pull in torch).
    config, X_train, y_train, X_val, y_val, fit_kwargs : various
        Forwarded to the model's ``fit`` / ``evaluate`` in the child.
    metric, task_type : str
        Forwarded to ``model.evaluate``.
    timeout_s : float
        Wall-clock timeout for this trial.
    env : dict, optional
        Environment variables to set in the child before any imports. Use
        this to propagate ``CUDA_VISIBLE_DEVICES``, ``AQUACONTAM_GPU_ID``,
        ``CUBLAS_WORKSPACE_CONFIG``.
    gpu_id : int, optional
        Index passed to ``_probe_gpu_post_kill`` on timeout to log VRAM
        leaks. Does not affect routing — that's via ``env``.
    model_key : str
        Registry key, used for the consecutive-timeout halt counter.

    Returns
    -------
    TrialResult
        ``(value, status, message)``. ``status`` is ``"ok"`` /
        ``"timeout"`` / ``"error"`` / ``"crash"``.

    Raises
    ------
    RuntimeError
        After ``MAX_CONSECUTIVE_TIMEOUTS`` consecutive timeouts on the same
        ``model_key`` — halts the sweep rather than letting it churn silently
        on a degraded GPU.
    """
    env = dict(env or {})

    # Pre-flight: surface pickling errors before spawning a child. A child
    # that fails to unpickle its args dies during interpreter bootstrap and
    # the parent only sees ``exitcode != 0`` (uninformative). Pickling here
    # is fast (the same arrays will be pickled by the spawn machinery).
    payload = (
        model_module,
        model_class_name,
        config,
        X_train,
        y_train,
        X_val,
        y_val,
        fit_kwargs,
        metric,
        task_type,
        env,
    )
    try:
        pickle.dumps(payload)
    except (pickle.PicklingError, TypeError, AttributeError) as exc:
        return TrialResult(None, "error", f"input not pickleable: {type(exc).__name__}: {exc}")

    ctx = mp.get_context("spawn")
    result_queue: Any = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_run_in_child,
        args=(*payload, result_queue),
        daemon=False,
    )
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        # Wall-clock timeout — wedged child.
        proc.kill()
        proc.join(timeout=10)
        _probe_gpu_post_kill(gpu_id)
        # Drain the queue so the underlying pipe / FD doesn't leak.
        try:
            while True:
                result_queue.get_nowait()
        except queue.Empty:
            pass
        result_queue.close()
        proc.close()

        if model_key:
            _consecutive_timeouts[model_key] = _consecutive_timeouts.get(model_key, 0) + 1
            if _consecutive_timeouts[model_key] >= MAX_CONSECUTIVE_TIMEOUTS:
                raise RuntimeError(
                    f"Trial subprocess wedged GPU {MAX_CONSECUTIVE_TIMEOUTS} times in a row "
                    f"on {model_key} — halting sweep to prevent silent loss"
                )
        return TrialResult(None, "timeout", f"killed after {timeout_s}s")

    # Child has exited. Determine outcome.
    try:
        value, status, msg = result_queue.get_nowait()
    except queue.Empty:
        # Child died before putting anything on the queue.
        result_queue.close()
        exitcode = proc.exitcode
        proc.close()
        return TrialResult(None, "crash", f"exitcode={exitcode}")
    result_queue.close()
    proc.close()

    if status == "ok" and model_key:
        _consecutive_timeouts.pop(model_key, None)
    return TrialResult(value, status, msg)


def run_trial_inproc(
    model_cls: Any,
    config: dict[str, Any],
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    fit_kwargs: dict[str, Any],
    metric: str,
    task_type: str,
) -> TrialResult:
    """In-process trial runner — debugging escape hatch.

    Use when ``--no-trial-isolation`` / ``AQUACONTAM_HPO_INPROC=1`` is set,
    or when the subprocess machinery itself is suspect. Provides no crash
    isolation — a wedged CUDA kernel will hang the calling process.
    """
    import numpy as np

    try:
        model = model_cls(config=config)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val, **fit_kwargs)
        metrics = model.evaluate(X_val, y_val, metrics=[metric], task_type=task_type)
        value = metrics.get(metric)
        if value is None or not np.isfinite(value):
            return TrialResult(None, "error", f"metric {metric!r} was None or non-finite")
        return TrialResult(float(value), "ok", "")
    except Exception as exc:
        return TrialResult(None, "error", f"{type(exc).__name__}: {exc}")
