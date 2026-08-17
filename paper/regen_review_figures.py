"""DEPRECATED (#55, 2026-07): out-of-band review-figure regenerator, retired.

This script is no longer on the active pipeline path. The root cause it worked
around -- the ICP's per-epoch training ``history`` not being persisted -- is now
fixed at source: ``train_and_evaluate`` retains the fitted ICP per task and
``export_icp_diagnostics`` writes its ``history`` into ``icp_diagnostics.json``,
so ``generate_figures.py`` draws ED Fig. 4a directly and the reliability-diagram
key-name mismatch is fixed. ``pipeline/export.py::generate_paper_assets`` no
longer shells out to this script. It is kept (not deleted) only because
``paper/DERIVATIONS.tsv`` PROV rows reference it as historical provenance.

Original purpose (historical):

Regenerate the two review figures whose data was missing.

Two display items in ``manuscript_review.docx`` rendered without data:

* **Extended Data Fig. 4a** (ICP adversary convergence). The frozen
  ``icp_diagnostics.json`` only stored final metrics (``results``), never the
  per-epoch training ``history`` that panel a needs, so panel a fell back to a
  "No ICP training history available" placeholder. The ICP model *does* record
  history during ``fit()`` -- it simply was not persisted. This script re-fits
  the ICP classifier on T1 (on CPU, as an illustrative run) to capture the real
  convergence curves, merges them with the **frozen** ICP ``results`` (so panel
  b keeps the cited T1 0.777 / T4 0.573 metrics), and regenerates the figure.

* **Supplementary Fig. 5** (reliability diagram). A key-name mismatch in
  ``generate_figures.py`` (now fixed) left the curves empty. This regenerates it
  from the **frozen** calibration data so the legend ECE values match the paper.

Only these two figures are regenerated, via direct function calls, to keep the
change set minimal -- a full ``generate_figures.py`` run would rebuild every
figure from the (drifted) live ``results/`` directory. The frozen snapshot
(``results/paper_frozen/``) is never modified.

Run::

    python paper/regen_review_figures.py
"""

from __future__ import annotations

# Force CPU before any torch import so this short run sidesteps multi-GPU device
# selection on Windows (torch's default ordering can bind the wrong device).
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Project root on sys.path so ``paper.generate_figures`` imports cleanly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# NOTE: matplotlib (via paper.generate_figures) and torch (via ICPClassifier /
# _prep_data) are imported LAZILY inside the functions that use them, so this module
# imports under the CI ``[test]`` extras (those are ``[paper]``/``[dl]`` extras). That
# keeps the pipeline-safety tests collectable/runnable in CI, and the graceful-skip path
# never touches the heavy deps. CUDA_VISIBLE_DEVICES="" is set at module top (above),
# still before any torch import, so the CPU-force guarantee holds.

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("regen_review_figures")

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FROZEN_DIR = RESULTS_DIR / "paper_frozen"
FIGURES_DIR = ROOT / "paper" / "figures"
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"


def _load_cached_data() -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Load cached WQ data + feature sets (mirrors ``reproduce.py:_ensure_data``)."""
    wq_path = DATA_DIR / "interim" / "merged_wq.parquet"
    if not wq_path.exists():
        raise FileNotFoundError(
            f"Missing cached WQ data: {wq_path}. Run the preprocess stage first."
        )
    wq_df = pd.read_parquet(wq_path)
    logger.info("Loaded cached WQ data: %d rows", len(wq_df))

    features_dir = DATA_DIR / "interim" / "features"
    feature_dfs = [pd.read_parquet(p) for p in sorted(features_dir.glob("*.parquet"))]
    if not feature_dfs:
        raise FileNotFoundError(f"No cached feature sets found in {features_dir}.")
    logger.info("Loaded %d cached feature sets", len(feature_dfs))
    return wq_df, feature_dfs


def _load_icp_config() -> dict[str, Any]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    models = cfg.get("models")
    if not models or "icp_classifier" not in models:
        raise KeyError("configs/experiment.yaml is missing models.icp_classifier")
    return dict(models["icp_classifier"])


def capture_icp_history() -> list[dict[str, float]]:
    """Re-fit the ICP classifier on T1 (CPU) and return its per-epoch history."""
    # Lazy torch imports (see module-top note): only the actual re-fit needs them.
    from aquacontam.benchmark._prep_data import prepare_t1_data
    from aquacontam.models.icp import ICPClassifier

    wq_df, feature_dfs = _load_cached_data()
    td = prepare_t1_data(ICPClassifier, wq_df, feature_dfs)
    if td is None:
        raise RuntimeError("prepare_t1_data returned None -- cannot re-fit ICP for T1.")
    logger.info(
        "T1 data: train=%d val=%d features=%d",
        len(td.X_train),
        len(td.X_val),
        td.X_train.shape[1],
    )

    # Fit the FULL epoch schedule (no early stopping). The convergence panel
    # illustrates the training dynamics; with early stopping the run truncates at
    # ~epoch 22 -- right as warmup ends and the adversary activates -- leaving no
    # post-warmup window to show. Omitting X_val/y_val disables EarlyStopping in
    # ICPClassifier.fit so all configured epochs run. Headline metrics still come
    # from the frozen snapshot, so this illustrative run changes no cited number.
    model = ICPClassifier(config=_load_icp_config())
    model.fit(td.X_train, td.y_train, **td.fit_extra)
    history = model.get_training_history()
    if not history:
        raise RuntimeError("ICP training produced no history.")
    logger.info("Captured ICP training history: %d epochs", len(history))
    return history


def _read_results_json(results_dir: Path, name: str) -> dict[str, Any] | None:
    """Read ``name`` from ``results_dir``, falling back to the frozen archive.

    When wired into the reproducibility pipeline, ``results_dir`` is the live run's output
    directory (e.g. ``results_reframe``); standalone, it is the frozen archive. Returns
    ``None`` if neither has the file, so the caller can skip gracefully rather than break
    a long pipeline run.
    """
    for base in (results_dir, FROZEN_DIR):
        p = base / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def build_icp_data(history: list[dict[str, float]], results_dir: Path) -> dict[str, Any]:
    """ICP results (cited metrics, from ``results_dir``) + freshly captured T1 history."""
    diag = _read_results_json(results_dir, "icp_diagnostics.json") or {}
    return {"results": diag.get("results", []), "history": {"T1": history}}


def _log_convergence_summary(history: list[dict[str, float]]) -> None:
    """Print an acceptance hint: does adversary loss rise after warmup (per caption)?"""
    warmup_end = next((h["epoch"] for h in history if h["lambda_adv"] > 0), None)
    post = [h["adv_loss"] for h in history if warmup_end is not None and h["epoch"] >= warmup_end]
    logger.info(
        "Convergence: task_loss %.4f->%.4f | adv_loss %.4f->%.4f | warmup_end=%s",
        history[0]["task_loss"],
        history[-1]["task_loss"],
        history[0]["adv_loss"],
        history[-1]["adv_loss"],
        warmup_end,
    )
    if post and len(post) >= 2:
        trend = "RISES" if post[-1] > post[0] else "does NOT rise"
        logger.info(
            "Post-warmup adversary loss %s (%.4f -> %.4f). Caption expects a rise; "
            "inspect panel a before shipping.",
            trend,
            post[0],
            post[-1],
        )


def regenerate(results_dir: Path, figures_dir: Path) -> bool:
    """Re-fit the ICP for its history and regenerate the two review figures.

    Reads ICP results + calibration from ``results_dir`` (frozen-archive fallback). Returns
    True if both figures were regenerated, False if inputs were missing (caller decides
    whether that is fatal). NEVER modifies the frozen archive.
    """
    calibration = _read_results_json(results_dir, "calibration_analysis.json")
    diag = _read_results_json(results_dir, "icp_diagnostics.json")
    if calibration is None or diag is None:
        logger.warning(
            "Skipping review-figure regeneration: missing calibration/icp_diagnostics in %s "
            "(and frozen fallback)",
            results_dir,
        )
        return False

    # Lazy matplotlib import (see module-top note): only reached once inputs are present,
    # so the graceful-skip path above never needs the [paper] extra.
    from paper.generate_figures import fig_icp_analysis, fig_reliability_diagram

    history = capture_icp_history()
    _log_convergence_summary(history)
    icp_data = build_icp_data(history, results_dir)

    # Persist merged diagnostics to results_dir (so a later full figure run over the same
    # dir reproduces panel a). For the frozen dir this is skipped — the archive is immutable.
    if results_dir.resolve() != FROZEN_DIR.resolve():
        (results_dir / "icp_diagnostics.json").write_text(
            json.dumps(icp_data, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Wrote %s/icp_diagnostics.json (results + fresh T1 history)", results_dir)

    figures_dir.mkdir(parents=True, exist_ok=True)
    p_icp = fig_icp_analysis(icp_data, figures_dir)
    p_cal = fig_reliability_diagram(calibration, figures_dir)
    logger.info("Regenerated %s", p_icp)
    logger.info("Regenerated %s", p_cal)
    return True


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results",
        default=str(FROZEN_DIR),
        help="Results directory holding icp_diagnostics.json + calibration_analysis.json "
        "(default: the frozen archive; the pipeline passes its live output dir).",
    )
    ap.add_argument("--figures", default=str(FIGURES_DIR), help="Figure output directory.")
    args = ap.parse_args()
    # A missing-input skip is intentional (regenerate() logs it) and NOT a hard failure,
    # so the pipeline's long run is never broken by absent optional figure inputs.
    regenerate(Path(args.results), Path(args.figures))
    return 0


if __name__ == "__main__":
    sys.exit(main())
