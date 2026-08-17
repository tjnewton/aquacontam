#!/usr/bin/env python
"""C10 (R5 M1): significance tests + CIs for every demographic group gap.

The abstract states the model "underperforms for high-people-of-color systems
(false-negative rate 0.50 versus 0.44)" with no CI or test. R5 (M1) shows the gap is
not statistically significant (two-proportion p ~ 0.16) and that the framing is selective
(the same split gives high-PoC systems HIGHER AUROC and BETTER conformal coverage; the FNR
gap is near-zero for education and reversed for low income). This COMPUTE formalizes it:
two-proportion tests + Wilson CIs for the FNR and conformal-coverage gaps, and a
Hanley-McNeil two-sample AUC test for the AUROC gap, for people-of-color / low-income /
less-than-HS-education, from the frozen counts. No retrain.

Reproduction gate (before anything is written):
  * the PoC FNR gap reproduces ``group_error_calibration.json`` fnr_gap (0.0624), and
  * the PoC AUROC-gap p-value reproduces ``inference_strengthening.json`` m3b (0.0596),
proving the derived counts and the reused ``auc_difference`` match the published archive.

Output: ``results/paper_frozen/group_gap_tests.json`` (then DERIVATIONS row, rehash, stamp,
markers). Interpretation map (pre-registered): a gap with p >= 0.05 ⇒ the abstract states it
is "directionally present but not individually significant (p ~ X)"; a gap with p < 0.05 ⇒
report the test and keep the claim. An outcome outside that map ⇒ escalate.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
OUT_PATH = FROZEN / "group_gap_tests.json"
GATE_TOL = 2e-3
DIMENSIONS = ("pct_people_of_color", "pct_low_income", "pct_less_hs_education")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("compute_group_gap_tests")


def _wilson_ci(x: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = x / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - half, center + half)


def _two_proportion(x1: int, n1: int, x2: int, n2: int) -> dict:
    """Two-sided pooled two-proportion z-test with Wilson CIs on each proportion."""
    from aquacontam.analysis.strengthening import _normal_cdf

    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    lo1, hi1 = _wilson_ci(x1, n1)
    lo2, hi2 = _wilson_ci(x2, n2)
    return {
        "p_high": p1,
        "n_high": n1,
        "ci_high": [lo1, hi1],
        "p_low": p2,
        "n_low": n2,
        "ci_low": [lo2, hi2],
        "gap": p1 - p2,
        "z": z,
        "p_value": p,
        "significant_05": bool(p < 0.05),
    }


def main() -> int:
    from aquacontam.analysis.strengthening import auc_difference

    gec = json.loads((FROZEN / "group_error_calibration.json").read_text(encoding="utf-8"))[
        "error_rates"
    ]
    infs = json.loads((FROZEN / "inference_strengthening.json").read_text(encoding="utf-8"))
    conf = json.loads((FROZEN / "group_conformal_demographic.json").read_text(encoding="utf-8"))[
        "results"
    ]

    out: dict[str, dict] = {}
    for dim in DIMENSIONS:
        d = gec.get(dim, {})
        hi, lo = d.get("high"), d.get("low")
        if not hi or not lo:
            logger.warning("%s: missing high/low group; skipping", dim)
            continue
        # Derive integer case counts from n + positive_rate; FN from fnr * positives.
        npos_h = round(hi["n"] * hi["positive_rate"])
        npos_l = round(lo["n"] * lo["positive_rate"])
        nneg_h, nneg_l = hi["n"] - npos_h, lo["n"] - npos_l
        fn_h, fn_l = round(hi["fnr"] * npos_h), round(lo["fnr"] * npos_l)

        fnr_test = _two_proportion(fn_h, npos_h, fn_l, npos_l)
        auroc_test = auc_difference(hi["auroc"], npos_h, nneg_h, lo["auroc"], npos_l, nneg_l)
        entry = {"fnr_gap_test": fnr_test, "auroc_gap_test": auroc_test}

        cov = conf.get(dim, {}).get("by_alpha", {}).get("0.05")
        if cov:
            hn, ln = int(cov["high_n"]), int(cov["low_n"])
            cov_test = _two_proportion(
                round(cov["high_group_coverage"] * hn),
                hn,
                round(cov["low_group_coverage"] * ln),
                ln,
            )
            entry["coverage_gap_test"] = cov_test
        out[dim] = entry

    # --- reproduction gate ---
    poc = out.get("pct_people_of_color", {})
    fnr_gap = poc.get("fnr_gap_test", {}).get("gap", float("nan"))
    auroc_p = poc.get("auroc_gap_test", {}).get("p_value", float("nan"))
    frozen_fnr_gap = gec["pct_people_of_color"]["fnr_gap"]
    frozen_auroc_p = infs["m3b_group_auroc_gap"]["p_value"]
    gate_fails = []
    if abs(fnr_gap - frozen_fnr_gap) > GATE_TOL:
        gate_fails.append(f"PoC FNR gap {fnr_gap:.4f} vs frozen {frozen_fnr_gap:.4f}")
    if abs(auroc_p - frozen_auroc_p) > GATE_TOL:
        gate_fails.append(f"PoC AUROC-gap p {auroc_p:.4f} vs frozen m3b {frozen_auroc_p:.4f}")
    if gate_fails:
        logger.error("REPRODUCTION GATE FAILED: %s — nothing written", gate_fails)
        return 1
    logger.info(
        "Reproduction gate OK: PoC FNR gap %.4f, AUROC-gap p %.4f reproduce frozen",
        fnr_gap,
        auroc_p,
    )

    payload = {
        "_meta": {
            "source": (
                "C10 (R5 M1) two-proportion FNR/coverage-gap tests (+ Wilson CIs) and "
                "Hanley-McNeil AUC-difference tests per demographic dimension, from frozen "
                "group_error_calibration.json / group_conformal_demographic.json counts. "
                "Reproduction gate ties PoC FNR gap to gec fnr_gap and PoC AUROC-gap p to "
                f"inference_strengthening m3b within {GATE_TOL}. No retrain."
            ),
        },
        **out,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for dim, e in out.items():
        logger.info(
            "%s: FNR gap %.4f p=%.3f%s | AUROC gap %.4f p=%.3f | %s",
            dim,
            e["fnr_gap_test"]["gap"],
            e["fnr_gap_test"]["p_value"],
            " (SIG)" if e["fnr_gap_test"]["significant_05"] else "",
            e["auroc_gap_test"]["delta"],
            e["auroc_gap_test"]["p_value"],
            f"cov gap p={e['coverage_gap_test']['p_value']:.3f}"
            if "coverage_gap_test" in e
            else "",
        )
    logger.info("wrote %s", OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
