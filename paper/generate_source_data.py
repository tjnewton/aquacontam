#!/usr/bin/env python
"""Generate Nature Water Source Data Excel files for each main-text figure.

Nature Water requires the source data underlying each figure as an Excel (.xlsx)
file. This script generates one file per **current** main figure, reading the
committed frozen archive so the numbers are exactly those the figures plot:

    Fig. 1  National-scale data integration   -> dataset summary
    Fig. 2  Two compounding confounds          -> split_comparison + detection_only_ablation
                                                  + feature_ablation
    Fig. 3  Models learn the monitoring process -> shap_T1 + causal_deconfounding
    Fig. 4  Inequitable monitoring             -> monitoring_inequity (a, b)
                                                  + monitoring_design (c)

There is no Figure 5 in the submitted manuscript (the national risk map is a
webapp asset). The freshness and figure↔data fidelity of these files is enforced
by the deterministic gate clause **G14** (`paper/final_gate.py`), which
regenerates from the frozen archive and asserts per-figure sentinel values — so a
stale layout (a historical defect class) cannot ship green again.

Usage::

    python paper/generate_source_data.py
    python paper/generate_source_data.py --results results/paper_frozen --output paper/source_data
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import click
import pandas as pd


def _load_json(path: Path) -> dict | list:
    """Load a JSON file, failing loud if it is missing.

    Fail-loud (not warn-and-None): a missing input must not silently produce a
    placeholder sheet — that is exactly how an earlier generator shipped Source
    Data that did not match the figures. G14 also depends on regeneration
    erroring rather than emitting placeholders.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required frozen input missing: {path}")
    data: dict | list = json.loads(path.read_text(encoding="utf-8"))
    return data


def generate_fig1_source_data(results_dir: Path, output_dir: Path) -> Path:
    """Fig. 1: Dataset summary — per-source records, analytes, period, censoring."""
    data = {
        "Source": [
            "UCMR5",
            "UCMR3",
            "SDWIS",
            "MI MPART",
            "CA GeoTracker",
            "NJ DEP",
            "WQP",
            "MO DNR",
            "WA DOH",
            "EJScreen",
        ],
        "Records": [
            1928117,
            1069174,
            916899,
            5640,
            324254,
            248107,
            35528,
            75971,
            9251,
            220000,
        ],
        "Analytes": [
            "29 PFAS + Li",
            "6 PFAS + 32",
            "Pb/Cu",
            "5 PFAS",
            "34 PFAS",
            "25 PFAS",
            "13 PFAS",
            "29 PFAS",
            "14 PFAS",
            "Demographics",
        ],
        "Period": [
            "2023",
            "2013-2015",
            "2016-2022",
            "2019-2023",
            "2019-2023",
            "2019-2023",
            "2020-2024",
            "2020-2023",
            "2023-2024",
            "2023",
        ],
        "Censoring_Rate": [
            0.971,
            0.764,
            None,
            None,
            None,
            None,
            0.387,
            0.989,
            None,
            None,
        ],
    }
    df = pd.DataFrame(data)
    path = output_dir / "Fig1_source_data.xlsx"
    df.to_excel(path, index=False, sheet_name="Dataset Summary")
    return path


def generate_fig2_source_data(results_dir: Path, output_dir: Path) -> Path:
    """Fig. 2: the two compounding confounds the figure actually plots.

    Panel (a) spatial leakage: random-vs-geographic AUPRC on the baseline feature
    set (``split_comparison.json`` ``simple``). Panel (b) ascertainment: T1 AUPRC
    with vs without detection-only sources (``detection_only_ablation.json``) and
    T4 AUPRC with vs without monitoring-intensity features (``feature_ablation.json``).
    """
    split = _load_json(results_dir / "split_comparison.json")
    det = _load_json(results_dir / "detection_only_ablation.json")
    feat = _load_json(results_dir / "feature_ablation.json")

    simple = split["simple"] if isinstance(split, dict) else {}
    leakage = pd.DataFrame(
        {
            "Split": ["Random", "Geographic"],
            "AUPRC": [
                simple["random_split_metrics"]["auprc"],
                simple["geographic_split_metrics"]["auprc"],
            ],
            "AUROC": [
                simple["random_split_metrics"]["auroc"],
                simple["geographic_split_metrics"]["auroc"],
            ],
        }
    )
    inflation = simple["inflation_ratio"]["auprc"]

    rows = []
    for e in det if isinstance(det, list) else []:
        if e.get("task") == "T1" and e.get("model") == "xgboost_classifier":
            rows.append(
                {
                    "Ablation": "T1: detection-only sources removed",
                    "Metric": "AUPRC",
                    "With": e["all_sources"]["auprc"],
                    "Without": e["excluding_detection_only"]["auprc"],
                }
            )
            rows.append(
                {
                    "Ablation": "T1: detection-only sources removed",
                    "Metric": "AUROC",
                    "With": e["all_sources"]["auroc"],
                    "Without": e["excluding_detection_only"]["auroc"],
                }
            )
            break
    for e in feat if isinstance(feat, list) else []:
        if e.get("task") == "T4" and e.get("category") == "monitoring_intensity":
            rows.append(
                {
                    "Ablation": "T4: monitoring-intensity features ablated",
                    "Metric": "AUPRC",
                    "With": e["baseline_score"],
                    "Without": e["ablated_score"],
                }
            )
            break
    ascertainment = pd.DataFrame(rows)

    path = output_dir / "Fig2_source_data.xlsx"
    with pd.ExcelWriter(path) as writer:
        leakage.to_excel(writer, index=False, sheet_name="a Spatial leakage")
        pd.DataFrame({"Inflation ratio (AUPRC)": [inflation]}).to_excel(
            writer, index=False, sheet_name="a Spatial leakage", startrow=len(leakage) + 2
        )
        ascertainment.to_excel(writer, index=False, sheet_name="b Ascertainment")
    return path


def generate_fig3_source_data(results_dir: Path, output_dir: Path) -> Path:
    """Fig. 3: SHAP importances (top features) + DML adjusted associations."""
    shap = _load_json(results_dir / "shap_T1.json")
    causal = _load_json(results_dir / "causal_deconfounding.json")

    mean_abs = shap.get("mean_abs_shap", {}) if isinstance(shap, dict) else {}
    top = sorted(mean_abs.items(), key=lambda kv: kv[1], reverse=True)[:15]
    df_shap = pd.DataFrame(top, columns=["Feature", "Mean_Abs_SHAP"])

    df_causal = pd.DataFrame(causal if isinstance(causal, list) else [])

    path = output_dir / "Fig3_source_data.xlsx"
    with pd.ExcelWriter(path) as writer:
        df_shap.to_excel(writer, index=False, sheet_name="SHAP importance (top 15)")
        df_causal.to_excel(writer, index=False, sheet_name="Adjusted associations (DML)")
    return path


#: Reader-facing labels for the monitoring-inequity demographic dimensions.
_INEQUITY_LABELS = {
    "pct_people_of_color": "% People of color",
    "pct_low_income": "% Low income",
    "pct_less_hs_education": "% Less than HS education",
    "pct_limited_english": "% Limited English",
}


#: Fig. 4c scopes (top to bottom) and design rows, from ``monitoring_design.json`` E1.
DESIGN_SCOPES: tuple[tuple[str, str], ...] = (
    ("national", "Nation (leave-one-region-out mean)"),
    ("test", "Test regions (EPA Regions 8–10)"),  # noqa: RUF001
    ("region9", "EPA Region 9"),
)
DESIGN_LABELS: dict[str, str] = {
    "pooled": "All sources (published estimate)",
    "ndr_pooled": "Programs reporting non-detects",
    "federal": "Federal census (UCMR3/UCMR5)",
    "state": "State programs",
    "state_ca": "State program (CA GeoTracker)",
}
#: Fig. 4c bottom block: the E4 primary California comparison.
CA_SCOPE = "California: census vs targeted"
CA_DESIGNS: tuple[tuple[str, str], ...] = (
    ("census", "Census (UCMR5)"),
    ("targeted_design", "Targeted (GeoTracker, first sampled 2019–2022)"),  # noqa: RUF001
)


def design_forest_rows(md: dict) -> list[dict]:
    """Fig. 4c rows: E1 design-restricted burden ratios by scope, then the E4 primary pair.

    Shared by ``generate_figures.fig_monitoring_inequity`` and the Fig. 4 Source
    Data so the plotted and tabulated values cannot drift. National rows are
    n_systems-weighted means of per-region ratios; their ``Systems`` sums the
    included regions.
    """
    e1 = md["e1_design_restricted"]
    rows: list[dict] = []
    for scope, scope_label in DESIGN_SCOPES:
        block = e1[scope]["rows"]
        per_region = e1[scope].get("per_region")
        for key, label in DESIGN_LABELS.items():
            est = block.get(key)
            if est is None:
                continue
            if per_region is not None:
                series = key.removesuffix("_pooled")
                n = sum(int(per_region[str(r)][series]["n_systems"]) for r in est["regions"])
                k = len(est["regions"])
                if k < len(per_region):
                    label = f"{label} ({k} region{'' if k == 1 else 's'})"
            else:
                n = int(est["n_systems"])
            rows.append(
                {
                    "Scope": scope_label,
                    "Design": label,
                    "Design_Key": key,
                    "Burden_Ratio": est["burden_ratio"],
                    "CI_Lower": est["ci_lower"],
                    "CI_Upper": est["ci_upper"],
                    "Systems": n,
                }
            )
    primary = md["e4_ca_census_vs_targeted"]["primary"]
    for key, label in CA_DESIGNS:
        est = primary[key]
        rows.append(
            {
                "Scope": CA_SCOPE,
                "Design": label,
                "Design_Key": key,
                "Burden_Ratio": est["burden_ratio"],
                "CI_Lower": est["ci_lower"],
                "CI_Upper": est["ci_upper"],
                "Systems": int(est["n_systems"]),
            }
        )
    return rows


def generate_fig4_source_data(results_dir: Path, output_dir: Path) -> Path:
    """Fig. 4: monitoring inequity and design dependence — the values the figure plots.

    Panel (a): mean sampling intensity in high- vs low-share systems and the
    high/low ratio per demographic dimension. Panel (b): population-size-adjusted
    regression coefficients with SEs. Source: ``monitoring_inequity.json``.
    Panel (c): people-of-color detection-burden ratios by monitoring design with
    95% CIs. Source: ``monitoring_design.json``.
    """
    ineq = _load_json(results_dir / "monitoring_inequity.json")
    rows = []
    for key, label in _INEQUITY_LABELS.items():
        d = ineq.get(key, {}) if isinstance(ineq, dict) else {}
        if not d:
            continue
        ratio = d.get("monitoring_ratio")
        rows.append(
            {
                "Dimension": label,
                "Mean_Samples_High": d.get("mean_high"),
                "Mean_Samples_Low": d.get("mean_low"),
                # An infinite ratio = no low-share systems (empty reference bin);
                # written as a string so the sheet is legible and openpyxl-safe.
                "Monitoring_Ratio": (
                    "inf (no low-share systems)"
                    if ratio is not None and math.isinf(ratio)
                    else ratio
                ),
                "N_High": d.get("n_high"),
                "N_Low": d.get("n_low"),
                "SizeAdj_Coef": d.get("size_adjusted_coef"),
                "SizeAdj_SE": d.get("size_adjusted_se"),
                "SizeAdj_P": d.get("size_adjusted_p"),
            }
        )
    df = pd.DataFrame(rows)
    md = _load_json(results_dir / "monitoring_design.json")
    design = pd.DataFrame(design_forest_rows(md)).drop(columns="Design_Key")  # type: ignore[arg-type]
    path = output_dir / "Fig4_source_data.xlsx"
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, index=False, sheet_name="Monitoring inequity")
        design.to_excel(writer, index=False, sheet_name="c Design decomposition")
    return path


@click.command()
@click.option("--results", default="results/paper_frozen", help="Frozen results directory.")
@click.option("--output", "-o", default="paper/source_data", help="Output directory.")
def main(results: str, output: str) -> None:
    """Generate Source Data Excel files for the four main-text figures."""
    results_dir = Path(results)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    generators = [
        ("Fig. 1", generate_fig1_source_data),
        ("Fig. 2", generate_fig2_source_data),
        ("Fig. 3", generate_fig3_source_data),
        ("Fig. 4", generate_fig4_source_data),
    ]
    for label, gen_fn in generators:
        path = gen_fn(results_dir, output_dir)
        click.echo(f"  {label}: {path}")
    click.echo("Done!")


if __name__ == "__main__":
    main()
