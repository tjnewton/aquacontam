# AquaContam Benchmark Leaderboard

*Last updated: 2026-08-05 — regenerated from the paper's frozen archive (results/paper_frozen)*

**28 baseline models**, regenerated from the paper's frozen archive (`results/paper_frozen/`)

## T1 PFAS-detection leaderboard (primary protocol: leave-one-region-out)

Ranked by **leave-one-region-out (LORO) mean AUROC** across the 10 EPA regions — the
leakage-resistant protocol this benchmark is built to reward. The fixed geographic-split
AUROC (West-only test: EPA Regions 8/9/10) is shown alongside; its i.i.d. bootstrap CI is
anti-conservative under residual spatial autocorrelation (Moran's I up to ~0.67), which is
why LORO, not the single split, is primary. Tree ensembles are the strong baseline to beat;
the top families are statistically comparable (the leaders differ by well under one AUROC
point). Non-converged families are flagged, never shown as a metric.

| Rank | Model | LORO AUROC (mean, 95% CI) | Fixed-split AUROC | LORO folds |
|------|-------|---------------------------|-------------------|------------|
| 1 | catboost_classifier | 0.790 [0.717-0.862] | 0.845 | 10/10 |
| 2 | voting_ensemble | 0.784 [0.707-0.861] | 0.850 | 10/10 |
| 3 | xgboost_classifier | 0.782 [0.706-0.859] | 0.864 | 10/10 |
| 4 | random_forest_classifier | 0.771 [0.693-0.848] | 0.850 | 10/10 |
| 5 | lightgbm_classifier | 0.766 [0.684-0.848] | 0.847 | 10/10 |
| 6 | tabpfn_classifier | 0.758 [0.678-0.838] | 0.856 | 10/10 |
| 7 | icp_classifier | 0.713 [0.623-0.804] | 0.731 | 10/10 |
| 8 | stacking_ensemble | 0.711 [0.585-0.837] | 0.849 | 10/10 |
| 9 | deep_tobit_classifier | 0.707 [0.611-0.802] | 0.602 | 10/10 |
| 10 | logistic_regression | 0.705 [0.607-0.802] | 0.753 | 10/10 |
| 11 | mlp_classifier | 0.704 [0.608-0.800] | 0.716 | 10/10 |
| 12 | gnn_gcn_classifier | 0.683 [0.594-0.772] | 0.642 | 10/10 |
| 13 | cnn1d_classifier | 0.670 [0.561-0.779] | 0.787 | 10/10 |
| 14 | gnn_sage_classifier | 0.652 [0.559-0.745] | 0.482 | 10/10 |

## Submit a model

The benchmark's primary evaluation is the leave-one-region-out (LORO) protocol above.
To add your model to the T1 leaderboard against the same leakage-resistant evaluation:

1. Implement your model against the `BaseModel` interface (`src/aquacontam/models/base.py`).
2. Run it through the committed LORO harness — the same code path that produced this
   table (`paper/compute_loro_full.py`, which calls `_run_loro_cv` over the 10 EPA-region
   folds). Do not re-tune on the held-out region.
3. Report the **LORO mean AUROC and its t(9) cluster-robust 95% CI**, plus the fixed
   geographic-split AUROC for reference, and your per-fold AUROCs.
4. Open a pull request adding your result. Include a reproducible config and a fixed seed
   so the harness regenerates your numbers.

Submissions that report only the single fixed split, or that re-tune on the test region,
are not comparable and will be marked as such.

## Multi-task rankings (context)

The per-task tables below are fixed-split reference results across all seven tasks; only T1 carries a converged LORO ranking (above) and T4 a three-model LORO check. T2/T3/T5/T7 are boundary or negative tasks.

### Overall Rankings

| Rank | Model | Entries | Mean Rank |
|------|-------|---------|-----------|
| 1 | mlp_regressor | 1 | 1.00 |
| 2 | random_forest_regressor | 1 | 2.00 |
| 3 | xgboost_classifier | 5 | 2.40 |
| 4 | xgboost_aft_regressor | 1 | 3.00 |
| 5 | catboost_classifier | 5 | 3.20 |
| 6 | hurdle_aft_ensemble | 1 | 4.00 |
| 6 | tabpfn_classifier | 3 | 4.00 |
| 8 | voting_ensemble | 5 | 4.20 |
| 9 | lightgbm_regressor | 1 | 5.00 |
| 10 | random_forest_classifier | 5 | 5.60 |
| 11 | xgboost_default | 5 | 5.80 |
| 12 | hurdle_regressor | 1 | 6.00 |
| 12 | lightgbm_classifier | 5 | 6.00 |
| 14 | xgboost_regressor | 1 | 7.00 |
| 15 | stacking_ensemble | 5 | 7.40 |
| 16 | catboost_regressor | 1 | 8.00 |
| 17 | cnn1d_regressor | 1 | 9.00 |
| 18 | icp_classifier | 5 | 9.40 |
| 18 | mlp_classifier | 5 | 9.40 |
| 20 | logistic_regression | 5 | 9.60 |
| 21 | zi_tobit_regressor | 1 | 10.00 |
| 22 | gnn_gcn_classifier | 3 | 10.33 |
| 23 | deep_tobit_classifier | 5 | 11.00 |
| 23 | icp_regressor | 1 | 11.00 |
| 25 | cnn1d_classifier | 5 | 11.60 |
| 26 | gnn_sage_classifier | 3 | 11.67 |
| 27 | deep_tobit_regressor | 1 | 12.00 |
| 28 | dummy_classifier | 5 | 14.20 |

### Per-Task Rankings

**T1** (metric: auprc)

| Rank | Model | Analyte | Value |
|------|-------|---------|-------|
| 1 | xgboost_classifier | PFOS | 0.7810 |
| 2 | tabpfn_classifier | PFOS | 0.7672 |
| 3 | lightgbm_classifier | PFOS | 0.7628 |
| 4 | catboost_classifier | PFOS | 0.7582 |
| 5 | voting_ensemble | PFOS | 0.7548 |
| 6 | stacking_ensemble | PFOS | 0.7539 |
| 7 | random_forest_classifier | PFOS | 0.7529 |
| 8 | xgboost_default | PFOS | 0.7491 |
| 9 | logistic_regression | PFOS | 0.6742 |
| 10 | icp_classifier | PFOS | 0.6722 |
| 11 | cnn1d_classifier | PFOS | 0.5067 |
| 12 | mlp_classifier | PFOS | 0.4765 |
| 13 | gnn_gcn_classifier | PFOS | 0.2848 |
| 14 | deep_tobit_classifier | PFOS | 0.2508 |
| 15 | dummy_classifier | PFOS | 0.2158 |
| 16 | gnn_sage_classifier | PFOS | 0.1953 |

**T2** (metric: rmse)

| Rank | Model | Analyte | Value |
|------|-------|---------|-------|
| 1 | mlp_regressor | PFOA | 0.1542 |
| 2 | random_forest_regressor | PFOA | 0.1574 |
| 3 | xgboost_aft_regressor | PFOA | 0.1576 |
| 4 | hurdle_aft_ensemble | PFOA | 0.1577 |
| 5 | lightgbm_regressor | PFOA | 0.1578 |
| 6 | hurdle_regressor | PFOA | 0.1578 |
| 7 | xgboost_regressor | PFOA | 0.1579 |
| 8 | catboost_regressor | PFOA | 0.1579 |
| 9 | cnn1d_regressor | PFOA | 0.1584 |
| 10 | zi_tobit_regressor | PFOA | 0.2097 |
| 11 | icp_regressor | PFOA | 1.2768 |
| 12 | deep_tobit_regressor | PFOA | 1.9332 |

**T3** (metric: macro_auprc)

| Rank | Model | Analyte | Value |
|------|-------|---------|-------|
| 1 | catboost_classifier | multi-PFAS | 0.3842 |
| 2 | xgboost_classifier | multi-PFAS | 0.3786 |
| 3 | lightgbm_classifier | multi-PFAS | 0.3698 |
| 4 | xgboost_default | multi-PFAS | 0.3583 |
| 5 | random_forest_classifier | multi-PFAS | 0.3549 |
| 6 | gnn_gcn_classifier | multi-PFAS | non-converged |
| 6 | gnn_sage_classifier | multi-PFAS | non-converged |
| 8 | deep_tobit_classifier | multi-PFAS | non-converged |
| 8 | icp_classifier | multi-PFAS | non-converged |
| 8 | stacking_ensemble | multi-PFAS | non-converged |
| 8 | tabpfn_classifier | multi-PFAS | non-converged |
| 8 | voting_ensemble | multi-PFAS | non-converged |
| 13 | logistic_regression | multi-PFAS | 0.2521 |
| 14 | mlp_classifier | multi-PFAS | 0.2096 |
| 15 | cnn1d_classifier | multi-PFAS | 0.2069 |
| 16 | dummy_classifier | multi-PFAS | 0.1553 |

**T4** (metric: auprc)

| Rank | Model | Analyte | Value |
|------|-------|---------|-------|
| 1 | catboost_classifier | lead | 0.4031 |
| 2 | voting_ensemble | lead | 0.3989 |
| 3 | stacking_ensemble | lead | 0.3969 |
| 4 | xgboost_classifier | lead | 0.3867 |
| 5 | lightgbm_classifier | lead | 0.3831 |
| 6 | xgboost_default | lead | 0.3761 |
| 7 | random_forest_classifier | lead | 0.3680 |
| 8 | logistic_regression | lead | 0.3217 |
| 9 | cnn1d_classifier | lead | 0.3138 |
| 10 | mlp_classifier | lead | 0.2799 |
| 11 | icp_classifier | lead | 0.2701 |
| 12 | gnn_gcn_classifier | lead | 0.2454 |
| 13 | gnn_sage_classifier | lead | 0.2237 |
| 14 | deep_tobit_classifier | lead | 0.2079 |
| 15 | dummy_classifier | lead | 0.2052 |

**T5** (metric: auprc)

| Rank | Model | Analyte | Value |
|------|-------|---------|-------|
| 1 | voting_ensemble | transfer | 0.3404 |
| 2 | xgboost_classifier | transfer | 0.3347 |
| 3 | random_forest_classifier | transfer | 0.3264 |
| 4 | xgboost_default | transfer | 0.3141 |
| 5 | deep_tobit_classifier | transfer | 0.2748 |
| 6 | catboost_classifier | transfer | 0.2686 |
| 7 | stacking_ensemble | transfer | 0.2641 |
| 8 | logistic_regression | transfer | 0.2627 |
| 9 | icp_classifier | transfer | 0.2413 |
| 10 | mlp_classifier | transfer | 0.2356 |
| 11 | lightgbm_classifier | transfer | 0.2051 |
| 12 | cnn1d_classifier | transfer | 0.1495 |
| 13 | dummy_classifier | transfer | 0.1187 |

**T7** (metric: auprc)

| Rank | Model | Analyte | Value |
|------|-------|---------|-------|
| 1 | mlp_classifier | temporal | 0.2488 |
| 2 | tabpfn_classifier | temporal | 0.2440 |
| 3 | xgboost_classifier | temporal | 0.2404 |
| 4 | catboost_classifier | temporal | 0.2267 |
| 5 | voting_ensemble | temporal | 0.2193 |
| 6 | random_forest_classifier | temporal | 0.2146 |
| 7 | xgboost_default | temporal | 0.2143 |
| 8 | lightgbm_classifier | temporal | 0.1938 |
| 9 | icp_classifier | temporal | 0.1898 |
| 10 | logistic_regression | temporal | 0.1875 |
| 11 | cnn1d_classifier | temporal | 0.1697 |
| 12 | dummy_classifier | temporal | 0.1298 |
| 13 | stacking_ensemble | temporal | 0.1202 |
| 14 | deep_tobit_classifier | temporal | 0.1130 |


## Citation

If you use this benchmark, please cite:

```bibtex
@article{newton2026aquacontam,
  title={AquaContam: machine-learning models of drinking-water contamination learn who is monitored as much as where contamination occurs},
  author={Newton, Tyler J.},
  journal={Nature Water},
  year={2026}
}
```
