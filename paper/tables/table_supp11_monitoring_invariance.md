## Supplementary Table 13: Monitoring-Invariance Decomposition

| Approach                  | Task   |   AUROC | AUPRC   | Notes                                          |
|:--------------------------|:-------|--------:|:--------|:-----------------------------------------------|
| Full model (XGBoost)      | T1     |   0.864 | 0.781   | All features                                   |
| Full model (XGBoost)      | T4     |   0.699 | 0.387   | All features                                   |
| Provenance-free (XGBoost) | T1     |   0.785 | 0.698   | Provenance features dropped (environment-only) |
| Provenance-free (XGBoost) | T4     |   0.55  | 0.244   | Provenance features dropped (environment-only) |
| ICP                       | T1     |   0.731 | 0.672   | Adversarially invariant                        |
| ICP                       | T4     |   0.587 | 0.270   | Adversarially invariant                        |
| DML adjusted              | T1     |   0.729 | —       | Post-hoc adjusted association                  |
