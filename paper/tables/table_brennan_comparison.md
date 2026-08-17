## Split Comparison: Feature Set × Split Strategy

| Model              | Features   | Split      | AUROC               |   AUPRC | DeLong p               |
|:-------------------|:-----------|:-----------|:--------------------|--------:|:-----------------------|
| XGBoost            | Fernandez  | Geographic | 0.699 [0.680-0.718] |   0.499 | —                      |
| XGBoost            | Fernandez  | Random     | 0.860 ± 0.011       |   0.722 | —                      |
| RandomForest       | Fernandez  | Geographic | 0.700 [0.682-0.719] |   0.532 | —                      |
| RandomForest       | Fernandez  | Random     | 0.862 ± 0.011       |   0.711 | —                      |
| LogisticRegression | Fernandez  | Geographic | 0.629 [0.607-0.648] |   0.451 | —                      |
| LogisticRegression | Fernandez  | Random     | 0.705 ± 0.013       |   0.427 | —                      |
| XGBoost            | Baseline   | Geographic | 0.699 [0.680-0.719] |   0.532 | —                      |
| XGBoost            | Baseline   | Random     | 0.869 ± 0.009       |   0.744 | —                      |
| RandomForest       | Baseline   | Geographic | 0.719 [0.702-0.738] |   0.551 | —                      |
| RandomForest       | Baseline   | Random     | 0.869 ± 0.010       |   0.718 | —                      |
| LogisticRegression | Baseline   | Geographic | 0.736 [0.717-0.756] |   0.559 | —                      |
| LogisticRegression | Baseline   | Random     | 0.749 ± 0.012       |   0.47  | —                      |
| XGBoost            | Full       | Geographic | 0.790 [0.774-0.806] |   0.672 | <0.001 (ΔAUROC +0.091) |
| XGBoost            | Full       | Random     | 0.901 ± 0.010       |   0.805 | —                      |
| RandomForest       | Full       | Geographic | 0.805 [0.788-0.820] |   0.688 | <0.001 (ΔAUROC +0.086) |
| RandomForest       | Full       | Random     | 0.889 ± 0.012       |   0.773 | —                      |
| LogisticRegression | Full       | Geographic | 0.654 [0.635-0.674] |   0.457 | <0.001 (ΔAUROC -0.082) |
| LogisticRegression | Full       | Random     | 0.873 ± 0.009       |   0.723 | —                      |

†DeLong p: paired test of Full-feature vs Baseline-feature AUROC on the shared geographic test set (reported on the geographic/Full rows); ΔAUROC = Full − Baseline, so a negative value means the Full feature set performs worse.
