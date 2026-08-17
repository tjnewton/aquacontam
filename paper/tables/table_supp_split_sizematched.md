## Split Comparison with Size-Matched Random Control

| Model              | Features   | Split                 |   n train | AUROC         |   AUPRC |
|:-------------------|:-----------|:----------------------|----------:|:--------------|--------:|
| XGBoost            | Fernandez  | Geographic            |     7,930 | 0.712         |   0.563 |
| XGBoost            | Fernandez  | Random                |    11,524 | 0.857 ± 0.009 |   0.725 |
| XGBoost            | Fernandez  | Random (size-matched) |     7,930 | 0.848 ± 0.016 |   0.708 |
| RandomForest       | Fernandez  | Geographic            |     7,930 | 0.728         |   0.6   |
| RandomForest       | Fernandez  | Random                |    11,524 | 0.854 ± 0.008 |   0.709 |
| RandomForest       | Fernandez  | Random (size-matched) |     7,930 | 0.848 ± 0.009 |   0.698 |
| LogisticRegression | Fernandez  | Geographic            |     7,930 | 0.587         |   0.374 |
| LogisticRegression | Fernandez  | Random                |    11,524 | 0.683 ± 0.011 |   0.394 |
| LogisticRegression | Fernandez  | Random (size-matched) |     7,930 | 0.681 ± 0.011 |   0.393 |
| XGBoost            | Baseline   | Geographic            |     7,930 | 0.684         |   0.567 |
| XGBoost            | Baseline   | Random                |    11,524 | 0.863 ± 0.011 |   0.74  |
| XGBoost            | Baseline   | Random (size-matched) |     7,930 | 0.856 ± 0.010 |   0.72  |
| RandomForest       | Baseline   | Geographic            |     7,930 | 0.751         |   0.622 |
| RandomForest       | Baseline   | Random                |    11,524 | 0.859 ± 0.010 |   0.715 |
| RandomForest       | Baseline   | Random (size-matched) |     7,930 | 0.853 ± 0.009 |   0.7   |
| LogisticRegression | Baseline   | Geographic            |     7,930 | 0.663         |   0.42  |
| LogisticRegression | Baseline   | Random                |    11,524 | 0.739 ± 0.010 |   0.45  |
| LogisticRegression | Baseline   | Random (size-matched) |     7,930 | 0.737 ± 0.010 |   0.448 |
| XGBoost            | Full       | Geographic            |     7,930 | 0.783         |   0.71  |
| XGBoost            | Full       | Random                |    11,524 | 0.895 ± 0.010 |   0.802 |
| XGBoost            | Full       | Random (size-matched) |     7,930 | 0.887 ± 0.009 |   0.79  |
| RandomForest       | Full       | Geographic            |     7,930 | 0.839         |   0.744 |
| RandomForest       | Full       | Random                |    11,524 | 0.883 ± 0.011 |   0.77  |
| RandomForest       | Full       | Random (size-matched) |     7,930 | 0.877 ± 0.010 |   0.762 |
| LogisticRegression | Full       | Geographic            |     7,930 | 0.690         |   0.527 |
| LogisticRegression | Full       | Random                |    11,524 | 0.870 ± 0.009 |   0.726 |
| LogisticRegression | Full       | Random (size-matched) |     7,930 | 0.867 ± 0.009 |   0.723 |

Random (size-matched): random 5-fold CV with each fold's training set stratified-subsampled to the geographic training-set size.
