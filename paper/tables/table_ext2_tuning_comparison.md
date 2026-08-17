## Extended Data Table 2: Hyperparameter Tuning Comparison

| Model         |   Configs |   Default AUROC |   Tuned AUROC |   Default AUPRC |   Tuned AUPRC |   ΔAUPRC |
|:--------------|----------:|----------------:|--------------:|----------------:|--------------:|---------:|
| XGBoost       |        18 |           0.828 |         0.831 |           0.727 |         0.725 |   -0.002 |
| Random Forest |        12 |           0.815 |         0.815 |           0.711 |         0.711 |    0     |
| lightgbm      |        18 |           0.819 |         0.826 |           0.714 |         0.732 |    0.018 |
| catboost      |        18 |           0.786 |         0.807 |           0.68  |         0.691 |    0.01  |
| MLP           |        18 |           0.703 |         0.713 |           0.601 |         0.568 |   -0.034 |
| CNN1D         |        18 |           0.794 |         0.775 |           0.631 |         0.606 |   -0.025 |
