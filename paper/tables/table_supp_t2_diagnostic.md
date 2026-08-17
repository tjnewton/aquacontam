## Supplementary Table 22: T2 Regression Diagnostics

| Model                   |   Overall R² |   Overall RMSE | Detected-only R²   |   Concordance Index |
|:------------------------|-------------:|---------------:|:-------------------|--------------------:|
| xgboost_regressor       |       -0.001 |         0.1579 | —                  |               0.639 |
| random_forest_regressor |        0.006 |         0.1574 | —                  |               0.724 |
| mlp_regressor           |        0.045 |         0.1542 | —                  |               0.526 |
| cnn1d_regressor         |       -0.007 |         0.1584 | —                  |               0.669 |
| lightgbm_regressor      |        0.001 |         0.1578 | —                  |               0.482 |
| catboost_regressor      |       -0.001 |         0.1579 | —                  |               0.458 |
| deep_tobit_regressor    |     -149.013 |         1.9332 | —                  |               0.652 |
| hurdle_regressor        |        0     |         0.1578 | —                  |               0.288 |
| xgboost_aft_regressor   |        0.003 |         0.1576 | —                  |               0.687 |
| zi_tobit_regressor      |       -0.765 |         0.2097 | —                  |               0.715 |
| icp_regressor           |      -64.438 |         1.2768 | —                  |               0.396 |
| hurdle_aft_ensemble     |        0.002 |         0.1577 | —                  |               0.525 |
