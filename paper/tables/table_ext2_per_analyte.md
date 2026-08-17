| Model                    | Task   | Analyte   |   auroc |   auprc |    f1 |   precision |   recall |   macro_auprc |
|:-------------------------|:-------|:----------|--------:|--------:|------:|------------:|---------:|--------------:|
| dummy_classifier         | T1     | PFOS      |   0.5   |   0.216 | 0     |       0     |    0     |               |
| logistic_regression      | T1     | PFOS      |   0.753 |   0.674 | 0.576 |       0.551 |    0.603 |               |
| xgboost_classifier       | T1     | PFOS      |   0.864 |   0.781 | 0.682 |       0.981 |    0.523 |               |
| xgboost_default          | T1     | PFOS      |   0.836 |   0.749 | 0.67  |       0.869 |    0.545 |               |
| random_forest_classifier | T1     | PFOS      |   0.85  |   0.753 | 0.637 |       0.767 |    0.545 |               |
| mlp_classifier           | T1     | PFOS      |   0.716 |   0.476 | 0.525 |       0.458 |    0.615 |               |
| cnn1d_classifier         | T1     | PFOS      |   0.787 |   0.507 | 0.474 |       0.339 |    0.788 |               |
| lightgbm_classifier      | T1     | PFOS      |   0.847 |   0.763 | 0.625 |       0.581 |    0.677 |               |
| catboost_classifier      | T1     | PFOS      |   0.845 |   0.758 | 0.534 |       0.402 |    0.792 |               |
| gnn_gcn_classifier       | T1     | PFOS      |   0.642 |   0.285 | 0.486 |       0.405 |    0.609 |               |
| gnn_sage_classifier      | T1     | PFOS      |   0.482 |   0.195 | 0.12  |       0.108 |    0.134 |               |
| deep_tobit_classifier    | T1     | PFOS      |   0.602 |   0.251 | 0     |       0     |    0     |               |
| tabpfn_classifier        | T1     | PFOS      |   0.856 |   0.767 | 0.674 |       0.943 |    0.525 |               |
| voting_ensemble          | T1     | PFOS      |   0.85  |   0.755 | 0.654 |       0.801 |    0.553 |               |
| stacking_ensemble        | T1     | PFOS      |   0.849 |   0.754 | 0.549 |       0.418 |    0.8   |               |
| icp_classifier           | T1     | PFOS      |   0.731 |   0.672 | 0.634 |       0.802 |    0.525 |               |
| dummy_classifier         | T3     | all       |         |         |       |             |          |         0.155 |
| logistic_regression      | T3     | all       |         |         |       |             |          |         0.252 |
| xgboost_classifier       | T3     | all       |         |         |       |             |          |         0.379 |
| xgboost_default          | T3     | all       |         |         |       |             |          |         0.358 |
| random_forest_classifier | T3     | all       |         |         |       |             |          |         0.355 |
| mlp_classifier           | T3     | all       |         |         |       |             |          |         0.21  |
| cnn1d_classifier         | T3     | all       |         |         |       |             |          |         0.207 |
| lightgbm_classifier      | T3     | all       |         |         |       |             |          |         0.37  |
| catboost_classifier      | T3     | all       |         |         |       |             |          |         0.384 |
| gnn_gcn_classifier       | T3     | all       |         |         |       |             |          |         0.345 |
| gnn_sage_classifier      | T3     | all       |         |         |       |             |          |         0.345 |
| deep_tobit_classifier    | T3     | all       |         |         |       |             |          |         0.26  |
| tabpfn_classifier        | T3     | all       |         |         |       |             |          |         0.26  |
| voting_ensemble          | T3     | all       |         |         |       |             |          |         0.26  |
| stacking_ensemble        | T3     | all       |         |         |       |             |          |         0.26  |
| icp_classifier           | T3     | all       |         |         |       |             |          |         0.26  |
| dummy_classifier         | T4     | lead      |   0.5   |   0.205 | 0     |       0     |    0     |               |
| logistic_regression      | T4     | lead      |   0.646 |   0.322 | 0.399 |       0.297 |    0.611 |               |
| xgboost_classifier       | T4     | lead      |   0.699 |   0.387 | 0.42  |       0.413 |    0.427 |               |
| xgboost_default          | T4     | lead      |   0.683 |   0.376 | 0.41  |       0.387 |    0.437 |               |
| random_forest_classifier | T4     | lead      |   0.689 |   0.368 | 0.387 |       0.43  |    0.351 |               |
| mlp_classifier           | T4     | lead      |   0.63  |   0.28  | 0.389 |       0.288 |    0.602 |               |
| cnn1d_classifier         | T4     | lead      |   0.641 |   0.314 | 0.35  |       0.341 |    0.359 |               |
| lightgbm_classifier      | T4     | lead      |   0.692 |   0.383 | 0     |       0     |    0     |               |
| catboost_classifier      | T4     | lead      |   0.695 |   0.403 | 0.428 |       0.342 |    0.573 |               |
| gnn_gcn_classifier       | T4     | lead      |   0.572 |   0.245 | 0.341 |       0.245 |    0.561 |               |
| gnn_sage_classifier      | T4     | lead      |   0.521 |   0.224 | 0.285 |       0.237 |    0.358 |               |
| deep_tobit_classifier    | T4     | lead      |   0.505 |   0.208 | 0.34  |       0.205 |    1     |               |
| voting_ensemble          | T4     | lead      |   0.704 |   0.399 | 0.44  |       0.381 |    0.52  |               |
| stacking_ensemble        | T4     | lead      |   0.696 |   0.397 | 0.417 |       0.306 |    0.656 |               |
| icp_classifier           | T4     | lead      |   0.587 |   0.27  | 0.205 |       0.315 |    0.152 |               |
| dummy_classifier         | T5     | all       |   0.5   |   0.119 | 0     |       0     |    0     |               |
| logistic_regression      | T5     | all       |   0.696 |   0.263 | 0.294 |       0.274 |    0.318 |               |
| xgboost_classifier       | T5     | all       |   0.715 |   0.335 | 0.306 |       0.433 |    0.236 |               |
| xgboost_default          | T5     | all       |   0.702 |   0.314 | 0.311 |       0.441 |    0.24  |               |
| random_forest_classifier | T5     | all       |   0.723 |   0.326 | 0.282 |       0.426 |    0.21  |               |
| mlp_classifier           | T5     | all       |   0.7   |   0.236 | 0.349 |       0.259 |    0.536 |               |
| cnn1d_classifier         | T5     | all       |   0.607 |   0.149 | 0.11  |       0.142 |    0.09  |               |
| lightgbm_classifier      | T5     | all       |   0.592 |   0.205 | 0     |       0     |    0     |               |
| catboost_classifier      | T5     | all       |   0.67  |   0.269 | 0.353 |       0.343 |    0.365 |               |
| deep_tobit_classifier    | T5     | all       |   0.729 |   0.275 | 0.212 |       0.119 |    0.991 |               |
| voting_ensemble          | T5     | all       |   0.721 |   0.34  | 0.327 |       0.414 |    0.27  |               |
| stacking_ensemble        | T5     | all       |   0.695 |   0.264 | 0.344 |       0.35  |    0.339 |               |
| icp_classifier           | T5     | all       |   0.647 |   0.241 | 0.202 |       0.351 |    0.142 |               |
| dummy_classifier         | T7     | PFOS      |   0.5   |   0.13  | 0     |       0     |    0     |               |
| logistic_regression      | T7     | PFOS      |   0.58  |   0.188 | 0.22  |       0.225 |    0.215 |               |
| xgboost_classifier       | T7     | PFOS      |   0.664 |   0.24  | 0     |       0     |    0     |               |
| xgboost_default          | T7     | PFOS      |   0.625 |   0.214 | 0.038 |       0.531 |    0.019 |               |
| random_forest_classifier | T7     | PFOS      |   0.628 |   0.215 | 0.068 |       0.434 |    0.037 |               |
| mlp_classifier           | T7     | PFOS      |   0.627 |   0.249 | 0.287 |       0.246 |    0.344 |               |
| cnn1d_classifier         | T7     | PFOS      |   0.582 |   0.17  | 0.242 |       0.143 |    0.781 |               |
| lightgbm_classifier      | T7     | PFOS      |   0.568 |   0.194 | 0.06  |       0.462 |    0.032 |               |
| catboost_classifier      | T7     | PFOS      |   0.633 |   0.227 | 0.208 |       0.294 |    0.161 |               |
| deep_tobit_classifier    | T7     | PFOS      |   0.463 |   0.113 | 0.084 |       0.07  |    0.104 |               |
| tabpfn_classifier        | T7     | PFOS      |   0.646 |   0.244 | 0     |       0     |    0     |               |
| voting_ensemble          | T7     | PFOS      |   0.633 |   0.219 | 0.047 |       0.532 |    0.025 |               |
| stacking_ensemble        | T7     | PFOS      |   0.418 |   0.12  | 0.207 |       0.119 |    0.808 |               |
| icp_classifier           | T7     | PFOS      |   0.577 |   0.19  | 0.047 |       0.569 |    0.025 |               |