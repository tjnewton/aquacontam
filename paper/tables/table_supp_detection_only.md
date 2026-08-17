## Supplementary Table 17: Detection-Only Source Ablation

| Model                    |   All AUROC |   Filtered AUROC |   Delta AUROC |   N All |   N Filtered |   N Excluded |
|:-------------------------|------------:|-----------------:|--------------:|--------:|-------------:|-------------:|
| dummy_classifier         |       0.5   |            0.5   |         0     |    1935 |         1935 |            0 |
| logistic_regression      |       0.501 |            0.501 |         0     |    1935 |         1935 |            0 |
| xgboost_classifier       |       0.838 |            0.714 |        -0.124 |    2322 |         2066 |          256 |
| xgboost_default          |       0.682 |            0.682 |         0     |    1935 |         1935 |            0 |
| random_forest_classifier |       0.712 |            0.712 |         0     |    1935 |         1935 |            0 |
| mlp_classifier           |       0.46  |            0.46  |         0     |    1935 |         1935 |            0 |
| cnn1d_classifier         |       0.614 |            0.614 |         0     |    1935 |         1935 |            0 |
| lightgbm_classifier      |       0.7   |            0.7   |         0     |    1935 |         1935 |            0 |
| catboost_classifier      |       0.689 |            0.689 |         0     |    1935 |         1935 |            0 |
| gnn_gcn_classifier       |       0.473 |            0.473 |         0     |    1935 |         1935 |            0 |
| gnn_sage_classifier      |       0.447 |            0.447 |         0     |    1935 |         1935 |            0 |
| deep_tobit_classifier    |       0.449 |            0.449 |         0     |    1935 |         1935 |            0 |
| tabpfn_classifier        |       0.709 |            0.709 |         0     |    1935 |         1935 |            0 |
| voting_ensemble          |       0.712 |            0.712 |         0     |    1935 |         1935 |            0 |
| stacking_ensemble        |       0.711 |            0.711 |         0     |    1935 |         1935 |            0 |
| icp_classifier           |       0.467 |            0.467 |         0     |    1935 |         1935 |            0 |
