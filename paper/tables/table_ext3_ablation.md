## Extended Data Table 3: Feature Category Ablation Results

| Task   | Category               |   N Features |   Baseline AUPRC |   Ablated AUPRC |   ΔAUPRC | p-value   | p (FDR)   |
|:-------|:-----------------------|-------------:|-----------------:|----------------:|---------:|:----------|:----------|
| T1     | all_features           |            0 |            0.665 |           0.665 |   0      | —         | —         |
| T1     | proximity              |           16 |            0.665 |           0.653 |  -0.0119 | 0.012     | 0.024     |
| T1     | land_use               |           28 |            0.665 |           0.667 |   0.0021 | 0.696     | 0.696     |
| T1     | hydrogeology           |           45 |            0.665 |           0.67  |   0.0042 | 0.368     | 0.421     |
| T1     | aquifer_type_only      |           43 |            0.665 |           0.656 |  -0.0098 | 0.038     | 0.061     |
| T1     | demographics           |            5 |            0.665 |           0.671 |   0.006  | 0.128     | 0.171     |
| T1     | n_samples_only         |            1 |            0.665 |           0.633 |  -0.0325 | <0.001    | <0.001    |
| T1     | monitoring_intensity   |            2 |            0.665 |           0.598 |  -0.0673 | <0.001    | <0.001    |
| T1     | system_characteristics |            4 |            0.665 |           0.597 |  -0.0681 | <0.001    | <0.001    |
| T4     | all_features           |            0 |            0.421 |           0.421 |   0      | —         | —         |
| T4     | proximity              |           16 |            0.421 |           0.42  |  -0.0012 | 0.104     | 0.119     |
| T4     | land_use               |           28 |            0.421 |           0.418 |  -0.0031 | <0.001    | <0.001    |
| T4     | hydrogeology           |           55 |            0.421 |           0.418 |  -0.0036 | <0.001    | <0.001    |
| T4     | aquifer_type_only      |           53 |            0.421 |           0.419 |  -0.0022 | <0.001    | <0.001    |
| T4     | demographics           |            5 |            0.421 |           0.422 |   0.001  | 0.444     | 0.444     |
| T4     | n_samples_only         |            1 |            0.421 |           0.302 |  -0.1194 | <0.001    | <0.001    |
| T4     | monitoring_intensity   |            1 |            0.421 |           0.302 |  -0.1194 | <0.001    | <0.001    |
| T4     | system_characteristics |            3 |            0.421 |           0.303 |  -0.1185 | <0.001    | <0.001    |
