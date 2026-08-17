## Supplementary Table 23: Validation Set Reuse Bias Decomposition

### Experimental Conditions

| Condition    | Description                                                        |   N train |   N val |   Mean AUROC |   SD AUROC |
|:-------------|:-------------------------------------------------------------------|----------:|--------:|-------------:|-----------:|
| Triple Use   | Train R1,3,4,5,6; val R2,7 for ES; test R8,9,10                    |      9087 |    3414 |        0.809 |      0.003 |
| Internal Val | Train ~85% of R1,3,4,5,6; internal 15% val for ES; R2,7 excluded   |      7724 |    1363 |        0.802 |      0.007 |
| No Es        | Train R1,3,4,5,6; no early stopping (all 500 rounds); test R8,9,10 |      9087 |       0 |        0.787 |      0.005 |


### Decomposition

| Component                       | ΔAUROC   | Description                                                                                                                                    |
|:--------------------------------|:---------|:-----------------------------------------------------------------------------------------------------------------------------------------------|
| Val Reuse Effect                | +0.0223  | AUROC difference: ES on external val regions vs no ES (same training data, same test). Isolates total effect of using R2,7 for early stopping. |
| Val Identity Effect             | +0.0073  | AUROC difference: ES on external R2,7 vs ES on internal val (~15% training holdout). Positive = external val helps more.                       |
| Es Benefit                      | +0.0150  | AUROC difference: ES on internal val vs no ES. Early stopping benefit independent of val source.                                               |
| % of holdout-LORO gap explained | 27.5%    | Gap = 0.0811                                                                                                                                   |

