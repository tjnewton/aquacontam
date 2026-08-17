## Extended Data Table 10: Causal Adjusted Associations

| Feature                                                        |   SHAP Importance |   Causal Effect |   Std Error |   p-value |   p-value (FDR) | DML Rank   | SHAP Rank   |
|:---------------------------------------------------------------|------------------:|----------------:|------------:|----------:|----------------:|:-----------|:------------|
| aquifer_type_California Coastal Basin aquifers                 |            0      |          0.0469 |      0.0034 |  0        |        0        | —          | —           |
| aquifer_type_Northern Atlantic Coastal Plain aquifer system    |            0.0128 |         -0.0441 |      0.0037 |  0        |        0        | 1          | 19          |
| aquifer_type_Basin and Range basin-fill aquifers               |            0      |         -0.0391 |      0.0031 |  0        |        0        | —          | —           |
| aquifer_type_Biscayne aquifer                                  |            0.043  |          0.0342 |      0.0033 |  0        |        0        | 2          | 4           |
| aquifer_type_Early Mesozoic basin aquifers                     |            0.0188 |          0.0337 |      0.0036 |  0        |        0        | 3          | 7           |
| n_landfill_10000m                                              |            0.0085 |          0.0312 |      0.0036 |  0        |        0        | 4          | 42          |
| prox_hazardous_waste                                           |            0.0072 |          0.0307 |      0.003  |  0        |        0        | 5          | 55          |
| n_industrial_10000m                                            |            0.0086 |          0.0284 |      0.0041 |  0        |        1e-10    | 6          | 38          |
| pct_developed_1km                                              |            0.0063 |          0.0239 |      0.0029 |  0        |        0        | 7          | 70          |
| aquifer_type_Piedmont and Blue Ridge crystalline-rock aquifers |            0.0214 |          0.0237 |      0.0036 |  0        |        2e-10    | 8          | 6           |
| nlcd_majority_class_developed                                  |            0      |          0.0236 |      0.0031 |  0        |        0        | —          | —           |
| aquifer_type_Central Valley aquifer system                     |            0      |          0.0233 |      0.0031 |  0        |        0        | —          | —           |
| nlcd_majority_class_agriculture                                |            0.0053 |         -0.0232 |      0.0024 |  0        |        0        | 9          | 78          |
| aquifer_type_Floridan aquifer system                           |            0.0183 |          0.0224 |      0.0037 |  1e-09    |        5.3e-09  | 10         | 8           |
| n_industrial_5000m                                             |            0.007  |          0.0218 |      0.0049 |  9.13e-06 |        2.89e-05 | 11         | 60          |
Original AUROC: 0.835, Adjusted AUROC: 0.729
Spearman rho = 0.24 across 80 features
