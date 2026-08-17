## Table S: Sensitivity Analysis

### Drop-NA Threshold

|   Threshold |   Features |   AUROC |   AUPRC |
|------------:|-----------:|--------:|--------:|
|         0.3 |        135 |   0.858 |  0.6882 |
|         0.4 |        135 |   0.858 |  0.6882 |
|         0.5 |        135 |   0.858 |  0.6882 |
|         0.6 |        135 |   0.858 |  0.6882 |
|         0.7 |        135 |   0.858 |  0.6882 |

### Buffer Radius Variants

| Variant   |   Proximity Cols |   AUROC |   AUPRC |
|:----------|-----------------:|--------:|--------:|
| all_radii |               19 |  0.858  |  0.6882 |
| 1km_only  |               19 |  0.8652 |  0.7043 |
| 5km_only  |               19 |  0.8638 |  0.7003 |
| 10km_only |               19 |  0.8537 |  0.6773 |
| dist_only |               13 |  0.8571 |  0.6823 |
