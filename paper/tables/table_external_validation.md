## External Validation and Domain Shift Analysis

| State DB      | EPA Region   | Split Role   | n Systems   | AUROC [95% CI]      | AUPRC   | Detection Rate   |
|:--------------|:-------------|:-------------|:------------|:--------------------|:--------|:-----------------|
| NJ DEP        | R2           | val          | 1310        | 0.789 [0.765-0.812] | 0.792   | 0.473            |
| MO DNR        | R7           | val          | 1208        | 0.829 [0.767-0.885] | 0.281   | 0.052            |
| CA GeoTracker | R9           | test         | 1421        | 0.813 [0.789-0.838] | 0.707   | 0.297            |
| MI MPART      | R5           | train        | 153         | —                   | —       | 1.000            |
| OH EPA        | R5           | train        | —           | —                   | —       | —                |
| WA DOH        | R10          | test         | 256         | —                   | —       | 1.000            |
| NC DEQ        | R4           | train        | —           | —                   | —       | —                |

*NJ DEP: DeLong test vs. chance (AUROC = 0.5): z = 22.85, p = 0.000.* *MO DNR: DeLong test vs. chance (AUROC = 0.5): z = 10.92, p = 0.000.* *CA GeoTracker: DeLong test vs. chance (AUROC = 0.5): z = 24.22, p = 0.000.*
