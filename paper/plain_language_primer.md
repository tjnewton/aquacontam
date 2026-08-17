# Understanding This Paper: A Guide for Water Scientists

*A plain-language companion for readers without a machine-learning background.*

## What the paper shows

Machine-learning models increasingly predict where drinking-water contamination such as PFAS or lead
will occur, guiding monitoring and regulation. But they are trained on administrative monitoring
records that capture who gets sampled, how often, and at what detection limit as much as where
contamination occurs. Standard evaluations therefore overstate skill: random splits let neighboring
systems fall in both training and test sets, and the models read fingerprints of the monitoring
program itself. Held out honestly by whole region, with monitoring inputs removed, the real
environmental signal is modest and uneven, and the same confound distorts the equity conclusions
drawn from these data. Analogy: predicting where fish are from a fleet's catch logs teaches the
fleet's habits, not the fish.

## Key terms, translated

- **Feature:** an input variable (industry distance, % wetland, aquifer type).
- **Train/validation/test split:** fit, tune, and judge on three separate sets.
- **Geographic holdout:** test on whole held-out regions, not a random mix.
- **Spatial leakage:** skill inflated when neighbors fall on both sides of a split.
- **LORO:** rotate testing across all 10 EPA regions; +/- is the regional spread.
- **AUROC:** ranking score for yes/no calls; 0.5 = coin flip, 1.0 = perfect.
- **AUPRC:** like AUROC but for rare detections; chance = the detection rate.
- **Gradient-boosted trees (XGBoost, Random Forest):** best tabular models here.
- **Provenance / ascertainment:** skill from the monitoring process, not nature.
- **SHAP / feature importance:** how much each input drives predictions.
- **Double machine learning (DML):** an input's link with other inputs removed.
- **Monitoring-invariant model (ICP):** ignores monitoring fingerprints by design.
- **Censored / non-detect:** below-limit values; Tobit/hurdle models keep them.
- **Burden ratio:** contamination in one group vs. a reference group.

## Reading the headline numbers

- **Lead, AUROC 0.962 → <!--pn:t4_pf_auroc-->0.550<!--/pn-->** (monitoring inputs removed): the model mostly tracked who gets sampled, not the environment.
- **PFAS, AUPRC <!--pn:leak_random_auprc-->0.744<!--/pn--> → <!--pn:leak_geo_auprc-->0.532<!--/pn-->** (random → geographic split): honest evaluation is far below prior random-split numbers.
- **PFAS, out-of-region AUROC <!--pn:loro_full_auroc-->0.782<!--/pn--> → <!--pn:loro_pf_auroc-->0.691<!--/pn-->** (monitoring inputs removed): the transportable environmental signal is modest and uneven.
- **<!--pn:mon_ratio_poc-->1.85<!--/pn-->×:** communities of color are monitored far more intensively than others, so monitoring itself is inequitably allocated.

*Full methods, numbers, and citations are in the manuscript.*
