# RESPONSE_BANK — point-by-point responses to R5 (Nature Water referee, 2026-07-05)

Every R5 finding writes its response paragraph here, keyed by finding id. R5 closely
simulates the journal's real referees, so this file is the rehearsal for the actual
response letter. Entries are drafted as their disposition lands (compute + prose) and
finalized at Phase 5. "Frozen artifact" refers to `results/paper_frozen/`; every number
below is marker-gated (G1) and, where computed this pass, reproduction-gated.

---

## Major points

### M1 — abstract's people-of-color underperformance claim (not significant; selectively framed)
We agree and have made the abstract's framing statistically honest. We computed
two-proportion tests and Wilson CIs for every demographic group gap
(`group_gap_tests.json`, C10). The people-of-color false-negative-rate gap is 0.062
(0.50 vs 0.44) with **z = 1.41, p = 0.16 — not significant**; the companion group-AUROC
gap is 0.050 (**p = 0.060**). On the same split, high-people-of-color systems in fact have
*higher* AUROC (0.799 vs 0.750) and *better* conformal coverage (0.919 vs 0.886, gap
p = 0.007 in their favour); the false-negative gap is essentially zero for education
(0.002) and *reversed* for low income (−0.077). The abstract and Discussion now state that
these group gaps are directionally present but not individually significant, and reframe the
deployment caution around single-threshold calibration under unequal base rates rather than
a discrimination deficit. [prose: Phase 3]

### M2 — the benchmark is over-stated; only T1 converged; T3 has degenerate cells
We reframed the benchmark honestly and hardened the tables against silent non-convergence.
The abstract and Discussion now describe a reproducible two-confound protocol with one
converged detection task (T1) plus documented negative (T2) and boundary (T3/T4/T5/T7) tasks,
not a populated multi-task leaderboard. Table 2 is now the LORO-ranked T1 leaderboard with the
primary metric stated per task. The T3 degeneracy the referee found is real: macro-AUROC is NaN
for every T3 model and two groups of models emit byte-identical micro metrics (five at
0.7533/0.3211; the two graph nets at 0.7528/0.4000), a silent fallback. We built a shared
degeneracy detector (`gate_lib.degenerate_metric_models`, unit-tested) and wired it into both
the Table 2 leaderboard and Supplementary Table 9, so those cells now render "non-converged"
rather than as plausible numbers. Finally, we dropped the ImageNet/GLUE equivalence: the
Discussion now contrasts AquaContam with general-purpose i.i.d. benchmarks (its point is that in
surveillance-generated data the evaluation protocol, not the model, decides most reported skill)
and cites GEO-Bench only as a domain-benchmark antecedent. [prose + generators: Phase 3]

### M3 — the leaderboard was a single West-only split; the LORO leader flips
We adopted the referee's first-choice remedy in full. Table 2 is re-ranked by leave-one-region-out
(LORO) mean AUROC across all 14 T1 families (`loro_cv_full.json`, the maintainer-funded full run),
with a t(9) cluster-robust CI as the primary column and the fixed West-only split (EPA Regions
8/9/10) AUROC shown alongside, its i.i.d. bootstrap CI captioned anti-conservative under the
documented residual spatial autocorrelation. The ranking flips: CatBoost leads under LORO (0.790)
while the fixed-split leader XGBoost (0.864 on the split) falls to third (LORO 0.782) — a direct
demonstration of why the single split misleads. The top tree/ensemble families are statistically
comparable (the benchmark is underpowered to separate the leading two), so we present the honest
protocol, not a single winner, and state that tree ensembles are the strong baseline to beat.
`LEADERBOARD.md` is now the canonical LORO-ranked T1 leaderboard with a "Submit a model" protocol
(run the same `compute_loro_full.py` / `_run_loro_cv` harness, report LORO mean + cluster CI, open
a PR) so third parties extend the benchmark against the same leakage-resistant evaluation.
[prose + generators + LEADERBOARD: Phase 3, R5-D3]

### m5 — the ICP separates weakly and no out-of-region ICP was reported
The full LORO run produces the out-of-region ICP number the referee asked for. The monitoring-
invariant ICP reaches LORO AUROC 0.713, close to its 0.731 in-region value, i.e. it transports
about as well as the provenance-reduced model's 0.691 upper bound. We now report the LORO-ICP
figure and soften the "stricter" language: the two honest estimators broadly agree rather than one
being decisively stricter than the other. [prose: Phase 3, from C18 loro_cv_full.json]

### M4 — robustness was reported on a different harness than the 0.864 headline
We ran the multi-seed check on the exact harness that produced the headline. Five seeds on
the benchmark task harness give a T1 AUROC mean of **0.864 ± 0.003**
(`headline_harness_stability.json`, C11; seed-42 reproduces the frozen 0.8636), so the
headline is seed-stable and the earlier `seed_inflation` `is_inflated` flag was a
**cross-harness artifact** (it compared the 0.864 headline against the 0.809 streamlined
harness), not evidence of a lucky seed. The main text now reconciles the several circulating
T1 figures as distinct harnesses and splits rather than conflicting estimates of one number:
0.864 (full-feature geographic holdout, the headline), 0.809 (streamlined single-model
robustness harness), 0.790 (full-feature split-comparison harness), and 0.782
(leave-one-region-out mean). [prose: Phase 3]

### M9 — the novelty framing over-claims for the venue
Adopted the honest reframe (R5-D1, pre-approved). The abstract and contribution list now lead with
the two defensible advances: the rigorous quantification and decomposition of a known ascertainment
confound in high-stakes US regulatory drinking-water ML (landing as the first enforceable PFAS MCLs
take effect), and the monitoring-inequity environmental-justice finding. We cite the presence-only /
sample-selection-bias antecedents (Phillips et al. 2009; Fithian et al. 2015) and the epidemiological
surveillance-bias literature, state explicitly that no new learning method is introduced, and describe
the benchmark honestly as a reproducible two-confound protocol with one converged detection task plus
documented negative and boundary tasks. The reframe is propagated to every pitch surface (cover
letter, plain-language primer, reporting summary, .zenodo, README), and a mechanical over-claim lexeme
sweep across the manuscript, SI, Extended Data, and pitch surfaces dispositioned every hit: the
"first/largest/genuinely novel/ImageNet-GLUE equivalence" claims are removed or recast as an
i.i.d.-contrast; the surviving "first enforceable PFAS MCLs" (a regulatory fact), the GEO-Bench
related-work citation, sequential ordinals, and descriptive superlatives ("the largest displacement")
are retained as legitimate. [prose + all pitch surfaces + sweep: Phase 3, R5-D1]

### M5 — the two most-cited out-of-region numbers rest on a 10-cluster block bootstrap
We agree the G = 10 region-block bootstrap is anti-conservative and adds little (frozen width
ratio block/naive = 1.002; the T4 "spatial" interval is actually *narrower* than naive, 0.845).
We computed t(G−1)- and cluster-bootstrap-calibrated intervals from the same ten per-region
values (`cluster_ci_calibration.json`, C12). The t(9) interval for the transportable AUROC is
**1.15× wider** than the block bootstrap (0.178 vs 0.154), but its lower bound (0.606) still
exceeds chance, so the "signal survives out of region" claim stands; the burden t(9) interval
[1.22, 2.04] still excludes 1.0. The manuscript now states G = 10, flags the anti-conservatism,
reads 0.691 as an upper-bound *indicative* estimate, and reconciles the "honest estimate" vs
"upper bound" wording (now uniformly "upper-bound estimate"). [prose: Phase 3, R5-D4]

### M7 — deployment/lift uses the detection target, not MCL exceedance
We agree the regulatorily-actionable target belongs in the main-text actionability analysis.
We computed top-decile lift for the MCL-exceedance target (`mcl_lift.json`, C14,
reproduction-gated to the frozen exceedance AUROC 0.719): the exceedance model concentrates
**2.68×** more exceedances in its top risk decile, versus 3.06× for detection — a real but
weaker prioritization signal, consistent with the 9-point AUROC gap (0.719 vs 0.807). The main
text now surfaces the MCL-exceedance AUROC/AUPRC and its top-decile lift, and softens the
"reasonable proxy" wording to match the measured gap. [prose: Phase 3]

---

## Minor points

### M6 — the T1 detection label is monitoring-defined (reporting-limit heterogeneity)
We added the promised main-text caveat and reported the AUPRC the referee asked for. The T1
label mixes reporting limits differing by an order of magnitude (UCMR5 quantifies PFOS to
~0.004 µg/L, UCMR3 to ~0.04 µg/L), so detection status depends partly on monitoring
sensitivity. The main-text task definition now states this explicitly and links it to the
paper's thesis. In the SI common-reporting-limit section we now report both metrics under
re-censoring to the common 0.04 µg/L limit: AUROC falls modestly (0.828 → 0.781, ~5 points)
but **AUPRC falls sharply, 0.727 → 0.102**, because re-censoring reclassifies **29,583** PFOS
detections and the positive class contracts. The ranking signal is robust to reporting-limit
harmonization; the label itself carries the monitoring dependence we document
(`common_rl_sensitivity.json`). [prose + SI markers: Phase 3]

### M8 — data availability: "publicly available" omits MN; NJ DEP basis; Zenodo license
We corrected the availability language, documented the legal bases, reconciled the license, and
implemented the fallback so submission is not blocked on the Minnesota reply. The manuscript,
Reporting Summary, and .zenodo description no longer call every source "publicly available":
all are public records, but Minnesota MDH is obtained by written request under the Minnesota
Government Data Practices Act and is not currently redistributable (permission pending, request
drafted in `paper/mdh_permission_request.md`), and New Jersey DEP is accessed via its public
WaterViewer portal (public records under the NJ Open Public Records Act; the scraper reads the
same public pages a browser would, with no access circumvention). The Reporting Summary gains an
MN MDH row. The Zenodo deposit license is corrected from MIT to CC BY 4.0 for the compiled data,
with the code remaining MIT (stated in Data and Code Availability and machine-checked by the
concept registry). Crucially, the thesis survives without Minnesota: the monitoring-confound
decomposition and the inequity analysis rest on the federal UCMR and SDWIS records, and Minnesota
serves only as a dense-monitoring corroboration, which the manuscript now states explicitly.
[prose + policy + human action: Phase 3, R5-D2]

### m12 — "fifteen sources" over-counts; count/license inconsistent across surfaces
Reconciled to one canonical rendering and gated it. The count is stated consistently as "fifteen
data sources (thirteen contributing)" in the manuscript, .zenodo.json (was "15+"), and the
CHANGELOG's latest release (was "12+"); the deposit license is CC BY 4.0 for data / MIT for code
everywhere. Two `CONCEPT_REGISTRY.tsv` rows (`data_source_count`, `dataset_deposit_license`) now
machine-check these across README, .zenodo, CHANGELOG (G9) and the manuscript (G10): the
conflicting-alias regexes for "15+/12+ sources" and a MIT-labeled data deposit fail the gate if any
surface regresses. The rows were added in the same commit as the prose fixes, so no surface carries
a conflict. [prose + G9/G10 rows: Phase 3]

### m3 — supplementary power table uses hardcoded AUROCs that do not match the frozen benchmark
Fixed at the root. The power-summary loader read a `delong_results.json` that never existed and
silently fell back to hardcoded AUROCs (0.873/0.857). It now derives the top-two T1 AUROCs
(0.864/0.856) directly from the committed `results.json` and **fails loud** if that source is
absent (with a test); the generator likewise raises rather than emitting a placeholder. The
regenerated `power_analysis.json` gives effect size **0.007** and power **0.11** — i.e. the
benchmark is underpowered to distinguish the top two T1 models, which we now state and connect
to the leave-one-region-out re-ranking (the leaderboard ordering among the leading models is
within sampling noise). [C16; SI S19 updated]

### m4 — "provenance-free" is a misnomer (source remains recoverable at AUROC 0.653)
Renamed throughout. The reduced feature set still permits source recovery at AUROC 0.653
(Supplementary S17), so "provenance-free" over-claimed; the manuscript, captions, and labels
now say **"provenance-reduced"**, and the 0.691 transportable-signal figure is uniformly read
as an **upper-bound estimate** (R5-D4). Frozen artifact filenames, JSON keys, and marker ids
are deliberately unchanged: renaming them would churn the tamper-evident archive (checksums
and derivation stamps) for zero reader value, so the archive keys retain the historical name
while every reader-facing surface uses the corrected term. [prose+labels: Phase 3, R5-D4]

### m9 — PFNA silently omitted from the MCL-exceedance union
Stated and quantified. The union covers the four individual MCLs (PFOS/PFOA/PFHxS/HFPO-DA) plus
PFBS via its Hazard-Index HBWC; the PFNA individual MCL is omitted for source-coverage parity.
We counted the systems this mislabels: of 112 systems exceeding the PFNA MCL, **5 are
PFNA-only exceeders** (they exceed no other regulated analyte) and are therefore currently
scored non-exceedant (`pfna_exceedance_count.json`, C17). The SI now states the omission and
this count. [prose: Phase 3]

### m1 — conformal coverage undershoots yet is framed as a strength
Reframed as observed coverage. The main text now states the marginal coverage (0.910 at
alpha = 0.05) is modestly below the nominal target and is reported as observed coverage, not a
delivered guarantee; the SI drops the claim that distribution shift makes the sets conservative
(over-covering) and instead reports the observed under-coverage (0.910 marginal, 0.878 in Region
9) directly. The referee's "0.947" sub-part was already fixed in the prior pass (the stale 0.947
was replaced by the gated 0.910 with a below-target note; REVIEW_LEDGER), so we refute that piece
as already resolved. [prose: Phase 3]

### m2 — subgroup ECE is sample-size biased with no CI
Replaced with equal-mass ECE plus bootstrap CIs. From a gated T1 reproduction
(`group_ece_debiased.json`, C15), equal-mass (equal-count) binning gives PoC ECE **0.179
[0.146, 0.213]** for high-share versus **0.052 [0.044, 0.077]** for low-share systems. The
intervals do not overlap, so the calibration gap is a real subgroup difference, not a binning
artifact; the SI table and prose now carry both the equal-width and equal-mass ECE with CIs.
[C15; SI Table 25]

### m6 — three burden estimands are conflated
Disambiguated. The Discussion now states these are distinct estimands rather than conflicting
values: the system-count-weighted national summary (1.48), the geographic deployment split
(1.88 unadjusted, 1.62 after inverse-propensity adjustment), and the single western test region
(1.34). We defend the national summary as the headline and read the single-region and
deployment-split figures as context; the IPW-adjusted figure is reported. [prose: Phase 3]

### m7 — the robust national signal and the fragile regional one are not distinguished
Added the distinction. The Discussion now states the two equity signals differ sharply in
robustness: the national people-of-color monitoring-intensity disparity (1.85-fold) is
significant under a stratified-by-region permutation at essentially full statistical power,
whereas the regional detection-burden disparity is significant in only a minority of regions
under a spatially-aware null; we rest the equity claim on the monitoring-intensity finding and
read the regional detection-burden pattern as suggestive. [prose: Phase 3]

### m8 — T7 temporal degradation is confounded with the reporting-limit change
Noted. The SI T7 diagnostics now state that the measured temporal degradation is confounded with
the UCMR3-to-UCMR5 reporting-limit change (the same monitoring-defined shift the
common-reporting-limit analysis quantifies for M6), so T7 degradation cannot be attributed to
environmental drift alone. [prose: Phase 3]

### m10 — the equity figure shows stars but no CIs; excluded unknown group unmentioned
Both fixed. The equity figure now carries bootstrap 95% CI error bars per burden ratio (from
`group_burden_ci.json`, C19; PoC 1.878 [1.62, 2.18]). The main text adds a sentence on the
excluded unknown-demographics group: **692** systems (**18.3%** of the test set) whose
demographics could not be attributed, on which the model is unusually accurate (AUROC **0.984**),
consistent with better-recorded systems, so their exclusion is conservative for the disparity
estimates. [C19; Fig. 4]

### m11 — borderline regional maxima presented without multiple-comparison caveat
Labeled exploratory. The Results now state that because multiple-comparison control is applied
within each demographic family but not across the per-region maxima, the single-region extremum
(region 7, 2.57, p_FDR ~ 0.05) is exploratory. [prose: Phase 3]

### m13 — reproducibility scope: the gate is not an independent re-derivation
Characterized honestly. Code Availability now states that the committed paper gate is a
consistency and tamper-evidence check, not an independent re-derivation: it verifies that every
reader-facing number matches the frozen archive and that generated artifacts regenerate and carry
input-hash stamps, without re-running the pipeline. It states the compute budget (a full raw
reproduction is a multi-day, GPU-dependent job dominated by the LORO and HPO stages) and gives two
existing minute-scale spot-check commands (`final_gate.py --check`, `regenerate_leaderboard.py
--check`), and documents the fresh-machine test-isolation caveat (a few unit tests assume prior-run
artifacts and are skipped in isolation; the gate checks do not). [prose: Phase 3]

# RESPONSE_BANK — R6 (Nature Water `naturewater_panel_2026-07-06` panel)

## R6 — 2026-07-06 panel

Six blind referees (31 findings) + five adversarial verifiers + editor synthesis. The panel's own
verification layer found **zero of nine Majors survived as Major**; no confirmed finding is a repo-S0
outside a `CLAIMS.tsv` band, so under the retirement policy this review does not reopen the manuscript.
Absorbed in one tiered pass (Tier A/B mandatory; C-in cheap verified wins land; C-out banked). Zero new
computes. Each paragraph below is the point-by-point response; the machine-state R6 block in
`revision_ledger.md` (G7) carries the lane-typed terminal state.

### R6-R5-1 — per-figure Source Data did not match Figs 2 and 4 (RE_DERIVE; Tier A)
Repaired. `generate_source_data.py` had drifted to a stale five-figure layout with `--results`
defaulting to the git-ignored `results/` and silent placeholders on missing inputs, so
`Fig2_source_data.xlsx` shipped the benchmark leaderboard (not the confound values), Fig 4's true
source (`monitoring_inequity.json`, 1.847) was in no file, and `Fig5_source_data.xlsx` was an
orphan — making skeleton.md:985 ("Source data … provided for all main-text figures") false. We
rewired the generator to the current four figures (Fig 2 ← split_comparison + detection_only_ablation
+ feature_ablation; Fig 4 ← monitoring_inequity; Fig 3 SHAP rows fixed), set `--results` to default to
`results/paper_frozen`, made `_load_json` fail-loud, regenerated, and deleted the Fig 5 orphan. An
independent verifier re-opened every xlsx against the frozen values and figure captions: Fig 2 now
carries 0.744/0.532 (+1.40×), 0.760→0.343, 0.421→0.302; Fig 4 carries the 1.847 monitoring ratio (not
the detection-burden 1.878); all cells match frozen. The figures and frozen values were always
correct — this was a mechanical packaging repair — and the new **G14** gate clause now regenerates and
sentinel-checks the Source Data on every run so the class cannot silently rot again (R6-R5-2 / R6-GATE).

### R6-R1-2 — lead T1 with the leakage-resistant estimate, not 0.864 (DECIDED R6-D1)
Adopted. The T1 Results prose now leads with the leave-one-region-out estimate (LORO mean AUROC 0.782;
CatBoost 0.790, XGBoost 0.782) and the transportable provenance-reduced out-of-region signal (0.691),
and frames the single fixed West-only 0.864 as the disclosed optimistic upper end — naming that its test
set contains region 10, the highest-prevalence (detection rate 0.433) and most separable (held-out
AUROC 0.971) region. The 0.864 value, its five-seed stability, and the LORO-ranked Table 2 are unchanged;
the abstract already led with the honest 0.785/0.691 framing. See DECISIONS.md `### R6-D1`.

### R6-R6-5 — no deployment advance is claimed; venue stands (DECIDED R6-D2)
The contribution is diagnostic and methodological; no predictive or deployment win is claimed. The
out-of-region signal is honestly modest (AUROC ~0.691, near-chance in roughly half of regions) and the
manuscript states no new learning method is introduced. We add one explicit sentence to that effect and
retain Nature Water (Analysis) as the venue — a rigorous field-correcting cautionary result at a moment
of acute regulatory relevance (April-2024 PFAS MCLs). See DECISIONS.md `### R6-D2`.

### R6-R3-4 — "tuned-vs-default headline" contradiction (REFUTED)
Refuted on the code and config. The T1 headline AUROC 0.864 is the fixed `xgboost_classifier`
configuration in `configs/experiment.yaml` run under the pipeline default `use_tuned_params=False`
(`optuna_tuning.json` is git-ignored/absent, so no tuned params can load); `xgboost_default` (0.836) is
a *separate* sklearn-defaults baseline, not "the untuned headline." The manuscript's "default-configuration"
billing is therefore accurate. The only defect was a stale internal code comment in
`compute_common_rl.py:47-51` calling 0.864 "Optuna-tuned"; corrected as code hygiene. No manuscript change.

### R6-R6-2 — benchmark "oversold" relative to what converges (BANKED, dedup of R5-D1)
Prior rationale (R5-D1): the benchmark was already reframed honestly as "a reproducible two-confound
protocol with one converged detection task and documented boundary tasks," and the abstract co-locates
the promotional counts with the "substantive evidence is concentrated in T1 … non-convergence
annotations rather than a populated leaderboard" concession (T2 labelled a negative result; T4 a
target-leakage cautionary case). New evidence mapped: the panel re-observes exactly this co-located
concession (verify_V4) — no unmapped item. Verifier V4 countersigns the concession is prominent and
backed by the tables. No further change.

### R6-R6-4 — EJ burden finding "not novel" (BANKED, Liddie already cited)
Liddie et al. 2023 is already cited as **ref 47**, and the manuscript frames the communities-of-color
detection burden as confirmatory of established literature while staking novelty on the *reflexive
monitoring-intensity* disparity (1.85×, "rock-solid") — not on the burden. Verifier V4 confirms ref 47
is present with an exact author/venue/year match and that the abstract's novelty statement claims only
the confound decomposition + monitoring-inequity + reproducibility apparatus. No uncited-novelty gap.

### R6-R2-4 — Rosenbaum Γ* is a t-statistic re-expression (BANKED, dedup of M8d)
Prior rationale (M8d): Γ* is already disclosed at skeleton.md:346 as "an adaptation computed on the
regression scale without a matched design — a heuristic robustness screen, not the original bounded
test," and the text pivots to E-values (uniformly small, stated as fragility). New evidence mapped:
verify layer confirms Γ* = exp(|t|−1.96) is a monotone function of the t-statistic — exactly the
"heuristic, not a matched-design bound" the disclosure already states. No unmapped item; no change.

### R6-R1-4 — `seed_inflation_check.json` `is_inflated:true` is a cross-harness artifact (BANKED)
The flag compares the seed-42 headline (0.8636) to a multi-seed mean (0.8091) drawn from a *different*
(streamlined) harness with fewer systems and a reduced feature set; `headline_harness_stability.json`
re-runs five seeds on the headline harness itself → 0.8640 ± 0.0029 (no seed cherry-picking), and the
manuscript discloses and reconciles the 0.809 value (skeleton.md:225-226). The frozen JSON's label is an
apples-to-oranges comparison, not evidence of inflation; per the no-re-freeze rule the JSON is left
untouched and the reconciliation stands in the SI (C11).

### R6-R1-6 — `power_analysis.json` `loro_tests` block is degenerate (BANKED, == R6-R2-6)
Every `loro_tests` entry carries placeholder `mean_auroc: 0.0` (a `.get("summary",{})` miss on a JSON
whose mean sits at top level). It is **not reader-facing** — Supplementary Table 15 surfaces only the
three core power tests (DeLong 0.11, EJ burden, DML), which are correct. No paper claim depends on it;
per the no-re-freeze rule the frozen JSON is left untouched. (R6-R2-6 is the same finding — see below.)

### R6-R2-6 — LORO power computed from placeholder mean_auroc=0.0 (BANKED, ≡ R6-R1-6, countersigned)
Identical to R6-R1-6: R1 (ML remit) and R2 (statistics remit) independently flagged the same dead
`loro_tests` block in `power_analysis.json`. Mapping: same file, same key, same root cause
(`_evaluation.py` reads `["summary"]["mean_auroc"]` absent from `loro_cv.json`), same non-reader-facing
status, same disposition (frozen JSON untouched; the reader-facing power claim cites the correct
`delong_auroc` power 0.11). The independent verifier countersigns the two are one finding.

### R6-R2-7 — feature-count 130-vs-131 bookkeeping (BANKED, dedup of minor-featurecount)
Prior disposition (minor-featurecount): the reader-facing count is reconciled to **130** and gated via
`pn:dml_total`; `causal_deconfounding.json` has 130 rows. The 131 in `deconfounded_auroc.json` /
`sensitivity_bounds.json` is a one-feature internal bookkeeping difference in non-reader-facing archive
files. New evidence mapped: the panel re-observes the 130/131 and 54/55 discrepancy across the same
three files — no unmapped item. Per the no-re-freeze rule the JSONs are untouched; the manuscript says
"54 of 130," gated.

### R6-R4-2 — linguistic-isolation Inf/p in the frozen archive (BANKED)
`monitoring_inequity.json` ships a degenerate `pct_limited_english` row (`monitoring_ratio: Infinity`,
`p_value: 1.0` — an empty reference bin because >50% of systems have zero limited-English share)
alongside a valid continuous `size_adjusted_p` 3.4e-9. The two measure different estimands. The
manuscript **does not propagate either as a reader claim**: linguistic isolation is excluded from the
burden table, and Fig. 4a's caption already states "an infinite ratio indicates no low-share systems."
No over-claim reaches the reader; per the no-re-freeze rule the JSON is left untouched.

### R6-R1-3 — LORO leader-flip and top-model inseparability (CONFIRMED true; banked as ammunition)
Verified true and honestly disclosed: `loro_cv_full.json` CatBoost 0.7896 > voting 0.7839 > XGBoost
0.7825 (fixed-split leader → 3rd), and `power_analysis.json` `delong_auroc` power 0.110 (top two
indistinguishable, underpowered), both stated at skeleton.md:237-241. Recorded as response-letter
ammunition: the panel validated the disclosure. No change.

### R6-R2-3 — conformal coverage undershoot fully disclosed (CONFIRMED true; banked as ammunition)
Verified: marginal coverage 0.910 vs nominal 0.95 with per-region variation and `reliable_guarantee:false`
/ `n_calibration:0`, all reported as observed coverage with the by-design exchangeability break stated
(skeleton.md:479-481; Methods 861-864). No over-claim; recorded as ammunition. No change.

### R6-R3-Lead3 — T4 target-leakage self-correction is correct (CONFIRMED true; banked as ammunition)
Verified: the leakage mechanism is precisely diagnosed and **fixed in code** (`sdwis.py:171-179` sets the
reporting limit NaN for detected rows, dropping the leaked feature), and the deflated numbers
(0.962→0.699→0.550) are in the abstract. Exemplary self-correction; recorded as ammunition. No change.

### R6-R3-Lead4 — UCMR5 MNAR selection correctly acknowledged (CONFIRMED true; banked as ammunition)
Verified: UCMR5's census-plus-representative-sample design and the resulting MNAR selection are
characterized in Limitations (skeleton.md:557-564), with IPW caveated as unable to rule out selection on
unobservables. Correct treatment; recorded as ammunition. No change.

### R6-R1-1 — ambient-source population asymmetry (14,335 vs 16,281) undisclosed (PROSE-FIX)
Disclosed. A Methods clause now names the asymmetry: the ambient sources (CA GeoTracker, WQP) are
reserved for external validation and excluded from the fixed-split benchmark but retained in the LORO and
split-comparison harnesses, so those harnesses run on a larger, ambient-inclusive population (16,281 vs
14,335 systems, both gated); because the ambient population is harder and structurally distinct, the
asymmetry makes the LORO/split comparators more conservative rather than inflating the fixed-split
headline. No conclusion changes; the decisive confound evidence is population-matched.

### R6-R2-1 — DML first-stage strength undisclosed (PROSE-FIX)
Disclosed and reframed as a strength. SI S14 now reports that the treatment-residualization nuisance
model has negative cross-fitted R² for 109 of 130 features (gated), i.e. most features are near-orthogonal
to the monitoring confounders. The adjustment removes little for them not because it is mis-specified but
because there is little monitoring confounding to remove — which strengthens the reading that the retained
environmental signal is not a monitoring artifact. No CLAIMS row depends on the DML.

### R6-R2-2 — G=10 block bootstrap anti-conservative; promote the t(9) interval (PROSE-FIX)
Done. The main-text transportable-signal CI now foregrounds the honest cluster-robust t(9) recalibration,
which widens the provenance-reduced interval to 0.606–0.783 (gated) and is reported as the headline
uncertainty; the block bootstrap is named anti-conservative given only ten region clusters, and the
interval still excludes chance.

### R6-R2-5 + R6-R4-3 — calibration gap underweighted; give it equal billing (PROSE-FIX)
Rebalanced. The Discussion now states that the one group disparity that is both robust and unfavourable is
calibration (ECE 0.18 vs 0.06, non-overlapping bootstrap CIs), and that this gap — not the non-significant
error-rate gaps — is what the recommended per-group recalibration must target. The favourable coverage gap
is disclosed as multiplicity-fragile (would not survive correction across the nine group comparisons).

### R6-R4-3 — calibration is the robust unfavourable metric (PROSE-FIX; paired with R6-R2-5)
Handled together with R6-R2-5 in the same Discussion edit: the calibration gap (ECE 0.18 vs 0.06,
non-overlapping CIs) is now named as the one group disparity that is both robust and unfavourable, and as
the specific quantity the recommended per-group recalibration must target.

### R6-R4-1 — favourable coverage uses a different partition than the error metrics (PROSE-FIX)
Disclosed. The Discussion now states that the favourable coverage gap rests on a validation-set median
split, a different demographic partition than the percentile split used for the discrimination, error-rate,
and calibration analyses; combined with the multiplicity caveat (R6-R2-5), we do not lean on it.

### R6-R3-1 — soften "robust"; report re-censored base rate (PROSE-FIX + BANKED subpart)
Softened. SI now scopes the robustness explicitly to ranking (AUROC) and states precision-recall does not
survive (AUPRC collapses to 0.102). The "report the re-censored base rate" subpart is BANKED: that value
is not in the frozen archive (`common_rl_sensitivity.json` records AUPRC 0.102 and n_test 3,780 but no
re-censored positive prevalence), and deriving it needs a re-run this pass forbids (DECISIONS R6-D3). Named
question for the maintainer: is a frozen source for the re-censored prevalence available, or does this stay
banked? It blocks nothing — the AUPRC-collapse direction is already disclosed.

### R6-R3-2 — MCL-exceedance target omits the mixture Hazard Index (PROSE-FIX)
Disclosed. SI S18 now flags that the scored target fires only on single-analyte exceedances and does not
compute the EPA mixture Hazard Index, so Hazard-Index-only violations (sub-threshold analytes summing to
HI ≥ 1) are scored non-exceedant; the scored "MCL exceedance" is an individual-MCL approximation used only
as a harder-target comparison, where the qualitative direction is robust.

### R6-R3-5 — deployment signal is regionally non-uniform (PROSE-FIX)
Disclosed at the deployment claim. The Discussion now states the out-of-region value is regionally
non-uniform (near-chance in several EPA regions, per-region provenance-reduced AUROC as low as 0.57), so a
deployer cannot assume uniform utility.

### R6-R1-5 — T3 degenerate shared-fallback cells shown as numbers (PROSE-FIX via generator)
Fixed in the generator. `table_benchmark_results` now applies the same degenerate-cell flagging as
`table_benchmark_full` (via `gate_lib.degenerate_metric_models`), so the T3 byte-identical micro-metric
cells render "non-converged" in `table2_benchmark_results.md` too, never as plausible numbers. Both
benchmark tables are now consistent; G4 regenerates and verifies.

### R6-R6-1 — uncited Nature Communications 2024 geospatial-ML review (PROSE-FIX / courtesy cite)
Added as reference 59 (Koldasbayeva et al., *Nat. Commun.* 15, 10700, 2024) and cited alongside the
spatial-leakage antecedents. Its content is already covered by refs 45, 51–57, so it is a courtesy
positioning citation.

### R6-R6-3 — uncited closely-adjacent STOTEN 2024 California EJ-ML paper (PROSE-FIX / cite + distinction)
Added as reference 58 (Karasaki, Morello-Frosch & Callaway, *Sci. Total Environ.* 951, 175730, 2024) with
a one-sentence distinction in the group-fairness Discussion: that California study audits a drinking-water
model for demographic error disparities; our contribution traces the disparity upstream to the inequitable
monitoring process that generates the labels. It is a single-jurisdiction fairness case study and does not
scoop the national two-confound / monitoring-intensity contribution.

### R6-R6-6 — ref-10 (Fernandez) title/volume mismatched (PROSE-FIX / citation correction)
Corrected. Reference 10 now reads Fernandez, N., Nejadhashemi, A. P. & Loveall, C., "Large-scale
assessment of PFAS compounds in drinking water sources using machine learning," *Water Res.* 243, 120307
(2023) — verified against CrossRef (PubMed 37480598) — the paper the Extended Data Table 3 comparison
actually characterizes, replacing the mismatched 245:120527 string.

### R6-R5-2 / R6-GATE — the deterministic gate had a source-data blind spot (POLICY + code, G14)
Closed by a new gate clause. Before R6, no clause regenerated or validated `paper/source_data/*.xlsx`
(G4 runs only `generate_tables.py` and `regenerate_leaderboard.py`), so a true-green gate could — and
did — ship the B1 defect. We added **G14** to `paper/final_gate.py` (registered in `CLAUSES`, not in the
fast subset): it regenerates the Source Data from `results/paper_frozen` to a temp dir, requires the
committed and regenerated file SETS to be equal (a lingering Fig5 or a dropped figure fails), requires
per-sheet cell-value equality via openpyxl (content only, since xlsx bytes are nondeterministic), and
asserts per-figure SENTINEL values read straight from the frozen JSON (Fig 4 must contain
`monitoring_inequity.json`'s 1.847; Fig 2 the confound values) so a generator that regenerates its own
mistake still fails. Five unit escape-replays (mutated cell / missing file / extra file /
generator-nonzero / sentinel logic) plus one live replay (corrupt a committed cell → gate red → restore →
green) verify the clause. The class the gate green-lit is now gate-enforced and cannot silently rot.

### R6-GATE — the gate blind-spot slice (POLICY + code; paired with R6-R5-2)
Closed jointly with R6-R5-2 by the new G14 clause: the class of defect a true-green gate could ship
(per-figure Source Data not regenerated or validated against the frozen archive) is now gate-enforced and
can never silently rot again.

### R6-R5-3 — below-chance WQP cross-source diagnostic undisclosed (PROSE-FIX)
Disclosed. The SI WQP entry now reports the pooled cross-source AUROC of 0.42 (gated), below chance,
and explains it as a Simpson's-paradox artifact of pooling regions with very different base rates (four of
five per-region AUROCs are above chance). It is a fair-transfer failure for ambient, non-PWS data rather
than a model defect; like the CA GeoTracker result, it reinforces that ambient geographic generalization
remains undemonstrated. No claim rests on it.

## Issue #55 corrections (pre-publication pass, 2026-07-11)

### issue55-refs — reference metadata + first-appearance renumber
Two entries with wrong bibliographic metadata were corrected against the publisher of record:
ref 9 now cites Tokranov et al., "Predictions of groundwater PFAS occurrence at drinking water
supply depths in the United States," *Science* 386, 748-755 (2024), and ref 45 now cites Stock,
Gregr & Chan, "Data leakage jeopardizes ecological applications of machine learning," *Nat. Ecol.
Evol.* 7, 1743-1745 (2023). Two miscitations were fixed: the neurodevelopment claim now cites the
Lanphear et al. (2005) pediatric-IQ pooled analysis rather than the 2018 adult-mortality cohort,
and the DWINSA infrastructure survey (ref 16) was moved off the SDWIS monitoring-records sentence
(ref 17 alone supports it) to the water-systems-count sentence it does support. Six minor metadata
errors were corrected (doc number, ZCTA vintage, NLCD year, CatBoost pages, Liddie title, Fithian
subtitle). TabPFN v2 (Hollmann et al., *Nature* 637, 319-326, 2025) is now cited alongside v1, and
three Supplementary-only citations (Barber 2023, Duan 1983, VanderWeele & Ding 2017) were added as a
Supplementary References section. The bibliography was renumbered to Nature order-of-first-appearance
and is guarded by a new marker<->entry bijection linter.

### issue55-ucmr3 — UCMR3 "38 contaminants" verified correct (no edit)
A prior audit flagged skeleton's "UCMR3 ... 38 contaminants including 6 PFAS" as possibly conflicting
with an EPA "30 contaminants" figure. Verified against the assembled data: UCMR3 monitored 38 distinct
contaminants, of which 6 are PFAS. The manuscript figure is correct; no edit was made.

### issue55-refcurrency — ref 7/31 currency notes (no defect)
Ref 7 (PFAS NPDWR, 89 FR 32532) has a later proposed partial rescission (91 FR 29413, 2026); the
manuscript's past-tense "established" is accurate for the rule as promulgated, so no change is needed.
Ref 31 (LCRR) was partly superseded by the LCRI (89 FR 86418, 2024); the SDWIS lead/copper samples
were collected under the LCR as amended and the sentence does not assert LCRR currency, so no change.

### issue55-gnn-tabpfn — false "failed (CUDA)" cause corrected with real LORO numbers
The Table 2 GNN cells previously read "failed (CUDA)"; the run log shows the real cause was that the
LORO driver never passed the coordinates the graph models require. Rather than relabel, we completed
the runs: GCN 0.683, GraphSAGE 0.652 (honest unconditional linear-fallback inference), TabPFN 0.758
across a full 10/10 folds under the disclosed 3,000-sample CPU protocol. CatBoost still leads LORO at
0.790. The SI note no longer claims the linear fallback "does not arise under random splitting" (it is
unconditional in this implementation).

### issue55-misgeocode — wrong-ZIP geocoding is immaterial to the headline
Some systems carry an out-of-state mailing/operator-ZIP centroid that places them in a foreign EPA
region. The geographic split derives from the PWSID prefix and is unaffected; the only exposure is
wrong-place environmental features for the flagged systems. The flag judges each coordinate
against exact US Census state boundary polygons (nearest state within 3.0 degrees for
water-jittered points): 1,818 of 87,450 geocoded systems. Excluding them end-to-end from
training and evaluation moves T1 AUROC 0.864->0.866 and T4 0.699->0.697 (frozen
`misgeocode_sensitivity_v2.json`, dual reproduction gates), so no headline result depends on the
mis-geocoded systems. The split map excludes them; Supplementary S7b reports the analysis.
*(2026-07-12: rule upgraded from state bounding boxes to exact polygons — the boxes falsely
flagged ~64% of their 6,889, correctly-located border systems, visibly banding the split map;
the superseded v1 artifact and its sensitivity are retained as the conservative broader-exclusion
record, decision `edfig1-D1`.)*

## R7 — 2026-07-12 simulated Nature Water referee report (terminal pre-submission screen)

Adjudicated under the maintainer's standing decision (2026-07-12): one scope-frozen screen of
the MAIN review file, fix reopen-key/must-fix/checkpoint-approved lanes, bank the rest. Zero S0
(independently audited, including M1). The rehearsed responses:

**M1 (the pivotal equity finding — source adjustment).** The reviewer is right that the pooled
"1.85x" monitoring-intensity ratio conflated cross-source corpus assembly with within-program
allocation. We re-estimated it (`source_adjusted_monitoring.py`, reproduction-gated on the frozen
1.847/0.060): with data-source fixed effects the size-adjusted association attenuates ~28%
(coef 0.060 to 0.043) but remains highly significant (p<1e-15); restricting to systems monitored
predominantly under a single federal program (UCMR5) lowers the ratio to 1.37x (coef 0.021,
p<1e-3). The disparity therefore SURVIVES source adjustment rather than collapsing; we dropped
"rock-solid", report the source-adjusted numbers, and added an explicit cross-source-confound
sentence. The equity claim stands, appropriately qualified.

**M2 (Article vs Analysis; length).** The manuscript is already targeted and framed as a Nature
Water *Analysis* (DECISIONS v2-D2 / R6-D2); the review's "submitted as an Article" premise is a
simulation artifact. No new-method claim is made. Length is an editorial matter for the Analysis
format; we will condense to the editor's limit on request.

**M3 (overclaim surfaces).** Softened throughout: ICP invariance "genuine rather than partial" ->
"substantial but partial"; IPW "not a monitoring artifact" -> "survives adjustment for observed
monitoring propensity"; SI S6 "calibrated uncertainty" -> "observed (approximate) coverage"; and
the abstract now carries the significant calibration gap (ECE 0.18 vs 0.06) beside the
non-significant FNR gap. The block-bootstrap CIs are labeled anti-conservative and the honest
t(9) interval is reported in Results; we removed the false "which we report as the headline
uncertainty" self-description (the full CI swap is banked, since both intervals exclude chance and
no conclusion flips).

**M4 (integrity items).** The SI S19 LORO power statistic was placeholder-derived (a nested-key
bug in the generator); we fixed the root cause and re-derived it from the real fold AUROCs (power
still >0.99). The DML Methods now state the actual estimand (residualized against a fixed
monitoring/size confounder set, not "all other features") and disclose the pooled 5-fold
cross-fitting (so 0.835 is an in-sample pooled value distinct from the 0.864 geographic headline;
the pair is a relative decomposition, not a generalization estimate).

**M5 (prior art / naming).** Added Peters, Bühlmann & Meinshausen 2016 with an explicit note that
our "Invariant Contamination Predictor" is distinct from their Invariant Causal Prediction; cited
the method lineage (Ganin/DANN, Arjovsky/IRM, Sagawa/Group DRO); cited and rebutted Wadoux et al.
2021 on spatial CV; and added Martinez-Morata, Nigra et al. 2022 as the nearest equity prior art.
Tokranov et al. 2024's "Random CV" label could not be verified (paywalled) and is carried as a
request to the editor.

**Minors + packaging (banked or fixed).** Corrected the intro to "five individual PFAS MCLs plus a
Hazard Index" (matching our own SI S18); added Minnesota MDH to the Methods source enumeration;
reconciled the Reporting Summary (the NJ headed-browser scraper is described, "all but two" sources
directly downloadable); softened the requirements-lock.txt claim to a non-portable advisory
snapshot; corrected the archive MANIFEST count (75 JSONs); regenerated software_versions.json with
the previously-omitted tabpfn / torch-geometric / optuna (13 key packages verified identical to the
freeze environment). The abstract "up to 58%" framing, the areal-apportionment attenuation, the
deployment-lift pairing, the Gamma E-value, and the cross-source overlap field were verified as
already-adequately-handled dedups against R1/R3/R5/M8d and are banked.

---

## Readability pass (2026-08-03)

### RP-1 — clm_seed_stable spelling (PROSE; paired CLAIMS.tsv diff)
Readers reported the manuscript was hard to follow, and a prose-only clarity pass was run over the
whole paper (no number, claim, or citation changed; the deterministic gate stayed green). One
`CLAIMS.tsv`-pinned sentence had to change text: `clm_seed_stable` contained the UK spelling
"favourable-seed artifact", which violated the manuscript's stated US-spelling convention and was
the last reader-facing UK spelling in the paper. The claim itself is untouched — the seed-stability
evidence (`headline_harness_stability.json`), the strength (DESCRIPTIVE), and every number are
identical; only the single word's spelling changed. The pinned row in `paper/CLAIMS.tsv` and the
sentence in `paper/skeleton.md` were edited in the same commit, so the paired diff is the witness.
No other pinned sentence changed text: every other clarity edit restructured the prose *around*
the pinned spans, which survive byte-identical.

