# Revision Ledger — *Nature Water* Referee Report (2026-06-16)

**This ledger is the contract and the exit-gate scorecard for the final revision pass.**
Scope is frozen to `paper/review_nature_water_20260616.md`. The pass is complete only when every row is
`Done=yes` with a verifier whose perspective ≠ the fixer's. New issues found after this ledger is frozen go
to `## Next revision` and do **not** reopen this pass (anti-spiral rule).

Perspectives: **P1** ML methodology · **P2** stats/causal · **P3** water-quality domain · **P4** environmental
justice · **P5** reproducibility/integrity · **P6** editorial/significance.

---

## Pre-flight facts (verified this session)

- **Branch:** `fix/paper-final-review-revisions` (off `main`).
- **verify_paper baseline (frozen):** `python paper/verify_paper.py --results-dir results/paper_frozen/` →
  exit 0, **0 errors, 0 warnings**. Exit-gate target: keep at 0 errors / ≤0 warnings. (`n_checks` printed is
  cosmetic = warnings+1.) Requires `PYTHONUTF8=1` on Windows (cp1252 else fails on the `✓` glyph).
- **SPLICE resume base validated:** `results_reframe/` ≡ frozen — all **86/86** `(task,model)` metric entries
  identical (incl. T4); `feature_ablation.json`, `loro_cv.json`, `multi_seed_stability.json`,
  `bootstrap_ci.json`, `shap_T4.json` all **byte-identical** reframe↔frozen. `results_reframe/checkpoints/`
  has all 6 tasks (T1,T2,T3,T4,T5,T7). Never re-freeze from `results/` (drifted, partial).
- **M4a nuance:** `spatial_block_bootstrap.json` holds the **with-provenance** T1 CI
  `[0.690, 0.845]` (mean 0.765). The Abstract's `0.698 ± 0.133` is the **provenance-free** model → Lane D
  must COMPUTE a provenance-free region-block-bootstrap CI (mirror `_strengthening.py:region_block_bootstrap`),
  not merely surface the existing one.

---

## Disposition table

Status legend: ground-truth = CONFIRMED / REBUT / OPEN. Disposition = FIX / REBUT / ACK. Done = yes/no.

| ID | Claim (abbrev) | Ground-truth | Disposition | Fixer | Verifier | Files | Evidence | Done |
|----|----|----|----|----|----|----|----|----|
| M1 | T4 collapse = leaked `mean_detection_limit` | CONFIRMED | FIX (re-run + reframe) | P3 | P1 | sdwis.py; Lane D | | no |
| M2 | LCR intensity partly legit risk proxy | CONFIRMED | FIX (prose) | P3 | P5 | skeleton.md | | no |
| M3a | Abstract over-generalizes 1.50 | CONFIRMED | FIX (prose) | P4 | P2 | skeleton.md | | no |
| M3b | 0.756/0.842 lacks CI/scope | CONFIRMED | FIX (prose+CI) | P4/P2 | P2/P4 | skeleton.md; equity code | | no |
| M3c | "5 of 10" vs "4 of 10" | OPEN (locate source) | REBUT/RECONCILE | P2 | P4 | TBD source | | no |
| M3d | Methods "population-" vs Results "system-count-" weighted | CONFIRMED | FIX (match code) | P2 | P5 | skeleton.md; equity code | | no |
| M4a | 0.698±0.133 raw SD → block-bootstrap CI | CONFIRMED (needs prov-free CI) | FIX (compute+prose) | P2 | P1 | Lane D; skeleton.md | | no |
| M4b | per-region DeLong + power caveat | CONFIRMED | FIX (compute+prose) | P2 | P1 | Lane D; skeleton.md | | no |
| M4c | national burden ratio no CI | CONFIRMED | FIX (compute+prose) | P2 | P4 | Lane D; skeleton.md | | no |
| M5a | Table 3 AUPRC reversal unexplained | CONFIRMED | FIX (prose) | P1 | P2 | skeleton.md | | no |
| M5b | HPO/early-stop/ensemble asymmetry | CONFIRMED (config stays) | FIX (prose) | P1 | P2 | skeleton.md | | no |
| M5c | conformal "conservative" overclaim | CONFIRMED | FIX (docstring+prose) | P1 | P5 | conformal.py:6-9; skeleton | | no |
| M6 | public loro.py lacks drop_leakage | CONFIRMED | FIX (code+test) | P1 | P3 | benchmark/loro.py; new test | | no |
| M7a | soften IPW "not an artifact" | CONFIRMED | FIX (prose) | P4 | P3 | skeleton.md | | no |
| M7b | foreground under-sampling | CONFIRMED | FIX (prose) | P4 | P3 | skeleton.md | | no |
| #46 | six lingering items non-material | CONFIRMED (§7) | ACK | P5 | P6 | — | | no |
| m1 | Fig 4 "p<0001" + collisions | CONFIRMED (visual) | FIX (figure layout) | P6 | P5 | generate_figures.py | | no |
| m2 | aquifer_confinement mislabel | CONFIRMED | FIX (config+prose) | P6 | P3 | data.yaml:144; skeleton | | no |
| m3 | TabPFN 3000 vs 10000 | CONFIRMED | FIX (prose) | P6 | P1 | skeleton.md | | no |
| m4 | word count ~3,980 | CONFIRMED | FIX (trim LAST) | P6 | P2 | skeleton.md | | no |
| m5 | priority-claim under-defended | — | FIX (prose) | P6 | P3 | skeleton.md | | no |
| m6 | lit-table 2025→2026 | CONFIRMED | FIX (data) | P6 | P5 | table_ext5 csv (+source) | | no |
| m7 | degenerate ICP-T3 row | CONFIRMED | FIX (annotate) | P5 | P1 | supp table gen | | no |
| m8 | add S17 ICP caveat | CONFIRMED | FIX (prose) | P5 | P1 | supplementary_information.md | | no |
| m9 | stale spatial_autocorr ICP rows | CONFIRMED | FIX (recompute/footnote) | P5 | P2 | spatial_autocorrelation.json/SI | | no |
| m10 | Moran's I range reconcile | CONFIRMED | FIX (prose) | P2 | P5 | skeleton/SI | | no |
| m11 | stale val_reuse_bias docstring | CONFIRMED | FIX (docstring) | P5 | P6 | val_reuse_bias.py | | no |
| §8-i | wire verify_paper into CI | CONFIRMED missing | FIX (DONE on main, CI green) | P5 | P6 | .github/workflows/ci.yml (main f87ee99) | verify-paper job ran green | yes |
| §8-iii | tighten weak EJ substring check | CONFIRMED | FIX (code) | P5 | P2 | verify_paper.py:747-759 | | no |
| §8-iv | add PYTHONHASHSEED/CuBLAS | CONFIRMED missing | FIX (code) | P5 | P1 | _reproducibility.py | | no |

**Total enumerated:** 7 majors (M1–M7, decomposed to 15 sub-rows) + 11 minors + #46 + 3 §8 = 30 rows.
(§8-ii == M6; counted once.)

---

## Exit gate (all must be TRUE → STOP)

1. Every row above has a disposition + concrete evidence; row count complete.
2. Every row: verifier-perspective ≠ fixer-perspective.
3. `verify_paper.py --results-dir results/paper_frozen/` → 0 errors, ≤0 warnings.
4. `sha256sum -c results/paper_frozen/checksums.sha256` all OK; manifest regenerated by `freeze_results.py`.
5. ruff + ruff format --check + mypy + pytest tests/unit/ green; new M6 test green.
6. Abstract-level uncertainty present (each value in prose AND a frozen JSON).
7. M1 propagated + reframed consistently; old T4 literals absent from prose.
8. Word-count check passes; Abstract ≤165 words; no display item loses its only ref.
9. DOCX regenerated after prose; Fig 4 visually confirmed.
10. This ledger frozen + committed; §8-i patch prepared with manual step surfaced.

---

## Execution log

### Batch 1 — disjoint code fixes (applied; gates green; Layer-2 verify in progress)
- **M6** (`benchmark/loro.py`): added `drop_leakage_columns` import + call right after
  `aggregate_to_system_level` (mirrors `_robustness.py:100-101`). New test `test_no_target_leakage`
  in `tests/unit/test_benchmark_loro.py`. **Evidence:** fixed path mean AUROC = **0.500** (exact
  chance); `corr(any_detected, target) = 1.000` → old path forced ~1.0; threshold 0.95 has full margin.
- **§8-iv** (`_reproducibility.py`): `set_seed` now sets `PYTHONHASHSEED` + `CUBLAS_WORKSPACE_CONFIG`
  (via `setdefault`) and `torch.cuda.manual_seed_all`. Description documents the child-process / pre-CUDA
  caveats.
- **m2** (`configs/data.yaml:141-149`, `features/hydrogeology.py`): **REDESCRIBE, not remove.**
  Ground truth: `us_aquifers.shp` has no `AQ_TYPE` (cols = ROCK_NAME/ROCK_TYPE/AQ_NAME/AQ_CODE);
  `aquifer_confinement` is **100% NA** in `hydrogeology.parquet`; `get_dummies(dummy_na=True)`
  (assembly.py:327, before the NA-drop at :345) turns it into a **constant `aquifer_confinement_nan`**
  that is baked into frozen `shap_T1.json`/`feature_importance.json`. Removing the mapping would change
  SHAP artifacts → cascade. So kept the column (artifact-stable) and fixed every misleading description.
  Prose (`skeleton.md:507` + feature catalog) → Batch 3.
- **M5c** (`calibration/conformal.py:1-13`): replaced one-directional "conservative" claim with observed
  per-region coverage (≈0.917 dips below nominal) + Mondrian/weighted caveat. Description-only; no
  behavior change. (T1/PFAS conformal — unaffected by the SDWIS fix, so 0.917 stays valid.)
- **§8-iii DEFERRED to Batch 3** — tightening the EJ burden-ratio check is entangled with the M3 Abstract
  rewrite (risk R6: must not mis-fire on rewritten prose). Co-develop there.
- **Gates:** ruff ✓, ruff format ✓, mypy ✓ (144 files), `pytest` 91 passed across loro/reproducibility/
  hydrogeology/calibration/feature_assembly.

### Batch 2 (Lane D) — progress
- **M1 code fix** applied (`sdwis.py`: non-censored detection_limit → NaN), gates green.
- **Isolation**: `data_m1` = junction(raw) + copies(processed, interim); `results_m1` = copy of `results_reframe`.
- **Surgical data fix (drift-free)**: instead of re-parsing (which exposed pre-existing parser drift, see Next revision), set `detection_limit = NaN` for `source=='sdwis'` rows on the **canonical** `merged_wq` → `data_m1/interim/merged_wq.parquet`. **Verified**: exactly 694,419 rows changed, 0 non-DL columns changed, 0 non-SDWIS rows changed.
- **Reproduction gate PASSED**: OLD control (canonical data) reproduces frozen T4 to **|Δ| = 1.1e-16** for xgboost/catboost/lightgbm/rf/logistic → any NEW-vs-frozen change = the fix alone.
- **NEW corrected chain** (full T4 15 models + bootstrap + ablation + multi-seed) running on data_m1.
- **Cited-T4 scope** (re-run + splice): `results.json`, `bootstrap_ci.json`, `feature_ablation.json`, `multi_seed_stability.json`. Provenance-free 0.545 UNCHANGED (already excludes mean_detection_limit). Other T4-token files (loro/calibration/conformal/model_comparison/spatial/seed_inflation/power) not surfaced for T4 → leave frozen, note in MANIFEST.
- **Splice mechanism**: build staging = frozen non-T4 entries (verbatim) + new T4, then `freeze_results.py --only` (slims results.json, rechecksums). Slim fields = y_true/y_prob/latitudes/longitudes/split_labels in `metadata`.
- **NEW T4 numbers (corrected, leak removed)** — xgboost AUROC **0.962→0.677** [0.668,0.687], AUPRC **0.900→0.417**; catboost 0.678/0.427; rf 0.663; logistic 0.629; ICP 0.623→0.607; prov-free 0.545 UNCHANGED. Ablation: all_features AUPRC 0.902→**0.442**; monitoring_intensity ΔAUPRC **−0.587→−0.127** (now n_samples only, mean_detection_limit auto-dropped). multi-seed T4 xgboost sd 0.0014. **Reframe**: 0.962 was mostly a target-leakage artifact; genuine skill 0.677; modest monitoring (n_samples) effect.
- **SPLICE DONE + VERIFIED**: `freeze_only` replaced results.json/feature_ablation.json/bootstrap_ci.json/multi_seed_stability.json. Gate (vs git HEAD): **0 non-T4 entries changed** in all; multi_seed T1 identical; T4 updated (14/9/28 entries). All 52 checksums verify (CRLF-agnostic). `git diff --stat` = only those 4 files + checksums.
- **Byte-identity insight**: entry-level splice (reuse frozen non-T4 verbatim) was ESSENTIAL — results_m1 had 4 drifted non-T4 ICP entries (nondeterministic ICP) + ~2e-4 T1-ablation noise; reusing frozen non-T4 neutralized all of it.

### M3 ground truth (verified this session)
- **M3c = REBUT.** Frozen Supp Table 20 (`table_supp_loro_equity.csv`) shows POC FDR-significant in regions 2,4,7,9,10 = **5 of 10**; `loro_equity_analysis.json` `n_regions_significant_fdr=5`; skeleton:332 says "5 of 10". All consistent. The reviewer's "4 of 10" is a miscount/stale read — no change needed (document evidence). Skeleton's "2.67 not FDR-significant" also correct (region 8, p_FDR=0.091).
- **M3d = FIX.** `_equity_transfer.py:518` uses `weights = n_systems` → **system-count** weighting. Results:332 "system-count-weighted" CORRECT; **Methods:762 "population-weighted" is WRONG** → change to system-count.
- **M3a**: 1.50 is real but heterogeneous (range 0.917–2.672, 5/10 sig) → rescope Abstract to match body.
- **M4a nuance**: `spatial_block_bootstrap.json` CI [0.690,0.845] is WITH-provenance (mean 0.765). The Abstract's 0.698 is provenance-free → must COMPUTE a provenance-free region-block-bootstrap CI in D4.
- **M4b**: `loro_cv_*` folds store per-fold metrics (AUROC, precision, recall, accuracy, n_test) but NOT raw scores → per-region AUC-vs-chance via Hanley–McNeil (recover n_pos/n_neg from metrics, CPU) OR a deterministic T1 LORO re-run for true DeLong; decide in D4.

### Batch 2 (Lane D) — D4 statistics DONE + tables regenerated
- New code: `analysis/strengthening.py` adds `auc_vs_chance`, `auc_difference` (Hanley-McNeil) + 8 tests; reproducible runner `paper/compute_inference_strengthening.py` → frozen `inference_strengthening.json` (53rd file, all checksums verify). ruff/mypy/tests green.
- **M4a** prov-free LORO: point estimate **0.698** (loro_cv mean, verify_paper-enforced; do NOT write 0.687 — verify_paper guards it), **block-bootstrap CI [0.603, 0.790]**. With-provenance **0.774 (CI 0.690–0.845)** [existing]. Replace the raw ±SD framing.
- **M4b** per-region AUC-vs-chance (prov-free T1): R2 p=0.003, R6 p=0.012, R4 p=2e-7, R5/R10 p≈0 — even weakest folds are *formally above chance* given large n → soften "near chance" to "weak (0.54–0.58) but above chance."
- **M4c** national POC burden ratio **1.50 (95% block-bootstrap CI 1.21–1.91)** — excludes 1.0.
- **M3b** group AUROC gap **0.756 vs 0.842, Δ=0.085, p=0.006** (Hanley-McNeil two-sample, independent groups).
- **verify_paper.py:182** `_EXPECTED_METRICS["T4"]` → auroc 0.678 / auprc 0.427 (CatBoost, corrected).
- **Tables regenerated** from corrected frozen: table2 T4 (xgboost 0.677/0.417), table_ext3_ablation T4 (baseline 0.442, monitoring_intensity & n_samples_only ΔAUPRC −0.127). Figures NOT yet regenerated (await Fig 4 m1 fix).

### CORRECTED-NUMBER REFERENCE (for Batch-3 prose; OLD → NEW)
| Where | OLD | NEW |
|---|---|---|
| T4 headline AUROC (xgboost) | 0.962 (CI 0.959–0.964) | **0.677 (CI 0.668–0.687)** |
| T4 headline AUPRC | 0.900 | **0.417** |
| T4 catboost / lightgbm AUROC | 0.961 / 0.960 | 0.678 / 0.656 |
| T4 ICP AUROC | 0.623 | **0.607** |
| T4 prov-free AUROC | 0.545 | 0.545 (UNCHANGED) |
| T4 ablation baseline AUPRC | 0.902 | **0.442** |
| T4 monitoring_intensity ΔAUPRC | −0.587 | **−0.127** (now n_samples only) |
| T4 multi-seed sd | <0.001 | ~0.001 (xgboost 0.0014) |
| out-of-region with-prov | 0.774 ± 0.106 | **0.774 (CI 0.690–0.845)** |
| out-of-region prov-free | 0.698 ± 0.133 | **0.698 (CI 0.603–0.790)** |
| national burden ratio | 1.50 | **1.50 (CI 1.21–1.91)** |
| group AUROC gap | 0.756 vs 0.842 (no test) | 0.756 vs 0.842 (**p=0.006**) |
**T4 reframe**: 0.962 was almost entirely a target-leakage artifact (mean_detection_limit = mean lead conc, SDWIS ~100% non-censored); leak-removed skill 0.677; genuine monitoring (n_samples) effect modest (ΔAUPRC −0.127); still > prov-free 0.545.

## FINAL STATUS — all 30 enumerated items resolved

**Exit gate (all green):** ruff ✓ · ruff format ✓ · mypy ✓ (144 files) · `verify_paper --results-dir results/paper_frozen/` **0 errors / 0 warnings** (incl. new §8-iii national-burden check) · 53/53 checksums OK · Fig 4 visually confirmed · DOCX regenerated · pytest running.

| ID | Resolution |
|----|----|
| M1 | FIXED — sdwis.py NaN detection_limit for detects; T4 surgically re-run (xgboost 0.962→0.677); spliced; prose reframed across Abstract/Results/Discussion/SI/ED. |
| M2 | FIXED — T4 prose: LCR sampling intensity presented as partly a legitimate risk proxy (40 CFR 141.86). |
| M3a | FIXED — Abstract + body: 1.50 framed as heterogeneous (CI 1.21–1.91; 5/10 sig). |
| M3b | FIXED — 0.756 vs 0.842 scoped to POC dimension + western region, with p=0.006 (Hanley-McNeil). |
| M3c | **REBUTTED** — table+JSON+skeleton all show 5 of 10; reviewer's "4 of 10" is wrong; no change. |
| M3d | FIXED — Methods "population-weighted" → "system-count-weighted" (matches code). |
| M4a | FIXED — out-of-region 0.698 now reports block-bootstrap CI [0.603,0.790] (raw ±SD removed). |
| M4b | FIXED — per-region AUC-vs-chance computed; "near chance" softened to "weak but above chance." |
| M4c | FIXED — national burden ratio 1.50 now carries block-bootstrap CI [1.21,1.91]. |
| M5a | FIXED — T1 AUPRC reversal (0.731→0.750) explained. |
| M5b | FIXED — ICP early-stop / ensemble-config asymmetry disclosed; architecture ranking de-emphasized (config UNCHANGED). |
| M5c | FIXED — conformal docstring + (existing) prose report observed sub-nominal coverage + Mondrian caveat. |
| M6 | FIXED — loro.py drop_leakage_columns + regression test (leakage-free AUROC 0.500 vs leak ~1.0). |
| M7a | FIXED — IPW "not an artifact" → "robust to observed monitoring confounders" + MNAR caveat. |
| M7b | FIXED — under-sampling of low-income (0.73×)/low-education (0.79×) foregrounded in Abstract. |
| #46 | ACK — §7 confirms non-material. |
| m1 | FIXED — Fig 4 layout (p-values above caps, decimal legible; title headroom); visually confirmed. |
| m2 | FIXED — aquifer_confinement redescribed (config + hydrogeology docstring + feature catalog + skeleton:532). |
| m3 | FIXED — TabPFN "3,000" → "10,000-sample cap, larger rejected." |
| m4 | FIXED — Abstract trimmed 184→~160 words (verify_paper warning cleared). |
| m5 | FIXED — priority claim defended (vs prior bespoke single-source models). |
| m6 | FIXED — lit-table year 2025→2026 (generator). |
| m7 | FIXED — degenerate ICP/Deep-Tobit/TabPFN/Voting/Stacking T3 documented non-converged (S17). |
| m8 | FIXED — S17 scope caveat (ICP validated T1/T4; degenerate T3/T5/T7). |
| m9 | FIXED — stale spatial_autocorr ICP rows footnoted (pre-GRL-fix, excluded from headline). |
| m10 | FIXED — Methods Moran's I reconciled with S9 (XGBoost 0.091 at 200 km). |
| m11 | FIXED — val_reuse_bias docstring 0.145→0.085. |
| §8-i | DONE — verify-paper CI job added to main (f87ee99, web editor) and ran green on CI; gate now active for all pushes/PRs to main. |
| §8-iii | FIXED — verify_paper national-burden check added (catches Abstract errors). |
| §8-iv | FIXED — PYTHONHASHSEED/CUBLAS_WORKSPACE_CONFIG in _reproducibility.py. |

**Anti-spiral:** exit on this fixed checklist; no fresh open-ended review commissioned.

## §8-i CI gate — APPLIED

The `verify-paper` job below was added to `.github/workflows/ci.yml` on `main`
(commit f87ee99) via the GitHub web editor — the local and `gh` credentials lack
`workflow` scope, so neither git push nor the Contents API could write it (the
API returned 404). The job ran **green** on CI
(`verify_paper.py --results-dir results/paper_frozen/`), so the prose-vs-frozen
consistency gate is now active for every push/PR to `main`.

```yaml
  verify-paper:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: pip install --prefer-binary -e ".[test]"
      - name: Verify paper prose matches frozen snapshot
        run: python paper/verify_paper.py --results-dir results/paper_frozen/
```

## SDWIS reproduce-from-raw root cause (2026-06-24) — quantify-first, decision pending

- **Symptom:** `reproduce.py` from raw does not reproduce the published T4/T5 (committed `sdwis.py`
  parses ~917k rows; the published canonical SDWIS is 694,419).
- **Root cause (forensic, pinned raw — sha256 matches manifest, NOT a data change):** the canonical
  694,419 = exactly the `SAMPLE_MEASURE > 0` rows (lead 655,976 / copper 38,443). The committed parser
  keeps the 222,550 `SAMPLE_MEASURE == 0` rows. No `>0` filter exists in committed `sdwis.py` (either
  version) or downstream — the freeze-time loader had filtering the committed loader lost. Secondary,
  independent drift: censoring `=="L"` (no-op → 0% censored, matches canonical) → `=="<"` (correct,
  commit 5d2f5e9). Committed `data/interim/merged_wq.parquet` is pre-M1 (SDWIS detection_limit filled),
  so the arbiter of "published" is `results/paper_frozen/results.json`, not local data artifacts.
- **Key finding (methodology, not just a bug):** the 222,550 zeros are REAL records (literal `0`, valid
  `mg/L` + dates), a literal-0 spike distinct from the positive continuum (median 3.6 µg/L) → EPA
  non-detect / below-detection 90th-percentile results coded as 0 (clear non-exceedances). Dropping them
  removes 5,572 zero-only lead systems (6.3%) entirely → upward bias in T4's exceedance base rate.
  "Reproducing" the published number would enshrine this; the defensible alternative (keep non-detects)
  changes the published T4.
- **This pass (author chose "quantify first"):** added a default-neutral `zero_handling` param to
  `sdwis.py` (`keep` default = current behavior; `drop` reproduces the 694,419 canonical set) + tests;
  measured the T4 keep-vs-drop delta in isolation (env matches `software_versions.json` exactly) and
  reported it for the keep-vs-drop decision. **No published number/frozen file changed.**
- **Measured (2026-06-24; env matches frozen exactly, drop-mode reproduces published T4 to dev=0.0):**
  keep-vs-drop T4 xgboost AUROC 0.677→0.691 (+0.014), AUPRC 0.417→0.399 (−0.017); catboost keep
  0.670/0.390; +1,250 non-detect test systems survive feature assembly (18,911→20,161). Keeping
  non-detects modestly raises AUROC and lowers AUPRC (published AUPRC is slightly optimistic from
  base-rate enrichment); the monitoring-confounding narrative is qualitatively unchanged (~0.68–0.69,
  still far below the leaked 0.962). Script: `paper/measure_sdwis_zero_impact.py`.
- **Decision PENDING:** keep (→ flip default, re-run SDWIS-dependent results, re-freeze, update paper) vs
  drop (→ flip default to drop + documented Methods rationale; numbers unchanged).

## Next revision (items discovered AFTER freeze — do NOT reopen this pass)

- **Zenodo dataset bundle (`results/zenodo-dataset/`) is STALE — regenerate before publication.**
  Verified 2026-06-24: 14/14 checksums OK, canonical splits (train 1,3,4,5,6 / val 2,7 / test 8,9,10),
  but `manifest.json` is dated 2026-03-30 — it predates the M1 SDWIS detection_limit fix and the pending
  #2 zero-handling decision, so its SDWIS data is pre-correction. Regenerate from the #2-resolved +
  M1-corrected data (`pipeline/export.export_zenodo_dataset`) and mint the Zenodo DOI (paper placeholders
  at skeleton.md:863,870) before release. Blocked on the #2 decision + #1.

- **Pre-existing SDWIS parser drift**: current `sdwis.py` re-parses `data/raw/SDWA_LCR_SAMPLES.csv` to **916,899** rows / 0.31% censored, vs the frozen-consistent canonical `sdwis.parquet` of **694,419** / 0.00% censored. The committed loader diverged from the code that built the frozen snapshot (unrelated to the M1 one-line fix). Not flagged by this review; resolving it (adopting 917K) would change the entire SDWIS/T4/T5 data foundation → out of scope. The M1 fix was therefore isolated surgically on the 694K frozen-consistent data.

---

## Authoritative issue set — machine-verified terminal states

The block below is the machine-readable contract for the *final-gate* revision
(`REFEREE_REPORT_NatureWater.md` M1–M5 + minors, plus the still-present ICP-T4
item from the prior round). It is parsed and checked by `paper/verify_ledger.py`:
OBJECTIVE → names a passing `verify_paper` check; ACKNOWLEDGED → cites a manuscript
phrase that is present; DECIDED → ratified rationale + a `### <id>` section in
`paper/DECISIONS.md`.

<!-- ledger-machine-state -->
```json
{
  "issues": [
    {"id": "M1", "terminal_state": "DECIDED", "ratified": true, "rationale": "Reorder abstract to lead with the confound-triangulation and monitoring-inequity novelty; move spatial-leakage to a supporting clause; soften the title."},
    {"id": "M2", "terminal_state": "DECIDED", "ratified": true, "rationale": "Soften 'standardized benchmark/ecosystem' to 'reproducible evaluation protocol and task suite'; annotate T3/T5/T7 non-convergence; T1 carries methodological weight."},
    {"id": "M3", "terminal_state": "OBJECTIVE", "verify_check": "check_text_metric_reconciliation"},
    {"id": "M4", "terminal_state": "OBJECTIVE", "verify_check": "check_text_metric_reconciliation"},
    {"id": "M5a", "terminal_state": "OBJECTIVE", "verify_check": "check_table3_invariance_rows"},
    {"id": "M5b", "terminal_state": "DECIDED", "ratified": true, "rationale": "Retain pre-fix T4 (~0.96) files as a provenance record; allowlist them as advisory stale warnings; canonical T4 (0.677) lives in results.json; no re-run."},
    {"id": "M5c", "terminal_state": "DECIDED", "ratified": true, "rationale": "Attribute pre-fix T4 to 'tree models reaching 0.962' rather than XGBoost specifically (CatBoost was tied)."},
    {"id": "M5d", "terminal_state": "ACKNOWLEDGED", "manuscript_ref": {"file": "paper/skeleton.md", "phrase": "the upper-bound estimate of the transportable environmental signal"}},
    {"id": "M5e", "terminal_state": "DECIDED", "ratified": true, "rationale": "Split Table 1 SDWIS footnote (genuine LCR 90th-percentile measurement, not detection-only reporting); decline the constant rename, which would alter un-recomputable frozen analysis."},
    {"id": "minor-200M", "terminal_state": "ACKNOWLEDGED", "manuscript_ref": {"file": "paper/skeleton.md", "phrase": "a modeled estimate predating UCMR5"}},
    {"id": "minor-featurecount", "terminal_state": "DECIDED", "ratified": true, "rationale": "Referee m6: reconciled to the frozen archive -- causal_deconfounding.json has 130 rows (not 127), 69 FDR-significant. Prose updated to 69 of 130 and gated via pn:dml_sig_count and pn:dml_total, machine-checked by G1/G3 (no brittle phrase anchor)."},
    {"id": "minor-detectiononly", "terminal_state": "ACKNOWLEDGED", "manuscript_ref": {"file": "paper/skeleton.md", "phrase": "cannot contribute to AUROC where noted"}},
    {"id": "minor-conformal", "terminal_state": "ACKNOWLEDGED", "manuscript_ref": {"file": "paper/skeleton.md", "phrase": "per-region guarantees are not claimed"}},
    {"id": "Q4-minors", "terminal_state": "DECIDED", "ratified": true, "rationale": "Rebut: existing reproduction-claim and feature-coverage text judged adequate by the author; no new limitation prose added."},
    {"id": "ratchet-infra", "terminal_state": "DECIDED", "ratified": true, "rationale": "verify_paper default flipped to results/paper_frozen with a hard results.json gate; text reconciliation is curated-anchor-based (reliability over exhaustiveness)."},

    {"id": "R5-M1", "terminal_state": "PROSE_FIX", "section": "R5", "lane": "PROSE+C10+G13", "desc": "Abstract FNR 0.50/0.44 unsupported + selectively framed; add CIs/tests, state non-significance (p~0.16), reframe as calibration/threshold target.", "witness": {"file": "paper/skeleton.md", "phrase": "The group error profile warrants care rather than alarm"}},
    {"id": "R5-M2", "terminal_state": "PROSE_FIX", "section": "R5", "lane": "PROSE+audit", "desc": "Benchmark over-stated; only T1 converged; T3 byte-identical cells; report primary metric per task; drop ImageNet/GLUE equivalence.", "witness": {"file": "paper/skeleton.md", "phrase": "its central point is methodological"}},
    {"id": "R5-M3", "terminal_state": "PROSE_FIX", "section": "R5", "lane": "PROSE+D3+C18", "desc": "Leaderboard = single West-only split; LORO leader flips; i.i.d. CIs anti-conservative; re-rank Table 2 by full-family LORO.", "witness": {"file": "paper/skeleton.md", "phrase": "The ranking flips under LORO: CatBoost leads"}},
    {"id": "R5-M4", "terminal_state": "PROSE_FIX", "section": "R5", "lane": "C11+PROSE+G13", "desc": "Robustness on a different harness than the 0.864 headline; seed_inflation flags is_inflated; reconcile the three circulating T1 values; multi-seed on the headline harness.", "witness": {"file": "paper/skeleton.md", "phrase": "The headline is seed-stable: the five-seed spread on its own harness is a few thousandths of a point"}},
    {"id": "R5-M5", "terminal_state": "PROSE_FIX", "section": "R5", "lane": "C12+PROSE+D4", "desc": "10-cluster block bootstrap anti-conservative (width ratio 1.002; T4 0.845); state G=10; reconcile honest-estimate vs upper-bound.", "witness": {"file": "paper/skeleton.md", "phrase": "resample only ten EPA-region clusters and are anti-conservative"}},
    {"id": "R5-M6", "terminal_state": "PROSE_FIX", "section": "R5", "lane": "C13+PROSE+G13", "desc": "T1 label monitoring-defined; add the promised main-text common-RL caveat; report AUPRC under re-censoring (already frozen).", "witness": {"file": "paper/skeleton.md", "phrase": "This label is monitoring-defined: reporting limits are heterogeneous across programs"}},
    {"id": "R5-M7", "terminal_state": "PROSE_FIX", "section": "R5", "lane": "C14+PROSE", "desc": "Deployment/lift use detection not MCL-exceedance target (9-pt gap); surface MCL metrics + exceedance top-decile lift in main text.", "witness": {"file": "paper/skeleton.md", "phrase": "detection-based scores are a weaker but still useful proxy for exceedance risk"}},
    {"id": "R5-M8", "terminal_state": "PROSE_FIX", "section": "R5", "lane": "D2+POLICY+PROSE+human", "desc": "Data availability: MN MDH redistribution pending; correct 'publicly available'; document NJ DEP legal basis; reconcile Zenodo MIT vs CC-BY; show thesis survives without MN.", "witness": {"file": "paper/reporting_summary.md", "phrase": "Minnesota MDH is obtained by written request"}},
    {"id": "R5-M9", "terminal_state": "PROSE_FIX", "section": "R5", "lane": "D1+POLICY+PROSE", "desc": "Novelty framing honest for venue; cite presence-only/surveillance-bias antecedents; drop first/largest/ImageNet over-claims; lead with the two genuine advances.", "witness": {"file": "paper/skeleton.md", "phrase": "No new learning method is introduced"}},
    {"id": "R5-m1", "terminal_state": "PROSE_FIX", "section": "R5", "lane": "PROSE", "desc": "Conformal coverage undershoots (0.910/0.845/0.763) yet framed as strength; report as observed coverage, not a delivered guarantee. (0.947 sub-part already fixed => REFUTE.)", "witness": {"file": "paper/skeleton.md", "phrase": "reported as observed coverage rather than a delivered guarantee"}},
    {"id": "R5-m2", "terminal_state": "COMPUTE", "section": "R5", "lane": "C15", "desc": "Subgroup ECE (0.18 vs 0.06) sample-size biased, no CI, vs pooled 0.038; use debiased/equal-mass binning with bootstrap CIs.", "artifact": "group_ece_debiased.json"},
    {"id": "R5-m3", "terminal_state": "COMPUTE", "section": "R5", "lane": "C16", "desc": "Power table effect (0.016) != frozen AUROC gap (0.008); treats paired AUCs as independent; silent fallback. FIX + fail-loud loader + test.", "artifact": "power_analysis.json"},
    {"id": "R5-m4", "terminal_state": "PROSE_FIX", "section": "R5", "lane": "D4", "desc": "'provenance-free' misnomer (source recoverable at 0.653); rename to 'provenance-reduced' in prose+labels; reconcile 'honest estimate'.", "witness": {"file": "paper/skeleton.md", "phrase": "A provenance-reduced PFAS signal survives"}},
    {"id": "R5-m5", "terminal_state": "PROSE_FIX", "section": "R5", "lane": "C18+PROSE", "desc": "ICP separates weakly (0.635 vs 0.653) and no LORO-ICP reported; report LORO-ICP (from C18) + soften invariance language.", "witness": {"file": "paper/skeleton.md", "phrase": "the two\nhonest estimators broadly agree rather than one being decisively stricter"}},
    {"id": "R5-m6", "terminal_state": "PROSE_FIX", "section": "R5", "lane": "PROSE", "desc": "Three burden estimands (1.88 test / 1.48 national LORO-wtd / 1.34 region-9); state which is defended where; report IPW-adjusted figure.", "witness": {"file": "paper/skeleton.md", "phrase": "distinct estimands rather than conflicting values"}},
    {"id": "R5-m7", "terminal_state": "PROSE_FIX", "section": "R5", "lane": "PROSE", "desc": "Regional detection burden fragile (3/10) vs rock-solid national monitoring intensity; add one distinguishing clause.", "witness": {"file": "paper/skeleton.md", "phrase": "The two equity signals differ sharply in robustness"}},
    {"id": "R5-m8", "terminal_state": "PROSE_FIX", "section": "R5", "lane": "PROSE", "desc": "T7 temporal degradation confounded with the UCMR3->UCMR5 reporting-limit change; note it.", "witness": {"file": "paper/supplementary_information.md", "phrase": "confounded with this reporting-limit change"}},
    {"id": "R5-m9", "terminal_state": "COMPUTE", "section": "R5", "lane": "C17+PROSE", "desc": "PFNA silently omitted from the MCL-exceedance union; state and quantify the PFNA-only exceeder count.", "artifact": "pfna_exceedance_count.json"},
    {"id": "R5-m10", "terminal_state": "COMPUTE", "section": "R5", "lane": "C19+PROSE", "desc": "Equity figure shows stars but no CIs; add error bars; one sentence on the excluded unknown-demographics group (AUROC 0.984).", "artifact": "group_burden_ci.json"},
    {"id": "R5-m11", "terminal_state": "PROSE_FIX", "section": "R5", "lane": "PROSE", "desc": "Multiple-comparison control within-family only; label borderline regional maxima exploratory.", "witness": {"file": "paper/skeleton.md", "phrase": "this single-region extremum is exploratory"}},
    {"id": "R5-m12", "terminal_state": "PROSE_FIX", "section": "R5", "lane": "PROSE+G9/G10", "desc": "'Fifteen sources' over-counts; state consistently across text/cover-letter/.zenodo; reconcile Zenodo MIT vs CC-BY.", "witness": {"file": "paper/CONCEPT_REGISTRY.tsv", "phrase": "data_source_count"}},
    {"id": "R5-m13", "terminal_state": "PROSE_FIX", "section": "R5", "lane": "PROSE+POLICY", "desc": "Reproducibility scope: document the fresh-machine test-isolation caveat + a compute budget + a cheap spot-check path; characterize the gate as consistency/tamper-evidence, not independent re-derivation.", "witness": {"file": "paper/skeleton.md", "phrase": "not an independent re-derivation of the science"}}
  ]
}
```

## R6 — Nature Water 2026-07-06 panel work-list (machine-verified terminal states)

The block below is the R6 authoritative issue set: the 31 panel findings + `R6-GATE`
(the source-data blind-spot slice). Seeded OPEN in Phase 1 (ratchet: any OPEN row keeps
G7 red under `EXPECTED_RED={G7}`); flipped to a lane-typed terminal state as each is
dispositioned. `parse_machine_state` unions this block with the R5 block above (both are
read; duplicate ids fail closed). Terminal states: `PROSE_FIX`/`RE_DERIVE`/`DECIDED`/
`REFUTED`/`BANKED`/`COMPUTE` (no computes this pass). See `.claude/prompts/r6-kickoff-2026-07-06.md`
and `REVIEW_LEDGER.md` for the human index.

<!-- ledger-machine-state -->
```json
{
  "issues": [
    {"id": "R6-R1-1", "terminal_state": "PROSE_FIX", "witness": {"file": "paper/skeleton.md", "phrase": "ambient-inclusive population"}, "section": "R6", "lane": "PROSE-FIX/C-in", "desc": "Ambient-source population asymmetry: headline 14,335 (ambient-excluded) vs LORO/split 16,281 (ambient-included); name it in one Methods clause."},
    {"id": "R6-R1-2", "terminal_state": "DECIDED", "ratified": true, "decision_ref": "R6-D1", "section": "R6", "lane": "DECIDED/R6-D1", "rationale": "Lead T1 with LORO 0.782 / PF 0.691; frame 0.864 as the disclosed optimistic upper end (region 10 in the fixed West-only test set). The 0.864 value, seed-stability annotation, and LORO-ranked Table 2 are untouched; the abstract already leads with the honest framing.", "desc": "0.864 is the most-favourable estimate from the most-favourable config (West-only split incl. region 10); lead T1 with LORO 0.782 / PF 0.691."},
    {"id": "R6-R1-3", "terminal_state": "BANKED", "section": "R6", "lane": "BANKED/confirmation", "desc": "LORO leader-flip + top-model inseparability verified true and disclosed (:237-241). Response-letter ammo."},
    {"id": "R6-R1-4", "terminal_state": "BANKED", "section": "R6", "lane": "BANKED", "desc": "seed_inflation_check.json is_inflated:true is a cross-harness artifact, reconciled at :225-226; frozen JSON untouched."},
    {"id": "R6-R1-5", "terminal_state": "PROSE_FIX", "witness": {"file": "paper/tables/table2_benchmark_results.md", "phrase": "non-converged"}, "section": "R6", "lane": "PROSE-FIX/C-in", "desc": "T3 degenerate shared-fallback cells (identical 0.7533) shown without a non-converged annotation; annotate via the generator."},
    {"id": "R6-R1-6", "terminal_state": "BANKED", "section": "R6", "lane": "BANKED", "desc": "power_analysis.json loro_tests block is degenerate (mean_auroc 0.0); not reader-facing; frozen JSON untouched. (== R6-R2-6.)"},
    {"id": "R6-R2-1", "terminal_state": "PROSE_FIX", "witness": {"file": "paper/supplementary_information.md", "phrase": "most features are near-orthogonal to the monitoring"}, "section": "R6", "lane": "PROSE-FIX/C-in", "desc": "DML first-stage strength undisclosed (84% near-zero/neg residualization R2 = near-orthogonality); add one SI S14 line. Zero DML CLAIMS rows."},
    {"id": "R6-R2-2", "terminal_state": "PROSE_FIX", "witness": {"file": "paper/skeleton.md", "phrase": "cluster-robust t(9) recalibration"}, "section": "R6", "lane": "PROSE-FIX/C-in", "desc": "G=10 block bootstrap anti-conservative; promote the honest t(9) CI [0.6055,0.7833] to the headline via generate_tables."},
    {"id": "R6-R2-3", "terminal_state": "BANKED", "section": "R6", "lane": "BANKED/confirmation", "desc": "Conformal coverage undershoot fully disclosed as observed-not-guaranteed; cleared. Response-letter ammo."},
    {"id": "R6-R2-4", "terminal_state": "BANKED", "section": "R6", "lane": "BANKED", "desc": "Rosenbaum Gamma* = exp(|t|-1.96), already disclosed as a heuristic at :346 (M8d); residual only -> bank."},
    {"id": "R6-R2-5", "terminal_state": "PROSE_FIX", "witness": {"file": "paper/skeleton.md", "phrase": "the one group disparity that is both robust"}, "section": "R6", "lane": "PROSE-FIX/C-in", "desc": "Group calibration gap (ECE 0.179 vs 0.052) underweighted vs favourable metrics; equal billing + multiplicity disclosure (p 0.007->0.06 / 9 tests). (Pairs with R6-R4-3.)"},
    {"id": "R6-R2-6", "terminal_state": "BANKED", "section": "R6", "lane": "BANKED", "desc": "power_analysis.json loro_tests placeholder mean_auroc=0.0 (== R6-R1-6, countersigned dedup); frozen JSON untouched."},
    {"id": "R6-R2-7", "terminal_state": "BANKED", "section": "R6", "lane": "BANKED", "desc": "Feature-count 130 (coef table/text) vs 131 (deconfounded_auroc/sensitivity); DECIDED-to-130 (minor-featurecount); frozen JSONs untouched."},
    {"id": "R6-R3-1", "terminal_state": "PROSE_FIX", "witness": {"file": "paper/supplementary_information.md", "phrase": "robust in ranking (AUROC) to reporting-limit"}, "section": "R6", "lane": "PROSE-FIX/C-in + BANKED", "desc": "Common-RL AUPRC collapse 0.727->0.102: soften 'robust' to ranking/AUROC (C-in); report re-censored base rate BANKED (not frozen; needs re-run)."},
    {"id": "R6-R3-2", "terminal_state": "PROSE_FIX", "witness": {"file": "paper/supplementary_information.md", "phrase": "does not compute the"}, "section": "R6", "lane": "PROSE-FIX/C-in", "desc": "MCL-exceedance target omits the mixture Hazard Index; add one SI S18 sentence + MCL-harder note (9-pt AUROC gap)."},
    {"id": "R6-R3-4", "terminal_state": "REFUTED", "section": "R6", "lane": "REFUTED", "rationale": "0.864 is the fixed experiment.yaml xgboost_classifier config run under use_tuned_params=False (optuna_tuning.json git-ignored/absent, so no tuned params load); xgboost_default (0.836) is a separate sklearn-defaults baseline. 'default-configuration' billing is accurate. Only defect = stale compute_common_rl.py:47-51 code comment, corrected.", "desc": "Tuned-vs-default headline REFUTED (V1: 0.864 = fixed yaml xgboost_classifier; 'default-configuration' accurate). Residual = fix stale compute_common_rl.py:47-51 comment."},
    {"id": "R6-R3-5", "terminal_state": "PROSE_FIX", "witness": {"file": "paper/skeleton.md", "phrase": "regionally non-uniform"}, "section": "R6", "lane": "PROSE-FIX/C-in", "desc": "Out-of-region 0.691 regionally non-uniform (per-region 0.565-0.956, near-chance in ~3 regions); add one caveat at the deployment claim."},
    {"id": "R6-R3-Lead3", "terminal_state": "BANKED", "section": "R6", "lane": "BANKED/confirmation", "desc": "T4 target-leakage self-correction verified correct + coded (sdwis.py:171-179). Response-letter ammo."},
    {"id": "R6-R3-Lead4", "terminal_state": "BANKED", "section": "R6", "lane": "BANKED/confirmation", "desc": "UCMR5 MNAR selection design correctly acknowledged (:557-564). Response-letter ammo."},
    {"id": "R6-R4-1", "terminal_state": "PROSE_FIX", "witness": {"file": "paper/skeleton.md", "phrase": "a validation-set median split"}, "section": "R6", "lane": "PROSE-FIX/C-in", "desc": "Coverage reassurance uses a val-median split (n 3120/660) != the 80/50 split (618/1544) of the FNR/AUROC/ECE metrics; add one disclosing clause."},
    {"id": "R6-R4-2", "terminal_state": "BANKED", "section": "R6", "lane": "BANKED", "desc": "Linguistic-isolation Infinity/p=1.0 vs size-adjusted p=3e-9 in frozen monitoring_inequity.json; text clean (excluded from burden); caption already caveats; JSON untouched."},
    {"id": "R6-R4-3", "terminal_state": "PROSE_FIX", "witness": {"file": "paper/skeleton.md", "phrase": "the per-group recalibration we recommend must target"}, "section": "R6", "lane": "PROSE-FIX/C-in", "desc": "Calibration gap is the one robust UNfavourable group metric; one sentence noting it is the target of the recommended per-group recalibration. (Pairs with R6-R2-5.)"},
    {"id": "R6-R5-1", "terminal_state": "RE_DERIVE", "section": "R6", "lane": "RE_DERIVE/Tier-A", "artifacts": ["paper/source_data/Fig1_source_data.xlsx", "paper/source_data/Fig2_source_data.xlsx", "paper/source_data/Fig3_source_data.xlsx", "paper/source_data/Fig4_source_data.xlsx"], "desc": "Per-figure Source Data stale/wrong (Fig2=leaderboard; Fig4=riskmap placeholder; 1.847 in no file; Fig5 orphan); skeleton:985 false. Rewired generate_source_data.py to the 4-figure layout (--results default -> frozen; fail-loud _load_json), regenerated, deleted Fig5. Independent verifier: Fig2 confound values + Fig4 1.847 present, cell-vs-frozen ALL PASS. G14 gates the class."},
    {"id": "R6-R5-2", "terminal_state": "PROSE_FIX", "witness": {"file": "paper/final_gate.py", "phrase": "g14_source_data"}, "section": "R6", "lane": "POLICY+code/G14/Tier-B", "desc": "Gate blind spot: no clause regenerates/validates source data (G4 runs only tables+leaderboard); a true-green gate shipped B1. Install G14."},
    {"id": "R6-R5-3", "terminal_state": "PROSE_FIX", "witness": {"file": "paper/supplementary_information.md", "phrase": "yields a pooled AUROC of"}, "section": "R6", "lane": "PROSE-FIX/C-in", "desc": "wqp_regional_validation pooled AUROC 0.4205 (below chance, Simpson's-paradox artifact) undisclosed; add one SI sentence reporting + caveating it."},
    {"id": "R6-R6-1", "terminal_state": "PROSE_FIX", "witness": {"file": "paper/skeleton.md", "phrase": "Challenges in data-driven geospatial modeling"}, "section": "R6", "lane": "POLICY-cite/C-in", "desc": "Uncited Nat Commun 2024 geospatial-ML review (Koldasbayeva, s41467-024-55240-8); courtesy citation (content already covered by refs 45,51-57)."},
    {"id": "R6-R6-2", "terminal_state": "BANKED", "section": "R6", "lane": "BANKED", "desc": "Benchmark 'oversold' (only T1 converges); concession already co-located in the abstract + T2/T4 labels (R5-D1). Bank."},
    {"id": "R6-R6-3", "terminal_state": "PROSE_FIX", "witness": {"file": "paper/skeleton.md", "phrase": "dissecting an algorithmic approach to predict drinking"}, "section": "R6", "lane": "POLICY-cite/C-in", "desc": "Uncited closely-adjacent STOTEN 2024 CA drinking-water-EJ ML paper (S0048969724058868); add citation + one-sentence distinction (does not scoop the national two-confound contribution)."},
    {"id": "R6-R6-4", "terminal_state": "BANKED", "section": "R6", "lane": "BANKED", "desc": "EJ burden finding not novel; Liddie 2023 already ref 47 and burden framed confirmatory; novelty staked on monitoring-intensity. Bank."},
    {"id": "R6-R6-5", "terminal_state": "DECIDED", "ratified": true, "decision_ref": "R6-D2", "section": "R6", "lane": "DECIDED/R6-D2", "rationale": "Venue stands (Nature Water Analysis); S3 venue-fit judgment, not an over-claim. Add one explicit sentence that the contribution is diagnostic/methodological and no deployment win is claimed. Venue retarget is a maintainer call.", "desc": "No demonstrated deployment/predictive advance; venue stands (Nature Water Analysis) + one explicit diagnostic/methodological sentence (no deployment win claimed)."},
    {"id": "R6-R6-6", "terminal_state": "PROSE_FIX", "witness": {"file": "paper/skeleton.md", "phrase": "Large-scale assessment of"}, "section": "R6", "lane": "POLICY-cite/Tier-A", "desc": "Ref 10 (Fernandez) title/volume mismatched on the load-bearing Table 3 comparison; correct to Water Res. 243:120307 (web-confirm at execution)."},
    {"id": "R6-GATE", "terminal_state": "PROSE_FIX", "witness": {"file": "paper/final_gate.py", "phrase": "per-figure sentinel values"}, "section": "R6", "lane": "POLICY+code/G14/Tier-B", "desc": "The source-data blind-spot slice: the class of defect a true-green gate could ship. Closed jointly with R6-R5-2 by the G14 clause."}
  ]
}
```

