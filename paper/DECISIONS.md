# Editorial & Deferral Decisions — Nature Water final revision (frozen)

This file records every **EDITORIAL** call and every deliberate **DECIDED** deferral
for the final revision pass. Each decision is frozen: it is reopened **only** on
authoritative external input (the journal editor or an assigned reviewer), never on a
future self-initiated review. Each entry gives the call, the rationale, a pre-written
rebuttal ("if a reviewer raises X, the answer is Y because Z"), and a reopen rule.

Machine-checked by `paper/verify_ledger.py`: every `DECIDED` issue in the ledger
machine-state must have a matching `### <id>` section here with a non-empty rationale
and a ratified flag.

Author ratification: the editorial calls (M1, M2, the M3/M4 scope, Q4-minors) were
selected by the corresponding author via an explicit decision batch on 2026-06-23.

---

### M1
**Lane:** EDITORIAL. **Call:** Reorder the abstract to lead with the genuinely novel
contributions — the ascertainment-confound triangulation (feature ablation +
monitoring-invariant predictor + double machine learning) and the monitoring-inequity
finding — moving the random-vs-geographic spatial-leakage result (prior art) to a
supporting clause; add "regionally heterogeneous / on average" qualifiers; retitle to a
named-benchmark hybrid that hedges the resource (cite-by-reuse) and finding (interest)
citation paths: "AquaContam: machine-learning models of drinking-water contamination
learn who is monitored, not the environment."
**Rationale:** The spatial-leakage story is conceded prior art (ref 45); the defensible
novelty is the triangulated ascertainment confound and the directional inequity result.
Leading with them matches the evidence. The title was reworked from the abstract
"predictions encode..." phrasing to lead with the named benchmark — for a resource paper
the dominant citation mechanism is reuse, so naming AquaContam maximizes citability while
the clause retains the provocative finding.
**Rebuttal:** If a reviewer says the spatial-leakage framing is unoriginal, the answer is
that it is now explicitly supporting context and credited to ref 45, while the lead claim
is the ascertainment triangulation, which is new for administratively generated water data.
**Reopen:** only on external editor input.

### M2
**Lane:** EDITORIAL. **Call:** Soften "the community's first standardized evaluation
ecosystem" to "a reproducible evaluation protocol and task suite"; state explicitly that
substantive evidence is concentrated in T1, and annotate T3/T5/T7 as non-converged rather
than presenting them as a populated leaderboard.
**Rationale:** Only T1 and T4 carry main display items; several models do not converge on
the sparser tasks. "Protocol and task suite" is the honest scope. Populating T3/T5/T7
would require new experiments (declined per surface-area policy).
**Rebuttal:** If a reviewer says the benchmark is over-sold, the text now claims only a
reproducible protocol with T1 carrying methodological weight, with explicit non-convergence
annotations — no ImageNet/GLUE-style breadth is claimed.
**Reopen:** only on external editor input.

### M5b
**Lane:** DECIDED (retain + document). **Call:** The pre-leakage-fix T4 (~0.96) values in
`spatial_block_bootstrap.json`, `seed_inflation_check.json`, and
`loro_detection_only_ablation.json` are retained as a provenance record and **allowlisted**
as advisory warnings from `check_stale_frozen_files`; they are not regenerated and back no
current claim. The text cites 0.962 only as the pre-fix "before" in the before/after narrative.
**Rationale:** Re-running these to refresh T4 would re-enter the pipeline the no-re-run rule
forbids (SDWIS parser has drifted to 916,899 vs the frozen 694,419 rows). The canonical T4
(0.677) lives in `results.json`; the verifier flags the stale files so no reader mistakes them
for current sources.
**Rebuttal:** If a reviewer re-derives a T4 CI from these files, the answer is that they are
flagged stale by `verify_paper` and superseded by `results.json` (T4 = 0.677); they are kept
only to document the artefact.
**Reopen:** only on external editor input.

### M5c
**Lane:** DECIDED (prose nuance). **Call:** Attribute the inflated pre-fix T4 to "tree models
reaching AUROC 0.962" rather than to XGBoost specifically.
**Rationale:** Pre-fix, CatBoost and XGBoost were essentially tied; naming one over-specifies.
**Rebuttal:** If a reviewer notes CatBoost was the pre-fix best, the text now says "tree models."
**Reopen:** only on external editor input.

### M5e
**Lane:** DECIDED (footnote split; code rename declined). **Call:** In Table 1, give SDWIS a
distinct footnote (§) explaining its 0% censoring reflects genuine Lead-and-Copper-Rule
90th-percentile measurements that are only rarely left-censored — categorically different from
the detection-only-reporting PFAS sources (†). The `DETECTION_ONLY_SOURCES` code constant is
**not** renamed.
**Rationale:** The footnote correction fully addresses the mislabelling. Renaming the constant
would change branches feeding the frozen detection-only ablation, an analysis the no-re-run rule
forbids recomputing.
**Rebuttal:** If a reviewer objects to grouping SDWIS with detection-only sources, the answer is
that Table 1 now distinguishes them by footnote; the internal constant name is an implementation
detail that does not affect the reported numbers.
**Reopen:** only on external editor input.

### Q4-minors
**Lane:** DECIDED (rebut). **Call:** Do **not** add (a) a softening of the "regenerates all
results" reproduction claim or (b) an under-represented-PFAS-source feature-coverage limitation,
beyond what the manuscript already states.
**Rationale:** The corresponding author judged the existing Data/Code Availability and limitations
text adequate on these two minor points; adding more limitation prose was declined to control
manuscript surface.
**Rebuttal:** If a reviewer raises reproduction completeness, the Code Availability section already
names the curated processed parquets as the reproduction entry point; if they raise feature
coverage, the modest 0.698 out-of-region signal is already framed as a floor, not a ceiling.
**Reopen:** only on external editor input.

### ratchet-infra
**Lane:** DECIDED (infrastructure). **Call:** (1) `verify_paper.py` now defaults
`--results-dir` to `results/paper_frozen` and hard-errors if `results.json` is absent.
(2) Text-vs-source reconciliation (`check_text_metric_reconciliation`) is **curated-anchor
based**, not a blanket float scan.
**Rationale:** The old default (`results/`) silently no-opped on a fresh clone where only the
frozen dir is committed. A blanket float scan over a 6,000-word manuscript is unreliable (false
positives on legitimately-shared literals such as the per-region LORO values 0.623 and 0.951);
curated anchors plus the table-vs-JSON and stale-file checks cover the headline numbers reliably.
**Rebuttal:** If a reviewer asks why reconciliation is not exhaustive over all numbers, the answer
is that exhaustiveness was traded for reliability (no false positives), with anchors covering every
abstract and headline metric and deterministic table-vs-JSON checks covering the tables.
**Reopen:** only on external editor input.

## v2 loop-close decision batch (2026-07-03; options selected by the assistant under the
## maintainer's delegation "pick the most robust defensible choices", which constitutes
## ratification per FINAL_FIX_CONTRACT_v2)

### v2-D1 (title/abstract calibration; reopens DECISIONS M1 on its named trigger: external editor input)
**Lane:** DECIDED. **Call:** Title comparative softened "more than" -> "as much as" (adopting the
de novo reviewer's own §1 paraphrase); abstract "dominates reported skill" -> "is the largest
single learnable signal"; the T1 environmental-signal survival is stated via the in-region
provenance-free AUROC in the abstract's transport sentence and at the discussion echo
(skeleton "encode who is monitored..."). PF numbers remain framed as upper bounds with ICP as
the stricter estimate (M4).
**Rationale:** "As much as" is supported across tasks (T1 delta full-vs-PF; T4; SHAP dominance)
without over-claiming on T1, and is the reviewer's own summary phrasing — the maximally
defensible calibration. **Reopen:** external editor input.

### v2-D2 (article type, m12)
**Lane:** DECIDED. **Call:** Submit as an Analysis-type contribution (benchmark + methodological
finding); cover letter mentions the fit. No manuscript-body change.

### v2-D3 (orphaned fig5, 9-iii)
**Lane:** DECIDED. **Call:** Remove fig5_national_risk_map from paper/figures and from
generate_figures' default paper outputs; the national risk map remains a webapp asset
(/deploy-map). No orphaned submission artifacts.

### v2-D4 (SDWIS reader-facing rendering, M7; executes the R1 evidence chain)
**Lane:** DECIDED. **Call:** Reader-facing SDWIS count = 916,899 (the keep-parse the canonical
archive is computed on) everywhere, gated via ds_sdwis. The Methods sentence documents the
zero-measure-rows-as-non-detects convention and names the legacy 694,419 parse as a
provenance fact (explicitly whitelisted numeral). The dataset_snapshot _provenance note is
corrected (it falsely claimed the frozen analyses used the 694,419 parse — stale M5b-era text).
**Reopen:** external editor input.

### v2-D5 (model-family count, M7)
**Lane:** DECIDED. **Call:** Canonical rendering = "20 model families (29 model classes)" —
the recorded census (15/23 at PR#25, +3 T2 families, +ICP, +GNNx2) and README's deliberate
count agree. Manuscript "sixteen" and .zenodo.json "15 (23)" are aligned to it; the concept
registry treats sixteen/15 renderings as conflicts.

### v2-D6 (headline systems count + labeling, M7)
**Lane:** DECIDED. **Call:** Keep 95,223 as the headline count, relabeled honestly as
"water systems and ambient monitoring locations" (the reviewer's complaint was the PWS label,
not the number; the SI already states WQP/CA GeoTracker are not PWS). The PWS-only ~91,061 is
NOT introduced (no frozen source; would rebase nothing while touching everything); the
pws_only_count concept row is removed.

### v2-D7 (verify_paper warning ratification)
**Lane:** DECIDED. **Call:** The four legitimately-persistent warnings are ratified into
allowed_warnings.txt verbatim (exact-normalized, counted): split-comparison 0.813
cross-reference (checker set-membership conservatism; 0.813 is the gated CA external AUROC),
CA GeoTracker DeLong p<0.05 (the narrative already says CA performs above chance; the checker
expectation is the outdated part, kept as a tripwire), conformal 0.910 vs 0.950 and Region-9
0.878 (true, disclosed properties of the method).

### v2-D8 (pre-fix 0.962 provenance record home)
**Lane:** DECIDED. **Call:** The retained-stale-files mechanism (M5b) is superseded by the v2
single-run archive invariant; the pre-fix "before" artifacts are preserved as repository
history, cited at commit 9cf4d56 in the SI next to the historical value (which carries an
explicit number_whitelist.tsv entry). CHANGELOG gains a [4.0.0] entry describing the current
dataset scale and the v2 pass (the merge of this branch is the release act; tagging via
/release).

## R5 loop-close decision batch (2026-07-05; absorbing the independent Nature Water referee
## report `paper/referee_report_naturewater_independent_2026-07-05.md`, FINAL_FIX_CONTRACT_v3)

### R5-D1 (honest reframe; the pivotal revision — M9 + M2)
**Lane:** DECIDED (PRE-APPROVED by the maintainer, 2026-07-05, in the kickoff that authored
this pass — `.claude/prompts/r5-kickoff-2026-07-05.md`; the direction is not re-opened).
**Call:** Adopt the honest reframe. Lead the abstract and contribution list with the two
genuine, defensible advances: (1) the rigorous quantification and decomposition of a known
ascertainment confound in high-stakes US regulatory drinking-water ML, landing as the first
enforceable PFAS MCLs take effect; and (2) the monitoring-inequity environmental-justice
finding. Cite the presence-only / sample-selection-bias antecedents (Phillips et al. 2009;
Fithian et al. 2015) and the epidemiological surveillance-bias literature. Drop the
"first / largest-learnable-signal / ImageNet-GLUE-GEO-Bench" over-claims. Describe the
benchmark honestly as a reproducible two-confound protocol with one converged detection task
(T1) plus documented negative (T2) and boundary (T3/T4/T5/T7) tasks, not a populated
multi-task leaderboard. State that no new learning method is introduced. The reframe
propagates to every pitch surface (cover letter, plain-language primer, reporting summary,
.zenodo.json, README).
**Rationale:** R5 (confidential comments) makes the reframe the gating condition for a
"lean accept": the residual novelty envelope, after the conceded spatial-leakage prior art
and the uncited ascertainment antecedent, is exactly these two advances plus the
reproducibility apparatus. Leading with them matches the evidence; the honesty already
present in the Results is simply carried into the abstract.
**Rebuttal:** If a reviewer says the confound is unoriginal, the answer is that we explicitly
credit the presence-only/surveillance-bias antecedents and claim only the domain
quantification/decomposition at a moment of acute regulatory relevance, plus the equity
finding — not conceptual novelty. **Reopen:** external editor input only.

### R5-D2 (data-availability strategy — M8)
**Lane:** DECIDED. **Call:** Implement the fallback NOW so submission is not blocked on the
Minnesota MDH reply: label MN-dependent results as non-redistributable corroboration with a
prose argument that the thesis stands without them (the main-line results are MN-independent;
the MN ascertainment "flip" is corroboration only); add the MDH row to the Reporting Summary
and correct "All data sources are publicly available"; document the NJ DEP legal basis (public
state records, non-self-serve access route); reconcile the Zenodo deposit license (CC BY 4.0
governs the data compilation; MIT stays on the code). Draft `paper/mdh_permission_request.md`
for the maintainer to send. Whether to wait for MDH's reply before submitting is a maintainer
call at submission time, not a blocker for this pass.
**Rationale:** Nature's data-availability standard requires either redistribution permission or
a clearly-labelled non-reproducible corroboration path plus a demonstration the thesis survives
without it. The fallback satisfies the standard immediately; the permission request pursues the
stronger outcome in parallel. **Reopen:** external editor input only.

### R5-D3 (leaderboard scope — M3 + m5)
**Lane:** DECIDED (maintainer bought the full run, 2026-07-05 decision batch Q2).
**Call:** Run the full 20-family Leave-One-Region-Out cross-validation on T1
(`paper/compute_loro_full.py`, reproduction-gated on the three committed tree-model LORO
values) and **re-rank Table 2 by LORO** for all families — fully resolving M3 rather than
merely demoting the table. State the West-only (EPA Regions 8/9/10) test composition; report
the CatBoost-vs-XGBoost LORO leader relationship; replace or caption the i.i.d. bootstrap CIs
as anti-conservative given the documented residual spatial autocorrelation (Moran's I up to
~0.67). The out-of-region ICP number that m5 asks for is produced by the same run (ICP is one
of the families).
**Rationale:** The referee's first-choice remedy is to rank the leaderboard by LORO for all
families; the maintainer authorized the compute, so we do the stronger thing. Per the
analysis-over-acknowledgement preference, running the analysis beats softening the language.
**Rebuttal:** If a reviewer questions the single fixed split, the answer is that Table 2 is now
ranked by leave-one-region-out across all families, with the West-only composition and
spatial-autocorrelation caveats stated. **Reopen:** external editor input only.

**R5-D3 refinement (leaderboard wiring; ratified 2026-07-06).** `loro_cv_full.json` is wired as a
**LORO-ranked T1 leaderboard in the main text** (the flagship benchmark display and the adoptable
resource), with the fixed geographic split shown alongside as a transparent secondary. Call: (1)
re-rank the T1 family table by LORO mean AUROC (primary, cluster-robust CI), keeping the fixed-split
AUROC as a secondary column with its i.i.d. bootstrap CI captioned anti-conservative (Moran's I
~0.67); prefer re-ranking the existing main-text Table 2, with the exhaustive 20-family detail
allowed to fall to Extended Data only if display budget forces it, but the LORO ranking + the
CatBoost-leads-0.790 / XGBoost-third-0.783 flip must appear in the main text; (2) state that the top
tree/ensemble models are statistically comparable at the top (per C16), so the value is the honest
protocol, not a single winner (adopter takeaway: tree ensembles are the baseline to beat); (3) flag
non-convergence via `gate_lib.flag_degenerate_metric_cells` (gnn_gcn/gnn_sage failed; TabPFN 2/10
folds), never as numbers — this also discharges the M2 degenerate-cell audit; (4) make
`LEADERBOARD.md` the canonical LORO-ranked leaderboard with a short "Submit a model" protocol (run a
model through the `_run_loro_cv` / `compute_loro_full.py` harness, report LORO mean AUROC + cluster
CI, open a PR). Rationale: it is the referee's first-choice remedy, thesis-aligned (LORO resists the
spatial leakage the paper indicts), gives the leaderboard one clearly-defined credible primary
protocol (the prerequisite for third-party adoption), and honesty about the top-of-table tie is what
makes a leaderboard trustworthy — turning M3's critique into a demonstration of the benchmark's
rigor. Reopen: external editor input only.

### R5-D4 (wording reconciliation — M5 + m4)
**Lane:** DECIDED. **Call:** Apply two terminology changes globally IN PROSE AND LABELS ONLY:
"honest estimate" → "upper-bound estimate" (the 0.691 transportable-signal figure); and
"provenance-free" → "provenance-reduced" (source remains recoverable at AUROC 0.653 from the
reduced feature set, so "free" over-claims). **Rename guard:** frozen JSON filenames, JSON
keys, `compute_*.py` script names, and `<!--pn:id-->` marker ids are NEVER renamed — that would
churn `checksums.sha256` and the DERIVATIONS stamps for zero reader value.
**Rationale:** Both are honest-labelling corrections the referee explicitly frames as minor;
the guard keeps the tamper-evident archive stable. **Rebuttal:** If a reviewer notes source is
still recoverable, the answer is that S17 bounds it (0.653) and the manuscript now says
"provenance-reduced" and "upper-bound estimate" throughout. **Reopen:** external editor input only.

## R6 decision batch (2026-07-06; absorbing the Nature Water `naturewater_panel_2026-07-06`
## panel, `/paper-triage` one tiered pass; three calls PRE-APPROVED by the maintainer in the
## kickoff that authored this pass, `.claude/prompts/r6-kickoff-2026-07-06.md` — the record)

### R6-D1 (emphasis reframe — editor A1 / referee R1-2; PRE-APPROVED 2026-07-06)
**Lane:** DECIDED (PRE-APPROVED by the maintainer, 2026-07-06, in the kickoff that authored this
pass; the direction is not re-opened — only wording was approved). **Call:** Lead the **T1 Results
prose** with the leakage-resistant leave-one-region-out (LORO) estimate (mean AUROC 0.782; CatBoost
0.790, XGBoost 0.782) and the transportable provenance-reduced out-of-region signal (0.691), and
frame the single fixed West-only 0.864 as the disclosed **optimistic upper end** — naming that its
test set contains region 10, the highest-prevalence (detection rate 0.433) and most separable
(held-out AUROC 0.971) region. The 0.864 **value**, its five-seed stability annotation (`clm_seed_stable`),
and the LORO-ranked Table 2 (`clm_loro_ranked`) are **untouched**: this changes lead ORDER only, and
rewords no `CLAIMS.tsv`-pinned sentence (the reworded sentence at skeleton.md ~:216 is not pinned).
The **abstract needs no change** — it already leads T1 with the honest survived-signal framing
(in-region 0.785 / out-of-region 0.691) and contains no "0.864" token.
**Rationale:** A paper whose thesis is that headline numbers are inflated by leakage should not lead
its own T1 narrative with 0.864, the maximum of its own estimates from its most favourable
configuration. Leading with the leakage-resistant numbers matches the evidence and the pitch surfaces
(cover letter, primer, reporting summary already lead with 0.782→0.691). **Rebuttal:** If a reviewer
says 0.864 is buried, the answer is that it is fully reported as the disclosed optimistic upper end
with its region-10 provenance and seed-stability, and Table 2 ranks by LORO. **Reopen:** external
editor input only.

### R6-D2 (venue stand + deployment framing — editor A3 / referee R6-5; PRE-APPROVED 2026-07-06)
**Lane:** DECIDED. **Call:** The venue **stands: Nature Water (Analysis).** R6-5/A3 is an S3
editorial/venue-fit judgment, not an over-claim (the manuscript already states no new method is
introduced). Add **one explicit sentence** stating the contribution is diagnostic/methodological and
that **no deployment or predictive advance is claimed**, given the modest, regionally heterogeneous
transportable signal (out-of-region AUROC ~0.691, near-chance in roughly half of regions). Only the
maintainer can retarget the venue.
**Rationale:** The out-of-region signal is honestly reported as modest; making the diagnostic framing
explicit forecloses the "no deployment win" reading without conceding the venue. A rigorous
field-correcting cautionary result with acute regulatory timing (April-2024 PFAS MCLs) is a
defensible Analysis. **Rebuttal:** If a reviewer asks for a predictive advance, the answer is that
the paper's stated contribution is the two-confound decomposition + monitoring-inequity finding +
reproducibility apparatus, explicitly not a better predictor. **Reopen:** external editor input only
(venue retarget is a maintainer call).

### R6-D3 (banked-residue record — the terminal disposition for "not worth a manuscript edit")
**Lane:** DECIDED. **Call:** R6 findings that do not warrant a manuscript edit are **terminal via a
`RESPONSE_BANK.md` paragraph (BANKED)** — a full disposition, not a deferral (the response bank is the
rehearsal for the journal's real referees). This records the banked classes: (a) **frozen-internal
oddities never touched** — R6-R1-4 (`seed_inflation` is_inflated cross-harness artifact), R6-R1-6 ≡
R6-R2-6 (`power_analysis` dead `loro_tests` block), R6-R2-7 (130-vs-131 feature count), R6-R4-2
(linguistic-isolation Inf/p) — the JSONs are never edited (no re-freeze); (b) **dedups of
already-closed items** — R6-R6-2 (benchmark scope, vs R5-D1), R6-R6-4 (EJ burden, Liddie = ref 47),
R6-R2-4 (Rosenbaum Γ* heuristic, vs M8d); (c) **four verified-true confirmations** — R6-R1-3, R6-R2-3,
R6-R3-Lead3, R6-R3-Lead4 (recorded as response-letter ammunition); and (d) the **re-censored-base-rate
subpart of R6-R3-1** — that value is not in the frozen archive (`common_rl_sensitivity.json` records
AUPRC 0.102 and n_test 3,780 but no re-censored positive prevalence), and deriving it would require a
re-run this pass forbids, so it is banked with a named question while R6-R3-1's soften-"robust" subpart
lands as prose. **Rationale:** BANKED is a real terminal state (verify_ledger witnesses the `### <id>`
heading); "not worth an edit" must itself be a provable disposition so the pass terminates.
**Reopen:** external editor input only.

## Issue #55 — pre-publication "fix everything" pass (2026-07-11)

The following decisions were pre-ratified by the approved plan
`.claude/plans/wise-cuddling-popcorn.md` (that approval is the per-item maintainer
ratification CLAUDE.md requires for each COMPUTE and each additive/corrective frozen-archive
edit in this pass). Tracked in GitHub issue #55; these are the durable rationale records.
None reopened the manuscript on an S0 basis — all were must-fix-before-submission
packaging/citation/figure defects or additive computes.

### issue55-D1 (GNN/TabPFN LORO completion, items 0/1)
**Lane:** DECIDED (COMPUTE, reproduction-gated). **Call:** Rather than a truthful relabel of the
false "failed (CUDA)" cells, thread coordinates into the LORO driver and run gnn_gcn/gnn_sage +
the 8 rejected TabPFN folds to a full 10/10 (`paper/compute_loro_gnn_tabpfn.py`), splicing
verbatim-preserving into frozen `loro_cv_full.json`. Result: GCN 0.683, SAGE 0.652 (honest
unconditional linear-fallback, bottom of table), TabPFN 0.758; CatBoost still leads LORO at
0.790, so the leader narrative holds. TabPFN methods disclose the two-stage 3,000-sample CPU
protocol; ref split into TabPFN v1 (ICLR 2023) + v2 (Nature 2025). **Rationale:** real numbers
close the "why didn't you run them?" question a relabel invites. **Reopen:** env-drift fallback F1
(label-only truth) — not triggered; reproduction gate passed.

### issue55-D2 (national risk map kept + frozen, items 3/6/8)
**Lane:** DECIDED. **Call:** Keep the supplementary national risk map (do not drop it as an
orphan). Freeze a slim single-task T1-PFOS surface `predictions_slim.json` (15,268 systems,
byte-exact reproduction gate in `paper/build_predictions_slim.py`, F-pred provenance from the
2026-06-29 inference parquets), point the figure at it, declare it as Supplementary Fig. 9, and
retitle the honest "AUROC/AUPRC by model" figure (curve arrays are deliberately not frozen).
**Rationale:** the map is a genuine deliverable and the webapp's visual; a single-task frozen
surface makes the "PFAS" title truthful and gate-reproducible. **Reopen:** external editor input.

### issue55-D3 (predictions included in the Zenodo deposit, item 3)
**Lane:** DECIDED. **Call:** Code Availability now states the per-system prediction export (git-
ignored in-repo) ships in the Zenodo archive alongside the harmonized dataset and full figure
arrays, and documents the `aquacontam webapp` entry point. The dead "(Supplementary Box 1)"
webapp pointer is repointed to "(Code Availability)". **Reopen:** external editor input.

### issue55-D4 (reference renumber to Nature first-appearance order, item 5)
**Lane:** DECIDED. **Call:** Renumber the whole bibliography to order-of-first-appearance (60
refs) via the scripted, self-validating `paper/renumber_references.py` (dry-run diff reviewed
before apply), remapping all 48 markers as comma-lists; add the new `check_bibliography_bijection`
verify_paper linter (auto-gated via G2) so the numbering cannot silently regress. **Rationale:**
strict Nature style requires ascending first-appearance; the scripted remap + bijection linter is
the safety net. **Reopen:** external editor input (a grouped-list article type would tolerate the
prior thematic numbering).

### issue55-D5 (ref 16 DWINSA rehome, item 5)
**Lane:** DECIDED. **Call:** The DWINSA infrastructure-needs survey (ref 16, corrected to doc#
EPA 810-R-23-001) does not support the SDWIS lead/copper monitoring-records sentence (ref 17
alone does); drop it there and rehome it to the "more than 150,000 public water systems" sentence
it actually supports. **Reopen:** external editor input.

### issue55-D6 (NJ acquisition wording harmonized, item 4)
**Lane:** DECIDED. **Call:** Describe the NJ DEP WaterViewer collection identically across the
manuscript, the scraper docstring, and `docs/data_requests/`: a headed-browser scraper reading the
same public pages a browser would, under the NJ Open Public Records Act, with no authentication or
access circumvention. The ranked CAPTCHA-workaround playbook is neutralized; `docs/data_requests/`
is de-personalized to office-level contacts; "correspondence templates" -> "request/response
records". **Reopen:** external editor input.

### issue55-D7 (git-history PII — SUPERSEDED: cleanup performed 2026-07-11)
**Lane:** DECIDED (superseded). **Original call:** accept a history residual (agency-staff
contact details in the `docs/data_requests/` history) because rewriting main's history was
forbidden by repo etiquette. **Superseded:** the maintainer authorized a one-time cleanup of
this private repository's history, which is retained only as the internal archive and is
never published; the public release is produced as a fresh-history copy (see the maintainer
release runbook). Correspondence records were consolidated to the agencies' office-level
public-records mailboxes (`openrecs@tceq.texas.gov`, `publicrecords@deq.nc.gov`), which are
correctly retained. **Verification:** the completeness sweep leaves only office mailboxes;
the frozen archive tree is byte-identical (blob hash = checksum); the deterministic gate is
GREEN. **Rationale:** the contacts are low-sensitivity public-agency addresses, but a clean
record is preferable before any public copy is produced. **Reopen:** none (terminal).

### issue55-D-followup (typing / ICP-durable / test-isolation fixes, 2026-07-11)
**Lane:** DECIDED. **Call:** The #55 follow-up pass fixed at the root the items the first pass had
substituted or worked around. (a) **Build hygiene:** the CI typecheck red was mypy 2.x (not
pandas-stubs) tightening `no-any-return` on pandas constructors; fixed the 64 flagged returns with
the repo's `cast(...)` idiom and relaxed the exact `mypy==1.20.2` pin to bounded ranges
(`mypy>=1.8,<3`, `pandas-stubs>=2.0,<4`) — honest reproducibility rails, not a mask. (b) **Item 7
durable fix:** ICP history is now persisted through the canonical `train_and_evaluate`→
`export_icp_diagnostics` path (3-tuple `icp_models` channel), the out-of-band `regen_review_figures`
shell-out is removed, and `icp_diagnostics.json` was re-frozen from a canonical T1 run that
reproduces the frozen ICP T1 AUROC exactly (0.7311), so ED Fig 4a panel a is the training trace of
the panel-b model (truthful `cuda:0` metadata). (c) **Test isolation:** the 6 full-suite
torch-reproducibility failures were local-GPU-only cuBLAS-handle pollution; the (device-independent)
determinism checks now run on CPU, matching CI. The figure-write test leak is closed by the
shell-out removal. **Reopen:** none (terminal; gate + CI green, PR #56 merged).

### issue55-D8 (mis-geocode handled by sensitivity + figure exclusion, not coordinate edits, item 6)
**Lane:** DECIDED (COMPUTE, reproduction-gated). **Call:** The 6,889 wrong-ZIP region-mismatched
systems are NOT corrected in the harmonized dataset (that would invalidate the frozen world).
Instead: exclude them from the split map (caption reports the count), and quantify the modeling
impact with `paper/compute_misgeocode_sensitivity.py` (frozen `misgeocode_sensitivity.json`) —
excluding them end-to-end moves T1 0.864->0.867 and T4 0.699->0.691, both inside the pre-registered
0.02 escalation band, so no S0. **Rationale:** the split is prefix-derived and uncontaminated; the
exposure is wrong-place features, bounded and shown immaterial. **Reopen:** F2 (material
degradation) — not triggered.

### minor-featurecount (R4 referee m6 — DML feature-count reconciliation)
**Lane:** DECIDED. **Call:** Reconciled to the frozen archive — `causal_deconfounding.json` has 130
coefficients (not the referee's 127), 69 FDR-significant. Prose reads "69 of 130", gated via
`pn:dml_sig_count` / `pn:dml_total` and machine-checked by G1/G3 (no brittle phrase anchor).
**Rationale:** the frozen archive is authoritative; gating the counts removes the drift risk.
**Reopen:** external editor input. *(Section added 2026-07-11 to clear the standing verify_ledger
witness warning; the machine-state row already recorded this DECIDED disposition.)*

### edfig1-D1 (mis-geocode flag rule upgraded to exact state polygons, ED Fig. 1 fix)
**Lane:** DECIDED (COMPUTE, reproduction-gated; maintainer-directed 2026-07-12, pre-submission).
**Call:** The issue55-D8 bounding-box flag rule is superseded for the ED Fig. 1 exclusion and
Supplementary S7b by an exact state-polygon rule (`paper/_geo_flags.py` against
`paper/assets/us_states_cb2023_5m.geojson`, US Census 1:5M, public domain; nearest state within
3.0 degrees for water-jittered points). The loose overlapping first-match boxes falsely flagged
~64% of their 6,889 (correctly-located border systems, e.g. St. Louis MO caught by Illinois's
rectangle), carving visible bands of dropped points along region borders in the split map. The
v2 rule flags 1,818 of 87,450; the figure keeps the rescued 5,183 border systems and still drops
true wrong-ZIP placements, so no wrong-color dots appear in foreign regions. Coordinates are
still never edited, and the pipeline-side bbox fallback in `assign_epa_region` is untouched
(benchmark split assignment unchanged). Quantified by `paper/compute_misgeocode_sensitivity_v2.py`
(frozen `misgeocode_sensitivity_v2.json`, additive; v1 retained): dual reproduction gates (A:
unperturbed T1/T4 xgboost reproduce frozen `results.json` AUROC exactly; B: the retained legacy
rule reproduces the v1 6,889/87,450 on the same frame), exclusion moves T1 0.864->0.866 and
T4 0.699->0.697, both inside the pre-registered 0.02 escalation band, so no S0. The figure's
renderer is pinned to matplotlib/cartopy (every committed version carries Matplotlib Creator
metadata; the newly-importable pygmt path would silently re-style the map). **Rationale:** the
exclusion idea was right, the geometry was wrong; exact polygons make the caption count mean
what it says while preserving the frozen world. **Reopen:** F2 (material degradation) — not
triggered; or a decidable S0 against the v2 artifact.
