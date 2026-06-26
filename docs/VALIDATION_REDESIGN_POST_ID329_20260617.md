# Validation redesign after ID329

Created: `2026-06-17 01:21:02` KST

## Public facts this validation must explain

| id | local CV | Public LB | role |
|---:|---:|---:|---|
| 178 | 0.538638 | 0.6014113009 | best confirmed recent public anchor |
| 287 | 0.378934 | 0.6698787318 | failed local-frontier direction |
| 329 | 0.516718 | 0.6017908577 | guarded practical candidate; safe but no public improvement |

The immediate target is not another Public `0.55` leap. The next credible target is a reproducible Public `< 0.60` signal, because ID329 proved that a guarded local improvement can still remain at the ID178 public plateau.

## Hard constraints before any new submission

- No same-family ID330/ID331 constrained-combo submission without a new decision note explaining why ID329 feedback does not apply.
- No local-frontier candidate may be promoted solely because local CV improves; ID287 makes that invalid.
- Every candidate must report distance to ID178, distance to ID329, projection to ID287 failed direction, local CV, and artifact family.
- A candidate must be uniquely named and matched across submission CSV, notebook, submission_log, metadata JSON, and base predictions before review.

## Redesigned validation layers

### Layer 0: artifact eligibility

Reject candidates whose provenance or file contract is unclear before considering score.

- unique immutable filenames
- base_oof/base_test present
- notebook/log/report present
- not blacklisted/risk-labelled unless explicitly reviewed

### Layer 1: strict local validation

Use only fold-safe OOF from subject-aware splits; no full-OOF residual recombination as proof.

- subject-hole split
- no same subject leakage
- feature preprocessing fit inside folds
- per-target logloss reported

### Layer 2: public-anchor geometry gate

Screen candidate movement using known public outcomes without pretending this is ground truth.

- mean/max delta vs ID178
- mean/max delta vs ID329
- projection to ID287 failed vector
- projection to ID329 tested-neutral vector
- family label

### Layer 3: anchor-consistency audit

A validation proxy is not trusted unless it qualitatively explains ID178 ~= ID329 << ID287 on Public.

- ID287 receives severe penalty
- ID329 same-family local gain is discounted
- ID178 remains a valid near-0.601 anchor

### Layer 4: pre-registered submission decision

Allow only one public probe after the above layers produce a specific reason to expect Public < 0.60.

- decision note
- exact SUB_PATH/SUB_MEMO
- explicit user approval
- no auto-submit

## Next experiments — no DACON submission

### V1: Demographic/static-feature strict CV audit

Rationale: Untracked demographic features from the ETRI paper may be useful, but they must be evaluated under strict subject-hole validation before any public use.

Success criteria: Improves strict fold-safe CV by target without increasing anchor-geometry risk; produces OOF/test artifacts only after review.

### V2: Public-anchor geometry reranker

Rationale: Existing candidate artifacts can be re-ranked using ID178/ID329 anchors and ID287 failed vector without making new predictions.

Success criteria: Produces a short ranked list where ID330/ID331 are not automatically promoted and at least one candidate has a distinct family rationale.

### V3: Source/report promotion audit

Rationale: The repo has 84 untracked scripts and 600+ bulk reports; authoritative artifacts must be separated from scratch outputs.

Success criteria: A commit/branch-ready list of files to promote, archive, or delete; no modeling side effects.

## Stop rule

Stop same-family submissions after one negative public feedback event unless a new validation layer predicts the observed failure/non-improvement.

## Related artifacts

- `docs/REPO_TRIAGE_POST_ID329_20260617.md`
- `reports/repo_triage_post_id329_20260617.json`
- `reports/public_anchor_geometry_post_id329_20260617.json`
- `reports/public_anchor_geometry_post_id329_20260617.md`

## Follow-up V2 rerank result

A no-submission public-anchor rerank was created:

- `reports/post_id329_public_anchor_rerank_20260617.json`
- `reports/post_id329_public_anchor_rerank_20260617.md`

Result: no existing low-movement candidate is a credible Public `< 0.60` submission candidate under the conservative ID178/ID329/ID287 screen. The top reviewable candidates are useful as study objects, not as immediate submissions. Same-family constrained-combo candidates remain frozen after ID329.

## Follow-up V1 demographic Kaggle audit result

A Kaggle report-only full audit was completed:

- `reports/strict_demographic_feature_audit_kaggle_20260617.json`
- `reports/strict_demographic_feature_audit_kaggle_20260617.md`
- `reports/strict_demographic_feature_audit_kaggle_review_20260617.json`
- `reports/strict_demographic_feature_audit_kaggle_review_20260617.md`

Result: demographic/static features are not Public `0.58`-plausible as a global overlay. They improve the subject-hole diagnostic (`compact_plus_demo__extratrees_d5_l8`, `0.616853`) but fail to dominate the stricter leave-one-subject-out stress split (`compact_covariates__logreg_l2_c020`, `0.669770`). Treat demographic features as target-specific hypotheses only; do not promote a CSV/notebook/submission-log candidate from this branch without a new fold-safe target gate and anchor-geometry review.

## Follow-up test split alignment diagnostic

A no-model split-shape diagnostic was created:

- `reports/test_split_fold_alignment_post_id329_20260617.json`
- `reports/test_split_fold_alignment_post_id329_20260617.md`

Result: the hidden test rows are structurally not a pure future holdout. Most subjects have two test blocks (middle and late), while `id04`, `id05`, `id06`, and `id08` have more fragmented later test rows. The diagnostic-only test-boundary neighbor holdout best matches the actual sample-row date geometry by relative-position histogram, but it must be treated as a stress test rather than a new optimization leaderboard. Future candidates should pass subject-hole, test-geometry stress, and LOSO-style robustness before anchor-geometry review.

## Follow-up OOF test-geometry stress result

A no-model scan of existing base OOF artifacts was created:

- `reports/oof_test_geometry_stress_post_id329_20260617.json`
- `reports/oof_test_geometry_stress_post_id329_20260617.md`

Result: post-hoc OOF subset losses must not be used as a promotion gate. ID287 still ranks near the top by boundary/band OOF losses even though its confirmed Public score collapsed to `0.6698787318`. Any future test-geometry stress gate must be fold-nested: candidate selection, calibration, and blending must be learned without the outer validation rows.

## Follow-up ID178 sequence robustness and rollback diagnostics

Two report-only diagnostics were created:

- `reports/random_conv_sequence_robustness_post_id329_20260617.json`
- `reports/random_conv_sequence_robustness_post_id329_20260617.md`
- `reports/id178_sequence_target_rollback_post_id329_20260617.json`
- `reports/id178_sequence_target_rollback_post_id329_20260617.md`

Result: the ID178 random-conv sequence overlay has small but consistent positive signal under subject-hole, LOSO, and test-boundary stress. Target-level stress, however, suggests S1 and S4 are fragile. A hypothetical `rollback_s1_s4` variant improves stress consistency versus the original ID178 overlay while staying close to ID178 geometry. This is a Public `<0.60` probe hypothesis only, not a Public `0.58`-plausible candidate and not a promoted artifact.

## Follow-up ID334 artifact preparation

The `rollback_s1_s4` hypothesis was promoted into a full generated-unsubmitted
artifact contract as ID334:

- `submissions/submission_id_334_id178_rollback_s1_s4_sequence_stressguard_cv0.540347_unsubmitted.csv`
- `notebooks/history/submission_334_id178_rollback_s1_s4_sequence_stressguard_cv0.540347_unsubmitted.ipynb`
- `reports/experiment_334_id178_rollback_s1_s4_sequence_stressguard_cv0.540347_unsubmitted.json`
- `reports/decision_334_id178_rollback_s1_s4_sequence_stressguard_cv0.540347_unsubmitted.md`
- `artifacts/base_predictions/base_oof_id_334_id178_rollback_s1_s4_sequence_stressguard_cv0.540347.csv`
- `artifacts/base_predictions/base_test_id_334_id178_rollback_s1_s4_sequence_stressguard_cv0.540347.csv`

No DACON submission was made. This artifact exists so the next public-probe
decision can be made from a clean immutable contract rather than from loose CSVs.

## Follow-up ID335 alpha refinement artifact preparation

Two report-only diagnostics were added:

- `reports/id178_sequence_alpha_scan_post_id334_20260617.json/.md`
- `reports/id178_sequence_alpha_refine_post_id334_20260617.json/.md`

The targeted refinement kept ID178 sequence overlay for `Q1/Q2/Q3/S2`, kept
`S1/S3` at ID173, and restored half of the `S4` overlay. It was promoted into a
full generated-unsubmitted artifact contract as ID335:

- `submissions/submission_id_335_id178_alpha_s4half_stressguard_cv0.539690_unsubmitted.csv`
- `notebooks/history/submission_335_id178_alpha_s4half_stressguard_cv0.539690_unsubmitted.ipynb`
- `reports/experiment_335_id178_alpha_s4half_stressguard_cv0.539690_unsubmitted.json`
- `reports/decision_335_id178_alpha_s4half_stressguard_cv0.539690_unsubmitted.md`
- `artifacts/base_predictions/base_oof_id_335_id178_alpha_s4half_stressguard_cv0.539690.csv`
- `artifacts/base_predictions/base_test_id_335_id178_alpha_s4half_stressguard_cv0.539690.csv`

No DACON submission was made. ID335 is a weak controlled-probe hypothesis only,
not Public `0.58` evidence.

## Follow-up target-specific demographic gate

A no-training gate report was created:

- `reports/target_specific_demographic_gate_post_id335_20260617.json/.md`

Allowed future demographic/static research targets are `Q2`, `S2`, and `S4`.
These are the only targets where demographic features beat no-demo alternatives
under both subject-hole and LOSO stress in the completed Kaggle audit. `Q3` is
watch-only; `Q1`, `S1`, and `S3` are rejected for now. This gate creates no
submission artifact and is not promotion evidence by itself.

## Follow-up demographic overlay rejection and post-ID335 rerank

Two report-only diagnostics were added:

- `reports/demographic_target_overlay_post_id335_20260617.json/.md`
- `reports/post_id335_public_anchor_rerank_20260617.json/.md`

The demographic overlay scan found no guarded positive alpha for `Q2`, `S2`, or
`S4` on top of ID335, so no candidate was promoted. The post-ID335 rerank
refreshes the candidate map with ID334/ID335 included. It does not authorize a
submission; the cleanest non-failed-family manual review target remains ID179,
while failed-direction/orthogonal rows should stay frozen unless they explain
ID329's public non-improvement.

## Follow-up ID179 versus ID335 review

A report-only review was created:

- `reports/id179_vs_id335_sequence_probe_review_20260617.json/.md`

ID179 is locally stronger and closer to ID178 than ID335, with a slightly better
slice-robust score. This makes ID179 the cleaner same-family manual-review
probe if the user explicitly asks for a cautious public check. It is still
non-strict, OOF-selected, and not Public `0.58`-plausible.

## Follow-up failed-direction candidate audit

A report-only audit was created:

- `reports/failed_direction_candidates_post_id335_20260617.json/.md`

The rerank-top failed-direction rows are not promoted. ID295/296/297 are
anti-failed-vector diagnostics only because they are worse than ID179. ID304,
ID305, ID306, ID310, and ID311 show large local gains per small test movement,
but that is precisely the non-nested full-OOF efficiency pattern that ID329
failed to convert into Public improvement. ID307 remains information-only. A
real promotion from this family requires a fold-nested validation path.

## Follow-up candidate outer-selector diagnostic

A selection-nested report over existing candidate OOF artifacts was created:

- `reports/candidate_outer_selector_post_id335_20260617.json/.md`

It selected mostly ID310/ID311 and reached local `0.521786` under subject-hole
choice nesting and `0.522195` under LOSO choice nesting. This confirms the
failed-direction artifacts remain locally attractive, but it does not solve the
underlying problem: the artifacts themselves were generated non-nested/full-OOF.
Treat the result as a blueprint for a true fold-nested rebuild, not as a
submission gate.

## Follow-up fold-internal failed-direction rebuild diagnostic

A stricter report-only diagnostic was created:

- `reports/fold_internal_failed_direction_rebuild_post_id335_20260617.json/.md`

It recomputed beta/projection/source choices inside each outer fold, while still
using existing source OOF artifacts. The best subject-hole row was
`id310_like_cap120_lam160` with local `0.522506` and robust `0.548380`; the
best LOSO row used the same setting with local `0.523274` and robust
`0.549054`.

This is a useful signal: the failed-direction idea is not purely a global
selector artifact. It is still not a submission gate, because the underlying
source predictions were already generated outside the outer-fold replay. The
next credible path is a true model-level fold-nested rebuild of the selected
source families, especially IDs `242`, `243`, `249`, `252`, `268`, `276`, and
`278`.

## Follow-up true nested rebuild blueprint

A no-submit blueprint was created:

- `reports/true_nested_rebuild_blueprint_post_id335_20260617.json/.md`

It confirms that the fold-internal selected sources are all non-nested. The
required rebuild should be split into three lanes: target-wise delta-capped
bridge selection (`249`), target-wise pair/logit blend selection (`268`, `276`,
`278`), and second-stage calibration (`242`, `243`, `252`). None of these lanes
can create a leaderboard-facing artifact until calibration, selection, and
scoring all happen inside fixed outer folds.

## Follow-up fold-internal pair/logit blend rebuild diagnostic

A stricter no-submit diagnostic was created:

- `reports/fold_internal_pair_blend_rebuild_post_id335_20260617.json/.md`

It rebuilt pair/logit blend choices inside each outer fold. The best row reached
`0.399920` subject-hole local and `0.400168` LOSO local, but it relied heavily
on ID252/ID255 for most targets and ID178/ID254 for S3. The best direct-ID252
excluded row was weaker but still notable at `0.433848` subject-hole local; it
is not truly ultra-free because ID254/ID255 are descendants of ultra calibration
artifacts.

This result is not a submission gate because all source artifacts are still
pre-existing OOF products. It is, however, the strongest current justification
for rebuilding the ID252/ID255/ID254 source path under true model-level nested
validation.

## Follow-up fold-internal second-stage calibration diagnostic

A no-submit calibration-layer diagnostic was created:

- `reports/fold_internal_second_stage_calibration_post_id335_20260617.json/.md`

It refit existing second-stage calibration recipes inside each outer fold. The
result splits the branch into two very different stories. Calibrating on top of
ID242 still looks extremely strong (`id250_like_on_id242` around local `0.262`),
but recreating ID242 itself from ID238 collapses badly (`0.585137`
subject-hole, `0.551329` LOSO, versus original full-OOF `0.357651`).

Therefore the current blocker is not the upper pair/logit blend selector; it is
the ID242 source layer. Any real rebuild must either make the ID242-like layer
survive true nested validation or exclude the ID242/250/251/252 dependency
chain.

## Follow-up conservative ID238 calibration salvage check

A no-submit diagnostic was created:

- `reports/id238_conservative_calibration_post_id335_20260617.json/.md`

It tested conservative temperature shifts and shrinked low-quantile/isotonic
calibrators on top of ID238, with all choices made inside outer folds. The only
stable improvement was a tiny temperature-shift signal: `0.461268`
subject-hole and `0.460105` LOSO, versus ID238 base `0.462365`. Low-quantile
and isotonic shrink variants worsened both splits.

This means the ID238 lower layer can maybe be mildly temperature-stabilized,
but it does not justify reviving the ID242 q240/isotonic branch. A Public
`0.58` path should not spend submission budget on ID242 descendants unless a
new true model-level source, not another OOF calibrator, replaces that layer.

## Follow-up surviving-candidate map excluding ID242 chain

A dependency-aware no-submit map was created:

- `reports/surviving_candidates_excluding_id242_post_id335_20260617.json/.md`

It scanned `238` base-test artifacts and found `93` descendants of the broken
ID239/ID242/ID249/ID250/ID251/ID252 chain. After excluding them, the top
reviewable survivor is ID179 (`0.538290` local, `0.563975` robust, max Δ178
`0.051824`, failed projection `+0.0011`). The rest of the leading survivors are
mostly sequence-temporal rows, not calibration-frontier rows.

This makes the post-ID242 strategy concrete: either continue with a stricter
ID179/sequence-temporal review, or build a genuinely new model-level source.
Do not keep mining the calibration-descendant frontier for submission probes.

## Follow-up sequence survivor review

A no-submit expanded sequence review was created:

- `reports/sequence_survivor_review_post_id335_20260617.json/.md`

ID179 remains the cleanest low-movement manual-review anchor after cutting the
ID242 chain. It improves ID178 only slightly (`+0.000348` local, `+0.000218`
robust) but stays very close to the confirmed public anchor (max Δ178
`0.051824`, failed projection `+0.0011`). Lower-local sequence descendants such
as ID193/198/200/203/206 move much farther from ID178 for small additional
local gains, while ID180 is rejected because its failed-direction projection is
too high (`+0.0348`).

This preserves ID179 as the cautious same-family review anchor, not as a Public
`0.58` claim or auto-submit candidate.

## Follow-up fold-internal sequence survivor selector

A no-submit selector diagnostic was created:

- `reports/fold_internal_sequence_survivor_selector_post_id335_20260617.json/.md`

Local-only target selection across sequence survivors reaches `0.537268`
subject-hole and `0.536920` LOSO, so the sequence-descendant branch has some
fold-internal target-wise signal. However, every movement/projection-controlled
scenario selects only ID178 and returns to local `0.538638`.

This is the current public-risk boundary for the sequence branch: lower-local
descendants are research targets, but they do not yet justify a leaderboard
probe under the post-ID178/ID329/ID287 evidence.

## Follow-up sequence source-generation plan

A no-submit command plan was created:

- `reports/sequence_source_generation_plan_post_id335_20260617.json/.md`

The target-specific source-generation lanes are now explicit: Q1 should study
SAX/PAA sequence patterns, Q2 sequence-summary embeddings, Q3/S2/S4
wavelet/spectral features, S1 random-conv refresh, and S3 compact
smartphone/light features. All commands are report-only and omit `--promote`.
Use Kaggle for the heavier raw-sequence jobs, but pull and review diagnostics
before creating any matched CSV/notebook/log/base-prediction artifact.

## Follow-up sequence source-generation execution review

Three report-only lanes were executed locally without `--promote` and summarized
in:

- `reports/sequence_source_generation_execution_review_post_id335_20260617.json/.md`

The diagnostics reproduce the existing sequence-source gains: sequence-summary
on ID191 improves the public-structure mean by `+0.001581`, smartphone/light on
ID193 improves `+0.000049`, and wavelet/spectral on ID203 improves `+0.000216`.
These are useful source-generation signals, but they remain small and selected
under public-structure diagnostics. They are not promotion evidence.

## Follow-up Kaggle sequence-source runner prep

A no-submit Kaggle runner was prepared:

- `notebooks/kaggle_sequence_source_runner/Sequence_Source_Post_ID335_Runner.ipynb`
- `notebooks/kaggle_sequence_source_runner/kernel-metadata.json`
- `reports/kaggle_sequence_source_runner_post_id335_20260617.json/.md`
- `reports/kaggle_sequence_source_remote_summary_post_id335_20260617.json/.md`

It packages the S1 random-conv refresh lane from the source-generation plan into
six sequential report-only jobs on ID173. The runner was pushed to Kaggle,
completed, and pulled. The best diagnostic was `nk512_seed45502`, improving the
ID173 diagnostic mean from `0.545253` to `0.542888`; most of that gain came from
S1 (`+0.010855`), with smaller Q1/S4 gains.

This confirms that S1 raw-sequence random-conv features still have source-level
signal, but the lane does not yet produce a whole-branch improvement beyond the
existing ID178/ID190 family. Treat it as a target-specific source to test under
stricter recombination, not as a submission candidate.

## Follow-up sequence anchor probe

A second no-submit Kaggle runner replayed the new S1 seed on stronger sequence
anchors:

- `notebooks/kaggle_sequence_anchor_probe_runner/Sequence_Anchor_Probe_Post_ID335_Runner.ipynb`
- `reports/kaggle_sequence_anchor_probe_remote_summary_post_id335_20260618.json/.md`
- `reports/sequence_anchor_probe_decision_post_id335_20260618.json/.md`

The S1 signal survived on ID178, ID179, ID190, and ID191. The cleanest
public-anchor construction would be ID178 plus S1-only
`seed45502/lr_c0_1/weight0.15` with estimated local `0.537664`; the best local
same-family construction would be ID179 plus the same S1-only source with
estimated local `0.537568`. These are candidate-construction leads only; they
still require artifact promotion and movement/projection gates before any
submission request.

## Follow-up ID336 controlled artifact

The conservative public-anchor construction was promoted as generated-unsubmitted
ID336:

- `submission_id_336_id178_s1_seed45502_randomconv_overlay_cv0.537663_unsubmitted.csv`
- `reports/id336_s1_only_public_risk_gate_20260618.json/.md`
- `reports/post_id336_public_anchor_rerank_20260618.json/.md`

ID336 changes only S1 from ID178. Non-S1 targets are byte-identical to ID178 in
OOF/test predictions. Its local CV is `0.537663`, with `+0.000974` gain versus
ID178 and S1 target loss improvement `+0.006820`. The post-ID336 public-anchor
rerank classifies it as `reviewable_low_movement`; failed-direction projection
is slightly negative (`-0.001662`).

This is the cleanest current candidate-construction result, but not a Public
`0.58` claim. If public budget is spent, ID336 should be considered a controlled
ID178-local S1 probe, not evidence that validation has been solved.

## Follow-up ID336 S1 weight sensitivity

A report-only sweep was created:

- `reports/id336_s1_weight_sensitivity_20260618.json/.md`

The local optimum remains weight `0.15` (ID336), with local `0.537663` and
`+0.000974` gain versus ID178. The conservative public-risk proxy instead
selects weight `0.05`, with local `0.538066`, smaller gain (`+0.000571`), but
much lower movement from ID178 (`max Δ178 0.037585` versus `0.112756`).

Validation consequence: this branch is useful as a controlled S1 probe, not as
a solved validation signal. Any next artifact from this lane must pre-register
the tradeoff between local gain and public-anchor movement and must receive a
new unique ID; ID336 must not be overwritten or auto-submitted.

## Follow-up ID337 low-movement S1 artifact

The lower-movement choice from the sensitivity sweep was promoted as ID337:

- `submission_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.csv`
- `reports/id337_s1_w005_public_risk_gate_20260618.json/.md`
- `reports/post_id337_public_anchor_rerank_20260618.json/.md`

ID337 keeps every non-S1 target identical to ID178 and uses S1
`seed45502/lr_c0_1/weight0.05`. Local CV is `0.538066`, weaker than ID336's
`0.537663` but still better than ID178. Its movement from ID178 is much smaller
than ID336 (`max Δ178 0.037585` vs `0.112756`), failed-direction projection is
negative (`-0.000554`), and the post-ID337 rerank places it first among
reviewable low-movement candidates by conservative proxy.

Validation consequence: ID337 is now the cleaner cautious public-probe object
than ID336, but it is not Public `0.58` evidence. The right interpretation is
“controlled ID178-relative S1 perturbation,” not validation repair.

## Follow-up ID179 anchor sweep and ID338 artifact

A second report-only sweep replayed the same S1 seed on ID179:

- `reports/id179_s1_seed45502_weight_sensitivity_20260618.json/.md`

The local-best weight `0.15` reached `0.537568`, but exceeded the low-movement
gate against ID178 (`max Δ178 0.144825`). The balanced/proxy weight `0.05`
reached `0.537861` and remained reviewable, so it was promoted as ID338:

- `submission_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.csv`
- `reports/id338_s1_w005_public_risk_gate_20260618.json/.md`
- `reports/post_id338_public_anchor_rerank_20260618.json/.md`
- `reports/id337_id338_s1_probe_tradeoff_20260618.json/.md`

Validation consequence: ID338 is locally stronger than ID337 by `0.000205`, but
worse by conservative proxy by `0.001811` because it starts from the unsubmitted
ID179 anchor and moves farther from ID178. ID337 remains the safety-first
controlled probe; ID338 is the local-gain sibling. Neither proves Public
`0.58`.

## Follow-up recent S1 probe stress review

A no-training stress review was created:

- `reports/recent_s1_probe_stress_review_20260618.json/.md`

It compares IDs `178`, `179`, `336`, `337`, `338`, `334`, and `335` using
existing OOF/test artifacts. ID336 is local/robust-best among the new S1
variants (`0.537663` / `0.563379`) but moves much more from ID178 (`max Δ178
0.112756`). ID337 is weaker locally (`0.538066`) but has far lower movement
(`max Δ178 0.037585`) and the best conservative proxy among the S1 probes.
ID338 improves local/robust over ID337 but worsens proxy.

Validation consequence: this confirms the current fork. Safety-first public
probe means ID337; local-gain sibling means ID338; local-best but higher
movement means ID336. None of the three is validation repair or Public `0.58`
evidence.

## Follow-up true-nested next-lane priority

Two no-submit reports were added:

- `reports/true_nested_next_lane_priority_post_id338_20260618.json/.md`
- `reports/b1_pair_blend_source_dependency_audit_20260618.json/.md`

Important correction: the pair/logit `direct_no_252` lane is not dependency
clean. It excludes direct ID252 at the selector layer, but its source artifacts
`254/255/256/257/258/266` depend on high/extreme full-OOF bridge/calibration
chains, including ID242/250/251/252. As a result, B1 remains a useful downstream
selector map, but the immediate model-level nested work should start at the
delta-capped bridge/source layer.

Validation consequence: prioritize A1 mid-risk delta-capped bridge rebuild
inside outer folds. Treat ID337 as a separate explicit-approval public probe
candidate, and keep the ID242/252 calibration chain blocked until the lower
source layer is rebuilt without full-OOF shortcuts.

## Follow-up fold-internal A1 bridge rebuild

A no-submit fold-internal replay was created:

- `reports/fold_internal_delta_capped_bridge_rebuild_post_id338_20260618.json/.md`

The A1 bridge/source signal survives when target-wise bridge choices are learned
inside outer folds: best all/q240 rows reach about `0.3954` local under both
subject-hole and LOSO, and the q240-free branch remains weaker but still
competitive around `0.4143`. This is validation-repair evidence only at the
selector layer. Because the source OOF artifacts are still pre-existing, it does
not authorize a CSV promotion or public submission. The next credible modeling
step is a true source-layer rebuild of the bridge candidates inside model-level
outer folds.

## Follow-up A1 bridge source priority

A source-layer priority map was created:

- `reports/a1_bridge_source_rebuild_priority_post_id338_20260618.json/.md`

Result: the strongest A1 selector is not clean enough to follow directly,
because the all-extreme row uses ID242/243 in `71.4%` of fold-target choices.
The q240-free row is weaker by roughly `0.019` local score but provides the
cleanest rebuild target: source IDs `238`, `239`, and `241`, with ID241 carrying
`Q1/Q2/Q3/S1/S2`, base/ID238 handling `S3`, and ID239 handling `S4`.

Validation consequence: prioritize the q240-free source-layer rebuild first.
Treat any ID242/243 or ID252 descendant as blocked until the lower source layer
is rebuilt without full-OOF shortcuts.

## Follow-up A1 q240-free source replay

A stricter no-submit replay was created:

- `reports/a1_q240_free_nested_source_replay_post_id338_20260618.json/.md`

It moved beyond selector-only validation by relearning the ID238-like pair
source, ID239/ID241 q240-free calibration, and A1 bridge choices inside each
outer fold. The signal largely disappeared: subject-hole replay was `0.491331`
and LOSO replay was `0.484186`, versus selector-only qfree around `0.4143` and
ID235 base `0.482255`.

Validation consequence: q240-free A1 is no longer the best immediate heavy
Kaggle rebuild lane. Keep it as a negative control and avoid creating any
submission artifact from this branch. The next search should either move below
the contaminated upstream sources `221/226/227/228/237`, or pivot to a different
non-calibration source family with stricter public-anchor behavior.

## Follow-up post-A1 sequence-summary pivot

Four report-only sequence-summary anchor replays were created:

- `reports/id178_sequence_summary_embedding_post_a1_20260618.json`
- `reports/id179_sequence_summary_embedding_post_a1_20260618.json`
- `reports/id190_sequence_summary_embedding_post_a1_20260618.json`
- `reports/id191_sequence_summary_embedding_post_a1_20260618.json`

Consolidation and gates:

- `reports/post_a1_sequence_pivot_priority_20260618.json/.md`
- `reports/sequence_summary_geometry_gate_post_a1_20260618.json/.md`
- `reports/sequence_summary_cap_sensitivity_post_a1_20260618.json/.md`

Result: sequence-summary is now the strongest surviving non-calibration signal.
The best public-anchor construction combines ID178 `Q1/Q2/Q3`
sequence-summary with ID337 S1 safety. Full weight gives local `0.536437` but
max Δ178 `0.121314`; scaled `0.80` gives local `0.536441` and max Δ178
`0.097051`.

Validation consequence: the scaled ID178-anchored sequence-summary + ID337-S1
construction was promoted as ID339 with a unique generated-unsubmitted artifact
contract. Its local CV is `0.536441`, max Δ178 is `0.097051`, and
failed-direction projection is `-0.012001`. It is still not Public `0.58`
evidence and still requires explicit submission approval.

Follow-up risk gate:
`reports/id339_seqsum_s080_s1_public_risk_gate_20260618.md` marks ID339 as
`reviewable_low_movement` but gives conservative public-risk proxy `0.609547`,
worse than ID337 by `0.004525`. This reinforces the validation conclusion:
local CV alone is not the selection criterion after ID287/ID329.

Compute parity guardrail:
`reports/mac_m4_kaggle_compute_parity_policy_20260618.md` records the local
Apple Silicon environment and requires Kaggle/CPU parity evidence before any
Mac-local TensorFlow/Keras/PyTorch-MPS artifact can be promoted.

Kaggle operational follow-up:
`reports/kaggle_offload_status_and_next_lanes_post_id339_20260618.md` confirms
the project Kaggle CLI works through `.venv/bin/kaggle`, the sequence-source and
sequence-anchor kernels are complete, and their outputs were pulled and
inspected, then transient build outputs were removed. The next remote run was
scoped as a report-only ID337/ID339 parity runner before any new heavy model
sweep.

That parity runner has been prepared locally and run on Kaggle CPU:
`reports/candidate_parity_runner_prep_post_id339_20260618.md`. The local smoke
test `reports/candidate_parity_post_id339_20260618.md` rebuilds ID337/ID339 and
matches stored OOF/test artifacts within floating-point noise. The remote
report `reports/remote_candidate_parity_post_id339_20260618.md` then shows that
Kaggle CPU failed only the original overly strict `1e-6` max-delta gate; under
the documented cross-platform gate both ID337 and ID339 pass with effectively
perfect correlations. This makes environment drift an unlikely explanation for
the ID178/287/329 public-local mismatch. The validation problem remains public
transfer and candidate selection, not Mac M4 computation.
