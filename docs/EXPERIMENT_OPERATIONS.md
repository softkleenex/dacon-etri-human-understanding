# Experiment Operations

This repository is a DACON competition workspace. Optimize for reproducible leaderboard movement, not for accumulating artifacts.

## Current Risk Posture

As of the authenticated 2026-06-24 KST submission-history correction, the best
confirmed public result is ID342: local `0.534000`, Public `0.5960832519`,
DACON submission `1467937`, rank `220`, submitted at `2026-06-23 23:30:37`
KST. This improves ID337 by `-0.004537376722` and ID178 by `-0.0053280490`,
but it is still far from Public `0.55` or prize range. The useful public
movement came from the ID178/ID221 feature-similarity shrink, not the ID338
S1-only random-conv probe. Treat ID342 as the current public anchor and keep
ID338/ID339 as scored comparison probes.

The older practical public probe ID329 scored Public `0.6017908577`, worse than
ID178 (`0.6014113009`) despite a much better local score. ID287 remains the
severe failed-local-frontier contrast: local `0.378934`, Public
`0.6698787318`. This means local frontier scans can still be actively
misleading. Future candidates must explain ID178, ID287, ID329, ID337/340,
ID338/339, and ID342 together.

The 2026-06-19 post-ID341 handoff refresh predates the submitted ID342
artifact. Its guardrail still applies in spirit: all unapproved local score
routes remain closed, and no new same-family local-only Q2 context or closed
Q3/S3 lane should be promoted without a fresh gated record. After the 2026-06-21
DACON slate, ID341/340/337 were consumed as micro-probes and stayed near the
`0.60` public band, so do not submit another same-family manual probe. The local
runner smoke is fresh, the prelaunch and guarded remote contracts pass, and the
operator handoff is ready only for the exact phrase
`Kaggle upstream rebuild no-submit 실행 승인`.

`reports/post_id341_public060_wall_root_cause_20260621.md` is now the concise
root-cause stop-line for the Public `0.60` wall. It ties together the tight
ID178/337/340 public band, ID287/329 local-CV optimism failures, the Q2-only
local lead, and the still-uncovered Q3/S3 gap. Treat it as the operator-facing
reason not to spend more DACON submissions or create same-family artifacts.
The stop-line is amended by the authenticated 2026-06-23 slate scores: ID342
feature-similarity shrink is the current best public anchor; ID339 Q123
sequence-summary and ID338 S1 random-conv also beat the old `0.60` band but do
not beat ID342. The next local work should be a no-submit, ID342-aware Q3/S3 or
independent-source probe, not another S1-only or Q-sequence cascade.

2026-06-24 ID342-aware local Q3/S3 follow-up: the source-priority report opened
raw temporal Q3/S3 sequence work only as report-only diagnostics. The raw
temporal smoke selected no target; Q3 showed an unstable partial lead
(`3/5` wins, mean delta `-0.001735`, max delta `+0.002487`) and S3 stayed
closed. The follow-up sequence-summary distance/kernel smoke selected no target
either; Q3 best was 2/5 wins with mean delta `+0.000055`, and S3 best was 1/5
wins with mean delta `+0.000092`. Because both local gates failed, there is no
current Kaggle wait state and no justified no-submit remote run from this
artifact family. Continue only with a materially stronger Q3/S3 representation
or a separately approved no-submit research runner; do not create submission
artifacts from these smokes.

The same rule now applies to the direct sleep-boundary/SOL and local tiny-CNN
follow-ups. `reports/post_id342_sleep_boundary_q3s3_rescue_smoke_20260624.md`
selected no Q3/S3 target against ID342 (Q3 0/5 wins, S3 1/5). The local
PixleepFlow/MIS-LSTM proxy
`reports/post_id342_tiny_temporal_cnn_q3s3_smoke_20260624.md` also selected no
target (Q3 0/5 wins, S3 2/5, both positive mean deltas). Do not launch a Kaggle
no-submit run for these exact architectures; remote compute should be reserved
for a materially different research design or for parity after a local gate
passes.

The separately approved post-ID342 self-supervised subject-state no-submit
workflow has now run end-to-end on Kaggle. The dedicated runner
`softkleenex/dacon-etri-ss-subject-state-id342` completed on version 4 after
slug-length, input-overlay, and `/kaggle/working` staging fixes. Its pulled
outputs live under
`build/kaggle_outputs/self_supervised_subject_state_post_id342/` and contain
only the remote JSON report, Markdown report, runner manifest, and Kaggle log.
The remote smoke read `1,839,621` raw payload rows, created `32,768` label-free
subject-state tokens, and reached final masked MSE `0.7820173501968384`.
`reports/post_id342_self_supervised_subject_state_remote_intake_20260624.md`
is the required post-pull gate and passes with `remote_intake_ready=True`, no
CSV or submission-like artifacts, no submission-log row, and no DACON
submission. This is research evidence only; it does not authorize a supervised
candidate, embedding feature matrix, submission CSV, or DACON action without a
separate artifact-contract review and exact DACON file/memo approval.

The same no-submit runner has now been extended with a raw subject-state
daily/window feature gate. Kaggle version 5 failed for a packaging reason, not
network instability: the new gate imported an ID342 helper script that was not
included in the small patch dataset. After making the gate self-contained,
Kaggle version 6 completed and pulled
`remote_post_id342_raw_subject_state_feature_gate_20260624.json/.md`. The full
scan read all `6,122,179` raw rows, matched `3,896,255` train-key rows, built
`969` in-memory features, and reached mean modality train-key coverage
`0.959074`. It reviewed `90` fold-internal ID342 candidates and passed `0`.
The best row is only a tiny S3 blend
`raw_subject_state_daily_windows_s3_lr_c0_02_w0.01`, projected Public
`0.596071791`, still `+0.046071791` from the `0.55` target. This closes the
daily/window raw-subject-state aggregate lane as negative evidence; do not
create a feature matrix artifact, prediction artifact, submission CSV,
submission-log row, Kaggle promotion run, or DACON submission from this result.

Kaggle runner version 7 then added the stronger S3/Q3 raw sequence
representation gate. The runner completed as a no-submit job in about `2703`
remote seconds, with the full daily/window gate still dominating runtime; the
new sequence gate itself ran after the full gate and wrote only JSON/Markdown
reports. It scanned five domain sequence families (`all_channels`,
`q3_usage_routine`, `s3_sleep_boundary`, `usage_evening`, and
`physio_light_sleep`) with 44 half-hour bins, `68` value/missingness channels,
`256` random-convolution kernels, seeds `45502/45602`, and fold-internal
ID342 blends for Q3/S3. The report
`reports/post_id342_s3q3_raw_sequence_representation_gate_20260624.md`
reviewed `240` candidates and passed `0`. The best row,
`raw_sequence_representation_q3_usage_routine_q3_lr_c0_20_seed45602_w0.005`,
has only `2/5` primary wins, mean all-target delta `-0.000000019`, worst
delta `+0.000033135`, projected Public `0.596083233`, and remains
`+0.046083233` from Public `0.55`. The remote intake report now checks this
sequence gate and still passes with `remote_intake_ready=True`, no CSV or
submission-like artifacts, no prediction/base artifacts, no submission-log row,
and no DACON submission. Treat raw random-convolution S3/Q3 sequence
representations as closed negative evidence under ID342; do not create a
candidate from this branch.

Post-ID342 public-response geometry now provides the next non-submission route.
`reports/post_id342_public_response_geometry_20260624.md` shows the 2026-06-23
slate local order and public order both ranked `[342, 339, 338]`, unlike the
failed ID287/329 frontier behavior. ID342's public-transfer movement is driven
by ID178/ID221 feature-similarity changes, with largest test movement on `S4`,
`Q1`, and `Q2`. The ID221-alpha decomposition in
`reports/post_id342_feature_similarity_source_decomposition_20260624.md` finds
small locally eligible ID342-neighborhood rows, but the best public proxy is
only about `0.59575` and remains far from Public `0.55`. Do not create a new
submission artifact or Kaggle run from this post-public alpha scan. The strict
artifact gate in
`reports/post_id342_feature_similarity_artifact_gate_20260624.md` confirms this
with `0` passing rows: the best proxy is `0.595751629`, still `+0.045751629`
from the `0.55` target, and every reviewed row fails the post-public scan
independence check. The next research route is a clean feature-similarity source
rebuild or materially new source, not a same-neighborhood target-alpha
submission.

The first paper-aligned rebuild check does not open that route yet.
`reports/post_id342_paper_aligned_feature_similarity_rebuild_gate_20260624.md`
uses DACON/ETRI priors to constrain Q1/S3 to movement/time, Q2/Q3 to app/time,
and S4 to HR/light/activity/time, with fold-internal config selection on top of
ID342. It is worse than ID342 locally (`0.534649` vs `0.534000`), wins only
`2/5` primary lenses, and projects to Public `0.596731761`; keep Kaggle and
DACON closed for this paper-aligned same-subject feature-similarity rebuild.

Past-only temporal target-state anchoring is closed as well.
`reports/post_id342_past_only_temporal_recency_gate_20260624.md` uses only
same-subject labels dated before each query row. It avoids the look-ahead
behavior that made earlier temporal-anchor probes risky, but still does not
beat ID342: candidate loss `0.534430`, delta vs ID342 `+0.000430`, primary wins
`0/5`, projected public proxy `0.596513661`. Do not run Kaggle or create a
candidate from simple temporal label persistence.

Cross-target consistency on top of ID342 is closed too.
`reports/post_id342_cross_target_consistency_gate_20260624.md` uses only other
ID342 target predictions as second-stage features for Q2/Q3/S3/S4, avoiding
same-row true sibling labels. The full-train/test preview moves only Q3 and
still fails the gate: candidate loss `0.534113`, delta vs ID342 `+0.000113`,
primary wins `0/5`, projected public proxy `0.596195808`. Do not use this as a
Kaggle/no-submit or DACON candidate route.

The registered material-source rescan is now the broad local stop-line for the
post-ID341 matrices under the ID342 anchor.
`reports/post_id342_registered_material_source_rescan_20260624.md` scanned
circadian HR/light phase, GPS path topology, phone/wearable discordance,
proximity/ambience co-occurrence, and usage semantic burst/Markov features
against ID342. It reviewed `450` candidates and passed `0`. The best row is a
tiny proximity/ambience S2 blend with projected Public `0.595911728`, still
`+0.045911728` from the `0.55` goal. Keep Kaggle and DACON closed for these
registered matrices.

The fused-material follow-up is also closed.
`reports/post_id342_registered_material_fusion_gate_20260624.md` concatenates
all five registered matrices into `322` features and reviews `90` fusion
candidates. Pass count is `0`; the best S3 row is effectively neutral
(`-0.000001844` mean all-target delta vs ID342), with projected Public
`0.596081408`. Do not launch Kaggle or create an artifact from lightweight
registered-material fusion.

The public-alignment validation-lens check is now the current validation
operating rule. `reports/post_id342_public_alignment_validation_lens_20260624.md`
uses the confirmed public-scored IDs `178/287/329/337/338/339/340/342`. It
shows that local CV is still unsafe across all IDs because ID287 dominates as
the local-best/public-worst failure (full OOF Spearman `-0.167`). Removing only
ID287 is not enough because ID329 remains locally over-optimistic (Spearman
`0.250`). Removing both failed-frontier IDs makes full OOF, subject-hole, and
LOSO exactly match the stable public order `[342, 339, 338, 337, 340, 178]`,
and every frozen split lens matches the 2026-06-23 slate `[342, 339, 338]`.
This does not open Kaggle or DACON. It means future no-submit gates must report
both failed-frontier-inclusive and failed-frontier-excluded views, and must not
promote a candidate that wins only because it moves toward ID287/ID329-style
frontier geometry.

The existing-candidate shelf has now been rechecked under that rule.
`reports/post_id342_stable_lens_existing_candidate_triage_20260624.md` scans
`247` OOF candidates and passes `0`. The apparent local-best rows are exactly
the unsafe frontier pattern: very low local losses, large OOF/test movement, and
high failed-direction alignment. ID222 and ID221 are the only stable-lens
lineage near-misses, but both are blocked because their submission-log notes
record public-structure and non-nested lineage. Do not submit, promote, or run
Kaggle for any existing shelf candidate from this result.

The first post-triage new-source check is
`reports/post_id342_env_social_routine_v2_id342_gate_20260624.md`. It is
research-aligned with DACON/ETRI daily-experience, sleep, affect, and stress
framing, but it only opens a Q2 micro-signal. Q2 wins `5/5` primary lenses at
weight `0.025`; Q3 and S3 win `0/5`. The projected Public proxy is
`0.595915153`, far above the `0.55` goal. Keep this as no-submit design
evidence and do not launch Kaggle or create an artifact from it.

The second post-triage lightweight source check is
`reports/post_id342_evening_state_transition_gate_20260624.md`. It tested
pre-sleep arousal and settling transitions using screen/app/HR, light,
charging, stillness, previous-day label-free deviations, and fold-internal
subject baselines. It does not open Q3 or S3: Q3's best row has only `2/5`
primary wins with mean delta `+0.000015535`, and S3's best row has `2/5` wins
with mean delta `+0.000063167`. Treat evening arousal/settling transitions as
closed local evidence under ID342; do not push Kaggle, create artifacts, edit
the submission log, or submit to DACON from this result.

`reports/post_id342_public_scored_target_swap_gate_20260624.md` then checks the
tempting target-swap shortcut directly. It finds `44` local wins from the known
failed-frontier donors ID287/ID329, including ID287 Q3 rows that project near
the Public `0.56` range locally, but these are exactly the public-failure
directions. Stable public-scored donors `178/337/338/339/340` pass `0` rows.
Do not promote public-scored Q3/S3 target swaps; they are anatomy evidence and
validation-warning evidence only.

`reports/post_id342_q3s3_slice_selector_anatomy_20260624.md` confirms that
fold-internal stable-donor selection does not rescue that route. Stable donors
win only `7` subject/target slices, failed-frontier donors win `20`, and the
selector has `0/5` primary wins for both Q3 and S3. Keep this lane closed for
Kaggle, artifacts, submission-log edits, and DACON. Future Q3/S3 work should
represent subject-state directly rather than selecting among old OOF donors.

`reports/post_id342_lightweight_subject_state_pca_gate_20260624.md` tests the
lightest safe version of that idea: fold-internal PCA over existing label-free
env/social/routine and registered material matrices. It passes `0` rows, with
the best row still worse than ID342 locally (`+0.000053036` mean delta on S3).
Do not push Kaggle, create artifacts, or submit from lightweight PCA. A future
subject-state route must be materially different, and heavy execution remains
behind the explicit heavy-compute approval phrase.

`reports/post_id342_processed_sleep_proxy_gate_20260624.md` formalizes the
processed sleep-proxy scratch check. It reads the existing
`data/processed/df_train_raw.parquet` table, aligns train rows one-to-one by
`subject_id` and `lifelog_date`, and tests Q3/S3 only with fold-internal LR/HGB
overlays on `sleep_proxy`, `sleep_proxy_no_ids`, and `core_sleep` feature
families. It passes `0` rows; the best S3 row has only `2/5` primary wins and
mean delta `+0.000019110` vs ID342. Re-run with
`make post-id342-processed-sleep-proxy-gate` if needed, but do not create
submission artifacts, edit `submission_log.csv`, push Kaggle, or submit to
DACON from this lane.

`reports/post_id342_processed_sleep_temporal_deviation_gate_20260624.md` then
tests the personal-baseline version of that lane: lag-1 sleep proxies, past
3-row rolling means/stds, and current deviations from those past baselines. It
also passes `0` rows; the best Q3 row has only `2/5` primary wins and mean
delta `+0.000111587` vs ID342. Re-run with
`make post-id342-processed-sleep-temporal-deviation-gate` if needed, but keep
the same no-artifact, no-log-edit, no-Kaggle, no-DACON contract.

`reports/post_id342_subject_prior_calibration_gate_20260624.md` checks whether
the Q3/S3 target definitions justify subject/global prior forcing. It does not:
all top rows have `0/5` primary wins, and the best Q3/S3 rows worsen ID342 by
`+0.000918056` and `+0.001384175` mean delta respectively. Re-run with
`make post-id342-subject-prior-calibration-gate` if needed, but do not use this
diagnostic to overwrite test subject means or create any public-facing artifact.

`reports/post_id342_probability_temperature_gate_20260624.md` verifies the
submission encoding and a minimal calibration idea. ID342's submission file is
continuous probability output exactly equal to `base_test_id_342`, so local
thresholding is not the artifact lever. The fold-internal Q3/S3
temperature/shift scan also passes `0` rows. Re-run with
`make post-id342-probability-temperature-gate` if needed, but do not create
another artifact or submit a temperature/threshold variant from this result.

`reports/post_id342_remaining_route_synthesis_20260624.md` is the current
post-ID342 stop-line. Re-run it with
`make post-id342-remaining-route-synthesis` after refreshing the input gates. It
closes `11/12` tracked local/lightweight routes, has no missing or review
routes, confirms no Kaggle wait state, and leaves Kaggle push/run plus DACON
submission set to false. The only remaining score-relevant route is the
materially different heavy self-supervised/subject-state representation; it is
not authorized without the exact heavy-compute no-submit approval phrase.

`reports/post_id342_self_supervised_route_research_packet_20260624.md` is the
blocked research packet for that route. Re-run it with
`make post-id342-self-supervised-route-research`. It maps Q3 to label-free
day/evening stress-state tokens, S3 to synchronized phone-wearable
sleep-transition tokens, and Q2 to a secondary sanity target. It is a planning
artifact only: do not read full raw streams, fit embeddings, write feature
matrices, push/run Kaggle, edit the submission log, or submit to DACON unless
the exact heavy-compute no-submit approval phrase is provided and the heavy
contract is refreshed first.

`reports/post_id342_self_supervised_heavy_compute_contract_20260624.md` is the
current heavy approval contract. Re-run it with
`make post-id342-self-supervised-heavy-contract`. It supersedes the post-ID341
contract for current execution decisions because the submission log now records
ID342 as a scored/submitted public anchor. Its current state is contract-ready
but execution-blocked: no Kaggle push/run, no raw full-stream read, no embedding
fit, no feature matrix, no submission-log edit, and no DACON submission. It
also records that the existing upstream rebuild notebook is not a sufficient
heavy runner.

`reports/post_id342_self_supervised_heavy_runner_spec_20260624.md` is the
dedicated runner specification that follows the heavy contract. Re-run it with
`make post-id342-self-supervised-heavy-runner-spec` after refreshing the
research packet or docs audit. It documents the no-submit Kaggle notebook path,
slug, data contract, model objective, resource cap, validation gates, and
after-approval command sequence. Its current decision is
`runner_spec_ready=True`, but `create_runner_files_now=False`,
`push_kaggle_now=False`, `run_kaggle_now=False`, and `dacon_submit_now=False`.
Treat it as readiness documentation, not execution permission.

`reports/post_id342_subject_state_domain_token_contract_20260624.md` is the
implementation meaning contract for the same route. Re-run it with
`make post-id342-subject-state-domain-token-contract`. It maps all 12 raw
phone/watch modalities into Q3 stress-state, S3 sleep-transition, and Q2 sanity
roles, then freezes day-grid, anchored-evening, sleep-transition, and
subject-baseline token windows. It incorporates the current ETRI/DACON target
definitions plus cross-modal SSL research, but still creates no raw-token cache,
embedding, feature matrix, prediction, Kaggle run, log row, or DACON
submission.

`reports/post_id342_subject_state_raw_schema_contract_20260624.md` is the
schema-only gate for that implementation. Re-run it with
`make post-id342-subject-state-raw-schema-contract`. It opens the 12 raw parquet
files through metadata only, validates required columns, and records `12/12`
schemas OK with `6,122,179` metadata rows and `0` payload rows read. This gate
can be run before approval; it still does not authorize token caches, feature
matrices, embeddings, Kaggle execution, submission-log edits, or DACON
submission.

`reports/post_id342_subject_state_heavy_approval_readiness_20260624.md` is the
single readiness checkpoint before asking for or consuming the exact heavy
approval. Re-run it with `make post-id342-subject-state-heavy-approval-readiness`.
Its current state is fail-closed: prerequisites are ready, the old upstream
Kaggle runner has no wait state (`KernelWorkerStatus.COMPLETE` from the manual
status check), but `exact_heavy_approval_observed=False`. Do not create the
dedicated runner, push Kaggle, read raw payloads, or edit any submission records
until the exact heavy approval phrase is present.

`reports/post_id342_subject_state_heavy_success_thresholds_20260624.md` is the
review ladder for future approved heavy outputs. Re-run it with
`make post-id342-subject-state-heavy-success-thresholds`. Treat results in the
ID342 `0.596` band as research leads only. Artifact review starts only at
projected public proxy `<=0.58` with stable Q3/S3 improvement, failed-frontier
resistance, worst-split control, and mandatory remote purity/coverage/artifact
separation reports. DACON candidacy starts only at proxy `<=0.55` and still
requires exact file/memo approval plus artifact/log/notebook matching.

`reports/post_id342_kaggle_input_availability_20260624.md` is the current
Kaggle CLI input check. Re-run it with
`ALLOW_KAGGLE_REMOTE=true make post-id342-kaggle-input-availability`. It lists
the required base and upstream-patch datasets through `.venv/bin/kaggle` and
confirms both are visible. The only issue is the known outdated Kaggle CLI
warning (`2.1.1` installed vs `2.2.2` suggested); there is no current dataset
slug, auth, or network blocker. This check does not push, run, pull outputs, or
authorize the heavy runner.

`reports/post_id342_kaggle_cli_health_20260624.md` is the broader CLI health
checkpoint. Re-run it with
`ALLOW_KAGGLE_REMOTE=true make post-id342-kaggle-cli-health`. It confirms the
local Kaggle binary is `.venv/bin/kaggle`, installed version `2.1.1`, required
dataset listings work, and the old upstream kernel status is
`KernelWorkerStatus.COMPLETE`. The CLI warning is version freshness only, not a
current auth/network/slug failure. Do not upgrade the CLI during a submission
window unless a concrete CLI failure appears.

`reports/post_id342_morning_closeout_20260624.md` is the current morning
operator snapshot. Re-run it with
`ALLOW_KAGGLE_REMOTE=true make post-id342-morning-closeout` if any prerequisite
state changes. It records that ID342 remains the public anchor, Public `0.55`
is not reached, no DACON submission is meaningful now, there is no Kaggle wait
state, and the Kaggle CLI is usable. Keep execution closed unless the exact
heavy approval phrase is present.

`reports/post_id342_external_research_priority_20260624.md` is the current
research-priority update for the remaining heavy route. Re-run it with
`make post-id342-external-research-priority` after refreshing the local heavy
route contracts. It does not use Kaggle or DACON; it records that the first
approved heavy runner should prioritize missingness-aware cross-modal masked
subject-state learning over image-only, calibration, alpha, or donor-swap
variants.

`reports/post_id342_heavy_implementation_feasibility_20260624.md` is the
current approved-later implementation audit. Re-run it with
`make post-id342-heavy-implementation-feasibility`. It checks installed
dependencies, raw input presence, existing heavy route contracts, and missing
after-approval runner/Make targets. Treat missing runner files and missing
targets as planned after-approval work, not permission to create them now.

`reports/post_id342_heavy_runtime_budget_20260624.md` is the current
metadata-only runtime budget for the approved-later heavy route. Re-run it with
`make post-id342-heavy-runtime-budget`. It reads parquet footers only and
records that all raw modality files have one row group, so a future runner
should use scanner/batch streaming instead of row-group reads. It does not
authorize payload reads, token caches, embeddings, Kaggle runs, or submissions.

`reports/post_id342_goal_state_audit_20260624.md` is the current active-goal
checkpoint. Re-run it with `make post-id342-goal-state-audit`; it refreshes
docs-audit first and then records the current score gap, local resource
snapshot, Kaggle/DACON action gates, and CSV/notebook/log matching for the
current candidate set. Its current decision is `goal_achieved=False`,
`run_kaggle_now=False`, and `dacon_submit_now=False`.

The local/Kaggle compute boundary is now explicit: local smoke validates only
the source.zip plus remote-output contract staging. The full upstream notebook
must not run on the Mac by default; it remains a guarded no-submit Kaggle action
behind the exact approval phrase above.

2026-06-23 Kaggle upstream packaging update: the previous remote failure was
not caused by Kaggle CLI authentication, kernel slug permission, or Mac/M4
runtime behavior. The pushed runner used the stale 2026-06-17
`softkleenex/dacon-etri-human-base` dataset, which contains large raw/processed
inputs but no `reports/` directory and no post-ID341 patch artifacts. The runner
therefore failed at the first required-path check. The approved no-submit
workflow now uses two Kaggle dataset sources:

1. `softkleenex/dacon-etri-human-base` for large raw/processed files.
2. `softkleenex/dacon-etri-upstream-patch-post-id341` for the latest
   `scripts/`, `reports/`, `submissions/`, `artifacts/base_predictions/`,
   external ETRI files, and registered post-ID341 feature parquet.

The runner overlays the patch after staging the base. Do not revert to relying
on a single large `source.zip` unless the patch path is explicitly retired and
revalidated. Upstream runner version 2 was pushed with both dataset sources and
passed the initial packaged-project required-path check (`Missing required
paths: []`), then generated many remote no-submit reports. It is no longer a
wait state: Kaggle finished with `KernelWorkerStatus.ERROR` at the final
upstream blueprint assertion because
`remote_post_id341_upstream_rebuild_blueprint_20260618.json` reported
`decision.blueprint_ready=False`; the pulled remote blueprint shows
`missing_required_files=["dataset-metadata.json"]`. Treat this as a Kaggle input
metadata exposure issue, not a Kaggle CLI/auth failure. The runner now
synthesizes minimal runtime metadata after overlay when Kaggle omits the file,
and the source self-test cleanup now removes only `build/kaggle_dataset` so
`build/kaggle_outputs` evidence is preserved. The canonical failure note is
`reports/kaggle_upstream_rebuild_v2_failure_20260623.md`. No DACON submission or
new candidate artifact is authorized by this remote run.

2026-06-24 CLI status check: the system shell path may not expose a global
`kaggle` command, but the project virtualenv binary `.venv/bin/kaggle` exists
and is the Makefile default. Running
`ALLOW_KAGGLE_REMOTE=true KAGGLE=.venv/bin/kaggle make kaggle-kernel-status-upstream-rebuild`
confirmed `KernelWorkerStatus.COMPLETE` for
`softkleenex/dacon-etri-upstream-rebuild-post-id341`. The installed CLI warns
that version `2.1.1` is older than `2.2.2`, but status retrieval works. There
is no current Kaggle wait state.

`phone_wearable_discordance_v1` has now completed its registered label-free
matrix write and deterministic validation-stop boundary. The design, bounded
85-column feature spec, in-memory feature-generation dry run, report-only
feature-matrix smoke contract, schema report, and validation stop all pass, but
this remains infrastructure rather than score evidence. It does not authorize
model training, ID342, Kaggle network use, submission-log edits, submission
artifact creation, or DACON submission. With this stop-line closed, the only
score-relevant next route is still the exact-approved Kaggle upstream rebuild
no-submit path.

`self_supervised_subject_state_embedding_v1` is now recorded only as a
metadata-level design gate at
`reports/post_id341_self_supervised_subject_state_embedding_design_20260619.md`.
It covers all 12 raw sensor streams and the Q2/Q3/S3 gap, but it is deliberately
classified as materially new but heavy. Do not fit embeddings, read full raw
streams for training, write a feature matrix, create ID342, run Kaggle, edit the
submission log, or submit from this design. If this route is ever pursued, it
needs a separate heavy-compute contract before execution; the current exact
approval phrase still applies only to the guarded upstream rebuild no-submit
path.
The local prelaunch chain now regenerates that design report and then refreshes
`reports/post_id341_material_new_source_queue_20260619.md`, where the route must
remain `design_gate_ready_heavy_deferred` with all artifact/model/submission
flags closed.

Post-ID341 GPS path-topology is now the first materially new local source family
that clears only the report-level dry-run contract: exact 700-row alignment,
32 bounded first-pass columns, and GPS-ready train/test rates above 0.93. Treat
`reports/post_id341_gps_path_topology_feature_generation_dryrun_20260619.md` as
permission to design a separate feature-matrix smoke contract, not as permission
to train a model, create ID342, submit to DACON, or bypass the explicit-approved
Kaggle upstream rebuild route.

That feature-matrix smoke contract now exists at
`reports/post_id341_gps_path_topology_feature_matrix_smoke_contract_20260619.md`.
It permits only a future no-model/no-submission write to
`data/processed/post_id341_gps_path_topology_features_v1.parquet` plus schema
report, followed by validation. It still does not authorize modeling, ID342, a
Kaggle network run, or DACON submission.

The registered GPS feature-matrix writer may create only that parquet and
`reports/post_id341_gps_path_topology_features_v1_schema.json/.md`. Treat the
matrix as a bounded, label-free raw-source cache, not as a score candidate:
do not train from it, create a submission artifact, modify `submission_log.csv`,
run Kaggle, or submit to DACON without a new explicit contract.

The first no-submit model smoke on that matrix,
`reports/post_id341_gps_path_topology_model_smoke_20260619.md`, does not open a
submission lane. It selects no Q3/S3 target under the frozen gate, with Q3
worsening and S3 only weakly positive on three of five primary splits. Do not
promote GPS-topology-only artifacts from this result.

The next materially new local source family is circadian HR/light phase shape.
`reports/post_id341_circadian_hr_light_phase_metadata_probe_20260619.md` shows
feasible raw coverage (`0.5622`/`0.6120` strong train/test phase-shape rates),
and `reports/post_id341_circadian_hr_light_phase_feature_spec_20260619.md`
freezes a 48-feature, no-label, predeclared-hour-bin design. The follow-up
dry run, `reports/post_id341_circadian_hr_light_phase_feature_generation_dryrun_20260619.md`,
assembles those features in memory over the exact 450 train + 250 test keys
and passes the strong/weak coverage gates. The follow-up smoke contract,
`reports/post_id341_circadian_hr_light_phase_feature_matrix_smoke_contract_20260619.md`,
now permits only a future no-model/no-submission write to the single registered
label-free parquet
`data/processed/post_id341_circadian_hr_light_phase_features_v1.parquet` plus
schema report. This is still not score evidence and does not authorize model
training, ID342, Kaggle network use, or DACON submission.

That registered circadian matrix has now been materialized and validated in
`reports/post_id341_circadian_hr_light_phase_features_v1_schema.md`. Treat it
as bounded, label-free feature infrastructure only. The follow-up contract and
bounded in-memory smoke are recorded in
`reports/post_id341_circadian_hr_light_phase_model_smoke_contract_20260619.md`
and `reports/post_id341_circadian_hr_light_phase_model_smoke_20260619.md`.
The smoke selected no Q3/S3/S2 target and opened no score route, so the
circadian lane is closed for local artifact work; do not rerun it with small
weight changes.

The current no-approval local step is now only the report-level usage semantic
burst/Markov design at
`reports/post_id341_usage_semantic_burst_markov_design_20260619.md`. It shows
raw usage/session coverage is sufficient for a later metadata/cardinality
smoke, but it creates no feature matrix, trains no model, and does not
authorize ID342, Kaggle network use, or DACON submission.

That metadata/cardinality smoke now exists at
`reports/post_id341_usage_semantic_burst_markov_metadata_smoke_20260619.md`.
It passes the raw state-space gate with all 7 predeclared app states, 48/49
Markov transition pairs, 4,686 sessions, and 99.85% usage-to-screen matching.
This still authorizes only a report-only feature spec; no feature matrix,
model training, ID342, Kaggle network use, or DACON submission is allowed.

That report-only feature spec now exists at
`reports/post_id341_usage_semantic_burst_markov_feature_spec_20260619.md`.
It freezes the exact 700-row row contract, 7 fixed app states, 49 fixed Markov
pairs, 7 feature groups, and a bounded 77-column first-pass schema. It is still
not score evidence. The only no-approval follow-up is an in-memory dry run that
checks alignment/schema/leakage; do not write a feature matrix, train a model,
create ID342, run Kaggle, or submit to DACON from the spec itself.

That in-memory dry run now exists at
`reports/post_id341_usage_semantic_burst_markov_feature_generation_dryrun_20260619.md`.
It assembles 450 train + 250 test rows with 77 finite feature columns and zero
NaNs, using the actual raw `mUsageStats`, `mScreenStatus`, and `mAmbience`
schemas. This still is not score evidence. The only no-approval follow-up is a
report-only feature-matrix smoke contract; do not write the matrix, train a
model, create ID342, run Kaggle, or submit to DACON from the dry run itself.

That report-only feature-matrix smoke contract now exists at
`reports/post_id341_usage_semantic_burst_markov_feature_matrix_smoke_contract_20260619.md`.
It permits only a future write to the single registered label-free path
`data/processed/post_id341_usage_semantic_burst_markov_features_v1.parquet`
plus schema report, then a hard stop. It still does not authorize model
training, ID342, Kaggle network use, or DACON submission.

The registered usage semantic burst/Markov matrix has now been materialized at
`data/processed/post_id341_usage_semantic_burst_markov_features_v1.parquet`,
with schema report
`reports/post_id341_usage_semantic_burst_markov_features_v1_schema.md`. The
matrix has 700 rows, 77 label-free feature columns, no NaNs, and finite values.
This remains infrastructure only: no model training, prediction/base artifact,
submission CSV, submission-log row, ID342, Kaggle run, or DACON submission is
authorized from this matrix alone. The queue now stops at schema validation
unless exact Kaggle/no-submit approval is given.

## Before Any Submission

1. Run `make normalize-submissions` to refresh `status_clean`, `risk_reason`, and `safe_for_default_ensemble` in `submissions/submission_log.csv`.
2. Run `make audit`. A critical failure means the workspace is not ready for trusted leaderboard work.
3. Confirm `reports/post_id341_public_local_gap_diagnosis_20260618.md` is fresh. `make audit` and `make public-local-gap-diagnosis` both regenerate it; do not promote local-CV-only candidates unless this report's recent public proxy gate is passed.
4. Confirm `reports/post_id341_manual_submission_probe_decision_20260618.md` is fresh. Its current default is `do_not_submit`; after ID341/340/337, there is no fresh same-family public sanity probe to recommend.
5. Use `make ensemble` only after normalization. The default ensemble excludes blacklisted files and candidates with unsafe CV-LB gaps.
6. Write or update a decision note before `make dacon-submit`; the note must name the exact file, validation split, local score, public-risk evidence, and reason it is not another ID287/ID329-style failure. The command itself must include `ALLOW_DACON_SUBMIT=true` plus exact `SUB_PATH` and `SUB_MEMO`.

After ID287/ID329, post-hoc full-OOF subset losses are not acceptable promotion
evidence. Any score-improving candidate must be supported by a fold-nested
validation path where candidate selection, calibration, and blending are learned
without the outer validation rows. Test-geometry diagnostics may be used as
stress tests, not as a standalone local leaderboard.

ID335 is the latest prepared-but-unsubmitted sequence refinement artifact. It
may be reviewed as a weak controlled Public `<0.60` probe hypothesis, but it is
not Public `0.58`-plausible and must not be submitted without explicit user
approval. ID334 remains the cleaner binary rollback reference.

Demographic/static features are globally frozen. Future demographic research is
allowed only target-specifically for `Q2`, `S2`, and `S4`, per
`reports/target_specific_demographic_gate_post_id335_20260617.md`; this gate is
not submission evidence by itself.

The follow-up overlay scan
`reports/demographic_target_overlay_post_id335_20260617.md` found no guarded
positive demographic alpha over ID335, so demographic/static features remain
research-only. The latest candidate map is
`reports/post_id335_public_anchor_rerank_20260617.md`; do not submit from this
proxy alone.

`reports/id179_vs_id335_sequence_probe_review_20260617.md` shows ID179 is the
cleaner same-family manual-review probe than ID334/ID335, but it remains
non-strict and not Public `0.58`-plausible. It still requires explicit approval
before any DACON submission.

`reports/failed_direction_candidates_post_id335_20260617.md` freezes the
rerank-top failed-direction/orthogonal rows as diagnostic-only. Do not promote
ID304/305/306/307/310/311 without a fold-nested path that directly addresses
ID329's public non-improvement.

`reports/candidate_outer_selector_post_id335_20260617.md` is only
selection-nested over existing artifacts. Its low local score does not authorize
submission because the selected artifacts remain non-nested/full-OOF.

`reports/fold_internal_failed_direction_rebuild_post_id335_20260617.md`
recomputes beta/projection/source choices inside outer folds and preserves a
meaningful local signal, but it still uses pre-existing source OOF artifacts.
Treat it as justification for a true model-level fold-nested rebuild, not as a
leaderboard-facing candidate.

`reports/true_nested_rebuild_blueprint_post_id335_20260617.md` is the current
map for that rebuild. It separates the selected sources into delta-capped
bridge, pair/logit blend, and second-stage calibration lanes. Do not collapse
these lanes back into a full-OOF recombination shortcut.

`reports/fold_internal_pair_blend_rebuild_post_id335_20260617.md` is the
strongest current local signal in this branch. It still does not authorize a
submission because the source artifacts are pre-existing OOF products and the
best choices depend heavily on ultra ID252. Treat it as a source-rebuild target,
not a leaderboard candidate.

`reports/fold_internal_second_stage_calibration_post_id335_20260617.md` shows
why that source rebuild must start below ID252: ID242-like calibration on ID238
collapses under fold-internal refitting, while upper calibrations on the already
non-nested ID242 artifact remain artificially strong. Do not spend submission
budget on descendants of ID242 until this lower layer is rebuilt or replaced.

`reports/id238_conservative_calibration_post_id335_20260617.md` confirms that
the lower layer has only a tiny conservative temperature-shift salvage signal.
Low-quantile and isotonic shrink alternatives worsen strict folds. Treat
ID242-descendant artifacts as deprioritized unless a true model-level source
rebuild replaces the q240/isotonic OOF shortcut.

`reports/surviving_candidates_excluding_id242_post_id335_20260617.md` is the
current dependency-aware candidate map after cutting the ID242 chain. ID179 is
the top survivor. Any next same-family public probe must be argued against the
confirmed ID178 public score and the ID179/ID335 review; otherwise prioritize
new model-level source generation over recombination.

`reports/sequence_survivor_review_post_id335_20260617.md` expands that review.
ID179 remains the cautious low-movement sequence anchor; lower-local sequence
descendants are watch-only because they move much farther from ID178, and ID180
is rejected by failed-direction projection. Do not submit any of these without
a fresh decision note and explicit user approval.

`reports/fold_internal_sequence_survivor_selector_post_id335_20260617.md`
shows that local-only target selection among sequence survivors improves strict
folds, but public-risk caps force the selector back to ID178. Treat sequence
descendants as source-generation research, not submission-ready recombination.

`reports/sequence_source_generation_plan_post_id335_20260617.md` is the current
command map for that research. Run these lanes report-only, preferably on
Kaggle for raw-sequence-heavy jobs, and never add `--promote` or create a
submission artifact until the diagnostics are reviewed.

`reports/sequence_source_generation_execution_review_post_id335_20260617.md`
summarizes the first executed report-only lanes. The gains reproduce but remain
small and non-nested. Do not promote them directly; use them to prioritize new
source generation and stricter validation.

`reports/kaggle_sequence_source_runner_post_id335_20260617.md` records the
Kaggle execution of the S1 random-conv source lane. The runner completed and was
summarized in
`reports/kaggle_sequence_source_remote_summary_post_id335_20260617.md`. It
staged only JSON diagnostics and did not create submissions, base predictions,
notebook/history artifacts, or submission-log rows. Reuse the S1 signal only
through a separate strict decision gate.

`reports/sequence_anchor_probe_decision_post_id335_20260618.md` is that first
replay gate. It confirms the S1 seed survives on ID178/179/190/191, but it
still authorizes only candidate construction, not submission. If promoted,
prefer a single S1-only artifact first, with the full unique
CSV/notebook/log/report/base-prediction contract and no extra target tweaks.

`reports/id336_s1_only_public_risk_gate_20260618.md` records that promoted
single-target artifact. ID336 is generated-unsubmitted and reviewable, with all
non-S1 targets identical to ID178 and a non-positive failed-direction
projection. It still requires explicit user approval for the exact CSV before
any DACON submission.

`reports/id336_s1_weight_sensitivity_20260618.md` weakens any automatic-submit
case for ID336. Weight `0.15` is the local optimum, but weight `0.05` is the
balanced/proxy choice because it keeps max movement from ID178 far smaller
(`0.037585` vs `0.112756`) while retaining a smaller local gain. Do not submit
ID336 merely because it has the best local score in this sweep; a lower-weight
variant would require a new unique artifact ID and its own decision note.

That lower-weight variant is ID337:
`submission_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.csv`.
It is the current cleanest low-movement S1 probe by the post-ID337 rerank, but
it remains an unsubmitted controlled-probe hypothesis only. No DACON submission
may be made without explicit approval for this exact file, and any score claim
must state that the proxy is not calibrated to Public LB.

ID338 is the ID179-anchor sibling:
`submission_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.csv`.
It is locally stronger than ID337, but its public-anchor proxy is worse because
it moves farther from ID178. Use `reports/id337_id338_s1_probe_tradeoff_20260618.md`
as the current decision note: safety-first means ID337; local-gain sibling means
ID338. Neither is Public `0.58` evidence or submission-approved.

`reports/recent_s1_probe_stress_review_20260618.md` confirms the same decision
under local slice/robust stress. ID336 is local/robust-best but moves more from
ID178; ID337 is the safety-first low-movement probe; ID338 is the local-gain
sibling. Do not present any of them as Public `0.58`-plausible. If the user
asks for one controlled public probe, default to reviewing ID337 first unless
they explicitly choose local gain over movement safety.

`reports/true_nested_next_lane_priority_post_id338_20260618.md` and
`reports/b1_pair_blend_source_dependency_audit_20260618.md` define the current
non-submission path. The immediate true-nested execution lane is A1
delta-capped bridge/source rebuild, not direct reuse of the B1 pair/logit
sources. The B1 selector remains useful downstream, but its current source IDs
254/255/256/257/258/266 are dependency-contaminated by high/extreme full-OOF
bridge/calibration artifacts. Rebuild the bridge/source layer inside outer
folds before replaying pair/logit selection. Keep ID242/252 descendants blocked.

`reports/fold_internal_delta_capped_bridge_rebuild_post_id338_20260618.md`
confirms that the A1 bridge selector still has strong fold-internal signal, but
only at the selector layer over pre-existing OOF sources. Treat it as the
priority map for a true source-layer rebuild. It does not create or authorize a
submission artifact.

`reports/a1_bridge_source_rebuild_priority_post_id338_20260618.md` narrows that
rebuild: start with q240-free sources `238/239/241`, not the q240-dominated
ID242/243 branch. The q240-free lane costs about `0.019` local score at the
selector layer but is the cleanest way to test whether A1 survives without the
blocked calibration chain. Run it report-only inside frozen outer folds.

`reports/a1_q240_free_nested_source_replay_post_id338_20260618.md` then tests
that lane more strictly and weakens it. Once ID238-like pair choice and
ID239/241 q240-free calibration are relearned inside outer folds, the replay
falls back to `0.491331` subject-hole and `0.484186` LOSO, close to or worse
than ID235 base. Do not spend heavy Kaggle/model-level compute on this exact
q240-free A1 stack unless a lower upstream source rebuild is defined first.

Post-A1, sequence-summary embeddings are the current best non-calibration
candidate-construction lane:
`reports/post_a1_sequence_pivot_priority_20260618.md`,
`reports/sequence_summary_geometry_gate_post_a1_20260618.md`, and
`reports/sequence_summary_cap_sensitivity_post_a1_20260618.md`. The preferred
variant was promoted as ID339:
`submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv`.
It uses ID178 `Q1/Q2/Q3` sequence-summary at scale `0.80` plus ID337 S1 safety,
with local `0.536441`, max Δ178 `0.097051`, and failed projection `-0.012001`.
The follow-up risk gate
`reports/id339_seqsum_s080_s1_public_risk_gate_20260618.md` keeps it in the
`reviewable_low_movement` tier but gives a conservative public-risk proxy of
`0.609547`, worse than ID337 by `0.004525`. It is generated-unsubmitted only,
not submission-approved, and not Public `0.58` evidence.

## Compute Parity Rules

The current local machine is Apple Silicon (`M4 Pro`, arm64) with project Python
`3.14.3`; TensorFlow is not importable in the current venv, PyTorch MPS is
available, and the Kaggle credential file exists. Bare `kaggle` is not on the
current shell `PATH`, but the project CLI `.venv/bin/kaggle` works and is what
the Makefile uses. See
`reports/mac_m4_kaggle_compute_parity_policy_20260618.md`.

Treat Mac-local TensorFlow/Keras/PyTorch-MPS artifacts as environment-risky.
They may be used for exploration, but they cannot become promoted submission
candidates unless a parity report compares the same seed/data slice against a
Kaggle or CPU-reference run. Minimum parity evidence: package/device versions,
OOF/test prediction correlation, max/mean absolute prediction deltas, and
per-target log-loss deltas. Use `.venv/bin/kaggle` or the existing Makefile
targets for offloaded runs.

Post-ID339 Kaggle status:
`reports/kaggle_offload_status_and_next_lanes_post_id339_20260618.md` confirms
the sequence-source and sequence-anchor Kaggle kernels are `COMPLETE` and their
outputs were pulled and inspected, then transient build outputs were removed.
The next remote job was deliberately scoped as a report-only candidate parity
runner for ID337/ID339 rather than another broad model sweep.

The parity runner has now been prepared locally and run on Kaggle CPU:
`reports/candidate_parity_runner_prep_post_id339_20260618.md`. It adds
`notebooks/kaggle_candidate_parity_runner/` and
`scripts/diagnose_candidate_parity_post_id339.py`. A local smoke test report
`reports/candidate_parity_post_id339_20260618.md` gives `review_pass` for both
ID337 and ID339 with OOF/test max deltas at floating-point noise. The Kaggle
CPU report `reports/remote_candidate_parity_post_id339_20260618.md` preserves
the remote output: both candidates failed the original strict `1e-6` max-delta
gate but pass the documented cross-platform tolerance gate with near-perfect
correlation. This clears the current Mac/M4 environment-suspicion lane, but it
does not improve or predict Public LB. The submission queue remains unchanged:
ID337 is still the safer controlled probe than ID339 if and only if the user
explicitly approves the exact CSV.

Post-parity candidate transfer gate:
`reports/validation_transfer_gate_post_remote_parity_20260618.md` compares the
current watch candidates against the recent ID178/ID329 public gap floor and
the post-parity conservative proxy. No current candidate passes the Public
`0.58` transfer gate. Do not create ID340 from local CV alone; the next useful
experiment must define a new source or validation split that lowers transfer
stress, not just another blend toward lower local CV.

The follow-up OOF geometry stress report
`reports/test_geometry_oof_stress_post_remote_parity_20260618.md` slices
existing OOF artifacts on train rows that resemble the hidden test row geometry.
It confirms a dangerous paradox: ID339 improves these slices, but known public
failures ID287/ID329 improve them even more. Therefore OOF geometry stress is a
warning lens only, not promotion evidence.

The pre-ID340 validation split manifest is
`reports/id340_validation_split_manifest_20260618.md`. Any ID340 source rebuild
must be selected inside these lenses: `subject_hole_5`,
`subject_timeseries_5`, `actual_test_band_holdout`,
`test_boundary_neighbor2_holdout`, `test_boundary_neighbor5_holdout`, and
`loso_10`. Existing full-OOF recombination alone is explicitly blocked.

The first source-level ID340 probes are now recorded:
`reports/id340_source_rebuild_split_probe_20260618.md` and
`reports/id340_sequence_summary_split_probe_20260618.md`. The lightweight
raw/tabular source models fail badly against the ID178 reference on every
primary split, so that lane is not worth promoting. The sequence-summary split
probe is the first credible ID340 construction lead: the exact design
`blend_id178_sequence_summary_lr_c0_3_w0.03` beats ID178 on all primary splits,
but loses slightly on `subject_timeseries_5`. See
`reports/id340_sequence_summary_candidate_decision_20260618.md`. This is still
not an artifact and not a submission candidate; the next required gate is a
report-only full-train/test prediction geometry check for that exact design.

The all-target exact design then failed the full geometry gate because its
proxy was worse than ID337:
`reports/id340_exact_sequence_full_geometry_gate_20260618.md`. The follow-up
subset scan `reports/id340_sequence_subset_geometry_scan_20260618.md` found a
safer Q2-only variant at weight `0.01`. That variant was promoted as full
contract generated-unsubmitted ID340:
`submission_id_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426_unsubmitted.csv`.
Its public-risk gate is
`reports/id340_q2_seqsum_w001_public_risk_gate_20260618.md`: proxy `0.604062`,
better than ID337/ID339 but still not Public `0.58` evidence. It must not be
submitted without explicit approval for the exact CSV.

The post-ID341 public transfer audit is
`reports/post_id341_public_transfer_audit_20260618.md`. It refreshes the
decision gate after ID341, using plausible public rows and the ID178/ID329
recent anchor gap floor. No current candidate passes the Public `0.58` transfer
gate; ID341 is only a low-movement control/probe. Do not continue same-family
local OOF recombination or micro-weight shrinking as a default path. The next
score-relevant lane is a report-only Kaggle upstream source rebuild, or a new
feature/source family that survives this transfer audit before any full artifact
contract is created.

`reports/post_id341_public_local_gap_diagnosis_20260618.md` makes that stop
rule numeric. Among 23 plausible public-scored rows, the recent ID85+ subset has
negative local/public rank transfer, and the ID178/ID329 recent gap floor says a
Public `0.58`-plausible local CV would need to be at least `0.517227`; the
ID329 stress gap would require about `0.494927`. Current watch candidates
ID334-341 all proxy to roughly `0.599` or worse under the recent floor, so do
not create ID342 or submit a CSV from local CV alone. Re-run with
`make public-local-gap-diagnosis` after any new real public score is recorded.

`reports/post_id341_data_source_coverage_audit_20260618.md` answers the
"are we actually using/considering all data?" concern. The official metric
files, processed train/test, all twelve raw 2025 modality parquet tables,
external ETRI 2019/2020 CSVs, and paper/data-spec references are present. The
external data is already routed through weak-prior evidence and remains
direct-concat blocked; the upstream Kaggle runner references the external,
raw/sequence phase-2, block-A/B, and clean-backbone checks. This means the
current blocker is not missing local data coverage; it is still transfer
validation and upstream rebuild evidence. Re-run with
`make data-source-coverage-audit`.

The new-source blueprint is
`reports/post_id341_new_source_blueprint_20260618.md`. It selects
`sleep_boundary_transition_v1` as the first clean representation lane after
ID341: use raw screen/activity/charge/light/heart-rate/pedometer transitions to
infer sleep boundaries and interruptions with fold-internal thresholds. This is
not a submission artifact and not a local-CV shortcut. Run it first as a
report-only feature-generation/frozen-split smoke; only promote later if it
beats the ID341 proxy and passes the post-ID341 transfer audit.

Sleep-boundary report-only results are now recorded in
`reports/sleep_boundary_transition_smoke_post_id341_20260618.md`,
`reports/sleep_boundary_transition_v2_gate_post_id341_20260618.md`,
`reports/sleep_boundary_transition_v3_scale_scan_post_id341_20260618.md`, and
`reports/sleep_boundary_prepromotion_gate_post_id341_20260618.md`. The lane is
real but still not promotable: scale `0.05` beats ID341 across all six frozen
lenses with max Δ vs ID341 `0.004759`, but its conservative proxy is `0.601697`,
above the actual ID178 public anchor and not Public `0.58` plausible. Treat it
as a source lead to combine with another independent source or upstream rebuild;
do not create ID342 from this lane alone.

Secondary new-source scans are recorded in
`reports/evening_load_recovery_smoke_post_id341_20260618.md`,
`reports/combined_new_source_gate_post_id341_20260618.md`,
`reports/combined_sequence_sleep_evening_gate_post_id341_20260618.md`, and
`reports/routine_stability_place_entropy_smoke_post_id341_20260618.md`. The
fixed raw place/social novelty follow-up is
`reports/place_social_novelty_smoke_fixedlist_post_id341_20260618.md`.
Evening-load/recovery adds only a tiny `S4` support signal and does not pass all
primary lenses by itself. Combining it with sleep-boundary lowers the best proxy
to `0.601490`, but that is still above ID178 Public `0.601411`; routine-stability
selects no target. The follow-up sequence+sleep+evening gate confirms that
sequence-summary should not be forced into the public-transfer stack now: the
best proxy still uses `sequence_scale=0`, and the best sequence-including
all-primary row is worse (`0.601831`). Raw WiFi/BLE/app identifier novelty was
also checked with corrected list parsing; coverage is good, but no target meets
the frozen selection gate. Therefore the artifact stop rule remains active: no
ID342 from the current new-source stack. Continue only with a materially
different independent raw source or the report-only Kaggle upstream rebuild.

The post-ID341 upstream rebuild phase-1 readiness report is
`reports/post_id341_upstream_rebuild_phase1_readiness_20260618.md`. It confirms
that the required reports, frozen split manifest, source artifacts, and public
anchor records are available for the CPU-only report-first Kaggle runner. This
is readiness evidence only: no submission artifact, no DACON submission, no
Kaggle push/run, and no heavy rebuild are authorized by this result alone.
The refreshed report also requires the latest local new-source closure reports
and asserts `local_new_source_stack_closed=True`; do not bypass that by creating
ID342 from the sequence/sleep/evening or place/social lanes.

The post-ID341 Kaggle upstream execution packet is
`reports/kaggle_upstream_execution_packet_post_id341_20260618.md`. It verifies
that the local source dataset package includes `reports/`, that pre-flight
passes, that the runner remains no-submit, and that the notebook regenerates and
stages the latest transfer-v2/external-prior/combined-external closure reports
and the phase-2 raw/sequence execution spec plus the phase-2 raw multimodal
execution plan before upstream planning. The runner contract now also includes
the block-A bridge, block-A Q1 proxy-gap, block-B S2 recovery closure, and
phase-3 usage temporal microstructure cells, all still no-submit. The package
contract also explicitly requires the Q3/S3 negative-evidence map and its
place/social novelty, routine-stability/place-entropy, and
sensor-adherence/dropout closure reports, so the 8-lane local stop line cannot
be lost by a stale source.zip. The runner now also regenerates and stages
`remote_post_id341_q3s3_negative_evidence_map_20260618.json` before the
remaining-route inventory; output intake treats that remote negative map as a
required report. The runner-intake contract now also proves that the remote
remaining-route inventory is generated with
`--negative-evidence-map {q3s3_negative_evidence_path}`, that the negative map
is staged, and that it asserts `8` closed lanes. After these reports were added,
the source.zip self-test, pre-flight, and local runner smoke were refreshed
under the new required-entry set; the packet is again
`ready_for_explicit_approval=True`, but still performs no push/run by itself. If
the user explicitly approves another Kaggle execution, use only the recorded
base-plus-patch sequence:
`ALLOW_KAGGLE_REMOTE=true make kaggle-push-upstream-patch-dataset`,
`ALLOW_KAGGLE_REMOTE=true make kaggle-check-base-dataset-files`,
`ALLOW_KAGGLE_REMOTE=true make kaggle-check-upstream-patch-dataset-files`,
`ALLOW_KAGGLE_REMOTE=true make kaggle-push-upstream-rebuild-runner`,
`ALLOW_KAGGLE_REMOTE=true make kaggle-kernel-status-upstream-rebuild`,
then `ALLOW_KAGGLE_REMOTE=true make kaggle-wait-and-pull-upstream-rebuild-output`.
The wait-and-pull step waits through running/queued/pending states, pulls only
after a complete/success status, and refuses to pull on unknown or failed
statuses.
`reports/kaggle_source_zip_selftest_post_id341_20260618.md` is the authoritative
evidence that the transient package is self-contained after `build/` removal.
The package now includes the full `submissions/` directory, the ID178/287/329/341
anchor CSVs required by joint-risk thresholds, `Makefile`, and
`dataset-metadata.json`; this fixed the local runner smoke failures where the
packaged project lacked anchor submissions and then failed the blueprint file
check. The current no-network local runner smoke is
`reports/post_id341_kaggle_runner_local_smoke_20260618.md`: it executed the
upstream rebuild notebook end-to-end, staged only 19 JSON and 18 Markdown files
plus `post_id341_upstream_rebuild_runner_manifest.json`, and kept
`create_submission_artifact=False`, `submit_now=False`, `push_kaggle_now=False`,
and `run_kaggle_now=False`.
The smoke report now records input fingerprints for the runner notebook,
package metadata, Makefile, guard scripts, and every script referenced by the
runner. `reports/post_id341_local_smoke_freshness_gate_20260618.md` must remain
fresh before any explicit approval is consumed; if the runner/package inputs
change after the smoke, rerun `make kaggle-local-runner-smoke` before prelaunch.
Before any push/run approval is consumed, also check
`reports/post_id341_kaggle_runner_intake_contract_20260618.md`. It compares the
remote report filenames generated by the runner, staged by the runner, and
required by the output-intake gate. This pre-push contract caught and fixed a
staging omission for the phase-2 plan, block-A, Q1 proxy-gap, block-B, and usage
temporal reports. It also scans the runner code cells for forbidden no-submit
violations such as `.to_csv(...)`, submission/base output paths, DACON submit
APIs, Kaggle network CLI calls, forbidden `True` decision flags, and base
prediction writes; the current forbidden-match counts are all zero. The current
contract passes but still authorizes no Kaggle execution by itself.
The final local prelaunch gate is
`reports/post_id341_kaggle_prelaunch_readiness_gate_20260618.md`. It aggregates
the approval note, execution packet, source.zip self-test, runner/intake
contract, guarded remote contract audit, public/local gap diagnosis, data/source
coverage audit, output-intake pending state, CPU/no-submit kernel metadata, and
Makefile dry-runs. The local target `make kaggle-upstream-prelaunch-local`
regenerates the runner/intake contract, upstream execution packet, local-source
closure synthesis, source.zip self-test, final packet/synthesis pair, launch
approval note, output-intake pending gate, public/local gap diagnosis,
data/source coverage audit, guarded remote contract audit, and prelaunch gate
in that order without network access.
The guarded remote contract audit now checks not only the top-level approval
sequence, but also imports the current prelaunch script's command sequence
instead of trusting a previously generated prelaunch report. It also verifies
that `make kaggle-review-upstream-rebuild-output` runs intake, remote summary,
next-action, and the Public-0.58 artifact-review threshold with the threshold
gate last.
The runtime fail-closed audit is
`reports/kaggle_remote_fail_closed_dryrun_audit_20260618.md`: it intentionally
calls the Kaggle-facing Make targets with `ALLOW_KAGGLE_REMOTE=false` and
`KAGGLE=/usr/bin/false`, and passes only if every target stops at the approval
guard before a Kaggle CLI command can run.
The post-pull review-chain contract is
`reports/post_id341_post_pull_review_chain_contract_20260618.md`; before remote
outputs exist it requires output intake, remote summary, remote next-action, and
the Public-0.58 threshold gate to be ordered, local-only, parseable,
output-pending, and no-artifact/no-submit.
The runner score-relevance contract is
`reports/post_id341_runner_score_relevance_contract_20260618.md`; it ties the
Q2/Q3/S3 missing-target inventory, Q3/S3 local-lane closure, runner/intake
coverage, and post-pull review chain to the clean ID221→ID226 route, so a safe
runner cannot stale-pass if it stops targeting the current score bottleneck.
Its current decision is
`prelaunch_ready_for_explicit_approval=True` while `execute_kaggle_now=False`,
`push_kaggle_now=False`, `run_kaggle_now=False`, `create_submission_artifact=False`,
and `submit_now=False`. Treat this as the launch console: if any input changes,
rerun the gate before consuming user approval.

The local-source closure synthesis is
`reports/post_id341_local_source_closure_synthesis_20260618.md`. It combines the
post-ID341 transfer gate, block-A/B, usage, ambience, external, place/social,
sensor, clean-backbone, and Kaggle packet evidence. Its decision is the current
operating boundary: all required local closure reports exist, no local lane is
promotable, no Public `0.58` reviewable artifact exists, and the next
score-relevant lane is explicit-user-approved report-first upstream
rebuild/Linux parity. Do not create a local ID342-style artifact before that
remote/parity evidence exists.

The exact Kaggle approval note is
`reports/post_id341_kaggle_launch_approval_note_20260618.md`. It is the only
approved command contract if the user explicitly authorizes the next remote
step. The sequence is:
`ALLOW_KAGGLE_REMOTE=true make kaggle-push-upstream-patch-dataset`,
`ALLOW_KAGGLE_REMOTE=true make kaggle-check-base-dataset-files`,
`ALLOW_KAGGLE_REMOTE=true make kaggle-check-upstream-patch-dataset-files`,
`ALLOW_KAGGLE_REMOTE=true make kaggle-push-upstream-rebuild-runner`,
`ALLOW_KAGGLE_REMOTE=true make kaggle-kernel-status-upstream-rebuild`, then
`ALLOW_KAGGLE_REMOTE=true make kaggle-wait-and-pull-upstream-rebuild-output` after
completion. This wait-and-pull wrapper refuses to pull unless Kaggle reports a
complete/success status. The `ALLOW_KAGGLE_REMOTE=true` guard is intentional: network targets
must fail closed unless explicit approval is being consumed.
This note itself performs no network action and still forbids submission CSVs,
base prediction artifacts, notebook-history artifacts, submission-log rows, and
DACON submission.

After any explicitly approved upstream rebuild run is pulled, run
`reports/post_id341_kaggle_output_intake_gate_20260618.md`'s source script:
`.venv/bin/python scripts/diagnose_post_id341_kaggle_output_intake_gate.py`.
This is a local intake gate for `build/kaggle_outputs/upstream_rebuild_post_id341`.
It expects the frozen remote report set plus
`post_id341_upstream_rebuild_runner_manifest.json`, flags parse errors or
accidental submission-like files, and keeps `create_submission_artifact=False`
and `submit_now=False` by default. If outputs are pending, incomplete,
suspicious, manifest-invalid, or contain remote decisions to create/submit/push/run,
artifact-contract review is not allowed. A pulled report signal can only open a
separate review; it never promotes a CSV by itself. The manifest branch is
covered by `tests/test_post_id341_kaggle_output_intake_gate.py`.
Then run
`.venv/bin/python scripts/summarize_post_id341_kaggle_remote_results.py --output reports/post_id341_kaggle_remote_result_summary_20260618.json`.
The summary extracts proxy scores, selected targets, review signals, forbidden
remote actions, and deltas versus actual ID178 Public `0.6014113009` and Public
`0.58`. It is still report-only: even if it marks a remote bundle reviewable, it
can only open a separate artifact-contract review and never creates or submits a
CSV. Its pending/review/forbidden branches are covered by
`tests/test_post_id341_kaggle_remote_summary.py`.
Finally run
`.venv/bin/python scripts/decide_post_id341_remote_next_action.py --output reports/post_id341_remote_next_action_decision_20260618.json`.
This final gate converts the intake+summary evidence into one of three safe
states: wait for outputs, close the remote lane for artifact creation, or allow
only a separate artifact-contract review. It still keeps
`create_submission_artifact=False` and `submit_now=False`; its branches are
covered by `tests/test_post_id341_remote_next_action.py`.
Then run
`.venv/bin/python scripts/decide_post_id341_remote_artifact_review_threshold.py --output reports/post_id341_remote_artifact_review_threshold_20260618.json`.
This stricter gate separates a mere ID178-improvement lead from a Public `0.58`
artifact-review opening. A remote bundle must pass intake, next-action,
public/local-gap, data/source-coverage, no-forbidden-action, ID178-beating, and
`min_proxy <= 0.58` checks before the Public-0.58 artifact-contract review can
open. If it only beats ID178 but stays above `0.58`, record it as an improvement
lead and do not create ID342.
The threshold gate also requires the best proxy row to name non-empty
`selected_targets` and the local source stack to be closed for submission
artifacts. A low proxy without reproducible target scope is not enough to open
artifact review.
The local-only shortcut for the whole post-pull chain is
`make kaggle-review-upstream-rebuild-output`. It runs intake, summary, and
next-action plus the strict artifact-review threshold in order and performs no
Kaggle/DACON network operation.
Use `make kaggle-upstream-prelaunch-local` before approval to refresh the launch
console without pushing or pulling anything.

The phase-2 raw/sequence execution spec is
`reports/post_id341_phase2_raw_source_execution_spec_20260618.md`. It is a
contract for the next approved runner revision, not permission to run it now.
Allowed next work is a report-only raw multimodal/sequence source smoke;
forbidden outputs include submission CSVs, base prediction artifacts,
notebook-history artifacts, and submission-log changes. Any temporary source
must pass transfer and joint-risk gates before artifact review.

The concrete phase-2 raw multimodal execution plan is
`reports/phase2_raw_multimodal_execution_plan_post_id341_20260618.md`. It
narrows the next Kaggle CPU runner revision to a block order, not a broad
search: first run `A_q1_s4_sleep_residual_bridge`, using raw sleep-boundary
signals plus Q1/S4 residual-drift clues. Stop if this first block does not beat
ID341 on all primary splits or remains above actual ID178 Public by proxy. Only
after that may S2 recovery retry or environment/social routine blocks be
considered. This plan creates no artifact and does not authorize Kaggle push/run.

The local block-A smoke is
`reports/phase2_block_a_q1_s4_sleep_residual_bridge_post_id341_20260618.md`.
It combines `294` sleep-boundary features with `399` residual features. Q1 is
selected (`hgb_l4_l2_5`, weight `0.05`) and the assembled Q1-only bridge beats
ID341 on all five primary splits, including `subject_hole_5` and `loso_10`.
However the projected proxy is still `0.603617`, above actual ID178 Public, so
`final_block_a_gate_pass=False`; this is partial source evidence, not a
submission path.

The follow-up Q1 proxy-gap report is
`reports/phase2_block_a_q1_proxy_gap_post_id341_20260618.md`. It is pure gap
math over the block-A report and trains no model. It confirms the Q1 signal is
directionally clean but too small: the current proxy is still `+0.002206` above
actual ID178 Public, and a Q1-only path would need roughly `11.37x` the current
target delta to beat ID178 and `65.02x` to become Public `0.58` reviewable. Do
not widen this into a larger Q1 overlay cascade; the next score-relevant work
needs an independent target/source such as the planned S2 recovery or another
raw upstream source.

The broader target/source gap budget is
`reports/post_id341_target_gap_budget_20260618.md`. It reads the existing
post-ID341 source reports and asks whether their target-level deltas are enough
before creating any artifact. Under the strict all-primary rule, only Q1 and S4
survive; their combined projected proxy is `0.602907`, still `+0.001496` above
ID178 Public. Even the looser 4/5 watch budget over Q1/S1/S2/S4 projects
`0.602256`, still above ID178. This makes the next focus explicit: do not spend
more local cycles on Q1/S4 micro-tuning; look for independent Q2/Q3/S3 signal or
run the approved no-submit Kaggle upstream rebuild.

The missing-target inventory is
`reports/post_id341_missing_target_signal_inventory_20260618.md`. It confirms
Q2/Q3/S3 have zero current local selected rows, while S1/S2 are only watch-level
signals. Treat Q2/Q3/S3 as an upstream-rebuild surface, not a local blend
surface: the next allowed score-relevant path is the clean ID221→ID226
report-first Kaggle rebuild after explicit approval, with no local ID342
artifact in between.
The follow-up route report is
`reports/post_id341_q2q3s3_source_route_20260618.md`. It ranks the explicit
approved clean ID221→ID226 upstream rebuild as the primary score-relevant route
because it covers all missing Q2/Q3/S3 targets. If approval is not given, the
only local fallback is a materially different Q2/Q3/S3-first
environment/social/routine design; more Q1/S4/S2 tuning is explicitly blocked
because it does not cover the missing-target gap.
That fallback design is now recorded in
`reports/post_id341_q2q3s3_env_social_routine_v2_design_20260618.md`. It
profiles the raw nested sources (`mWifi`, `mBle`, `mUsageStats`, `mAmbience`)
and confirms the lane must avoid direct BSSID/address/app-name memorization.
The next local step, if no Kaggle approval is given, is feature-generation
metadata smoke only: Q2/Q3/S3 routine stability, place/social novelty, and
ambient-load context features fitted inside each outer fold before any model
training or artifact review.
That metadata smoke is now recorded in
`reports/post_id341_q2q3s3_feature_generation_metadata_smoke_20260618.md`. It
builds label-free daily raw-source features for all `450` train and `250` test
keys, with `247` raw feature columns across the routine, place/social, and
ambient context blocks. The leakage guard passes: no direct
BSSID/address/app-name model columns, no target columns in the feature frame, no
model training, no submission CSV, no submission-log row, no Kaggle push/run,
and no DACON submission. The next local step is therefore limited to a
fold-internal Q2/Q3/S3 model smoke against ID178/ID341 anchors; it is still not
an artifact-promotion step.
The fold-internal model smoke is
`reports/post_id341_q2q3s3_fold_internal_model_smoke_20260618.md`. It trains
only small fold-internal logistic source models on the metadata-smoke features
and compares them with ID178/ID341 OOF anchors. Q2 shows a real local signal:
`blend_id341_q2q3s3_lr_c0_08_w0.05` wins 5/5 primary split lenses with mean
delta `-0.002758` versus ID341 and max delta `-0.000201`. Q3 and S3 do not pass
even tiny positive-movement blends. Treat this as a Q2 lead only, not Public
`0.58` evidence and not an artifact-promotion authorization. The next allowed
step is a no-artifact geometry gate for the Q2 lead; if it fails, close the
local fallback and return to the explicit-approved upstream rebuild.
The Q2 lead geometry gate is
`reports/post_id341_q2q3s3_q2_lead_geometry_gate_20260618.md`. It computes
full-test predictions in memory only and writes no prediction artifact. The Q2
lead is low-movement-reviewable at scale `0.25` with effective weight `0.0125`,
5/5 primary split wins, local subject-hole loss `0.538556`, and proxy
`0.601911`; however this is still far above the Public `0.58` target and Q3/S3
remain uncovered. Keep the Q2 lead as context only. Do not create a submission
artifact from this lane unless a later independent Q3/S3 source or the
explicit-approved upstream rebuild provides broader missing-target coverage.
The processed-feature paper-family smoke is
`reports/post_id341_q2q3s3_paper_family_fold_smoke_20260619.md`. It tests the
reference-style families directly: Q2 uses `mUsageStats+DW`, Q3 uses
`wPedo+mUsageStats+DW`, and S3 uses `wPedo+mLight/wLight+DW`. Despite full
processed train/test coverage and low missingness, no target passes the frozen
primary gates: Q2 has 0/5 wins in its best rows, and Q3/S3 only reach 1/5 with
positive mean deltas. Treat this as another closed no-approval local lane, not
as evidence against the paper itself; it says this repo's current processed
paper-family implementation is not a Public `0.58` path.
The Q3/S3 anchor-failure profile is
`reports/post_id341_q3s3_anchor_failure_profile_20260619.md`. It compares
existing OOF anchors against known public outcomes. ID287 and ID329 both show
Q3/S3 local OOF gains with 5/5 primary wins, yet both score worse than ID178 on
public. ID226 still has Q3/S3 OOF signal, but this profile classifies that as
rebuild-prior evidence only: do not submit ID226 directly and do not create a
new local artifact just because Q3/S3 OOF improves.
`make kaggle-upstream-prelaunch-local` now regenerates this full no-artifact
fallback chain in order: source route, env/social/routine design, metadata
smoke, fold-internal model smoke, Q2 geometry gate, Q3/S3 local closure, and
local fallback-chain contract. This prevents stale Q2-only evidence from being
used in the launch console.
The independent Q3/S3 physio-light-onset smoke is
`reports/post_id341_q3s3_physio_light_onset_smoke_20260618.md`. It tests
wearable heart-rate, pedometer movement, wearable/mobile light, and charging
transition features for stress/SOL-like onset behavior. Coverage is complete
(`450` train / `250` test, `121` raw feature columns), but no target passes:
Q3's best row wins only 3/5 primary lenses with positive mean delta, and S3 wins
0/5. This lane is therefore closed locally and creates no artifact.
The resulting local-lane closure is
`reports/post_id341_q3s3_local_lane_closure_20260618.md`. It keeps the Q2 lead
as context only, records Q3/S3 as still uncovered, and closes the no-approval
local fallback for submission artifacts. The next score-relevant route is the
explicit-approved clean ID221→ID226 no-submit upstream rebuild.

`make kaggle-upstream-prelaunch-local` now regenerates both
`post_id341_target_gap_budget_20260618` and
`post_id341_missing_target_signal_inventory_20260618` before rebuilding the
source package self-test. The final prelaunch gate requires
`missing_target_inventory_ready=True`, so explicit approval cannot be consumed
from a stale launch console that forgot the Q2/Q3/S3 local stop line.

The S2 recovery retry is recorded in
`reports/phase2_block_b_s2_sleep_recovery_retry_post_id341_20260618.md`. It is
a bounded exception to the block-A stop rule, run only to test whether an
independent S2 sleep/recovery source exists. It combines `294` sleep-boundary,
`399` subject-residual, and `148` evening/recovery columns, but no S2 config
passes the fixed eligibility rule; the best rows win at most 3/5 primary target
lenses and have positive mean delta. The assembled candidate therefore keeps
ID341 unchanged with proxy `0.603830`. Treat this as closure evidence: do not
create an S2 recovery submission artifact from this feature family.

The post-A/B usage temporal microstructure smoke is
`reports/phase3_usage_temporal_microstructure_smoke_post_id341_20260618.md`.
It tests a materially different raw source, `mUsageStats`, after removing
direct app-name columns and keeping only coarse category/time-bin load, entropy,
concentration, and late-use features. Coverage is good (`158,426` app events,
`302` usage feature columns, all train/test rows covered), but no target passes
the fixed eligibility rule. The best rows are isolated split gains rather than
transfer-safe target signals, and the assembled candidate remains ID341 with
proxy `0.603830`. Do not create ID342 from usage temporal microstructure.

The ambience context closure audit is
`reports/phase3_ambience_context_closure_post_id341_20260618.md`. It reads the
existing strict semantic `mAmbience` diagnostics rather than training again.
This family has a tempting historical S4 full-OOF signal, but it is exactly the
kind of trap we are trying to avoid: the best broad S4 row improves full OOF by
`-0.002880` and then worsens nested validation by `+0.003467`; the narrow confirm
still worsens nested by `+0.000642`. Do not replay ambience overlays or create
ID342 from this source unless a future upstream rebuild nests source generation
before any selector choice.

Subject residual sequence-drift smoke is recorded in
`reports/subject_residual_sequence_drift_smoke_post_id341_20260618.md`. It
uses processed numeric covariates only, builds past-only per-subject
drift/volatility features, and fits compact fold-internal source models on the
primary frozen splits. Result: Q1 wins 5/5 target lenses, S2 wins 4/5, and S4
wins 3/5, but the combined best candidate beats ID341 on only 1/5 primary
splits. The projected proxy is `0.603816`, still above the actual ID178 public
anchor and far from Public `0.58`. Treat it as feature/source-design evidence,
not a submission candidate; do not create ID342 from this lane alone.

The target-restricted follow-up is
`reports/subject_residual_target_restricted_gate_post_id341_20260618.md`. It
fixes the scan to Q1/S2/S4 and selects only Q1 and S4 tiny ID178-anchored
ExtraTrees residual overlays (`w=0.01`). The assembled candidate improves 3/5
primary splits but still fails `subject_hole_5` and `loso_10`; projected proxy
is `0.603784`, not below actual ID178 Public. This confirms residual drift is a
small design hint, not a promotable local artifact.

The upstream source priority matrix is
`reports/post_id341_upstream_source_priority_matrix_20260618.md`. It ranks the
existing source artifacts as rebuild targets, not as submission candidates. The
first clean rebuild focus is `ID221` plus `ID226`; `ID227/228/237` and A1/B1
artifacts remain replay/hold targets because their stronger local losses are
paired with full-OOF or known-failure geometry exposure. Do not create ID342
from these existing artifacts.

The clean-backbone rebuild design is
`reports/post_id341_clean_backbone_rebuild_design_20260618.md`. It is the
contract for the next heavy Kaggle revision if the user explicitly approves a
push/run: implement only `clean_id221_feature_similarity_root` and
`clean_id226_targetwise_convex_backbone`, using nested/fold-internal fitting and
selection. Keep ID227/228/237 and A1/B1 replay blocked until those clean reports
pass the transfer gates.

Clean-ID221 local report-only probes are recorded in
`reports/post_id341_clean_id221_feature_similarity_smoke_20260618.md`,
`reports/post_id341_clean_id221_inner_selection_probe_20260618.md`, and
`reports/post_id341_clean_backbone_next_step_decision_20260618.md`. The fixed
and inner-selected ID221-like feature-similarity variants beat ID178/ID341, but
do not beat ID218 or historical ID221. Treat clean ID221 as an anchor/control;
the next useful no-submit lane is clean ID226 fold-internal convex/calibration
smoke, not more ID221 tuning.

Clean-ID226 and clean-backbone closure are recorded in
`reports/post_id341_clean_id226_calibration_smoke_20260618.md` and
`reports/post_id341_clean_backbone_lane_closure_20260618.md`. Conservative
fold-internal calibration on ID218 does not beat ID218 and fails the primary
ID178/ID341 gates, so the clean-backbone lane is closed. Do not replay
ID227/228/237 or create ID342 from this family. Move next to a stronger
raw/sequence source model or another independent source family.

Sensor-adherence/dropout source smoke is recorded in
`reports/sensor_adherence_dropout_smoke_post_id341_20260618.md`. It uses
label-free row-level modality completeness/dropout/adherence features and no
submission artifacts. No target passed the frozen selection gate; Q1/S4 were
only 3/5 primary-win near misses. Treat this lane as closed standalone and do
not promote or combine it unless a separate independent source first passes the
post-ID341 gate.

The public joint-distribution risk diagnostic is
`reports/public_joint_distribution_risk_post_id341_20260618.md`. It uses only
scored submission CSVs and train-label structure to add report-level warning
metrics: target-mean drift, pairwise co-occurrence drift, prediction
extremeness/entropy, temporal volatility, and distance from ID178. Use these as
candidate-selection cautions before any future artifact review; do not tune a
submission directly to these metrics or claim Public `0.58` evidence from them.

The joint-risk candidate rerank is
`reports/joint_risk_candidate_rerank_post_id341_20260618.md`. It applies the
same warning lens to all loadable logged candidate CSVs. The local-CV frontier
is mostly high-risk; ID337/339/340/341 remain low-warning controls, with ID341
lowest-movement. This is not a candidate selector by itself: continue to require
the post-ID341 transfer gate, artifact contract, and explicit approval before
any public-facing action.

The combined post-ID341 transfer gate v2 is
`reports/post_id341_transfer_gate_v2_joint_risk_20260618.md`. It recomputes the
transfer lens and joint-risk lens together and requires both to pass before a
candidate can be called Public `0.58` reviewable. Current result: no candidate
passes. A low-warning candidate that remains above the transfer proxy gate is
not submission-ready, and a low-local candidate rejected by joint risk is not a
frontier candidate. Do not submit or promote any same-family local recombination
unless this combined gate first passes and a separate decision note names the
exact CSV.

The external ETRI prior smoke is
`reports/external_etri_prior_smoke_post_id341_20260618.md`. It confirms the
2019/2020 same-institution CSVs can provide weak priors and feature ideas, but
not direct rows for current training. Objective sleep prevalences differ
materially from the current 2024 challenge, and the best tiny prior overlay
still fails the combined Public `0.58` path. Do not concatenate external rows
into the current train set unless a future leak-free validation note proves it;
for now use them only as reference priors or source-design hints.

The combined sleep/evening/external-prior gate is
`reports/combined_sleep_evening_external_prior_gate_post_id341_20260618.md`.
It tests whether the surviving non-sequence weak leads combine into a safer
candidate. Current result: the best proxy row uses external scale `0`; external
priors do not improve the evening-only micro lead and no row becomes reviewable.
Treat this as closure for local weak-source stacking around ID341 unless a new
raw source family enters the gate.

The Q2/Q3/S3 source-hypothesis matrix is
`reports/post_id341_q2q3s3_source_hypothesis_matrix_20260618.md`. It records
that Q2, Q3, and S3 are all missing locally, while historical ID226 has
negative target deltas and all-fold wins for those targets. Use it as the
target-specific design compass for the explicit-approved no-submit Kaggle
upstream rebuild. It does not authorize ID342, a submission artifact, Kaggle
execution, or DACON submission.
The upstream runner, output-intake gate, execution packet, and source.zip
self-test now require the same matrix as a staged remote report, so any approved
Kaggle no-submit run must carry this Q2/Q3/S3 design evidence into post-pull
review.

## Submission State Rules

Use `status_clean` as the operational state, not raw `is_submitted`.

- `scored`: real public LB score exists and is plausible.
- `generated_unsubmitted`: local artifact exists but has not been public-scored.
- `submitted_unscored` / `submitted_placeholder_score`: historical automation state; do not use for default ensembling.
- `cv_lb_gap_risk`: local CV and public LB diverged too much; treat as validation failure.
- `invalid_public_score`: penalty-scale or malformed score; never ensemble by default.

## Artifact Rules

Keep raw data, model binaries, OOF files, notebook logs, and training logs out of Git. Track summaries and scripts, not bulk artifacts. Generated experiment logs under `lightning_logs/`, `catboost_info/`, `checkpoints/`, and `scratch/` are ignored; if they were previously tracked, remove them with `git rm --cached` only, never by deleting local files.

For `reports/`, promote only curated decision records that are needed to
understand a submitted or submission-ready candidate. Bulk seed sweeps, grid
scans, and temporary diagnostics should remain untracked or move to ignored
`artifacts/` unless explicitly selected for the permanent record.

## Session and Tooling Rules

Use a fresh session before switching from audit/docs work into modeling. This
prevents overloaded context from mixing cleanup, validation, and candidate
generation decisions. Do not claim that subagents were used unless the runtime
actually provides subagent execution; local parallel audit commands are not
subagents.

## Commit Boundaries

Commit hygiene changes separately from modeling changes. A good cleanup commit includes scripts, docs, Makefile updates, `.gitignore`, and normalized metadata. It should not include new model binaries, raw data, or leaderboard submissions unless explicitly intended.

Post-ID341 Q3/S3 event-cadence/silence-gap smoke is recorded in
`reports/post_id341_q3s3_event_cadence_smoke_20260618.md`. It uses only raw
timestamp cadence, inter-event gaps, and modality silence windows across phone
and watch streams. It selects no Q3/S3 target and is folded into the negative
evidence map as a closed local lane. The same negative evidence map also folds
in the previously report-only place/social novelty, routine-stability/place
entropy, and sensor-adherence/dropout smokes. Current Q3/S3 closed local lane
count is `8`, and none authorize a submission artifact. Do not reopen these as
submission artifact paths; the score-relevant route remains the
explicit-approved clean ID221→ID226 no-submit upstream rebuild.

Post-ID341 Q3/S3 fold-nested sibling bridge smoke is recorded in
`reports/post_id341_q3s3_fold_nested_sibling_bridge_smoke_20260619.md`. It
tests the natural Q3←Q2 and S3←S2 label-structure idea using only nested sibling
predictions, never same-row sibling labels. The smoke selects no Q3/S3 target,
keeps Public `0.58` evidence false, and remains report-only. Do not create
ID342 or submit a local sibling-bridge artifact from this result; keep it as
upstream design context unless a future independent source covers Q3/S3.
The remaining-route inventory, prelaunch readiness gate, and runner
score-relevance contract now require this closure explicitly, so a green Kaggle
handoff also means the sibling-bridge shortcut is closed locally.
The local prelaunch target and no-submit Kaggle runner regenerate the same
sequence in order: Q3/S3 negative-evidence map, fold-nested sibling smoke, then
remaining-route inventory. If that order breaks, approval readiness should fail
closed rather than relying on a stale sibling report.

Before consuming any explicit Kaggle approval, prefer
`make kaggle-upstream-ready-local`. This wrapper remains local/no-network and
runs the focused approval contract tests, the local runner smoke, the local
smoke freshness gate, and final prelaunch readiness in one safe order. It does
not replace the approval phrase, and it must not be followed by any Kaggle
network command unless the user separately approves the exact no-submit remote
sequence.

Post-ID341 external Public `0.5917` residual audit is recorded in
`reports/post_id341_external_public05917_residual_audit_20260619.md`. It keeps
the reference notebook useful as validation philosophy only: anchor blending,
Subject-Hole awareness, stability-selected features, and low-variance candidate
ranking remain good ideas, but the notebook is not reproducible as a direct
artifact route and provides no Public `0.58` evidence after the ID178/287/329
public checks. The no-approval frontier and local prelaunch target now require
this closure, so do not reopen the external notebook as a direct local ID342
submission path.

Post-ID341 GPS path-topology is now closed as a local artifact route. The valid
feature matrix and model smoke are useful negative evidence, but the smoke
selected no Q3/S3 target and opened no Public `0.58` route. Do not replay GPS
metadata/spec/matrix/model-smoke with tiny weight changes as ID342 work.

Post-ID341 circadian HR/light phase-shape metadata is recorded in
`reports/post_id341_circadian_hr_light_phase_metadata_probe_20260619.md`. It
uses raw `wHr`, `wLight`, `mLight`, and `wPedo` only as a resource/coverage
probe. Strong phase-shape coverage is `0.5622` train and `0.6120` test; weak
coverage is `0.7489` train and `0.8280` test. This is enough for a report-only
feature spec, not model training, not a submission artifact, and not Public
`0.58` evidence.

Post-ID341 phone/wearable discordance v1 now has a report-only feature spec, an
in-memory feature-generation dry run, a feature-matrix smoke contract, a
registered label-free matrix/schema report, and a validation-stop report:
`reports/post_id341_phone_wearable_discordance_feature_spec_20260619.md` and
`reports/post_id341_phone_wearable_discordance_feature_generation_dryrun_20260619.md`,
plus
`reports/post_id341_phone_wearable_discordance_feature_matrix_smoke_contract_20260619.md`,
`reports/post_id341_phone_wearable_discordance_features_v1_schema.md`, and
`reports/post_id341_phone_wearable_discordance_feature_matrix_validation_stop_20260619.md`.
The dry run assembles the frozen 85-column `pwd_` schema over 450 train and 250
test rows with finite bounded values, and the writer materializes only the
single registered label-free parquet. The validation-stop confirms that no
model, prediction file, submission CSV, submission-log row, Kaggle run, DACON
submission, or ID342 is authorized from this matrix.

Post-ID341 DACON submission spend policy is now enforced before the DACON API
call. `make dacon-submit` still requires `ALLOW_DACON_SUBMIT=true` plus exact
`DACON_APPROVED_SUB_PATH` and `DACON_APPROVED_SUB_MEMO`, but it also runs
`scripts/audit_dacon_submit_approval_contract.py` against
`reports/post_id341_submission_spend_policy_20260619.json`. Under the current
policy, score-improvement submission is closed and no fresh same-family manual
sanity probe is recommended. Other generated-unsubmitted files, including
lower-local-CV or lower-proxy probes, fail closed even if their path and memo
are matched.
The same DACON submit approval contract also checks freshness. The policy report
must be newer than `submission_log.csv` and its decision inputs
(`manual_submission_probe_decision`, `safe_wait_state`, `public058_promotion`,
`public_local_gap`, and `next_material_route_triage`). If any of those inputs is
refreshed after the policy report, `make dacon-submit` fails before importing the
DACON API. Re-run `make post-id341-submission-spend-policy` after refreshing
those decision inputs and before asking for exact DACON approval.

2026-06-21 DACON daily-slate reconciliation:

- ID341 is marked submitted-unscored from the earlier 2026-06-19 DACON HTTP-200
  response that occurred before a local pandas dtype failure. It must not be
  resubmitted just to repair local metadata.
- ID340 submitted successfully and scored `0.601379175253277`.
- ID337 submitted successfully and scored `0.6006206286219425`.
- ID221 was attempted but rejected with `day_max_count`; ID255 was not
  attempted after that stop.

Operationally, this means the daily slate did buy useful information but did not
unlock a Public `0.58` route. Future same-neighborhood ID178 overlays should be
treated as low-value probes unless they are tied to a validation/source repair
hypothesis that explains ID178/287/329/337/340 together.

2026-06-23 Kaggle upstream rebuild v3 result:

- Status: `KernelWorkerStatus.COMPLETE`.
- Fixed issue: v2's missing `dataset-metadata.json` gate was repaired by
  runtime metadata synthesis; the remote blueprint is now `blueprint_ready=True`.
- Review result: output is no-submit and complete, but not promotable. The
  sanitized post-pull review found all 25 expected reports, no parse errors, no
  forbidden remote actions, no review-signal rows, no proxy below ID178, and no
  Public `0.58` evidence.
- Best proxy: `0.6014904915736387`, slightly worse than ID178 Public
  `0.6014113009`.
- Operational decision: do not create any new CSV/base/notebook/log artifact
  from Kaggle v3. If the clock reaches the user's 23:30 KST submission-spend
  line without a stronger gated candidate, use
  `reports/dacon_three_slot_slate_20260623.json` as
  `DACON_SUBMIT_POLICY_REPORT`.

2026-06-23 23:30 KST submission-spend result:

- ID338 submitted successfully with DACON `isSubmitted=True`, HTTP 200.
- ID342 submitted successfully with DACON `isSubmitted=True`, HTTP 200.
- ID339 submitted successfully with DACON `isSubmitted=True`, HTTP 200.
- Authenticated submission history confirmed ID342 as the team-best row:
  DACON submission `1467937`, score `0.5960832519`, rank `220`, time
  `2026-06-23 23:30:37`. This improves the previous confirmed best
  `0.6006206286219425` by `-0.004537376722`.
- Authenticated submission history corrected the initial unauthenticated
  leaderboard inference: ID342 scored `0.5960832519`, ID339 scored
  `0.5985187787`, and ID338 scored `0.5992145177`.
- Current corrected result record:
  `reports/dacon_three_slot_submission_result_20260623.md/.json`.
- The prior `reports/post_id338_public_feedback_next_action_20260623.md/.json`
  remains useful only as an ID338-neighborhood diagnostic. It should not be read
  as the current public-anchor record after the authenticated ID342 correction.

2026-06-23 post-ID338 material Q3/S3 follow-up:

- `reports/post_id338_material_q3s3_probe_20260623.md/.json` combines the
  registered material raw-source matrices and evaluates only Q3/S3 overlays
  against ID338. Q3 remains closed. S3 selects a tiny phone/wearable
  discordance overlay: logistic `C=0.20`, weight `0.005`, 4/5 primary split
  wins, mean delta vs ID338 `-0.000062`, and mean S3 movement `0.000573`.
- `reports/post_id338_material_q3s3_geometry_gate_20260623.md/.json` rebuilds
  that S3 overlay in memory on test rows and passes the no-artifact geometry
  gate: mean all-target movement vs ID338 `0.00011334`, max movement
  `0.00365242`, no risk flags, and no Public `0.58` evidence.
- ID343 was generated as a full-lineage, unsubmitted reserve artifact:
  `submissions/submission_id_343_id338_s3_phone_wearable_micro_w0005_cv0.537854_unsubmitted.csv`.
  It changes only S3, keeps all other targets at ID338, records local CV
  `0.537854`, and has matching base OOF/test, metadata, decision note, notebook
  history, and submission-log row. It is not a DACON submission recommendation
  by itself; use it only as a future controlled S3 probe candidate after fresh
  exact approval.
- `reports/post_id338_s3_phone_wearable_weight_geometry_scan_20260624.md/.json`
  compares S3 phone/wearable weights after ID343. Weight `0.02` passes the
  no-artifact geometry/probe gate with 4/5 primary wins, mean delta vs ID338
  `-0.000192`, max split delta `+0.000267`, mean all-target movement
  `0.00045335`, max all-target movement `0.01460969`, and no risk flags.
- ID344 was generated as the bolder full-lineage, unsubmitted S3 reserve
  artifact:
  `submissions/submission_id_344_id338_s3_phone_wearable_w0020_cv0.537839_unsubmitted.csv`.
  It changes only S3 at weight `0.02`, improves local CV vs ID338 by
  `-0.00002210`, and has matching base OOF/test, metadata, decision note,
  notebook history, and submission-log row. It is still not Public `0.58`
  evidence and still requires fresh exact approval before any DACON submission.
- `reports/post_id338_q3_deep_material_scan_20260623.md/.json` then expanded
  Q3 material-source scanning across low-capacity LR, HGB, ExtraTrees, and
  RandomForest variants. No Q3 row passed the ID338-preserving gate; the best
  rows remain circadian HR/light micro-blends with only 2/5 primary wins and
  positive mean delta vs ID338. Do not promote a Q3 material-source artifact
  from this family.

2026-06-24 post-ID342 missingness subject-state no-submit closeout:

- Kaggle runner:
  `softkleenex/dacon-etri-missingness-subject-state-id342`, version 3
  complete.
- Pulled outputs:
  `build/kaggle_outputs/missingness_subject_state_post_id342/`.
- Intake:
  `reports/post_id342_missingness_subject_state_remote_intake_20260624.md/.json`,
  `remote_intake_ready=True`.
- Coverage: all `6,122,179` raw rows streamed, `5,678,719` train/test-key rows
  matched, mean modality-key coverage `0.976905`, train/test any observed
  `1.0` / `1.0`.
- Score gate: `0` stable passes; best config
  `missingness_subject_state_q3_lr_c0_02_w0.005`; projected Public proxy
  `0.596101196`, gap `+0.046101196` to Public `0.55`.
- Decision: no feature matrix, prediction artifact, submission CSV,
  submission-log row, or DACON submission. This route is research evidence
  only and should not consume a leaderboard submission slot.

2026-06-24 Public-0.55 OOF-shelf gap attribution:

- Report:
  `reports/post_id342_public055_gap_attribution_20260624.md/.json`.
- OOF files scanned: `247`.
- Stable public-minus-local median gap: `0.062318739`.
- Required full local for Public `0.55`: `0.487681261`.
- Existing rows below required local: `64`.
- Credible rows below required local after frontier/direct-lite/delta-pair
  /riskpen and distribution-move filters: `0`.
- Best low-movement shelf row: ID211, projected Public `0.582424274`.
- Decision: no existing shelf candidate is worth a DACON slot. Continue only
  with public-gap/validation repair or a materially new source hypothesis.

2026-06-24 public-gap risk lens:

- Report: `reports/post_id342_public_gap_risk_lens_20260624.md/.json`.
- OOF files scanned: `247`; scored rows: `8`.
- Stable-shape envelope rows: `11`.
- Public-0.55 shape-lens pass count: `0`.
- Best shape-envelope row: ID342, projected Public `0.596318745` under the
  stable gap.
- Strong warning signals: mean movement vs ID342 Pearson `0.991` with public
  score; extreme-probability rate Pearson `0.997` with public score.
- Decision: next experiments must stay inside the stable movement envelope or
  repair public-gap validation. Do not enlarge frontier/direct-lite/delta-pair
  /riskpen blends.

2026-06-25 stable-envelope microblend and multi-source optimizer:

- Reports:
  `reports/post_id342_stable_envelope_microblend_scan_20260625.md/.json` and
  `reports/post_id342_stable_envelope_multisource_optimizer_20260625.md/.json`.
- The single-source scan reviewed `3198` ID342-convex microblend rows from
  `246` sources. The best envelope row is legacy-risk ID250 at projected Public
  `0.563921703`, so it is not a candidate. The best nonlegacy gate row is ID285
  alpha `0.12`, projected Public `0.569887753`.
- The nonlegacy multi-source optimizer rescanned the full shelf, found `101`
  nonlegacy gate sources, enforced the same stable-shape envelope, and retained
  `16153` states. The best row is ID285 weight `0.125` plus ID278 weight
  `0.015`, local `0.504044060`, projected Public `0.566362799`, still
  `+0.016362799` from Public `0.55`.
- Decision: no Public-0.55 envelope rows, no artifact review, no Kaggle action,
  and no DACON submission. Treat this as evidence that same-neighborhood
  post274/post278/post282/post285 recombination is compressed; the next useful
  work needs independent source signal or a validation/public-gap redesign.

2026-06-25 target-wise reallocation optimizer:

- Report:
  `reports/post_id342_target_reallocation_optimizer_20260625.md/.json`.
- Strict exclusions: confirmed failed-public IDs `287/329` plus public-guided
  frontier/direct-lite/delta-pair/riskpen/public-probe/outer-supported/oracle
  tokens.
- The scan reviewed `420` target-specific ID342-convex options and retained
  `2462` envelope-safe beam states. Allowing each target to choose a different
  source improves the stable-gap projection only from `0.566362799` to
  `0.566166687`; Public-0.55 envelope rows remain `0`.
- Target read: Q2/Q3 can be pushed down locally by ID285-like directions, but
  this spends nearly all mean movement budget. S4/WASO remains the largest
  target loss at `0.567825`, and S3/SOL changes only `-0.002828` vs ID342.
- Decision: strict existing-source target reallocation is closed. Continue with
  genuinely new S4/S3 signal or a validation/public-gap redesign; do not create
  artifacts or DACON submissions from this result.

2026-06-25 S4/S3 processed sleep-fragmentation and raw event feasibility:

- Reports:
  `reports/post_id342_s4s3_sleep_fragmentation_gate_20260625.md/.json` and
  `reports/post_id342_raw_sleep_event_reconstruction_feasibility_20260625.md/.json`.
- The S4/S3 gate tests WASO/fragmentation core features, sleep-onset context,
  sleep HR/awake-light ratios, and their union. It passes `0` rows. S4's best
  row (`sleep_hr_awake_light_s4_hgb_l7_w0.05`) has mean delta `-0.000272711`
  but fails stability with positive subject-hole/LOSO deltas; S3's best row is
  positive on average (`+0.000075639`).
- The raw feasibility audit scans all `12` public raw parquet files and finds
  `0` direct sleep-stage, SOL, or WASO event columns. Seven proxy modalities are
  available (`wHr`, `wLight`, `mLight`, `mScreenStatus`, `mACStatus`,
  `mActivity`, `wPedo`), but those proxy families already failed the processed
  S4/S3 gate.
- Decision: no processed-column overlay, no raw direct-event reconstruction,
  no Kaggle action, and no DACON submission. The next meaningful route is
  either a genuinely sequence-level HR/light event model with a fresh local
  gate, or a validation/public-gap redesign.

2026-06-25 S4/S3 raw HR/light sequence event gate:

- Report:
  `reports/post_id342_sequence_hr_light_event_gate_20260625.md/.json`.
- Raw sources: `wHr`, `wLight`, `mLight`, `mScreenStatus`, `mACStatus`,
  `mActivity`, and `wPedo` over an 18:00-to-next-noon relative window.
- Feature contract: `807` numeric event columns before ranking, `640` used,
  mean missing rate `0.222455`, no feature matrix materialized.
- S4 result: `5` eligible rows. The selected row is
  `s4_id342_sequence_hr_light_event_s3_onset_settling_lr_c0_10_w0.01`,
  with `4/5` primary wins, mean delta `-0.000254566`, and max delta
  `+0.000695438`.
- S3 result: `0` eligible rows. The best row has `3/5` wins, mean delta
  `-0.000049330`, and positive subject-hole/LOSO-style instability.
- Projection: estimated all-target delta only `-0.000036367`; projected Public
  proxy `0.596046885`, still `+0.046046885` from Public `0.55`.
- Decision: this is research evidence only. It does not justify a feature
  artifact, prediction artifact, submission CSV, submission-log row, Kaggle
  run, or DACON submission. Local raw event reconstruction is closed as an
  immediate ID342 overlay route unless a stronger validation/public-gap design
  is introduced.

2026-06-25 public-gap penalty proxy and current route synthesis:

- Report:
  `reports/post_id342_public_gap_penalty_proxy_redesign_20260625.md/.json`.
- The validation redesign audit scans `247` OOF files and `8` public-scored
  rows. It fits a small nonnegative shape-excess penalty over the stable ID342
  envelope.
- The proxy improves pairwise public ordering to `1.0`, but LOO MAE is unstable
  (`0.106258281`), so it is a stricter review lens rather than a score
  predictor.
- Public-0.55 penalty proxy pass count: `0`; credible Public-0.55 pass count:
  `0`.

2026-06-25 MIS-LSTM-style block-sequence feasibility gate:

- Report:
  `reports/post_id342_mis_lstm_block_sequence_feasibility_gate_20260625.md/.json`.
- The gate builds in-memory 3-hour/4-hour block features from `wHr`, `wLight`,
  `mLight`, `mScreenStatus`, `mACStatus`, `mActivity`, and `wPedo`, separating
  continuous sensor channels from sparse screen/charging/activity events.
- Feature contract: `1257` numeric columns before ranking, `720` used, mean
  missing rate `0.271774`, no feature matrix materialized.
- Result: S4 selects `s4_id342_mis_blockseq_sparse_event_channels_lr_c0_10_w0.03`
  with `4/5` wins and mean delta `-0.001468349`; Q3 and S3 pass `0` rows.
- Projection: estimated all-target delta `-0.000209764`; projected Public proxy
  `0.595873488`, still `+0.045873488` from Public `0.55`.
- Decision: no MIS-LSTM Kaggle runner, no candidate artifact, no submission-log
  row, and no DACON submission.

2026-06-25 validation-weighted pseudo-public split audit and closeout:

- Report:
  `reports/post_id342_validation_weighted_pseudo_public_split_audit_20260625.md/.json`.
- The audit evaluates `15` target-weighted full/subject-hole/LOSO/test-neighbor
  lens configs over `247` OOF files and `8` public-scored rows.
- Best audit lens: `full_oof__id342_target_bottleneck`, stable public-order
  pairwise `1.0`, stable LOO MAE `0.000759294`, recent slate order
  `[342, 339, 338]`.
- Blocking issue: failed-frontier separation remains negative
  (`-0.014237111`), review-lens pass count is `0`.
- Public-0.55 weighted pass count: `0`; credible Public-0.55 pass count: `0`.
- Public-0.58 review pass count: `22`; credible Public-0.58 review pass count:
  `0`.

2026-06-25 solar/seasonal/holiday external prior gate:

- Report:
  `reports/post_id342_solar_seasonal_holiday_prior_gate_20260625.md/.json`.
- Feature source: Seoul/Korea photoperiod approximation, seasonal phase,
  weekday/weekend, Korean public-holiday proximity, and small interaction terms.
  It uses no raw sensor payloads, processed sleep columns, or prediction-derived
  features.
- Feature families: `solar_photoperiod` (`13` features),
  `korean_holiday_calendar` (`11`), `solar_holiday_interaction` (`7`), and
  `solar_holiday_all` (`28`), all with mean missing rate `0.0`.
- Best row: `solar_holiday_interaction_S4_lr_c0_08_w0.05`, wins `3/5`, mean
  delta `-0.002238508`, max delta `+0.001238751`.
- Eligible rows: `0`; selected targets: `[]`; projected Public proxy remains
  ID342 at `0.596083252`.
- Decision: no artifact, Kaggle run, submission-log row, or DACON submission.
  The external-calendar prior is useful negative evidence only.

2026-06-25 KMA ASOS weather-context gate:

- Secret handling and source contract:
  `docs/EXTERNAL_WEATHER_API_PLAN.md` records the external API boundary. Public
  data keys must stay in environment variables such as
  `KMA_ASOS_SERVICE_KEY`; fetched API caches stay under ignored
  `data/external/weather/`.
- Fetch script: `scripts/fetch_kma_asos_hourly_context.py`.
- Gate report:
  `reports/post_id342_kma_weather_context_gate_20260625.md/.json`.
- Source: KMA ASOS hourly observations from Daejeon station `133`, fixed before
  modeling to avoid GPS-derived nearest-station leakage.
- Cache: `4200` hourly rows covering `2024-06-01 00:00` through
  `2024-11-22 23:00`; metadata records `secret_values_stored=false`.
- Feature families: thermal load (`156` features), precipitation/pressure
  (`108`), solar/ground (`36`), and all weather (`300`).
- Best row: `kma_weather_precip_pressure_s4_lr_c0_10_w0.05`, wins `2/5`,
  mean delta `-0.000539108`, max delta `+0.000515499`, max failed-direction
  corr `+0.269367`.
- Projection: Public proxy `0.596006236`, still `+0.046006236` from Public
  `0.55`.
- Decision: local gate passes `0`, review gate passes `0`, DACON gate passes
  `0`; no artifact, Kaggle run, submission-log row, or DACON submission.

2026-06-25 public-data weather/air API smoke:

- Report:
  `reports/public_data_weather_air_api_smoke_20260625.md/.json`.
- Script: `scripts/smoke_public_data_weather_air_api.py`; Make target:
  `make smoke-public-data-weather-air-api`.
- Secret handling: all service keys are read from environment variables only
  (`KMA_ASOS_SERVICE_KEY`, `KMA_LIFE_INDEX_SERVICE_KEY`,
  `AIRKOREA_SERVICE_KEY`, or `PUBLIC_DATA_SERVICE_KEY`). The smoke report keeps
  summary metadata only; raw API payload caches are not retained by default,
  and optional troubleshooting caches must record `service_key_value_stored=false`.
- KMA ASOS `/getWthrDataList`: real historical call for Daejeon station `133`
  on `2024-06-01` returned `24/24` hourly rows. This confirms 2024 historical
  availability; the downstream KMA gate above still failed all promotion gates.
- KMA Living Weather Index V5: current `/getUVIdxV5` and
  `/getAirDiffusionIdxV5` calls succeeded, but the `2024060106` historical UV
  probe failed with result code `99` and a one-day data-window message. Treat
  this service as forecast/report-only, not a 2024 train/test feature source.
- AirKorea `ArpltnInforInqireSvc`: all five attempted information endpoints
  returned normal responses. Province realtime had `15` Daejeon rows; station
  `3MONTH` measurement returned `1000` page-1 rows and `totalCount=2190`, with
  page-1 times from `2026-05-14 22:00` to `2026-06-25 16:00`; fine-dust
  forecast returned `4` rows; weekly forecast and bad-or-worse CAI station list
  returned `0` rows for the smoke parameters.
- Decision: AirKorea API access works, but the approved information service
  exposes realtime/forecast/recent station data rather than a date-bounded
  2024 historical train/test feature source. No feature matrix, artifact,
  Kaggle run, submission-log row, or DACON submission is opened.

2026-06-25 downloaded AirKorea 2024 hourly archive audit:

- User download:
  `/Volumes/samsd/download_samsd/2024.zip`, moved to ignored
  `data/external/weather/airkorea_2024/airkorea_2024_hourly_all_stations_xlsx.zip`.
- Audit script: `scripts/audit_airkorea_2024_hourly_zip.py`; Make target:
  `make audit-airkorea-2024-hourly-zip`.
- Audit report:
  `reports/airkorea_2024_hourly_zip_audit_20260625.md/.json`.
- Normalized ignored cache:
  `data/external/weather/airkorea_2024/airkorea_2024_daejeon_hourly.csv`.
- Schema: monthly XLSX files with columns `지역`, `망`, `측정소코드`,
  `측정소명`, `측정일시`, `SO2`, `CO`, `O3`, `NO2`, `PM10`, `PM25`, and `주소`.
- Coverage: `5,819,357` national hourly rows, `114,192` Daejeon rows,
  `13` Daejeon stations, and Daejeon time range `2024-01-01 01:00:00` through
  `2025-01-01 00:00:00`.
- Competition-window coverage: `2024-06-01 00:00:00` through
  `2024-11-22 23:00:00`, `4,200/4,200` timestamps, `54,600` Daejeon
  station-hour rows, and all 13 stations at full timestamp coverage.
- Decision: this is a valid historical external source. It creates no model
  feature matrix, prediction file, submission artifact, Kaggle run,
  submission-log row, or DACON submission. It required a report-only ID342
  AirKorea gate before any promotion; that gate is recorded below and did not
  pass.

2026-06-25 downloaded AirKorea 2024 ID342 context gate:

- Gate script: `scripts/diagnose_post_id342_airkorea_2024_context_gate.py`;
  Make target: `make post-id342-airkorea-2024-context-gate`.
- Gate report:
  `reports/post_id342_airkorea_2024_context_gate_20260625.md/.json`.
- Scope: report-only, in-memory hourly/daily Daejeon pollution features from
  the ignored normalized cache. No persisted feature matrix, prediction file,
  submission artifact, Kaggle action, submission-log row, or DACON submission.
- Reviewed candidates: `216` over Q3/S3/S4, compact AirKorea families,
  low-capacity LR/HGB models, and tiny blend weights.
- Result: `0` local passes, `0` Public-0.58 review passes, and `0` DACON
  passes.
- Best row: `airkorea_selected_station_s4_hgb_l7_w0.025`, wins `3/5`, mean
  delta `-0.000190312`, max delta `+0.000819150`, failed-frontier correlation
  `+0.262872`, and projected Public proxy `0.596056065`.
- Decision: AirKorea 2024 is a valid historical source but not a score-relevant
  promotion route. Keep as negative external-source evidence; do not create an
  artifact, Kaggle run, submission-log row, or DACON submission.

2026-06-25 external ETRI sleep/survey prior audit:

- Prior script: `scripts/diagnose_post_id342_external_etri_sleep_survey_prior.py`;
  Make target: `make post-id342-external-etri-sleep-survey-prior`.
- Prior report:
  `reports/post_id342_external_etri_sleep_survey_prior_20260625.md/.json`.
- Scope: report-only analysis of local `data/external/etri_lifelog` historical
  survey/sleep CSVs plus 2026 processed-schema metadata. No feature matrix,
  model fit, prediction, artifact, Kaggle action, submission-log row, or DACON
  submission.
- Pairing result: `523` 2020 PM survey rows from `22` users matched to
  next-day sleep records after collapsing duplicate user/date sleep rows.
- Domain result: Q2 fatigue has coherent next-sleep coupling
  (`durationtosleep` delta `-407.064`, `wakeupduration` delta `-745.873`,
  S3 0-30m rate delta `+0.061719`, S4 WASO<20m delta `+0.038814`), but Q3
  stress does not support a simple sleep-quality proxy
  (`durationtosleep` delta `+58.624`, `wakeupduration` delta `+58.421`, S3
  0-30m delta `-0.059160`, S4 delta `-0.029484`).
- Decision: useful domain prior, not a direct 2026 feature route. Do not use it
  to create artifacts, Kaggle runs, submission-log rows, or DACON submissions.

2026-06-25 deadline-aware operating policy:

- Policy script: `scripts/diagnose_post_id342_deadline_operating_policy.py`;
  Make target: `make post-id342-deadline-operating-policy`.
- Policy report:
  `reports/post_id342_deadline_operating_policy_20260625.md/.json`.
- Official deadline sources checked on 2026-06-25: DACON overview search result
  lists algorithm close as `2026-07-23`; the ICTC official paper-submission
  page lists paper submission deadline `2026-06-26`, corroborated by the DACON
  talkboard notice for the ETRI workshop track. DACON rules require the
  IWETRIAI paper to use the Standard IEEE conference template as a `4`-page
  full paper.
- Current time is recorded in the generated policy report.
- Decision: deadline mode active, current public anchor remains ID342
  `0.5960832519`, `32/32` registered routes are closed, no current candidate
  exists, `run_kaggle_now=false`, and `dacon_submit_now=false`.
- Operating action: freeze long Kaggle/heavy runs and low-evidence DACON
  submissions; prioritize paper/reproducibility packaging and only run
  report-only or local gates expected to finish within 30-60 minutes.
- Paper draft created at `docs/PAPER_SUBMISSION_DRAFT_20260626.md`. It is a
  technical draft from verified records, not a final formatted EDAS paper.
- Paper checklist created at `docs/PAPER_SUBMISSION_CHECKLIST_20260626.md` to
  separate human-only EDAS details from model/reproducibility claims.
- IEEE source draft created at `docs/paper_submission_20260626/paper.tex` with
  bibliography and package README. It is a source draft only; it creates no
  predictions, submission CSV, Kaggle run, or DACON action.
- Paper package audit script: `scripts/audit_paper_submission_package.py`;
  Make target: `make paper-submission-package-audit`; report:
  `reports/paper_submission_package_audit_20260625.md/.json`.
- EDAS handoff checklist:
  `docs/paper_submission_20260626/edas_submission_handoff.md`, including the
  ICTC official EDAS submission link and human-only upload steps.
- Source archive builder:
  `scripts/package_paper_submission_sources.py`; Make target:
  `make paper-submission-source-archive`; report:
  `reports/paper_submission_source_archive_20260625.md/.json`; ignored archive:
  `build/paper_submission_20260626/paper_submission_20260626_source.zip`.
- EDAS metadata extractor:
  `scripts/extract_paper_submission_metadata.py`; Make target:
  `make paper-submission-metadata`; outputs:
  `docs/paper_submission_20260626/edas_metadata.json/.md`.
- Paper length budget audit:
  `scripts/audit_paper_length_budget.py`; Make target:
  `make paper-submission-length-audit`; outputs:
  `docs/paper_submission_20260626/paper_length_budget.json/.md`.

2026-06-25 cross-source split stability audit:

- Report:
  `reports/post_id342_cross_source_split_stability_audit_20260625.md/.json`.
- Scope: existing weak sequence-event, MIS-LSTM proxy, processed sleep,
  solar/holiday, and registered material-source reports only. No raw payloads,
  models, predictions, feature matrices, Kaggle actions, or DACON submissions.
- The audit extracts `128` candidate rows and reduces weight-only variants to
  `13` independent family-target representatives.
- S3 has a tiny stable alignment from phone/wearable discordance and GPS path
  topology, but its best projected Public proxy is only `0.596049063`.
- S4 aligns mainly on test-neighbor holdouts and conflicts on LOSO; Q3 has no
  cross-source alignment.
- Review signal targets: `[]`; gate pass: `False`.

2026-06-25 S3 phone/GPS interaction gate:

- Report:
  `reports/post_id342_s3_phone_gps_interaction_gate_20260625.md/.json`.
- Scope: registered processed phone/wearable discordance and GPS path-topology
  matrices only. No raw payloads, feature matrix artifacts, predictions,
  Kaggle actions, submission-log edits, or DACON submissions.
- Feature frame: `37` selected phone columns, `15` selected GPS columns, and
  `230` bounded interaction columns.
- Best row: `s3_phone_gps_selected_concat_s3_lr_c0_20_w0.05`, wins `5/5`
  primary splits, target mean delta `-0.001635922`, target max delta
  `-0.000068993`.
- Projection: Public proxy `0.595849549`, still `+0.045849549` from Public
  `0.55`.
- Decision: local gate pass count `0`; no artifact, Kaggle run,
  submission-log row, or DACON submission.

2026-06-25 final-day independent signal sweep:

- Report:
  `reports/post_id342_final_day_independent_signal_sweep_20260625.md/.json`.
- Scope: S3 phone/GPS temporal deviations, S4 processed sleep-fragmentation x
  solar/calendar joint state, and compact Q3/S3/S4 joint-state features. No
  feature matrix artifacts, predictions, Kaggle actions, submission-log edits,
  or DACON submissions.
- Best row: `s4_sleep_solar_joint_s4_lr_c0_10_w0.05`, wins `4/5`, mean delta
  `-0.002013473`, max delta `+0.000266583`, max failed-direction corr
  `+0.189500`.
- Projection: Public proxy `0.595795613`, still `+0.045795613` from Public
  `0.55`.
- Decision: local gate passes `4`, review gate passes `0`, DACON gate passes
  `0`; no artifact, Kaggle run, submission-log row, or DACON submission.

2026-06-25 registered-route closeout:

- Route synthesis:
  `reports/post_id342_remaining_route_synthesis_20260625.md/.json` now closes
  `30/30` registered routes.
- External research priority:
  `reports/post_id342_external_research_priority_20260625.md/.json` now sets
  top priority to `materially_new_external_source_hypothesis`.
- Decision: no candidate artifact, no Kaggle run, no submission-log row, and no
  DACON submission. Future work must start from a materially new source
  hypothesis and pass local/stable-shape/failed-frontier gates.
