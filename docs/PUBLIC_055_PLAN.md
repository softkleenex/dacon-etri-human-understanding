# Public 0.55 Plan

## Current checkpoint and stop rule

Public `0.55` remains unproven. The next step is not another same-family
submission; it is validation repair and artifact triage.

- Confirmed anchor: ID342 Public `0.5960832519`, local `0.534000`, DACON
  submission `1467937`, rank `220`, timestamp `2026-06-23 23:30:37`.
- ID342 improves ID337 by `-0.004537376722` and ID178 by `-0.0053280490`.
  The transferable public signal was the ID178/ID221 feature-similarity
  shrink, not the ID338 S1-only random-conv overlay.
- Previous anchor: ID178 Public `0.6014113009`.
- Severe failed local frontier: ID287 local `0.378934`, Public `0.6698787318`.
- Latest practical guarded probe: ID329 local `0.516718`, Public
  `0.6017908577`; this avoided collapse but did not improve ID178.
- 2026-06-21 micro-probe slate: ID337 scored `0.6006206286219425` and ID340
  scored `0.601379175253277`. These stayed near the old `0.60` band; ID342 is
  the first confirmed break below it, but still not Public `0.58` evidence.
- 2026-06-23 23:30 KST submission-spend slate: ID338, ID342, and ID339 were
  submitted after Kaggle v3 closed with no promotable artifact. Authenticated
  submission history later confirmed ID342 `0.5960832519`, ID339
  `0.5985187787`, and ID338 `0.5992145177`.
- Public `0.60` wall root cause:
  `reports/post_id341_public060_wall_root_cause_20260621.md` records the current
  explanation: recent micro-probes compress into the same public band, ID287/329
  prove local CV can be dangerously optimistic, Q2-only local signal is
  insufficient, and Q3/S3 remain the missing independent targets. The 2026-06-24
  authenticated correction refines this: feature-similarity shrink transferred
  best, while S1 random-conv and Q123 sequence-summary were useful but weaker
  probes. The next credible route must be ID342-aware Q3/S3 or materially
  independent raw-source work.
- Stop rule: do not automatically submit ID330/ID331 or other bolder
  constrained-combo variants until a fresh validation note explains why the
  candidate should behave differently from ID329.
- Post-ID341 manual probe decision:
  `reports/post_id341_manual_submission_probe_decision_20260618.md` keeps the
  default action at `do_not_submit`. ID341/340/337 have now been consumed as
  micro-probes and stayed near the `0.60` public band, so there is no fresh
  manual DACON sanity probe to recommend from this family.
- Post-ID341 no-approval frontier / Kaggle handoff refresh:
  `reports/post_id341_no_approval_frontier_20260619.md` confirmed, before the
  later reviewed/submitted ID342 artifact, that all local no-approval score
  routes were closed with no DACON submission authorized. The refreshed local
  runner smoke, freshness gate, prelaunch gate, guarded remote contract, and
  `reports/post_id341_operator_handoff_contract_20260619.md` are ready for
  explicit approval only; the exact approval phrase remains
  `Kaggle upstream rebuild no-submit 실행 승인`.
- Post-ID341 local/Kaggle compute boundary:
  `reports/post_id341_local_remote_compute_boundary_contract_20260619.md`
  keeps local work to fast source.zip / remote-output contract staging. The full
  upstream notebook is not a Mac-local default action; it is reserved for the
  exact-approved no-submit Kaggle run.
- 2026-06-23 Kaggle upstream patch run:
  the first upstream rebuild failure was traced to stale Kaggle input packaging,
  not to the Kaggle CLI itself. The old
  `softkleenex/dacon-etri-human-base` input exposed a 2026-06-17 project tree
  without `reports/`, so the runner died before any score-relevant report could
  run. A new private patch dataset,
  `softkleenex/dacon-etri-upstream-patch-post-id341`, was created to overlay
  the latest `scripts/`, `reports/`, `submissions/`,
  `artifacts/base_predictions/`, external ETRI files, and registered post-ID341
  feature parquet on top of the base raw/processed dataset. Upstream runner
  version 2 was pushed with both dataset sources; it passed the initial
  packaged-project path check and generated remote no-submit reports, then
  ended as `KernelWorkerStatus.ERROR` at the final upstream blueprint assertion
  because `remote_post_id341_upstream_rebuild_blueprint_20260618.json` reported
  `decision.blueprint_ready=False`; the pulled remote blueprint shows
  `missing_required_files=["dataset-metadata.json"]`. This is no longer a wait
  state. The runner now synthesizes runtime metadata after overlay when Kaggle
  omits `dataset-metadata.json`; the next action is an explicitly approved
  no-submit patch/runner refresh, not ID342 or DACON submission. See
  `reports/kaggle_upstream_rebuild_v2_failure_20260623.md`.
- Post-ID338 public-feedback next action:
  `reports/post_id338_public_feedback_next_action_20260623.md` was generated
  before authenticated submission-history correction and should now be read as
  an ID338-neighborhood diagnostic, not as the current anchor record. The
  current anchor record is
  `reports/dacon_three_slot_submission_result_20260623.md/.json`, corrected to
  ID342.
- Post-ID342 raw/sequence Q3/S3 follow-up:
  `reports/post_id342_research_gap_and_source_priority_20260624.md` identified
  `raw_temporal_q3s3_sequence_image` as the next report-only source family
  after the ID342 public correction. The first raw temporal smoke,
  `reports/post_id342_raw_temporal_q3s3_sequence_image_smoke_20260624.md`,
  did not pass: Q3 had only an unstable partial same-subject kNN lead
  (3/5 primary wins, mean delta `-0.001735`, max delta `+0.002487`) and S3
  remained closed. The stronger sequence-summary distance/kernel smoke,
  `reports/post_id342_sequence_summary_distance_kernel_smoke_20260624.md`,
  also selected no target: Q3 best was 2/5 wins with mean delta `+0.000055`,
  and S3 best was 1/5 wins with mean delta `+0.000092`. Do not create a DACON
  artifact or Kaggle no-submit run from these local smokes; they are negative
  evidence that simple ID342-aware raw sequence summaries are not enough for
  Public `0.55`.
- Post-ID342 sleep-boundary and tiny-CNN checks:
  `reports/post_id342_sleep_boundary_q3s3_rescue_smoke_20260624.md` retested
  direct sleep-onset-latency boundary features against ID342 for Q3/S3 only.
  It selected no target; Q3 best was 0/5 wins with mean delta `+0.000175`, and
  S3 best was 1/5 wins with mean delta `+0.000101`. The local
  PixleepFlow/MIS-LSTM proxy
  `reports/post_id342_tiny_temporal_cnn_q3s3_smoke_20260624.md` also selected
  no target; Q3 best was 0/5 wins with mean delta `+0.000275`, and S3 best was
  2/5 wins with mean delta `+0.000012`. This closes the current ID342-aware
  local raw/sequence/SOL/tiny-neural branch for Kaggle no-submit escalation.
  The next work must either explain ID342's public transfer through submitted
  candidate geometry or design a materially different source, not replay these
  architectures with tiny weight changes.
- Post-ID342 public-response and feature-similarity decomposition:
  `reports/post_id342_public_response_geometry_20260624.md` confirms that the
  2026-06-23 slate is one of the few recent batches where local order matched
  public order: `[342, 339, 338]` under both lenses. ID342's useful movement is
  feature-similarity movement on `S4`, `Q1`, and `Q2`, with local target-loss
  advantage over ID339 led by `S4`, `Q1`, and `Q3`. The follow-up
  `reports/post_id342_feature_similarity_source_decomposition_20260624.md`
  finds small local-eligible ID342-neighborhood variants, especially ID221
  alpha `0.65` on `Q1/Q2/Q3` or `Q1/S4`. The best proxy remains only about
  Public `0.59575`, still `+0.04575` from the target `0.55`, so this is not a
  DACON or Kaggle escalation route. The strict artifact gate,
  `reports/post_id342_feature_similarity_artifact_gate_20260624.md`, reviewed
  the locally eligible rows and passed `0`: best projected public proxy
  `0.595751629`, gap to Public `0.55` `+0.045751629`, no artifact creation,
  no Kaggle push/run, and no DACON submission. Treat it as evidence that the
  next credible work is a clean feature-similarity source rebuild/decomposition
  or a materially new source, not a Q3/S3-only rescue or another ID342 alpha
  tweak.
- Post-ID342 paper-aligned feature-similarity rebuild gate:
  `reports/post_id342_paper_aligned_feature_similarity_rebuild_gate_20260624.md`
  tests the strongest current research prior directly: Q1/S3 use
  movement/time families, Q2/Q3 use app/time families, and S4 uses
  HR/light/activity/time families, with fold-internal config selection on top
  of ID342. It does not beat ID342: candidate loss `0.534649`, delta vs ID342
  `+0.000649`, primary wins `2/5`, projected public proxy `0.596731761`, gap to
  Public `0.55` `+0.046731761`. Do not push this to Kaggle or create a
  submission artifact; it is negative evidence for same-subject
  paper-aligned feature-similarity as the immediate source rebuild.
- Post-ID342 past-only temporal recency gate:
  `reports/post_id342_past_only_temporal_recency_gate_20260624.md` tests the
  safer version of same-subject temporal label anchors: validation and test
  queries can use only labels from earlier dates. It still does not beat ID342:
  candidate loss `0.534430`, delta vs ID342 `+0.000430`, primary wins `0/5`,
  projected public proxy `0.596513661`, gap to Public `0.55` `+0.046513661`.
  This closes simple temporal persistence as an immediate Public-0.55 route.
- Post-ID342 cross-target consistency gate:
  `reports/post_id342_cross_target_consistency_gate_20260624.md` tests
  second-stage Q2/Q3/S3/S4 consistency using only other ID342 target
  predictions, not true same-row sibling labels. It also does not beat ID342:
  candidate loss `0.534113`, delta vs ID342 `+0.000113`, primary wins `0/5`,
  projected public proxy `0.596195808`, gap to Public `0.55` `+0.046195808`.
  Keep it as negative evidence; no Kaggle, artifact, or DACON action.
- Post-ID342 registered material-source rescan:
  `reports/post_id342_registered_material_source_rescan_20260624.md` rescans
  the registered circadian, GPS topology, phone/wearable discordance,
  proximity/ambience, and usage semantic matrices against ID342. It reviews
  `450` fold-internal candidates and passes `0`. The best row is
  `proximity_ambience_cooccurrence_s2_lr_c0_20_w0.05`, mean all-target delta
  `-0.000171524` vs ID342, projected public proxy `0.595911728`, gap to Public
  `0.55` `+0.045911728`. This is not a Kaggle or DACON escalation route.
- Post-ID342 registered material fusion gate:
  `reports/post_id342_registered_material_fusion_gate_20260624.md` concatenates
  all five registered material matrices into `322` features. It reviews `90`
  fusion candidates and passes `0`; the best S3 row has mean all-target delta
  only `-0.000001844` vs ID342, projected public proxy `0.596081408`, gap to
  Public `0.55` `+0.046081408`. Lightweight registered-material fusion is not
  the missing Public-0.55 route.
- Post-ID342 public-alignment validation lens:
  `reports/post_id342_public_alignment_validation_lens_20260624.md` uses
  confirmed public-scored OOF IDs `178/287/329/337/338/339/340/342` to measure
  which local lenses explain public outcomes. The result is cautionary rather
  than promotable: across all scored IDs, full OOF has Spearman `-0.167`
  because ID287 is the local-best/public-worst failure. Excluding only ID287 is
  still weak (Spearman `0.250`) because ID329 remains a local-frontier trap.
  Excluding both failed-frontier IDs, full OOF, subject-hole, and LOSO exactly
  match the public order `[342, 339, 338, 337, 340, 178]`, and the recent slate
  `[342, 339, 338]` matches across all frozen split lenses. This updates the
  gate policy only: future candidates must beat ID342 on stable near-anchor
  lenses and explicitly show ID287/ID329 inclusion/exclusion behavior. It does
  not authorize Kaggle, artifact creation, or DACON submission.
- Post-ID342 stable-lens existing-candidate triage:
  `reports/post_id342_stable_lens_existing_candidate_triage_20260624.md` applies
  the new lens policy to `247` existing OOF candidates. It passes `0`. The raw
  local-best rows are failed-frontier-shaped, with large movement and high
  ID287/ID329 direction alignment. The only lineage near-misses are ID222 and
  ID221, but ID222 is a public-structure tempered calibration probe that is not
  strict Subject-Hole/fold-nested, and ID221 is the public-structure
  feature-similarity root already used by ID342. This closes the existing
  candidate shelf for Kaggle/DACON escalation; Public `0.55` needs a clean
  source rebuild or materially new representation, not a recycled local-frontier
  artifact.
- Post-ID342 env/social/routine v2 ID342 gate:
  `reports/post_id342_env_social_routine_v2_id342_gate_20260624.md` retests the
  raw-derived routine/place/social/ambient-load feature frame against ID342
  after the stable-lens triage. It finds a narrow Q2-only signal:
  `blend_id342_envsocial_lr_c0_20_w0.025` wins `5/5` primary lenses with mean
  delta `-0.001177` vs ID342. Q3 and S3 both have `0/5` wins, so the missing
  target problem remains. Estimated all-target delta is only `-0.000168`,
  projected Public proxy `0.595915153`, gap to Public `0.55` `+0.045915153`.
  This is not a Kaggle or DACON route.
- Post-ID342 evening state-transition gate:
  `reports/post_id342_evening_state_transition_gate_20260624.md` tests
  pre-sleep screen/app/HR arousal, settling into darkness/charging/stillness,
  previous-day label-free deviations, and fold-internal subject baselines
  against ID342. It uses `520` raw-derived transition columns and reports
  ID287/ID329 failed-frontier direction alignment. It selects no target: Q3
  best is `2/5` primary wins with mean delta `+0.000015535`, and S3 best is
  `2/5` with mean delta `+0.000063167`. This closes lightweight
  arousal/settling transitions for Kaggle, artifact creation, and DACON
  submission.
- Post-ID342 public-scored Q3/S3 target-swap gate:
  `reports/post_id342_public_scored_target_swap_gate_20260624.md` confirms
  that apparent Q3/S3 local improvement is dominated by failed-frontier donors.
  ID287/ID329 create `44` local trap rows, with ID287 Q3 weight `1.0`
  projecting locally near Public `0.564688`, but those donors are confirmed
  public failures. Stable public-scored donors `178/337/338/339/340` pass `0`
  rows; the best stable Q3 rows have only `3/5` primary wins and no S3 row
  opens. This is not a Kaggle, artifact, or DACON route.
- Post-ID342 Q3/S3 slice-selector anatomy:
  `reports/post_id342_q3s3_slice_selector_anatomy_20260624.md` checks whether
  the isolated stable donor wins can be selected fold-internally by subject or
  split. They cannot. Stable donors beat ID342 on only `7` subject/target
  slices, while failed-frontier donors beat it on `20`. The fold-internal
  selector picks no target: Q3 has `0/5` primary wins with mean delta
  `+0.000063533`, and S3 has `0/5` with mean delta `+0.000103967`. This closes
  stable-donor selection and points back to materially new subject-state
  representation work.
- Post-ID342 lightweight subject-state PCA gate:
  `reports/post_id342_lightweight_subject_state_pca_gate_20260624.md` tests a
  non-heavy proxy for subject-state modeling by applying fold-internal PCA and
  low-capacity LR to existing label-free env/social/routine and registered
  material matrices. It passes `0` rows. The best row is registered-material S3
  PCA16/LR at weight `0.005`, only `2/5` primary wins with mean delta
  `+0.000053036`. This closes lightweight PCA over existing matrices; the
  remaining subject-state path must be materially different and, if heavy, must
  use the separate heavy-compute approval phrase.
- Post-ID342 processed sleep-proxy gate:
  `reports/post_id342_processed_sleep_proxy_gate_20260624.md` checks the
  already-materialized processed sleep-stage, sleep-HR, activity, screen, light,
  and pre-sleep proxy columns against ID342 on Q3/S3 only. It scans
  `sleep_proxy` (48 columns), `sleep_proxy_no_ids` (47), and `core_sleep` (42)
  with fold-internal LR/HGB overlays and tiny weights. It passes `0` rows; the
  best row is S3 HGB weight `0.005` with only `2/5` primary wins, mean delta
  `+0.000019110`, projected Public proxy `0.596083252`, and gap to Public
  `0.55` `+0.046083252`. Keep these features as negative/domain evidence, not
  as a Kaggle, artifact, or DACON escalation route.
- Post-ID342 processed sleep temporal-deviation gate:
  `reports/post_id342_processed_sleep_temporal_deviation_gate_20260624.md`
  tests the more domain-specific personal-baseline version of the same source:
  previous-row sleep proxies, past 3-row rolling means/stds, and current
  deviations from those baselines. It passes `0` rows. The best row is Q3
  temporal-deviation LR weight `0.005`, only `2/5` primary wins with mean delta
  `+0.000111587`, projected Public proxy `0.596083252`, and gap to Public
  `0.55` `+0.046083252`. This closes lightweight processed sleep
  lag/deviation as an immediate route.
- Post-ID342 subject-prior calibration gate:
  `reports/post_id342_subject_prior_calibration_gate_20260624.md` checks the
  target-definition idea directly. Q3 is participant-relative stress and S3 is
  guideline adherence, but fold-internal subject/global prior forcing hurts
  ID342: the best Q3 row has `0/5` primary wins and mean delta `+0.000918056`,
  while the best S3 row has `0/5` wins and mean delta `+0.001384175`. Projected
  Public proxy remains `0.596083252`, gap to Public `0.55` `+0.046083252`.
  Therefore do not calibrate the test file toward train subject priors.
- Post-ID342 probability/temperature gate:
  `reports/post_id342_probability_temperature_gate_20260624.md` confirms that
  the ID342 submission CSV is a probability file matching its base-test artifact
  exactly, not a binary-thresholded file. Max absolute delta vs base-test is
  `0.000000000000`, and `all_targets_binary=False`. A fold-internal Q3/S3
  temperature/shift scan also fails: S3 has `0/5` primary wins with mean delta
  `+0.001856684`, and Q3 has only `2/5` wins with mean delta `+0.003552168`.
  This closes simple threshold/temperature handling as an immediate Public
  `0.55` route.
- Post-ID342 remaining-route synthesis:
  `reports/post_id342_remaining_route_synthesis_20260624.md` combines the
  post-ID342 no-submit diagnostics into the current operating decision. It
  records ID342 as the anchor at Public `0.5960832519`, gap `+0.0460832519` to
  Public `0.55`, closes `11/12` tracked routes, has `0` missing/review routes,
  and confirms no current Kaggle wait state. The decision fields keep
  `push_kaggle_now=False`, `run_kaggle_now=False`, `dacon_submit_now=False`,
  and `dacon_submit_meaningful_now=False`. The only remaining materially
  different route is heavy self-supervised/subject-state representation work,
  still requiring the exact heavy-compute no-submit approval phrase before any
  execution.
- Post-ID342 self-supervised route research packet:
  `reports/post_id342_self_supervised_route_research_packet_20260624.md`
  converts that remaining route into a blocked execution packet. It uses the
  ETRI/DACON target definitions and PixleepFlow-style multimodal synchronization
  prior to define Q3 as a stress-state route, S3 as a sleep-transition route,
  and Q2 as a secondary sanity target. It confirms the post-ID341 design and
  heavy-compute contract are ready, but keeps `heavy_compute_allowed_now=False`,
  `push_kaggle_now=False`, `run_kaggle_now=False`, and `dacon_submit_now=False`.
  The exact required phrase remains
  `Self-supervised subject-state heavy compute no-submit 실행 승인`; the Kaggle
  upstream rebuild approval phrase does not authorize this route.
- Post-ID342 self-supervised heavy-compute contract:
  `reports/post_id342_self_supervised_heavy_compute_contract_20260624.md`
  refreshes the approval boundary after ID342 became a scored anchor. The
  post-ID341 contract should not be reused for current execution decisions
  because it assumed ID342 was still generated-unsubmitted. The refreshed
  contract marks the current prerequisites ready, accepts ID342 Public
  `0.5960832519` as the anchor, records next numeric ID `345` only if later
  authorized, and keeps `heavy_compute_allowed_now=False`,
  `push_kaggle_now=False`, and `run_kaggle_now=False`. It also records that a
  dedicated heavy runner is required; the upstream rebuild runner is not an
  embedding-training runner.
- Post-ID342 goal-state audit:
  `reports/post_id342_goal_state_audit_20260624.md` maps the active objective
  to evidence. It marks `goal_achieved=False` and `public_055_reached=False`
  because the best confirmed score is still ID342 Public `0.5960832519`, gap
  `+0.0460832519`. It confirms `dacon_submit_now=False`, `push_kaggle_now=False`,
  `run_kaggle_now=False`, and that current candidate IDs `338/339/342/343/344`
  have matching submission CSV and notebook-history files. Re-run with
  `make post-id342-goal-state-audit` after any major state change.
- Post-ID342 self-supervised heavy runner spec:
  `reports/post_id342_self_supervised_heavy_runner_spec_20260624.md` is the
  blocked no-submit Kaggle runner specification for the remaining heavy route.
  It is ready only as a report: proposed notebook directory
  `notebooks/kaggle_self_supervised_subject_state_runner`, proposed slug
  `softkleenex/dacon-etri-self-supervised-subject-state-post-id342`, target
  focus `Q2/Q3/S3`, and validation gates are documented. It creates no runner,
  feature matrix, prediction, submission artifact, Kaggle run, or log row, and
  keeps `create_runner_files_now=False`, `push_kaggle_now=False`,
  `run_kaggle_now=False`, and `dacon_submit_now=False`. Any execution still
  requires `Self-supervised subject-state heavy compute no-submit 실행 승인`.
- Post-ID342 subject-state domain/token contract:
  `reports/post_id342_subject_state_domain_token_contract_20260624.md` converts
  the heavy route into domain-level implementation requirements. It sets Q3/S3
  as the primary score targets, Q2 as a sanity target, maps all 12 raw phone
  and wearable modalities to stress-state or sleep-transition token roles, and
  defines day/evening/sleep-transition/subject-baseline token windows. It is
  report-only: no raw parquet payload read, token cache, embedding, feature
  matrix, prediction, Kaggle run, log edit, or DACON submission is authorized.
- Post-ID342 subject-state raw schema contract:
  `reports/post_id342_subject_state_raw_schema_contract_20260624.md` verifies
  the heavy route against raw parquet metadata only. All 12 expected modalities
  exist and pass required-column checks; row counts are captured from metadata
  only (`6,122,179` total), with `raw_payload_rows_read=0`. This makes the
  future no-submit tokenization implementation less ambiguous but still
  authorizes no raw payload read, token cache, embedding, feature matrix,
  Kaggle run, log edit, or DACON submission.
- Post-ID342 subject-state heavy approval readiness:
  `reports/post_id342_subject_state_heavy_approval_readiness_20260624.md`
  consolidates the heavy route gates. All prerequisites are ready and the
  latest manual Kaggle status check found the old upstream runner at
  `KernelWorkerStatus.COMPLETE`, but `exact_heavy_approval_observed=False`.
  Therefore approval is the only missing runner-creation gate; it still keeps
  runner files, raw payload reads, token caches, embeddings, feature matrices,
  Kaggle run, log edit, artifact creation, and DACON submission blocked.
- Post-ID342 subject-state heavy success thresholds:
  `reports/post_id342_subject_state_heavy_success_thresholds_20260624.md`
  defines the future remote-output review ladder. A below-ID342 but above-0.58
  proxy is only a research lead. Artifact review requires public proxy `<=0.58`,
  stable failed-frontier-excluded Q3/S3 gain, worst primary split control, and
  embedding/privacy/artifact-separation reports. DACON candidacy requires proxy
  `<=0.55`, exact artifact matching, and fresh exact file/memo approval. This
  creates no runner, raw read, token cache, embedding, Kaggle run, artifact,
  log edit, or DACON submission now.
- Post-ID342 Kaggle input availability:
  `reports/post_id342_kaggle_input_availability_20260624.md` uses the
  project-local Kaggle CLI to verify future heavy-runner input datasets. Both
  `softkleenex/dacon-etri-human-base` and
  `softkleenex/dacon-etri-upstream-patch-post-id341` are visible (`2/2`
  datasets available, first page lists `20` files each). The CLI still emits
  the known `2.1.1` outdated-version warning, but input availability is not the
  current blocker. No push, kernel run, output pull, artifact, log edit, or
  DACON submission is authorized.
- Post-ID342 Kaggle CLI health:
  `reports/post_id342_kaggle_cli_health_20260624.md` verifies `.venv/bin/kaggle`
  is usable for required status/list checks. Installed version is `2.1.1`, the
  CLI emits the known `2.2.2` upgrade warning, dataset listing works, the old
  upstream kernel is `KernelWorkerStatus.COMPLETE`, and
  `auth_network_slug_blocker_observed=False`. This confirms the current blocker
  is not Kaggle CLI access; it is the missing exact heavy approval.
- Post-ID342 morning closeout:
  `reports/post_id342_morning_closeout_20260624.md` is the current operating
  state near 10:00 KST. It records `morning_closeout_ready=True`,
  `goal_achieved=False`, `public_055_reached=False`, best public anchor ID342
  at `0.5960832519`, gap `+0.0460832519`, `kaggle_waiting_now=False`,
  `kaggle_cli_blocker_now=False`, `dacon_submit_meaningful_now=False`, and no
  runner/artifact/log action now. The next score-relevant action is only the
  exact-approved self-supervised subject-state no-submit heavy route.
- Post-ID342 external research priority:
  `reports/post_id342_external_research_priority_20260624.md` updates the
  remaining heavy-route design from external ETRI/DACON and wearable SSL
  research. The priority is now missingness-aware cross-modal masked
  subject-state learning, combining AIM-style inherited/artificial masks,
  CroSSL-style modality encoders and latent masking, TimeMAE-style semantic
  windows, and strict fold-internal Q3/S3 downstream heads. This is research
  prioritization only: `heavy_compute_allowed_now=False`,
  `run_kaggle_now=False`, `dacon_submit_now=False`, and no artifact/log action
  is opened.
- Post-ID342 heavy implementation feasibility:
  `reports/post_id342_heavy_implementation_feasibility_20260624.md` confirms
  the current runtime and data inputs can support the approved-later heavy
  route: core Python dependencies are installed and all 12 raw parquet inputs
  are present. It also records the real after-approval implementation gap:
  dedicated no-submit runner files and Make targets do not exist yet and should
  be created only after exact heavy approval. It opens no raw payload read,
  embedding, runner, Kaggle action, artifact, log row, or DACON submission.
- Post-ID342 heavy runtime budget:
  `reports/post_id342_heavy_runtime_budget_20260624.md` estimates the future
  heavy runner's input and token budget from file sizes and parquet footer
  metadata only. It confirms `6,122,179` raw rows from metadata, about `122.8`
  MiB compressed raw files, and one row group per raw modality file. Therefore
  the approved-later runner must stream with scanner batches rather than
  row-group reads. It recommends `65,536` scanner rows and keeps local work to
  tiny no-submit smoke batches; no payload read, token cache, embedding,
  Kaggle action, artifact, log row, or DACON submission is opened now.
- Post-ID342 S3/Q3 raw sequence representation gate:
  `reports/post_id342_s3q3_raw_sequence_representation_gate_20260624.md` is
  the completed no-submit Kaggle version 7 follow-up to the raw subject-state
  runner. It tested five random-convolution sequence families over the 44
  half-hour raw tensor with `256` kernels and seeds `45502/45602`. The gate
  reviewed `240` ID342-aware fold-internal Q3/S3 candidates and passed `0`.
  The best row is a Q3 usage-routine micro-blend with projected Public
  `0.596083233`, still `+0.046083233` from Public `0.55`. The updated remote
  intake passed and found no submission-like output. This closes the raw
  random-convolution sequence branch as a 0.55 route; no artifact, log row,
  Kaggle promotion run, or DACON submission is authorized from it.
- Post-ID338 material Q3/S3 probe:
  `reports/post_id338_material_q3s3_probe_20260623.md` and
  `reports/post_id338_material_q3s3_geometry_gate_20260623.md` found one tiny
  S3-only independent signal from phone/wearable discordance features. Q3 did
  not open. The S3 geometry gate passed with mean all-target movement
  `0.00011334` vs ID338 and no risk flags, so ID343 was created as a
  generated-unsubmitted reserve artifact:
  `submission_id_343_id338_s3_phone_wearable_micro_w0005_cv0.537854_unsubmitted.csv`.
  This is not Public `0.58` evidence and not a current submit recommendation.
- Post-ID338 S3 weight geometry scan:
  `reports/post_id338_s3_phone_wearable_weight_geometry_scan_20260624.md`
  selects the bolder weight `0.02` S3 phone/wearable overlay. ID344 was created
  as a generated-unsubmitted reserve artifact:
  `submission_id_344_id338_s3_phone_wearable_w0020_cv0.537839_unsubmitted.csv`.
  It changes only S3, improves local CV vs ID338 by `-0.00002210`, and remains
  a controlled reserve candidate rather than a current submit recommendation.
- Post-ID338 Q3 deep material scan:
  `reports/post_id338_q3_deep_material_scan_20260623.md` tried low-capacity LR,
  HGB, ExtraTrees, and RandomForest variants over the same material matrices.
  Q3 still has no selected config; top rows are circadian micro-blends with
  only 2/5 primary wins and positive mean delta vs ID338. Do not promote Q3 from
  this material-source family.
- Post-ID341 phone/wearable discordance design:
  `reports/post_id341_phone_wearable_discordance_design_20260619.md` freezes a
  new report-only source family over fixed disagreement between phone rest
  state (`mScreenStatus`, `mACStatus`, `mActivity`) and wearable activation
  (`wPedo`, `wHr`, `wLight`, `mLight`) for Q2/Q3/S3. It observes feasible
  discordance bins and adequate raw coverage, but it is not Public `0.58`
  evidence and opens only a future feature-spec step; no matrix, model, ID342,
  Kaggle run, or DACON submission is authorized.
- Post-ID341 phone/wearable discordance validation stop:
  `reports/post_id341_phone_wearable_discordance_feature_matrix_validation_stop_20260619.md`
  confirms the registered 700-row, 85-feature, label-free matrix at
  `data/processed/post_id341_phone_wearable_discordance_features_v1.parquet`
  matches its deterministic rebuild and opens no score route. This closes the
  local no-approval phone/wearable lane at infrastructure-only status; no model,
  prediction file, ID342, submission-log row, Kaggle run, or DACON submission is
  authorized from it.
- Post-ID341 self-supervised subject-state embedding design:
  `reports/post_id341_self_supervised_subject_state_embedding_design_20260619.md`
  is a metadata-only design gate for a possible label-free all-sensor day-state
  embedding route over Q2/Q3/S3. It confirms 12/12 raw modalities, at least
  6M raw rows, and the 450/250 processed row contract, but opens no score route
  and authorizes no full raw-stream read, embedding fit, feature matrix, model,
  ID342, Kaggle run, submission-log row, or DACON submission. This route is
  heavy and requires a separate execution contract if pursued later.
  `reports/post_id341_material_new_source_queue_20260619.md` now records it as
  `design_gate_ready_heavy_deferred`, so it is visible in prelaunch without
  becoming a local no-approval score route.
- Post-ID341 GPS path-topology dry run:
  `reports/post_id341_gps_path_topology_feature_generation_dryrun_20260619.md`
  confirms a materially new Q3/S3 raw-source family can be aligned in memory:
  700/700 train+test keys match exactly, the first virtual frame has 32 columns
  under the 64-column bound, and GPS-ready rates are `0.9333` train / `0.9360`
  test. This is still not Public `0.58` evidence and does not authorize ID342;
  it only makes a later feature-matrix smoke contract reviewable.
- Post-ID341 GPS feature-matrix smoke contract:
  `reports/post_id341_gps_path_topology_feature_matrix_smoke_contract_20260619.md`
  now permits only a future no-model/no-submission writer for the single
  registered path
  `data/processed/post_id341_gps_path_topology_features_v1.parquet`. The
  contract explicitly keeps `feature_matrix_write_allowed_now=False`,
  Public `0.58` evidence false, and score-route creation false.
- Post-ID341 GPS feature-matrix materialization:
  `reports/post_id341_gps_path_topology_features_v1_schema.md` records the
  registered label-free parquet after validation. This is infrastructure for a
  later separately approved review only; it still does not authorize ID342,
  model training, Kaggle network use, or DACON submission.
- Post-ID341 GPS model smoke:
  `reports/post_id341_gps_path_topology_model_smoke_20260619.md` tests the
  registered matrix with tiny fold-internal Q3/S3 models against ID178/ID341.
  It selects no target under the frozen promotion gate: Q3 is directionally bad
  and S3 has only a weak 3/5 split micro-blend signal. Therefore the GPS
  topology matrix is not an ID342/local-submission route by itself.
- Post-ID341 circadian HR/light phase-shape spec:
  `reports/post_id341_circadian_hr_light_phase_feature_spec_20260619.md`
  advances the next materially new local source only to a bounded design: 48
  planned features from `wHr`, `wLight`, `mLight`, and `wPedo`, predeclared
  hour bins, fold-only baseline guards, and no labels/predictions/public scores
  in feature generation. It is not Public `0.58` evidence and authorizes only a
  report-only no-matrix dry run, not ID342, model training, Kaggle network use,
  or DACON submission.
- Post-ID341 circadian HR/light feature-generation dry run:
  `reports/post_id341_circadian_hr_light_phase_feature_generation_dryrun_20260619.md`
  confirms the 48-feature design can be assembled in memory with exact
  700-row train/test alignment, finite bounded columns, and strong/weak
  phase-shape gates still passing. This only makes a future report-only
  feature-matrix smoke contract reviewable; it still does not authorize a
  feature parquet, model, ID342, Kaggle network use, or DACON submission.
- Post-ID341 circadian HR/light feature-matrix smoke contract:
  `reports/post_id341_circadian_hr_light_phase_feature_matrix_smoke_contract_20260619.md`
  now permits only a future no-model/no-submission writer for the single
  registered path
  `data/processed/post_id341_circadian_hr_light_phase_features_v1.parquet` plus
  schema report. It keeps feature-matrix writes disallowed in the contract
  step itself, Public `0.58` evidence false, and score-route creation false.
- Post-ID341 circadian HR/light feature-matrix materialization:
  `reports/post_id341_circadian_hr_light_phase_features_v1_schema.md` records
  the registered 700-row, 48-feature, label-free parquet after validation. This
  is infrastructure for a later separately gated model-smoke review only; it
  does not authorize ID342, model training, Kaggle network use, or DACON
  submission.
- Post-ID341 circadian HR/light model-smoke closure:
  `reports/post_id341_circadian_hr_light_phase_model_smoke_contract_20260619.md`
  is now idempotent after the bounded smoke has completed, and
  `reports/post_id341_circadian_hr_light_phase_model_smoke_20260619.md` selects
  no Q3/S3/S2 target. This closes the circadian lane as local score evidence;
  do not create ID342 or rerun tiny circadian variants from it.
- Post-ID341 usage semantic burst/Markov design:
  `reports/post_id341_usage_semantic_burst_markov_design_20260619.md` is the
  next allowed no-approval local step after circadian closure. It is design-only:
  raw `mUsageStats` coverage is `0.9778` train / `1.0000` test with 586 unique
  app names, and `mScreenStatus` coverage is complete, but the report creates no
  feature matrix, trains no model, and gives no Public `0.58` evidence. A later
  step may only be metadata/cardinality smoke unless a new contract is written.
- Post-ID341 usage semantic burst/Markov metadata smoke:
  `reports/post_id341_usage_semantic_burst_markov_metadata_smoke_20260619.md`
  passes the design's raw feasibility gate: all 7 predeclared app-category
  states appear, 48/49 category transition pairs are observed, 12,945 evening
  transitions exist, 4,686 sessions are found, and usage rows match screen state
  within five minutes at `0.9985`. This is still not score evidence and opens
  only a report-only feature-spec step, not feature generation, ID342, model
  training, Kaggle, or DACON submission.
- Post-ID341 usage semantic burst/Markov feature spec:
  `reports/post_id341_usage_semantic_burst_markov_feature_spec_20260619.md`
  freezes a 700-row label-free row contract, 7 fixed app states, 49 fixed
  Markov pairs, 7 feature groups, and a bounded 77-column first-pass schema.
  This is still not Public `0.58` evidence. It opens only a future report-only
  in-memory dry run, not a feature parquet, model, ID342, Kaggle network run, or
  DACON submission.
- Post-ID341 usage semantic burst/Markov feature-generation dry run:
  `reports/post_id341_usage_semantic_burst_markov_feature_generation_dryrun_20260619.md`
  proves the 77-column schema can be assembled in memory over the exact
  450 train + 250 test keys with finite values and no NaNs. It is still not
  Public `0.58` evidence and opens only a future report-only feature-matrix
  smoke contract, not a feature parquet, model, ID342, Kaggle run, or DACON
  submission.
- Post-ID341 usage semantic burst/Markov feature-matrix smoke contract:
  `reports/post_id341_usage_semantic_burst_markov_feature_matrix_smoke_contract_20260619.md`
  now permits only a future single registered label-free matrix writer for
  `data/processed/post_id341_usage_semantic_burst_markov_features_v1.parquet`
  plus schema report. It is still not Public `0.58` evidence and does not
  authorize model training, ID342, Kaggle network use, or DACON submission.

See `docs/WORKFLOW_REVIEW.md` for the current workflow assessment and session
handoff.

## 2026-06-17 latest checkpoint

### 2026-06-17 21:50 KST update: validation stress diagnostics

- Kaggle full demographic audit completed with no submission. It found
  subject-hole gains but LOSO weakness, so demographic/static features are
  hypothesis-only rather than a global overlay.
- Test split alignment diagnostic confirmed that the hidden test rows are not a
  pure future holdout. Most subjects have middle and late test blocks; several
  subjects have fragmented later rows.
- Existing base OOF artifacts were scanned on test-geometry holdouts. This
  failed the critical sanity check: ID287 still ranked near the top by OOF
  subset losses despite its confirmed Public `0.6698787318` collapse.
- ID178 sequence settings were stress-tested under subject-hole, LOSO, and
  test-boundary holdouts. The signal is small but consistently positive. A
  hypothetical `rollback_s1_s4` variant is more stress-consistent than original
  ID178, but it remains a same-family hypothesis and was not promoted.
- Decision: no Public `0.58`-plausible candidate is promoted. If the user wants
  the next controlled public probe, the first reviewable path is to create a new
  fully matched artifact for the ID178 `rollback_s1_s4` hypothesis; otherwise,
  the next candidate must come from a fold-nested path, not from full-OOF
  recalibration, post-hoc OOF subset ranking, or another same-family constrained
  combo.
- Supporting reports:
  - `reports/strict_demographic_feature_audit_kaggle_review_20260617.json/.md`
  - `reports/test_split_fold_alignment_post_id329_20260617.json/.md`
  - `reports/oof_test_geometry_stress_post_id329_20260617.json/.md`
  - `reports/random_conv_sequence_robustness_post_id329_20260617.json/.md`
  - `reports/id178_sequence_target_rollback_post_id329_20260617.json/.md`

### 2026-06-17 22:03 KST update: ID334 rollback artifact prepared, not submitted

- A fully matched generated-unsubmitted artifact was created for the ID178
  `rollback_s1_s4` hypothesis:
  `submission_id_334_id178_rollback_s1_s4_sequence_stressguard_cv0.540347_unsubmitted.csv`.
- Matching files:
  - notebook:
    `notebooks/history/submission_334_id178_rollback_s1_s4_sequence_stressguard_cv0.540347_unsubmitted.ipynb`
  - metadata:
    `reports/experiment_334_id178_rollback_s1_s4_sequence_stressguard_cv0.540347_unsubmitted.json`
  - decision note:
    `reports/decision_334_id178_rollback_s1_s4_sequence_stressguard_cv0.540347_unsubmitted.md`
  - base OOF/test:
    `artifacts/base_predictions/base_oof_id_334_id178_rollback_s1_s4_sequence_stressguard_cv0.540347.csv`,
    `artifacts/base_predictions/base_test_id_334_id178_rollback_s1_s4_sequence_stressguard_cv0.540347.csv`
- Local OOF CV is `0.540347`, worse than ID178 local `0.538638`, but stress
  diagnostics prefer the rollback because S1/S4 were fragile under stricter
  checks.
- Decision: no DACON submission was made. ID334 is a weak/moderate Public
  `<0.60` controlled-probe hypothesis, not a Public `0.58`-plausible candidate.
  It requires explicit user approval before any leaderboard submission.

### 2026-06-17 22:15 KST update: ID335 alpha refinement artifact prepared, not submitted

- A report-only alpha scan around ID334 found that restoring half of the S4
  sequence overlay gives a tiny local improvement while keeping the stress
  diagnostics close to ID334:
  `reports/id178_sequence_alpha_scan_post_id334_20260617.json/.md` and
  `reports/id178_sequence_alpha_refine_post_id334_20260617.json/.md`.
- A fully matched generated-unsubmitted artifact was created:
  `submission_id_335_id178_alpha_s4half_stressguard_cv0.539690_unsubmitted.csv`.
- Matching files:
  - notebook:
    `notebooks/history/submission_335_id178_alpha_s4half_stressguard_cv0.539690_unsubmitted.ipynb`
  - metadata:
    `reports/experiment_335_id178_alpha_s4half_stressguard_cv0.539690_unsubmitted.json`
  - decision note:
    `reports/decision_335_id178_alpha_s4half_stressguard_cv0.539690_unsubmitted.md`
  - base OOF/test:
    `artifacts/base_predictions/base_oof_id_335_id178_alpha_s4half_stressguard_cv0.539690.csv`,
    `artifacts/base_predictions/base_test_id_335_id178_alpha_s4half_stressguard_cv0.539690.csv`
- Local OOF CV is `0.539690`, between ID178 (`0.538638`) and ID334
  (`0.540347`). The targeted stress row reports stress mean `0.573622`, min
  stress improvement `+0.001896`, and failed-vector projection `+0.0032`.
- Decision: no DACON submission was made. ID335 is a weak Public `<0.60`
  controlled-probe hypothesis only. It is still same-family with ID178/ID334
  and is not Public `0.58`-plausible.

### 2026-06-17 22:19 KST update: target-specific demographic gate

- A no-training/no-submission demographic gate report was created:
  `reports/target_specific_demographic_gate_post_id335_20260617.json/.md`.
- Using the completed Kaggle demographic audit, only `Q2`, `S2`, and `S4` are
  allowed for future target-specific demographic research because demo features
  beat no-demo alternatives under both subject-hole and LOSO stress.
- `Q3` is watch-only because it wins LOSO but not subject-hole. `Q1`, `S1`,
  and `S3` are rejected for now because LOSO does not support them.
- This is not promotion evidence and creates no submission artifact.

### 2026-06-17 22:25 KST update: demographic overlay rejected

- A report-only target overlay scan was created:
  `reports/demographic_target_overlay_post_id335_20260617.json/.md`.
- The scan tested tiny demographic/static overlays on the allowed targets
  `Q2`, `S2`, and `S4` over ID335. No target found a guarded positive alpha.
- Result: no CSV, notebook, base artifact, submission-log row, or DACON
  submission was created. Demographic/static features remain research-only and
  globally frozen.

### 2026-06-17 22:27 KST update: post-ID335 anchor rerank

- A no-model rerank of existing base-test artifacts was created:
  `reports/post_id335_public_anchor_rerank_20260617.json/.md`.
- The scan now includes ID334 and ID335. ID335 remains a prepared sequence
  reference with local `0.539690`, max delta vs ID178 `0.111949`, and failed
  projection `+0.0032`; ID334 is slightly farther from ID178 and has local
  `0.540347`.
- Top reviewable low-movement rows by the conservative screening proxy are
  mostly failed-direction/orthogonal artifacts (`307`, `295`, `304`, `297`),
  which should not be trusted directly after ID329. The cleanest non-failed
  row to manually review remains ID179, the ID178/ID177 tempered sequence
  anchor blend.
- Decision: no DACON submission and no new artifact promotion from rerank
  alone.

### 2026-06-17 22:31 KST update: ID179 vs ID335 manual review

- A report-only sequence probe review was created:
  `reports/id179_vs_id335_sequence_probe_review_20260617.json/.md`.
- ID179 remains locally stronger than ID335 (`0.538290` vs `0.539690`), has a
  slightly better slice-robust score (`0.563975` vs `0.565340`), and is much
  closer to the confirmed public anchor ID178 (`max Δ178 0.051824` vs
  `0.111949`).
- Interpretation: if a same-family controlled public probe is ever explicitly
  approved, ID179 is cleaner than ID334/ID335. It is still non-strict,
  OOF-selected, same-family with ID178, and not Public `0.58`-plausible.
- Decision: no DACON submission and no new artifact promotion.

### 2026-06-17 22:36 KST update: failed-direction candidates frozen

- A report-only audit of the post-ID335 rerank top failed-direction candidates
  was created:
  `reports/failed_direction_candidates_post_id335_20260617.json/.md`.
- ID295/296/297 are useful as anti-failed-vector diagnostics, but they are
  worse than ID179 locally and robustly, so they are not probe candidates.
- ID304/305/306/310/311 gain local CV with small test movement, but the gain
  comes from non-nested/full-OOF recombination. ID329 already showed this class
  of evidence can preserve the public plateau instead of improving it.
- ID307 is low-movement and negative-projection, but the local gain over ID179
  is only `+0.000498` and still comes from non-nested full-OOF selection.
- Decision: no DACON submission and no new artifact promotion. These rows stay
  diagnostic-only unless a fold-nested validation path is built.

### 2026-06-17 22:38 KST update: candidate outer-selector diagnostic

- A selection-nested report over existing OOF artifacts was created:
  `reports/candidate_outer_selector_post_id335_20260617.json/.md`.
- Subject-hole choice-nested overall was `0.521786`; LOSO choice-nested overall
  was `0.522195`. Both mostly select ID310/ID311, with ID306 appearing for S2.
- Interpretation: the failed-direction artifacts remain locally attractive even
  when the choice among artifacts is made inside outer folds, but this only
  nests the selector, not the artifact generation. The selected candidates are
  still non-nested/full-OOF products.
- Decision: no DACON submission and no artifact promotion. Use this only as a
  map for a future true fold-nested rebuild of the failed-direction idea.

### 2026-06-17 22:44 KST update: fold-internal failed-direction rebuild diagnostic

- A stricter report-only diagnostic was created:
  `reports/fold_internal_failed_direction_rebuild_post_id335_20260617.json/.md`.
- It recomputed beta, projection, and source selection inside each outer fold,
  while still using existing source OOF artifacts.
- Best subject-hole setting: `id310_like_cap120_lam160`, local `0.522506`,
  robust `0.548380`, selected source IDs `[242, 243, 249, 252, 268, 276, 278]`.
- Best LOSO setting: `id310_like_cap120_lam160`, local `0.523274`, robust
  `0.549054`, selected source IDs `[242, 249, 252, 268, 276, 278]`.
- Interpretation: a meaningful part of the failed-direction signal survives
  fold-internal recomputation, so a true model-level fold-nested rebuild is
  justified.
- Decision: no CSV, notebook, submission-log row, artifact promotion, or DACON
  submission was created. This is still not Public `0.58` evidence because the
  source artifacts were generated before the outer-fold replay.

### 2026-06-17 22:49 KST update: true nested rebuild blueprint

- A no-submit rebuild blueprint was created:
  `reports/true_nested_rebuild_blueprint_post_id335_20260617.json/.md`.
- It audits the fold-internal selected source IDs `242`, `243`, `249`, `252`,
  `268`, `276`, and `278`.
- Result: every selected source is non-nested. The set splits into
  second-stage full-OOF calibration (`242`, `243`, `252`), target-wise
  delta-capped bridge selection (`249`), and target-wise pair/logit blend
  selection (`268`, `276`, `278`).
- Decision: still no CSV, notebook, submission-log row, artifact promotion, or
  DACON submission. The next credible modeling task is a true model-level
  nested rebuild lane, not another public-guided recombination.

### 2026-06-17 22:55 KST update: fold-internal pair/logit blend rebuild diagnostic

- A stricter no-submit pair/logit selection diagnostic was created:
  `reports/fold_internal_pair_blend_rebuild_post_id335_20260617.json/.md`.
- It rebuilt target-wise pair/logit choices inside each outer fold while still
  using existing source OOF artifacts.
- Best subject-hole row: `full_upstream_cap030`, local `0.399920`, robust
  `0.423760`.
- Best LOSO row: `full_upstream_cap030`, local `0.400168`, robust `0.423989`.
- The best row that directly excludes ID252 was `direct_no_252_cap022`, local
  `0.433848`, robust `0.457077`; this is not truly ultra-free because ID254
  and ID255 are themselves descendants of ultra calibration artifacts.
- Interpretation: the pair/logit selector signal survives fold-internal replay,
  but the winning choices depend heavily on ultra source ID252 plus ID255, with
  S3 using an ID178/ID254 blend.
- Decision: no CSV, notebook, submission-log row, artifact promotion, or DACON
  submission. This strengthens the case for a true model-level nested rebuild
  of the ID252/ID255/ID254 source path; it does not create Public `0.58`
  evidence by itself.

### 2026-06-17 22:59 KST update: fold-internal second-stage calibration diagnostic

- A no-submit calibration-layer diagnostic was created:
  `reports/fold_internal_second_stage_calibration_post_id335_20260617.json/.md`.
- It refit existing second-stage calibration recipes inside each outer fold
  while still using pre-existing base OOF artifacts.
- Critical finding: upper calibrations on already extreme ID242 survive
  fold-internal replay (`id250_like_on_id242` around local `0.262`), but the
  lower source layer `id242_like_on_id238` collapses from full-OOF `0.357651`
  to `0.585137` subject-hole and `0.551329` LOSO.
- Interpretation: the attractive ID252/ID255 pair-blend signal is downstream
  of ID242, and ID242 itself is not validated under fold-internal calibration.
- Decision: no CSV, notebook, submission-log row, artifact promotion, or DACON
  submission. Do not rebuild only the upper pair-blend/calibration layers; the
  next credible path must either rebuild the ID242 source layer truly nested or
  avoid that branch.

### 2026-06-17 23:08 KST update: conservative ID238 calibration salvage check

- A no-submit conservative calibration diagnostic was created:
  `reports/id238_conservative_calibration_post_id335_20260617.json/.md`.
- It tested weak temperature shifts plus shrinked low-quantile/isotonic
  alternatives on top of pre-existing ID238 OOF predictions, with all choices
  made inside outer folds.
- Best subject-hole row: `temp_safe_cap012`, local `0.461268`, robust
  `0.483738`, improvement versus ID238 base `+0.001097`.
- Best LOSO row: `temp_safe_cap012`, local `0.460105`, robust `0.482918`,
  improvement versus ID238 base `+0.002260`.
- Low-quantile and isotonic shrink variants were worse than ID238 base under
  both subject-hole and LOSO.
- Interpretation: the ID238 lower layer has only a tiny conservative
  temperature-calibration salvage signal. It does not support rebuilding the
  ID242 q240/isotonic branch as-is.
- Decision: no CSV, notebook, submission-log row, artifact promotion, or DACON
  submission. Future ID242-descendant work should be deprioritized unless a
  true model-level rebuild creates a new non-OOF source signal.

### 2026-06-17 23:12 KST update: surviving-candidate map excluding ID242 chain

- A dependency-aware no-submit survivor map was created:
  `reports/surviving_candidates_excluding_id242_post_id335_20260617.json/.md`.
- It scanned `238` base-test candidates and marked `93` as descendants of the
  broken ID239/ID242/ID249/ID250/ID251/ID252 chain.
- After excluding that chain, `21` reviewable survivors remained under the
  conservative movement/projection screen.
- Top survivor: ID179, local `0.538290`, robust `0.563975`, proxy `0.606255`,
  max Δ178 `0.051824`, failed projection `+0.0011`.
- The next survivors are mostly sequence-temporal family rows (`190`, `191`,
  `193`, `177`, `198`, `200`, `203`, `206`) rather than calibration-frontier
  rows.
- Decision: no CSV, notebook, submission-log row, artifact promotion, or DACON
  submission. The practical branch after cutting ID242 is ID179/sequence
  review or a genuinely new model-level source, not another calibration
  descendant.

### 2026-06-17 23:16 KST update: sequence survivor review

- A no-submit expanded sequence-survivor review was created:
  `reports/sequence_survivor_review_post_id335_20260617.json/.md`.
- It compared ID179 against the leading non-ID242 survivor rows:
  `190`, `191`, `193`, `177`, `198`, `200`, `203`, `206`, plus references
  `178`, `334`, `335`, and the low-local ID180 stress row.
- ID179 remains the cleanest low-movement review anchor: local `0.538290`,
  robust `0.563975`, local gain vs ID178 `+0.000348`, max Δ178 `0.051824`,
  failed projection `+0.0011`.
- ID193/198/200/203/206 improve local CV more, but require much larger
  movement from ID178 (`~0.122-0.136` max Δ178) for only small local gains.
- ID180 has stronger local gain, but failed-direction projection is too high
  (`+0.0348`), so it is rejected as a public probe.
- Decision: no CSV, notebook, submission-log row, artifact promotion, or DACON
  submission. ID179 can be treated as the cautious manual-review anchor only;
  it is still not Public `0.58` evidence.

### 2026-06-17 23:20 KST update: fold-internal sequence survivor selector

- A no-submit fold-internal target selector was created:
  `reports/fold_internal_sequence_survivor_selector_post_id335_20260617.json/.md`.
- It selected among fixed sequence-survivor OOF artifacts inside each outer
  fold and target. Source artifacts remain pre-existing/non-nested.
- Local-only selection improved subject-hole to `0.537268` and LOSO to
  `0.536920`, selecting IDs `[178, 179, 190, 191, 193, 198, 200, 203, 206]`
  across targets/folds.
- Once movement and failed-projection caps/penalties were applied, every
  risk-controlled scenario selected only ID178 and returned to local
  `0.538638`.
- Interpretation: sequence descendants have fold-internal local signal, but it
  is not strong enough to survive the current public-anchor movement budget.
- Decision: no CSV, notebook, submission-log row, artifact promotion, or DACON
  submission. This supports using sequence descendants as research targets for
  new source generation, not as immediate public probes.

### 2026-06-17 23:24 KST update: sequence source-generation plan

- A no-submit report-only source-generation plan was created:
  `reports/sequence_source_generation_plan_post_id335_20260617.json/.md`.
- It translates the local-only fold-internal selector into target-specific
  diagnostic lanes:
  - `Q1`: ID200-like SAX/PAA sequence pattern, support `86.67%`.
  - `Q2`: ID193-like sequence-summary embedding, support `100.00%`.
  - `Q3`: ID206-like wavelet/spectral sequence, support `80.00%`.
  - `S1`: ID190-like random-conv sequence refresh, support `93.33%`.
  - `S2`: ID206-like wavelet/spectral sequence, support `53.33%`.
  - `S3`: ID198-like smartphone/light compact source, support `80.00%`.
  - `S4`: ID206-like wavelet/spectral sequence, support `73.33%`.
- The report includes concrete no-`--promote` commands for each lane. Heavier
  raw-sequence jobs, especially S1 random-conv refresh, should run on Kaggle if
  local memory/CPU becomes a concern.
- Decision: no Kaggle push, CSV, notebook, submission-log row, artifact
  promotion, or DACON submission. These commands are source-generation
  diagnostics only and require review before any matched artifact can be
  promoted.

### 2026-06-17 23:34 KST update: sequence source-generation execution review

- Three high-priority report-only source diagnostics were executed without
  `--promote`:
  - `reports/id191_sequence_summary_embedding_post_id335_20260617.json`
  - `reports/id193_smartphone_light_compact_post_id335_20260617.json`
  - `reports/id203_wavelet_spectral_sequence_post_id335_20260617.json`
- A summary review was created:
  `reports/sequence_source_generation_execution_review_post_id335_20260617.json/.md`.
- Results:
  - Q2 sequence-summary lane: base `0.538009` to best `0.536429`,
    improvement `+0.001581`; changed targets `Q1/Q2/Q3`.
  - S3 smartphone/light lane: base `0.536429` to best `0.536380`,
    improvement `+0.000049`; changed target `S3`.
  - Q3/S2/S4 wavelet-spectral lane: base `0.535980` to best `0.535764`,
    improvement `+0.000216`; changed targets `Q3/S2/S4`.
- Interpretation: the planned sequence-source gains are reproducible as
  public-structure diagnostics, but still small and not model-level nested.
- Decision: no Kaggle push, CSV, notebook, submission-log row, artifact
  promotion, or DACON submission. These results guide future source generation
  only.

### 2026-06-17 00:45 KST update: ID329 public result

- ID329 was submitted to DACON as submission `1465639`:
  `submission_id_329_post328_combo4_cap080_budget050_practical_cv0.516718_unsubmitted.csv`.
- Submitted at `2026-06-17 00:45:07`; Public LB `0.6017908577`.
- Local CV was `0.516718`, so the public gap is `+0.085073`.
- Compared with confirmed anchor ID178 Public `0.6014113009`, ID329 is slightly worse by
  `+0.0003795568`. Compared with failed ID287 Public `0.6698787318`, ID329 is much safer.
- Interpretation: the failed-direction guard prevented an ID287-style collapse, but this practical constrained-combo family did not improve toward Public `0.55`.
- Decision: do not automatically submit ID330/ID331; they are same-family bolder moves. Re-rank using ID178 and ID329 as public anchors and ID287 as the failed vector.
- Supporting reports:
  - `reports/dacon_submission_result_id329_public0.6017908577.json`
  - `reports/public_score_response_diagnostic_post_id329_20260617.json/.md`

### 2026-06-16 21:50 KST update: fifth-pass no-promotion check

- A fifth-pass constrained-combo scan included IDs `329-333` as sources:
  `reports/post333_failed_direction_constrained_combo_fifthpass_scan_20260616.json/.md`.
- Result: no new submission artifact was promoted. The practical cap-0.08 row
  improved only from ID329 local `0.516718` to `0.516601`, and the risk geometry
  was unchanged. That gain is too small to justify another unique submission
  file/log/notebook bundle.
- Decision note:
  `reports/post333_fifthpass_no_promotion_decision_20260616.json/.md`.
- Current recommendation remains ID329 as the practical next public candidate if
  a DACON submission is explicitly approved. ID330/ID331 are bolder; ID332/ID333
  are diagnostic/frontier-stress only.
- No DACON submission was made.

### 2026-06-16 21:35 KST update: fourth-pass constrained-combo scan

- A fourth-pass constrained-combo scan included IDs `323-328` as sources:
  `reports/post328_failed_direction_constrained_combo_fourthpass_scan_20260616.json/.md`.
- New unsubmitted artifacts:
  - ID329:
    `submission_id_329_post328_combo4_cap080_budget050_practical_cv0.516718_unsubmitted.csv`.
    Local `0.516718`, max delta vs ID178 `0.079192`, aggregate projection
    `+0.0499`. This narrowly supersedes ID323 as the current practical
    candidate.
  - ID330:
    `submission_id_330_post328_combo4_cap120_budget050_tightscore_cv0.511769_unsubmitted.csv`.
    Local `0.511769`, max delta `0.113916`, projection `+0.0499`; bolder
    score-pressure card.
  - ID331:
    `submission_id_331_post328_combo4_cap150_budget050_scoreguard_cv0.509930_unsubmitted.csv`.
    Local `0.509930`, max delta `0.149464`, projection `+0.0498`; high-movement
    scoreguard.
  - ID332:
    `submission_id_332_post328_combo4_cap240_budget050_wideguard_cv0.503202_unsubmitted.csv`.
    Local `0.503202` under projection budget `+0.05`, but max delta `0.227831`;
    wide diagnostic, not first probe.
  - ID333:
    `submission_id_333_post328_combo4_cap320_budget280_frontier_cv0.466963_unsubmitted.csv`.
    New local frontier, ID178-gap proxy around `0.529736`, but projection
    `+0.2693` and max delta `0.317996`; stress only.
- Current recommendation: ID329 is the practical next candidate if a new public
  probe is explicitly approved. ID330/ID331 are bolder. ID332/ID333 show local
  pressure toward the 0.55 target zone, but the public-failure geometry says not
  to lead with them.
- Supporting manifest:
  `reports/candidate_tier_manifest_post_id333_fourthpass_combo_20260616.json/.md`.
- No DACON submission was made for IDs 329-333 while creating this update.

### 2026-06-16 21:10 KST update: third-pass constrained-combo scan

- A third-pass constrained-combo scan included IDs `318-322` as sources:
  `reports/post322_failed_direction_constrained_combo_thirdpass_scan_20260616.json/.md`.
- New unsubmitted artifacts:
  - ID323:
    `submission_id_323_post322_combo3_cap080_budget050_practical_cv0.516767_unsubmitted.csv`.
    Local `0.516767`, max delta vs ID178 `0.077587`, aggregate projection
    `+0.0498`. This narrowly supersedes ID318 as the current practical
    candidate.
  - ID324:
    `submission_id_324_post322_combo3_cap100_budget050_balanced_cv0.514188_unsubmitted.csv`.
    Local `0.514188`, max delta `0.099643`, projection `+0.0498`; bolder
    budget-0.05 candidate.
  - ID325:
    `submission_id_325_post322_combo3_cap150_budget050_scoreguard_cv0.510506_unsubmitted.csv`.
    Local `0.510506`, max delta `0.149716`, projection `+0.0498`; strong but
    high movement.
  - ID326:
    `submission_id_326_post322_combo3_cap240_budget050_wideguard_cv0.503613_unsubmitted.csv`.
    Local `0.503613` under projection budget `+0.05`, but max delta `0.227831`;
    wide-stress, not first probe.
  - ID327:
    `submission_id_327_post322_combo3_cap280_budget000_zerowide_cv0.511571_unsubmitted.csv`.
    Near-zero projection but max delta `0.270086`; diagnostic only.
  - ID328:
    `submission_id_328_post322_combo3_cap240_budget240_frontier_cv0.473842_unsubmitted.csv`.
    New local frontier, projection `+0.2399`; stress only.
- Current recommendation: ID323 is the practical next candidate if a new public
  probe is approved. ID324/ID325 are bolder. ID326/ID328 show the local score can
  be pushed much lower, but movement is too wide for a first public check.
- Supporting manifest:
  `reports/candidate_tier_manifest_post_id328_thirdpass_combo_20260616.json/.md`.

### 2026-06-16 20:55 KST update: constrained-combo projection-budget scan

- A Pareto/dynamic-program constrained-combo scan was added:
  `reports/post317_failed_direction_constrained_combo_scan_20260616.json/.md`.
  This directly optimizes target combinations under fixed aggregate projection
  budgets instead of relying on a Lagrangian penalty.
- New unsubmitted artifacts:
  - ID318:
    `submission_id_318_post317_combo_cap080_budget050_practical_cv0.516844_unsubmitted.csv`.
    Local `0.516844`, max delta vs ID178 `0.079086`, aggregate projection
    `+0.0497`. This is the current best practical score-pressure candidate.
  - ID319:
    `submission_id_319_post317_combo_cap100_budget050_balanced_cv0.514544_unsubmitted.csv`.
    Local `0.514544`, max delta `0.098294`, projection `+0.0496`; bolder than
    ID318.
  - ID320:
    `submission_id_320_post317_combo_cap120_budget050_scoreguard_cv0.511972_unsubmitted.csv`.
    Local `0.511972`, max delta `0.113916`, projection `+0.0499`; strongest
    budget-0.05 scoreguard but higher movement.
  - ID321:
    `submission_id_321_post317_combo_cap200_budget000_zerowide_cv0.515882_unsubmitted.csv`.
    Local `0.515882`, projection `-0.0002`, but max delta `0.195733`; wide
    diagnostic/stress.
  - ID322:
    `submission_id_322_post317_combo_cap200_budget160_frontier_cv0.489676_unsubmitted.csv`.
    Local `0.489676`, projection `+0.1597`; lower projection than ID317 but
    still frontier stress.
- Current recommendation: ID318 supersedes ID314 as the best practical next
  candidate. ID319/ID320 are bolder score-pressure probes. ID321/ID322 should
  wait for public feedback because they are structurally wide or frontier-stress.
- Supporting manifest:
  `reports/candidate_tier_manifest_post_id322_constrained_combo_20260616.json/.md`.

### 2026-06-16 20:35 KST update: second-pass aggregate budget

- A second-pass aggregate-budget scan reused IDs `308-313` as sources:
  `reports/post313_failed_direction_aggregate_budget_secondpass_scan_20260616.json/.md`.
- New unsubmitted artifacts:
  - ID314:
    `submission_id_314_post313_secondpass_cap080_lam120_practical_cv0.520647_unsubmitted.csv`.
    Local `0.520647`, max delta vs ID178 `0.078300`, aggregate projection to
    the ID287 failed direction `+0.0266`. This is now the strongest practical
    score-pressure candidate.
  - ID315:
    `submission_id_315_post313_secondpass_cap180_lam120_scoreguard_cv0.509564_unsubmitted.csv`.
    Local `0.509564`, projection `+0.0416`, but max delta `0.175338`; this is
    high-movement and should wait for public feedback.
  - ID316:
    `submission_id_316_post313_secondpass_cap200_lam160_zerowide_cv0.517120_unsubmitted.csv`.
    Local `0.517120`, projection `-0.0036`, but max delta `0.194423`; wide
    zero-projection stress card.
  - ID317:
    `submission_id_317_post313_secondpass_cap200_lam000_frontierstress_cv0.481968_unsubmitted.csv`.
    New local frontier, but projection `+0.2495` and max delta `0.196587`.
    Treat only as local frontier/stress, not a first public probe.
- Current recommendation: ID314 is the best practical next candidate if a new
  submission is approved. ID308/ID309 remain cleaner fallbacks. ID315/ID316/ID317
  prove the local frontier can move, but are too structurally wide for a first
  post-ID287 public check.
- Supporting manifest:
  `reports/candidate_tier_manifest_post_id317_secondpass_aggregate_budget_20260616.json/.md`.

### 2026-06-16 20:20 KST update: aggregate failed-direction budget and under-0.50 local stress cards

- The safe orthogonal cards were still conservative locally, so a new aggregate
  failed-direction budget scan was added:
  `reports/post307_failed_direction_aggregate_budget_scan_20260616.json/.md`.
  Unlike ID304-ID307, this scan lets targets trade risk against each other under
  one aggregate projection budget to push local CV lower.
- New unsubmitted artifacts:
  - ID308:
    `submission_id_308_post307_aggbudget_cap060_lam120_scoreguard_cv0.521679_unsubmitted.csv`.
    Local `0.521679`, max delta vs ID178 `0.059530`, aggregate projection to the
    ID287 failed direction `+0.0340`. This is now the best controlled
    score-pressure probe.
  - ID309:
    `submission_id_309_post307_aggbudget_cap050_lam120_balanced_cv0.523249_unsubmitted.csv`.
    Local `0.523249`, max delta `0.049147`, failed-direction projection
    `+0.0328`. This is the best balanced controlled probe.
  - ID310:
    `submission_id_310_post307_aggbudget_cap120_lam160_zeroproj_cv0.521885_unsubmitted.csv`.
    Local `0.521885`, near-zero projection, but max delta `0.114878`; treat as
    wide-movement stress, not a first probe.
  - ID311:
    `submission_id_311_post307_aggbudget_cap100_lam160_zeroproj_cv0.524195_unsubmitted.csv`.
    Local `0.524195`, near-zero projection, max delta `0.091284`; high movement.
  - ID312:
    `submission_id_312_post307_aggbudget_cap080_lam000_scorepressure_cv0.496824_unsubmitted.csv`.
    This breaks local `0.50`, but projection is `+0.2113` toward the ID287
    failure direction. It is a local-score stress artifact, not a recommended
    first submission.
  - ID313:
    `submission_id_313_post307_aggbudget_cap120_lam000_frontierstress_cv0.488205_unsubmitted.csv`.
    Current local frontier (`0.488205`), but projection is `+0.2412` and max
    delta `0.113424`. Do not submit unless intentionally stress-testing the
    positive-failed-direction family.
- Current recommendation: if submitting next, prefer ID309 for balanced risk or
  ID308 for slightly stronger local score. ID312/ID313 prove the local score can
  be pushed below `0.50`, but ID287 already showed this family can collapse
  publicly. Public `0.55` remains unproven.
- Supporting manifest:
  `reports/candidate_tier_manifest_post_id313_aggregate_budget_20260616.json/.md`.

### 2026-06-16 20:00 KST update: ID287 public failure response and failed-direction orthogonal probes

- ID287 was submitted to DACON as submission id `1465470` at
  `2026-06-16 19:08:14` and scored Public `0.6698787318` despite local
  `0.378934`. The gap is `+0.2909447318`, so the fixed-temperature/full-OOF
  local-frontier family is demoted as a severe public failure.
- ID178 remains the confirmed public anchor: local `0.538638`, Public
  `0.6014113009`.
- Temporal recovery candidates were promoted but remain risky:
  - ID301: local `0.517635`, exact prediction duplicate of ID289; do not spend
    a submit slot on it.
  - ID302: local `0.519801`, max4 temporal partial; still projects toward the
    ID287 failure direction.
  - ID303: local `0.505102`, stronger temporal score card; also projects toward
    ID287.
- New post-failure scan:
  `reports/post303_failed_direction_orthogonal_scan_20260616.json/.md`.
  It treats the ID178→ID287 test-prediction vector as a concrete failed public
  direction and removes that component from existing stronger artifacts.
- New unsubmitted orthogonal recovery artifacts:
  - ID304:
    `submission_id_304_post303_orthogonal_cap030_proj000_safe_cv0.532349_unsubmitted.csv`.
    Local `0.532349`, max delta vs ID178 `0.029765`, aggregate projection to
    the ID287 failed direction approximately `0.0000`. This is the preferred
    first public probe if we submit again.
  - ID305:
    `submission_id_305_post303_orthogonal_cap050_proj000_balanced_cv0.529570_unsubmitted.csv`.
    Local `0.529570`, max delta `0.048772`, failed-direction projection
    approximately `0.0000`. Use only after ID304 or if we accept more movement.
  - ID306:
    `submission_id_306_post303_orthogonal_cap080_proj000_aggressive_cv0.526367_unsubmitted.csv`.
    Local `0.526367`, max delta `0.074735`, failed-direction projection
    approximately `0.0000`. Stronger local but too much movement for a first
    post-ID287 probe.
  - ID307:
    `submission_id_307_post303_orthogonal_cap030_neg010_probe_cv0.537792_unsubmitted.csv`.
    Local `0.537792`, max delta `0.026555`, projection `-0.0170`. This is an
    information card for whether moving opposite ID287 helps public.
- Current recommendation: if the user approves one submission, submit ID304
  first. If ID304 is not worse than ID178, ID305 becomes the next score-recovery
  probe. ID306 waits for confirmation; ID307 is diagnostic, not a direct
  Public-0.55 chase. Public `0.55` remains unproven.
- Supporting manifest:
  `reports/candidate_tier_manifest_post_id307_failed_direction_orthogonal_20260616.json/.md`.


### 2026-06-16 17:35 KST update: ID247 S3 remote sweep and ID178 public-anchor bridges

- Kaggle strict runner v25 completed the ID247 S3 random-conv expansion sweep
  (`512/768/1024` kernels, six seeds). All six S3 diagnostics selected `base`;
  S3 stayed at `0.46511496294445753`. No random-conv S3 artifact was promoted.
  Supporting review: `reports/id247_s3_remote_random_conv_v25_review_20260616.json/.md`.
- Because S3 did not open, the next move shifted from larger sequence expansion to
  a confirmed-public-anchor bridge. ID178 is the best confirmed Public artifact
  (`0.6014113009`, local `0.538638`, gap `+0.0627733009`).
- New scan: `reports/id178_confirmed_public_anchor_bridge_scan_20260616.json/.md`
  moves target-wise from ID178 toward IDs `247`, `248`, and `252` while capping
  max test-delta versus ID178.
- Promoted unsubmitted candidates:
  - ID253: `submission_id_253_id178_anchor_cap012_frontier_bridge_cv0.477131_unsubmitted.csv`.
    Cap `0.12`, max test delta vs ID178 `0.106724`, ID178-gap estimate about
    `0.539904`. This is now the preferred conservative public probe candidate,
    but it remains full-OOF selected and not public proof.
  - ID254: `submission_id_254_id178_anchor_cap015_frontier_bridge_cv0.471985_unsubmitted.csv`.
    Cap `0.15`, max test delta vs ID178 `0.146734`, ID178-gap estimate about
    `0.534758`. This is the bolder second probe candidate.
- Updated candidate ordering: ID253 first, ID254 second, ID247 third. DACON
  submission still requires explicit approval.


### 2026-06-16 17:45 KST update: fine ID178 public-anchor bridge

- A fine-grained ID178 confirmed-public-anchor bridge scan expanded frontier IDs
  to `247/248/249/250/251/252/242/243` and added smaller alpha values.
  Supporting scan: `reports/id178_confirmed_public_anchor_bridge_fine_scan_20260616.json/.md`.
- New unsubmitted artifacts:
  - ID255: `submission_id_255_id178_anchor_fine_cap012_frontier_bridge_cv0.469816_unsubmitted.csv`.
    Same nominal cap bucket as ID253 (`0.12`), local `0.469816`, max test delta
    vs ID178 `0.115233`, ID178-gap estimate about `0.532590`. It is sharper
    than ID253 but uses ultra-frontier IDs 250/251 in small weights.
  - ID256: `submission_id_256_id178_anchor_fine_cap014_frontier_bridge_cv0.464121_unsubmitted.csv`.
    Cap `0.14`, local `0.464121`, max delta `0.131771`, ID178-gap estimate
    about `0.526894`.
  - ID257: `submission_id_257_id178_anchor_fine_cap018_frontier_bridge_cv0.433079_unsubmitted.csv`.
    Cap `0.18`, local `0.433079`, max delta `0.176195`; high-risk stress card.
- Updated probe ordering: ID255 first, ID253 cleaner fallback, ID256 sharper
  second probe, ID254 fallback, ID257 high-risk only. Public `0.55` remains
  unproven until a DACON result is recorded.


### 2026-06-16 17:55 KST update: clean fine bridge and robustness review

- Robustness diagnostics were added for the public-anchor bridge family:
  `reports/post_id257_public_anchor_bridge_robustness_20260616.json/.md` and
  `reports/post_id259_public_anchor_bridge_robustness_20260616.json/.md`.
- A clean fine scan excluded the most extreme source IDs `250/251/242/243` and
  used only IDs `247/248/249/252`:
  `reports/id178_confirmed_public_anchor_bridge_clean_fine_scan_20260616.json/.md`.
- New unsubmitted artifacts:
  - ID258: `submission_id_258_id178_anchor_clean_cap016_frontier_bridge_cv0.457647_unsubmitted.csv`.
    Cap `0.16`, local `0.457647`, max delta vs ID178 `0.159452`, ID178-gap
    estimate about `0.520420`; cleaner than ID256 because it avoids IDs
    `250/251/242/243`.
  - ID259: `submission_id_259_id178_anchor_clean_cap018_frontier_bridge_cv0.443567_unsubmitted.csv`.
    Cap `0.18`, local `0.443567`, max delta `0.178005`; clean high-movement
    stress card.
- Updated probe ordering: ID255 remains first because it keeps cap `0.12`; ID258
  is now the best sharper clean alternative; ID253 remains the cleaner fallback;
  ID259/ID257 are stress cards only.


### 2026-06-16 post-ID262 update: delta-constrained pair blends

- A target-wise pair probability/logit blend scan was run around confirmed
  public anchor ID178 and recent bridge candidates. Supporting scan:
  `reports/id178_delta_constrained_pair_blend_scan_20260616.json/.md`.
- New unsubmitted artifacts:
  - ID260: `submission_id_260_id178_delta_pair_cap012_blend_cv0.463684_unsubmitted.csv`.
    Local `0.463684`, robustness `0.493703`, max test delta vs ID178
    `0.119513`, ID178-gap estimate about `0.526457`. This is now the best
    score-targeting first public probe **if** full-OOF pair-blend risk is
    acceptable.
  - ID261: `submission_id_261_id178_delta_pair_cap016_blend_cv0.441919_unsubmitted.csv`.
    Local `0.441919`, robustness `0.471868`, max delta `0.159583`; sharper but
    uses ID252-style second-stage sources, so treat it as a second/stress card.
  - ID262: `submission_id_262_id178_delta_pair_cap018_blend_cv0.440031_unsubmitted.csv`.
    Local `0.440031`, robustness `0.469165`, max delta `0.179554`; strongest
    local pair-blend stress card, not a first probe.
- Updated probe ordering: ID260 first for score pressure, ID255 first if the
  priority is structural simplicity, then ID258/ID253 as cleaner alternatives.
  ID261/ID262 should wait for public feedback from a safer candidate.
- Supporting review:
  `reports/post_id262_public_anchor_pair_blend_robustness_20260616.json/.md`
  and `reports/candidate_tier_manifest_post_id262_20260616.json/.md`.


### 2026-06-16 post-ID263 update: direct-lite pair blend

- A follow-up direct-lite pair-blend scan excluded direct stress IDs
  `252/257/259/260/261/262` from the candidate pool and used only
  `178/253/255/258`. Supporting scan:
  `reports/id178_direct_lite_pair_blend_scan_20260616.json/.md`.
- New unsubmitted artifact:
  - ID263: `submission_id_263_id178_direct_lite_pair_cap012_blend_cv0.464978_unsubmitted.csv`.
    Local `0.464978`, robustness `0.495013`, max test delta vs ID178
    `0.119513`, ID178-gap estimate about `0.527751`. It is only `0.001294`
    worse than ID260 locally, while avoiding direct ID252/259 stress-source
    selection in the pair scan.
- Updated probe ordering: ID263 is now the preferred first public probe if we
  want the best balance of score pressure and risk story. ID260 remains the
  slightly sharper but wilder score-pressure candidate. ID255 is the simplest
  conservative fallback. ID261/ID262 remain stress cards only.
- Supporting review:
  `reports/post_id263_direct_lite_pair_blend_robustness_20260616.json/.md`
  and `reports/candidate_tier_manifest_post_id263_20260616.json/.md`.


### 2026-06-16 post-ID266 update: fine direct-lite ladder

- A fine direct-lite pair-blend scan reran the `178/253/255/258` source pool at
  `0.01` weight resolution while continuing to exclude direct stress IDs
  `252/257/259/260/261/262`. Supporting scan:
  `reports/id178_direct_lite_pair_blend_fine_scan_20260616.json/.md`.
- New unsubmitted artifacts:
  - ID264: `submission_id_264_id178_direct_lite_fine_cap012_blend_cv0.464758_unsubmitted.csv`.
    Local `0.464758`, robustness `0.494762`, max test delta vs ID178
    `0.119995`, ID178-gap estimate about `0.527532`. This supersedes ID263 as
    the preferred first public probe.
  - ID265: `submission_id_265_id178_direct_lite_fine_cap014_blend_cv0.458478_unsubmitted.csv`.
    Local `0.458478`, robustness `0.488017`, max delta `0.139926`; intermediate
    clean sharper card.
  - ID266: `submission_id_266_id178_direct_lite_fine_cap015_blend_cv0.457894_unsubmitted.csv`.
    Local `0.457894`, robustness `0.487400`, max delta `0.149809`; best clean
    sharper second-probe card because it nearly matches ID258 while reducing
    max movement.
- Updated probe ordering: ID264 first, ID260 if we prefer a slightly sharper
  but less clean cap0.12 candidate, ID266 as the clean sharper second probe,
  then ID265/ID255/ID258. ID261/ID262 remain stress cards only.
- A follow-up 4-way convex grid over the same clean source pool
  `178/253/255/258` did not produce a material new artifact:
  `reports/id178_direct_lite_convex4_grid_scan_20260616.json/.md`. The best
  cap0.15 result was local `0.457888`, only `0.000006` better than ID266, so no
  submission artifact was promoted from this diagnostic.
- Supporting review:
  `reports/post_id266_direct_lite_fine_pair_blend_robustness_20260616.json/.md`
  and `reports/candidate_tier_manifest_post_id266_20260616.json/.md`.


### 2026-06-16 post-ID269 update: controlled risk-penalized broader pair blends

- A broader controlled pair-blend scan added ID253-ID266 around the confirmed
  public ID178 anchor and reported source-risk penalty variants. Supporting
  scan: `reports/post266_controlled_pair_blend_risk_scan_20260616.json/.md`.
- New unsubmitted artifacts:
  - ID267: `submission_id_267_post266_riskpen_cap012_p005_blend_cv0.461960_unsubmitted.csv`.
    Source-risk-penalized cap `0.12`, local `0.461960`, robustness `0.492124`,
    max delta vs ID178 `0.119969`, ID178-gap estimate about `0.524733`. It is
    the new score-targeting cap0.12 candidate, but it borrows limited
    target-wise signal from stress sources ID257/261.
  - ID268: `submission_id_268_post266_riskpen_cap015_p010_blend_cv0.448976_unsubmitted.csv`.
    Source-risk-penalized cap `0.15`, local `0.448976`, robustness `0.478923`,
    max delta `0.149985`, ID178-gap estimate about `0.511750`. This is the best
    balanced sharper Public-0.55 chase card.
  - ID269: `submission_id_269_post266_riskpen_cap016_p010_blend_cv0.443780_unsubmitted.csv`.
    Source-risk-penalized cap `0.16`, local `0.443780`, robustness `0.473962`,
    max delta `0.159906`, ID178-gap estimate about `0.506553`. Treat as a
    stress card, not a first probe.
- Updated probe ordering: ID267 if we accept limited stress-source exposure for
  a lower cap0.12 score, ID264 if we want the cleaner cap0.12 first probe, ID268
  as the best sharper second probe, and ID269 only after public feedback
  suggests this family is not overfitting.
- Supporting review:
  `reports/post_id269_controlled_risk_pair_blend_robustness_20260616.json/.md`
  and `reports/candidate_tier_manifest_post_id269_20260616.json/.md`.

- Confirmed leaderboard truth is unchanged: best submitted result is ID178,
  Public `0.6014113009`. Public `0.55` has not been proven.
- Current best cleaner/non-ID172 local branch is ID221:
  `submission_id_221_id218_public_feature_similarity_overlay_cv0.532990_unsubmitted.csv`.
  It improves ID218 from local `0.534103` to `0.532990`, and the split
  robustness lens from `0.565633` to `0.564811`.
- ID221 is not strict Subject-Hole proof. It is a public-structure
  feature-similarity overlay selected from a narrow full-grid diagnostic.
- Temporal-anchor refinements on ID218 were tested and rejected: all-target,
  Q123-only, and KNN temporal refinements all worsened local CV.
- The external `[Public_0.5917] LGBM+XGB anchor/Subject-hole/stability
  feature-selection/blend` notebook is useful as strategy guidance, not as a
  plug-and-play pipeline. It reinforces conservative anchoring,
  subject-hole/stability validation, and public-gap risk ranking, but it lacks
  the anchor artifacts and full feature build needed for reproduction.
- Follow-up transfer diagnostics from that notebook did **not** produce a new
  promotable signal on the current ladder:
  - `reports/public_sleep_guideline_overlay_diagnostic_20260616.json` and
    `reports/subject_hole_sleep_guideline_overlay_diagnostic_20260616.json`
    tested lightweight objective sleep-guideline overlays on ID221/226/231.
    ID226 and ID231 selected `base` for every S-target under both public
    interleaved and Subject-Hole lenses.
  - `reports/public_booster_stability_overlay_fast_scan_20260616.json` tested
    shallow stability-selected LGBM/XGB overlays on ID221/226/231. Even the
    optimistic full-OOF scan selected `base` for every target, so there is no
    reason to promote a booster-stability artifact from this path.
- ID221-to-ID172 bridge probes were generated after this checkpoint:
  - ID222 delta06 bridge: local `0.521966`, robustness `0.553200`.
  - ID223 delta08 bridge: local `0.518724`, robustness `0.549816`.
  These are lower than the previous ID219/ID220 bridge pair, but inherit the
  ID172 non-nested calibration risk. Treat ID223 as the current risky bridge
  frontier, not as public proof.
- The local frontier was extended again:
  - ID224 cap-0.15 convex bridge: local `0.509110`, robustness `0.539895`.
  - ID225 cap-0.30 convex bridge: local `0.496691`, robustness `0.528599`.
  - ID226 target-wise convex frontier: local `0.491771`, robustness `0.524870`.
  - ID227 second-stage isotonic/q160 frontier: local `0.464409`, robustness
    `0.495627`.
  ID227 is the first local score with enough nominal room to survive an
  ID178-sized public gap, but it is **ultra-risk**: second-stage full-OOF
  calibration, especially S4 q160. It is not public proof and should not be
  submitted without explicit approval.
- Stability-selected raw-feature overlays on ID221 and ID226 were tested and
  selected `base` for every target, so that route did not add a new stable
  signal.
- A target-wise oracle over 129 strict-era artifacts found the existing
  artifact recombination limit at ID226 (`0.491771`); going lower required the
  new second-stage calibration frontier ID227.
- A tempered second-stage ladder was added after ID227:
  - ID228 isotonic-only second-stage: local `0.482245`, robustness `0.513849`.
  - ID229 isotonic + S4 q40: local `0.479973`, robustness `0.510142`.
  - ID230 isotonic + S4 q80: local `0.472703`, robustness `0.504400`.
  ID228 is now the least aggressive second-stage candidate with nominal margin
  under the observed ID178 public gap (`0.482245 + 0.062774 ~= 0.545018`), but
  it is still full-OOF calibrated and not public proof.
- An outer/nested calibration diagnostic on ID226 rejected most broad
  second-stage gains: ID228 is `0.482245` full-OOF, but outer calibration
  selection scored around `0.4950`, worse than ID226's `0.491771`. Q2 was the
  only target with positive outer support. ID231 promotes only Q2
  `temp_t0.85_s0`: local `0.491287`, robustness `0.524343`, and max test delta
  vs ID226 `0.036313`. ID231 is safer but lacks Public `0.55` score margin.
- External ETRI prior reconstruction was tested as diagnostics only. Approximate
  external prevalence differed sharply from current train labels for S1/S3/S4,
  and constant prior shrink selected `base` for ID221/224/226/228/231. No
  external-prior submission was promoted.
- ID231 anchor/cross-target meta-model diagnostics also selected `base` for
  every target (`0.491287` -> `0.491287`). Current target-structure meta
  features do not provide a stable next improvement.
- A narrower partial second-stage ladder was added after rejecting direct
  external-notebook transfer:
  - ID232:
    `submission_id_232_id226_q2temp_s4iso_partial_frontier_cv0.486685_unsubmitted.csv`.
    It keeps the outer-supported Q2 temperature repair from ID231 and changes
    only S4 via isotonic calibration. Local `0.486685`, robustness `0.516793`,
    and nominal ID178-gap estimate about `0.549458`. This is now the least
    reckless new Public-0.55-margin card, but still uses full-OOF S4
    calibration and is not public proof.
  - ID233:
    `submission_id_233_id226_q2temp_s4q40_partial_frontier_cv0.484413_unsubmitted.csv`.
    It keeps Q2 temp and changes only S4 via q40 calibration. Local `0.484413`,
    robustness `0.513300`, nominal ID178-gap estimate about `0.547187`. It is
    sharper than ID232 but has materially larger S4 test movement.
  Supporting review:
  `reports/post_id233_partial_s4_calibration_robustness_20260616.json` and
  `reports/candidate_tier_manifest_post_id233_20260616.json`. DACON submission
  still requires explicit approval.
- The partial ladder was extended with small non-S4 temperature transforms
  instead of broad non-S4 isotonic calibration:
  - ID234:
    `submission_id_234_id226_tempgrid_s4iso_frontier_cv0.484527_unsubmitted.csv`.
    Q1/Q2/Q3/S1/S2 use small temperature shifts, S3 stays base, and S4 uses
    isotonic. Local `0.484527`, robustness `0.515224`.
  - ID235:
    `submission_id_235_id226_tempgrid_s4q40_frontier_cv0.482255_unsubmitted.csv`.
    Same non-S4 temp grid, S4 q40. Local `0.482255`, robustness `0.511518`.
    This now looks like the best balanced score-margin card: nearly ID228 local
    strength with better robustness and less broad non-S4 calibration.
  - ID236:
    `submission_id_236_id226_tempgrid_s4q80_frontier_cv0.474985_unsubmitted.csv`.
    Same non-S4 temp grid, S4 q80. Local `0.474985`, robustness `0.505914`.
    Lower local, high S4 public-gap risk.
  - ID237:
    `submission_id_237_id226_tempgrid_s4q160_frontier_cv0.466691_unsubmitted.csv`.
    Same non-S4 temp grid, S4 q160. Local `0.466691`, robustness `0.497142`.
    It is not as low as ID227, but is a cleaner low-local alternative because
    non-S4 targets use small temperature transforms rather than isotonic maps.
  Supporting review:
  `reports/post_id237_tempgrid_s4_frontier_robustness_20260616.json` and
  `reports/candidate_tier_manifest_post_id237_20260616.json`.
- A target-wise clean-anchor/ultra-calibration logit oracle was added:
  - ID238:
    `submission_id_238_id227_clean_anchor_logit_oracle_frontier_cv0.462365_unsubmitted.csv`.
    It uses the diagnostic
    `reports/id221_237_narrow_pair_prob_logit_oracle_scan_20260616.json` to
    logit-blend clean ID221 anchors with ID227 ultra calibration target-wise,
    keeps S3 from ID226, and blends S4 between ID227 and ID228. Local
    `0.462365`, robustness `0.493658`, and nominal ID178-gap estimate about
    `0.525138`. This is the current local frontier, but it is also a
    high-risk full-OOF oracle recombination, not validation proof. ID235 remains
    the more balanced next public probe; ID238 is the aggressive local-frontier
    card.
  Supporting review:
  `reports/post_id238_clean_anchor_logit_oracle_robustness_20260616.json` and
  `reports/candidate_tier_manifest_post_id238_20260616.json`.
- ID238 was then used as a recalibration stress-test base:
  - ID239:
    `submission_id_239_id238_iso_recal_frontier_cv0.437967_unsubmitted.csv`.
    Isotonic recalibration on Q1/Q2/Q3/S1/S2/S4, S3 base. Local `0.437967`.
  - ID240:
    `submission_id_240_id238_q80_s4iso_recal_frontier_cv0.417349_unsubmitted.csv`.
    q80 on Q1/Q2/Q3/S1/S2, S3 base, S4 isotonic. Local `0.417349`.
  - ID241:
    `submission_id_241_id238_q160_s4iso_recal_frontier_cv0.383459_unsubmitted.csv`.
    q160 on Q1/Q2/Q3/S1/S2, S3 base, S4 isotonic. Local `0.383459`.
  - ID242:
    `submission_id_242_id238_q240_s4iso_recal_frontier_cv0.357651_unsubmitted.csv`.
    q240 on Q1/Q2/Q3/S1/S2, S3 base, S4 isotonic. This is the current raw
    local minimum, but is an extreme q240-after-oracle calibration stress test
    with severe public-overfit risk.
  - ID243:
    `submission_id_243_id238_q240_all_recal_frontier_cv0.360420_unsubmitted.csv`.
    q240 on Q1/Q2/Q3/S1/S2/S4, S3 base. Slightly worse than ID242 but less
    S4-isotonic extremeness.
  Supporting review:
  `reports/post_id243_id238_recalibration_frontier_robustness_20260616.json`
  and `reports/candidate_tier_manifest_post_id243_20260616.json`. These
  artifacts are useful as local frontier stress tests, not as public proof.

This plan is the working strategy after the 2026-06-14 leaderboard probe where
`submission_id_085_score0.591466.csv` scored Public `0.6166556393`, and the
2026-06-16 probe where the lower local-CV random-conv branch
`submission_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638_unsubmitted.csv`
scored Public `0.6014113009`.

## Current position

- Best confirmed Public in the local log after updating submitted rows:
  `0.6014113009` from ID178. This is better than ID085's `0.6166556393`, but
  still far from the Public `0.55` target.
- Recent submitted probes:
  - ID085: local CV `0.591466` -> Public `0.6166556393`, gap `+0.025190`.
  - ID178: local public-structure CV `0.538638` -> Public `0.6014113009`, gap
    `+0.062774`.
- Operational implication: local CV is not merely optimistic; the more
  aggressive high-variance public-structure branch became **more** optimistic.
  A candidate that only improves local CV by a few thousandths is not enough.
  Public `0.55` requires validation repair, conservative anchoring, and a
  genuinely new stable signal source, not just another high-variance stacker.
- Current best non-calibration local artifact: `id=167`
  (`submission_id_167_s12_shallow_tree_rule_overlay_cv0.562280_unsubmitted.csv`).
  It extends the strict-era local ladder from ID142 (`0.576995`) through the
  feature-similarity, threshold-rule, and shallow tree-rule families. The latest
  target losses are Q1 `0.578471`, Q2 `0.610795`, Q3 `0.612616`, S1
  `0.489164`, S2 `0.538720`, S3 `0.491428`, and S4 `0.614764`. This is a strong
  non-calibration probe, but the most recent feature-similarity/threshold/tree
  choices are locally tuned from full-OOF diagnostics rather than untouched
  independent nested proof.
- Latest post-ID167 public-structure branch:
  - `id=168`
    (`submission_id_168_id167_public_multi_anchor_probe_cv0.549010_unsubmitted.csv`)
    applies fixed target-wise same-subject temporal/KNN anchors on top of ID167
    under interleaved within-subject blocked CV. It is the cleanest current
    sub-`0.55` public-structure artifact that does **not** depend on the ID136
    full-OOF extreme calibration ladder.
  - `id=169`
    (`submission_id_169_id168_fixed_public_calibration_probe_cv0.545627_unsubmitted.csv`)
    applies fixed target-wise calibration on top of ID168. It improves the same
    public-structure lens to `0.545627`, but calibration choices came from the
    ID168 diagnostic and therefore remain leaderboard-mapping evidence, not
    strict Subject-Hole proof.
  - `id=170`
    (`submission_id_170_id168_outer_supported_partial_calibration_probe_cv0.546579_unsubmitted.csv`)
    keeps only the Q2/Q3/S1 calibration targets after an outer-fold diagnostic
    showed full target-wise selection does not survive. It is slightly worse
    than ID169 but is the more conservative post-ID167 calibration rung.
  - `id=171`
    (`submission_id_171_id163_q40_extreme_public_probe_cv0.493975_unsubmitted.csv`)
    applies full-OOF q40 quantile calibration on top of the already high-risk
    ID163 public-structure probe. This is the lowest generated local
    public-structure score so far, but it is explicitly non-nested and should be
    treated as an extreme leaderboard-response probe, not as proof that robust
    local `0.50` quality has been achieved.
  - `id=172`
    (`submission_id_172_id163_targetwise_extreme_public_probe_cv0.492762_unsubmitted.csv`)
    tightens ID171 by using q40 for Q1/Q2/Q3/S1/S2 and isotonic for S3/S4. This
    is now the lowest generated local public-structure score, with the same
    extreme non-nested caveat.
  - `id=173`
    (`submission_id_173_id169_s4_hourly_sequence_overlay_cv0.545253_unsubmitted.csv`)
    is the cleaner post-ID167 branch plus a small S4-only raw hourly-sequence
    overlay. It is much less dramatic than ID172, but it is the latest artifact
    where a raw sensor sequence signal changes a calibrated public-structure
    base.
  - `id=174`
    (`submission_id_174_id173_multiseed_random_conv_sequence_overlay_cv0.543431_unsubmitted.csv`)
    extends ID173 with deterministic ROCKET-style random-convolution features
    over 34 value channels plus 34 presence channels from raw daily sensor
    sequences. This is the strongest non-extreme-calibration post-ID167 branch
    so far, although target/seed/model/weight choices are selected from local
    public-structure diagnostics rather than nested proof.
  - `id=175`
    (`submission_id_175_id173_multiseed_random_conv_sequence_overlay_v2_cv0.543336_unsubmitted.csv`)
    adds two more random-conv seed diagnostics and replaces only the S1 residual
    model versus ID174. The improvement is tiny, but ID175 is now the strongest
    non-extreme-calibration post-ID167 branch.
  - `id=176`
    (`submission_id_176_id173_multiseed_random_conv_sequence_overlay_v3_cv0.542270_unsubmitted.csv`)
    extends the random-conv seed pool through seed `43025` and updates Q2/Q3/S2/S4
    residual choices versus ID175.
  - `id=177`
    (`submission_id_177_id173_multiseed_random_conv_sequence_overlay_v4_cv0.540607_unsubmitted.csv`)
    extends the random-conv sweep through 27 diagnostics, including Kaggle-run
    192-kernel seeds `43026`–`43037` and 384-kernel seeds `43101`–`43106`.
    It updates Q1/Q3/S1/S4 versus ID176. The caveat remains important:
    target/seed/kernel-count/model/weight choices are selected from
    public-structure diagnostics, not nested Subject-Hole evidence.
  - `id=178`
    (`submission_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638_unsubmitted.csv`)
    fine-scans the same random-conv residual branch with denser blend weights
    and high-C logistic residual models. It is now the lowest
    non-extreme-calibration post-ID167 branch, but it is also more aggressive
    than ID177 because the raw residual models are weak alone and only help as
    small high-variance corrections to the ID173 base. This file was submitted
    to DACON as submission `1465318` on `2026-06-16 11:03:11` and scored Public
    `0.6014113009`, producing a severe `+0.062774` local-to-public gap. Treat
    the high-C random-conv branch as a useful diagnostic family, not as a
    trusted leaderboard template without validation repair.
  - `id=179`
    (`submission_id_179_id178_tempered_id177_random_conv_anchor_blend_cv0.538290_unsubmitted.csv`)
    tempers ID178 against the lower-variance ID177 random-conv branch after the
    ID178 public-gap failure. Target-wise primary weights are Q1 `0.90`, Q2
    `1.00`, Q3 `0.70`, S1 `0.55`, S2 `0.90`, S3 `1.00`, and S4 `0.85`.
    This improves the same public-structure local CV from ID178's `0.538638` to
    `0.538290` while reducing high-C residual exposure on Q1/Q3/S1/S2/S4. It is
    still non-strict and OOF-selected, so it should be treated as a
    public-gap-aware tempering probe, not as proof of Public `0.55` quality.
  - `id=180`
    (`submission_id_180_id179_tempered_id172_extreme_anchor_probe_cv0.529579_unsubmitted.csv`)
    uses ID179 as the clean/public-gap-aware anchor and blends in only `11%` of
    the ID172 target-wise extreme calibration artifact. This is the first
    post-ID178 artifact to cross the nominal local `0.53` line, with target
    losses Q1 `0.541125`, Q2 `0.545164`, Q3 `0.556696`, S1 `0.462898`, S2
    `0.519118`, S3 `0.487215`, and S4 `0.594836`. It is deliberately much less
    exposed than pure ID172, but it still inherits non-nested full-OOF
    calibration risk; treat it as a bold leaderboard-mapping counterprobe, not
    as validation proof.
  - `id=181`
    (`submission_id_181_id179_targetwise_delta06_id172_extreme_probe_cv0.526430_unsubmitted.csv`)
    keeps the same clean anchor ID179 and extreme source ID172, but allocates
    the extreme weight target-wise under a maximum test-prediction movement of
    about `0.06` versus ID179. The chosen extreme weights are Q1 `0.111`, Q2
    `0.149`, Q3 `0.165`, S1 `0.177`, S2 `0.115`, S3 `0.320`, and S4 `0.159`,
    with the selection scan saved as
    `reports/id179_id172_targetwise_extreme_delta_cap_scan_20260616.json`. This
    improves the public-structure local score to `0.526430`, with target losses
    Q1 `0.541034`, Q2 `0.541382`, Q3 `0.552206`, S1 `0.459187`, S2 `0.518680`,
    S3 `0.480851`, and S4 `0.591671`. It is still a non-nested extreme-blend
    counterprobe, but it uses the same approximate max-drift budget as ID180
    more efficiently.
  - `id=182`
    (`submission_id_182_id179_targetwise_delta08_id172_extreme_probe_cv0.522909_unsubmitted.csv`)
    extends the same target-wise ID179/ID172 bridge with a maximum test
    prediction movement of about `0.08` versus ID179. Extreme weights are Q1
    `0.149`, Q2 `0.199`, Q3 `0.220`, S1 `0.236`, S2 `0.153`, S3 `0.427`, and
    S4 `0.213`. Local public-structure score improves to `0.522909`. This is a
    higher-risk rung than ID181, saved to preserve the calibrated-risk ladder
    rather than as strict validation evidence.
  - `id=183`
    (`submission_id_183_id179_targetwise_delta10_id172_extreme_probe_cv0.519611_unsubmitted.csv`)
    pushes the target-wise ID179/ID172 bridge to a maximum test movement of
    about `0.10`. Extreme weights are Q1 `0.186`, Q2 `0.249`, Q3 `0.275`, S1
    `0.296`, S2 `0.191`, S3 `0.534`, and S4 `0.266`. Local public-structure
    score improves to `0.519611`, with the extended cap scan saved as
    `reports/id179_id172_targetwise_extreme_delta_cap_extended_scan_20260616.json`.
    This is the current strongest tempered local-score rung after ID178's public
    result, but it is still non-nested and should not be represented as a safe
    model.
  - `id=184`
    (`submission_id_184_id179_targetwise_delta12_id172_extreme_probe_cv0.516499_unsubmitted.csv`)
    extends the same target-wise bridge to a maximum test movement of about
    `0.12` versus ID179. Extreme weights are Q1 `0.223`, Q2 `0.299`, Q3
    `0.331`, S1 `0.355`, S2 `0.230`, S3 `0.641`, and S4 `0.319`. Local
    public-structure score improves to `0.516499`. This rung is materially more
    exposed to the ID172 full-OOF calibration source than ID181/ID182.
  - `id=185`
    (`submission_id_185_id179_targetwise_delta15_id172_extreme_probe_cv0.512540_unsubmitted.csv`)
    pushes the target-wise bridge to about `0.15` max movement versus ID179.
    Extreme weights are Q1 `0.279`, Q2 `0.373`, Q3 `0.413`, S1 `0.444`, S2
    `0.287`, S3 `0.700`, and S4 `0.399`. Local public-structure score improves
    to `0.512540`. This is the current lowest tempered post-ID178 local artifact
    below the pure ID172 extreme probe, but it should be treated as a high-risk
    leaderboard counterprobe only.
  - `id=186`
    (`submission_id_186_id179_targetwise_delta20_id172_extreme_probe_cv0.506162_unsubmitted.csv`)
    extends the target-wise bridge to about `0.20` max movement versus ID179.
    Extreme weights are Q1 `0.372`, Q2 `0.498`, Q3 `0.551`, S1 `0.592`, S2
    `0.383`, S3 `1.000`, and S4 `0.532`, producing local public-structure
    `0.506162`. This is now very exposed to ID172, including a fully extreme S3
    column.
  - `id=187`
    (`submission_id_187_id179_targetwise_delta25_id172_extreme_probe_cv0.501633_unsubmitted.csv`)
    pushes the target-wise bridge to about `0.25` max movement versus ID179.
    Extreme weights are Q1 `0.465`, Q2 `0.623`, Q3 `0.689`, S1 `0.740`, S2
    `0.479`, S3 `1.000`, and S4 `0.666`, producing local public-structure
    `0.501633`. This is the lowest generated tempered bridge short of going
    directly to ID172, but it is a high-risk counterprobe rather than a safe
    candidate.
  - `id=188`
    (`submission_id_188_id179_targetwise_delta30_id172_extreme_probe_cv0.498094_unsubmitted.csv`)
    pushes the target-wise bridge to about `0.30` max movement versus ID179 and
    is the first tempered bridge artifact to cross the local `0.50` line.
    Extreme weights are Q1 `0.558`, Q2 `0.747`, Q3 `0.827`, S1 `0.888`, S2
    `0.575`, S3 `1.000`, and S4 `0.799`, producing local public-structure
    `0.498094`. This is close to the pure ID172 extreme-calibration regime and
    should only be used as a high-risk leaderboard counterprobe.
  - `id=189`
    (`submission_id_189_id172_s4_tempered_id179_extreme_frontier_probe_cv0.492595_unsubmitted.csv`)
    is the current local-score frontier artifact. It uses ID172 for
    Q1/Q2/Q3/S1/S2/S3 and keeps only `12.4%` of the cleaner ID179 anchor on S4,
    producing public-structure local `0.492595` with target losses Q1
    `0.497013`, Q2 `0.491685`, Q3 `0.516014`, S1 `0.433412`, S2 `0.477167`,
    S3 `0.465115`, and S4 `0.567760`. The frontier scan is saved as
    `reports/id179_id172_targetwise_extreme_delta_cap_frontier_scan_20260616.json`.
    This is essentially the ID172 full-OOF calibration regime with a small S4
    repair, not a public-safe candidate.
  - `id=190`
    (`submission_id_190_id173_remote_s1_random_conv_refresh_overlay_cv0.537956_unsubmitted.csv`)
    incorporates the completed Kaggle remote random-conv sweep. The only clearly
    useful new cleaner-branch signal was S1 from `n_kernels=512`, seed `43203`,
    model `lr_c0_1`, weight `0.20`, lowering S1 public-structure loss to
    `0.466858`. The full artifact improves the ID178-like random-conv branch to
    local `0.537956`, but it remains public-structure/non-strict and still sits
    in the family that produced the ID178 Public gap.
  - `id=191`
    (`submission_id_191_id190_tempered_id177_remote_s1_anchor_blend_cv0.538009_unsubmitted.csv`)
    tempers ID190 against the lower-variance ID177 branch using the same
    target-wise weights as ID179. It is slightly worse than ID190 locally but
    better than ID179 (`0.538009` vs `0.538290`), and is the current
    post-Kaggle cleaner anchor-like reference.
  - `id=192`
    (`submission_id_192_id191_targetwise_delta06_id172_extreme_probe_cv0.526261_unsubmitted.csv`)
    rebuilds the moderate ID172 bridge from the ID191 anchor under the same
    approximate `0.06` max test-movement budget used by ID181. It improves the
    moderate bridge from `0.526430` to `0.526261`, mainly through the refreshed
    S1 anchor. It is still non-nested and still inherits ID172 calibration risk.
  - `id=193`
    (`submission_id_193_id191_sequence_summary_embedding_overlay_cv0.536429_unsubmitted.csv`)
    adds a new deterministic sequence-summary representation on top of ID191.
    It extracts window means, slopes, difference statistics, autocorrelation,
    and coarse FFT energy from the same 44-bin daily raw-sensor tensor, then
    lightly blends logistic summary models into Q1/Q2/Q3 only. This improves the
    cleaner/non-ID172 branch from `0.538009` to `0.536429`, with target losses
    Q1 `0.549258`, Q2 `0.550689`, and Q3 `0.564204`. This is the current lowest
    cleaner branch, but remains public-structure/non-strict.
  - `id=194`
    (`submission_id_194_id193_targetwise_delta06_id172_extreme_probe_cv0.525179_unsubmitted.csv`)
    rebuilds the moderate ID172 bridge from the new ID193 cleaner anchor under
    the same approximate `0.06` max movement budget. It improves ID192 from
    `0.526261` to `0.525179`, mostly by carrying the Q1/Q2/Q3 sequence-summary
    gains into the bridge. This supersedes ID192 locally, but remains
    non-nested and inherits ID172 calibration risk.
  - `id=195`
    (`submission_id_195_id193_targetwise_delta08_id172_extreme_probe_cv0.521836_unsubmitted.csv`)
    extends the ID193/ID172 target-wise bridge to the approximate `0.08` max
    movement budget. Extreme weights are Q1 `0.138`, Q2 `0.191`, Q3 `0.209`,
    S1 `0.227`, S2 `0.153`, S3 `0.427`, and S4 `0.213`, producing local
    public-structure `0.521836`. This is a riskier moderate rung than ID194,
    not independent Public `0.55` evidence.
  - `id=196`
    (`submission_id_196_id193_targetwise_delta10_id172_extreme_probe_cv0.518710_unsubmitted.csv`)
    extends the same bridge to the approximate `0.10` max movement budget.
    Extreme weights are Q1 `0.172`, Q2 `0.238`, Q3 `0.262`, S1 `0.284`, S2
    `0.191`, S3 `0.534`, and S4 `0.266`, producing local public-structure
    `0.518710`. Treat this as a bold counterprobe because the gain mainly comes
    from additional ID172 non-nested calibration exposure.
  - `id=197`
    (`submission_id_197_id193_targetwise_delta12_id172_extreme_probe_cv0.515748_unsubmitted.csv`)
    extends the ID193/ID172 bridge to the approximate `0.12` max movement
    budget. Extreme weights are Q1 `0.207`, Q2 `0.286`, Q3 `0.314`, S1
    `0.341`, S2 `0.230`, S3 `0.641`, and S4 `0.319`, producing local
    public-structure `0.515748`. This is the lowest post-ID193 bridge score so
    far, but it should be tiered with the bold/high-risk probes rather than
    treated as a safe submission candidate.
  - `id=198`
    (`submission_id_198_id193_s3_smartphone_light_compact_overlay_cv0.536380_unsubmitted.csv`)
    tests the external-notebook `mScreenStatus` hint as a compact
    smartphone/light side model on top of ID193. The diagnostic selected only S3:
    `0.05` blend weight from a small HGB model over 124 phone/light/usage/charge
    features. It improves the cleaner branch from `0.536429` to `0.536380`,
    with S3 moving from `0.490893` to `0.490553`. This is a real but tiny
    low-variance signal, not a leaderboard breakthrough.
  - `id=199`
    (`submission_id_199_id198_targetwise_delta06_id172_extreme_probe_cv0.525173_unsubmitted.csv`)
    carries the ID198 S3 improvement into the moderate ID172 bridge using the
    same target-wise clean weights as ID194. It slightly improves ID194 from
    `0.525179` to `0.525173`, with S3 moving from `0.480851` to `0.480803`.
    This now slightly supersedes ID194 as the preferred moderate-low bridge, but
    it remains non-nested and inherits ID172 calibration risk.
  - `id=200`
    (`submission_id_200_id198_sax_paa_sequence_overlay_cv0.536190_unsubmitted.csv`)
    adds a deterministic SAX/PAA sequence-pattern representation on top of
    ID198. It converts the aligned 44-bin daily raw-sensor tensor into PAA-8
    segment mean/std/delta features plus SAX symbol, segment, and transition
    frequencies. A low-variance logistic side model is lightly blended into
    Q1 (`0.05`), Q3 (`0.03`), S1 (`0.01`), and S4 (`0.01`), improving the
    cleaner branch from `0.536380` to `0.536190`. Target losses are Q1
    `0.548258`, Q2 `0.550689`, Q3 `0.563914`, S1 `0.467737`, S2 `0.529522`,
    S3 `0.490553`, and S4 `0.602656`. This is the current best cleaner
    non-ID172 branch, but remains public-structure/non-strict.
  - `id=201`
    (`submission_id_201_id200_targetwise_delta06_id172_extreme_probe_cv0.524999_unsubmitted.csv`)
    carries ID200 into the same moderate `delta06` ID172 bridge weights used by
    ID194 and ID199. It improves the moderate-low bridge from ID199 `0.525173`
    to `0.524999`, with most of the additional gain on Q1/Q3/S4. This slightly
    supersedes ID199/ID194 as the preferred moderate-low counterprobe, but it
    remains non-nested and inherits ID172 calibration risk.
  - `id=202`
    (`submission_id_202_id200_targetwise_delta08_id172_extreme_probe_cv0.521667_unsubmitted.csv`)
    carries ID200 into the more aggressive `delta08` ID172 bridge weights used
    by ID195. It improves the comparable ID195 rung from `0.521836` to
    `0.521667`, with target losses Q1 `0.535616`, Q2 `0.532580`, Q3
    `0.546623`, S1 `0.455158`, S2 `0.515447`, S3 `0.477907`, and S4
    `0.588340`. This is a moderate counterprobe, not the preferred
    moderate-low candidate; ID201 remains cleaner because it moves less toward
    ID172.
  - `id=203`
    (`submission_id_203_id200_s4_recurrence_gaf_sequence_overlay_cv0.535980_unsubmitted.csv`)
    adds a recurrence/GAF-style sequence-image summary on top of ID200. It
    summarizes each 44-bin raw-sensor daily tensor with recurrence rates,
    lagged recurrence, Gramian Angular Field aggregates, and compact
    cross-channel correlation features. The diagnostic selected only S4 with a
    `0.10` blend from a low-C logistic side model, improving the cleaner branch
    from `0.536190` to `0.535980` and S4 from `0.602656` to `0.601188`. This
    is now the current best cleaner non-ID172 branch, still public-structure
    and non-strict.
  - `id=204`
    (`submission_id_204_id203_targetwise_delta06_id172_extreme_probe_cv0.524761_unsubmitted.csv`)
    carries ID203 into the same `delta06` ID172 bridge weights used by ID201.
    It improves the preferred moderate-low bridge from `0.524999` to
    `0.524761`, almost entirely through the S4 recurrence/GAF gain. This now
    supersedes ID201 as the preferred moderate-low counterprobe, while keeping
    the same non-nested ID172 caveat.
  - `id=205`
    (`submission_id_205_id203_targetwise_delta08_id172_extreme_probe_cv0.521424_unsubmitted.csv`)
    carries ID203 into the more aggressive `delta08` bridge weights used by
    ID202. It improves ID202 from `0.521667` to `0.521424`, with S4 moving from
    `0.588340` to `0.586639`. This is a stronger but riskier moderate
    counterprobe, not the preferred moderate-low candidate.
  - `id=206`
    (`submission_id_206_id203_wavelet_spectral_sequence_overlay_cv0.535764_unsubmitted.csv`)
    adds deterministic wavelet/spectral sequence summaries on top of ID203:
    Haar-like multiscale contrasts, lagged roughness, FFT coefficient/band
    shares, and window-level volatility over the aligned 44-bin raw-sensor
    tensor. The diagnostic selected light logistic blends for Q3 (`0.05`), S2
    (`0.005`), and S4 (`0.05`), improving the cleaner branch from `0.535980` to
    `0.535764`. Target losses are Q1 `0.548258`, Q2 `0.550689`, Q3 `0.562877`,
    S1 `0.467737`, S2 `0.529516`, S3 `0.490553`, and S4 `0.600718`. This is now
    the current best cleaner non-ID172 branch, but remains public-structure and
    non-strict.
  - `id=207`
    (`submission_id_207_id206_targetwise_delta06_id172_extreme_probe_cv0.524518_unsubmitted.csv`)
    carries ID206 into the same `delta06` ID172 bridge weights used by ID204. It
    improves the preferred moderate-low bridge from `0.524761` to `0.524518`.
    This now supersedes ID204 as the preferred moderate-low counterprobe, while
    keeping the same non-nested ID172 caveat.
  - `id=208`
    (`submission_id_208_id206_targetwise_delta08_id172_extreme_probe_cv0.521177_unsubmitted.csv`)
    carries ID206 into the more aggressive `delta08` bridge weights used by
    ID205. It improves ID205 from `0.521424` to `0.521177`, with S4 moving from
    `0.586639` to `0.585842`. This is a stronger but riskier moderate
    counterprobe, not the preferred moderate-low candidate.
  - `id=209`
    (`submission_id_209_id206_shapelet_motif_sequence_overlay_cv0.534530_unsubmitted.csv`)
    adds a deterministic shapelet/motif distance representation on top of ID206.
    It samples a label-free motif bank from high-variance 44-bin sensor channels
    and transforms each day by z-normalized subsequence distance, motif position,
    and late-vs-early distance contrast features. The diagnostic selected Q1
    (`0.03` blend from `lr_c0_1`) and S4 (`0.20` blend from `lr_c0_01`),
    improving the cleaner branch from `0.535764` to `0.534530`. S4 moves from
    `0.600718` to `0.592263`, making this the strongest clean non-ID172 repair
    in the recent sequence family. It remains public-structure/non-strict, not
    public proof.
  - `id=210`
    (`submission_id_210_id209_targetwise_delta06_id172_extreme_probe_cv0.523400_unsubmitted.csv`)
    carries ID209 into the same `delta06` ID172 bridge weights used by ID207. It
    improves the preferred moderate-low bridge from `0.524518` to `0.523400`.
    This now supersedes ID207 as the preferred moderate-low counterprobe, while
    keeping the same non-nested ID172 caveat.
  - `id=211`
    (`submission_id_211_id209_targetwise_delta08_id172_extreme_probe_cv0.520106_unsubmitted.csv`)
    carries ID209 into the more aggressive `delta08` bridge weights used by
    ID208. It improves ID208 from `0.521177` to `0.520106`, with S4 moving from
    `0.585842` to `0.578428`. This is a stronger but riskier moderate
    counterprobe, not the preferred moderate-low candidate.
  - `id=212`
    (`submission_id_212_id206_shapelet_motif_sequence_wide_weight_overlay_cv0.534469_unsubmitted.csv`)
    widens the S4 shapelet/motif blend from ID209 after a weight scan showed S4
    was still improving at the previous `0.20` cap. It keeps Q1 at `0.03` with
    `lr_c0_1` and moves S4 to `0.25` with `lr_c0_01`, improving the cleaner
    branch from ID209 `0.534530` to `0.534469` and S4 from `0.592263` to
    `0.591833`. This is now the current best cleaner non-ID172 branch, but it is
    a small public-structure refinement rather than public proof.
  - `id=213`
    (`submission_id_213_id212_targetwise_delta06_id172_extreme_probe_cv0.523287_unsubmitted.csv`)
    carries ID212 into the same `delta06` ID172 bridge weights used by ID210. It
    improves the preferred moderate-low bridge from `0.523400` to `0.523287`.
    This now supersedes ID210 as the preferred moderate-low counterprobe, while
    keeping the same non-nested ID172 caveat.
  - `id=214`
    (`submission_id_214_id212_targetwise_delta08_id172_extreme_probe_cv0.519982_unsubmitted.csv`)
    carries ID212 into the more aggressive `delta08` bridge weights used by
    ID211. It improves ID211 from `0.520106` to `0.519982`, with S4 moving from
    `0.578428` to `0.577561`. This is the first local public-structure artifact
    below `0.52` in the current ladder, but it is still a riskier non-nested
    ID172 bridge counterprobe, not a confirmed public result.
  - `id=215`
    (`submission_id_215_id212_shapelet_motif_alltop40_sequence_overlay_cv0.534135_unsubmitted.csv`)
    expands the shapelet/motif family to the top 40 sequence channels and
    modifies Q1, S1, and S4. It improves the clean branch from ID212 `0.534469`
    to `0.534135`; the split-robustness score also improves from `0.566994` to
    `0.565863`. This is the current best clean/non-ID172 public-structure
    branch, but it is still not strict Subject-Hole proof.
  - `id=216`
    (`submission_id_216_id215_targetwise_delta06_id172_extreme_probe_cv0.522875_unsubmitted.csv`)
    carries ID215 into the same `delta06` ID172 bridge weights used by ID213.
    It improves the preferred moderate-low bridge from ID213 `0.523287` to
    `0.522875`, with split-robustness improving from `0.555222` to `0.554029`.
    This supersedes ID213 as the preferred moderate-low counterprobe, while
    retaining the non-nested ID172 caveat.
  - `id=217`
    (`submission_id_217_id215_targetwise_delta08_id172_extreme_probe_cv0.519556_unsubmitted.csv`)
    carries ID215 into the more aggressive `delta08` ID172 bridge weights used
    by ID214. It improves ID214 from `0.519982` to `0.519556`, with
    split-robustness improving from `0.551765` to `0.550573`. This is the lowest
    non-extreme bridge score in the current ladder, but it is riskier than ID216
    and remains a non-nested ID172 bridge artifact.
  - `id=218`
    (`submission_id_218_id215_multi_signal_clean_overlay_cv0.534103_unsubmitted.csv`)
    combines several tiny clean side signals on ID215 instead of chaining them
    as separate submissions: random-conv for Q1/S4, window-pair for Q3, and
    smartphone/light for S3. It improves the clean branch from ID215 `0.534135`
    to `0.534103`, with split-robustness improving from `0.565863` to
    `0.565633`. This is the current best clean/non-ID172 branch, but the gain is
    very small and should not be mistaken for public evidence.
  - `id=219`
    (`submission_id_219_id218_targetwise_delta06_id172_extreme_probe_cv0.522874_unsubmitted.csv`)
    carries ID218 into the same `delta06` ID172 bridge weights used by ID216. It
    very slightly improves ID216 from `0.522875` to `0.522874`, with
    split-robustness improving from `0.554029` to `0.553879`. This is the
    preferred moderate-low counterprobe, while retaining the non-nested ID172
    caveat.
  - `id=220`
    (`submission_id_220_id218_targetwise_delta08_id172_extreme_probe_cv0.519561_unsubmitted.csv`)
    carries ID218 into the more aggressive `delta08` bridge weights used by
    ID217. It is fractionally worse than ID217 by raw local CV (`0.519561` vs
    `0.519556`) but fractionally better by split-robustness (`0.550442` vs
    `0.550573`). Treat ID217 as the raw-local delta08 frontier and ID220 as the
    robustness-score delta08 reference; both remain high-risk ID172 bridge
    artifacts.
- Separate non-strict public-structure probes:
  - Best fixed-anchor probe: `id=122`
    (`submission_id_122_q123s1234_mixed_temporal_anchor_public_probe_cv0.560861_unsubmitted.csv`).
    It applies fixed same-subject nearest/KNN temporal-anchor blends to all seven
    targets under interleaved within-subject blocked CV.
  - Lowest public-structure probe score: `id=126`
    (`submission_id_126_extreme_isotonic_quantile_calibration_public_probe_cv0.529691_unsubmitted.csv`).
    It keeps the high-risk ID125 calibrated base and applies full-OOF fitted
    isotonic/quantile calibration to all seven targets. This is an
    **extreme-risk**, non-nested leaderboard calibration probe. The nested
    calibration reference is `0.562377`, which is much less attractive than the
    full-OOF score and slightly worse than the cleaner ID122 fixed-anchor probe.
    Do not treat ID126 as a proven strict-local improvement or a confirmed
    Public `0.55` candidate without a real DACON result.
  - Current tempered sub-`0.55` probe candidate: `id=127`
    (`submission_id_127_tempered_extreme_calibration_public_probe_cv0.549454_unsubmitted.csv`).
    It uses a fixed `0.80 * ID125 + 0.20 * ID126` blend. This is still
    non-strict and not fold-nested, but it is deliberately less aggressive than
    ID126 while crossing the nominal public-structure `0.55` line.
  - Cleaner-anchor tempered sub-`0.55` counterprobe: `id=128`
    (`submission_id_128_clean_anchor_tempered_extreme_calibration_public_probe_cv0.549956_unsubmitted.csv`).
    It uses `0.75 * ID122 + 0.25 * ID126`, anchoring the blend on the cleaner
    fixed temporal-anchor artifact rather than the high-risk ID125 calibration.
    This makes it a useful comparison probe if ID127 is submitted.
  - Latest GPS-base public-structure counterprobe: `id=131`
    (`submission_id_131_gpsbase_tempered_extreme_calibration_public_probe_cv0.549876_unsubmitted.csv`).
    It first updates the cleaner mixed-anchor probe to the strict ID129 GPS base
    as ID130 (`0.560757`), then applies `0.75 * ID130 + 0.25 * ID126`. This is
    slightly better than ID128 while keeping the cleaner-anchor interpretation.
  - New lower public-structure probe: `id=133`
    (`submission_id_133_refined_anchor_tempered_extreme_public_probe_cv0.545875_unsubmitted.csv`).
    It first refines the clean ID130 anchor as ID132 (`0.560259`) with low-weight
    KNN temporal-anchor updates on Q2/Q3/S2/S4, then uses
    `0.65 * ID132 + 0.35 * ID126`. This gives more leaderboard-probe buffer than
    ID127/ID131, but is also more exposed to ID126's non-nested calibration risk.
  - Lowest generated public-structure probe: `id=136`
    (`submission_id_136_extreme_calibration_on_balanced_refined_anchor_public_probe_cv0.523144_unsubmitted.csv`).
    It applies full-OOF isotonic/quantile calibration on top of the balanced
    refined-anchor blend ID134. This satisfies the nominal local `0.53` line,
    but it is the highest-risk current probe and must not be treated as a
    verified leaderboard improvement without a real DACON score.
  - Current lowest high-risk public-structure probe: `id=163`
    (`submission_id_163_id162_fine_targetwise_tempered_id136_extreme_public_probe_cv0.521143_unsubmitted.csv`).
    It remains the local public-structure leader, but it is dominated by the
    ID136 extreme-calibration ladder and should be treated as the boldest
    leaderboard-mapping candidate rather than as validation evidence.
  - Current lowest extreme public-structure file: `id=171`
    (`submission_id_171_id163_q40_extreme_public_probe_cv0.493975_unsubmitted.csv`),
    now superseded locally by `id=172`
    (`submission_id_172_id163_targetwise_extreme_public_probe_cv0.492762_unsubmitted.csv`).
    These cross any nominal local target, but their calibration is fitted and
    scored on the same OOF rows.
  - A strict-hole-base public anchor rerun, `id=138`
    (`submission_id_138_q123s1234_mixed_temporal_anchor_holebase_public_probe_cv0.562347_unsubmitted.csv`),
    did **not** beat the cleaner ID132 public-structure anchor (`0.560259`), so
    it is a documented counterexample rather than a current public-probe leader.
- Current productive paths are now split: keep strict target-specific feature
  work alive, especially the newly useful same-subject temporal-hole anchor
  family, but prioritize validation/public-gap repair before further
  leaderboard-facing promotion. ID178 showed that chasing lower public-structure
  CV alone can move in the wrong direction on the actual leaderboard; ID179 is
  the first small corrective step that explicitly incorporates that lesson, and
  ID180-ID214 form the current tempered bridge/frontier ladder between that
  safer anchor and the extreme local-score ladder.

The public-gap/risk maps for the post-ID178 candidate ladder are:

- `reports/public_gap_risk_map_post_id185_20260616.json`
- `reports/public_gap_risk_map_post_id187_20260616.json`
- `reports/public_gap_risk_map_post_id188_20260616.json`
- `reports/public_gap_adjusted_candidate_ranking_post_id188_20260616.json`
- `reports/id179_id188_oof_stability_concentration_diagnostic_20260616.json`
- `reports/candidate_tier_manifest_post_id188_20260616.json`
- `reports/candidate_tier_manifest_post_id189_20260616.json`
- `reports/remote_random_conv_sweep_review_post_id192_20260616.json`
- `reports/id191_id172_targetwise_extreme_delta_cap_scan_20260616.json`
- `reports/candidate_tier_manifest_post_id192_20260616.json`
- `reports/id191_sequence_summary_embedding_diagnostic.json`
- `reports/id193_id172_targetwise_extreme_delta_cap_scan_20260616.json`
- `reports/sequence_summary_embedding_review_post_id194_20260616.json`
- `reports/candidate_tier_manifest_post_id194_20260616.json`
- `reports/id193_id172_targetwise_extreme_delta_cap_review_post_id197_20260616.json`
- `reports/candidate_tier_manifest_post_id197_20260616.json`
- `reports/id193_smartphone_light_compact_diagnostic_20260616.json`
- `reports/smartphone_light_compact_review_post_id199_20260616.json`
- `reports/candidate_tier_manifest_post_id199_20260616.json`
- `reports/id198_sax_paa_sequence_diagnostic_20260616.json`
- `reports/sax_paa_sequence_review_post_id201_20260616.json`
- `reports/candidate_tier_manifest_post_id201_20260616.json`
- `reports/id178_id201_split_robustness_diagnostic_20260616.json`
- `reports/id200_anchor_feature_meta_model_diagnostic_20260616.json`
- `reports/id200_outer_calibration_selection_diagnostic_20260616.json`
- `reports/post_id202_risk_and_bridge_review_20260616.json`
- `reports/candidate_tier_manifest_post_id202_20260616.json`
- `reports/id200_recurrence_gaf_sequence_diagnostic_20260616.json`
- `reports/recurrence_gaf_sequence_review_post_id205_20260616.json`
- `reports/candidate_tier_manifest_post_id205_20260616.json`
- `reports/id178_id205_split_robustness_diagnostic_20260616.json`
- `reports/id203_wavelet_spectral_sequence_diagnostic_20260616.json`
- `reports/wavelet_spectral_sequence_review_post_id206_20260616.json`
- `reports/candidate_tier_manifest_post_id206_20260616.json`
- `reports/id178_id206_split_robustness_diagnostic_20260616.json`
- `reports/post_id208_bridge_review_20260616.json`
- `reports/candidate_tier_manifest_post_id208_20260616.json`
- `reports/id178_id208_split_robustness_diagnostic_20260616.json`
- `reports/id206_shapelet_motif_sequence_diagnostic_20260616.json`
- `reports/shapelet_motif_sequence_review_post_id211_20260616.json`
- `reports/candidate_tier_manifest_post_id211_20260616.json`
- `reports/id178_id211_split_robustness_diagnostic_20260616.json`
- `reports/id206_shapelet_motif_sequence_wide_weight_diagnostic_20260616.json`
- `reports/shapelet_motif_wide_weight_review_post_id214_20260616.json`
- `reports/candidate_tier_manifest_post_id214_20260616.json`
- `reports/id178_id214_split_robustness_diagnostic_20260616.json`
- `reports/shapelet_motif_alltop40_review_post_id217_20260616.json`
- `reports/candidate_tier_manifest_post_id217_20260616.json`
- `reports/id178_id217_split_robustness_diagnostic_20260616.json`
- `reports/multi_signal_clean_review_post_id220_20260616.json`
- `reports/candidate_tier_manifest_post_id220_20260616.json`
- `reports/id178_id220_split_robustness_diagnostic_20260616.json`
- `reports/external_public05917_notebook_review_20260616.json`
- `reports/id179_id188_split_robustness_diagnostic_20260616.json`
- `reports/id179_id172_targetwise_extreme_delta_cap_frontier_scan_20260616.json`

These reports should be checked before any DACON submission because ID178's real
Public result proved that a much lower public-structure local CV can still map
poorly to the leaderboard.

The post-ID188 adjusted-risk ranking deliberately separates "lowest local" from
"most plausible probe." Under the current heuristic, ID179 remains the safest
post-ID178 anchor-like candidate, ID218 is the refreshed cleaner anchor-like
candidate, ID219 is the preferred moderate-low counterprobe, ID217/ID220/ID214/ID211/ID208/ID205/ID202/ID199/ID195/ID194/ID192/ID181/ID182
are moderate-delta counterprobes, ID196/ID197/ID183-ID185 are bold high-risk
rungs, and ID186-ID189 are extreme-delta
counterprobes whose local scores are increasingly dominated by the non-nested
ID172 calibration source. This ranking is not a public-score prediction; it is a
guardrail against repeating the ID178 local-to-public failure.

The OOF stability/concentration diagnostic adds a second lens: the ID180-ID188
bridge improvements are not concentrated in only one subject or one target under
the available OOF rows; all ten subjects and all seven targets have positive
mean OOF improvement versus ID179. That is encouraging, but it does **not**
remove the main caveat because those improvements still come from OOF-selected,
non-nested ID172 calibration exposure.

The candidate-tier manifests freeze the current operational interpretation:
ID218/ID215/ID212/ID209/ID206/ID203/ID200/ID198/ID193/ID191/ID179 are the anchor-like references, ID219/ID216/ID213/ID210/ID207/ID204/ID201/ID199/ID194 are the
preferred moderate-low bridges, ID217/ID220/ID214/ID211/ID208/ID205/ID202/ID195/ID192/ID181/ID182/ID180 are moderate counterprobes,
ID196/ID197/ID183-ID185 are bold counterprobes, and ID186-ID188 are
extreme-delta counterprobes only. ID189 is even closer to the ID172 full-OOF
frontier and should be considered a local frontier marker, not a preferred
submission. Use the post-ID220 tiering before any future DACON submission
discussion.

The post-ID202 risk review adds two negative results that should prevent
duplicated work: ID200 anchor-feature meta models selected `base` for all seven
targets, and ID200 outer calibration worsened the base under all guarded
narrow/wide variants (`0.539254` best guarded mean versus base `0.536190`).
This supports continuing with new stable representations or offloaded Kaggle
experiments rather than applying more calibration to ID200.

The post-ID206 work adds three more exhaustion notes on top of ID203: compact
smartphone/light rerun improved by only about one micro-point, stability-selected
raw-feature overlays selected `base` for all targets, and deterministic
sequence-summary rerun improved by only about one micro-point. The promoted new
signal was the wavelet/spectral sequence family, and even that was a small clean
improvement rather than a breakthrough.

The refreshed ID178-ID205 split robustness diagnostic keeps the same fragility
formula and shows the recurrence/GAF update moving in the right direction:
ID203 robust score `0.568318` versus ID200 `0.568848`, ID204 `0.556611` versus
ID201 `0.557148`, and ID205 `0.553144` versus ID202 `0.553676`. This is useful
supporting evidence that the S4 recurrence/GAF repair is not obviously
concentrated in a worse split, but it remains OOF evidence rather than a public
leaderboard guarantee.

The ID178-ID206 split robustness diagnostic adds the wavelet/spectral cleaner
update. Under that refreshed formula, ID206 moves the clean branch from ID203
overall `0.535980` to `0.535764` and robustness score from `0.568839` to
`0.568760`. This is small but directionally consistent clean-branch evidence,
not public leaderboard proof.

The post-ID208 bridge review carries the same ID172 delta weights onto ID206:
ID207 improves the moderate-low bridge from ID204 `0.524761` to `0.524518`, and
ID208 improves the stronger delta08 bridge from ID205 `0.521424` to `0.521177`.
The refreshed split-robustness lens also moves slightly in the right direction
(ID204 `0.557089` -> ID207 `0.556920`; ID205 `0.553610` -> ID208 `0.553419`),
but these bridge improvements inherit the same non-nested ID172 calibration risk.

The post-ID211 shapelet/motif update is the strongest recent cleaner-branch
sequence repair: ID209 improves ID206 from `0.535764` to `0.534530`, mainly by
repairing S4 from `0.600718` to `0.592263`. The same robustness lens also moves
from ID206 `0.568760` to ID209 `0.567055`, so the gain is not obviously
concentrated in a worse split. Carrying the existing ID172 delta weights forward
gives ID210 `0.523400` and ID211 `0.520106`; those bridge gains remain local
counterprobes because they inherit ID172 non-nested calibration exposure.

The post-ID214 wide-weight refinement pushes the same clean family a little
further: ID212 improves ID209 from `0.534530` to `0.534469` and robustness from
`0.567055` to `0.566994`. Carrying the bridge weights forward gives ID213
`0.523287` and ID214 `0.519982`; ID214 is the first artifact in this ladder below
local `0.52`, but this threshold is not a public leaderboard claim because ID214
is still a non-nested ID172 bridge.

The post-ID217 all-top40 refinement makes the clean shapelet/motif branch a
little stronger again: ID215 improves ID212 from `0.534469` to `0.534135` and
robustness from `0.566994` to `0.565863`. Carrying the same bridge weights
forward gives ID216 `0.522875` and ID217 `0.519556`. ID216 is the preferred
moderate-low counterprobe; ID217 is lower local but riskier because it uses a
larger ID172 exposure.

The post-ID220 multi-signal clean overlay adds only a tiny improvement, but it
keeps the artifact ladder tidy by combining several one-target side signals into
one clean artifact. ID218 improves ID215 from `0.534135` to `0.534103` and
robustness from `0.565863` to `0.565633`. Carrying the bridge weights forward
gives ID219 `0.522874` and ID220 `0.519561`. ID219 is the current preferred
moderate-low bridge. ID217 remains fractionally lower by raw local CV than
ID220, while ID220 is fractionally better by robustness.

The split robustness diagnostic adds a third lens across subject, date quartile,
date half, subject parity, within-subject phase, and target. Within the existing
OOF artifacts, the ID179->ID188 bridge lowers not only the overall score but also
the worst-subject, worst-date-bin, and worst-target losses monotonically. That is
useful evidence that the OOF gain is broad, but it is still not public proof
because the ladder remains increasingly exposed to the non-nested ID172
calibration source.

## External reference notebook: Public 0.5917 LGBM+XGB anchor blend

The shared notebook
`/Volumes/samsd/download_samsd/[Public_0.5917] LGBM+XGB 앵커·Subject-hole CV·안정성 피처선별·블렌드.ipynb`
is not a fully reproducible artifact by itself: it describes a V152 anchor,
stability-filtered features, and LGBM/XGB/Cat training functions, but does not
include the missing large feature matrices, V152 OOF/test predictions, or a
complete executable feature build. Its reported Local OOF `0.514` -> Public
`0.5917` gap is also large.

However, it is strategically useful because it independently points at the same
failure mode now confirmed by ID178: local CV can be dramatically overconfident.
The transferable ideas are:

- use a conservative anchor instead of letting high-variance residual models
  dominate final predictions;
- compare subject-hole/within-subject temporal folds rather than relying on one
  public-structure lens;
- prefer stability-selected, low-variance features over very large full-OOF
  feature pools;
- evaluate candidate submissions by public-gap risk signals such as prediction
  extremeness, distance from known submitted anchors, and target-wise drift, not
  just by the minimum local CV.

## Artifact contract from id 94 onward

Every experiment row from `id >= 94` must have all three artifacts, with globally
unique filenames:

1. `submissions/submission_id_<id>_<slug>_cv<score>_<status>.csv`
2. `notebooks/history/submission_<id>_<slug>_cv<score>_<status>.ipynb`
3. one matching `submissions/submission_log.csv` row with `notebook_file`

`make artifact-contract-audit` enforces this. Existing legacy rows are kept for
history, but new rows must not reuse generic or previously logged filenames.

## Immediate probe

`id=94` re-registers the preserved historical low-CV candidate:

- CSV: `submission_id_094_nvidia4stage_refreshed_probe_cv0.562183_unsubmitted.csv`
- Notebook: `submission_094_nvidia4stage_refreshed_probe_cv0.562183_unsubmitted.ipynb`
- Purpose: bold leaderboard calibration probe. It is not a confirmed improvement.

If this scores:

- `<= 0.590`: the old low-CV 4-stage stacker family still contains useful signal.
- `0.590–0.610`: use it as a diversity component, not a final model.
- `> 0.610`: treat the low CV as leakage/overfit and prioritize validation repair.

Do not submit without explicit approval.

## 2026-06-14 strict local experiments

The first strict-contract iteration deliberately favored validation integrity
over leaderboard-chasing:

- `id=95`: rebuilt v32 base models with official paper demographics and fixed
  logistic meta-stacking. Local CV `0.629446`. This did not improve the legacy
  v13 base family; demographics helped some slices but hurt the overall v32
  stack in this form.
- `id=96`: recovered archived v10-v13 base submission files under unique
  `id096_recovered_*` tags, then ran fixed logistic meta-stacking over all
  recovered base OOF/submission pairs. Local CV `0.633301`.
- `id=97`: same recovered base pool, but with a fixed trimmed mean and no
  learned weights. Local CV `0.617675`, the best strict artifact from this
  batch but still far from the `0.56` target.
- `id=98`: fixed raw-feature ExtraTrees with fold-wise same-subject priors and
  no future target interpolation. Local CV `0.607161`. This is the first strict
  improvement from a new raw covariate signal family rather than another
  recovered-base stacker.
- `id=99`: nested raw-feature selector over fixed base, subject-prior, and
  past-only target-anchor candidates. Local CV `0.604937`. Candidate selection
  for the reported OOF score is fold-nested; final test predictions select the
  best full-training OOF candidate per target. This is cleaner than the legacy
  low-CV family but still not close enough to justify a DACON submission.
- `id=100`: shutdown-sequence raw features plus the same nested raw selector.
  Local CV `0.603036`. Features are rebuilt from raw 18:00-09:00 screen,
  activity, light, heart-rate, and app-usage windows keyed by lifelog date.
  This mainly improved `Q1` and `S1`, confirming that sequence/order features
  are useful, but `Q2/Q3` remain the main blocker.
- `id=101`: fold-nested selector between `id=100`-style shutdown features and
  subject-relative shutdown features. Local CV `0.600495`. This improved `Q1`
  to `0.611071` and `S1` to `0.499176`; `Q2`/`Q3` remain high at
  `0.673299`/`0.664645`, so the next work should be Q2/Q3-specific rather than
  another all-target stack.
- `id=102`: Q2/Q3-only pre-sleep strain overlay on top of `id=101`. Local CV
  `0.597333`. It keeps `id=101` predictions for `Q1/S1/S2/S3/S4` and replaces
  only `Q2/Q3` through a fold-nested selector over same-day 12:00-24:00
  GPS/WiFi/app-usage/ambience/BLE context features. `Q2` improved materially to
  `0.649822`; nested `Q3` slightly worsened to `0.665986`, although the full OOF
  RF candidate scored `0.657076`. The WiFi home-anchor feature is label-free but
  transductive because it uses all available unlabeled context rows; treat this
  as a useful final-inference candidate, not a complete validation proof.
- `id=103`: fold-nested target-wise selector over strict artifacts `id=98`–`102`.
  Local CV `0.595284`. It selects `id=101/102/99/101/98/98-or99/98-or99`
  families by target/fold, improving `Q3` and `S3` while keeping the `Q2` gain
  from `id=102`. This is the current best strict local artifact, but `S4`
  worsened to `0.650090` under nested fold selection even though full-OOF choice
  prefers `id=98` at `0.644442`.
- `id=104`: Q3-only raw+Q23 overlay on top of `id=103`. Local CV `0.594182`.
  It preserves `id=103` for all non-Q3 targets and replaces Q3 with a
  fold-nested selector over raw processed features plus pre-sleep Q23 context
  features. The selected `rf6_l6` candidate improved Q3 from `0.660847` to
  `0.653132`; the generated submission hash differs from `id=103`, so this is a
  real new inference candidate rather than a validation-only reshuffle.
- `id=105`: S4-only raw+Q23 overlay on top of `id=104`. Local CV `0.592125`.
  It preserves all non-S4 predictions from `id=104` and replaces S4 with the
  fold-nested `et8_l10` raw+Q23 candidate, improving S4 from `0.650090` to
  `0.635691`. This is the current best strict local artifact.
- `id=106`: Q2-only q23 GB-grid overlay on top of `id=105`. Local CV `0.590854`.
  It preserves all non-Q2 predictions from `id=105` and replaces Q2 through a
  fold-nested selector over a compact GB/HGB neighborhood on the pre-sleep Q23
  context features. Nested Q2 improved from `0.649822` to `0.640924`; the final
  test Q2 candidate is `gb_n180_lr0.05_d1_l12`. The candidate neighborhood was
  narrowed after a local diagnostic, so this is a strong local candidate rather
  than an untouched independent proof.
- `id=107`: Q1-only morning-recovery overlay on top of `id=106`. Local CV
  `0.589452`. It preserves every non-Q1 target from `id=106` and replaces Q1
  with a fold-nested choice between the ID106 base and a fixed 75/25 blend of
  ID106 Q1 plus a morning-only logistic model. The morning features summarize
  post-sleep 04:00-12:00 HR, steps, activity, phone screen, charging, and light
  signals anchored back to the previous lifelog date. Nested Q1 improved from
  `0.611071` to `0.601254`. The direction was promoted after
  `reports/strict_q1_morning_recovery_diagnostic.json`, and the strict script
  uses a narrower core morning-recovery feature module, so treat it as a strong
  local candidate rather than an untouched independent proof.
- `id=108`: Q3-only compact q23 overlay on top of `id=107`. Local CV
  `0.589345`. The broad 248-candidate q23 diagnostic found a strong full-OOF
  Q3 blend (`0.643374`) but failed broad fold-nested selection (`0.656126`), so
  it was not promoted directly. A narrower compact pool from the top q23 blend
  neighborhood improved nested Q3 slightly from `0.653132` to `0.652383`. The
  artifact preserves every non-Q3 target from `id=107` and records the broad-grid
  caveat in metadata.
- `id=109`: Q2-only compact tabular-booster overlay on top of `id=108`. Local CV
  `0.588692`. The broad XGBoost/LightGBM diagnostic found attractive Q2
  candidates but the broad nested pool was not stable enough to promote. A
  narrower compact booster pool improved nested Q2 from `0.640924` to
  `0.636358`, and the generated submission/test predictions differ from
  `id=108` only for Q2. This is a real inference candidate rather than a
  validation-only OOF reshuffle.
- `id=110`: Q3-only compact tabular-booster overlay on top of `id=109`. Local CV
  `0.588687`. A broad Q3 XGBoost/LightGBM diagnostic showed only a tiny nested
  edge, but the candidate family was stable enough to confirm with a one-family
  compact LightGBM blend. Nested Q3 improved from `0.652383` to `0.652343`, and
  the generated submission/test predictions differ from `id=109` only for Q3.
  Because the improvement is only `0.000040` on Q3, this is mainly a clean
  incremental probe and not evidence that booster tuning alone can reach Public
  `0.55`.
- `id=111`: S4-only compact cross-target OOF-stacker overlay on top of `id=110`.
  Local CV `0.587978`. A broad cross-target diagnostic overfit under nested
  selection, but a one-family compact HGB blend over Q1/Q2/Q3/S1/S2/S3 OOF
  predictions improved nested S4 from `0.635691` to `0.630732`. The generated
  submission/test predictions differ from `id=110` only for S4. This is the first
  successful S4 improvement after the sleep-continuity failures and is a useful
  signal that the model can benefit from latent sleep-state distillation, not
  just more raw night features.
- `id=112`: Q3-only compact cross-target OOF-stacker overlay on top of `id=111`.
  Local CV `0.587748`. Q2 cross-target diagnostics found no new signal, but Q3
  improved after narrowing the broad cross-target pool to a compact logistic
  blend over Q1/Q2/S1/S2/S3/S4 OOF predictions. Nested Q3 improved from
  `0.652343` to `0.650732`, and the generated submission/test predictions differ
  from `id=111` only for Q3.
- `id=113`: S2-only compact cross-target OOF-stacker overlay on top of `id=112`.
  Local CV `0.585700`. Broad and compact cross-target diagnostics both supported
  a stable HGB blend over Q1/Q2/Q3/S1/S3/S4 OOF predictions. Nested S2 improved
  from `0.568228` to `0.553892`, and the generated submission/test predictions
  differ from `id=112` only for S2. Q1 and S3 cross-target diagnostics were not
  promoted because nested validation worsened despite small full-OOF attractions.
- `id=114`: S4-only cross-target refresh overlay on top of `id=113`. Local CV
  `0.585079`. Re-running the compact S4 HGB blend after the S2 shift improved
  nested S4 from `0.630732` to `0.626383`, while a Q3 refresh diagnostic worsened
  nested validation and was not promoted. The generated submission/test
  predictions differ from `id=113` only for S4.
- `id=115`: Q2-only compact top-app-token overlay on top of `id=114`. Local CV
  `0.584545`. Raw `mUsageStats` app names were converted into top-app
  duration/share features over afternoon/evening/late windows. The broad
  app-token pool overfit slightly under nested selection, but the one-family
  compact LGBM blend improved nested Q2 from `0.636358` to `0.632620`, and the
  generated submission/test predictions differ from `id=114` only for Q2. A
  more granular app-token v2 diagnostic did not improve nested Q2 and was not
  promoted.
- `id=116`: Q3-only compact subject-relative q23 overlay on top of `id=115`.
  Local CV `0.584376`. Q23 strain features were converted into per-subject
  IQR-relative and rank features using unlabeled train/test covariates. The
  compact LGBM blend improved nested Q3 from `0.650732` to `0.649550`, and the
  generated submission/test predictions differ from `id=115` only for Q3.
  Calendar/phase q23 diagnostics and post-ID116 S2/S4 refresh diagnostics were
  not promoted because nested validation selected the base or worsened.
- `id=117`: Q3-only semantic app-category confirm overlay on top of `id=116`.
  Local CV `0.584316`. Raw `mUsageStats` app names were mapped into auditable
  semantic categories and time-window summaries, avoiding raw app-name one-hot
  expansion. The broad app-category diagnostic showed a full-OOF Q3 edge but
  worsened under broad nested selection; the narrow confirm retained only the
  `appcat_abs` LightGBM family with blend weights capped at `0.20`, improving
  nested Q3 from `0.649550` to `0.649133`. The generated submission/test
  predictions differ from `id=116` only for Q3.
- `id=118`: Q1-only temperature confirm overlay on top of `id=117`. Local CV
  `0.584228`. The script keeps every non-Q1 target unchanged and applies only a
  fixed probability-temperature transform to Q1. Fold-nested selection improved
  Q1 from `0.601254` to `0.600637`, and the final test Q1 uses the full-OOF
  best candidate `temp_0.85`. Broad all-target calibration was not promoted;
  only this narrow Q1 improvement passed the nested check.
- `id=119`: Q1/Q2/Q3 temporal-anchor public-structure probe on top of `id=118`.
  Public-structure blocked CV `0.565628`; this is **not** strict Subject-Hole CV
  and is not directly comparable to IDs 95-118. It preserves S1/S2/S3/S4 from
  `id=118` and replaces Q1/Q2/Q3 with fixed same-subject temporal-anchor blends
  validated by interleaved within-subject blocked folds. The generated test
  predictions change only Q1/Q2/Q3. This is the first artifact near the working
  public-target zone, but it should be treated as a leaderboard calibration probe
  rather than a proven strict local improvement. No DACON submission was made.
- `id=120`: fine Q1/Q2/Q3 temporal-anchor public-structure probe on top of
  `id=118`. Public-structure blocked CV `0.565280`, improving the ID119 probe by
  tightening fixed alpha/half-life values (`Q1` alpha `0.725`, half-life `0.5`;
  `Q2` alpha `0.595`, half-life `84`; `Q3` alpha `0.570`, half-life `84`). It
  still changes only Q1/Q2/Q3 and remains non-strict/non-submitted.
- `id=121`: fine all-target temporal-anchor public-structure probe on top of
  `id=118`. Public-structure blocked CV `0.562941`, improving the ID120 probe by
  keeping the Q1 optimum, extending Q2/Q3 anchor half-life to `336`, and adding
  light S-target same-subject anchors (`S1` alpha `0.875`, half-life `0.25`;
  `S2` alpha `0.940`, half-life `0.25`; `S3` alpha `0.910`, half-life `0.25`;
  `S4` alpha `0.870`, half-life `10`). It changes Q2/Q3/S1/S2/S3/S4 versus
  ID120 and all seven targets versus strict ID118. It remains non-strict and
  was not submitted to DACON.
- `id=122`: mixed nearest/KNN all-target temporal-anchor public-structure probe
  on top of `id=118`. Public-structure blocked CV `0.560861`, improving ID121 by
  replacing nearest anchors with KNN label-memory where that was stronger:
  `Q2` uses `k=3`, alpha `0.502`; `S2` uses `k=5`, alpha `0.779`; `S4` uses
  `k=7`, alpha `0.736`. Q1 keeps the ID120/ID121 nearest anchor. Q3/S1/S3 use
  slightly fine-tuned nearest anchors. It changes Q2/Q3/S1/S2/S3/S4 versus
  ID121 and all seven targets versus strict ID118. It remains non-strict and was
  not submitted to DACON.
- `id=123`: high-risk full-OOF convex temporal-anchor public-structure probe on
  top of `id=118`. Public-structure full-OOF probe CV `0.559964`, improving ID122
  by fitting convex weights over the base prediction and multiple fold-blocked
  temporal-anchor components per target. It changes every target versus ID122.
  The score is **not fold-nested**; a stricter nested convex diagnostic worsened
  to about `0.570330`, so this is a leaderboard calibration probe only, not a
  verified local improvement. No DACON submission was made.
- `id=124`: S3/S4 raw hourly-sequence public-structure overlay on top of
  `id=123`. Public-structure probe CV `0.559501`, improving ID123 by preserving
  Q1/Q2/Q3/S1/S2 and replacing only S3/S4 with fixed blends of the base
  prediction and raw hourly-sequence OOF models. The sequence features flatten
  lifelog-date `12:00` through next-day morning into hour-bin summaries for
  screen, charging, activity, phone/watch light, heart rate, and pedometer
  signals. This is inspired by Pixleep-style sensor-to-image/sequence modeling
  but remains a lightweight local probe. It is not strict Subject-Hole CV and no
  DACON submission was made.
- `id=125`: high-risk target-wise temperature/shift calibration on top of
  `id=124`. Public-structure full-OOF calibration probe CV `0.557413`, improving
  every target under the non-nested calibration lens (`Q1` `0.572564`, `Q2`
  `0.565015`, `Q3` `0.593850`, `S1` `0.485383`, `S2` `0.547914`, `S3`
  `0.523634`, `S4` `0.613530`). It changes all target columns versus ID124.
  Parameters were selected by a full-OOF calibration scan, not fold-nested
  selection, so this is a leaderboard calibration probe only. No DACON
  submission was made.
- `id=126`: extreme-risk isotonic/quantile calibration on top of `id=125`.
  Public-structure full-OOF calibration probe CV `0.529691`, crossing the
  nominal local/probe `0.55` threshold only under a highly optimistic lens. All
  seven target columns change versus ID125 (`Q1` `0.542938`, `Q2` `0.529484`,
  `Q3` `0.570140`, `S1` `0.463027`, `S2` `0.520444`, `S3` `0.495874`, `S4`
  `0.585929`). Target methods are a mixture of isotonic and quantile mapping
  fitted on full OOF predictions, not fold-nested. The nested calibration
  reference mean is `0.562377`, so ID126 is useful as a bold leaderboard probe
  and calibration diagnostic, not as robust evidence that the true local target
  has been reached. No DACON submission was made.
- `id=127`: tempered extreme-calibration blend between `id=125` and `id=126`.
  Public-structure full-OOF blend probe CV `0.549454`, using the fixed formula
  `0.80 * ID125 + 0.20 * ID126` for all seven targets. Target scores are `Q1`
  `0.562342`, `Q2` `0.556105`, `Q3` `0.588041`, `S1` `0.479966`, `S2`
  `0.538624`, `S3` `0.517348`, `S4` `0.603753`. It changes all target columns
  versus ID125 but with much smaller test deltas than ID126. This is the current
  risk-controlled sub-`0.55` public-structure probe candidate, not a strict or
  nested validation win. No DACON submission was made.
- `id=128`: cleaner-anchor tempered extreme-calibration blend between `id=122`
  and `id=126`. Public-structure full-OOF blend probe CV `0.549956`, using the
  fixed formula `0.75 * ID122 + 0.25 * ID126` for all seven targets. Target
  scores are `Q1` `0.561898`, `Q2` `0.558154`, `Q3` `0.587455`, `S1`
  `0.482906`, `S2` `0.536943`, `S3` `0.516352`, `S4` `0.605986`. It changes
  all target columns versus ID122. It is slightly worse than ID127 under the
  optimistic public-structure metric, but its base anchor is cleaner, so it is a
  useful submission-order counterprobe. No DACON submission was made.
- `id=129`: strict Q3-only GPS routine overlay on top of `id=118`. Local
  Subject-Hole CV `0.584178`, improving ID118 by replacing only Q3 with a
  fold-nested selector over privacy-safe GPS routine numeric summaries. Q3
  improves from `0.649133` to `0.648784`; every non-Q3 target is unchanged. Raw
  GPS coordinates are used only transiently to build features and are not stored
  in metadata. This is a real strict local improvement, but the gain is small
  and still far from the public `0.55` target. No DACON submission was made.
- `id=130`: mixed nearest/KNN all-target temporal-anchor public-structure probe
  on top of strict `id=129`. Public-structure blocked CV `0.560757`, slightly
  improving the original ID122 cleaner fixed-anchor probe (`0.560861`) by
  carrying the ID129 Q3 GPS-routine strict improvement into the public-structure
  base before applying the same mixed-anchor parameters. It remains non-strict
  and was not submitted to DACON.
- `id=131`: GPS-base tempered extreme-calibration blend between `id=130` and
  `id=126`. Public-structure full-OOF blend probe CV `0.549876`, using
  `0.75 * ID130 + 0.25 * ID126`. It is slightly better than ID128 under the same
  cleaner-anchor risk-control idea because ID130 includes the ID129 Q3 GPS base.
  It remains non-strict/non-nested and was not submitted to DACON.
- `id=132`: refined KNN temporal-anchor public-structure probe on top of
  `id=130`. Public-structure blocked CV `0.560259`, improving ID130 by applying
  fixed low-weight KNN same-subject refinements to Q2 (`k=4`, weight `0.10`),
  Q3 (`k=3`, weight `0.10`), S2 (`k=21`, weight `0.20`), and S4 (`k=21`, weight
  `0.20`). Q1/S1/S3 remain unchanged. This is a cleaner anchor, not a strict
  Subject-Hole artifact.
- `id=133`: refined-anchor tempered extreme-calibration blend between `id=132`
  and `id=126`. Public-structure full-OOF blend probe CV `0.545875`, using
  `0.65 * ID132 + 0.35 * ID126`. It changes every target versus ID132, ID131,
  and ID127. This is the current lowest tempered public-structure probe, but it
  is less conservative than ID127/ID131 because it uses a larger ID126 extreme
  calibration share.
- `id=134`: balanced refined-anchor tempered blend between `id=132` and
  `id=126`. Public-structure full-OOF blend probe CV `0.540925`, using
  `0.50 * ID132 + 0.50 * ID126`. This is a stronger calibration-exposure rung
  than ID133, not strict/nested evidence.
- `id=135`: aggressive refined-anchor tempered blend between `id=132` and
  `id=126`. Public-structure full-OOF blend probe CV `0.535430`, using
  `0.30 * ID132 + 0.70 * ID126`. It approaches ID126 and should be treated as a
  high-variance leaderboard probe only.
- `id=136`: extreme calibration on the balanced refined-anchor blend `id=134`.
  Public-structure full-OOF calibration probe CV `0.523144`, using full-OOF
  isotonic/quantile calibrators on all seven targets. This is the current lowest
  local public-structure artifact, but it is double-calibrated and explicitly
  non-nested; the known nested calibration reference remains `0.562377`.
- S4 sleep-continuity diagnostics after `id=108` were not promoted: full OOF
  looked promising, but broad, compact, and single-candidate nested checks all
  worsened S4. The next S4 attempt should therefore use a different signal
  family rather than more variants of the same continuity blend.
- A temporary pair-selector artifact was removed because it improved validation
  by selecting older OOF predictions for S2/S3 but produced an unchanged
  submission CSV. From here, no candidate should be promoted unless the actual
  test/submission predictions change versus the prior best.

Interpretation: the low-CV `0.562183` family is not reproduced by simple
leak-safe stacking. Its advantage likely comes from aggressive same-OOF
post-processing, hidden recovered assets not yet matched, or a validation
contract that is too optimistic. The raw-feature path is now the preferred
research path because it improved `id=97` by `0.017180` without using
future-label interpolation. Treat `id=94` as a calibration probe, not a proven
model family. After `id=118`, the strict Subject-Hole path nearly plateaued, but
ID129 shows that Q3 still has small gains from new raw-routine features. The
clearest strict paths are still Q2/Q3 reductions, while the public-structure path
now has an extreme-risk full-OOF calibrated probe at `0.529691`, a tempered
sub-`0.55` blend probe at `0.549454`, a high-risk target calibration probe at
`0.557413`, a raw-sequence-enhanced probe at `0.559501`, and a cleaner
fixed-anchor probe at `0.560861`. The ID126 score
should be interpreted against its nested reference `0.562377`, not as a robust
standalone validation win. The most promising new signal families have shifted:
Q2 benefits from raw app-token
usage, while Q3 benefits from subject-relative q23 normalization and semantic
app-category compression. Calendar-only, broad cross-target refreshes, and broad
app-category selectors should not be promoted without compact nested
confirmation. Calibration should also remain target-specific and narrow: the
current evidence supports Q1 temperature scaling only, while Q2/Q3/S-target
calibration worsened under nested validation.

## Signal plan

### 1. Official paper demographics

The ETRI paper discloses subject-level gender, age group, employment, and BMI for
`id01`–`id10`. The baseline paper reports that demographic and weekday features
improve performance. Add these as static subject features, with no row-level
leakage:

- gender,
- age decade / `age_ge_40`,
- employment,
- BMI / BMI bucket.

This is a high-priority feature track because the current code search shows no
clear active demographic feature integration.

Status: implemented in `src/features/demographics.py` and merged by
`src/data/make_dataset.py`. The first v32 run did not improve local CV, so keep
these features available but do not assume they are a positive signal without
target-specific selection.

### 2. Target-specific feature selection

The paper explicitly warns that selective feature groups outperform using all
features together. Use target-specific families:

- `Q1`: 3-hour step windows, weekday/weekend, sleep-duration proxies, morning
  activity suppression vs afternoon activity.
- `Q2/Q3`: app category usage, late-night phone usage, ambient sound context,
  HR-with-low-activity stress proxy, routine disruption.
- `S1`: sleep-window duration proxy from screen-off, stillness, low light, HR dip.
- `S2`: sleep efficiency proxy from consolidated stillness and low wake events.
- `S3`: sleep onset latency proxy from light drop -> screen-off -> stillness.
- `S4`: WASO proxy from night screen-on bursts, light spikes, HR spikes, and
  activity fragmentation.

Status for Q2/Q3: implemented in `src/features/q_context.py` and promoted by
`scripts/run_strict_q23_strain_experiment.py` as `id=102`, then combined by
`scripts/run_strict_artifact_selector_experiment.py` as `id=103`, then combined
with raw features by `scripts/run_strict_q3_raw_q23_experiment.py` as `id=104`.
The feature family is useful for Q2 and, when combined with raw covariates, Q3.

Status for Q1: implemented in `src/features/morning_recovery.py` and promoted by
`scripts/run_strict_q1_morning_recovery_experiment.py` as `id=107`. This uses
post-sleep morning recovery signals rather than only pre-sleep shutdown signals,
which better matches the official Q1 timing.

Status for Q1 probability calibration: promoted by
`scripts/run_strict_q1_temperature_confirm_experiment.py` as `id=118`. This is a
narrow temperature-only post-processing overlay on the current strict base, not a
broad recalibration layer.

Status for Q3 compact q23 blending: promoted by
`scripts/run_strict_q3_q23_compact_experiment.py` as `id=108`. The broad q23
grid was too unstable under nested selection, so only the compact top-neighborhood
diagnostic was promoted, with an explicit caveat.

Status for Q2 compact tabular boosters: promoted by
`scripts/run_strict_q2_booster_compact_experiment.py` as `id=109`. The compact
Q2 booster pool improved nested Q2 and changed Q2 test predictions; broad booster
and unchanged-submission selectors remain diagnostic-only.

Status for Q3 compact tabular boosters: promoted by
`scripts/run_strict_q3_booster_compact_experiment.py` as `id=110`. The compact
Q3 LightGBM blend improved nested Q3 by a very small margin and changed Q3 test
predictions; the result is useful for continuity but not a large enough signal
for the Public `0.55` target by itself.

Status for S4 cross-target distillation: promoted by
`scripts/run_strict_s4_cross_target_compact_experiment.py` as `id=111`. Broad
cross-target stacking overfit, but the compact HGB blend over other-target OOF
predictions improved nested S4 and changed S4 test predictions. This is now the
preferred S4 track over more sleep-continuity variants.

Status for Q3/S2/S4 cross-target distillation: promoted by
`scripts/run_strict_q3_cross_target_compact_experiment.py` as `id=112`,
`scripts/run_strict_s2_cross_target_compact_experiment.py` as `id=113`, and a
base-id-113 refresh of `scripts/run_strict_s4_cross_target_compact_experiment.py`
as `id=114`. Q3 uses a compact logistic blend; S2 and S4 use compact HGB blends.
Q2, Q1, and S3 cross-target diagnostics have not been promoted because their
nested checks failed or selected the base artifact; this includes Q2 refreshes
after ID114 and cross-target refreshes after ID115/ID116.

Status for Q2 app-token features: promoted by
`scripts/run_strict_q2_app_token_compact_experiment.py` as `id=115`. This uses
raw app-name duration/share features from `mUsageStats`; top-app vocabulary is
label-free transductive and recorded in metadata.

Status for Q3 subject-relative q23 features: promoted by
`scripts/run_strict_q3_subject_relative_compact_experiment.py` as `id=116`. This
uses per-subject IQR/rank normalization over unlabeled q23 covariates and remains
a useful Q3 track over more calendar-only or app-token variants.

Status for Q3 semantic app-category features: promoted by
`scripts/run_strict_q3_app_category_confirm_experiment.py` as `id=117`. The
broad `scripts/run_strict_q3_app_category_compact_experiment.py` diagnostic
found a full-OOF Q3 edge but failed broad nested selection; only the narrowed
confirm pool was promoted.

Status for Q3 GPS routine features: promoted by
`scripts/run_strict_q3_gps_routine_overlay_experiment.py` as `id=129`. This uses
privacy-safe GPS movement/routine summaries such as evening home distance,
location-cell diversity, movement timing, and path deltas. Q2 failed the same
diagnostic, but Q3 passed nested Subject-Hole selection with a small improvement.

Post-ID116 diagnostics, not promoted:

- `reports/strict_q2_subject_relative_app_diagnostic_id116.json`: combining the
  successful Q2 app-token family with per-subject IQR/rank normalization worsened
  Q2 (`0.632620 -> 0.633343` nested/local), so no ID117 artifact was created.
- `reports/strict_q1_morning_recovery_refresh_diagnostic_id116.json`: refreshing
  the older morning-recovery overlay on top of ID116 selected the base for every
  fold; Q1 remained `0.601254`, so no promotion.
- `reports/strict_q2_env_token_compact_diagnostic_id116.json`: Kaggle-offloaded
  compact hashed WiFi/BLE environment-token features produced a weak raw Q2
  model (`0.689999`) and all nested folds selected ID116 base; no promotion.
- `reports/strict_q2_app_category_compact_diagnostic_id116.json`: semantic
  app-category features showed a small full-OOF Q2 edge (`0.632620 -> 0.632385`)
  but broad nested selection worsened Q2 to `0.635974`; no promotion.
- `reports/strict_q3_app_category_compact_diagnostic_id116.json`: semantic
  app-category features showed a useful full-OOF Q3 edge (`0.649550 ->
  0.648407`) but broad nested selection worsened Q3 to `0.651358`; this led to
  the narrower confirmation script rather than direct promotion.
- `reports/strict_q3_app_category_confirm_diagnostic_id116.json`: narrow Q3
  app-category confirmation improved local CV to `0.584316` and nested Q3 to
  `0.649133`; promoted as `id=117`.
- `reports/strict_q2_app_category_confirm_diagnostic_id117.json`: narrow Q2
  app-category confirmation preserved the small full-OOF attraction
  (`0.632620 -> 0.632385`) but worsened nested Q2 to `0.634151`; no promotion.
- `reports/strict_s4_cross_target_compact_diagnostic_id118.json`: refreshing
  S4 cross-target distillation after the ID118 Q1 calibration selected the base
  for every fold and full OOF; no promotion.
- `reports/strict_q3_cross_target_compact_diagnostic_id118.json`: refreshing
  Q3 cross-target distillation after ID118 worsened nested Q3 to `0.650969`;
  no promotion.
- `reports/strict_s2_cross_target_compact_diagnostic_id118.json`: refreshing
  S2 cross-target distillation after ID118 selected the base for every fold and
  full OOF; no promotion.
- `reports/strict_q3_joint_q23_app_category_diagnostic_id118.json`: feature-level
  joint training over subject-relative q23 and semantic app-category features
  underperformed the existing prediction-level ID118 base (`0.649133` base Q3
  vs best joint blend `0.649463` full OOF, nested Q3 `0.650459`); no promotion.
- `reports/strict_s4_ambience_context_diagnostic_id118.json`: semantic ambience context had a strong full-OOF S4 attraction (`0.626383 -> 0.623503`) but worsened nested S4 to `0.629850`; no promotion.
- `reports/strict_s4_ambience_context_confirm_diagnostic_id118.json`: the narrower focused ambience confirmation still worsened nested S4 (`0.626383 -> 0.627025`) despite full-OOF improvement (`0.624385`); no promotion.
- `reports/strict_q2_app_token_reblend_diagnostic_id118.json`: second-stage Q2 app-token reblend improved full OOF (`0.632620 -> 0.632370`) but worsened nested Q2 to `0.633127`; no promotion.
- Local temperature-scaling scan after ID118 showed full-OOF attractions for several targets, but fold-nested selection worsened every scanned target; no calibration promotion beyond the existing ID118 Q1 transform.
- `reports/strict_hourly_sequence_overlay_diagnostic_id118.json`: lightweight Pixleep-inspired hourly sequence features over Q2/Q3/S3/S4 showed small full-OOF attractions for Q3/S3/S4, but nested Subject-Hole selection worsened every checked target (`Q2` `0.632620 -> 0.632918`, `Q3` `0.649133 -> 0.651291`, `S3` `0.527756 -> 0.530641`, `S4` `0.626383 -> 0.630638`); no strict artifact was promoted. This supports moving heavier sequence/image modeling to Kaggle rather than adding more shallow local sequence blends.
- `reports/experiment_129_q3_gps_routine_overlay_cv0.584178_unsubmitted.json`: Q3 GPS routine summaries passed nested Subject-Hole selection (`0.649133 -> 0.648784`) and were promoted as ID129. The feature family summarizes movement/routine relative to subject-level evening GPS centroids without storing raw coordinates in metadata. The improvement is small but strict-comparable.
- `reports/strict_gps_routine_all_targets_diagnostic_id129.json`: reusing the GPS routine family on top of ID129 did not produce another strict promotion. Q1/Q2/Q3/S2/S3/S4 worsened under nested selection and S1 selected the base; no new artifact was created.
- `reports/strict_cross_target_refresh_diagnostic_id129.json`: refreshing Q3/S2/S4 cross-target compact stackers after the ID129 Q3 shift did not improve nested validation (`Q3` worsened to `0.650979`, S2 stayed at base, S4 worsened to `0.628156`); no strict artifact was created.
- `reports/legacy_user_embedding_overlay_diagnostic_id129.json`: legacy
  `user_embedding.weight`-style subject embeddings were checked only as a
  diagnostic. They appear supervised/high-risk, and nested validation did not
  improve any target over ID129, so they must not be promoted or used as strict
  evidence.
- `reports/strict_chronos_embedding_overlay_diagnostic_id129.json`: Kaggle CPU
  runner extracted Chronos-T5 mini embeddings from nine known daily sensor
  covariate sequences and tested Q2/Q3/S3/S4 overlays on top of ID129. Q2 and
  S4 had full-OOF attractions (`Q2` `0.632620 -> 0.630602`, `S4`
  `0.626383 -> 0.625875`), but nested validation worsened them (`Q2`
  `0.634789`, `S4` `0.629095`); Q3 and S3 selected the base under full OOF.
  No strict artifact was promoted.
- `reports/strict_tree_model_refresh_diagnostic_id129.json`: a strict
  Subject-Hole refresh with LR/HGB/ET/LGBM/XGBoost/CatBoost over numeric raw
  features found full-OOF attractions for Q2/S2/S3/S4, but nested validation
  worsened every checked target. No strict artifact was promoted.
- `reports/public_structure_temporal_anchor_diagnostic_id118.json`: a separate within-subject blocked diagnostic, not comparable to strict Subject-Hole CV, suggests same-subject temporal label anchors are powerful for Q targets only. Fixed Q1/Q2/Q3 anchor blends scored `0.565628` as ID119; fine fixed values scored `0.565280` as ID120 under this public-structure lens while keeping S-targets at ID118. Because this changes validation philosophy, do not treat these as strict-local improvements and do not submit without explicit user approval.
- `reports/experiment_126_extreme_isotonic_quantile_calibration_public_probe_cv0.529691_unsubmitted.json`: ID126 proves that aggressive full-OOF isotonic/quantile calibration can drive the public-structure probe metric below `0.55`, but the nested reference (`0.562377`) says most of that gain is likely calibration overfit. Use it only as a DACON probe candidate after explicit approval, and keep ID122/ID125 as safer comparison anchors.
- `reports/experiment_127_tempered_extreme_calibration_public_probe_cv0.549454_unsubmitted.json`: ID127 uses only 20% of the ID126 extreme calibration signal on top of ID125. It is still non-strict/non-nested, but it is a more conservative sub-`0.55` public-structure probe than ID126 and changes all seven target columns with smaller test deltas.
- `reports/experiment_128_clean_anchor_tempered_extreme_calibration_public_probe_cv0.549956_unsubmitted.json`: ID128 uses 25% of the ID126 extreme calibration signal on top of the cleaner ID122 fixed-anchor probe. It is slightly worse than ID127 but gives a cleaner-anchor sub-`0.55` counterprobe for leaderboard mapping.
- `reports/experiment_130_q123s1234_mixed_temporal_anchor_gpsbase_public_probe_cv0.560757_unsubmitted.json`: ID130 reruns the cleaner mixed temporal-anchor probe on top of strict ID129, slightly improving ID122 (`0.560861 -> 0.560757`) through the Q3 GPS base change.
- `reports/experiment_131_gpsbase_tempered_extreme_calibration_public_probe_cv0.549876_unsubmitted.json`: ID131 applies the same 25% ID126 extreme-calibration tempering to ID130, becoming the latest cleaner-anchor sub-`0.55` public-structure counterprobe.
- `reports/experiment_132_knn_temporal_refine_public_probe_cv0.560259_unsubmitted.json`: ID132 improves the clean GPS-base temporal-anchor path (`ID130` `0.560757 -> 0.560259`) with fixed low-weight KNN refinements on Q2/Q3/S2/S4. It is non-strict but cleaner than calibration-heavy probes.
- `reports/experiment_133_refined_anchor_tempered_extreme_public_probe_cv0.545875_unsubmitted.json`: ID133 blends the refined clean anchor with ID126 at `0.65/0.35`, lowering the public-structure probe below ID127/ID131. It is a stronger leaderboard probe, but the larger ID126 share means higher non-nested calibration risk.
- `reports/experiment_134_balanced_refined_anchor_extreme_public_probe_cv0.540925_unsubmitted.json`: ID134 increases ID126 exposure to a balanced `0.50/0.50` blend with ID132, lowering public-structure CV to `0.540925`.
- `reports/experiment_135_aggressive_refined_anchor_extreme_public_probe_cv0.535430_unsubmitted.json`: ID135 uses `0.30 * ID132 + 0.70 * ID126`, giving an aggressive pre-extreme calibration probe at `0.535430`.
- `reports/experiment_136_extreme_calibration_on_balanced_refined_anchor_public_probe_cv0.523144_unsubmitted.json`: ID136 applies a second full-OOF isotonic/quantile calibration on top of ID134, producing the lowest local public-structure CV so far (`0.523144`) but also the highest overfit risk.
- `reports/experiment_137_temporal_hole_anchor_selector_cv0.580548_unsubmitted.json`: ID137 promotes a strict Subject-Hole temporal-anchor selector on top of ID129. It uses the same `make_subject_hole_folds` lens as the strict artifacts and fold-nested candidate selection over base plus narrow same-subject exponential/KNN label anchors. The mean strict CV improves from ID129 `0.584178` to `0.580548`, with the largest gains on Q1 and Q2, but Q3 and S1 slightly worsen under the nested selector.
- `reports/experiment_138_q123s1234_mixed_temporal_anchor_holebase_public_probe_cv0.562347_unsubmitted.json`: ID138 reruns the mixed public-structure temporal-anchor probe on top of ID137. It scores `0.562347`, worse than the cleaner public anchor ID132 (`0.560259`), so the stricter ID137 base does not automatically improve the public-structure ladder.
- `reports/experiment_139_guarded_temporal_hole_anchor_selector_cv0.580390_unsubmitted.json`: ID139 is the guarded strict version of ID137. It keeps the temporal-hole anchor selector only where nested validation improves over ID129, and guards Q3/S1 back to the base. This becomes the current best strict Subject-Hole artifact (`0.580390`) while preserving lower-risk test predictions for the two non-improving targets.
- `reports/strict_anchor_meta_model_diagnostic_id139.json`: a nested logistic/HGB/ExtraTrees meta-model over base plus many same-subject anchor features failed to beat ID139 on every target. This closed the "let a small model read all anchors" path for now; the base/handpicked anchors were more reliable.
- `reports/strict_periodic_anchor_diagnostic_id139.json`: weekly/weekday same-subject anchors found nested improvements for S1 and S2. This led to ID140, which applies guarded periodic anchors on top of ID139 and improves strict CV from `0.580390` to `0.579624`.
- `reports/experiment_140_periodic_anchor_overlay_cv0.579624_unsubmitted.json`: ID140 modifies only S1/S2. S1 improves from `0.499176` to `0.495602` with weekly-lag anchors, and S2 improves from `0.550380` to `0.548592` with subject-weekend anchors.
- `reports/strict_feature_similarity_anchor_diagnostic_id140.json`: same-subject feature-similarity KNN anchors over label-free train/test covariates found a strong nested S3 date-time similarity signal and a tiny S2 screen/app signal. The diagnostic also had attractive full-OOF Q1/Q2/Q3/S1/S4 candidates, but those failed nested selection and were not promoted as strict improvements.
- `reports/experiment_141_feature_similarity_anchor_overlay_cv0.577197_unsubmitted.json`: ID141 promotes the nested feature-similarity signals from ID140. S3 improves materially from `0.527549` to `0.510644`, and S2 improves slightly from `0.548592` to `0.548508`.
- `reports/experiment_142_s3_refined_date_similarity_anchor_cv0.576995_unsubmitted.json`: ID142 refines the S3 date-time KNN neighborhood after ID141. The nested S3 score improves from `0.510644` to `0.509224`; the full-OOF candidate selected for final test is `dt_k7_dw0.05_w0.5` with S3 full-OOF `0.503023`, so this remains a strong but locally tuned S3 inference candidate.
- `reports/strict_fixed_feature_similarity_confirmation_id142.json`: a fixed
  full-OOF confirmation on top of ID142 found attractive Q2/Q3 screen/app
  feature-similarity candidates (`Q2` `0.626180 -> 0.618793`, `Q3`
  `0.648784 -> 0.640376`). S4 was attractive only under the diagnostic's
  original `env` family definition.
- `reports/experiment_143_fixed_feature_similarity_overlay_cv0.575037_unsubmitted.json`:
  ID143 applied fixed Q2/Q3/S4 candidates on top of ID142, but the recreated
  S4 `env` family had 47 columns rather than the diagnostic's 24-column family
  and worsened S4 (`0.625986 -> 0.628075`). Keep this artifact as a documented
  mismatch/probe, not as the preferred fixed feature-similarity candidate.
- `reports/experiment_144_q23_fixed_feature_similarity_overlay_cv0.574738_unsubmitted.json`:
  ID144 keeps only the reproducible Q2/Q3 screen/app fixed candidates from
  ID143 and preserves S4 at ID142. It lowers fixed Subject-Hole OOF CV from
  ID142 `0.576995` to `0.574738`, but it is still locally tuned from the ID142
  confirmation diagnostic and therefore weaker evidence than a fully
  fold-nested strict promotion.
- `reports/experiment_145_q23s4_env24_fixed_feature_similarity_overlay_cv0.574355_unsubmitted.json`:
  ID145 reconstructs the diagnostic's 24-column S4 environment family
  (`ambience`/`wifi`/`ble`) and combines it with the same Q2/Q3 screen/app fixed
  anchors. It reproduces the ID142 confirmation losses for Q2, Q3, and S4 and
  becomes the preferred fixed feature-similarity leaderboard probe (`0.574355`),
  while ID142 remains the preferred fully nested strict-integrity base.
- `reports/experiment_146_q123s14_fixed_feature_similarity_overlay_cv0.573671_unsubmitted.json`:
  ID146 adds the Q1 date-time and S1 light/HR/sleep fixed candidates to ID145.
  It is the lowest fixed feature-similarity local OOF probe so far (`0.573671`),
  but Q1/S1 have weaker fold stability than Q2/Q3 and should be treated as an
  aggressive leaderboard-mapping candidate rather than a strict-validation
  improvement.
- `reports/strict_feature_similarity_grid_id146.json`: a broader fixed
  feature-similarity grid on top of ID146 found another layer of full-OOF
  attractions: Q3 all-feature nearest-neighbor (`0.640376 -> 0.627009`), S3
  screen/app (`0.509224 -> 0.500989`), Q2 all-feature (`0.618793 -> 0.614668`),
  S2 move/GPS (`0.548508 -> 0.546074`), and S4 light/HR (`0.623305 ->
  0.621332`). Candidate choice is not nested.
- `reports/experiment_147_q23s234_refined_grid_feature_similarity_overlay_cv0.569366_unsubmitted.json`:
  ID147 promotes the ID146 grid winners for Q2/Q3/S2/S3/S4 while preserving the
  ID146 Q1/S1 anchors. This becomes the strongest fixed feature-similarity local
  OOF probe so far (`0.569366`), still below the public-structure calibration
  ladder but materially better than ID146 without using isotonic/quantile
  calibration.
- `reports/strict_feature_similarity_grid_id147.json`: a residual grid over
  ID147 found a smaller second layer of fixed feature-similarity gains for all
  searched targets, led by Q3 move/GPS (`0.627009 -> 0.622701`) and S3 move/GPS
  (`0.500989 -> 0.498936`).
- `reports/experiment_148_q23s234_second_refined_grid_feature_similarity_overlay_cv0.567949_unsubmitted.json`:
  ID148 promotes the ID147 residual-grid winners. It is now the strongest
  non-calibration feature-similarity probe (`0.567949`), but it is two layers of
  full-OOF-selected fixed anchors and therefore high tuning risk.
- `reports/strict_feature_similarity_grid_id148.json`: an all-target residual
  grid over ID148 found small but consistent fixed-anchor gains on every target,
  led by S1 screen/app (`0.493657 -> 0.491655`) and Q3 screen/app (`0.622701 ->
  0.620218`).
- `reports/experiment_151_q123s1234_third_refined_grid_feature_similarity_overlay_cv0.566666_unsubmitted.json`:
  ID151 promotes the ID148 residual-grid winners on all seven targets and is the
  current strongest non-calibration feature-similarity probe (`0.566666`). It is
  useful as a cleaner counterweight to calibration-heavy probes, but it is three
  layers of full-OOF-selected anchors and should be treated as locally tuned.
- `reports/strict_feature_similarity_grid_id151.json`: a fourth, lower-weight
  residual grid over ID151 still found small gains on every target, led by Q1
  light/HR (`0.580383 -> 0.579012`), S3 screen/app (`0.497646 -> 0.496423`),
  and S1 environment (`0.491655 -> 0.490554`).
- `reports/experiment_153_q123s1234_fourth_refined_grid_feature_similarity_overlay_cv0.565947_unsubmitted.json`:
  ID153 promotes the ID151 residual-grid winners and becomes the current
  strongest non-calibration feature-similarity probe (`0.565947`). The monotone
  improvement is encouraging, but this is now four layers of full-OOF-selected
  anchors and should not be mistaken for a nested strict improvement.
- `reports/strict_feature_similarity_grid_id153.json`: a fifth, very low-weight
  residual grid over ID153 still found small all-target gains, with the largest
  remaining signal on S3 environment similarity (`0.496423 -> 0.495596`).
- `reports/experiment_155_q123s1234_fifth_refined_grid_feature_similarity_overlay_cv0.565677_unsubmitted.json`:
  ID155 promotes the ID153 residual-grid winners and is the current strongest
  non-calibration feature-similarity probe (`0.565677`). The absolute gain over
  ID153 is small, so this should be considered an aggressively tuned probe
  rather than a robust validation breakthrough.
- `reports/id148_id136_tempered_blend_scan.json`: blending the non-calibrated
  ID148 feature-similarity probe with the extreme calibration ID136 showed that
  a global 10/90 ID148/ID136 blend scores `0.522802`, while target-wise weights
  score `0.521895`. This is leaderboard-mapping evidence only, not robust
  validation.
- `reports/experiment_149_id148_tempered_id136_extreme_public_probe_cv0.522802_unsubmitted.json`:
  ID149 is the global 10% ID148 + 90% ID136 tempered calibration probe. It is a
  tiny local improvement over ID136, still dominated by non-nested extreme
  calibration.
- `reports/experiment_150_id148_targetwise_tempered_id136_extreme_public_probe_cv0.521895_unsubmitted.json`:
  ID150 uses target-wise ID148/ID136 weights from the blend scan and is the
  lowest local public-structure probe so far. It has the highest tuning risk in
  the current ladder and must not be treated as a strict local achievement.
- `reports/id151_id136_tempered_blend_scan.json`: replacing ID148 with the
  cleaner ID151 base in the extreme-calibration blend scan improves the
  target-wise local probe slightly (`0.521895 -> 0.521641`).
- `reports/experiment_152_id151_targetwise_tempered_id136_extreme_public_probe_cv0.521641_unsubmitted.json`:
  ID152 is the new lowest local public-structure probe. It uses target-wise
  ID151/ID136 weights and is even more leaderboard-mapping oriented than ID150;
  do not interpret it as a robust validation result.
- `reports/id153_id136_tempered_blend_scan.json`: replacing ID151 with the
  cleaner ID153 base in the extreme-calibration blend scan improves the
  target-wise local probe slightly (`0.521641 -> 0.521538`).
- `reports/experiment_154_id153_targetwise_tempered_id136_extreme_public_probe_cv0.521538_unsubmitted.json`:
  ID154 is the new lowest local public-structure probe. It uses target-wise
  ID153/ID136 weights and remains an extreme leaderboard-mapping candidate, not
  a strict validation result.
- `reports/id155_id136_tempered_blend_scan.json`: replacing ID153 with ID155 in
  the extreme-calibration blend scan improves the target-wise local probe again
  (`0.521538 -> 0.521411`).
- `reports/experiment_156_id155_targetwise_tempered_id136_extreme_public_probe_cv0.521411_unsubmitted.json`:
  ID156 is the new lowest local public-structure probe. It uses target-wise
  ID155/ID136 weights and remains an extreme leaderboard-mapping candidate, not
  a strict validation result.
- `reports/experiment_157_id155_extreme_calibration_public_probe_cv0.538161_unsubmitted.json`:
  direct full-OOF extreme calibration on ID155 scored `0.538161`, worse than the
  ID155/ID136 tempered route. This confirms that the current low public-structure
  path is the ID136 blend ladder, not recalibrating every new feature-similarity
  base from scratch.
- `reports/strict_feature_similarity_grid_id155_q1s1s3.json`: a narrow residual
  grid over ID155 found small further gains on Q1, S1, and S3. The largest gain
  was S3 screen/app similarity (`0.495596 -> 0.494955`); Q1 and S1 gains were
  much smaller but directionally consistent.
- `reports/experiment_158_q1s1s3_sixth_refined_grid_feature_similarity_overlay_cv0.565563_unsubmitted.json`:
  ID158 promotes the Q1/S1/S3 residual-grid winners on top of ID155 and improved
  the non-calibration feature-similarity probe to `0.565563`. The gain is small,
  so this is useful mostly as a cleaner component for downstream probes rather
  than as a robust validation breakthrough.
- `reports/id158_id136_tempered_blend_scan.json` and
  `reports/id158_id136_tempered_blend_fine_scan.json`: replacing ID155 with ID158
  in the ID136 blend ladder improved the target-wise public-structure probe from
  `0.521411` to `0.521199` after a fine scan around the coarse target weights.
- `reports/experiment_159_id158_fine_targetwise_tempered_id136_extreme_public_probe_cv0.521199_unsubmitted.json`:
  ID159 lowered the local public-structure probe to `0.521199`. It uses fine
  target-wise ID158/ID136 weights and is leaderboard-mapping evidence only.
- `reports/strict_feature_similarity_grid_id158_q23s24.json`: a residual grid on
  the remaining ID158 blockers found a useful S2 screen/app similarity gain
  (`0.543065 -> 0.540815`) plus smaller Q2/Q3/S4 improvements.
- `reports/experiment_160_q23s24_seventh_refined_grid_feature_similarity_overlay_cv0.565136_unsubmitted.json`:
  ID160 promotes the ID158 Q2/Q3/S2/S4 residual-grid winners and improved the
  non-calibration feature-similarity ladder to `0.565136`.
- `reports/id160_id136_tempered_blend_scan.json`: replacing ID158 with ID160 in
  the ID136 blend ladder improved the target-wise public-structure probe from
  `0.521199` to `0.521151`, mostly through the S2 clean-weight adjustment.
- `reports/experiment_161_id160_fine_targetwise_tempered_id136_extreme_public_probe_cv0.521151_unsubmitted.json`:
  ID161 lowered the local public-structure probe to `0.521151`. It is still an
  extreme non-nested leaderboard-mapping candidate, not strict validation.
- `reports/strict_feature_similarity_grid_id160_q23s4.json`: a narrower residual
  grid on ID160 found tiny additional Q2/Q3/S4 gains, led by Q2 screen/app
  similarity (`0.611639 -> 0.611376`).
- `reports/experiment_162_q23s4_eighth_refined_grid_feature_similarity_overlay_cv0.565080_unsubmitted.json`:
  ID162 promotes those tiny Q2/Q3/S4 gains and is now the current strongest
  non-calibration feature-similarity probe (`0.565080`). The marginal gain is
  small enough that further KNN-style residual stacking is likely near
  exhaustion.
- `reports/id162_id136_tempered_blend_fine_scan.json` and
  `reports/experiment_163_id162_fine_targetwise_tempered_id136_extreme_public_probe_cv0.521143_unsubmitted.json`:
  ID163 is the current lowest local public-structure probe (`0.521143`). It uses
  fine target-wise ID162/ID136 weights and is the highest-risk
  leaderboard-mapping candidate in the current artifact set; do not treat it as
  strict validation evidence.
- `reports/external_sleep_prior_threshold_diagnostic_id162.json`: external ETRI
  sleep/survey summaries were used as priors for a current-data threshold search
  rather than as extra training rows. Direct current sleep-duration/efficiency
  proxies were weak, but simple proxy rules found useful blended gains on S3
  (`0.494955 -> 0.491595`) and S4 (`0.619903 -> 0.615128`).
- `reports/experiment_164_s234_external_sleep_proxy_threshold_overlay_cv0.563831_unsubmitted.json`:
  ID164 promotes fixed S2/S3/S4 sleep/activity proxy threshold rules on top of
  ID162 and is now the current strongest non-calibration probe (`0.563831`).
  Rule probabilities are fit inside each Subject-Hole training fold, but the
  threshold/config choices are still full-OOF tuned.
- `reports/id164_id136_tempered_blend_fine_scan.json`: blending ID164 with ID136
  worsened the target-wise public-structure probe (`0.521143 -> 0.521334`), so
  no ID165 calibration-blend artifact was promoted from ID164.
- `reports/broad_threshold_rule_grid_id164.json`: a broader numeric-covariate
  threshold scan on ID164 found the largest remaining non-calibration gain on Q3
  via a weekly circadian-light rule (`0.619704 -> 0.613260`), with smaller Q2,
  S2, S3, S4, and Q1 gains.
- `reports/experiment_165_q123s234_broad_threshold_rule_overlay_cv0.562646_unsubmitted.json`:
  ID165 promotes those broad threshold rules on Q1/Q2/Q3/S2/S3/S4 and improves
  the non-calibration ladder to `0.562646`. This is useful signal but even more
  full-OOF tuned than ID164.
- `reports/id165_id136_tempered_blend_fine_scan.json`: blending ID165 with ID136
  worsened the target-wise public-structure probe (`0.521143 -> 0.521498`), so
  no calibration-blend artifact was promoted from ID165.
- `reports/broad_threshold_rule_grid_id165.json` and
  `reports/experiment_166_q3_second_broad_threshold_rule_overlay_cv0.562554_unsubmitted.json`:
  a second residual threshold scan found only a small Q3 wrist-light quantile
  rule (`0.613260 -> 0.612616`). ID166 promotes that rule and lowers the
  non-calibration ladder to `0.562554`.
- `reports/id166_id136_tempered_blend_fine_scan.json`: blending ID166 with ID136
  again worsened the target-wise public-structure probe (`0.521143 -> 0.521508`).
- `reports/shallow_tree_rule_overlay_diagnostic_id166.json` and
  `reports/experiment_167_s12_shallow_tree_rule_overlay_cv0.562280_unsubmitted.json`:
  shallow decision-tree rule overlays helped S1 (`0.490232 -> 0.489164`) and S2
  (`0.539572 -> 0.538720`). ID167 is now the current strongest non-calibration
  local probe (`0.562280`), but tree config selection is still full-OOF tuned.
- `reports/id167_id136_tempered_blend_fine_scan.json`: blending ID167 with ID136
  did not improve the public-structure ladder (`0.521143 -> 0.521504`), leaving
  ID163 as the current lowest public-structure probe.
- `reports/id167_id163_direct_blend_scan.json`: direct blending between the
  strongest non-calibration local artifact (ID167) and the current lowest
  public-structure probe (ID163) selected pure ID163 for every target. This
  confirms that ID163 already dominates the local public-structure mapping lens,
  although it remains non-nested and high-risk.
- `reports/strict_hourly_sequence_overlay_diagnostic_id167.json` and
  `reports/id167_post_ladder_light_diagnostics.json`: after ID167, additional
  light diagnostics did not justify an ID168 promotion. Recent-artifact selector
  scans worsened ID167 unless high-risk public-calibration artifacts were
  included; repeat threshold/tree rules did not improve; a 292-feature
  single-threshold residual scan found only negligible target-loss gains; hourly
  sequence, Q3 GPS routine, Q2 WiFi/BLE environment tokens, and Q2/Q3/S4
  ambience-context overlays either selected ID167 as the base or worsened nested
  validation. This closes the current local shallow-feature search branch.
- `reports/public_temporal_all_targets_grid_id167.json` and
  `reports/public_multi_anchor_grid_id167.json`: after confirming train/test use
  the same ten subjects on non-overlapping date rows, the public-structure lens
  was rerun on top of ID167. Same-subject temporal/KNN anchors improved the
  interleaved within-subject blocked CV to `0.549010`, with the strongest fixed
  settings on Q1/Q2/Q3/S1/S2/S4 and S3 left at the ID167 base.
- `reports/experiment_168_id167_public_multi_anchor_probe_cv0.549010_unsubmitted.json`:
  ID168 promotes those fixed public-structure anchors on top of ID167. It is the
  cleanest current sub-`0.55` public-structure artifact outside the ID136
  extreme-calibration ladder, but it is not strict Subject-Hole comparable.
- `reports/id168_blend_scans.json`: blending ID168 with ID163/ID136 did not beat
  the existing ID163 high-risk public-structure leader. Blending ID168 with ID167
  gave only a tiny target-wise improvement under the same lens, so no separate
  blend artifact was promoted.
- `reports/id168_calibration_diagnostic.json` and
  `reports/experiment_169_id168_fixed_public_calibration_probe_cv0.545627_unsubmitted.json`:
  fixed target-wise calibration on top of ID168 improves the public-structure
  probe to `0.545627` (`Q1` `0.563826`, `Q2` `0.565374`, `Q3` `0.573709`, `S1`
  `0.477483`, `S2` `0.536675`, `S3` `0.490893`, `S4` `0.611429`). This is less
  extreme than the ID136/ID163 ladder but still not independent proof because
  the calibration choices came from a diagnostic on the same base.
- `reports/id168_outer_calibration_selection_diagnostic.json`: choosing
  calibration inside an outer interleaved-within-subject fold worsened ID168
  overall (`0.549010 -> 0.550982` for the narrow pool, `0.551768` for the wide
  pool). This is a warning that the full ID169 calibration score is still
  selection-sensitive. The only targets with directional outer support were
  Q2/Q3/S1.
- `reports/experiment_170_id168_outer_supported_partial_calibration_probe_cv0.546579_unsubmitted.json`:
  ID170 promotes only Q2/Q3/S1 from ID169 and leaves Q1/S2/S3/S4 at ID168. It is
  a more conservative public-structure calibration rung (`0.546579`) and should
  be preferred over ID169 when risk control matters more than the last
  `0.000952` of local probe score.
- `reports/id170_blend_scans.json`: blending ID170/ID169 with the existing
  ID163/ID136 extreme-calibration ladder did not create a worthwhile new
  artifact. The best global ID170/ID163 blend is pure ID163, and the best
  target-wise variant only moves `0.521143 -> 0.521112`, too small to justify a
  new high-risk promoted submission.
- `reports/id170_anchor_feature_meta_model_diagnostic.json`: a small
  public-structure meta-model over base predictions plus 130 same-subject
  anchor/date/subject features failed to beat ID170. All fixed LR/HGB candidates
  worsened the outer interleaved-fold score, and target-wise selection chose the
  base for every target. This closes the "let a small model read all anchors"
  branch for the current base.
- `reports/experiment_171_id163_q40_extreme_public_probe_cv0.493975_unsubmitted.json`:
  ID171 applies full-OOF q40 quantile calibration to all seven ID163 targets and
  becomes the lowest generated public-structure artifact (`0.493975`; target
  losses Q1 `0.497013`, Q2 `0.491685`, Q3 `0.516014`, S1 `0.433412`, S2
  `0.477167`, S3 `0.468456`, S4 `0.574075`). This score is not nested; the same
  OOF rows fit and score the quantile bins.
- `reports/id163_outer_calibration_selection_diagnostic.json`: when calibration
  selection is hidden inside outer interleaved folds, ID163 calibration no longer
  supports the `0.493975` headline. Guarded outer means land around
  `0.524372`–`0.528227`, and the guard often falls back to identity. This still
  suggests the public-structure ladder is under `0.53`, but not that ID171's
  full q40 score is reliable.
- `reports/experiment_172_id163_targetwise_extreme_public_probe_cv0.492762_unsubmitted.json`:
  ID172 improves the ID171 extreme probe by keeping q40 for Q1/Q2/Q3/S1/S2 and
  switching S3/S4 to isotonic. It becomes the lowest generated local
  public-structure artifact (`0.492762`; target losses Q1 `0.497013`, Q2
  `0.491685`, Q3 `0.516014`, S1 `0.433412`, S2 `0.477167`, S3 `0.465115`, S4
  `0.568928`). This is still full-OOF method selection and fitting, not nested
  evidence.
- `reports/id163_targetwise_extreme_outer_calibration_selection_diagnostic.json`:
  the outer calibration-selection diagnostic for ID172 repeats the warning from
  ID171: the headline full-OOF score is much lower than guarded outer estimates
  (`0.524372`–`0.528227`). Treat ID172 as a leaderboard-response probe only.
- `reports/id168_169_170_hourly_sequence_rescan.json`: rerunning the lightweight
  hourly sequence overlay on ID168/ID169/ID170 found no S3 gain, but ID169 S4
  improved with the ET sequence model around blend weight `0.18`
  (`0.611429 -> 0.608814`).
- `reports/experiment_173_id169_s4_hourly_sequence_overlay_cv0.545253_unsubmitted.json`:
  ID173 promotes that S4-only raw hourly-sequence overlay on top of ID169,
  improving the cleaner post-ID167 calibrated branch from `0.545627` to
  `0.545253`. The gain is small and non-strict, but it is a real raw-sequence
  contribution rather than another calibration-only transform.
- `reports/id173_random_conv_sequence_diagnostic.json`,
  `reports/id173_random_conv_sequence_diagnostic_seed43018.json`, and
  `reports/id173_random_conv_sequence_diagnostic_seed43019.json`: deterministic
  ROCKET-style random convolution features over raw daily sensor sequences found
  repeatable small improvements over ID173. The diagnostic uses 44 half-hour
  bins, 34 value channels, 34 presence channels, and 192 random kernels per
  seed. Best single-seed means were `0.543951`, `0.544657`, and `0.544431`.
- `reports/experiment_174_id173_multiseed_random_conv_sequence_overlay_cv0.543431_unsubmitted.json`:
  ID174 promotes the best target-wise random-conv overlays from those three
  diagnostics on top of ID173. It modifies Q1/Q2/Q3/S1/S2/S4, keeps S3 at base,
  and improves the cleaner public-structure branch to `0.543431` (Q1
  `0.558725`, Q2 `0.561752`, Q3 `0.573674`, S1 `0.475861`, S2 `0.536380`, S3
  `0.490893`, S4 `0.606733`). The raw sequence models alone are weak, but small
  blends act as useful residual correctors. This is still target-wise
  full-diagnostic selection, not strict nested evidence.
- `reports/id173_random_conv_sequence_diagnostic_seed43020.json` and
  `reports/id173_random_conv_sequence_diagnostic_seed43021.json`: two additional
  random-conv seeds kept the same pattern alive. Seed `43020` gave a better S1
  residual than ID174's selected seed.
- `reports/experiment_175_id173_multiseed_random_conv_sequence_overlay_v2_cv0.543336_unsubmitted.json`:
  ID175 updates only the S1 random-conv residual from ID174, using seed `43020`
  logistic blend weight `0.10`. It improves the cleaner random-conv branch from
  `0.543431` to `0.543336`; this is a very small local gain and remains
  target-wise full-diagnostic public-structure selection.
- `reports/id173_random_conv_sequence_diagnostic_seed43022.json` through
  `reports/id173_random_conv_sequence_diagnostic_seed43025.json`: extending the
  random-conv seed pool found stronger residuals for Q2, Q3, S2, and S4. Seed
  `43024` was especially useful for Q3/S4, while seed `43023` improved S2 and
  seed `43025` improved Q2.
- `reports/experiment_176_id173_multiseed_random_conv_sequence_overlay_v3_cv0.542270_unsubmitted.json`:
  ID176 promotes the best target-wise random-conv overlays from seeds
  `43017`–`43025` on top of ID173. It lowers the cleaner raw-sequence branch to
  `0.542270` (Q1 `0.558725`, Q2 `0.559258`, Q3 `0.570902`, S1 `0.475198`, S2
  `0.534362`, S3 `0.490893`, S4 `0.606554`). It is a meaningful improvement
  over ID175, but still uses target-wise full-diagnostic selection rather than
  nested proof.
- `reports/id173_random_conv_sequence_diagnostic_seed43026.json` through
  `reports/id173_random_conv_sequence_diagnostic_seed43037.json`, plus
  `reports/id173_random_conv_sequence_diagnostic_nk384_seed43101.json` through
  `reports/id173_random_conv_sequence_diagnostic_nk384_seed43106.json`: a
  Kaggle CPU strict-runner sweep completed on 2026-06-16 and broadened the
  random-conv pool without local memory pressure. The best single report mean
  was seed `43037` at `0.543152`; the best target-wise recombination across all
  27 reports was lower, around `0.540607`, mainly from better Q1, Q3, S1, and
  S4 residuals.
- `reports/experiment_177_id173_multiseed_random_conv_sequence_overlay_v4_cv0.540607_unsubmitted.json`:
  ID177 promotes that broader target-wise random-conv selection on top of ID173.
  It uses Q1 seed `43103` with 384 kernels, Q2 seed `43025` with 192 kernels, Q3
  seed `43102` with 384 kernels, S1 seed `43029`, S2 seed `43023`, keeps S3 at
  base, and uses S4 seed `43030`. The local public-structure score is `0.540607`
  (Q1 `0.554516`, Q2 `0.559258`, Q3 `0.569734`, S1 `0.472216`, S2 `0.534362`,
  S3 `0.490893`, S4 `0.603270`). This is the current strongest cleaner
  raw-sequence branch, but it is still a tuned public-structure probe.
- `reports/id173_random_conv_fine_selection_diagnostic.json`,
  `reports/id173_random_conv_fine_selection_lrwide_diagnostic.json`,
  `reports/id173_random_conv_fine_selection_lrultra_diagnostic.json`, and
  `reports/id173_random_conv_fine_selection_highc_diagnostic.json`: fine
  scanning the ID177 random-conv residual neighborhood showed that stronger LR
  regularization settings kept improving local public-structure loss. The scan
  moved the target-wise recombination from `0.540607` to `0.538638`, but the
  best Q1/Q2/S2 choices sit at high C and the raw residual models have poor
  standalone losses, so this is a higher-variance local probe rather than a
  cleaner validation proof.
- `reports/experiment_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638_unsubmitted.json`:
  ID178 promotes the fine/high-C random-conv residual selection. It uses Q1
  seed `43103`/384 kernels with `lr_c100`, Q2 seed `43025` with `lr_c100`, Q3
  seed `43037` with `lr_c50`, S1 seed `43034` with `lr_c50`, S2 seed `43023`
  with `lr_c100`, keeps S3 at base, and uses S4 seed `43030` with `lr_c0_7`.
  The local public-structure score is `0.538638` (Q1 `0.551916`, Q2 `0.556679`,
  Q3 `0.567090`, S1 `0.471630`, S2 `0.529556`, S3 `0.490893`, S4 `0.602700`).
  This is the current lowest cleaner-branch artifact, with a stronger
  high-variance warning than ID177.

### 3. External ETRI 2018–2020 data

Use external data as priors and feature design material, not direct row
concatenation:

- derive threshold intuition for sleep score, wake duration, wake count, duration
  to sleep, HR min/max/average, snoring/breathing disturbance;
- learn broad monotonic directions for survey variables such as fatigue, stress,
  caffeine, alcohol;
- a 2026-06-16 threshold diagnostic found that current sleep-duration and
  efficiency proxies are weak direct classifiers, but external-prior-inspired
  activity/screen proxy rules can still help when blended lightly with a strong
  base; this produced ID164 without concatenating external rows;
- do not merge external participants into current train/test unless a separate
  leak-free validation note proves benefit.
- the external `[Public_0.5917] LGBM+XGB` notebook reviewed on 2026-06-16 is
  useful as direction only, not as a directly reproducible artifact: it depends
  on missing V152/full-feature artifacts and contains placeholder window-pair
  feature code. Its transferable lesson is to preserve a strong anchor and use
  subject-aware, stability-controlled blending rather than chasing raw local
  minima.

### 4. Validation repair

Keep Subject-Hole CV, but add a second reliability lens:

- maintain per-target CV and per-target Public-gap history;
- penalize candidates whose prediction variance or target means resemble known
  overfit probes;
- report a “public-adjusted expected score” using the empirical `+0.02–0.03`
  optimism range until a better validation/public mapping is established.

### 5. Modeling order

1. If the user explicitly approves a DACON probe, choose among risk rungs:
   `id=142` if the priority is strict-validation integrity, `id=145` if the
   priority is a locally tuned Q2/Q3/S4 feature-similarity leaderboard probe,
   `id=167` if the priority is the strongest non-calibration probe,
   `id=166`/`id=165`/`id=164`/`id=162`/`id=160`/`id=158`/`id=155`/`id=153`/`id=151`/`id=148`/`id=147`/`id=146`
   if an earlier less-tuned non-calibration rung is desired,
   `id=168` if the priority is a post-ID167 public-structure sub-`0.55` probe
   without the ID136 extreme calibration ladder, `id=169` if the priority is a
   lower calibrated version of that post-ID167 public-structure branch, `id=170`
   if the priority is a more conservative outer-supported subset of ID169,
   `id=173` if the priority is the latest cleaner post-ID167 branch with a small
   raw hourly-sequence S4 contribution, `id=174` if the priority is the strongest
   cleaner post-ID167 branch with random-conv raw sequence residuals, `id=175`
  if an earlier tiny refinement of that branch is desired, `id=176` if a
   lower 192-kernel-only random-conv branch is desired, `id=177` if a broader
   but still moderate random-conv branch is desired, `id=178` if the priority is
   the current lowest cleaner-branch local score and the user accepts higher
   high-C residual-selection risk, `id=218` if the priority is the latest
   cleaner multi-signal branch, `id=219` if the priority is the preferred
   moderate-low ID172 bridge, `id=217` if the priority is the lower raw-local
   delta08 bridge, or `id=220` if the priority is the slightly better
   robustness-score delta08 bridge, `id=235` if the priority is the more
   balanced post-ID226 calibration card, `id=247` if the priority is the current
   best safer target-wise delta-capped bridge, `id=248` if the priority is a
   stronger mid-risk local improvement, `id=252` if an ultra-frontier stress
   probe is desired with somewhat less discrete provenance than ID250/ID251,
   `id=244` if a more conservative bridge is desired, `id=246` if a slightly
   less S4-isotonic alternative to ID244 is desired, `id=245` only as a bolder
   uniform bridge, `id=249` only as a high-risk target-wise frontier bridge,
   `id=250`/`id=251` only as raw-local ultra-frontier stress tests, and
   `id=242`/`id=243` only as extreme local-frontier stress tests,
   `id=131` if the
   priority is cleaner-anchor public-structure risk control, `id=133`/`id=134`
   if the priority is a lower public-structure score with moderate-to-high ID126
   exposure, `id=156`/`id=159`/`id=161`/`id=163` only as bold ID136-blend leaderboard mapping
   probes, `id=171`/`id=172` only as extreme q40/isotonic
   leaderboard-response probes, or `id=136` as the underlying extreme
   calibration reference.
   `id=127` remains a useful older comparison, and `id=126` is the original
   extreme calibrator.
2. Keep ID122, ID130, ID132, ID125, ID128, ID131, ID127, ID133, ID134, ID135, ID136, and ID126 together as a calibration ladder:
   cleaner fixed anchor (`0.560861`) -> GPS-base cleaner anchor (`0.560757`) ->
   refined GPS-base cleaner anchor (`0.560259`) -> high-risk target calibration
   (`0.557413`) -> cleaner-anchor tempered sub-`0.55` blend (`0.549956`) ->
   GPS-base cleaner tempered blend (`0.549876`) -> tempered sub-`0.55` blend
   (`0.549454`) -> refined-anchor stronger tempered blend (`0.545875`) ->
   balanced refined-anchor blend (`0.540925`) -> aggressive refined-anchor blend
   (`0.535430`) -> double-calibrated refined-anchor extreme (`0.523144`) ->
   ID148-tempered extreme (`0.522802`) -> target-wise ID148-tempered extreme
   (`0.521895`) -> target-wise ID151-tempered extreme (`0.521641`) ->
   target-wise ID153-tempered extreme (`0.521538`) -> target-wise
   ID155-tempered extreme (`0.521411`) ->
   target-wise ID158-tempered extreme (`0.521199`) ->
   target-wise ID160-tempered extreme (`0.521151`) ->
   target-wise ID162-tempered extreme (`0.521143`) ->
   original extreme calibration (`0.529691`).
   Keep ID168 (`0.549010`), ID169 (`0.545627`), and ID170 (`0.546579`) as a
   separate post-ID167 branch: they are cleaner than the ID136-dominated ladder
   in calibration provenance, but still public-structure/non-strict.
   Keep ID171 (`0.493975`) and ID172 (`0.492762`) outside that ladder as special
   extreme q40/isotonic probes; their outer calibration diagnostics are much
   weaker (`0.524372`–`0.528227`). ID173 (`0.545253`) belongs to the cleaner
   post-ID167 branch, not the ID163 extreme branch; ID174 (`0.543431`) is the
   strong member of that cleaner raw-sequence branch, with ID178 (`0.538638`) as
   the current lowest cleaner random-conv successor and ID177 (`0.540607`) as
   the less aggressive variant. ID215 (`0.534135`) is now the current lowest
   cleaner shapelet/motif successor, while ID218 (`0.534103`) is the latest
   cleaner multi-signal successor. ID219 (`0.522874`) is the current
   moderate-low ID172 bridge, with ID217 (`0.519556`) and ID220 (`0.519561`) as
   the delta08 risk references.
   ID235 (`0.482255`) is the balanced post-ID226 calibration card. ID238
   (`0.462365`) opened the aggressive clean-anchor/logit oracle frontier, and
   ID242 (`0.357651`) / ID243 (`0.360420`) are extreme q240 recalibration stress
   tests, not public proof. ID244 (`0.437262`) blends ID235 with ID242 at alpha
   `0.225` and is the current best risk-aware next public probe; ID245
   (`0.415035`) is the bolder alpha `0.375` bridge, while ID246 (`0.438034`) is
   the ID243-based alpha `0.225` alternative. ID247 (`0.415523`) improves the
   safer region using target-wise prob/logit bridge choices under max test delta
   `0.20` versus ID235. ID248 (`0.391735`) extends the same idea to cap `0.30`
   and is the current best mid-risk local improvement. ID249 (`0.374465`) uses
   cap `0.40`; it is useful for mapping the frontier but too aggressive to call
   safe. A full scan of existing base OOF artifacts found that S3 does not
   improve below `0.465115`; the next local gains therefore come from non-S3
   calibration only. ID250 (`0.261184`) applies isotonic second-stage
   calibration to ID242 for Q1/Q2/Q3/S1/S2 while keeping S3/S4 unchanged and is
   the new raw local minimum. ID251 (`0.263237`) is the quantile-only version of
   that ultra-frontier idea. ID252 (`0.274420`) applies a mixed second-stage
   calibration to ID249 and is the sanest ultra-frontier stress card, but all
   three are full-OOF/non-nested and should not be treated as public proof.
   A post-ID252 S3 hard-wall review then checked all base OOF artifacts, S3
   isotonic/quantile/temperature calibration, strict hourly-sequence overlays on
   ID247/ID249, and processed HR/BLE/screen/demo compact side models. None
   improved S3 below `0.465115`; the compact side models all selected blend
   weight `0`. Save this as a negative result: do not keep repeating the same
   S3 local feature families.
   A real DACON score from this ladder is the fastest way to learn which
   validation lens maps to Public.
3. Continue strict Subject-Hole work only where it adds a genuinely new signal:
   target-specific app/context/GPS-routine features, same-subject temporal-hole
   anchors, S3/S4 sequence features, and raw sensor summaries were all useful in
   earlier rungs, but the ID167 post-ladder diagnostics show that simply
   re-running these shallow families on the latest base is now exhausted. Use
   ID167 as the default non-calibration base for local diagnostics, ID163/ID136
   only for explicit public-structure calibration probes, and ID129/ID142 only
   when an older baseline is needed to reproduce a specific diagnostic. The next
   strict improvement likely needs a genuinely different representation rather
   than more KNN/threshold/hourly/context overlays.
4. Use Kaggle for heavier Pixleep-style sensor-to-image/sequence models and
   larger seed/feature ablation batches after local audit passes.
   A CPU-only strict runner now lives in `notebooks/kaggle_strict_runner/`, with
   `make kaggle-push-strict-runner` and `make kaggle-pull-strict-output`. Keep
   local work to light diagnostics, lint/audit, and artifact promotion review.
   The runner has already completed one Chronos-T5 mini embedding diagnostic on
   ID129 without nested improvement and one broader ID173 random-conv sweep that
   produced ID177. Inspect and change the runner command before each push; the
   next heavy attempt should use a genuinely different S3 representation:
   per-subject multichannel minute grids with ROCKET/MiniRocket-style random
   convolutions, a small 1D CNN embedding, or another self-supervised time-series
   embedding. Do not simply rerun HR/BLE/screen/hourly compact overlays locally.
5. DACON submission remains blocked until explicit user approval, even when a
   generated artifact is auditable and below the public-structure `0.55` line.

## Success criteria

- No new experiment without matching notebook/CSV/log row.
- No duplicate filenames for new artifacts.
- No public-score claim without a real DACON result.
- Public `0.55` is not considered achieved until DACON returns a real score.
  ID168-ID252 plus ID133/ID134/ID135/ID136/ID127/ID126/ID159/ID161/ID163 only
  satisfy non-strict public-structure probe targets.

## 2026-06-16 post-ID271 update

ID270 and ID271 were added as controlled risk-penalized pair-blend stress follow-ups after ID267-ID269. Their artifact contract is complete: each promoted candidate has a unique submission CSV, history notebook, submission log row, metadata JSON, and base OOF/test prediction pair.

Current practical ladder:

| role | id | local CV | robustness | max delta vs ID178 | note |
|---|---:|---:|---:|---:|---|
| clean first probe | 264 | 0.464758 | 0.494762 | 0.119995 | clean cap0.12 option; no direct stress-source exposure |
| sharper first probe | 267 | 0.461960 | 0.492124 | 0.119969 | best cap0.12 local, but borrows limited stress-source signal |
| balanced Public-0.55 chase | 268 | 0.448976 | 0.478923 | 0.149985 | best current balance between local gain and movement cap |
| cap0.16 stress follow-up | 270 | 0.440505 | 0.470154 | 0.159931 | stronger than ID269/261 locally, but stress-heavy |
| low-local stress card | 271 | 0.435576 | 0.466134 | 0.176195 | locally tempting, but heavily inherits ID257-style full-OOF oracle behavior |

Recommendation after the ID178 and ID085 public gaps: do not treat lower local CV alone as progress. The next DACON feedback should map one controlled candidate family first. If the goal is to protect submission budget, choose ID264 or ID267. If the goal is to chase Public 0.55 with meaningful but still bounded movement, choose ID268. Hold ID270/ID271 until at least one safer cap0.12/cap0.15 member proves that this family’s local-to-public gap is not catastrophically larger than ID178’s `+0.0627733009` gap.

Supporting reports:

- `reports/post_id271_controlled_risk_pair_blend_robustness_20260616.json`
- `reports/post_id271_controlled_risk_pair_blend_robustness_20260616.md`
- `reports/candidate_tier_manifest_post_id271_20260616.json`
- `reports/candidate_tier_manifest_post_id271_20260616.md`

No DACON submission was made while creating this update.

## 2026-06-16 post-ID274 update

ID272-ID274 extend the local frontier after two new diagnostics:

- `reports/post271_recursive_pair_blend_risk_scan_20260616.json/.md`
- `reports/post271_ultra_gated_pair_blend_scan_20260616.json/.md`

Promoted candidates:

| role | id | local CV | robustness | max delta vs ID178 | note |
|---|---:|---:|---:|---:|---|
| balanced cap0.15 update | 272 | 0.448575 | 0.478539 | 0.149985 | slightly improves ID268 while avoiding recursive ID267-ID271 ingredients in selected configs |
| tempered ultra stress | 273 | 0.433773 | 0.463138 | 0.179612 | lower than ID271, but uses ID249/ID252 ultra-frontier ingredients |
| promoted local frontier | 274 | 0.423144 | 0.452672 | 0.199402 | current promoted local frontier; useful as a map, not public proof |

Recommendation: ID272 now replaces ID268 as the best balanced local candidate. ID273 is the saner stress probe below ID271 if the next feedback allows more risk. ID274 demonstrates that the current artifact family can reach local `0.423144`, but it relies directly on ID252-style ultra-frontier behavior and should not be mistaken for a trustworthy Public `0.55` claim.

Supporting reports:

- `reports/post_id274_recursive_ultra_pair_blend_robustness_20260616.json`
- `reports/post_id274_recursive_ultra_pair_blend_robustness_20260616.md`
- `reports/candidate_tier_manifest_post_id274_20260616.json`
- `reports/candidate_tier_manifest_post_id274_20260616.md`

No DACON submission was made while creating this update.

## 2026-06-16 post-ID278 update

ID275-ID278 were promoted from the post-ID274 refined recursive scan:

- `reports/post274_refined_recursive_pair_blend_scan_20260616.json`
- `reports/post274_refined_recursive_pair_blend_scan_20260616.md`

Promoted candidates:

| role | id | local CV | robustness | max delta vs ID178 | note |
|---|---:|---:|---:|---:|---|
| refined cap0.22 bridge | 275 | 0.416609 | 0.446035 | 0.219213 | lower than ID274 with moderate additional movement |
| smoother cap0.26 frontier | 276 | 0.411217 | 0.439765 | 0.256619 | p0.010 penalty version with lower mean source risk than raw/p0.005 |
| first promoted sub-0.40 | 277 | 0.399355 | 0.427818 | 0.276333 | proves promoted local sub-0.40 is reachable |
| current promoted local frontier | 278 | 0.389120 | 0.418446 | 0.298720 | current lowest promoted local, but high-risk and non-nested |

Recommendation: ID272 remains the most plausible public-transfer candidate in this family. ID275/ID276 are the bridge cards if we decide to accept more movement for stronger local. ID277 and ID278 are frontier mapping cards: valuable for understanding the local landscape, but not credible Public `0.55` proof without DACON feedback. The local frontier is no longer the immediate problem; local-to-public calibration risk is now the main problem.

Supporting reports:

- `reports/post_id278_refined_frontier_pair_blend_robustness_20260616.json`
- `reports/post_id278_refined_frontier_pair_blend_robustness_20260616.md`
- `reports/candidate_tier_manifest_post_id278_20260616.json`
- `reports/candidate_tier_manifest_post_id278_20260616.md`

No DACON submission was made while creating this update.

## 2026-06-16 post-ID282 update

ID279-ID282 were promoted from the public-gap-aware shrinkage scan:

- `reports/post278_public_gap_aware_shrinkage_scan_20260616.json`
- `reports/post278_public_gap_aware_shrinkage_scan_20260616.md`

The scan uses a diagnostic proxy, not a public-score claim:

`overall + ID178 public gap + 0.20 * max(0, max_delta_all - 0.15) + 0.005 * mean_source_risk`

Promoted candidates:

| role | id | local CV | robustness | max delta vs ID178 | note |
|---|---:|---:|---:|---:|---|
| gap-aware cap0.22 bridge | 279 | 0.414169 | 0.443291 | 0.219676 | improves over ID275 at similar movement |
| gap-aware cap0.24 bridge | 280 | 0.406704 | 0.435637 | 0.239676 | strong middle bridge between ID279 and ID281 |
| gap-aware cap0.26 near-sub0.40 | 281 | 0.399583 | 0.428407 | 0.259827 | near sub-0.40 with lower max delta than ID277 |
| gap-aware cap0.28 frontier alternative | 282 | 0.393400 | 0.422599 | 0.279793 | lower movement than ID278, slightly higher local |

Recommendation: ID272 remains the best public-transfer lead. ID279-ID281 are now the preferred bridge ladder if we want to move beyond ID272 while keeping movement more controlled than the raw frontier ladder. ID282 is a better-behaved alternative to ID278, but still belongs to the stress/frontier family. None of these is public proof without a real DACON result.

Supporting reports:

- `reports/post_id282_gapaware_frontier_blend_robustness_20260616.json`
- `reports/post_id282_gapaware_frontier_blend_robustness_20260616.md`
- `reports/candidate_tier_manifest_post_id282_20260616.json`
- `reports/candidate_tier_manifest_post_id282_20260616.md`

No DACON submission was made while creating this update.

## 2026-06-16 post-ID285 second-stage calibration update

ID283-ID285 were promoted from the post-ID282 second-stage calibration scan:

- `reports/post282_second_stage_calibration_scan_20260616.json`
- `reports/post282_second_stage_calibration_scan_20260616.md`

These candidates use full-OOF second-stage calibration, so the local scores are useful for mapping the frontier but remain high-risk and non-nested. The confirmed ID178 feedback is still the main public-gap anchor: local `0.538638` -> Public `0.6014113009`, gap `+0.062773`.

Promoted candidates:

| role | id | base | local CV | score-pressure proxy | stability-veto proxy | max delta vs ID178 | note |
|---|---:|---:|---:|---:|---:|---:|---|
| calibrated public-transfer lead | 283 | 272 | 0.440206 | 0.502979 | 0.523259 | 0.200699 | smallest calibrated movement profile; best calibrated lead if we want safer public transfer |
| calibrated score-pressure card | 284 | 280 | 0.385005 | 0.447778 | 0.579899 | 0.409299 | much stronger local, but large movement and isotonic full-OOF risk |
| calibrated local frontier stress | 285 | 278 | 0.356330 | 0.419103 | 0.578461 | 0.474428 | lowest promoted local so far, but too much movement to treat as public proof |

Recommendation: ID283 now joins ID272 as the first-tier public-transfer pair. ID284 is the more plausible score-pressure calibration probe if we decide to accept second-stage calibration risk. ID285 proves the local public-structure frontier can be pushed to `0.356330`, but it should be treated as a diagnostic/stress artifact rather than the next safe submission.

A public-score-response diagnostic was also created. It confirms that scalar leaderboard scores are too underdetermined to reconstruct public labels; the practical use is a veto layer based on movement/sharpness, not direct leaderboard probing.

Supporting reports:

- `reports/post_id285_second_stage_calibration_robustness_20260616.json`
- `reports/post_id285_second_stage_calibration_robustness_20260616.md`
- `reports/candidate_tier_manifest_post_id285_20260616.json`
- `reports/candidate_tier_manifest_post_id285_20260616.md`
- `reports/public_score_response_diagnostic_post_id285_20260616.json`
- `reports/public_score_response_diagnostic_post_id285_20260616.md`

No DACON submission was made while creating this update.

## 2026-06-16 post-ID288 fixed-temperature update

A fixed temperature/shift-only scan was added after ID285:

- `reports/post285_temperature_only_fixed_scan_20260616.json`
- `reports/post285_temperature_only_fixed_scan_20260616.md`

This scan intentionally avoids fitted isotonic/quantile calibration. It still uses target-wise full-OOF selection, so it is not public proof, but the promoted artifacts are less calibration-heavy than ID284/ID285.

Promoted candidates:

| role | id | base | local CV | score-pressure proxy | stability-veto proxy | max delta vs ID178 | max delta vs base | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fixed-temp safer bridge | 286 | 280 | 0.395368 | 0.458141 | 0.496510 | 0.259624 | 0.041869 | conservative fixed transform; improves ID280 without fitted calibration |
| fixed-temp balanced lead | 287 | 282 | 0.378934 | 0.441707 | 0.493668 | 0.298460 | 0.049781 | strongest non-fitted score-pressure card so far |
| fixed-temp frontier stress | 288 | 278 | 0.372773 | 0.435546 | 0.498594 | 0.330135 | 0.057850 | local frontier without fitted calibration, but above the preferred ID178 movement band |

Recommendation: ID287 now becomes the leading no-fitted-calibrator score-pressure candidate. ID286 is the safer fixed-temp bridge if movement control is prioritized. ID288 is useful because it gets close to the frontier without isotonic/quantile fitting, but its ID178 movement is high enough that it remains a stress probe.

Updated inspection order without new DACON feedback: ID287, ID286, ID283, ID272, ID279, ID280, ID281, ID282, ID288, ID278, ID284, ID285.

Supporting reports:

- `reports/post_id288_fixedtemp_robustness_20260616.json`
- `reports/post_id288_fixedtemp_robustness_20260616.md`
- `reports/candidate_tier_manifest_post_id288_20260616.json`
- `reports/candidate_tier_manifest_post_id288_20260616.md`

No DACON submission was made while creating this update.

## 2026-06-16 ID287 public failure and post-ID291 recovery update

ID287 was submitted to DACON as submission `1465470` on `2026-06-16 19:08:14` and scored Public `0.6698787318`:

- file: `submission_id_287_post285_base282_fixedtemp_balanced_cv0.378934_unsubmitted.csv`
- local CV: `0.378934`
- Public LB: `0.6698787318`
- gap: `+0.290945`

This is a decisive failure of the fixed-temperature local-frontier strategy. Very low full-OOF local scores are now treated as suspicious unless the prediction movement from confirmed Public anchor ID178 is tightly capped.

Two post-failure scans were created:

- `reports/post287_failure_confirmed_anchor_bridge_scan_20260616.json/.md`
- `reports/post287_failure_clean_anchor_bridge_scan_20260616.json/.md`

The clean scan excludes submitted-failed ID287 and frontier-stress ID288, then moves from confirmed Public anchor ID178 toward cleaner bridge candidates only.

Promoted recovery candidates:

| role | id | cap vs ID178 | local CV | optimistic ID178-gap proxy | post-failure risk proxy | max delta vs ID178 | note |
|---|---:|---:|---:|---:|---:|---:|---|
| safest recovery probe | 289 | 0.030 | 0.517635 | 0.580408 | 0.580408 | 0.029698 | first choice if submitting again after ID287 failure |
| balanced recovery probe | 290 | 0.050 | 0.504921 | 0.567694 | 0.583507 | 0.049218 | only if accepting more risk for score pressure |
| score-pressure recovery probe | 291 | 0.060 | 0.498891 | 0.561664 | 0.586086 | 0.059485 | should wait until safer feedback arrives |

Recommendation changed: ID287, ID288, and the larger fixed-temp/frontier family are demoted. If another submission is made immediately, ID289 is the correct public-information probe. ID290 is the more aggressive alternative, but after Public `0.6698787318`, movement control is more important than local score.

Supporting reports:

- `reports/dacon_submission_result_id287_public0.6698787318.json`
- `reports/post_id291_post287_failure_anchor_recovery_robustness_20260616.json`
- `reports/post_id291_post287_failure_anchor_recovery_robustness_20260616.md`
- `reports/public_score_response_diagnostic_post_id287_20260616.json`
- `reports/public_score_response_diagnostic_post_id287_20260616.md`
- `reports/candidate_tier_manifest_post_id291_after_id287_public_failure_20260616.json`
- `reports/candidate_tier_manifest_post_id291_after_id287_public_failure_20260616.md`

No DACON submission was made for IDs 289-291 while creating this update.

## 2026-06-16 post-ID294 anti-failed recovery update

After ID287 scored Public `0.6698787318`, two anti-failure scans were added:

- `reports/post287_anti_failure_anchor_blend_scan_20260616.json/.md`
- `reports/post287_recovery_anti_failed_blend_scan_20260616.json/.md`

The idea is not to chase lower local CV. Instead, it preserves the clean ID178-anchor recovery move while applying a tiny negative blend away from the public-failed ID287 direction.

Promoted anti-failed recovery candidates:

| role | id | local CV | ID178-gap proxy | post-failure proxy | max delta vs ID178 | note |
|---|---:|---:|---:|---:|---:|---|
| ultra-safe anti-failed recovery | 292 | 0.518466 | 0.581239 | 0.581239 | 0.028837 | ID289-like movement, slightly farther from failed ID287 |
| balanced anti-failed recovery | 293 | 0.510898 | 0.573671 | 0.585579 | 0.039357 | best balanced anti-failed card from ID290 base |
| score anti-failed recovery | 294 | 0.507898 | 0.570671 | 0.589135 | 0.044288 | more score-oriented, still less aggressive than clean ID290 by failure-aware proxy |

Recommendation: ID292 is now the safest next information probe if another submission is made, because it stays inside the ID289 movement band while hedging away from the failed ID287 vector. ID293 is the balanced alternative; ID294 is the score-pressure alternative. ID290/291 remain useful references but should not lead immediately after the ID287 failure.

Supporting reports:

- `reports/post_id294_antifailed_recovery_robustness_20260616.json`
- `reports/post_id294_antifailed_recovery_robustness_20260616.md`
- `reports/candidate_tier_manifest_post_id294_antifailed_recovery_20260616.json`
- `reports/candidate_tier_manifest_post_id294_antifailed_recovery_20260616.md`

No DACON submission was made for IDs 292-294 while creating this update.

## 2026-06-16 post-ID297 wide anti-vector update

A wider direct anti-vector scan was added:

- `reports/post287_wide_anchor_antivector_scan_20260616.json`
- `reports/post287_wide_anchor_antivector_scan_20260616.md`

This scan moves directly from confirmed Public anchor ID178 away from public-failed ID287. It is not a local-score-improvement strategy; it deliberately sacrifices local CV to test whether the public distribution lies opposite the ID287 failure direction.

Promoted direct anti-vector candidates:

| role | id | local CV | ID178-gap proxy | max delta vs ID178 | note |
|---|---:|---:|---:|---:|---|
| mild direct anti-vector | 295 | 0.548583 | 0.611356 | 0.014923 | cleanest pure anti-ID287 information probe |
| stronger direct anti-vector | 296 | 0.558807 | 0.621580 | 0.029846 | stronger opposite-of-ID287 test, but local is worse |
| mild logit anti-vector | 297 | 0.550861 | 0.613634 | 0.018164 | logit-space alternative to ID295 |

Recommendation: ID292 remains the next score-recovery candidate. ID295 is a different kind of card: a direct public-response information probe that could reveal whether moving opposite ID287 improves Public relative to ID178. ID296 is stronger but should only be used if we explicitly want to test that hypothesis. ID295/296 should not be judged by local CV alone.

Supporting reports:

- `reports/post_id297_wide_antivector_robustness_20260616.json`
- `reports/post_id297_wide_antivector_robustness_20260616.md`
- `reports/candidate_tier_manifest_post_id297_wide_antivector_20260616.json`
- `reports/candidate_tier_manifest_post_id297_wide_antivector_20260616.md`

No DACON submission was made for IDs 295-297 while creating this update.

## 2026-06-16 post-ID300 partial-target recovery update

A partial-target recovery scan was added:

- `reports/post297_partial_target_recovery_scan_20260616.json`
- `reports/post297_partial_target_recovery_scan_20260616.md`

This scan keeps ID178 as the confirmed Public anchor and swaps only selected targets from recovery candidates. It helps diagnose whether the ID287 public failure is concentrated in Q targets, S targets, or specific sleep metrics.

Promoted partial-target candidates:

| role | id | selected targets | local CV | ID178-gap proxy | max delta vs ID178 | note |
|---|---:|---|---:|---:|---:|---|
| Q-only diagnostic | 298 | Q1,Q2,Q3 | 0.527647 | 0.590420 | 0.029697 | isolates subjective/survey target movement |
| S-only diagnostic | 299 | S1,S2,S3,S4 | 0.528626 | 0.591399 | 0.029698 | isolates objective sleep metric movement |
| no-S3 balanced partial | 300 | Q1,Q2,Q3,S1,S2,S4 | 0.511734 | 0.574507 | 0.039357 | stronger recovery card while leaving S3 anchored |

Recommendation: ID292 remains the full recovery lead if the next submission should maximize score recovery while staying near ID178. ID298/ID299 are the clean diagnostic pair for Q-vs-S public-risk decomposition. ID300 is useful if S3 is suspected to be unstable: it keeps S3 at ID178 while moving the other six targets.

Supporting reports:

- `reports/post_id300_partial_target_recovery_robustness_20260616.json`
- `reports/post_id300_partial_target_recovery_robustness_20260616.md`
- `reports/candidate_tier_manifest_post_id300_partial_target_20260616.json`
- `reports/candidate_tier_manifest_post_id300_partial_target_20260616.md`

No DACON submission was made for IDs 298-300 while creating this update.

## 2026-06-16 post-ID333 fourth-pass constrained-combo update

A fourth-pass constrained-combo scan was added:

- `reports/post328_failed_direction_constrained_combo_fourthpass_scan_20260616.json/.md`

Promoted fourth-pass candidates:

| role | id | local CV | ID178-gap proxy | max delta vs ID178 | failed projection | note |
|---|---:|---:|---:|---:|---:|---|
| practical | 329 | 0.516718 | 0.579492 | 0.079192 | +0.0499 | narrowly supersedes ID323 |
| tight score-pressure | 330 | 0.511769 | 0.574543 | 0.113916 | +0.0499 | stronger but bolder |
| scoreguard | 331 | 0.509930 | 0.572703 | 0.149464 | +0.0498 | high movement |
| wide diagnostic | 332 | 0.503202 | 0.565975 | 0.227831 | +0.0499 | not first probe |
| frontier stress | 333 | 0.466963 | 0.529736 | 0.317996 | +0.2693 | target-zone proxy but public-risky |

Recommendation: ID329 is the current practical candidate if another DACON public probe is approved. ID330/ID331 are bolder alternatives. ID332/ID333 should wait for public feedback because ID287 proved that local frontier pressure can fail badly on Public.

Supporting reports:

- `reports/candidate_tier_manifest_post_id333_fourthpass_combo_20260616.json`
- `reports/candidate_tier_manifest_post_id333_fourthpass_combo_20260616.md`

No DACON submission was made for IDs 329-333 while creating this update.

## 2026-06-17 ID329 public result update

ID329 was submitted and scored:

- DACON submission id: `1465639`
- File: `submission_id_329_post328_combo4_cap080_budget050_practical_cv0.516718_unsubmitted.csv`
- Submitted at: `2026-06-17 00:45:07`
- Local CV: `0.516718`
- Public LB: `0.6017908577`
- Gap: `+0.085073`

This is an important negative signal. ID329 was safe relative to the ID287 collapse, but it did not beat ID178 (`0.6014113009`). The constrained-combo budget-0.05 direction should now be treated as public-neutral/slightly negative unless a new orthogonal rationale appears.

Supporting reports:

- `reports/dacon_submission_result_id329_public0.6017908577.json`
- `reports/public_score_response_diagnostic_post_id329_20260617.json`
- `reports/public_score_response_diagnostic_post_id329_20260617.md`

## 2026-06-17/18 post-ID335 Kaggle sequence-source runner update

A dedicated report-only Kaggle runner was prepared for the S1 random-conv
sequence-source lane:

- Runner: `notebooks/kaggle_sequence_source_runner/Sequence_Source_Post_ID335_Runner.ipynb`
- Metadata: `notebooks/kaggle_sequence_source_runner/kernel-metadata.json`
- Push target: `make kaggle-push-sequence-source-runner`
- Pull target: `make kaggle-pull-sequence-source-output`

The runner executed six ID173-based S1 random-conv diagnostics with
512/768/1024 kernels on Kaggle, completed successfully, and staged only JSON
reports plus a manifest. The best result was `nk512_seed45502`: ID173 base mean
`0.545253`, targetwise best mean `0.542888`, improvement `+0.002365`. The main
source signal was S1 (`+0.010855`), with smaller Q1/S4 signals.

This remains source-generation evidence only. It does not beat the confirmed
ID178/ID190 random-conv branch as a whole, and it does not create or authorize a
DACON submission, submission-log row, promoted notebook, or base-prediction
artifact.

Supporting reports:

- `reports/kaggle_sequence_source_runner_post_id335_20260617.json`
- `reports/kaggle_sequence_source_runner_post_id335_20260617.md`
- `reports/kaggle_sequence_source_remote_summary_post_id335_20260617.json`
- `reports/kaggle_sequence_source_remote_summary_post_id335_20260617.md`

## 2026-06-18 post-ID335 sequence anchor probe update

The S1 random-conv seed found by the Kaggle source runner (`nk512_seed45502`)
was replayed report-only on the stronger sequence anchors ID178/179/190/191:

- Runner: `notebooks/kaggle_sequence_anchor_probe_runner/Sequence_Anchor_Probe_Post_ID335_Runner.ipynb`
- Push target: `make kaggle-push-sequence-anchor-probe-runner`
- Pull target: `make kaggle-pull-sequence-anchor-probe-output`

The S1 signal survived every replay base. S1-only estimated means:

| base | base mean | S1 improvement | S1-only estimated mean |
|---:|---:|---:|---:|
| 178 | `0.538638` | `+0.006820` | `0.537664` |
| 179 | `0.538290` | `+0.005051` | `0.537568` |
| 190 | `0.537956` | `+0.001385` | `0.537758` |
| 191 | `0.538009` | `+0.002306` | `0.537680` |

This is the first useful post-ID335 source-generation lead: an S1-only
candidate is now worth constructing if promotion is approved. However, it is
still public-structure evidence, not Public `0.58` evidence. Do not submit or
promote automatically.

Supporting reports:

- `reports/kaggle_sequence_anchor_probe_remote_summary_post_id335_20260618.json`
- `reports/kaggle_sequence_anchor_probe_remote_summary_post_id335_20260618.md`
- `reports/sequence_anchor_probe_decision_post_id335_20260618.json`
- `reports/sequence_anchor_probe_decision_post_id335_20260618.md`

## 2026-06-18 ID336 S1-only controlled artifact

ID336 was generated as the first post-anchor-probe artifact:

- File: `submission_id_336_id178_s1_seed45502_randomconv_overlay_cv0.537663_unsubmitted.csv`
- Local CV: `0.537663`
- Base: ID178 confirmed public anchor
- Change: S1 only, random-conv `nk512 seed45502`, `lr_c0_1`, weight `0.15`
- DACON submission: no

The artifact contract is complete: unique submission CSV, history notebook,
metadata report, decision note, base OOF, base test, and submission-log row all
exist for ID336. Non-S1 targets are exactly identical to ID178 in both OOF and
test predictions.

Gate summary:

- Local gain vs ID178: `+0.000974`
- S1 loss improvement: `+0.006820`
- Mean Δ vs ID178: `0.004008`
- Max Δ vs ID178: `0.112756`
- Failed-direction projection: `-0.001662`
- Rerank tier: `reviewable_low_movement`
- Naive ID178-gap projected public: `0.600436`

Interpretation: ID336 is cleaner than the broad post-ID287 combo attempts
because it modifies one target on top of a known public anchor and does not
project positively into the ID287 failed direction. It is still not Public
`0.58` evidence and must not be submitted automatically.

Supporting reports:

- `reports/experiment_336_id178_s1_seed45502_randomconv_overlay_cv0.537663_unsubmitted.json`
- `reports/decision_336_id178_s1_seed45502_randomconv_overlay_cv0.537663_unsubmitted.md`
- `reports/id336_s1_only_public_risk_gate_20260618.json`
- `reports/id336_s1_only_public_risk_gate_20260618.md`
- `reports/post_id336_public_anchor_rerank_20260618.json`
- `reports/post_id336_public_anchor_rerank_20260618.md`

## 2026-06-18 ID336 S1 weight sensitivity

A report-only weight sweep was run for the same ID178 + S1 `seed45502`
construction:

- Report: `reports/id336_s1_weight_sensitivity_20260618.json/.md`
- Weights: `0.05`, `0.07`, `0.10`, `0.12`, `0.15`, `0.18`, `0.20`
- Submission artifact: none
- DACON submission: no

Result:

| pick | weight | local CV | local gain vs ID178 | max Δ178 | conservative proxy |
|---|---:|---:|---:|---:|---:|
| best local / ID336 | `0.15` | `0.537663` | `+0.000974` | `0.112756` | `0.607945` |
| best proxy / balanced | `0.05` | `0.538066` | `+0.000571` | `0.037585` | `0.605022` |

Interpretation: ID336's `0.15` weight remains the best local choice, but the
public-risk proxy prefers the much lower-movement `0.05` weight. This weakens
the case for submitting ID336 as-is: it is a clean controlled probe, but not a
Public `0.58`-plausible candidate. If a new artifact is created from this
branch, it should use the next unique ID and explicitly justify whether it
chooses local gain (`0.15`) or movement safety (`0.05`); do not silently reuse
or overwrite ID336.

## 2026-06-18 ID337 low-movement S1 artifact

The movement-safe weight `0.05` variant from the ID336 sensitivity sweep was
promoted as a new generated-unsubmitted artifact:

- File: `submission_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.csv`
- Local CV: `0.538066`
- Base: ID178 confirmed public anchor
- Change: S1 only, random-conv `nk512 seed45502`, `lr_c0_1`, weight `0.05`
- DACON submission: no

The artifact contract is complete: unique submission CSV, history notebook,
metadata report, decision note, base OOF, base test, and submission-log row all
exist for ID337. Non-S1 targets are exactly identical to ID178 in both OOF and
test predictions.

Gate summary:

- Local gain vs ID178: `+0.000572` by rounded log score
- S1 loss improvement: `+0.003999`
- Mean Δ vs ID178: `0.001336`
- Max Δ vs ID178: `0.037585`
- Failed-direction projection: `-0.000554`
- Rerank tier: `reviewable_low_movement`
- Conservative public-risk proxy: `0.605022`
- Naive ID178-gap projected public: `0.600839`

The post-ID337 rerank places ID337 first among reviewable low-movement
candidates by the conservative proxy, ahead of ID307 and ID179. This is still
not a calibrated public prediction. Interpretation: ID337 is cleaner than ID336
for a cautious ID178-relative public probe, but it gives up some local gain and
does not establish a Public `0.58` path. It must not be submitted without
explicit approval for the exact CSV.

Supporting reports:

- `reports/experiment_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.json`
- `reports/decision_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.md`
- `reports/id337_s1_w005_public_risk_gate_20260618.json`
- `reports/id337_s1_w005_public_risk_gate_20260618.md`
- `reports/post_id337_public_anchor_rerank_20260618.json`
- `reports/post_id337_public_anchor_rerank_20260618.md`

## 2026-06-18 ID179 anchor S1 sweep and ID338 artifact

The same S1 `seed45502` source was swept over ID179:

- Report: `reports/id179_s1_seed45502_weight_sensitivity_20260618.json/.md`
- Best local: weight `0.15`, local `0.537568`, but it fails the low-movement
  gate because max Δ vs ID178 rises to `0.144825`.
- Best proxy/balanced: weight `0.05`, local `0.537861`, max Δ vs ID178
  `0.073989`, proxy `0.606833`.

The balanced weight `0.05` variant was promoted as ID338:

- File: `submission_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.csv`
- Base: ID179 sequence anchor
- Local CV: `0.537861`
- Change: S1 only, random-conv `nk512 seed45502`, `lr_c0_1`, weight `0.05`
- DACON submission: no

Tradeoff versus ID337:

| id | base | local CV | max Δ178 | failed projection | proxy | interpretation |
|---:|---:|---:|---:|---:|---:|---|
| 337 | 178 | `0.538066` | `0.037585` | `-0.000554` | `0.605022` | safer proxy choice |
| 338 | 179 | `0.537861` | `0.073989` | `+0.000549` | `0.606833` | stronger local, less conservative |

Post-ID338 rerank keeps ID337 ahead of ID338 by the conservative proxy; ID338
appears as a reviewable low-movement candidate but not a replacement for ID337.
Neither is Public `0.58` evidence. If a public probe is approved, ID337 is the
cleaner safety-first file, while ID338 is the local-gain sibling.

Supporting reports:

- `reports/experiment_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.json`
- `reports/decision_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.md`
- `reports/id338_s1_w005_public_risk_gate_20260618.json`
- `reports/id338_s1_w005_public_risk_gate_20260618.md`
- `reports/post_id338_public_anchor_rerank_20260618.json`
- `reports/post_id338_public_anchor_rerank_20260618.md`
- `reports/id337_id338_s1_probe_tradeoff_20260618.json`
- `reports/id337_id338_s1_probe_tradeoff_20260618.md`

## 2026-06-18 recent S1 probe stress review

A report-only stress review compared the recent S1 candidates with existing
OOF/test artifacts only:

- Report: `reports/recent_s1_probe_stress_review_20260618.json/.md`
- Candidate IDs: `178`, `179`, `336`, `337`, `338`, `334`, `335`
- Model training: no
- DACON submission: no

Key rows:

| id | local | robust | max Δ178 | failed projection | proxy | read |
|---:|---:|---:|---:|---:|---:|---|
| 337 | `0.538066` | `0.563741` | `0.037585` | `-0.000554` | `0.605022` | safety-first S1 probe |
| 338 | `0.537861` | `0.563600` | `0.073989` | `+0.000549` | `0.606833` | local-gain sibling |
| 336 | `0.537663` | `0.563379` | `0.112756` | `-0.001662` | `0.607945` | local-best but higher movement |

Interpretation: ID336 remains the local/robust-best S1 variant, but ID337 is
the cleaner public-anchor candidate because it keeps movement from ID178 much
smaller. ID338 sits between them: local stronger than ID337, but less
conservative. None of these repairs validation or proves Public `0.58`; if a
controlled public probe is explicitly approved, ID337 remains the safety-first
file.

## 2026-06-18 true-nested next-lane priority update

Two report-only consolidation audits were added:

- `reports/true_nested_next_lane_priority_post_id338_20260618.json/.md`
- `reports/b1_pair_blend_source_dependency_audit_20260618.json/.md`

Initial intuition was that the pair/logit `direct_no_252` lane might be the
cleanest high-signal rebuild target. The dependency audit corrected that:
although the selector excludes direct ID252, the source artifacts
`254/255/256/257/258/266` still trace through high/extreme full-OOF bridge or
calibration chains, including ID242/250/251/252. Therefore the immediate true
nested execution lane is not “reuse B1 sources”; it is the lower
delta-capped bridge/source rebuild first.

Current execution split:

| lane | role | decision |
|---|---|---|
| ID337 S1 probe | controlled public probe only | hold unless exact submission is approved |
| A1 delta-capped bridge midrisk rebuild | immediate true-nested execution lane | rebuild bridge choices inside folds first |
| B1 pair/logit direct-no-ID252 | downstream selector map | replay only after source layer is rebuilt |
| C2 ID242/252 calibration chain | blocked | do not spend submission budget |

No submission artifact or DACON submission was created by these reports.

## 2026-06-18 fold-internal A1 bridge rebuild diagnostic

A report-only fold-internal replay was created for the A1 delta-capped bridge
lane:

- `reports/fold_internal_delta_capped_bridge_rebuild_post_id338_20260618.json/.md`

The selector choices were learned inside each outer fold, while the underlying
base/extreme source artifacts remained existing OOF products. The bridge signal
survived this stricter replay: best subject-hole was
`midrisk_caps_all_extremes` cap `0.30` with local `0.395423` and robust
`0.418444`; best LOSO was the same scenario/cap with local `0.395415` and
robust `0.418415`. The cleaner `no_q240_extremes` branch was weaker but still
nontrivial at cap `0.30` (`0.414272` subject-hole, `0.414440` LOSO).

Decision: this supports rebuilding the bridge/source layer inside true
model-level outer folds, but it is not submission evidence because the source
OOF products are still pre-existing. No CSV, notebook, submission-log row, base
artifact promotion, or DACON submission was created.

## 2026-06-18 A1 bridge source rebuild priority

A report-only source-priority map was created:

- `reports/a1_bridge_source_rebuild_priority_post_id338_20260618.json/.md`

The A1 all-extreme fold-internal best row is strong but q240-dominated: ID242
or ID243 appears in `71.4%` of fold-target choices under both subject-hole and
LOSO. The q240-free alternative gives up about `0.018849` subject-hole local
score and `0.019026` LOSO local score, but avoids direct ID242/243 dependence.
Its stable target map is mostly ID241 for `Q1/Q2/Q3/S1/S2`, base/ID238 for
`S3`, and ID239 for `S4`.

Decision: the next validation-repair lane is
`A1_q240_free_source_layer_rebuild` using source IDs `238/239/241` inside true
outer folds, then replaying the cap `0.30` selector report-only. Keep the
q240 branch as explanatory stress only; do not create a DACON-facing artifact
from the existing q240-free OOF products.

## 2026-06-18 A1 q240-free source replay result

A stricter no-submit source replay was created:

- `reports/a1_q240_free_nested_source_replay_post_id338_20260618.json/.md`

This replay refits the ID238-like pair/logit source choice, the ID239/ID241
q240-free calibrators, and the A1 cap `0.30` bridge choices inside each outer
fold, while still using the lower upstream OOF artifacts `221/226/227/228/237`.

Result:

| split | replay local | replay robust | selector-only qfree | delta |
|---|---:|---:|---:|---:|
| subject-hole source replay | `0.491331` | `0.511750` | `0.414272` | `+0.077058` |
| LOSO source replay | `0.484186` | `0.507550` | `0.414440` | `+0.069746` |

ID235 base is `0.482255` / robust `0.502595`, so the q240-free source replay
does not preserve enough of the selector-only signal. Decision: deprioritize
the q240-free A1 lane before spending Kaggle-heavy compute on a model-level
rebuild. This creates no submission artifact and no DACON submission.

## 2026-06-18 post-A1 sequence-summary pivot

After the A1 q240-free replay failed, sequence-summary embeddings were replayed
on the main sequence anchors:

- `reports/id178_sequence_summary_embedding_post_a1_20260618.json`
- `reports/id179_sequence_summary_embedding_post_a1_20260618.json`
- `reports/id190_sequence_summary_embedding_post_a1_20260618.json`
- `reports/id191_sequence_summary_embedding_post_a1_20260618.json`
- consolidation: `reports/post_a1_sequence_pivot_priority_20260618.json/.md`

All four anchors improved locally. The best raw local replay was ID190
`0.537956 -> 0.536326`, while the clean public-anchor replay was ID178
`0.538638 -> 0.536786`. The ID178 replay modifies `Q1/Q2/Q3/S1`, with the
largest movement on `Q2`.

A geometry gate was then run:

- `reports/sequence_summary_geometry_gate_post_a1_20260618.json/.md`

Best gated construction was
`id178_q123_sequence_summary_plus_id337_s1_safety_gate`, local `0.536437`,
robust `0.561965`, max Δ178 `0.121314`, failed projection `-0.014863`. This is
the strongest non-calibration candidate-construction lead so far, but it was not
promoted because the movement slightly exceeds the internal `0.12` cap and
changes four targets.

A cap-sensitivity scan was then run:

- `reports/sequence_summary_cap_sensitivity_post_a1_20260618.json/.md`

The cleaner report-only variant is
`id178_q123_summary_scale0.80_plus_id337_s1`: local `0.536441`, robust
`0.562034`, max Δ178 `0.097051`, failed projection `-0.012001`. This preserves
almost all of the local gain while bringing max movement below `0.10`.

ID339 was then promoted as a generated-unsubmitted artifact:

- `submissions/submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv`
- `reports/experiment_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.json`
- `reports/decision_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.md`

ID339 local CV is `0.536441`, robust score `0.562034`, max Δ178 `0.097051`,
and failed-direction projection `-0.012001`. It is not Public `0.58` evidence
and no DACON submission was made.

Follow-up risk gate:

- `reports/id339_seqsum_s080_s1_public_risk_gate_20260618.json/.md`

The gate keeps ID339 in `reviewable_low_movement` but assigns conservative
public-risk proxy `0.609547`, worse than ID337 by `0.004525` despite ID339's
local gain. Decision: ID339 is a valid full-contract candidate-construction
lead, but it should not jump ahead of the lower-movement ID337 safety probe
without explicit approval.

## 2026-06-18 Mac M4 / Kaggle parity rule

The local machine is Apple Silicon (`M4 Pro`, arm64) with project Python
`3.14.3`; TensorFlow is not importable in the current venv, PyTorch MPS is
available, and the Kaggle credential file exists. Bare `kaggle` is not on the
current shell `PATH`, but `.venv/bin/kaggle` works (`2.1.1`) and the Makefile
already uses it.

Report:

- `reports/mac_m4_kaggle_compute_parity_policy_20260618.json/.md`

This does not explain ID339 directly because ID339 is not a TensorFlow GPU
training artifact. But from now on, any Mac-local TensorFlow/Keras/PyTorch-MPS
artifact must pass a Kaggle/CPU parity gate before promotion: same seed/data
slice, recorded package/device versions, prediction correlation, max/mean
absolute deltas, and per-target log-loss deltas. No package installation,
Kaggle push, or DACON submission was made.

Follow-up Kaggle status:

- `reports/kaggle_offload_status_and_next_lanes_post_id339_20260618.json/.md`

The existing sequence-source and sequence-anchor Kaggle kernels are `COMPLETE`;
their outputs were pulled and inspected; the transient `build/` pull directory
was then removed to keep structure audit clean. The next remote job was scoped
as a report-only ID337/ID339 candidate parity runner, not a broad heavy model
sweep.

The candidate parity runner was then prepared and run on Kaggle CPU:

- `scripts/diagnose_candidate_parity_post_id339.py`
- `notebooks/kaggle_candidate_parity_runner/Candidate_Parity_Post_ID339_Runner.ipynb`
- `reports/candidate_parity_post_id339_20260618.json/.md`
- `reports/candidate_parity_runner_prep_post_id339_20260618.json/.md`
- `reports/candidate_parity_crossplatform_gate_post_id339_20260618.json/.md`
- `reports/remote_candidate_parity_post_id339_20260618.json/.md`

Local smoke test result: ID337 and ID339 both `review_pass`; OOF/test max deltas
are `1.11e-16`, with local-CV deltas only from six-decimal log rounding. This
does not authorize a submission; it only proves the local candidate rebuild.

Kaggle CPU result: the original strict `1e-6` max-delta gate marked both
candidates `fail`, but the differences were tiny cross-platform/package-version
floating-point deltas: ID337 max prediction delta `6.264e-06`, ID339 max
prediction delta `2.249e-05`, and min correlations effectively `1.0`. Under the
documented cross-platform gate (`3e-5` max prediction delta, `1e-5` local-CV
delta, `0.999999` min correlation), both candidates pass. This means Mac M4 /
Kaggle environment drift is not the credible explanation for the public-local
gap. It does not imply Public `0.58`; the bottleneck remains validation-public
transfer and candidate selection.

Post-parity transfer gate:

- `reports/post_remote_parity_public_anchor_rerank_post_id339_20260618.json/.md`
- `reports/post_remote_parity_next_lane_decision_20260618.json/.md`
- `reports/validation_transfer_gate_post_remote_parity_20260618.json/.md`

The gate uses the recent ID178/ID329 public gap floor and conservative
post-parity proxy as a stress estimate. No current watch candidate passes the
Public `0.58` plausibility gate. ID337 remains the safest controlled probe only
if the user explicitly approves the exact CSV; it is not an improvement-proof
candidate. ID340 should not be created from local CV alone.

Post-parity test-geometry OOF stress:

- `reports/test_geometry_oof_stress_post_remote_parity_20260618.json/.md`
- `reports/id340_validation_split_manifest_20260618.json/.md`

ID339 improves hidden-test-shaped OOF slices versus ID178, but ID287 and ID329
also look strong on the same slices while failing publicly. This confirms that
full-OOF geometry slicing is useful as a trap detector but not as promotion
evidence. The ID340 split manifest freezes the next source-rebuild lenses:
subject-hole, same-subject timeseries, known test-band, boundary-neighbor, and
LOSO sanity. Existing full-OOF recombination alone is blocked.

ID340 source-rebuild probe status:

- `reports/id340_source_rebuild_split_probe_20260618.json/.md`
- `reports/id340_sequence_summary_split_probe_20260618.json/.md`
- `reports/id340_sequence_summary_candidate_decision_20260618.json/.md`

The raw/tabular source lane fails against ID178 on every primary split. The
sequence-summary lane is materially better: the fixed design
`blend_id178_sequence_summary_lr_c0_3_w0.03` beats ID178 on all primary split
lenses (`subject_hole_5`, test-band, boundary-neighbor 2/5, and LOSO), but is
slightly worse on the non-primary `subject_timeseries_5` stress lens. Treat it
as the first credible ID340 construction lead, not as a generated artifact. The
next required step is a full-train/test prediction geometry gate for this exact
design before any CSV/notebook/log/base-prediction contract is created.

ID340 artifact promotion:

- `reports/id340_exact_sequence_full_geometry_gate_20260618.json/.md`
- `reports/id340_sequence_subset_geometry_scan_20260618.json/.md`
- `reports/id340_q2_seqsum_w001_public_risk_gate_20260618.json/.md`
- `submissions/submission_id_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426_unsubmitted.csv`

The exact all-target design was too broad: proxy `0.606723`, worse than ID337.
The safer subset scan selected `q2_only_w0.01`, which passes all primary split
lenses and improves the conservative proxy versus both ID337 and ID339. It was
promoted as generated-unsubmitted ID340 with full artifact contract. This is a
controlled low-movement probe candidate only: proxy `0.604062`, Public `0.58`
plausible `False`, and no DACON submission made.

Post-ID340 micro-control follow-up:

- `reports/post_id340_micro_sequence_target_scan_20260618.json/.md`
- `reports/id341_q2_seqsum_w00025_public_risk_gate_20260618.json/.md`
- `submissions/submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv`

The micro scan tested single/pair target subsets at weights `0.0025`–`0.02`.
It selected `q2_w0.0025` as the safest ID178-neighborhood move, with proxy
`0.603830`, max delta vs ID178 `0.002025`, and all frozen primary split lenses
slightly better than ID178. This was promoted as generated-unsubmitted ID341
with full artifact contract, but it is deliberately a control/probe artifact:
Public `0.58` plausible remains `False`, and no DACON submission was made.

Post-ID341 A1 q240-free closure:

- `reports/a1_q240_free_target_salvage_post_id341_20260618.json/.md`
- `reports/a1_s4_salvage_geometry_gate_post_id341_20260618.json/.md`
- `reports/a1_s4_id178_anchor_micro_scan_post_id341_20260618.json/.md`
- `reports/post_id341_a1_qfree_lane_closure_20260618.json/.md`

The q240-free A1 source replay has one real surviving target, `S4`: nested replay
improves S4 by about `0.106` versus ID235 and the target-only hybrid improves
ID235 by about `0.015`. However, the direct ID235 S4 salvage is rejected by
public-risk geometry (`max Δ178 0.574855`, failed-direction projection
`+0.521897`), and the safer ID178-anchored S4 micro scan does not beat ID341's
proxy. Therefore the local A1 qfree artifact lane is closed unless a true
upstream model-level Kaggle rebuild replaces the existing OOF source artifacts.

Post-ID341 family priority:

- `reports/demographic_target_overlay_post_id341_20260618.json/.md`
- `reports/post_id341_family_priority_20260618.json/.md`

The ID341-anchored demographic/static overlay scan also selects alpha `0` for
all allowed targets (`Q2`, `S2`, `S4`), so demographic local artifact reuse is
closed. The live priority is no longer local OOF artifact recombination: either
prepare a report-only Kaggle upstream model-level rebuild for the contaminated
A1/B1 source layers, or design a genuinely new feature source that can lower the
transfer proxy below ID341.

Post-ID341 upstream rebuild prep:

- `reports/post_id341_upstream_rebuild_blueprint_20260618.json/.md`
- `reports/post_id341_upstream_rebuild_runner_prep_20260618.json/.md`
- `reports/post_id341_upstream_rebuild_phase1_readiness_20260618.json/.md`
- `reports/post_id341_upstream_source_priority_matrix_20260618.json/.md`
- `reports/post_id341_clean_backbone_rebuild_design_20260618.json/.md`
- `notebooks/kaggle_upstream_rebuild_runner/Upstream_Rebuild_Post_ID341_Runner.ipynb`

A CPU-only Kaggle runner scaffold now exists for the upstream rebuild blueprint,
with Makefile targets `make kaggle-push-upstream-rebuild-runner` and
`make kaggle-wait-and-pull-upstream-rebuild-output`. Local research pre-flight passed.
The phase-1 readiness audit is green: required reports, frozen split manifest,
source artifacts, and public-anchor records are present for a report-first
upstream rebuild lane. The runner now validates both the blueprint and phase-1
readiness and stages only report artifacts. It also regenerates and stages the
latest transfer-v2, external-prior, and combined sleep/evening/external closure
reports in the Kaggle working copy before upstream planning, so pulled runner
outputs can prove that the local closure survives Kaggle CPU/runtime parity. It
still creates no submission files. Do not push or extend it into a long rebuild
without explicit approval.

The refreshed readiness audit now also requires the latest local new-source
closure reports: sequence+sleep+evening, fixed-list place/social novelty,
sensor-adherence/dropout, transfer-v2, external ETRI prior, and combined
sleep/evening/external prior.
It records `local_new_source_stack_closed=True`, meaning the current local
source stack must not create ID342; the next useful approved action is a
CPU-only report-first upstream rebuild runner before any artifact promotion.

Phase-2 raw/sequence execution spec:

- `reports/post_id341_phase2_raw_source_execution_spec_20260618.json/.md`

This freezes the next approved runner revision after phase-1 closure. The
recommended next revision is `remote_phase2_raw_multimodal_sequence_source_smoke`:
nightly multimodal bins, environment/social routine bins, and fold-internal
subject-residual sequence drift. It explicitly forbids submission CSVs, base
prediction artifacts, notebook-history artifacts, and submission-log changes.
Any temporary source output must pass transfer and joint-risk gates against
ID178/287/329/341 before artifact review. The spec is ready, but does not
authorize Kaggle push/run or heavy compute by itself.

Execution packet:

- `reports/kaggle_upstream_execution_packet_post_id341_20260618.json/.md`
- `reports/kaggle_source_zip_selftest_post_id341_20260618.json/.md`
- `reports/post_id341_kaggle_runner_intake_contract_20260618.json/.md`
- `reports/post_id341_kaggle_output_intake_gate_20260618.json/.md`
- `reports/post_id341_kaggle_prelaunch_readiness_gate_20260618.json/.md`

The packet found and fixed a practical packaging issue: the Kaggle source
dataset zip must include `reports/`, otherwise the runner cannot validate the
refreshed readiness and local-source closure reports on Kaggle. The local
`source.zip` is transient, but the self-test rebuilds it, verifies the required
reports/base predictions, extracts it, runs packaged phase-1 readiness, records
the SHA256 hash, and removes `build/`. After adding the phase-2 raw multimodal
execution plan, the packet now also requires the plan script/report and notebook
cell. The refreshed self-test passed under the new required-entry set
(`source.zip` size `150.744` MB, transient `build/` removed), and the execution
packet is again `ready_for_explicit_approval=True`. The exact
approval-required sequence remains
`ALLOW_KAGGLE_REMOTE=true make kaggle-push-upstream-patch-dataset`,
`ALLOW_KAGGLE_REMOTE=true make kaggle-check-base-dataset-files`,
`ALLOW_KAGGLE_REMOTE=true make kaggle-check-upstream-patch-dataset-files`,
`ALLOW_KAGGLE_REMOTE=true make kaggle-push-upstream-rebuild-runner`,
`ALLOW_KAGGLE_REMOTE=true make kaggle-kernel-status-upstream-rebuild`, then
`ALLOW_KAGGLE_REMOTE=true make kaggle-wait-and-pull-upstream-rebuild-output`. Do not run
those commands without explicit approval; the Makefile now fails Kaggle network
targets closed unless `ALLOW_KAGGLE_REMOTE=true` is supplied. The wait-and-pull
target waits through running/queued/pending states and refuses to pull unless
Kaggle reports complete/success. The output intake
gate is the required local check after any future pull: with no pulled Kaggle
output present it reports
`output_pending=True`, `artifact_contract_review_allowed=False`, and keeps
submission/artifact creation blocked. If remote reports later appear, they must
all parse cleanly, contain no submission-like files, and show no forbidden
remote action before a separate artifact-contract review can even begin.
The local no-network runner smoke now confirms this package can execute the
upstream rebuild notebook end-to-end before any Kaggle approval is consumed. It
stages only report artifacts and `post_id341_upstream_rebuild_runner_manifest.json`
— no CSVs, base predictions, notebook-history files, submission-log rows, Kaggle
network calls, or DACON submissions. The package contract now explicitly includes
the full `submissions/` directory, ID178/287/329/341 anchor CSVs, `Makefile`, and
`dataset-metadata.json`, because those inputs are required for joint-risk
thresholds and blueprint readiness inside the packaged project.
The local smoke report also records fingerprints of the runner/package inputs,
and the prelaunch console now requires
`reports/post_id341_local_smoke_freshness_gate_20260618.json/.md` to confirm
those fingerprints still match current files. If any runner notebook, Makefile,
metadata, guard script, or runner-referenced source script changes, do not use
the previous smoke result for approval; rerun `make kaggle-local-runner-smoke`
and then `make kaggle-upstream-prelaunch-local`.
The runner/intake contract diagnostic is the required pre-push counterpart: it
now confirms every report expected by the intake gate is both generated and
staged by the notebook. This fixed a pre-run omission where five phase-2 reports
would have been created but not copied into Kaggle outputs. It now also scans
the runner code for no-submit violations (`.to_csv(...)`, submission/base output
paths, DACON submit APIs, Kaggle network CLI calls, forbidden `True` decision
flags, and base-prediction writes); all forbidden-match counts are currently
zero. The output-intake gate also requires the pulled runner manifest; tests now
cover the critical branches: manifest missing blocks review, valid manifest plus
a review signal can only open a separate artifact review, any submission-like
file blocks review again, and the runner safety contract blocks submission-like
actions before push.
The prelaunch readiness gate is the single current launch-console report. It
aggregates the approval note, execution packet, source.zip self-test,
runner/intake contract, output-intake pending state, kernel metadata, and
Makefile dry-runs. Current status: ready to wait for explicit approval, but no
Kaggle push/run, artifact creation, or DACON submission is authorized.
After a future approved pull, the remote result summary
`reports/post_id341_kaggle_remote_result_summary_20260618.json/.md` is the
required post-intake interpretation layer. It compares remote proxy evidence
against ID178 Public and the Public `0.58` line, surfaces forbidden remote
actions, and still keeps artifact/submission creation blocked unless a separate
contract review is started.
The final post-remote decision gate
`reports/post_id341_remote_next_action_decision_20260618.json/.md` turns that
evidence into the next operational state: wait, close this remote lane, or open
a separate artifact-contract review. It never creates a candidate by itself.
Operational shortcut: after an approved pull, run
`make kaggle-review-upstream-rebuild-output` to execute intake, remote summary,
and next-action decision in order. Before approval, `make
kaggle-upstream-prelaunch-local` refreshes only the local readiness console.

The upstream source priority matrix narrows the first true rebuild target:
existing artifacts still are not promotable, and the clean rebuild should start
with `ID221` feature-similarity root plus `ID226` target-wise convex backbone.
Although `ID227`, `ID237`, `ID239`, and `ID241` look much better locally, they
also expose large known-failure geometry and/or full-OOF calibration lineage.
Hold `ID227/228/237` and A1/B1 replay until the clean `ID221/ID226` backbone is
rebuilt and passes the same transfer gates.

The clean-backbone rebuild design is ready. It fixes the next heavy-compute
scope to `clean_id221_then_clean_id226`: rebuild an ID221-like
feature-similarity root with fold-internal feature scaling/imputation and
inner-fold target selection, then rebuild an ID226-like target-wise convex
backbone without reusing the existing ID172 extreme full-OOF calibration
artifact. The design explicitly blocks ID227/228/237 calibration derivatives
and A1/B1 replay until the clean backbone reports are green.

Clean-ID221 smoke/probe follow-up:

- `reports/post_id341_clean_id221_feature_similarity_smoke_20260618.json/.md`
- `reports/post_id341_clean_id221_inner_selection_probe_20260618.json/.md`
- `reports/post_id341_clean_backbone_next_step_decision_20260618.json/.md`

Both clean ID221 reads beat ID178/ID341 on the primary frozen lenses, but neither
beats the ID218 base or historical ID221. The inner-selection probe has low
failed-direction exposure (`-0.0154` onto ID287), so it is useful as an
anchor/control, not as a standalone improvement source. Do not spend more local
work tuning ID221 now; move the next report-only clean-backbone lane to ID226
fold-internal convex/calibration smoke.

Clean-ID226 and clean-backbone closure:

- `reports/post_id341_clean_id226_calibration_smoke_20260618.json/.md`
- `reports/post_id341_clean_backbone_lane_closure_20260618.json/.md`

The conservative fold-internal ID226-like calibration smoke starts from ID218
and deliberately does not reuse ID172's extreme full-OOF calibration artifact.
It improves versus ID178/ID341 in full OOF, but fails the primary frozen gates
and does not beat ID218 (`+0.000338`). Therefore the clean-backbone lane is
closed: do not replay ID227/228/237 and do not create ID342 from this family.
The next score-relevant no-submit work should be a stronger raw/sequence source
model or another independent source family, then transfer-gated against
ID178/287/329/341.

Post-ID341 public transfer audit:

- `reports/post_id341_public_transfer_audit_20260618.json/.md`
- `reports/post_id341_public_local_gap_diagnosis_20260618.json/.md`
- `reports/post_id341_data_source_coverage_audit_20260618.json/.md`

The refreshed transfer audit uses 23 plausible public rows and the ID178/ID329
recent anchor gap floor. The local/public correlation on plausible rows is only
`0.156420`, so local CV remains a weak candidate selector. None of the current
rows passes the Public `0.58` transfer gate. ID341 remains the lowest-movement
control/probe, not an improvement-proof submission. The next useful work is
therefore the report-only Kaggle upstream source rebuild or a genuinely new
source family evaluated through the same transfer gate, not additional local
OOF recombination.

The public/local gap diagnosis makes the local-CV target explicit. Using the
recent ID178/ID329 floor gap, a Public `0.58`-plausible candidate would need
local CV `<=0.517227`; using the ID329 stress gap it would need `<=0.494927`.
Current watch candidates ID334-341 proxy to `0.599+` even under the recent
floor. This means "iterate until local `0.53`" is still not enough unless the
validation design or source family also proves transfer. Re-run
`make public-local-gap-diagnosis` after any new actual public score is added.

The data/source coverage audit confirms the issue is not simply that we forgot a
major local asset: official metric/spec files, processed train/test, all twelve
raw 2025 modality parquet tables, external ETRI 2019/2020 CSVs, and paper/data
references are present. External data remains weak-prior/reference-only and
direct concatenation is blocked; the no-submit upstream Kaggle runner references
the current external, raw/sequence, phase-2 block-A/B, and clean-backbone checks.
So the next score-relevant action is still upstream rebuild/parity evidence, not
another local artifact created from the existing coverage.

Post-ID341 new source blueprint:

- `reports/post_id341_new_source_blueprint_20260618.json/.md`

The raw coverage review confirms that the processed table already contains many
daily aggregate features, but the most promising non-recombination lane is a
separate `sleep_boundary_transition_v1` source: infer sleep onset, wake time,
night interruptions, HR recovery, and charge/screen/light/activity transition
features with fold-internal per-subject baselines. This targets `S1`–`S4` and
`Q1` first. `evening_load_recovery_v1` and
`routine_stability_place_entropy_v1` are secondary lanes. External ETRI data
remains a weak regularizer/feature-idea source only, not direct labeled training
data.

Sleep-boundary transition smoke and gates:

- `reports/sleep_boundary_transition_smoke_post_id341_20260618.json/.md`
- `reports/sleep_boundary_transition_v2_gate_post_id341_20260618.json/.md`
- `reports/sleep_boundary_transition_v3_scale_scan_post_id341_20260618.json/.md`
- `reports/sleep_boundary_prepromotion_gate_post_id341_20260618.json/.md`

This is the first post-ID341 new-source lane with real validation signal. The
v2 target-restricted candidate (`Q1`, `S1`, `S2`, `S4`) beats ID341 on every
frozen split, and the v3 scale scan finds a low-movement scale `0.05` row that
also beats ID341 on all six frozen lenses. Its movement is tiny (max Δ vs ID341
`0.004759`) and its proxy `0.601697` is better than the ID341 proxy, but it is
still above the actual ID178 Public `0.601411` and far from Public `0.58`.
Therefore it is a validated feature/source lead, not a submission artifact:
do not create ID342 from this lane alone.

Secondary new-source follow-up:

- `reports/evening_load_recovery_smoke_post_id341_20260618.json/.md`
- `reports/combined_new_source_gate_post_id341_20260618.json/.md`
- `reports/combined_sequence_sleep_evening_gate_post_id341_20260618.json/.md`
- `reports/routine_stability_place_entropy_smoke_post_id341_20260618.json/.md`
- `reports/place_social_novelty_smoke_fixedlist_post_id341_20260618.json/.md`

The evening-load/recovery lane contributes only a tiny `S4` support signal. Its
best standalone proxy is `0.601507`, but it wins only 4/5 primary lenses. The
combined sleep-boundary + evening scan improves the best proxy to `0.601490`,
but still does not go below the actual ID178 Public anchor `0.601411`; no row is
Public `0.58` plausible. The routine-stability/place-entropy lane selects no
target under the frozen gate. Adding the surviving ID340 sequence-summary source
to the combined scan does not help transfer: the best proxy row keeps
`sequence_scale=0`, and the best sequence-including all-primary row is worse
(`0.601831`). A fixed-list raw place/social novelty smoke over WiFi BSSID, BLE
address, and app-name novelty has good coverage after parser correction
(train missing mean `0.0198`) but still selects no targets; Q1/Q2 top rows win
only 3/5 primary lenses. Current conclusion: these are useful source leads, but
not enough for ID342 promotion. The next meaningful step is either a materially
different raw source or the report-only Kaggle upstream rebuild.

Sensor-adherence/dropout smoke:

- `reports/sensor_adherence_dropout_smoke_post_id341_20260618.json/.md`

This report tested a different label-free source family: row-level sensor
completeness, modality dropout, family presence entropy, and device-adherence
signals from `299` numeric processed/source columns. Coverage was complete after
feature construction (`0.0000` train/test feature missing mean), but no target
met the frozen selection gate. The best near misses were only 3/5 primary wins
for Q1 and S4, so this must not create ID342 or be widened locally without a new
independent reason. The next useful path remains the report-only Kaggle upstream
rebuild or a materially different raw source.

Subject residual sequence-drift smoke:

- `reports/subject_residual_sequence_drift_smoke_post_id341_20260618.json/.md`
- `reports/subject_residual_target_restricted_gate_post_id341_20260618.json/.md`

This report tests the phase-2 `subject_residual_sequence_drift_v1` idea without
creating submission artifacts. It selects `56` processed numeric covariates and
turns them into `399` past-only per-subject drift/volatility features. There is
real target-level signal: Q1 wins 5/5 primary target lenses, S2 wins 4/5, and
S4 wins 3/5. But the overall best blended candidate beats ID341 on only 1/5
primary splits, and the projected public proxy is `0.603816`, not below actual
ID178 Public `0.601411` and nowhere near Public `0.58`. This is a useful
feature-design clue for a future remote raw-source runner, not an ID342
promotion path. The target-restricted follow-up selects Q1/S4 only, both as
tiny ID178-anchored ExtraTrees residual overlays at weight `0.01`; it improves
3/5 primary splits with mean Δ `-0.000046`, but still fails `subject_hole_5`
and `loso_10`, with projected proxy `0.603784`. Therefore this lane is closed
for local artifact promotion and should only inform phase-2 raw/source design.

Public joint-distribution risk:

- `reports/public_joint_distribution_risk_post_id341_20260618.json/.md`

This report compares the test prediction structure of `23` plausible scored
submissions with train-label target means, pairwise co-occurrence, target
correlations, prediction extremeness/entropy, and within-subject temporal
volatility. It does not create candidates. The useful read is a warning lens:
ID287's collapse is paired with much higher extremeness (`0.1623` at 5/95) and
larger ID178 movement (`0.1154` mean absolute delta), while ID178/ID329 have
near-identical target-mean drift and public scores around `0.601`. Treat these
metrics as candidate-selection cautions, not as a new hard public-fit recipe.

Joint-risk candidate rerank:

- `reports/joint_risk_candidate_rerank_post_id341_20260618.json/.md`

This applies the warning lens to `330` locally loadable logged candidate CSVs.
Among unscored rows, the local-CV frontier is dominated by
`reject_high_joint_risk` rows, matching the ID287 failure mode. Recent
ID337/339/340/341 all remain `low_joint_warning`; ID341 is the lowest-movement
control by this lens, but this does not override the post-ID341 transfer gate or
authorize a submission. Use the rerank only as a warning overlay before manual
artifact review.

Transfer gate v2 with joint-risk:

- `reports/post_id341_transfer_gate_v2_joint_risk_20260618.json/.md`

This recomputes the post-ID341 transfer rows (`244`) and joint-risk rows (`330`)
and requires both gates at once: a candidate must pass the conservative Public
`0.58` transfer-proxy cutoff and remain low/watch on joint-risk warnings.
Result: `0` candidates pass. The closest recent controls such as ID340/337/341
are joint-safe but transfer-blocked around proxy `0.602`–`0.604`, while the
low-local frontier is mostly rejected by joint risk. Therefore there is still no
local Public `0.58` candidate to submit; the next useful path remains an
explicit-approved report-first Kaggle upstream rebuild or a materially new raw
source that passes this combined gate.

External ETRI prior smoke:

- `reports/external_etri_prior_smoke_post_id341_20260618.json/.md`

The older same-institution 2019/2020 summary CSVs were converted into weak
target-like priors: survey sleep/stress/fatigue relative to per-user means for
`Q1`/`Q2`/`Q3`, and sleep-duration/efficiency/latency/WASO heuristics for
`S1`–`S4`. This gives `6135` external long target rows across `42` users, but
the distribution mismatch is severe for objective sleep targets (`S1` external
mean `0.1447` vs current train `0.6822`; `S4` external `0.3118` vs current
`0.5600`). Tiny ID341 overlays find only weak near-signals: the best proxy is
an `S2` prior at `0.601564`, still above actual ID178 Public `0.601411` and far
from Public `0.58`; no external-prior row is reviewable or promotable. External
data is therefore useful as feature/reference inspiration, not as direct
training rows or a submission source.

Combined sleep/evening/external-prior gate:

- `reports/combined_sleep_evening_external_prior_gate_post_id341_20260618.json/.md`

The independent weak-source stack was scanned without sequence-summary deltas:
sleep-boundary, evening-load/recovery, and older-ETRI external priors. The best
proxy row keeps sleep scale `0`, evening scale `0.025`, and external-prior scale
`0`, with proxy `0.601490`. It wins 4/5 primary splits but still misses the
all-primary gate and remains above ID178 Public. This closes the local
non-sequence weak-source combination lane: external priors do not add useful
transfer signal on top of the existing tiny evening lead.

Phase-2 raw multimodal execution plan:

- `reports/phase2_raw_multimodal_execution_plan_post_id341_20260618.json/.md`
- `reports/phase2_block_a_q1_s4_sleep_residual_bridge_post_id341_20260618.json/.md`
- `reports/phase2_block_a_q1_proxy_gap_post_id341_20260618.json/.md`
- `reports/phase2_block_b_s2_sleep_recovery_retry_post_id341_20260618.json/.md`
- `reports/phase3_usage_temporal_microstructure_smoke_post_id341_20260618.json/.md`

This is the next score-relevant direction after local weak-source closure. It
does not create predictions or a submission; it fixes the next approved Kaggle
CPU runner to a small block order. The first block is
`A_q1_s4_sleep_residual_bridge`: raw screen/activity/charge/light/HR/pedo
sleep-boundary features plus fold-internal residual clues for Q1/S4. Its gate is
deliberately harsh: beat ID341 on all primary splits and get the projected proxy
below actual ID178 Public `0.6014113009` before any artifact review. If block A
fails, stop instead of widening into more local blends. S2 recovery and
environment/social routine blocks are secondary and should not run as broad
searches unless the first block gives a stronger anchor. The local block-A smoke
is a meaningful partial positive: Q1 alone is selected and the assembled bridge
beats ID341 on all five primary splits, including `subject_hole_5` and
`loso_10`. But its projected proxy is still `0.603617`, so it does not beat
actual ID178 Public and is not Public `0.58` reviewable. Do not promote it; use
it as the current strongest remote block-A parity target.

The Q1 proxy-gap follow-up makes the stop rule sharper. The Q1 bridge is real
enough to beat ID341 on all primary splits, but not large enough to matter on
the public-transfer scale: the projected proxy is still `+0.002206` above the
actual ID178 Public score, and Q1 alone would need about `11.37x` the current
target-level signal to beat ID178, or about `65.02x` to reach the Public `0.58`
review line. The right next move is not a larger Q1-only overlay; it is a new
independent signal source or target block.

The planned S2 recovery retry was run as a narrow report-only closure check. It
uses sleep-boundary, subject-residual, and evening/recovery features for S2
only, but no config passes the fixed eligibility rule. The top S2 rows win at
most 3/5 primary target lenses and have positive mean delta, so the assembled
candidate remains the ID341 reference with proxy `0.603830`. This closes S2
sleep/recovery as a standalone Public `0.58` path; do not revive it without a
materially different raw source.

The next materially different local source checked was raw `mUsageStats`
temporal microstructure. It has strong coverage (`158,426` parsed app events,
`302` coarse usage features, every train/test row covered), and direct app-name
columns were intentionally excluded to reduce memorization risk. Even so, no
target config passes the fixed gate; the best gains are one- or two-split
artifacts, not transfer-safe signals. This closes usage temporal microstructure
as a standalone Public `0.58` path.

Semantic `mAmbience` context was also audited instead of re-run. The historical
S4 broad ambience diagnostic had a real-looking full-OOF gain
(`-0.002880` vs base), but nested validation reversed it to `+0.003467`; the
narrow confirm still worsened nested by `+0.000642`. This is a clean example of
the full-OOF trap that produced earlier public failures. Do not create a new
ambience overlay unless ambience source generation is rebuilt inside a true
upstream/nested training contract.

Post-ID341 local-source closure synthesis:

- `reports/post_id341_local_source_closure_synthesis_20260618.json/.md`
- `reports/post_id341_target_gap_budget_20260618.json/.md`
- `reports/post_id341_missing_target_signal_inventory_20260618.json/.md`
- `reports/post_id341_kaggle_launch_approval_note_20260618.json/.md`

This is now the controlling local stop line. It reads the transfer gate, block-A
Q1 gap, block-B S2 recovery, usage temporal microstructure, ambience closure,
external/combined weak-source reports, place/social and sensor-adherence
closures, clean-backbone closure, and Kaggle packet readiness. All required
reports are present, no local lane is promotable, and no Public `0.58` reviewable
artifact exists. Therefore do not create another local ID342-style candidate
from these sources; the next score-relevant action is report-first upstream
rebuild or Kaggle/Linux parity, followed by a separate transfer-gated artifact
review only if pulled reports are safe, beat ID178, and show `min_proxy <= 0.58`.
An ID178-only improvement above `0.58` is an improvement lead, not ID342.

The target gap budget explains why the local stop line should stay closed. The
strict all-primary budget finds only Q1 and S4 target signals, with projected
proxy `0.602907`; this is still `+0.001496` above ID178 Public and needs another
`-0.010472` sum target delta just to beat ID178. The looser 4/5 watch budget
adds S1/S2 but still projects `0.602256`, above ID178 and nowhere near the
Public-0.58 review cutoff. Therefore the useful missing surface is not more
Q1/S4 tuning; it is independent Q2/Q3/S3 raw/upstream signal or the explicit
approved no-submit Kaggle upstream rebuild.

The missing-target signal inventory makes that instruction concrete. Q2, Q3,
and S3 have zero current local selected rows in the post-ID341 source reports;
S1/S2 are only watch-level, not strict all-primary evidence. The only meaningful
Q2/Q3/S3 surface in the current repo is the historical clean ID221→ID226
upstream backbone, so it must be rebuilt report-first under the explicit Kaggle
approval path before any artifact review. Do not create local ID342 from these
missing targets.
`reports/post_id341_q2q3s3_source_route_20260618.json/.md` is the route table
for that gap. It sets the primary score-relevant route to
`explicit_approved_clean_upstream_id221_to_id226`, keeps
`remote_no_submit_prelaunch_ready=True`, and records the no-approval fallback as
a new Q2/Q3/S3-first environment/social/routine design only. It also blocks more
Q1/S4 tuning because that cannot close the Q2/Q3/S3 budget.
`reports/post_id341_q2q3s3_env_social_routine_v2_design_20260618.json/.md`
concretizes that fallback. Raw `mWifi`/`mBle`/`mUsageStats` are nested,
high-cardinality context streams (`mWifi` BSSID and `mBle` address cardinalities
are very large), so the design forbids direct full-train one-hot memorization
and limits the next local step to fold-internal feature-generation metadata
smoke. The only allowed local feature families are Q2/Q3/S3-first routine
stability, place/social novelty, and ambient-load context; no model training or
ID342 artifact is authorized by this design.
`reports/post_id341_q2q3s3_feature_generation_metadata_smoke_20260618.json/.md`
executes that no-approval fallback at metadata level only. It covers all `700`
train/test keys, creates `247` raw label-free feature columns plus missingness
flags, and passes the high-cardinality/leakage guard. This is useful because it
proves the raw Q2/Q3/S3 lane can be generated cleanly, but it is not leaderboard
evidence: no model was trained, no local CV was claimed, no CSV/log row was
created, and no Kaggle/DACON action was taken. The next permitted local step is
only a fold-internal Q2/Q3/S3 model smoke compared against ID178/ID341.
`reports/post_id341_q2q3s3_fold_internal_model_smoke_20260618.json/.md` runs
that fold-internal model smoke. It finds a Q2-only local lead:
`blend_id341_q2q3s3_lr_c0_08_w0.05` wins 5/5 primary split lenses with mean
delta `-0.002758` versus ID341. The same smoke does not find Q3 or S3 signal,
so this is not Public `0.58` evidence and does not authorize a submission CSV.
The only reasonable continuation is a no-artifact geometry gate for the Q2 lead,
followed by closure if the geometry is not safer than ID178/ID341.
`reports/post_id341_q2q3s3_q2_lead_geometry_gate_20260618.json/.md` runs that
gate. The best scale is `0.25` with effective Q2 weight `0.0125`, 5/5 primary
wins, and proxy `0.601911`. This is safe enough to remember as a Q2 component
but still nowhere near Public `0.58`, and Q3/S3 remain uncovered. Therefore the
local fallback should not produce a submission artifact; it should either wait
for an independent Q3/S3 source or return to the explicit-approved clean
ID221→ID226 upstream rebuild path.
`reports/post_id341_q2q3s3_paper_family_fold_smoke_20260619.json/.md` checks
one such independent-looking local idea using the processed reference-paper
families: Q2=`mUsageStats+DW`, Q3=`wPedo+mUsageStats+DW`, and
S3=`wPedo+mLight/wLight+DW`. It selects no target under the frozen primary
gates and creates no CSV/log/notebook/base artifact. This keeps the processed
paper-family lane as design context only; it does not reopen local ID342.
`reports/post_id341_q3s3_anchor_failure_profile_20260619.json/.md` adds the
current cautionary counterweight: ID287 and ID329 are now explicit Q3/S3
OOF-public traps. Both improve Q3/S3 locally with 5/5 primary wins, but public
is worse than ID178. Therefore ID226's Q3/S3 OOF signal can guide the approved
clean upstream rebuild, but it is not direct-submit evidence and not a local
ID342 shortcut.
The prelaunch console now regenerates this entire Q2/Q3/S3 no-artifact fallback
sequence before the local fallback-chain contract and runner score-relevance
contract. A stale Q2-only report can no longer keep the Kaggle launch path green.
`reports/post_id341_q3s3_physio_light_onset_smoke_20260618.json/.md` tests
that independent Q3/S3 source idea using wearable heart-rate, movement, light,
and charging/onset transition features. It has full row coverage and `121` raw
feature columns, but it selects no target: Q3 reaches only 3/5 primary wins with
positive mean delta, and S3 reaches 0/5. The follow-up closure report,
`reports/post_id341_q3s3_local_lane_closure_20260618.json/.md`, therefore keeps
the Q2 lead as context only, records Q3/S3 as uncovered, and closes the
no-approval local fallback for submission artifacts. Public `0.55` work should
not continue by widening this local lane; the next score-relevant path is the
explicit-approved clean ID221→ID226 no-submit upstream rebuild.

The Kaggle launch approval note freezes the exact no-submit remote sequence and
keeps execution blocked until explicit user approval. It also adds a dataset file
listing check between source dataset push and runner push to reduce stale
payload risk. No network action, runner push, output pull, submission artifact,
or DACON submission was made by creating the note. The paired output-intake gate
currently confirms the expected safe state before any run: outputs are pending,
artifact-contract review is not allowed, and no CSV/log/notebook/base-prediction
artifact has been created.

The local prelaunch target now refreshes the public/local gap diagnosis,
data/source coverage audit, target-gap budget, missing-target inventory,
runner/intake contract, source.zip self-test, upstream execution packet,
local-source closure synthesis, launch approval note, output-intake pending
gate, guarded remote contract audit, and final prelaunch gate before any
approval is consumed. The final gate requires local-CV-only promotion to remain
blocked, external direct concatenation to remain blocked, Q2/Q3/S3 to remain
non-promotable locally but present as an upstream-rebuild surface, the package
self-test to pass, and the data/source coverage plus runner/intake audits to be
ready. This keeps the Kaggle path score-relevant without reopening a local ID342
shortcut.
It now also requires
`reports/kaggle_remote_fail_closed_dryrun_audit_20260618.json/.md`, which calls
the Kaggle-facing Make targets with `ALLOW_KAGGLE_REMOTE=false` and
`KAGGLE=/usr/bin/false` and passes only if the approval guard blocks before any
Kaggle CLI command can run.
The same prelaunch console now also requires
`reports/post_id341_post_pull_review_chain_contract_20260618.json/.md`, proving
the local output-intake, remote-summary, next-action, and Public-0.58 threshold
reports are ordered, local-only, parseable, output-pending, and no-artifact
before any remote approval is consumed.
The Public-0.58 threshold gate is intentionally stricter than a proxy-number
check: even if a future pulled report shows `min_proxy <= 0.58`, artifact review
still stays closed unless the best proxy row has non-empty `selected_targets`
and the local source stack is closed for submission artifacts. This prevents a
low but non-reproducible remote signal from becoming ID342.
It also requires
`reports/post_id341_runner_score_relevance_contract_20260618.json/.md`, which
checks that the approved no-submit runner still addresses the current
Q2/Q3/S3 missing-target bottleneck through the clean ID221→ID226 upstream route
rather than merely passing safety gates.
The guarded remote contract audit now also inspects the body of
`make kaggle-review-upstream-rebuild-output`, requiring the local post-pull chain
to run intake, remote summary, next-action, and the Public-0.58 threshold gate
with the threshold step last. For the launch command sequence, it now treats the
current prelaunch script constant as authoritative instead of trusting a
previously generated prelaunch report, avoiding stale-pass after script changes.
The preferred local pre-approval wrapper is now
`make kaggle-upstream-ready-local`: it runs the focused approval contract tests,
the no-network local runner smoke, the freshness gate, and the final prelaunch
readiness gate in a single local-only sequence. This wrapper is a safety
rehearsal, not remote approval; Kaggle push/run/pull still requires the exact
approval phrase and the guarded `ALLOW_KAGGLE_REMOTE=true` commands.

The post-pull local review chain now ends with
`reports/post_id341_remote_artifact_review_threshold_20260618.json/.md`. This
strict threshold gate requires safe intake, safe next-action, current
public/local-gap evidence, current data/source-coverage evidence, no forbidden
remote action, ID178-beating evidence, and `min_proxy <= 0.58` before a
Public-0.58 artifact-contract review can open. A remote result that only beats
ID178 but remains above `0.58` is recorded as an improvement lead, not a local
ID342 artifact path.

`reports/post_id341_q3s3_event_cadence_smoke_20260618.json/.md` tests the last
lightweight no-approval Q3/S3 idea in this branch: raw timestamp cadence,
inter-event gaps, and silence windows rather than sensor values or state labels.
It has full train/test key coverage but selects no Q3/S3 target under the frozen
primary gates. The negative evidence map now also includes the prior
place/social novelty, routine-stability/place-entropy, and
sensor-adherence/dropout smokes, all of which selected no Q3/S3 target or no
reviewable low-movement row. It therefore counts `8` closed local Q3/S3 lanes.
There is still no local no-approval score route; the only score-relevant open
route remains the explicit-approved clean ID221→ID226 no-submit upstream
rebuild.

Post-ID341 external Public `0.5917` residual audit:

- `reports/post_id341_external_public05917_residual_audit_20260619.json/.md`

The external LGBM+XGB anchor/Subject-hole/stability notebook remains useful as
a design reference, but not as a direct local submission route. The audit marks
`residual_direct_notebook_route_exists=False`, `public_058_evidence_present=False`,
and `q3_s3_still_uncovered=True`: its transferable pieces have already been
absorbed as validation principles or tested as closed overlays, while Q3/S3
coverage is still missing. This closure is now part of the no-approval frontier
and the local Kaggle prelaunch target, so a future ID342 requires either the
approved no-submit upstream rebuild output or a materially new source, not
another direct replay of the external notebook.
The upstream runner now stages this stop line as
`remote_post_id341_q3s3_negative_evidence_map_20260618.json`, and the output
intake gate requires it before any post-pull artifact-contract review can open.

`reports/post_id341_q3s3_fold_nested_sibling_bridge_smoke_20260619.json/.md`
adds one more leakage-safe local check: Q3 is allowed to consume only nested
predictions of Q2, and S3 only nested predictions of S2. This protects against
the tempting but invalid shortcut of using same-row sibling labels. The result
still selects no Q3/S3 target and has best proxy around the ID178 public-anchor
region, not Public `0.58`. Therefore the sibling bridge is not an ID342 route;
the only score-relevant open route remains the explicit-approved clean
ID221→ID226 no-submit upstream rebuild.
This closure is now part of the remaining-route inventory and the prelaunch
score-relevance contract, so future local candidates must bring a genuinely new
Q3/S3 source rather than reusing the Q2/S2 sibling structure as a shortcut.
The approval path now also regenerates this sibling smoke locally and inside the
no-submit Kaggle runner before remaining-route inventory, keeping the remote
review chain aligned with the local closure evidence.

Post-ID341 GPS/circadian local-source update:

- `reports/post_id341_q3s3_remaining_route_inventory_20260618.json/.md` now
  explicitly closes GPS path-topology after the model smoke selected no Q3/S3
  target and no score route.
- `reports/post_id341_material_new_source_queue_20260619.json/.md` now treats
  GPS as closed and advances the no-approval local work to circadian HR/light
  phase-shape feature-spec-only mode.
- `reports/post_id341_circadian_hr_light_phase_metadata_probe_20260619.json/.md`
  passes as a metadata/resource probe with strong phase-shape coverage of
  `0.5622` train / `0.6120` test, but it creates no predictions and is not
  Public `0.58` evidence. The only score-relevant route remains the exact
  approval-gated clean ID221→ID226 no-submit Kaggle rebuild.

Post-ID341 phone/wearable discordance feature-matrix smoke contract:

- `reports/post_id341_phone_wearable_discordance_feature_spec_20260619.json/.md`
- `reports/post_id341_phone_wearable_discordance_feature_generation_dryrun_20260619.json/.md`
- `reports/post_id341_phone_wearable_discordance_feature_matrix_smoke_contract_20260619.json/.md`
- `reports/post_id341_phone_wearable_discordance_features_v1_schema.json/.md`
- `reports/post_id341_phone_wearable_discordance_feature_matrix_validation_stop_20260619.json/.md`
- source family: `phone_wearable_discordance_v1`
- target focus: Q2/Q3/S3
- feature count: 85 bounded `pwd_` columns
- dry-run frame: 450 train rows + 250 test rows, finite bounded values, no
  duplicate keys
- feature matrix write allowed now: false
- writer/validation boundary: exactly one registered label-free matrix plus
  schema report, followed by deterministic validation-stop
- Public `0.58` evidence: false
- score route opened: false
- model/submission/Kaggle/DACON actions: false

This closes the local no-approval phone/wearable lane as infrastructure only. It
does not change the score-relevant route: the only open score path remains the
exact-approved clean ID221→ID226 no-submit Kaggle upstream rebuild.

2026-06-21 DACON daily-slate result:

- ID340 Q2 sequence-summary controlled probe scored `0.601379175253277`.
- ID337 S1 random-conv controlled probe scored `0.6006206286219425`.
- ID341 is recorded as a reconciled HTTP-200 submission from 2026-06-19, but its
  public score still needs independent DACON-page confirmation.
- ID221 was attempted next, but DACON returned `day_max_count`; ID221 and ID255
  remain unsubmitted.

This slate gives a small public improvement over the ID178 anchor only through
ID337, not a step toward Public `0.58`. Treat ID178-neighborhood single-target
micro-overlays as calibration probes, not as the main Public `0.55` path. The
score-relevant route therefore remains validation/source repair and the clean
upstream rebuild lane rather than further public-guided micro-cascade work.

## 2026-06-23 Kaggle upstream rebuild v3 closeout

The approved no-submit v3 runner completed on Kaggle and fixed the v2
`dataset-metadata.json` failure. The log shows runtime metadata synthesis,
`Missing required paths: []`, and the remote upstream blueprint reports
`blueprint_ready=True`.

The pulled review bundle was sanitized to contain only remote reports, the
runner manifest, and the log. The post-pull gates found all 25 expected remote
reports and no forbidden remote actions, but no score-improvement signal:
`remote_review_signal_present=False`, `proxy_below_id178_rows=0`,
`proxy_below_058_rows=0`, and best proxy `0.6014904915736387` is slightly worse
than ID178 Public `0.6014113009`.

Decision: do not promote a new submission artifact from v3. The time-sensitive
leaderboard action remains the exact three-slot slate in
`reports/dacon_three_slot_slate_20260623.json`: ID338, ID342, then ID339 if the
23:30 KST submission-spend line is reached without a stronger gated candidate.

At 23:30 KST, the slate was spent after exact guard re-audit. ID338, ID342, and
ID339 each returned DACON `isSubmitted=True` with HTTP 200. Authenticated
submission history later corrected the unauthenticated leaderboard inference:
ID342 is the team-best row, submission `1467937`, score `0.5960832519`, rank
`220`, timestamp `2026-06-23 23:30:37`; ID339 scored `0.5985187787`
(`1467938`, `23:30:48`); ID338 scored `0.5992145177` (`1467936`,
`23:30:25`). ID342 beats the previous confirmed team best
`0.6006206286219425` by `-0.004537376722` and finally breaks the recent Public
`0.600` wall. The exact result record is
`reports/dacon_three_slot_submission_result_20260623.md/.json`.

## 2026-06-24 Missingness Subject-State Heavy No-Submit Closeout

The remaining high-coverage heavy route has now been executed on Kaggle without
DACON submission. Runner
`softkleenex/dacon-etri-missingness-subject-state-id342` version 3 completed
after package-path repairs and produced only reports/logs under
`build/kaggle_outputs/missingness_subject_state_post_id342/`. The intake report
`reports/post_id342_missingness_subject_state_remote_intake_20260624.md/.json`
is clean and sets `remote_intake_ready=True`, with no submission-like artifacts.

The route covered the intended raw domain: all `6,122,179` raw rows were
streamed, `5,678,719` train/test-key rows matched, mean modality-key coverage
was `0.976905`, and train/test any-observed coverage was `1.0` / `1.0`.
However, the downstream score gate is negative. Stable pass count is `0`, best
projected Public proxy is `0.596101196`, and the gap to Public `0.55` is still
`+0.046101196`. This closes the missingness-aware cross-modal subject-state
embedding branch under ID342 as a direct Public `0.55` route. Do not create a
feature matrix, prediction artifact, submission CSV, submission-log row, or
DACON submission from this branch.

## 2026-06-24 Public-0.55 Gap Attribution

The existing OOF shelf is not enough by itself. The report
`reports/post_id342_public055_gap_attribution_20260624.md/.json` scans all
`247` base OOF files and uses stable scored rows to estimate the public gap.
Stable public-minus-local median is `0.062318739`; therefore a candidate would
need full local `0.487681261` to project to Public `0.55` under the same gap
regime. ID342 is still `+0.046318745` local above that threshold.

There are `64` existing shelf rows below the required local threshold, but the
credible count is `0` after excluding scored failed-frontier traps,
frontier/zerowide families, legacy direct-lite/delta-pair/riskpen blends, and
large-distribution moves. The best low-movement shelf row is ID211 and still
projects to `0.582424274`, well short of `0.55`.

Decision: do not submit an existing shelf row just because its local CV is low.
The next real path must either prove a local-frontier candidate without the
known public-trap movement pattern, or change the public-minus-local gap through
validation/source repair. Same-family micro-cascades and raw subject-state
reruns are closed until a materially new source hypothesis appears.

The follow-up risk lens
`reports/post_id342_public_gap_risk_lens_20260624.md/.json` makes that
constraint sharper. It compares prediction-shape features for all `247` OOF
files and the `8` scored rows. Public failure is strongly associated with
movement away from ID342 and extreme probability shape: mean movement vs ID342
has Pearson `0.991` with public score, and extreme-probability rate has Pearson
`0.997`. The stable-shape envelope leaves only `11` shelf rows; ID342 remains
the best projected row inside it at `0.596318745`, and the Public-0.55
shape-lens pass count is `0`.

Operational consequence: a 0.55 attempt cannot be a larger local-frontier blend.
It must either add independent signal while staying inside the stable movement
envelope, or first repair the validation/public-gap model so local gains stop
turning into public gap expansion.

The stable-envelope microblend follow-up
`reports/post_id342_stable_envelope_microblend_scan_20260625.md/.json` shows
why this remains hard. A one-source ID342-convex scan finds `1124` gate rows,
but the best projected row is legacy-risk ID250 at `0.563921703`; after the
nonlegacy gate, ID285 is best at `0.569887753`. The multi-source follow-up
`reports/post_id342_stable_envelope_multisource_optimizer_20260625.md/.json`
rescans the full shelf, finds `101` nonlegacy gate sources, and runs a bounded
beam search under the same stable-shape envelope. Its best row is ID285 weight
`0.125` plus ID278 weight `0.015`, projected Public `0.566362799`, still
`+0.016362799` from the `0.55` target. Public-0.55 envelope rows remain `0`,
so this does not authorize a submission artifact, Kaggle run, or DACON
submission.

Operational consequence: ID285/ID278-style stable-envelope recombination is a
research ceiling, not a solution. The next credible path must bring a genuinely
independent signal that moves Q3/S3 or the public-gap model, not another
same-family post274/post278/post282/post285 weight tweak.

The target-wise reallocation follow-up
`reports/post_id342_target_reallocation_optimizer_20260625.md/.json` closes the
obvious escape hatch. It excludes confirmed failed-public IDs `287/329` and
public-guided frontier/probe/oracle tokens, then lets each target choose its own
ID342-convex source direction under the stable-shape envelope. It scans `420`
target options and retains `2462` beam states, but the best projected Public is
only `0.566166687`; Public-0.55 envelope rows remain `0`. The remaining target
profile is instructive: S4/WASO is still the largest loss (`0.567825`), Q1 is
next (`0.519604`), Q3/Q2 are already heavily moved by ID285-like directions,
and S3/SOL barely improves (`-0.002828` vs ID342).

Operational consequence: existing-source target reallocation is not the missing
0.55 route. Prioritize a new S4 sleep-fragmentation/WASO source and a new S3
sleep-onset-latency source, or redesign the validation/public-gap model before
spending more work on same-shelf blending.

The direct S4/S3 processed sleep-source follow-up now narrows that instruction.
`reports/post_id342_s4s3_sleep_fragmentation_gate_20260625.md/.json` tests
WASO/fragmentation core columns, sleep-onset context, sleep HR/awake-light
ratios, and their union. It passes `0` rows. S4's best row has mean delta
`-0.000272711`, but it loses on subject-hole/LOSO and is not stable. S3's best
row is positive on average (`+0.000075639`). The raw schema feasibility audit
`reports/post_id342_raw_sleep_event_reconstruction_feasibility_20260625.md/.json`
then checks whether a direct sleep-stage/SOL/WASO route exists in the public raw
files; it scans all `12` raw parquet files and finds `0` direct event columns.

Operational consequence: the next 0.55 attempt should not be another processed
sleep-column overlay or direct raw sleep-stage reconstruction. Either build a
new sequence-level HR/light event model with its own local gate, or move to
validation/public-gap redesign.

The sequence-level HR/light event follow-up has now run locally as
`reports/post_id342_sequence_hr_light_event_gate_20260625.md/.json`. It uses
the available proxy modalities (`wHr`, `wLight`, `mLight`, `mScreenStatus`,
`mACStatus`, `mActivity`, and `wPedo`) over an 18:00-to-next-noon window and
derives label-free low-HR, dark, no-screen, charging, stillness, and movement
event features. It finds only a small S4 lead: the selected S4 row wins `4/5`
primary lenses with mean delta `-0.000254566`, but S3 passes `0` rows. The
estimated all-target delta is just `-0.000036367`, projecting to Public
`0.596046885`, still `+0.046046885` from the `0.55` target.

Operational consequence: local raw HR/light/screen event reconstruction is also
closed as an immediate ID342 overlay route. It is not a Kaggle or DACON lane.
The next useful work should shift to validation/public-gap redesign, or to a
materially stronger source that can move S3 and S4 without expanding the known
failed public gap.

The cross-source split stability audit
`reports/post_id342_cross_source_split_stability_audit_20260625.md/.json`
checks whether the weak post-ID342 signals line up on the same splits instead
of failing independently. It reads only existing JSON reports and reduces `128`
candidate rows to `13` independent family-target representatives so weight-only
variants are not counted as independent evidence. The only stable clue is a
small S3 phone/GPS micro-alignment, but its best Public proxy is still
`0.596049063`. S4 has broader test-neighbor alignment but conflicts on LOSO,
and Q3 has no cross-source alignment. Review signal targets are `[]`; no
submission artifact, Kaggle run, submission-log row, or DACON submission is
opened.

Operational consequence: this gives a design hint, not a candidate. A future
S3 route can revisit phone/wearable plus GPS state, but only if it creates a
materially larger fold-internal signal and passes the stable public-gap gates.
The current evidence remains far from Public `0.55`.

The focused S3 phone/GPS follow-up
`reports/post_id342_s3_phone_gps_interaction_gate_20260625.md/.json` tests that
design hint directly using the registered processed phone/wearable discordance
and GPS path-topology matrices. It selects `37` phone columns and `15` GPS
columns, then adds `230` bounded phone×GPS interaction columns. The strongest
row is actually selected concat rather than interaction:
`s3_phone_gps_selected_concat_s3_lr_c0_20_w0.05` wins `5/5` primary splits with
S3 target mean delta `-0.001635922`, but projects only to Public `0.595849549`.
Local gate passes are `0`.

Operational consequence: the S3 phone/GPS clue is real enough to explain part
of the stable split behavior, but not remotely large enough for a Public
`0.55` attempt. Do not create an artifact, Kaggle run, submission-log row, or
DACON submission from this route.

The final-day independent signal sweep
`reports/post_id342_final_day_independent_signal_sweep_20260625.md/.json`
implements the remaining last-day plan without creating artifacts: S3
phone/GPS temporal deviations, S4 sleep-fragmentation x solar/calendar joint
state, and compact Q3/S3/S4 joint-state features. The best row is
`s4_sleep_solar_joint_s4_lr_c0_10_w0.05`, with `4/5` wins, mean delta
`-0.002013473`, max delta `+0.000266583`, and projected Public
`0.595795613`. Local gate passes are `4`, but review gate passes are `0` and
DACON gate passes are `0`.

Operational consequence: even the last-day independent sweep stays in the
Public `0.596` band. Do not create a submission artifact or spend a DACON slot
from this result.

The public-gap penalty proxy redesign
`reports/post_id342_public_gap_penalty_proxy_redesign_20260625.md/.json` now
tests whether a stricter validation proxy can rescue any OOF-shelf row. It fits
a small nonnegative penalty for prediction-shape excess over the stable ID342
envelope. The proxy matches public ordering better, but LOO error is unstable
and it finds `0` credible Public-0.55 rows.

The block-sequence feasibility contract has now been executed as
`reports/post_id342_mis_lstm_block_sequence_feasibility_gate_20260625.md/.json`.
It is a bounded local proxy for MIS-LSTM rather than a neural remote runner:
3-hour/4-hour continuous and sparse-event blocks are built in memory and tested
with fold-internal low-capacity heads. It selects only S4 (`4/5` wins, mean
delta `-0.001468349`), while Q3 and S3 both pass `0` rows. Projected Public
proxy is `0.595873488`, still `+0.045873488` from `0.55`.

Operational consequence: do not launch a MIS-LSTM Kaggle runner from the
current evidence.

The validation-weighted pseudo-public split audit has now run as
`reports/post_id342_validation_weighted_pseudo_public_split_audit_20260625.md/.json`.
It evaluates `15` target-weighted full/subject-hole/LOSO/test-neighbor lens
configs over `247` OOF files and `8` public-scored rows. The best audit lens
matches the stable public order exactly, but failed-frontier separation remains
negative; review-lens pass count is `0`. It finds `0` credible Public-0.55 rows
and `0` credible Public-0.58 review rows.

The first external-date source gate has also run as
`reports/post_id342_solar_seasonal_holiday_prior_gate_20260625.md/.json`.
It tests Seoul/Korea photoperiod, seasonal phase, weekday, and Korean
public-holiday proximity. It finds only unstable S4 movement: best mean delta
`-0.002238508`, but only `3/5` wins and max delta `+0.001238751`; eligible rows
are `0`.

The first historical weather source gate has also run as
`reports/post_id342_kma_weather_context_gate_20260625.md/.json`. The API
contract is documented in `docs/EXTERNAL_WEATHER_API_PLAN.md`, with keys kept
only in environment variables and raw caches ignored under
`data/external/weather/`. The gate fetched Daejeon ASOS station `133`
observations for `2024-06-01` through `2024-11-22` and tested thermal,
precipitation/pressure, solar/ground, and combined weather families against
ID342. It finds a tiny S4 precipitation/pressure hint:
`kma_weather_precip_pressure_s4_lr_c0_10_w0.05`, wins `2/5`, mean delta
`-0.000539108`, max delta `+0.000515499`, max failed-direction corr
`+0.269367`, projected Public `0.596006236`. Local/review/DACON pass counts
are all `0`; no artifact, Kaggle run, submission-log row, or DACON submission
is opened.

The approved public-data API smoke has also run as
`reports/public_data_weather_air_api_smoke_20260625.md/.json`, covering all
three user-provided service families. KMA ASOS historical access is confirmed:
the `2024-06-01` Daejeon station `133` probe returned `24/24` hourly rows.
KMA Living Weather Index current forecast calls work, but a `2024060106`
historical probe returns the service's one-day-window failure, so it cannot
support the 2024 train/test window. AirKorea `ArpltnInforInqireSvc` access
also works: the province realtime, station measurement, fine-dust forecast,
weekly forecast, and bad-or-worse CAI list endpoints all return normal API
responses. However, the station measurement endpoint is recent-only for this
competition purpose: the `3MONTH` smoke returned page-1 rows from
`2026-05-14 22:00` through `2026-06-25 16:00`, not the 2024 target period.

Operational consequence: the three approved API services are now actually
validated, not only read from docs. Only KMA ASOS is a confirmed 2024
historical feature source, and it has already failed the ID342 gate. KMA
Living Weather Index and the approved AirKorea information API stay
report-only unless a separate date-bounded historical/statistics source is
approved and then passes an ID342 local gate.

That separate AirKorea historical source is now present. The user-downloaded
`2024.zip` was moved to
`data/external/weather/airkorea_2024/airkorea_2024_hourly_all_stations_xlsx.zip`
and audited in
`reports/airkorea_2024_hourly_zip_audit_20260625.md/.json`. The archive has
monthly 2024 hourly nationwide air-quality XLSX files with `SO2`, `CO`, `O3`,
`NO2`, `PM10`, and `PM25`. The audit extracted ignored
`data/external/weather/airkorea_2024/airkorea_2024_daejeon_hourly.csv`,
covering 13 Daejeon stations and every hour in the `2024-06-01 00:00` through
`2024-11-22 23:00` external-feature window.

Operational consequence: AirKorea is reopened only as a new downloaded
historical-source gate, not as a submission candidate. That gate has now run as
`reports/post_id342_airkorea_2024_context_gate_20260625.md/.json`. It reviewed
`216` Q3/S3/S4 compact AirKorea overlays and passed `0` local gates, `0`
Public-0.58 review gates, and `0` DACON gates. The best row,
`airkorea_selected_station_s4_hgb_l7_w0.025`, only projects to Public
`0.596056065`, still `+0.046056065` from Public `0.55`, and fails stability
with wins `3/5`, max delta `+0.000819150`, and failed-frontier correlation
`+0.262872`.

The refreshed route synthesis now closes `30/30` registered routes, with ID342
still the best credible shape row under the validation-weighted projection at
`0.596211532`.

Operational consequence: the current plan is no longer "find another candidate
inside the registered route shelf" or "try another weather/air-quality date
prior." The top research route is now
`materially_new_external_source_hypothesis`, excluding the unstable
solar/holiday prior, the too-small KMA weather-context hint, and the failed
AirKorea 2024 context gate. No artifact, Kaggle run, or DACON submission is
justified until a materially different source passes local and
stable-shape/failed-frontier gates.

The historical ETRI sleep/survey prior was also audited in
`reports/post_id342_external_etri_sleep_survey_prior_20260625.md/.json`.
It matched `523` 2020 PM survey rows from `22` users to next-day sleep records.
This prior supports Q2 fatigue-to-sleep coupling, but it does not rescue the
needed Q3/S3/S4 path: lower-stress days did not align with better sleep
outcomes (`durationtosleep` delta `+58.624`, `wakeupduration` delta
`+58.421`, S3 0-30m delta `-0.059160`, S4 WASO<20m delta `-0.029484`), and no
direct 2026 feature route is available. Treat it as a warning against simple
Q3-from-sleep coupling, not as a candidate route.

On `2026-06-25 KST`, deadline-aware operating policy was generated in
`reports/post_id342_deadline_operating_policy_20260625.md/.json`. Public
sources checked that night show algorithm close on `2026-07-23`; the ICTC
official paper-submission page lists paper submission deadline `2026-06-26`,
corroborated by the DACON talkboard notice for the ETRI workshop track. Because
`32/32` routes are closed and no current candidate clears
local/stable-shape/failed-frontier gates, the policy is `run_kaggle_now=false`
and `dacon_submit_now=false`. The next deadline-critical artifact is the
technical paper/reproducibility draft
`docs/PAPER_SUBMISSION_DRAFT_20260626.md`, while leaderboard work remains gated
to short report-only or local Q3/S3/S4 experiments. The paired submission
checklist is `docs/PAPER_SUBMISSION_CHECKLIST_20260626.md`; DACON rules require
the IWETRIAI paper to use the Standard IEEE `4`-page full-paper format. The
working IEEE source draft is `docs/paper_submission_20260626/paper.tex`.
