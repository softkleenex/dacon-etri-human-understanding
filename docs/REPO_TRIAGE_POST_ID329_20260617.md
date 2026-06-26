# Repo triage after ID329 public result

Created: `2026-06-17 01:18:11` KST

No files were deleted, moved, or submitted while creating this triage ledger.

## Current dirty-tree shape

| bucket | count |
|---|---:|
| ` M` | 26 |
| `??` | 763 |

## Top-level counts

| top-level path | count |
|---|---:|
| `reports` | 679 |
| `scripts` | 86 |
| `docs` | 7 |
| `src` | 5 |
| `tests` | 4 |
| `notebooks` | 2 |
| `.gitignore` | 1 |
| `AGENTS.md` | 1 |
| `CHANGELOG.md` | 1 |
| `Makefile` | 1 |
| `README.md` | 1 |
| `submissions` | 1 |

## Triage categories

| category | count | immediate decision |
|---|---:|---|
| `bulk_experiment_report_archive_or_review` | 602 | archive or leave untracked; do not treat all as authoritative |
| `experiment_script_review_required` | 84 | freeze; promote only scripts needed to reproduce accepted artifacts |
| `generated_report_archive_or_review` | 61 | archive/ignore unless explicitly referenced by docs |
| `curated_report_promote_or_review` | 16 | promote as concise decision evidence |
| `operational_docs_promote_or_review` | 11 | promote/review as guardrails before more modeling |
| `source_or_tests_review_required` | 9 | review before use; do not mix into public-guided experiments silently |
| `audit_tooling_promote_or_review` | 2 | promote if needed by guardrails |
| `notebook_review_required` | 2 | review manually; notebooks are not authoritative without matching log/report |
| `canonical_submission_record_promote` | 1 | promote; this is canonical ID329 public evidence |
| `misc_review_required` | 1 | manual review |

## Blocking findings

- There are source/test changes and 84 untracked scripts; new modeling should not start until these are promoted, archived, or reverted intentionally.
- There are hundreds of untracked reports; only curated decision reports should be treated as authoritative.
- Makefile LINT_PATHS references many untracked scripts, so lint success currently depends on files not yet promoted into git.

## Safe next actions

- Promote/commit operational guardrails separately from experiment artifacts.
- Decide whether demographic feature code is part of the next validation redesign branch before using it in modeling.
- Archive bulk scan reports outside tracked decision docs, keeping only concise JSON/MD decision records.
- Do not submit any candidate until a decision note passes the ID178/ID329/ID287 public-anchor checklist.

## Samples by category

### audit_tooling_promote_or_review (2)

- `scripts/workspace_cleanup_manifest.py`
- `scripts/audit_experiment_artifacts.py`

### bulk_experiment_report_archive_or_review (602)

- `reports/candidate_tier_manifest_post_id188_20260616.json`
- `reports/candidate_tier_manifest_post_id188_20260616.md`
- `reports/candidate_tier_manifest_post_id189_20260616.json`
- `reports/candidate_tier_manifest_post_id189_20260616.md`
- `reports/candidate_tier_manifest_post_id192_20260616.json`
- `reports/candidate_tier_manifest_post_id192_20260616.md`
- `reports/candidate_tier_manifest_post_id194_20260616.json`
- `reports/candidate_tier_manifest_post_id194_20260616.md`
- `reports/candidate_tier_manifest_post_id197_20260616.json`
- `reports/candidate_tier_manifest_post_id197_20260616.md`
- `reports/candidate_tier_manifest_post_id199_20260616.json`
- `reports/candidate_tier_manifest_post_id199_20260616.md`
- `reports/candidate_tier_manifest_post_id201_20260616.json`
- `reports/candidate_tier_manifest_post_id201_20260616.md`
- `reports/candidate_tier_manifest_post_id202_20260616.json`
- `reports/candidate_tier_manifest_post_id202_20260616.md`
- `reports/candidate_tier_manifest_post_id205_20260616.json`
- `reports/candidate_tier_manifest_post_id205_20260616.md`
- `reports/candidate_tier_manifest_post_id206_20260616.json`
- `reports/candidate_tier_manifest_post_id206_20260616.md`
- `reports/candidate_tier_manifest_post_id208_20260616.json`
- `reports/candidate_tier_manifest_post_id208_20260616.md`
- `reports/candidate_tier_manifest_post_id211_20260616.json`
- `reports/candidate_tier_manifest_post_id211_20260616.md`
- `reports/candidate_tier_manifest_post_id214_20260616.json`
- ... 577 more

### canonical_submission_record_promote (1)

- `submissions/submission_log.csv`

### curated_report_promote_or_review (16)

- `reports/docs_records_audit.json`
- `reports/docs_records_audit.md`
- `reports/repository_structure_audit.json`
- `reports/repository_structure_audit.md`
- `reports/submissions_audit.json`
- `reports/submissions_audit.md`
- `reports/workspace_cleanup_manifest.md`
- `reports/candidate_tier_manifest_post_id333_fourthpass_combo_20260616.json`
- `reports/candidate_tier_manifest_post_id333_fourthpass_combo_20260616.md`
- `reports/dacon_submission_result_id178_public0.6014113009.json`
- `reports/dacon_submission_result_id287_public0.6698787318.json`
- `reports/dacon_submission_result_id329_public0.6017908577.json`
- `reports/post333_fifthpass_no_promotion_decision_20260616.json`
- `reports/post333_fifthpass_no_promotion_decision_20260616.md`
- `reports/public_score_response_diagnostic_post_id329_20260617.json`
- `reports/public_score_response_diagnostic_post_id329_20260617.md`

### experiment_script_review_required (84)

- `scripts/train_ultimate_pipeline_clean_v32.py`
- `scripts/diagnose_public_anchor_feature_meta_model.py`
- `scripts/diagnose_public_booster_stability_overlay.py`
- `scripts/diagnose_public_failed_direction_constrained_combo.py`
- `scripts/diagnose_public_outer_calibration_selection.py`
- `scripts/diagnose_public_random_conv_fine_selection.py`
- `scripts/diagnose_public_random_conv_sequence.py`
- `scripts/diagnose_public_sequence_summary_embedding.py`
- `scripts/diagnose_public_sleep_guideline_overlay.py`
- `scripts/run_public_all_temporal_anchor_fine_probe_experiment.py`
- `scripts/run_public_confirmed_anchor_bridge_probe_experiment.py`
- `scripts/run_public_convex_temporal_anchor_probe_experiment.py`
- `scripts/run_public_extreme_calibration_probe_experiment.py`
- `scripts/run_public_failed_direction_aggregate_budget_probe_experiment.py`
- `scripts/run_public_failed_direction_orthogonal_probe_experiment.py`
- `scripts/run_public_feature_similarity_overlay_experiment.py`
- `scripts/run_public_id163_targetwise_extreme_calibration_probe_experiment.py`
- `scripts/run_public_id168_calibration_probe_experiment.py`
- `scripts/run_public_knn_temporal_refine_probe_experiment.py`
- `scripts/run_public_mixed_temporal_anchor_probe_experiment.py`
- `scripts/run_public_multi_anchor_probe_experiment.py`
- `scripts/run_public_multi_signal_clean_overlay_experiment.py`
- `scripts/run_public_multiseed_random_conv_sequence_overlay_experiment.py`
- `scripts/run_public_outer_supported_calibration_probe_experiment.py`
- `scripts/run_public_pair_logit_blend_probe_experiment.py`
- ... 59 more

### generated_report_archive_or_review (61)

- `reports/submission_blacklist.json`
- `reports/submission_log_normalized.csv`
- `reports/workspace_audit.json`
- `reports/workspace_audit.md`
- `reports/all_base_oof_target_loss_map_20260616.json`
- `reports/broad_candidate_delta_map_post_id266_20260616.json`
- `reports/broad_threshold_rule_grid_id164.json`
- `reports/broad_threshold_rule_grid_id165.json`
- `reports/external_etri_prior_target_reconstruction_20260616.json`
- `reports/external_etri_prior_target_reconstruction_20260616.md`
- `reports/external_prior_constant_shrink_scan_20260616.json`
- `reports/external_prior_constant_shrink_scan_20260616.md`
- `reports/external_public05917_notebook_review_20260616.json`
- `reports/external_public05917_notebook_review_20260616.md`
- `reports/external_sleep_prior_threshold_diagnostic_id162.json`
- `reports/kaggle_next_signal_plan_id164.json`
- `reports/legacy_user_embedding_overlay_diagnostic_id129.json`
- `reports/multi_signal_clean_review_post_id220_20260616.json`
- `reports/multi_signal_clean_review_post_id220_20260616.md`
- `reports/recurrence_gaf_sequence_review_post_id205_20260616.json`
- `reports/recurrence_gaf_sequence_review_post_id205_20260616.md`
- `reports/remote_random_conv_sweep_review_post_id192_20260616.json`
- `reports/remote_random_conv_sweep_review_post_id192_20260616.md`
- `reports/remote_strict_manifest.json`
- `reports/remote_strict_manifest_id247_s3_random_conv_v25_20260616.json`
- ... 36 more

### misc_review_required (1)

- `.gitignore`

### notebook_review_required (2)

- `notebooks/kaggle_runner/SOTA_DL_Anchor_Training_Pipeline.ipynb`
- `notebooks/kaggle_strict_runner/`

### operational_docs_promote_or_review (11)

- `AGENTS.md`
- `CHANGELOG.md`
- `Makefile`
- `README.md`
- `docs/EXPERIMENT_OPERATIONS.md`
- `docs/MASTER_PLAN.md`
- `docs/infrastructure/INFRA_EVALUATION_AND_PLAN.md`
- `docs/infrastructure/README.md`
- `docs/infrastructure/cloud_env/guide.md`
- `docs/PUBLIC_055_PLAN.md`
- `docs/WORKFLOW_REVIEW.md`

### source_or_tests_review_required (9)

- `src/data/make_dataset.py`
- `src/features/demographics.py`
- `src/features/morning_recovery.py`
- `src/features/q_context.py`
- `src/features/shutdown.py`
- `tests/test_demographics.py`
- `tests/test_morning_recovery_features.py`
- `tests/test_q_context_features.py`
- `tests/test_shutdown_features.py`
