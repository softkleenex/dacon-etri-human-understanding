.PHONY: help install lint format test audit docs-audit submissions-audit artifact-contract-audit public-local-gap-diagnosis data-source-coverage-audit q2q3s3-source-hypothesis-matrix manual-submission-probe-decision post-id341-submission-spend-policy post-id341-safe-wait-state post-id341-next-route-triage post-id341-sleep-proxy-design post-id341-local-compute-boundary post-id342-research-gap post-id342-raw-temporal-q3s3-smoke post-id342-sequence-summary-distance-smoke post-id342-sleep-boundary-q3s3-smoke post-id342-tiny-temporal-cnn-q3s3-smoke post-id342-public-response-geometry post-id342-public-alignment-lens post-id342-stable-lens-triage post-id342-env-social-routine-gate post-id342-evening-state-transition-gate post-id342-public-scored-target-swap-gate post-id342-q3s3-slice-selector-anatomy post-id342-lightweight-subject-state-pca-gate post-id342-processed-sleep-proxy-gate post-id342-processed-sleep-temporal-deviation-gate post-id342-subject-prior-calibration-gate post-id342-probability-temperature-gate post-id342-remaining-route-synthesis post-id342-self-supervised-heavy-contract post-id342-self-supervised-route-research post-id342-goal-state-audit post-id342-feature-similarity-decomp post-id342-feature-similarity-artifact-gate post-id342-paper-aligned-featuresim-gate post-id342-past-only-temporal-gate post-id342-cross-target-consistency-gate post-id342-material-source-rescan post-id342-material-fusion-gate post-id342-s3q3-raw-sequence-representation-gate structure-audit normalize-submissions cleanup-manifest run-local build-kaggle-dataset build-kaggle-upstream-patch-dataset kaggle-init-code kaggle-push-code kaggle-push-upstream-patch-dataset kaggle-init-notebook kaggle-push-notebook kaggle-push-only kaggle-push-strict-runner kaggle-push-sequence-source-runner kaggle-push-sequence-anchor-probe-runner kaggle-push-candidate-parity-runner kaggle-push-upstream-rebuild-runner kaggle-check-base-dataset-files kaggle-check-upstream-patch-dataset-files kaggle-kernel-status-upstream-rebuild kaggle-pull-output kaggle-pull-strict-output kaggle-pull-sequence-source-output kaggle-pull-sequence-anchor-probe-output kaggle-pull-candidate-parity-output kaggle-pull-upstream-rebuild-output kaggle-wait-and-pull-upstream-rebuild-output kaggle-upstream-prelaunch-local kaggle-approval-contract-test dacon-submit-approval-contract-test kaggle-local-runner-smoke kaggle-upstream-ready-local kaggle-review-upstream-rebuild-output require-kaggle-remote-approval require-dacon-submit-approval dacon-submit ensemble
.PHONY: post-id342-self-supervised-heavy-runner-spec
.PHONY: post-id342-subject-state-domain-token-contract
.PHONY: post-id342-subject-state-raw-schema-contract
.PHONY: post-id342-subject-state-heavy-approval-readiness
.PHONY: post-id342-subject-state-heavy-success-thresholds
.PHONY: post-id342-kaggle-input-availability
.PHONY: post-id342-kaggle-cli-health
.PHONY: post-id342-public055-gap-attribution
.PHONY: post-id342-public-gap-risk-lens
.PHONY: post-id342-public-gap-penalty-proxy-redesign
.PHONY: post-id342-validation-weighted-pseudo-public-split-audit
.PHONY: post-id342-solar-seasonal-holiday-prior-gate
.PHONY: fetch-kma-asos-hourly-context
.PHONY: post-id342-kma-weather-context-gate
.PHONY: smoke-public-data-weather-air-api
.PHONY: audit-airkorea-2024-hourly-zip
.PHONY: post-id342-airkorea-2024-context-gate
.PHONY: post-id342-external-etri-sleep-survey-prior
.PHONY: post-id342-deadline-operating-policy
.PHONY: paper-submission-metadata
.PHONY: paper-submission-pdf
.PHONY: paper-submission-length-audit
.PHONY: paper-submission-source-archive
.PHONY: paper-submission-package-audit
.PHONY: post-id342-stable-envelope-multisource-optimizer
.PHONY: post-id342-target-reallocation-optimizer
.PHONY: post-id342-s4s3-sleep-fragmentation-gate
.PHONY: post-id342-raw-sleep-event-feasibility
.PHONY: post-id342-sequence-hr-light-event-gate
.PHONY: post-id342-mis-lstm-block-sequence-feasibility-gate
.PHONY: post-id342-morning-closeout
.PHONY: post-id342-external-research-priority
.PHONY: post-id342-heavy-implementation-feasibility
.PHONY: post-id342-heavy-runtime-budget
.PHONY: post-id342-materially-new-source-hypothesis
.PHONY: post-id342-self-supervised-subject-state-local-smoke
.PHONY: post-id342-self-supervised-subject-state-remote-intake
.PHONY: post-id342-missingness-subject-state-local-smoke
.PHONY: post-id342-missingness-subject-state-remote-intake
.PHONY: post-id342-raw-subject-state-feature-gate
.PHONY: build-kaggle-self-supervised-subject-state-patch-dataset
.PHONY: kaggle-push-self-supervised-subject-state-patch-dataset
.PHONY: kaggle-check-self-supervised-subject-state-patch-dataset-files
.PHONY: kaggle-push-self-supervised-subject-state-runner
.PHONY: kaggle-push-missingness-subject-state-runner
.PHONY: kaggle-kernel-status-self-supervised-subject-state
.PHONY: kaggle-kernel-status-missingness-subject-state
.PHONY: kaggle-pull-self-supervised-subject-state-output
.PHONY: kaggle-pull-missingness-subject-state-output
.PHONY: kaggle-wait-and-pull-self-supervised-subject-state-output
.PHONY: kaggle-wait-and-pull-missingness-subject-state-output

KAGGLE ?= .venv/bin/kaggle
PAPER_PYTHON ?= python3
TECTONIC ?= tectonic
KAGGLE_DATASET_DIR := build/kaggle_dataset
KAGGLE_UPSTREAM_PATCH_DATASET_DIR := build/kaggle_upstream_patch_dataset
KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_DIR := build/kaggle_self_supervised_subject_state_patch_dataset
KAGGLE_UPSTREAM_PATCH_DATASET_SLUG ?= softkleenex/dacon-etri-upstream-patch-post-id341
KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_SLUG ?= softkleenex/dacon-etri-ss-subject-state-patch-id342
KAGGLE_NOTEBOOK_DIR ?= notebooks/kaggle_runner
KAGGLE_SLUG ?= softkleenex/dacon-etri-dl-tuning-v10
KAGGLE_STRICT_SLUG ?= softkleenex/dacon-etri-strict-runner
KAGGLE_SEQUENCE_SOURCE_SLUG ?= softkleenex/dacon-etri-sequence-source-post-id335
KAGGLE_SEQUENCE_ANCHOR_PROBE_SLUG ?= softkleenex/dacon-etri-sequence-anchor-probe-post-id335
KAGGLE_CANDIDATE_PARITY_SLUG ?= softkleenex/dacon-etri-candidate-parity-post-id339
KAGGLE_UPSTREAM_REBUILD_SLUG ?= softkleenex/dacon-etri-upstream-rebuild-post-id341
KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_SLUG ?= softkleenex/dacon-etri-ss-subject-state-id342
KAGGLE_MISSINGNESS_SUBJECT_STATE_SLUG ?= softkleenex/dacon-etri-missingness-subject-state-id342
KAGGLE_STATUS_OBSERVATION ?= not_checked_by_make
KAGGLE_ENABLE_GPU ?= false
KAGGLE_ENABLE_TPU ?= true
PULL_DIR ?= ./submissions
AUTO_SUBMIT ?= false
ALLOW_KAGGLE_REMOTE ?= false
ALLOW_DACON_SUBMIT ?= false
DACON_APPROVED_SUB_PATH ?=
DACON_APPROVED_SUB_MEMO ?=
DACON_SUBMIT_POLICY_REPORT ?= reports/post_id341_submission_spend_policy_20260619.json
COMPETITION_ID ?= 236690 # Update this to your actual Dacon competition ID if needed

help:
	@echo "Available commands:"
	@echo "  make install           - Install dependencies"
	@echo "  make lint              - Run Ruff linter"
	@echo "  make format            - Run Ruff formatter"
	@echo "  make test              - Run Pytest"
	@echo "  make audit             - Audit workspace health and experiment hygiene"
	@echo "  make docs-audit        - Audit docs, commands, and submission record integrity"
	@echo "  make submissions-audit - Audit submission artifacts and log integrity"
	@echo "  make artifact-contract-audit - Audit notebook/submission/log experiment matching"
	@echo "  make q2q3s3-source-hypothesis-matrix - Refresh report-only Q2/Q3/S3 source plan"
	@echo "  make manual-submission-probe-decision - Refresh report-only manual DACON probe decision"
	@echo "  make post-id341-submission-spend-policy - Refresh report-only DACON submission spend policy"
	@echo "  make post-id341-safe-wait-state - Refresh report-only post-ID341 safe-wait contract"
	@echo "  make post-id341-next-route-triage - Refresh report-only post-ID341 next route triage"
	@echo "  make post-id341-sleep-proxy-design - Refresh report-only sleep-proxy backlog design"
	@echo "  make post-id341-local-compute-boundary - Validate local/Kaggle compute boundary contract"
	@echo "  make post-id342-research-gap - Refresh report-only ID342 source-priority diagnosis"
	@echo "  make post-id342-raw-temporal-q3s3-smoke - Run report-only raw temporal Q3/S3 smoke"
	@echo "  make post-id342-sequence-summary-distance-smoke - Run report-only ID342 sequence-summary distance/kernel Q3/S3 smoke"
	@echo "  make post-id342-sleep-boundary-q3s3-smoke - Run report-only ID342 sleep-boundary/SOL Q3/S3 smoke"
	@echo "  make post-id342-tiny-temporal-cnn-q3s3-smoke - Run report-only ID342 tiny temporal-CNN Q3/S3 smoke"
	@echo "  make post-id342-public-response-geometry - Explain ID342 public response from scored slate geometry"
	@echo "  make post-id342-public-alignment-lens - Diagnose which local validation lenses explain public scores"
	@echo "  make post-id342-public055-gap-attribution - Quantify Public 0.55 local-gap requirements over the OOF shelf"
	@echo "  make post-id342-public-gap-risk-lens - Derive a conservative public-gap risk lens from scored OOF prediction shapes"
	@echo "  make post-id342-public-gap-penalty-proxy-redesign - Fit a stricter shape-penalty public-gap proxy over the OOF shelf"
	@echo "  make post-id342-validation-weighted-pseudo-public-split-audit - Audit target-weighted pseudo-public validation lenses"
	@echo "  make post-id342-solar-seasonal-holiday-prior-gate - Gate external solar/seasonal/Korean-holiday priors"
	@echo "  make fetch-kma-asos-hourly-context - Fetch ignored KMA ASOS hourly weather cache from env key"
	@echo "  make post-id342-kma-weather-context-gate - Gate KMA ASOS weather context against ID342"
	@echo "  make smoke-public-data-weather-air-api - Smoke-test approved KMA/AirKorea APIs from env keys"
	@echo "  make audit-airkorea-2024-hourly-zip - Audit downloaded 2024 AirKorea hourly XLSX archive"
	@echo "  make post-id342-airkorea-2024-context-gate - Gate downloaded AirKorea 2024 context against ID342"
	@echo "  make post-id342-external-etri-sleep-survey-prior - Audit historical ETRI sleep/survey priors"
	@echo "  make post-id342-deadline-operating-policy - Refresh deadline-aware operating policy"
	@echo "  make paper-submission-metadata - Extract EDAS title/abstract/keywords from paper.tex"
	@echo "  make paper-submission-pdf - Compile the IEEE paper with Tectonic"
	@echo "  make paper-submission-length-audit - Audit paper.tex source length budget"
	@echo "  make paper-submission-source-archive - Build Overleaf/TeX-ready paper source ZIP"
	@echo "  make paper-submission-package-audit - Audit deadline IEEE paper package claims and hygiene"
	@echo "  make post-id342-stable-envelope-multisource-optimizer - Test nonlegacy multi-source blends inside the stable envelope"
	@echo "  make post-id342-target-reallocation-optimizer - Test strict target-wise source reallocation inside the stable envelope"
	@echo "  make post-id342-s4s3-sleep-fragmentation-gate - Gate S4/S3 processed WASO/SOL sleep-fragmentation features"
	@echo "  make post-id342-raw-sleep-event-feasibility - Audit raw schema support for direct S4/S3 sleep-event reconstruction"
	@echo "  make post-id342-sequence-hr-light-event-gate - Gate raw HR/light/screen sequence event proxies for S4/S3"
	@echo "  make post-id342-mis-lstm-block-sequence-feasibility-gate - Gate MIS-LSTM-style block-sequence proxies locally"
	@echo "  make post-id342-stable-lens-triage - Triage existing OOF candidates under the stable-lens policy"
	@echo "  make post-id342-env-social-routine-gate - Gate env/social/routine v2 features against ID342"
	@echo "  make post-id342-evening-state-transition-gate - Gate evening arousal/settling transitions against ID342"
	@echo "  make post-id342-public-scored-target-swap-gate - Gate public-scored Q3/S3 target swaps against ID342"
	@echo "  make post-id342-q3s3-slice-selector-anatomy - Diagnose Q3/S3 subject slices and stable-donor selector"
	@echo "  make post-id342-lightweight-subject-state-pca-gate - Gate lightweight PCA subject-state proxies against ID342"
	@echo "  make post-id342-processed-sleep-proxy-gate - Gate processed sleep proxy columns against ID342"
	@echo "  make post-id342-processed-sleep-temporal-deviation-gate - Gate processed sleep lag/deviation columns against ID342"
	@echo "  make post-id342-subject-prior-calibration-gate - Gate Q3/S3 subject-prior calibration against ID342"
	@echo "  make post-id342-probability-temperature-gate - Gate ID342 probability contract and temperature calibration"
	@echo "  make post-id342-remaining-route-synthesis - Synthesize closed ID342 routes and remaining score-relevant work"
	@echo "  make post-id342-self-supervised-heavy-contract - Refresh the post-ID342 heavy approval boundary"
	@echo "  make post-id342-self-supervised-route-research - Freeze the blocked heavy subject-state route research packet"
	@echo "  make post-id342-self-supervised-heavy-runner-spec - Define the blocked no-submit heavy Kaggle runner spec"
	@echo "  make post-id342-subject-state-domain-token-contract - Freeze the heavy route modality/token contract"
	@echo "  make post-id342-subject-state-raw-schema-contract - Validate heavy route raw schemas without payload reads"
	@echo "  make post-id342-subject-state-heavy-approval-readiness - Audit heavy route readiness before exact approval"
	@echo "  make post-id342-subject-state-heavy-success-thresholds - Define review thresholds for future approved heavy output"
	@echo "  make post-id342-kaggle-input-availability - Check future heavy runner Kaggle input datasets"
	@echo "  make post-id342-kaggle-cli-health - Check Kaggle CLI version/list/status health"
	@echo "  make post-id342-morning-closeout - Consolidate current post-ID342 morning operating state"
	@echo "  make post-id342-external-research-priority - Prioritize the remaining heavy route from external research"
	@echo "  make post-id342-heavy-implementation-feasibility - Audit approved-later heavy route implementation readiness"
	@echo "  make post-id342-heavy-runtime-budget - Estimate approved-later heavy route runtime budget from metadata"
	@echo "  make post-id342-materially-new-source-hypothesis - Define non-repeated Q3/S3/S4 source hypotheses after closed routes"
	@echo "  make post-id342-self-supervised-subject-state-local-smoke - Run bounded no-submit raw subject-state smoke"
	@echo "  make post-id342-self-supervised-subject-state-remote-intake - Inspect pulled no-submit subject-state Kaggle outputs"
	@echo "  make post-id342-missingness-subject-state-local-smoke - Run bounded no-submit missingness subject-state smoke"
	@echo "  make post-id342-missingness-subject-state-remote-intake - Inspect pulled no-submit missingness subject-state Kaggle outputs"
	@echo "  make post-id342-raw-subject-state-feature-gate - Gate raw subject-state daily/window features against ID342"
	@echo "  make post-id342-s3q3-raw-sequence-representation-gate - Run bounded local S3/Q3 raw sequence representation gate"
	@echo "  make post-id342-goal-state-audit - Audit current active-goal status and action gates"
	@echo "  make post-id342-feature-similarity-decomp - Decompose ID342's ID221 feature-similarity source"
	@echo "  make post-id342-feature-similarity-artifact-gate - Strictly gate ID342 alpha rows before artifacts/Kaggle"
	@echo "  make post-id342-paper-aligned-featuresim-gate - Gate paper-aligned ID342 feature-similarity rebuild"
	@echo "  make post-id342-past-only-temporal-gate - Gate past-only temporal recency anchors on ID342"
	@echo "  make post-id342-cross-target-consistency-gate - Gate cross-target consistency on ID342"
	@echo "  make post-id342-material-source-rescan - Rescan registered material matrices against ID342"
	@echo "  make post-id342-material-fusion-gate - Gate fused registered material matrices against ID342"
	@echo "  make structure-audit   - Audit recursive repository structure"
	@echo "  make normalize-submissions - Normalize submission log and blacklist risky files"
	@echo "  make cleanup-manifest  - Summarize intended cleanup vs review-required changes"
	@echo "  make run-local         - Run full local SOTA training/stacking pipeline (heavy)"
	@echo "  make build-kaggle-dataset - Package source code for Kaggle Dataset"
	@echo "  make build-kaggle-upstream-patch-dataset - Package the small post-ID341 upstream patch dataset"
	@echo "  make build-kaggle-self-supervised-subject-state-patch-dataset - Package minimal post-ID342 no-submit runner patch dataset"
	@echo "  make kaggle-init-code  - Initialize Kaggle dataset metadata"
	@echo "  make kaggle-push-code  - Push packaged source code to Kaggle (requires ALLOW_KAGGLE_REMOTE=true)"
	@echo "  make kaggle-push-upstream-patch-dataset - Push upstream patch dataset (requires ALLOW_KAGGLE_REMOTE=true)"
	@echo "  make kaggle-push-self-supervised-subject-state-patch-dataset - Push minimal no-submit runner patch dataset (requires ALLOW_KAGGLE_REMOTE=true)"
	@echo "  make kaggle-push-notebook - Push Kaggle notebook/kernel metadata"
	@echo "  make kaggle-push-strict-runner - Push CPU-only strict experiment runner"
	@echo "  make kaggle-push-sequence-source-runner - Push CPU-only post-ID335 sequence source runner"
	@echo "  make kaggle-push-sequence-anchor-probe-runner - Push CPU-only post-ID335 sequence anchor probe runner"
	@echo "  make kaggle-push-candidate-parity-runner - Push CPU-only post-ID339 candidate parity runner"
	@echo "  make kaggle-push-upstream-rebuild-runner - Push CPU-only post-ID341 upstream rebuild blueprint runner (requires ALLOW_KAGGLE_REMOTE=true)"
	@echo "  make kaggle-push-self-supervised-subject-state-runner - Push CPU-only post-ID342 no-submit subject-state runner (requires ALLOW_KAGGLE_REMOTE=true)"
	@echo "  make kaggle-push-missingness-subject-state-runner - Push CPU-only post-ID342 missingness subject-state runner (requires ALLOW_KAGGLE_REMOTE=true)"
	@echo "  make kaggle-check-base-dataset-files - List Kaggle base dataset files (requires ALLOW_KAGGLE_REMOTE=true)"
	@echo "  make kaggle-check-upstream-patch-dataset-files - List Kaggle upstream patch dataset files (requires ALLOW_KAGGLE_REMOTE=true)"
	@echo "  make kaggle-check-self-supervised-subject-state-patch-dataset-files - List no-submit runner patch dataset files (requires ALLOW_KAGGLE_REMOTE=true)"
	@echo "  make kaggle-kernel-status-upstream-rebuild - Check upstream runner status (requires ALLOW_KAGGLE_REMOTE=true)"
	@echo "  make kaggle-kernel-status-self-supervised-subject-state - Check no-submit subject-state runner status (requires ALLOW_KAGGLE_REMOTE=true)"
	@echo "  make kaggle-kernel-status-missingness-subject-state - Check missingness subject-state runner status (requires ALLOW_KAGGLE_REMOTE=true)"
	@echo "  make kaggle-wait-and-pull-upstream-rebuild-output - Wait for upstream runner and pull outputs (requires ALLOW_KAGGLE_REMOTE=true)"
	@echo "  make kaggle-wait-and-pull-self-supervised-subject-state-output - Wait for no-submit subject-state runner and pull outputs (requires ALLOW_KAGGLE_REMOTE=true)"
	@echo "  make kaggle-wait-and-pull-missingness-subject-state-output - Wait for missingness subject-state runner and pull outputs (requires ALLOW_KAGGLE_REMOTE=true)"
	@echo "  make kaggle-upstream-prelaunch-local - Run local no-submit prelaunch gates"
	@echo "  make kaggle-approval-contract-test - Run local focused Kaggle approval/guard contract checks"
	@echo "  make kaggle-local-runner-smoke - Validate upstream runner output contract locally without Kaggle network"
	@echo "  make kaggle-upstream-ready-local - Run full no-network readiness chain before Kaggle approval"
	@echo "  make kaggle-review-upstream-rebuild-output - Run local no-submit post-pull intake/summary/decision"
	@echo "  make dacon-submit-approval-contract-test - Run local fail-closed DACON submit guard checks"
	@echo "  make kaggle-pull-output - Pull outputs from Kaggle notebook (requires ALLOW_KAGGLE_REMOTE=true)"
	@echo "  make dacon-submit      - Submit to Dacon only with exact ALLOW_DACON_SUBMIT/DACON_APPROVED_SUB_PATH/DACON_APPROVED_SUB_MEMO"

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e .
	@echo "Installing Dacon API from global backup..."
	.venv/bin/pip install ~/sysp/dacon_api/dacon_submit_api-0.0.4-py3-none-any.whl

LINT_PATHS := src tests scripts/audit_competition_workspace.py scripts/ensemble.py scripts/normalize_submission_log.py scripts/workspace_cleanup_manifest.py scripts/diagnose_post_id341_kaggle_output_intake_gate.py scripts/diagnose_post_id341_kaggle_runner_intake_contract.py scripts/diagnose_post_id341_kaggle_prelaunch_readiness_gate.py scripts/audit_kaggle_guarded_remote_contract.py scripts/audit_kaggle_remote_fail_closed_dryrun.py scripts/diagnose_kaggle_source_zip_selftest_post_id341.py scripts/diagnose_kaggle_upstream_execution_packet_post_id341.py scripts/diagnose_post_id341_kaggle_launch_approval_note.py scripts/diagnose_post_id341_target_gap_budget.py scripts/diagnose_post_id341_missing_target_signal_inventory.py scripts/diagnose_post_id341_q2q3s3_source_hypothesis_matrix.py scripts/diagnose_post_id341_q2q3s3_paper_family_fold_smoke.py scripts/diagnose_post_id341_q3s3_anchor_failure_profile.py scripts/diagnose_post_id341_q2q3s3_local_fallback_chain_contract.py scripts/diagnose_post_id341_local_smoke_freshness_gate.py scripts/run_post_id341_kaggle_runner_local_smoke.py scripts/summarize_post_id341_kaggle_remote_results.py scripts/decide_post_id341_remote_next_action.py scripts/convert_pdfs_to_markdown.py scripts/audit_docs_records.py scripts/audit_repository_structure.py scripts/audit_submissions.py scripts/audit_experiment_artifacts.py scripts/run_strict_meta_stacking_experiment.py scripts/run_strict_raw_feature_experiment.py scripts/run_strict_shutdown_feature_experiment.py scripts/run_strict_shutdown_feature_set_selector_experiment.py scripts/run_strict_q23_strain_experiment.py scripts/run_strict_artifact_selector_experiment.py scripts/run_strict_q3_raw_q23_experiment.py scripts/run_strict_q2_q23_grid_experiment.py scripts/run_strict_q1_morning_recovery_experiment.py scripts/run_strict_q1_temperature_confirm_experiment.py scripts/run_strict_q3_q23_compact_experiment.py scripts/run_strict_q2_booster_compact_experiment.py scripts/run_strict_q3_booster_compact_experiment.py scripts/run_strict_s4_cross_target_compact_experiment.py scripts/run_strict_q3_cross_target_compact_experiment.py scripts/run_strict_s2_cross_target_compact_experiment.py scripts/run_strict_q2_app_token_compact_experiment.py scripts/run_strict_q2_subject_relative_app_experiment.py scripts/run_strict_q2_env_token_compact_experiment.py scripts/run_strict_q2_app_category_compact_experiment.py scripts/run_strict_q2_app_category_confirm_experiment.py scripts/run_strict_q3_app_category_compact_experiment.py scripts/run_strict_q3_app_category_confirm_experiment.py scripts/run_strict_q3_joint_q23_app_category_diagnostic.py scripts/run_strict_q3_subject_relative_compact_experiment.py scripts/run_strict_ambience_context_experiment.py scripts/run_strict_s4_ambience_context_confirm_experiment.py scripts/run_strict_q2_app_token_reblend_experiment.py scripts/run_strict_hourly_sequence_overlay_diagnostic.py scripts/run_strict_q3_gps_routine_overlay_experiment.py scripts/run_strict_chronos_embedding_overlay_diagnostic.py scripts/run_strict_temporal_hole_anchor_selector_experiment.py scripts/run_strict_periodic_anchor_overlay_experiment.py scripts/run_strict_feature_similarity_anchor_overlay_experiment.py scripts/run_strict_s3_refined_date_similarity_anchor_experiment.py scripts/run_strict_fixed_feature_similarity_overlay_experiment.py scripts/run_strict_feature_similarity_grid_diagnostic.py scripts/run_strict_fixed_threshold_rule_overlay_experiment.py scripts/run_strict_fixed_tree_rule_overlay_experiment.py scripts/run_public_temporal_anchor_probe_experiment.py scripts/run_public_temporal_anchor_fine_probe_experiment.py scripts/run_public_all_temporal_anchor_fine_probe_experiment.py scripts/run_public_mixed_temporal_anchor_probe_experiment.py scripts/run_public_convex_temporal_anchor_probe_experiment.py scripts/run_public_sequence_s34_overlay_experiment.py scripts/run_public_target_calibration_probe_experiment.py scripts/run_public_extreme_calibration_probe_experiment.py scripts/run_public_tempered_extreme_calibration_probe_experiment.py scripts/run_public_knn_temporal_refine_probe_experiment.py scripts/run_public_refined_anchor_extreme_blend_probe_experiment.py scripts/run_public_multi_anchor_probe_experiment.py scripts/run_public_id168_calibration_probe_experiment.py scripts/diagnose_public_outer_calibration_selection.py scripts/run_public_outer_supported_calibration_probe_experiment.py scripts/run_public_q40_extreme_calibration_probe_experiment.py scripts/run_public_id163_targetwise_extreme_calibration_probe_experiment.py scripts/run_public_s4_hourly_sequence_overlay_experiment.py scripts/diagnose_public_random_conv_sequence.py scripts/diagnose_public_sequence_summary_embedding.py scripts/diagnose_public_random_conv_fine_selection.py scripts/run_public_multiseed_random_conv_sequence_overlay_experiment.py scripts/run_public_random_conv_tempered_anchor_blend_experiment.py scripts/run_public_sequence_summary_embedding_overlay_experiment.py scripts/diagnose_public_anchor_feature_meta_model.py scripts/diagnose_public_sleep_guideline_overlay.py scripts/diagnose_public_booster_stability_overlay.py scripts/run_public_smartphone_light_compact_overlay_experiment.py scripts/run_public_sax_paa_sequence_overlay_experiment.py scripts/run_public_recurrence_gaf_sequence_overlay_experiment.py scripts/run_public_stability_selected_raw_overlay_experiment.py scripts/run_public_wavelet_spectral_sequence_overlay_experiment.py scripts/run_public_shapelet_motif_sequence_overlay_experiment.py scripts/run_public_shapelet_motif_alltop40_sequence_overlay_experiment.py scripts/run_public_window_pair_sequence_overlay_experiment.py scripts/run_public_multi_signal_clean_overlay_experiment.py scripts/run_public_feature_similarity_overlay_experiment.py scripts/run_public_pair_logit_blend_probe_experiment.py scripts/run_public_second_stage_calibration_probe_experiment.py scripts/run_public_targetwise_delta_capped_bridge_probe_experiment.py scripts/run_public_confirmed_anchor_bridge_probe_experiment.py scripts/run_public_failed_direction_orthogonal_probe_experiment.py scripts/run_public_failed_direction_aggregate_budget_probe_experiment.py scripts/diagnose_public_failed_direction_constrained_combo.py scripts/diagnose_post_id341_post_pull_review_chain_contract.py scripts/diagnose_post_id341_q3s3_semantic_routine_v2_design.py scripts/diagnose_post_id341_q3s3_semantic_routine_metadata_smoke.py scripts/diagnose_post_id341_q3s3_semantic_routine_resource_gate.py scripts/diagnose_post_id341_q3s3_semantic_routine_model_smoke.py scripts/diagnose_post_id341_q3s3_semantic_routine_lane_closure.py scripts/diagnose_post_id341_q3s3_calendar_sleep_prior_smoke.py scripts/diagnose_post_id341_q3s3_subject_drift_gate.py scripts/diagnose_post_id341_q3s3_event_cadence_smoke.py scripts/diagnose_post_id341_q3s3_cross_modal_disagreement_smoke.py scripts/diagnose_post_id341_q3s3_negative_evidence_map.py scripts/diagnose_post_id341_q3s3_target_anatomy.py scripts/diagnose_post_id341_q3s3_sibling_target_bridge_design.py scripts/diagnose_post_id341_q3s3_fold_nested_sibling_bridge_smoke.py scripts/diagnose_post_id341_q3s3_remaining_route_inventory.py scripts/diagnose_post_id341_manual_submission_probe_decision.py scripts/diagnose_post_id341_no_approval_frontier.py scripts/diagnose_post_id341_operator_handoff_contract.py scripts/diagnose_post_id341_runner_score_relevance_contract.py scripts/diagnose_post_id341_material_new_source_queue.py scripts/diagnose_post_id341_circadian_hr_light_phase_metadata_probe.py scripts/diagnose_post_id341_circadian_hr_light_phase_feature_spec.py scripts/diagnose_post_id341_circadian_hr_light_phase_feature_generation_dryrun.py scripts/diagnose_post_id341_gps_path_topology_metadata_probe.py scripts/diagnose_post_id341_gps_path_topology_feature_spec.py scripts/diagnose_post_id341_gps_path_topology_feature_generation_dryrun.py scripts/diagnose_post_id341_gps_path_topology_feature_matrix_smoke_contract.py scripts/write_post_id341_gps_path_topology_feature_matrix.py scripts/diagnose_post_id341_gps_path_topology_model_smoke.py
KAGGLE_APPROVAL_CONTRACT_LINT_PATHS := tests/test_kaggle_remote_make_guards.py tests/test_kaggle_guarded_remote_contract_audit.py tests/test_kaggle_remote_fail_closed_dryrun_audit.py tests/test_post_id341_kaggle_output_intake_gate.py tests/test_post_id341_kaggle_prelaunch_gate.py tests/test_post_id341_kaggle_remote_summary.py tests/test_post_id341_kaggle_runner_intake_contract.py tests/test_post_id341_kaggle_source_package_contract.py tests/test_post_id341_kaggle_ready_local_target.py tests/test_post_id341_post_pull_review_chain_contract.py tests/test_post_id341_remote_artifact_review_threshold.py tests/test_post_id341_remote_next_action.py tests/test_post_id341_operator_handoff_contract.py scripts/audit_kaggle_guarded_remote_contract.py scripts/audit_kaggle_remote_fail_closed_dryrun.py scripts/diagnose_kaggle_source_zip_selftest_post_id341.py scripts/diagnose_kaggle_upstream_execution_packet_post_id341.py scripts/diagnose_post_id341_kaggle_launch_approval_note.py scripts/diagnose_post_id341_kaggle_output_intake_gate.py scripts/diagnose_post_id341_kaggle_prelaunch_readiness_gate.py scripts/diagnose_post_id341_kaggle_runner_intake_contract.py scripts/diagnose_post_id341_post_pull_review_chain_contract.py scripts/diagnose_post_id341_operator_handoff_contract.py scripts/summarize_post_id341_kaggle_remote_results.py scripts/decide_post_id341_remote_next_action.py scripts/decide_post_id341_remote_artifact_review_threshold.py
KAGGLE_APPROVAL_CONTRACT_TESTS := tests/test_kaggle_remote_make_guards.py tests/test_kaggle_guarded_remote_contract_audit.py tests/test_kaggle_remote_fail_closed_dryrun_audit.py tests/test_post_id341_kaggle_output_intake_gate.py tests/test_post_id341_kaggle_prelaunch_gate.py tests/test_post_id341_kaggle_remote_summary.py tests/test_post_id341_kaggle_runner_intake_contract.py tests/test_post_id341_kaggle_source_package_contract.py tests/test_post_id341_kaggle_ready_local_target.py tests/test_post_id341_post_pull_review_chain_contract.py tests/test_post_id341_remote_artifact_review_threshold.py tests/test_post_id341_remote_next_action.py tests/test_post_id341_operator_handoff_contract.py

USAGE_SEMANTIC_BURST_MARKOV_SCRIPTS := scripts/diagnose_post_id341_usage_semantic_burst_markov_design.py scripts/diagnose_post_id341_usage_semantic_burst_markov_metadata_smoke.py scripts/diagnose_post_id341_usage_semantic_burst_markov_feature_spec.py scripts/diagnose_post_id341_usage_semantic_burst_markov_feature_generation_dryrun.py scripts/diagnose_post_id341_usage_semantic_burst_markov_feature_matrix_smoke_contract.py scripts/write_post_id341_usage_semantic_burst_markov_feature_matrix.py scripts/diagnose_post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop.py
USAGE_SEMANTIC_BURST_MARKOV_TESTS := tests/test_post_id341_usage_semantic_burst_markov_design.py tests/test_post_id341_usage_semantic_burst_markov_metadata_smoke.py tests/test_post_id341_usage_semantic_burst_markov_feature_spec.py tests/test_post_id341_usage_semantic_burst_markov_feature_generation_dryrun.py tests/test_post_id341_usage_semantic_burst_markov_feature_matrix_smoke_contract.py tests/test_post_id341_usage_semantic_burst_markov_feature_matrix_writer.py tests/test_post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop.py

PROXIMITY_AMBIENCE_COOCCURRENCE_SCRIPTS := scripts/diagnose_post_id341_proximity_ambience_cooccurrence_design.py scripts/diagnose_post_id341_proximity_ambience_cooccurrence_metadata_smoke.py scripts/diagnose_post_id341_proximity_ambience_cooccurrence_feature_spec.py scripts/diagnose_post_id341_proximity_ambience_cooccurrence_feature_generation_dryrun.py scripts/diagnose_post_id341_proximity_ambience_cooccurrence_feature_matrix_smoke_contract.py scripts/write_post_id341_proximity_ambience_cooccurrence_feature_matrix.py scripts/diagnose_post_id341_proximity_ambience_cooccurrence_feature_matrix_validation_stop.py
PROXIMITY_AMBIENCE_COOCCURRENCE_TESTS := tests/test_post_id341_proximity_ambience_cooccurrence.py

PHONE_WEARABLE_DISCORDANCE_SCRIPTS := scripts/diagnose_post_id341_phone_wearable_discordance_design.py scripts/diagnose_post_id341_phone_wearable_discordance_feature_spec.py scripts/diagnose_post_id341_phone_wearable_discordance_feature_generation_dryrun.py scripts/diagnose_post_id341_phone_wearable_discordance_feature_matrix_smoke_contract.py scripts/write_post_id341_phone_wearable_discordance_feature_matrix.py scripts/diagnose_post_id341_phone_wearable_discordance_feature_matrix_validation_stop.py
PHONE_WEARABLE_DISCORDANCE_TESTS := tests/test_post_id341_phone_wearable_discordance_design.py tests/test_post_id341_phone_wearable_discordance_feature_spec.py tests/test_post_id341_phone_wearable_discordance_feature_generation_dryrun.py tests/test_post_id341_phone_wearable_discordance_feature_matrix_smoke_contract.py tests/test_post_id341_phone_wearable_discordance_feature_matrix_writer.py tests/test_post_id341_phone_wearable_discordance_feature_matrix_validation_stop.py

SELF_SUPERVISED_SUBJECT_STATE_EMBEDDING_SCRIPTS := scripts/diagnose_post_id341_self_supervised_subject_state_embedding_design.py scripts/diagnose_post_id341_self_supervised_heavy_compute_contract.py
SELF_SUPERVISED_SUBJECT_STATE_EMBEDDING_TESTS := tests/test_post_id341_self_supervised_subject_state_embedding_design.py tests/test_post_id341_self_supervised_heavy_compute_contract.py

MATERIAL_SOURCE_REMOTE_COVERAGE_SCRIPTS := scripts/diagnose_post_id341_material_source_remote_coverage_contract.py
MATERIAL_SOURCE_REMOTE_COVERAGE_TESTS := tests/test_post_id341_material_source_remote_coverage_contract.py
PRELAUNCH_FRESHNESS_CONTRACT_SCRIPTS := scripts/diagnose_post_id341_prelaunch_freshness_contract.py
PRELAUNCH_FRESHNESS_CONTRACT_TESTS := tests/test_post_id341_prelaunch_freshness_contract.py
PUBLIC058_ARTIFACT_PROMOTION_CONTRACT_SCRIPTS := scripts/diagnose_post_id341_public058_artifact_promotion_contract.py
PUBLIC058_ARTIFACT_PROMOTION_CONTRACT_TESTS := tests/test_post_id341_public058_artifact_promotion_contract.py
SAFE_WAIT_STATE_CONTRACT_SCRIPTS := scripts/diagnose_post_id341_safe_wait_state_contract.py
SAFE_WAIT_STATE_CONTRACT_TESTS := tests/test_post_id341_safe_wait_state_contract.py
NEXT_MATERIAL_ROUTE_TRIAGE_SCRIPTS := scripts/diagnose_post_id341_next_material_route_triage.py
NEXT_MATERIAL_ROUTE_TRIAGE_TESTS := tests/test_post_id341_next_material_route_triage.py
SLEEP_PROXY_THRESHOLD_RESIDUAL_SCRIPTS := scripts/diagnose_post_id341_sleep_proxy_threshold_residual_design.py
SLEEP_PROXY_THRESHOLD_RESIDUAL_TESTS := tests/test_post_id341_sleep_proxy_threshold_residual_design.py
LOCAL_REMOTE_COMPUTE_BOUNDARY_SCRIPTS := scripts/diagnose_post_id341_local_remote_compute_boundary_contract.py
LOCAL_REMOTE_COMPUTE_BOUNDARY_TESTS := tests/test_post_id341_local_remote_compute_boundary_contract.py
LOCAL_SOURCE_CLOSURE_SYNTHESIS_SCRIPTS := scripts/diagnose_post_id341_local_source_closure_synthesis.py
LOCAL_SOURCE_CLOSURE_SYNTHESIS_TESTS := tests/test_post_id341_local_source_closure_synthesis.py
SUBMISSION_SPEND_POLICY_SCRIPTS := scripts/diagnose_post_id341_submission_spend_policy.py
SUBMISSION_SPEND_POLICY_TESTS := tests/test_post_id341_submission_spend_policy.py
DACON_SUBMIT_APPROVAL_CONTRACT_SCRIPTS := scripts/audit_dacon_submit_approval_contract.py
DACON_SUBMIT_APPROVAL_CONTRACT_TESTS := tests/test_dacon_submit_approval_contract.py tests/test_dacon_submit_make_guard.py

LINT_PATHS += $(USAGE_SEMANTIC_BURST_MARKOV_SCRIPTS) $(PROXIMITY_AMBIENCE_COOCCURRENCE_SCRIPTS) $(PHONE_WEARABLE_DISCORDANCE_SCRIPTS) $(SELF_SUPERVISED_SUBJECT_STATE_EMBEDDING_SCRIPTS) $(MATERIAL_SOURCE_REMOTE_COVERAGE_SCRIPTS) $(PRELAUNCH_FRESHNESS_CONTRACT_SCRIPTS) $(PUBLIC058_ARTIFACT_PROMOTION_CONTRACT_SCRIPTS) $(SAFE_WAIT_STATE_CONTRACT_SCRIPTS) $(NEXT_MATERIAL_ROUTE_TRIAGE_SCRIPTS) $(SLEEP_PROXY_THRESHOLD_RESIDUAL_SCRIPTS) $(LOCAL_REMOTE_COMPUTE_BOUNDARY_SCRIPTS)
KAGGLE_APPROVAL_CONTRACT_LINT_PATHS += $(USAGE_SEMANTIC_BURST_MARKOV_SCRIPTS) $(USAGE_SEMANTIC_BURST_MARKOV_TESTS) $(PROXIMITY_AMBIENCE_COOCCURRENCE_SCRIPTS) $(PROXIMITY_AMBIENCE_COOCCURRENCE_TESTS) $(PHONE_WEARABLE_DISCORDANCE_SCRIPTS) $(PHONE_WEARABLE_DISCORDANCE_TESTS) $(SELF_SUPERVISED_SUBJECT_STATE_EMBEDDING_SCRIPTS) $(SELF_SUPERVISED_SUBJECT_STATE_EMBEDDING_TESTS) $(MATERIAL_SOURCE_REMOTE_COVERAGE_SCRIPTS) $(MATERIAL_SOURCE_REMOTE_COVERAGE_TESTS) $(PRELAUNCH_FRESHNESS_CONTRACT_SCRIPTS) $(PRELAUNCH_FRESHNESS_CONTRACT_TESTS) $(PUBLIC058_ARTIFACT_PROMOTION_CONTRACT_SCRIPTS) $(PUBLIC058_ARTIFACT_PROMOTION_CONTRACT_TESTS) $(SAFE_WAIT_STATE_CONTRACT_SCRIPTS) $(SAFE_WAIT_STATE_CONTRACT_TESTS) $(NEXT_MATERIAL_ROUTE_TRIAGE_SCRIPTS) $(NEXT_MATERIAL_ROUTE_TRIAGE_TESTS) $(SLEEP_PROXY_THRESHOLD_RESIDUAL_SCRIPTS) $(SLEEP_PROXY_THRESHOLD_RESIDUAL_TESTS) $(LOCAL_REMOTE_COMPUTE_BOUNDARY_SCRIPTS) $(LOCAL_REMOTE_COMPUTE_BOUNDARY_TESTS)
KAGGLE_APPROVAL_CONTRACT_TESTS += $(USAGE_SEMANTIC_BURST_MARKOV_TESTS) $(PROXIMITY_AMBIENCE_COOCCURRENCE_TESTS) $(PHONE_WEARABLE_DISCORDANCE_TESTS) $(SELF_SUPERVISED_SUBJECT_STATE_EMBEDDING_TESTS) $(MATERIAL_SOURCE_REMOTE_COVERAGE_TESTS) $(PRELAUNCH_FRESHNESS_CONTRACT_TESTS) $(PUBLIC058_ARTIFACT_PROMOTION_CONTRACT_TESTS) $(SAFE_WAIT_STATE_CONTRACT_TESTS) $(NEXT_MATERIAL_ROUTE_TRIAGE_TESTS) $(SLEEP_PROXY_THRESHOLD_RESIDUAL_TESTS) $(LOCAL_REMOTE_COMPUTE_BOUNDARY_TESTS)
LINT_PATHS += $(LOCAL_SOURCE_CLOSURE_SYNTHESIS_SCRIPTS)
KAGGLE_APPROVAL_CONTRACT_LINT_PATHS += $(LOCAL_SOURCE_CLOSURE_SYNTHESIS_SCRIPTS) $(LOCAL_SOURCE_CLOSURE_SYNTHESIS_TESTS)
KAGGLE_APPROVAL_CONTRACT_TESTS += $(LOCAL_SOURCE_CLOSURE_SYNTHESIS_TESTS)
LINT_PATHS += $(SUBMISSION_SPEND_POLICY_SCRIPTS)
KAGGLE_APPROVAL_CONTRACT_LINT_PATHS += $(SUBMISSION_SPEND_POLICY_SCRIPTS) $(SUBMISSION_SPEND_POLICY_TESTS)
KAGGLE_APPROVAL_CONTRACT_TESTS += $(SUBMISSION_SPEND_POLICY_TESTS)
LINT_PATHS += $(DACON_SUBMIT_APPROVAL_CONTRACT_SCRIPTS)
LINT_PATHS += scripts/diagnose_post_id341_public060_wall_root_cause.py
LINT_PATHS += scripts/diagnose_post_id342_research_gap_and_source_priority.py
LINT_PATHS += scripts/diagnose_post_id342_raw_temporal_q3s3_sequence_image_smoke.py
LINT_PATHS += scripts/diagnose_post_id342_sequence_summary_distance_kernel_smoke.py
LINT_PATHS += scripts/diagnose_post_id342_sleep_boundary_q3s3_rescue_smoke.py
LINT_PATHS += scripts/diagnose_post_id342_tiny_temporal_cnn_q3s3_smoke.py
LINT_PATHS += scripts/diagnose_post_id342_public_response_geometry.py
LINT_PATHS += scripts/diagnose_post_id342_public_alignment_validation_lens.py
LINT_PATHS += scripts/diagnose_post_id342_stable_lens_existing_candidate_triage.py
LINT_PATHS += scripts/diagnose_post_id342_env_social_routine_v2_id342_gate.py
LINT_PATHS += scripts/diagnose_post_id342_evening_state_transition_gate.py
LINT_PATHS += scripts/diagnose_post_id342_public_scored_target_swap_gate.py
LINT_PATHS += scripts/diagnose_post_id342_q3s3_slice_selector_anatomy.py
LINT_PATHS += scripts/diagnose_post_id342_lightweight_subject_state_pca_gate.py
LINT_PATHS += scripts/diagnose_post_id342_processed_sleep_proxy_gate.py
LINT_PATHS += scripts/diagnose_post_id342_processed_sleep_temporal_deviation_gate.py
LINT_PATHS += scripts/diagnose_post_id342_subject_prior_calibration_gate.py
LINT_PATHS += scripts/diagnose_post_id342_probability_temperature_gate.py
LINT_PATHS += scripts/diagnose_post_id342_remaining_route_synthesis.py
LINT_PATHS += scripts/diagnose_post_id342_self_supervised_heavy_compute_contract.py
LINT_PATHS += scripts/diagnose_post_id342_self_supervised_route_research_packet.py
LINT_PATHS += scripts/diagnose_post_id342_self_supervised_heavy_runner_spec.py
LINT_PATHS += scripts/diagnose_post_id342_subject_state_domain_token_contract.py
LINT_PATHS += scripts/diagnose_post_id342_subject_state_raw_schema_contract.py
LINT_PATHS += scripts/diagnose_post_id342_subject_state_heavy_approval_readiness.py
LINT_PATHS += scripts/diagnose_post_id342_subject_state_heavy_success_thresholds.py
LINT_PATHS += scripts/diagnose_post_id342_kaggle_input_availability.py
LINT_PATHS += scripts/diagnose_post_id342_kaggle_cli_health.py
LINT_PATHS += scripts/diagnose_post_id342_morning_closeout.py
LINT_PATHS += scripts/diagnose_post_id342_external_research_priority.py
LINT_PATHS += scripts/diagnose_post_id342_heavy_implementation_feasibility.py
LINT_PATHS += scripts/diagnose_post_id342_heavy_runtime_budget.py
LINT_PATHS += scripts/diagnose_post_id342_goal_state_audit.py
LINT_PATHS += scripts/diagnose_post_id342_feature_similarity_source_decomposition.py
LINT_PATHS += scripts/diagnose_post_id342_feature_similarity_artifact_gate.py
LINT_PATHS += scripts/diagnose_post_id342_paper_aligned_feature_similarity_rebuild_gate.py
LINT_PATHS += scripts/diagnose_post_id342_past_only_temporal_recency_gate.py
LINT_PATHS += scripts/diagnose_post_id342_cross_target_consistency_gate.py
LINT_PATHS += scripts/diagnose_post_id342_registered_material_source_rescan.py
LINT_PATHS += scripts/diagnose_post_id342_registered_material_fusion_gate.py
LINT_PATHS += scripts/diagnose_post_id342_public_gap_penalty_proxy_redesign.py
LINT_PATHS += scripts/diagnose_post_id342_stable_envelope_microblend_scan.py
LINT_PATHS += scripts/diagnose_post_id342_stable_envelope_multisource_optimizer.py
LINT_PATHS += scripts/diagnose_post_id342_target_reallocation_optimizer.py
LINT_PATHS += scripts/diagnose_post_id342_s4s3_sleep_fragmentation_gate.py
LINT_PATHS += scripts/diagnose_post_id342_raw_sleep_event_reconstruction_feasibility.py
LINT_PATHS += scripts/diagnose_post_id342_sequence_hr_light_event_gate.py
LINT_PATHS += scripts/diagnose_post_id342_mis_lstm_block_sequence_feasibility_gate.py
LINT_PATHS += scripts/diagnose_post_id342_cross_source_split_stability_audit.py
LINT_PATHS += scripts/diagnose_post_id342_s3_phone_gps_interaction_gate.py
LINT_PATHS += scripts/diagnose_post_id342_final_day_independent_signal_sweep.py
LINT_PATHS += scripts/fetch_kma_asos_hourly_context.py
LINT_PATHS += scripts/smoke_public_data_weather_air_api.py
LINT_PATHS += scripts/audit_airkorea_2024_hourly_zip.py
LINT_PATHS += scripts/diagnose_post_id342_airkorea_2024_context_gate.py
LINT_PATHS += scripts/diagnose_post_id342_external_etri_sleep_survey_prior.py
LINT_PATHS += scripts/diagnose_post_id342_deadline_operating_policy.py
LINT_PATHS += scripts/audit_paper_submission_package.py
LINT_PATHS += scripts/package_paper_submission_sources.py
LINT_PATHS += scripts/extract_paper_submission_metadata.py
LINT_PATHS += scripts/audit_paper_length_budget.py
LINT_PATHS += scripts/run_post_id342_self_supervised_subject_state_no_submit.py
LINT_PATHS += scripts/diagnose_post_id342_self_supervised_subject_state_remote_intake.py
LINT_PATHS += scripts/diagnose_post_id342_raw_subject_state_feature_gate.py
LINT_PATHS += scripts/diagnose_post_id342_s3q3_raw_sequence_representation_gate.py

lint:
	.venv/bin/ruff check $(LINT_PATHS)

format:
	.venv/bin/ruff check --fix $(LINT_PATHS)
	.venv/bin/ruff format $(LINT_PATHS)

test:
	PYTHONPATH=. .venv/bin/pytest tests/ -v

audit:
	.venv/bin/python scripts/diagnose_post_id341_public_local_gap.py --output reports/post_id341_public_local_gap_diagnosis_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_data_source_coverage.py --output reports/post_id341_data_source_coverage_audit_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_manual_submission_probe_decision.py --output reports/post_id341_manual_submission_probe_decision_20260618.json
	.venv/bin/python scripts/audit_competition_workspace.py
	.venv/bin/python scripts/audit_docs_records.py
	.venv/bin/python scripts/audit_submissions.py
	.venv/bin/python scripts/audit_experiment_artifacts.py
	.venv/bin/python scripts/audit_repository_structure.py

docs-audit:
	.venv/bin/python scripts/audit_docs_records.py

submissions-audit:
	.venv/bin/python scripts/audit_submissions.py

artifact-contract-audit:
	.venv/bin/python scripts/audit_experiment_artifacts.py

public-local-gap-diagnosis:
	.venv/bin/python scripts/diagnose_post_id341_public_local_gap.py --output reports/post_id341_public_local_gap_diagnosis_20260618.json

data-source-coverage-audit:
	.venv/bin/python scripts/diagnose_post_id341_data_source_coverage.py --output reports/post_id341_data_source_coverage_audit_20260618.json

q2q3s3-source-hypothesis-matrix:
	.venv/bin/python scripts/diagnose_post_id341_q2q3s3_source_hypothesis_matrix.py --output reports/post_id341_q2q3s3_source_hypothesis_matrix_20260618.json

manual-submission-probe-decision:
	.venv/bin/python scripts/diagnose_post_id341_manual_submission_probe_decision.py --output reports/post_id341_manual_submission_probe_decision_20260618.json

post-id341-submission-spend-policy:
	.venv/bin/python scripts/diagnose_post_id341_submission_spend_policy.py --output reports/post_id341_submission_spend_policy_20260619.json

post-id341-safe-wait-state:
	.venv/bin/python scripts/diagnose_post_id341_safe_wait_state_contract.py --output reports/post_id341_safe_wait_state_contract_20260619.json

post-id341-next-route-triage:
	.venv/bin/python scripts/diagnose_post_id341_next_material_route_triage.py --output reports/post_id341_next_material_route_triage_20260619.json

post-id341-sleep-proxy-design:
	.venv/bin/python scripts/diagnose_post_id341_sleep_proxy_threshold_residual_design.py --output reports/post_id341_sleep_proxy_threshold_residual_design_20260619.json

post-id341-local-compute-boundary:
	.venv/bin/python scripts/diagnose_post_id341_local_remote_compute_boundary_contract.py --output reports/post_id341_local_remote_compute_boundary_contract_20260619.json

post-id342-research-gap:
	.venv/bin/python scripts/diagnose_post_id342_research_gap_and_source_priority.py --output reports/post_id342_research_gap_and_source_priority_20260624.json

post-id342-raw-temporal-q3s3-smoke: post-id342-research-gap
	.venv/bin/python scripts/diagnose_post_id342_raw_temporal_q3s3_sequence_image_smoke.py --output reports/post_id342_raw_temporal_q3s3_sequence_image_smoke_20260624.json

post-id342-sequence-summary-distance-smoke: post-id342-raw-temporal-q3s3-smoke
	.venv/bin/python scripts/diagnose_post_id342_sequence_summary_distance_kernel_smoke.py --output reports/post_id342_sequence_summary_distance_kernel_smoke_20260624.json

post-id342-sleep-boundary-q3s3-smoke: post-id342-sequence-summary-distance-smoke
	.venv/bin/python scripts/diagnose_post_id342_sleep_boundary_q3s3_rescue_smoke.py --output reports/post_id342_sleep_boundary_q3s3_rescue_smoke_20260624.json

post-id342-tiny-temporal-cnn-q3s3-smoke: post-id342-sleep-boundary-q3s3-smoke
	.venv/bin/python scripts/diagnose_post_id342_tiny_temporal_cnn_q3s3_smoke.py --output reports/post_id342_tiny_temporal_cnn_q3s3_smoke_20260624.json

post-id342-public-response-geometry:
	.venv/bin/python scripts/diagnose_post_id342_public_response_geometry.py --output reports/post_id342_public_response_geometry_20260624.json

post-id342-public-alignment-lens:
	.venv/bin/python scripts/diagnose_post_id342_public_alignment_validation_lens.py --output reports/post_id342_public_alignment_validation_lens_20260624.json

post-id342-public055-gap-attribution:
	.venv/bin/python scripts/diagnose_post_id342_public055_gap_attribution.py --output reports/post_id342_public055_gap_attribution_20260624.json

post-id342-public-gap-risk-lens:
	.venv/bin/python scripts/diagnose_post_id342_public_gap_risk_lens.py --output reports/post_id342_public_gap_risk_lens_20260624.json

post-id342-public-gap-penalty-proxy-redesign: post-id342-public-gap-risk-lens
	.venv/bin/python scripts/diagnose_post_id342_public_gap_penalty_proxy_redesign.py --output reports/post_id342_public_gap_penalty_proxy_redesign_20260625.json

post-id342-validation-weighted-pseudo-public-split-audit: post-id342-public-gap-penalty-proxy-redesign post-id342-public-alignment-lens
	.venv/bin/python scripts/diagnose_post_id342_validation_weighted_pseudo_public_split_audit.py --output reports/post_id342_validation_weighted_pseudo_public_split_audit_20260625.json

post-id342-solar-seasonal-holiday-prior-gate:
	.venv/bin/python scripts/diagnose_post_id342_solar_seasonal_holiday_prior_gate.py --output reports/post_id342_solar_seasonal_holiday_prior_gate_20260625.json

fetch-kma-asos-hourly-context:
	.venv/bin/python scripts/fetch_kma_asos_hourly_context.py --output data/external/weather/kma_asos_hourly_20240601_20241122.csv

post-id342-kma-weather-context-gate:
	.venv/bin/python scripts/diagnose_post_id342_kma_weather_context_gate.py --weather-cache data/external/weather/kma_asos_hourly_20240601_20241122.csv --output reports/post_id342_kma_weather_context_gate_20260625.json

smoke-public-data-weather-air-api:
	.venv/bin/python scripts/smoke_public_data_weather_air_api.py --output reports/public_data_weather_air_api_smoke_20260625.json

audit-airkorea-2024-hourly-zip:
	.venv/bin/python scripts/audit_airkorea_2024_hourly_zip.py --zip-path data/external/weather/airkorea_2024/airkorea_2024_hourly_all_stations_xlsx.zip --daejeon-output data/external/weather/airkorea_2024/airkorea_2024_daejeon_hourly.csv --output reports/airkorea_2024_hourly_zip_audit_20260625.json

post-id342-airkorea-2024-context-gate:
	.venv/bin/python scripts/diagnose_post_id342_airkorea_2024_context_gate.py --airkorea-cache data/external/weather/airkorea_2024/airkorea_2024_daejeon_hourly.csv --output reports/post_id342_airkorea_2024_context_gate_20260625.json

post-id342-external-etri-sleep-survey-prior:
	.venv/bin/python scripts/diagnose_post_id342_external_etri_sleep_survey_prior.py --output reports/post_id342_external_etri_sleep_survey_prior_20260625.json

post-id342-deadline-operating-policy:
	.venv/bin/python scripts/diagnose_post_id342_deadline_operating_policy.py --output reports/post_id342_deadline_operating_policy_20260625.json

paper-submission-metadata:
	$(PAPER_PYTHON) scripts/extract_paper_submission_metadata.py --tex docs/paper_submission_20260626/paper.tex --output docs/paper_submission_20260626/edas_metadata.json

paper-submission-pdf:
	mkdir -p build/paper_submission_20260626/compile
	cd docs/paper_submission_20260626 && $(TECTONIC) paper.tex --outdir ../../build/paper_submission_20260626/compile --keep-logs --keep-intermediates

paper-submission-length-audit: paper-submission-pdf
	$(PAPER_PYTHON) scripts/audit_paper_length_budget.py --tex docs/paper_submission_20260626/paper.tex --output docs/paper_submission_20260626/paper_length_budget.json

paper-submission-source-archive: paper-submission-metadata paper-submission-length-audit
	$(PAPER_PYTHON) scripts/package_paper_submission_sources.py --archive build/paper_submission_20260626/paper_submission_20260626_source.zip --output reports/paper_submission_source_archive_20260625.json

paper-submission-package-audit: paper-submission-source-archive
	$(PAPER_PYTHON) scripts/audit_paper_submission_package.py --output reports/paper_submission_package_audit_20260625.json

post-id342-stable-envelope-multisource-optimizer: post-id342-public-gap-risk-lens
	.venv/bin/python scripts/diagnose_post_id342_stable_envelope_microblend_scan.py --output reports/post_id342_stable_envelope_microblend_scan_20260625.json
	.venv/bin/python scripts/diagnose_post_id342_stable_envelope_multisource_optimizer.py --output reports/post_id342_stable_envelope_multisource_optimizer_20260625.json

post-id342-target-reallocation-optimizer: post-id342-public-gap-risk-lens
	.venv/bin/python scripts/diagnose_post_id342_target_reallocation_optimizer.py --output reports/post_id342_target_reallocation_optimizer_20260625.json

post-id342-s4s3-sleep-fragmentation-gate:
	.venv/bin/python scripts/diagnose_post_id342_s4s3_sleep_fragmentation_gate.py --output reports/post_id342_s4s3_sleep_fragmentation_gate_20260625.json

post-id342-raw-sleep-event-feasibility: post-id342-s4s3-sleep-fragmentation-gate
	.venv/bin/python scripts/diagnose_post_id342_raw_sleep_event_reconstruction_feasibility.py --output reports/post_id342_raw_sleep_event_reconstruction_feasibility_20260625.json

post-id342-sequence-hr-light-event-gate: post-id342-raw-sleep-event-feasibility
	.venv/bin/python scripts/diagnose_post_id342_sequence_hr_light_event_gate.py --output reports/post_id342_sequence_hr_light_event_gate_20260625.json

post-id342-mis-lstm-block-sequence-feasibility-gate: post-id342-external-research-priority
	.venv/bin/python scripts/diagnose_post_id342_mis_lstm_block_sequence_feasibility_gate.py --output reports/post_id342_mis_lstm_block_sequence_feasibility_gate_20260625.json

post-id342-cross-source-split-stability-audit:
	.venv/bin/python scripts/diagnose_post_id342_cross_source_split_stability_audit.py --output reports/post_id342_cross_source_split_stability_audit_20260625.json

post-id342-s3-phone-gps-interaction-gate: post-id342-cross-source-split-stability-audit
	.venv/bin/python scripts/diagnose_post_id342_s3_phone_gps_interaction_gate.py --output reports/post_id342_s3_phone_gps_interaction_gate_20260625.json

post-id342-final-day-independent-signal-sweep: post-id342-s3-phone-gps-interaction-gate
	.venv/bin/python scripts/diagnose_post_id342_final_day_independent_signal_sweep.py --output reports/post_id342_final_day_independent_signal_sweep_20260625.json

post-id342-stable-lens-triage: post-id342-public-alignment-lens
	.venv/bin/python scripts/diagnose_post_id342_stable_lens_existing_candidate_triage.py --output reports/post_id342_stable_lens_existing_candidate_triage_20260624.json

post-id342-env-social-routine-gate: post-id342-stable-lens-triage
	.venv/bin/python scripts/diagnose_post_id342_env_social_routine_v2_id342_gate.py --output reports/post_id342_env_social_routine_v2_id342_gate_20260624.json

post-id342-evening-state-transition-gate: post-id342-env-social-routine-gate
	.venv/bin/python scripts/diagnose_post_id342_evening_state_transition_gate.py --output reports/post_id342_evening_state_transition_gate_20260624.json

post-id342-public-scored-target-swap-gate: post-id342-evening-state-transition-gate
	.venv/bin/python scripts/diagnose_post_id342_public_scored_target_swap_gate.py --output reports/post_id342_public_scored_target_swap_gate_20260624.json

post-id342-q3s3-slice-selector-anatomy: post-id342-public-scored-target-swap-gate
	.venv/bin/python scripts/diagnose_post_id342_q3s3_slice_selector_anatomy.py --output reports/post_id342_q3s3_slice_selector_anatomy_20260624.json

post-id342-lightweight-subject-state-pca-gate: post-id342-q3s3-slice-selector-anatomy
	.venv/bin/python scripts/diagnose_post_id342_lightweight_subject_state_pca_gate.py --output reports/post_id342_lightweight_subject_state_pca_gate_20260624.json

post-id342-processed-sleep-proxy-gate: post-id342-lightweight-subject-state-pca-gate
	.venv/bin/python scripts/diagnose_post_id342_processed_sleep_proxy_gate.py --output reports/post_id342_processed_sleep_proxy_gate_20260624.json

post-id342-processed-sleep-temporal-deviation-gate: post-id342-processed-sleep-proxy-gate
	.venv/bin/python scripts/diagnose_post_id342_processed_sleep_temporal_deviation_gate.py --output reports/post_id342_processed_sleep_temporal_deviation_gate_20260624.json

post-id342-subject-prior-calibration-gate: post-id342-processed-sleep-temporal-deviation-gate
	.venv/bin/python scripts/diagnose_post_id342_subject_prior_calibration_gate.py --output reports/post_id342_subject_prior_calibration_gate_20260624.json

post-id342-probability-temperature-gate: post-id342-subject-prior-calibration-gate
	.venv/bin/python scripts/diagnose_post_id342_probability_temperature_gate.py --output reports/post_id342_probability_temperature_gate_20260624.json

post-id342-remaining-route-synthesis: post-id342-probability-temperature-gate post-id342-paper-aligned-featuresim-gate post-id342-past-only-temporal-gate post-id342-cross-target-consistency-gate post-id342-material-fusion-gate post-id342-public-gap-penalty-proxy-redesign post-id342-validation-weighted-pseudo-public-split-audit post-id342-solar-seasonal-holiday-prior-gate post-id342-kma-weather-context-gate post-id342-airkorea-2024-context-gate post-id342-external-etri-sleep-survey-prior post-id342-stable-envelope-multisource-optimizer post-id342-target-reallocation-optimizer post-id342-sequence-hr-light-event-gate post-id342-cross-source-split-stability-audit post-id342-s3-phone-gps-interaction-gate post-id342-final-day-independent-signal-sweep
	.venv/bin/python scripts/diagnose_post_id342_remaining_route_synthesis.py --output reports/post_id342_remaining_route_synthesis_20260625.json

post-id342-self-supervised-heavy-contract: post-id342-remaining-route-synthesis post-id342-goal-state-audit
	.venv/bin/python scripts/diagnose_post_id342_self_supervised_heavy_compute_contract.py --output reports/post_id342_self_supervised_heavy_compute_contract_20260624.json

post-id342-self-supervised-route-research: post-id342-self-supervised-heavy-contract
	.venv/bin/python scripts/diagnose_post_id342_self_supervised_route_research_packet.py --output reports/post_id342_self_supervised_route_research_packet_20260624.json

post-id342-self-supervised-heavy-runner-spec: post-id342-self-supervised-heavy-contract post-id342-self-supervised-route-research docs-audit
	.venv/bin/python scripts/diagnose_post_id342_self_supervised_heavy_runner_spec.py --output reports/post_id342_self_supervised_heavy_runner_spec_20260624.json

post-id342-subject-state-domain-token-contract: post-id342-self-supervised-heavy-runner-spec
	.venv/bin/python scripts/diagnose_post_id342_subject_state_domain_token_contract.py --output reports/post_id342_subject_state_domain_token_contract_20260624.json

post-id342-subject-state-raw-schema-contract: post-id342-subject-state-domain-token-contract
	.venv/bin/python scripts/diagnose_post_id342_subject_state_raw_schema_contract.py --output reports/post_id342_subject_state_raw_schema_contract_20260624.json

post-id342-subject-state-heavy-approval-readiness: post-id342-subject-state-raw-schema-contract docs-audit
	.venv/bin/python scripts/diagnose_post_id342_subject_state_heavy_approval_readiness.py --manual-kaggle-status $(KAGGLE_STATUS_OBSERVATION) --output reports/post_id342_subject_state_heavy_approval_readiness_20260624.json

post-id342-subject-state-heavy-success-thresholds: post-id342-subject-state-heavy-approval-readiness
	.venv/bin/python scripts/diagnose_post_id342_subject_state_heavy_success_thresholds.py --output reports/post_id342_subject_state_heavy_success_thresholds_20260624.json

post-id342-kaggle-input-availability: require-kaggle-remote-approval post-id342-subject-state-heavy-success-thresholds
	ALLOW_KAGGLE_REMOTE=true KAGGLE=$(KAGGLE) .venv/bin/python scripts/diagnose_post_id342_kaggle_input_availability.py --allow-remote --output reports/post_id342_kaggle_input_availability_20260624.json

post-id342-kaggle-cli-health: require-kaggle-remote-approval post-id342-kaggle-input-availability
	ALLOW_KAGGLE_REMOTE=true KAGGLE=$(KAGGLE) .venv/bin/python scripts/diagnose_post_id342_kaggle_cli_health.py --allow-remote --output reports/post_id342_kaggle_cli_health_20260624.json

post-id342-morning-closeout: post-id342-kaggle-cli-health
	.venv/bin/python scripts/diagnose_post_id342_morning_closeout.py --output reports/post_id342_morning_closeout_20260624.json

post-id342-external-research-priority: post-id342-subject-state-heavy-success-thresholds
	.venv/bin/python scripts/diagnose_post_id342_external_research_priority.py --output reports/post_id342_external_research_priority_20260625.json

post-id342-heavy-implementation-feasibility: post-id342-external-research-priority
	.venv/bin/python scripts/diagnose_post_id342_heavy_implementation_feasibility.py --output reports/post_id342_heavy_implementation_feasibility_20260624.json

post-id342-heavy-runtime-budget: post-id342-heavy-implementation-feasibility
	.venv/bin/python scripts/diagnose_post_id342_heavy_runtime_budget.py --output reports/post_id342_heavy_runtime_budget_20260624.json

post-id342-materially-new-source-hypothesis:
	.venv/bin/python scripts/diagnose_post_id342_materially_new_source_hypothesis.py --output reports/post_id342_materially_new_source_hypothesis_20260625.json

post-id342-self-supervised-subject-state-local-smoke: post-id342-heavy-runtime-budget
	.venv/bin/python scripts/run_post_id342_self_supervised_subject_state_no_submit.py --mode local_smoke --max-rows-per-modality 2048 --batch-size 1024 --token-limit 2048 --epochs 1 --output reports/post_id342_self_supervised_subject_state_local_smoke_20260624.json

post-id342-self-supervised-subject-state-remote-intake:
	.venv/bin/python scripts/diagnose_post_id342_self_supervised_subject_state_remote_intake.py --pull-dir build/kaggle_outputs/self_supervised_subject_state_post_id342 --output reports/post_id342_self_supervised_subject_state_remote_intake_20260624.json

post-id342-missingness-subject-state-local-smoke: post-id342-heavy-runtime-budget
	.venv/bin/python scripts/run_post_id342_missingness_subject_state_heavy_no_submit.py --mode local_smoke --max-rows-per-modality 2048 --batch-size 1024 --embedding-dim 8 --epochs 1 --output-dir build/local_missingness_subject_state_heavy_smoke

post-id342-missingness-subject-state-remote-intake:
	.venv/bin/python scripts/diagnose_post_id342_missingness_subject_state_remote_intake.py --pull-dir build/kaggle_outputs/missingness_subject_state_post_id342 --output reports/post_id342_missingness_subject_state_remote_intake_20260624.json

post-id342-raw-subject-state-feature-gate:
	.venv/bin/python scripts/diagnose_post_id342_raw_subject_state_feature_gate.py --max-rows-per-modality 32768 --batch-size 16384 --output reports/post_id342_raw_subject_state_feature_gate_20260624.json

post-id342-s3q3-raw-sequence-representation-gate:
	.venv/bin/python scripts/diagnose_post_id342_s3q3_raw_sequence_representation_gate.py --n-kernels 64 --seeds 45502 --families q3_usage_routine,s3_sleep_boundary --output reports/post_id342_s3q3_raw_sequence_representation_gate_local_smoke_20260624.json

post-id342-goal-state-audit: post-id342-remaining-route-synthesis docs-audit
	.venv/bin/python scripts/diagnose_post_id342_goal_state_audit.py --output reports/post_id342_goal_state_audit_20260624.json

post-id342-feature-similarity-decomp: post-id342-public-response-geometry
	.venv/bin/python scripts/diagnose_post_id342_feature_similarity_source_decomposition.py --output reports/post_id342_feature_similarity_source_decomposition_20260624.json

post-id342-feature-similarity-artifact-gate: post-id342-feature-similarity-decomp
	.venv/bin/python scripts/diagnose_post_id342_feature_similarity_artifact_gate.py --output reports/post_id342_feature_similarity_artifact_gate_20260624.json

post-id342-paper-aligned-featuresim-gate: post-id342-feature-similarity-artifact-gate
	.venv/bin/python scripts/diagnose_post_id342_paper_aligned_feature_similarity_rebuild_gate.py --output reports/post_id342_paper_aligned_feature_similarity_rebuild_gate_20260624.json

post-id342-past-only-temporal-gate:
	.venv/bin/python scripts/diagnose_post_id342_past_only_temporal_recency_gate.py --output reports/post_id342_past_only_temporal_recency_gate_20260624.json

post-id342-cross-target-consistency-gate:
	.venv/bin/python scripts/diagnose_post_id342_cross_target_consistency_gate.py --output reports/post_id342_cross_target_consistency_gate_20260624.json

post-id342-material-source-rescan:
	.venv/bin/python scripts/diagnose_post_id342_registered_material_source_rescan.py --output reports/post_id342_registered_material_source_rescan_20260624.json

post-id342-material-fusion-gate: post-id342-material-source-rescan
	.venv/bin/python scripts/diagnose_post_id342_registered_material_fusion_gate.py --output reports/post_id342_registered_material_fusion_gate_20260624.json

structure-audit:
	.venv/bin/python scripts/audit_repository_structure.py

normalize-submissions:
	.venv/bin/python scripts/normalize_submission_log.py --update-canonical

cleanup-manifest:
	.venv/bin/python scripts/workspace_cleanup_manifest.py

jupyter:
	.venv/bin/jupyter lab

run-local:
	.venv/bin/python scripts/run_full_sota_pipeline.py

build-kaggle-dataset:
	@mkdir -p $(KAGGLE_DATASET_DIR)
	@rm -rf $(KAGGLE_DATASET_DIR)/*
	@zip -r $(KAGGLE_DATASET_DIR)/source.zip src conf scripts pyproject.toml README.md Makefile dataset-metadata.json reports data/raw data/processed data/external artifacts/base_predictions submissions -x "*/__pycache__/*" "*.pyc" ".DS_Store" "*/.DS_Store" ".pytest_cache/*" ".ruff_cache/*"
	@if [ -f dataset-metadata.json ]; then cp dataset-metadata.json $(KAGGLE_DATASET_DIR)/; fi
	@echo "Packaged project files into $(KAGGLE_DATASET_DIR)/"

build-kaggle-upstream-patch-dataset:
	@mkdir -p $(KAGGLE_UPSTREAM_PATCH_DATASET_DIR)
	@rm -rf $(KAGGLE_UPSTREAM_PATCH_DATASET_DIR)/*
	@mkdir -p $(KAGGLE_UPSTREAM_PATCH_DATASET_DIR)/data/processed $(KAGGLE_UPSTREAM_PATCH_DATASET_DIR)/data/external $(KAGGLE_UPSTREAM_PATCH_DATASET_DIR)/artifacts $(KAGGLE_UPSTREAM_PATCH_DATASET_DIR)/submissions
	@cp -R src conf scripts reports $(KAGGLE_UPSTREAM_PATCH_DATASET_DIR)/
	@cp pyproject.toml README.md Makefile $(KAGGLE_UPSTREAM_PATCH_DATASET_DIR)/
	@cp data/processed/post_id341_usage_semantic_burst_markov_features_v1.parquet $(KAGGLE_UPSTREAM_PATCH_DATASET_DIR)/data/processed/
	@cp -R data/external/etri_lifelog $(KAGGLE_UPSTREAM_PATCH_DATASET_DIR)/data/external/
	@cp -R artifacts/base_predictions $(KAGGLE_UPSTREAM_PATCH_DATASET_DIR)/artifacts/
	@cp -R submissions $(KAGGLE_UPSTREAM_PATCH_DATASET_DIR)/
	@printf '{\n  "title": "Dacon ETRI Upstream Patch Post ID341",\n  "id": "$(KAGGLE_UPSTREAM_PATCH_DATASET_SLUG)",\n  "licenses": [{"name": "CC0-1.0"}]\n}\n' > $(KAGGLE_UPSTREAM_PATCH_DATASET_DIR)/dataset-metadata.json
	@find $(KAGGLE_UPSTREAM_PATCH_DATASET_DIR) \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -type d -prune -exec rm -rf {} +
	@find $(KAGGLE_UPSTREAM_PATCH_DATASET_DIR) \( -name '*.pyc' -o -name .DS_Store \) -type f -delete
	@echo "Packaged upstream patch files into $(KAGGLE_UPSTREAM_PATCH_DATASET_DIR)/"

build-kaggle-self-supervised-subject-state-patch-dataset:
	@mkdir -p $(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_DIR)
	@rm -rf $(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_DIR)/*
	@mkdir -p $(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_DIR)/scripts $(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_DIR)/data/raw $(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_DIR)/reports $(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_DIR)/artifacts/base_predictions
	@cp scripts/run_post_id342_self_supervised_subject_state_no_submit.py $(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_DIR)/scripts/
	@cp scripts/run_post_id342_missingness_subject_state_heavy_no_submit.py $(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_DIR)/scripts/
	@cp scripts/diagnose_post_id341_q2q3s3_fold_internal_model_smoke.py $(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_DIR)/scripts/
	@cp scripts/diagnose_post_id341_q2q3s3_feature_generation_metadata_smoke.py $(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_DIR)/scripts/
	@cp scripts/diagnose_post_id342_raw_subject_state_feature_gate.py $(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_DIR)/scripts/
	@cp scripts/diagnose_post_id342_s3q3_raw_sequence_representation_gate.py $(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_DIR)/scripts/
	@cp data/raw/ch2026_metrics_train.csv $(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_DIR)/data/raw/
	@cp data/raw/ch2026_submission_sample.csv $(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_DIR)/data/raw/
	@cp reports/id340_validation_split_manifest_20260618.json $(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_DIR)/reports/
	@cp artifacts/base_predictions/base_oof_id_342_*.csv $(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_DIR)/artifacts/base_predictions/
	@printf '{\n  "title": "Dacon ETRI SS Subject State Patch ID342",\n  "id": "$(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_SLUG)",\n  "licenses": [{"name": "CC0-1.0"}]\n}\n' > $(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_DIR)/dataset-metadata.json
	@echo "Packaged self-supervised subject-state patch files into $(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_DIR)/"

kaggle-init-code: build-kaggle-dataset
	$(KAGGLE) datasets init -p $(KAGGLE_DATASET_DIR)
	cp $(KAGGLE_DATASET_DIR)/dataset-metadata.json ./
	@echo "Created dataset-metadata.json. Please edit it in the root directory and then run 'make kaggle-push-code'."

require-kaggle-remote-approval:
	@if [ "$(ALLOW_KAGGLE_REMOTE)" != "true" ]; then \
		echo "Error: Kaggle remote action blocked. Re-run with ALLOW_KAGGLE_REMOTE=true after explicit approval."; \
		exit 2; \
	fi

kaggle-push-code: require-kaggle-remote-approval build-kaggle-dataset
	@if [ ! -f dataset-metadata.json ]; then echo "Error: dataset-metadata.json not found in root. Run make kaggle-init-code first."; exit 1; fi
	@.venv/bin/python scripts/pre_flight_check.py
	$(KAGGLE) datasets version -p $(KAGGLE_DATASET_DIR) -m "Update source code" -r zip

kaggle-push-upstream-patch-dataset: require-kaggle-remote-approval build-kaggle-upstream-patch-dataset
	@if $(KAGGLE) datasets files $(KAGGLE_UPSTREAM_PATCH_DATASET_SLUG) >/dev/null 2>&1; then \
		$(KAGGLE) datasets version -p $(KAGGLE_UPSTREAM_PATCH_DATASET_DIR) -m "Refresh post-ID341 upstream patch payload" -r zip; \
	else \
		$(KAGGLE) datasets create -p $(KAGGLE_UPSTREAM_PATCH_DATASET_DIR) -r zip; \
	fi

kaggle-push-self-supervised-subject-state-patch-dataset: require-kaggle-remote-approval build-kaggle-self-supervised-subject-state-patch-dataset
	@if $(KAGGLE) datasets files $(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_SLUG) >/dev/null 2>&1; then \
		$(KAGGLE) datasets version -p $(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_DIR) -m "Refresh post-ID342 no-submit subject-state runner patch" -r zip; \
	else \
		$(KAGGLE) datasets create -p $(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_DIR) -r zip; \
	fi

kaggle-init-notebook:
	@mkdir -p $(KAGGLE_NOTEBOOK_DIR)
	$(KAGGLE) kernels init -p $(KAGGLE_NOTEBOOK_DIR)
	@echo "Created kernel-metadata.json in $(KAGGLE_NOTEBOOK_DIR). Please edit it and then run 'make kaggle-push-notebook'."

kaggle-push-notebook: kaggle-push-only

kaggle-push-only: require-kaggle-remote-approval
	@sed -i '' -e "s|\"id\": \".*\"|\"id\": \"$(KAGGLE_SLUG)\"|" $(KAGGLE_NOTEBOOK_DIR)/kernel-metadata.json
	@sed -i '' -e "s|\"title\": \".*\"|\"title\": \"$(shell echo $(KAGGLE_SLUG) | cut -d'/' -f2)\"|" $(KAGGLE_NOTEBOOK_DIR)/kernel-metadata.json
	@sed -i '' -e "s|\"enable_gpu\": [a-zA-Z]*|\"enable_gpu\": $(KAGGLE_ENABLE_GPU)|" $(KAGGLE_NOTEBOOK_DIR)/kernel-metadata.json
	@sed -i '' -e "s|\"enable_tpu\": [a-zA-Z]*|\"enable_tpu\": $(KAGGLE_ENABLE_TPU)|" $(KAGGLE_NOTEBOOK_DIR)/kernel-metadata.json
	@.venv/bin/python scripts/pre_flight_check.py --notebook_dir $(KAGGLE_NOTEBOOK_DIR) --research
	$(KAGGLE) kernels push -p $(KAGGLE_NOTEBOOK_DIR)
	@echo "Kernel $(KAGGLE_SLUG) pushed successfully on Kaggle. GPU=$(KAGGLE_ENABLE_GPU), TPU=$(KAGGLE_ENABLE_TPU)."

kaggle-push-strict-runner:
	$(MAKE) kaggle-push-only KAGGLE_NOTEBOOK_DIR=notebooks/kaggle_strict_runner KAGGLE_SLUG=$(KAGGLE_STRICT_SLUG) KAGGLE_ENABLE_GPU=false KAGGLE_ENABLE_TPU=false

kaggle-push-sequence-source-runner:
	$(MAKE) kaggle-push-only KAGGLE_NOTEBOOK_DIR=notebooks/kaggle_sequence_source_runner KAGGLE_SLUG=$(KAGGLE_SEQUENCE_SOURCE_SLUG) KAGGLE_ENABLE_GPU=false KAGGLE_ENABLE_TPU=false

kaggle-push-sequence-anchor-probe-runner:
	$(MAKE) kaggle-push-only KAGGLE_NOTEBOOK_DIR=notebooks/kaggle_sequence_anchor_probe_runner KAGGLE_SLUG=$(KAGGLE_SEQUENCE_ANCHOR_PROBE_SLUG) KAGGLE_ENABLE_GPU=false KAGGLE_ENABLE_TPU=false

kaggle-push-candidate-parity-runner:
	$(MAKE) kaggle-push-only KAGGLE_NOTEBOOK_DIR=notebooks/kaggle_candidate_parity_runner KAGGLE_SLUG=$(KAGGLE_CANDIDATE_PARITY_SLUG) KAGGLE_ENABLE_GPU=false KAGGLE_ENABLE_TPU=false

kaggle-push-upstream-rebuild-runner:
	$(MAKE) kaggle-push-only KAGGLE_NOTEBOOK_DIR=notebooks/kaggle_upstream_rebuild_runner KAGGLE_SLUG=$(KAGGLE_UPSTREAM_REBUILD_SLUG) KAGGLE_ENABLE_GPU=false KAGGLE_ENABLE_TPU=false

kaggle-push-self-supervised-subject-state-runner:
	$(MAKE) kaggle-push-only KAGGLE_NOTEBOOK_DIR=notebooks/kaggle_self_supervised_subject_state_runner KAGGLE_SLUG=$(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_SLUG) KAGGLE_ENABLE_GPU=false KAGGLE_ENABLE_TPU=false

kaggle-push-missingness-subject-state-runner:
	$(MAKE) kaggle-push-only KAGGLE_NOTEBOOK_DIR=notebooks/kaggle_missingness_subject_state_runner KAGGLE_SLUG=$(KAGGLE_MISSINGNESS_SUBJECT_STATE_SLUG) KAGGLE_ENABLE_GPU=false KAGGLE_ENABLE_TPU=false

kaggle-check-base-dataset-files: require-kaggle-remote-approval
	$(KAGGLE) datasets files softkleenex/dacon-etri-human-base

kaggle-check-upstream-patch-dataset-files: require-kaggle-remote-approval
	$(KAGGLE) datasets files $(KAGGLE_UPSTREAM_PATCH_DATASET_SLUG)

kaggle-check-self-supervised-subject-state-patch-dataset-files: require-kaggle-remote-approval
	$(KAGGLE) datasets files $(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_SLUG)

kaggle-kernel-status-upstream-rebuild: require-kaggle-remote-approval
	$(KAGGLE) kernels status $(KAGGLE_UPSTREAM_REBUILD_SLUG)

kaggle-kernel-status-self-supervised-subject-state: require-kaggle-remote-approval
	$(KAGGLE) kernels status $(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_SLUG)

kaggle-kernel-status-missingness-subject-state: require-kaggle-remote-approval
	$(KAGGLE) kernels status $(KAGGLE_MISSINGNESS_SUBJECT_STATE_SLUG)

kaggle-pull-output: require-kaggle-remote-approval
	$(KAGGLE) kernels output $(KAGGLE_SLUG) -p $(PULL_DIR) --force
	@echo "Downloaded outputs to $(PULL_DIR) directory."

kaggle-pull-strict-output:
	$(MAKE) kaggle-pull-output KAGGLE_SLUG=$(KAGGLE_STRICT_SLUG) PULL_DIR=build/kaggle_outputs/strict

kaggle-pull-sequence-source-output:
	$(MAKE) kaggle-pull-output KAGGLE_SLUG=$(KAGGLE_SEQUENCE_SOURCE_SLUG) PULL_DIR=build/kaggle_outputs/sequence_source_post_id335

kaggle-pull-sequence-anchor-probe-output:
	$(MAKE) kaggle-pull-output KAGGLE_SLUG=$(KAGGLE_SEQUENCE_ANCHOR_PROBE_SLUG) PULL_DIR=build/kaggle_outputs/sequence_anchor_probe_post_id335

kaggle-pull-candidate-parity-output:
	$(MAKE) kaggle-pull-output KAGGLE_SLUG=$(KAGGLE_CANDIDATE_PARITY_SLUG) PULL_DIR=build/kaggle_outputs/candidate_parity_post_id339

kaggle-pull-upstream-rebuild-output: require-kaggle-remote-approval
	$(MAKE) kaggle-pull-output KAGGLE_SLUG=$(KAGGLE_UPSTREAM_REBUILD_SLUG) PULL_DIR=build/kaggle_outputs/upstream_rebuild_post_id341

kaggle-pull-self-supervised-subject-state-output: require-kaggle-remote-approval
	$(MAKE) kaggle-pull-output KAGGLE_SLUG=$(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_SLUG) PULL_DIR=build/kaggle_outputs/self_supervised_subject_state_post_id342

kaggle-pull-missingness-subject-state-output: require-kaggle-remote-approval
	$(MAKE) kaggle-pull-output KAGGLE_SLUG=$(KAGGLE_MISSINGNESS_SUBJECT_STATE_SLUG) PULL_DIR=build/kaggle_outputs/missingness_subject_state_post_id342

kaggle-wait-and-pull-upstream-rebuild-output: require-kaggle-remote-approval
	$(MAKE) kaggle-wait-and-pull KAGGLE_SLUG=$(KAGGLE_UPSTREAM_REBUILD_SLUG) PULL_DIR=build/kaggle_outputs/upstream_rebuild_post_id341

kaggle-wait-and-pull-self-supervised-subject-state-output: require-kaggle-remote-approval
	$(MAKE) kaggle-wait-and-pull KAGGLE_SLUG=$(KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_SLUG) PULL_DIR=build/kaggle_outputs/self_supervised_subject_state_post_id342

kaggle-wait-and-pull-missingness-subject-state-output: require-kaggle-remote-approval
	$(MAKE) kaggle-wait-and-pull KAGGLE_SLUG=$(KAGGLE_MISSINGNESS_SUBJECT_STATE_SLUG) PULL_DIR=build/kaggle_outputs/missingness_subject_state_post_id342

kaggle-upstream-prelaunch-local:
	.venv/bin/python scripts/diagnose_post_id341_target_gap_budget.py --output reports/post_id341_target_gap_budget_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_missing_target_signal_inventory.py --output reports/post_id341_missing_target_signal_inventory_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_q3s3_physio_light_onset_smoke.py --output reports/post_id341_q3s3_physio_light_onset_smoke_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_q3s3_semantic_routine_v2_design.py --output reports/post_id341_q3s3_semantic_routine_v2_design_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_q3s3_semantic_routine_metadata_smoke.py --output reports/post_id341_q3s3_semantic_routine_metadata_smoke_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_q3s3_semantic_routine_resource_gate.py --output reports/post_id341_q3s3_semantic_routine_resource_gate_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_q3s3_semantic_routine_model_smoke.py --output reports/post_id341_q3s3_semantic_routine_model_smoke_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_local_smoke_freshness_gate.py --output reports/post_id341_local_smoke_freshness_gate_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_q3s3_semantic_routine_lane_closure.py --output reports/post_id341_q3s3_semantic_routine_lane_closure_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_public_local_gap.py --output reports/post_id341_public_local_gap_diagnosis_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_data_source_coverage.py --output reports/post_id341_data_source_coverage_audit_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_local_smoke_freshness_gate.py --output reports/post_id341_local_smoke_freshness_gate_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_q2q3s3_source_hypothesis_matrix.py --output reports/post_id341_q2q3s3_source_hypothesis_matrix_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_q2q3s3_source_route.py --output reports/post_id341_q2q3s3_source_route_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_q2q3s3_env_social_routine_v2_design.py --output reports/post_id341_q2q3s3_env_social_routine_v2_design_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_q2q3s3_feature_generation_metadata_smoke.py --output reports/post_id341_q2q3s3_feature_generation_metadata_smoke_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_q2q3s3_fold_internal_model_smoke.py --output reports/post_id341_q2q3s3_fold_internal_model_smoke_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_q2q3s3_q2_lead_geometry_gate.py --output reports/post_id341_q2q3s3_q2_lead_geometry_gate_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_q2q3s3_paper_family_fold_smoke.py --output reports/post_id341_q2q3s3_paper_family_fold_smoke_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_q3s3_anchor_failure_profile.py --output reports/post_id341_q3s3_anchor_failure_profile_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_q3s3_calendar_sleep_prior_smoke.py --output reports/post_id341_q3s3_calendar_sleep_prior_smoke_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_q3s3_subject_drift_gate.py --output reports/post_id341_q3s3_subject_drift_gate_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_q3s3_event_cadence_smoke.py --output reports/post_id341_q3s3_event_cadence_smoke_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_q3s3_local_lane_closure.py --output reports/post_id341_q3s3_local_lane_closure_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_q3s3_negative_evidence_map.py --output reports/post_id341_q3s3_negative_evidence_map_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_q3s3_fold_nested_sibling_bridge_smoke.py --output reports/post_id341_q3s3_fold_nested_sibling_bridge_smoke_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_q3s3_remaining_route_inventory.py --output reports/post_id341_q3s3_remaining_route_inventory_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_q2q3s3_local_fallback_chain_contract.py --output reports/post_id341_q2q3s3_local_fallback_chain_contract_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_external_public05917_residual_audit.py --output reports/post_id341_external_public05917_residual_audit_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_material_new_source_queue.py --output reports/post_id341_material_new_source_queue_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_usage_semantic_burst_markov_design.py --output reports/post_id341_usage_semantic_burst_markov_design_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_usage_semantic_burst_markov_metadata_smoke.py --output reports/post_id341_usage_semantic_burst_markov_metadata_smoke_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_usage_semantic_burst_markov_feature_spec.py --output reports/post_id341_usage_semantic_burst_markov_feature_spec_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_usage_semantic_burst_markov_feature_generation_dryrun.py --output reports/post_id341_usage_semantic_burst_markov_feature_generation_dryrun_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_usage_semantic_burst_markov_feature_matrix_smoke_contract.py --output reports/post_id341_usage_semantic_burst_markov_feature_matrix_smoke_contract_20260619.json
	.venv/bin/python scripts/write_post_id341_usage_semantic_burst_markov_feature_matrix.py
	.venv/bin/python scripts/diagnose_post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop.py --output reports/post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_proximity_ambience_cooccurrence_design.py --output reports/post_id341_proximity_ambience_cooccurrence_design_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_proximity_ambience_cooccurrence_metadata_smoke.py --output reports/post_id341_proximity_ambience_cooccurrence_metadata_smoke_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_proximity_ambience_cooccurrence_feature_spec.py --output reports/post_id341_proximity_ambience_cooccurrence_feature_spec_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_proximity_ambience_cooccurrence_feature_generation_dryrun.py --output reports/post_id341_proximity_ambience_cooccurrence_feature_generation_dryrun_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_proximity_ambience_cooccurrence_feature_matrix_smoke_contract.py --output reports/post_id341_proximity_ambience_cooccurrence_feature_matrix_smoke_contract_20260619.json
	.venv/bin/python scripts/write_post_id341_proximity_ambience_cooccurrence_feature_matrix.py
	.venv/bin/python scripts/diagnose_post_id341_proximity_ambience_cooccurrence_feature_matrix_validation_stop.py --output reports/post_id341_proximity_ambience_cooccurrence_feature_matrix_validation_stop_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_phone_wearable_discordance_design.py --output reports/post_id341_phone_wearable_discordance_design_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_phone_wearable_discordance_feature_spec.py --output reports/post_id341_phone_wearable_discordance_feature_spec_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_phone_wearable_discordance_feature_generation_dryrun.py --output reports/post_id341_phone_wearable_discordance_feature_generation_dryrun_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_phone_wearable_discordance_feature_matrix_smoke_contract.py --output reports/post_id341_phone_wearable_discordance_feature_matrix_smoke_contract_20260619.json
	.venv/bin/python scripts/write_post_id341_phone_wearable_discordance_feature_matrix.py
	.venv/bin/python scripts/diagnose_post_id341_phone_wearable_discordance_feature_matrix_validation_stop.py --output reports/post_id341_phone_wearable_discordance_feature_matrix_validation_stop_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_material_new_source_queue.py --output reports/post_id341_material_new_source_queue_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_kaggle_runner_intake_contract.py --output reports/post_id341_kaggle_runner_intake_contract_20260618.json
	.venv/bin/python scripts/diagnose_kaggle_upstream_execution_packet_post_id341.py --output reports/kaggle_upstream_execution_packet_post_id341_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_local_source_closure_synthesis.py --output reports/post_id341_local_source_closure_synthesis_20260618.json
	.venv/bin/python scripts/diagnose_kaggle_source_zip_selftest_post_id341.py --output reports/kaggle_source_zip_selftest_post_id341_20260618.json
	.venv/bin/python scripts/diagnose_kaggle_upstream_execution_packet_post_id341.py --output reports/kaggle_upstream_execution_packet_post_id341_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_local_source_closure_synthesis.py --output reports/post_id341_local_source_closure_synthesis_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_kaggle_launch_approval_note.py --output reports/post_id341_kaggle_launch_approval_note_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_kaggle_output_intake_gate.py --output reports/post_id341_kaggle_output_intake_gate_20260618.json
	.venv/bin/python scripts/summarize_post_id341_kaggle_remote_results.py --output reports/post_id341_kaggle_remote_result_summary_20260618.json
	.venv/bin/python scripts/decide_post_id341_remote_next_action.py --output reports/post_id341_remote_next_action_decision_20260618.json
	.venv/bin/python scripts/decide_post_id341_remote_artifact_review_threshold.py --output reports/post_id341_remote_artifact_review_threshold_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_public058_artifact_promotion_contract.py --output reports/post_id341_public058_artifact_promotion_contract_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_post_pull_review_chain_contract.py --output reports/post_id341_post_pull_review_chain_contract_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_runner_score_relevance_contract.py --output reports/post_id341_runner_score_relevance_contract_20260618.json
	.venv/bin/python scripts/audit_kaggle_guarded_remote_contract.py --output reports/kaggle_guarded_remote_contract_audit_20260618.json
	.venv/bin/python scripts/audit_kaggle_remote_fail_closed_dryrun.py --output reports/kaggle_remote_fail_closed_dryrun_audit_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_local_smoke_freshness_gate.py --output reports/post_id341_local_smoke_freshness_gate_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_local_remote_compute_boundary_contract.py --output reports/post_id341_local_remote_compute_boundary_contract_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_no_approval_frontier.py --output reports/post_id341_no_approval_frontier_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_operator_handoff_contract.py --output reports/post_id341_operator_handoff_contract_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_self_supervised_subject_state_embedding_design.py --output reports/post_id341_self_supervised_subject_state_embedding_design_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_sleep_proxy_threshold_residual_design.py --output reports/post_id341_sleep_proxy_threshold_residual_design_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_material_new_source_queue.py --output reports/post_id341_material_new_source_queue_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_material_source_remote_coverage_contract.py --output reports/post_id341_material_source_remote_coverage_contract_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_kaggle_prelaunch_readiness_gate.py --output reports/post_id341_kaggle_prelaunch_readiness_gate_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_prelaunch_freshness_contract.py --output reports/post_id341_prelaunch_freshness_contract_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_self_supervised_heavy_compute_contract.py --output reports/post_id341_self_supervised_heavy_compute_contract_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_safe_wait_state_contract.py --output reports/post_id341_safe_wait_state_contract_20260619.json
	.venv/bin/python scripts/diagnose_post_id341_next_material_route_triage.py --output reports/post_id341_next_material_route_triage_20260619.json

kaggle-approval-contract-test:
	.venv/bin/ruff check $(KAGGLE_APPROVAL_CONTRACT_LINT_PATHS)
	.venv/bin/ruff format --check $(KAGGLE_APPROVAL_CONTRACT_LINT_PATHS)
	PYTHONPATH=. .venv/bin/pytest -q $(KAGGLE_APPROVAL_CONTRACT_TESTS)

dacon-submit-approval-contract-test:
	.venv/bin/ruff check $(DACON_SUBMIT_APPROVAL_CONTRACT_SCRIPTS) $(DACON_SUBMIT_APPROVAL_CONTRACT_TESTS)
	.venv/bin/ruff format --check $(DACON_SUBMIT_APPROVAL_CONTRACT_SCRIPTS) $(DACON_SUBMIT_APPROVAL_CONTRACT_TESTS)
	PYTHONPATH=. .venv/bin/pytest -q $(DACON_SUBMIT_APPROVAL_CONTRACT_TESTS)

kaggle-local-runner-smoke:
	.venv/bin/python scripts/run_post_id341_kaggle_runner_local_smoke.py --output reports/post_id341_kaggle_runner_local_smoke_20260618.json

kaggle-upstream-ready-local:
	$(MAKE) kaggle-local-runner-smoke
	.venv/bin/python scripts/diagnose_post_id341_local_smoke_freshness_gate.py --output reports/post_id341_local_smoke_freshness_gate_20260618.json
	$(MAKE) kaggle-upstream-prelaunch-local
	$(MAKE) kaggle-approval-contract-test

kaggle-review-upstream-rebuild-output:
	.venv/bin/python scripts/diagnose_post_id341_kaggle_output_intake_gate.py --output reports/post_id341_kaggle_output_intake_gate_20260618.json
	.venv/bin/python scripts/summarize_post_id341_kaggle_remote_results.py --output reports/post_id341_kaggle_remote_result_summary_20260618.json
	.venv/bin/python scripts/decide_post_id341_remote_next_action.py --output reports/post_id341_remote_next_action_decision_20260618.json
	.venv/bin/python scripts/decide_post_id341_remote_artifact_review_threshold.py --output reports/post_id341_remote_artifact_review_threshold_20260618.json
	.venv/bin/python scripts/diagnose_post_id341_public058_artifact_promotion_contract.py --output reports/post_id341_public058_artifact_promotion_contract_20260619.json

kaggle-wait-and-pull: require-kaggle-remote-approval
	@echo "Waiting for Kaggle kernel $(KAGGLE_SLUG) to finish..."
	@while true; do \
		status_output="$$($(KAGGLE) kernels status $(KAGGLE_SLUG) 2>&1)"; \
		echo "$$status_output"; \
		if echo "$$status_output" | grep -Eiq 'complete|success'; then \
			break; \
		fi; \
		if echo "$$status_output" | grep -Eiq 'running|queued|pending'; then \
			echo "Kernel is not complete yet... sleeping for 60 seconds."; \
			sleep 60; \
		else \
			echo "Kernel status is not complete/running/queued/pending; refusing to pull outputs."; \
			exit 3; \
		fi; \
	done; \
	echo "Kernel finished. Pulling output..."; \
	$(KAGGLE) kernels output $(KAGGLE_SLUG) -p $(PULL_DIR) --force
	@echo "Downloaded outputs to $(PULL_DIR) directory."

ensemble:
	.venv/bin/python scripts/ensemble.py

SUB_PATH ?= "submissions/submission_ensemble_latest.csv"
SUB_MEMO ?= "Automated submission"

require-dacon-submit-approval:
	@if [ "$(ALLOW_DACON_SUBMIT)" != "true" ]; then \
		echo "Error: DACON submission blocked. Re-run with ALLOW_DACON_SUBMIT=true only after explicit approval of the exact SUB_PATH and SUB_MEMO."; \
		exit 2; \
	fi; \
	if [ "$(DACON_APPROVED_SUB_PATH)" != "$(SUB_PATH)" ]; then \
		echo "Error: DACON exact path approval mismatch. DACON_APPROVED_SUB_PATH must exactly equal SUB_PATH."; \
		exit 3; \
	fi; \
	if [ "$(DACON_APPROVED_SUB_MEMO)" != "$(SUB_MEMO)" ]; then \
		echo "Error: DACON exact memo approval mismatch. DACON_APPROVED_SUB_MEMO must exactly equal SUB_MEMO."; \
		exit 4; \
	fi
	@.venv/bin/python scripts/audit_dacon_submit_approval_contract.py --sub-path "$(SUB_PATH)" --sub-memo "$(SUB_MEMO)" --policy-report "$(DACON_SUBMIT_POLICY_REPORT)"

dacon-submit: require-dacon-submit-approval
	@.venv/bin/python -c "import os; \
from dacon_submit_api import dacon_submit_api; \
dacon_token = os.environ.get('DACON_TOKEN'); \
team_name = os.environ.get('DACON_TEAMNAME'); \
sub_path = '$(SUB_PATH)'; \
sub_memo = '$(SUB_MEMO)'; \
assert dacon_token and team_name, 'Error: DACON_TOKEN or DACON_TEAMNAME not set. Please export them in your shell.'; \
print(f'Submitting {sub_path} to Dacon...'); \
result = dacon_submit_api.post_submission_file( \
    file_path=sub_path, \
    token=dacon_token, \
    cpt_id='$(strip $(COMPETITION_ID))', \
    team_name=team_name, \
    memo=sub_memo \
); \
print('Result:', result);"
