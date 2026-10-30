# Competition Workspace Audit

## Executive Summary

Found **2 issues**: 0 critical, 2 high, 0 medium.

## Issues

### 1. [HIGH] Historical submitted rows lack real public LB scores

- Area: `submissions`

- Detail: 45 rows are now labeled via status_clean/risk_reason.

- Recommendation: Use safe_for_default_ensemble and status_clean instead of raw is_submitted for candidate selection.

### 2. [HIGH] Large CV-LB gaps exist

- Area: `validation`

- Detail: 16 scored rows have |public-local| >= 0.03.

- Recommendation: Treat these as validation failures; compare their feature sets and fold strategy before ensembling them.

## Key Stats

- `git_status_by_code`: `{"M": 8}`

- `git_status_entries`: `8`

- `largest_cv_lb_gaps`: `[{"id": "287", "file": "submission_id_287_post285_base282_fixedtemp_balanced_cv0.378934_unsubmitted.csv", "local": "0.378934", "public": "0.6698787318", "gap": 0.290945}, {"id": "9", "file": "submission_id_009_score0.6258.csv", "local": "0.6258", "public": "0.8208847151", "gap": 0.195085}, {"id": "74", "file": "submission_id_074_score0.5986.csv", "local": "0.5986", "public": "0.7436431908", "gap": 0.145043}, {"id": "73", "file": "submission_id_073_score0.6022.csv", "local": "0.6022", "public": "0.7454473391", "gap": 0.143247}, {"id": "329", "file": "submission_id_329_post328_combo4_cap080_budget050_practical_cv0.516718_unsubmitted.csv", "local": "0.516718", "public": "0.6017908577", "gap": 0.085073}, {"id": "340", "file": "submission_id_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426_unsubmitted.csv", "local": "0.538426", "public": "0.601379175253277", "gap": 0.062953}, {"id": "178", "file": "submission_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638_unsubmitted.csv",`

- `logged_files_missing`: `0`

- `make_targets`: `["DACON_SUBMIT_APPROVAL_CONTRACT_SCRIPTS", "DACON_SUBMIT_APPROVAL_CONTRACT_TESTS", "KAGGLE_APPROVAL_CONTRACT_LINT_PATHS", "KAGGLE_APPROVAL_CONTRACT_TESTS", "KAGGLE_DATASET_DIR", "KAGGLE_SELF_SUPERVISED_SUBJECT_STATE_PATCH_DATASET_DIR", "KAGGLE_UPSTREAM_PATCH_DATASET_DIR", "LINT_PATHS", "LOCAL_REMOTE_COMPUTE_BOUNDARY_SCRIPTS", "LOCAL_REMOTE_COMPUTE_BOUNDARY_TESTS", "LOCAL_SOURCE_CLOSURE_SYNTHESIS_SCRIPTS", "LOCAL_SOURCE_CLOSURE_SYNTHESIS_TESTS", "MATERIAL_SOURCE_REMOTE_COVERAGE_SCRIPTS", "MATERIAL_SOURCE_REMOTE_COVERAGE_TESTS", "NEXT_MATERIAL_ROUTE_TRIAGE_SCRIPTS", "NEXT_MATERIAL_ROUTE_TRIAGE_TESTS", "PHONE_WEARABLE_DISCORDANCE_SCRIPTS", "PHONE_WEARABLE_DISCORDANCE_TESTS", "PRELAUNCH_FRESHNESS_CONTRACT_SCRIPTS", "PRELAUNCH_FRESHNESS_CONTRACT_TESTS", "PROXIMITY_AMBIENCE_COOCCURRENCE_SCRIPTS", "PROXIMITY_AMBIENCE_COOCCURRENCE_TESTS", "PUBLIC058_ARTIFACT_PROMOTION_CONTRACT_SCRIPTS", "PUBLIC058_ARTIFACT_PROMOTION_CONTRACT_TESTS", "SAFE_WAIT_STATE_CONTRACT_SCRIPTS", "SAFE_WAIT_STATE_CONTRACT`

- `missing_markdown_links`: `[]`

- `missing_scripts_referenced_by_makefile`: `[]`

- `referenced_missing_make_targets`: `[]`

- `size_checkpoints`: `0B`

- `size_data`: `475.9MB`

- `size_docs`: `1.1MB`

- `size_lightning_logs`: `0B`

- `size_models`: `0B`

- `size_notebooks`: `4.8MB`

- `size_scratch`: `0B`

- `size_scripts`: `9.1MB`

- `size_src`: `428.9KB`

- `size_submissions`: `18.4MB`

- `submission_rows`: `349`

- `submitted_missing_public_or_zero`: `45`

- `submitted_rows`: `76`

- `tracked_generated_files`: `0`
