# Competition Workspace Audit

## Executive Summary

Found **3 issues**: 1 critical, 2 high, 0 medium.

## Issues

### 1. [CRITICAL] Working tree is too dirty for reliable experiments

- Area: `repo hygiene`

- Detail: git status has 1185 entries.

- Recommendation: Create a checkpoint branch, commit/restash intentional code, and keep generated artifacts ignored.

### 2. [HIGH] Historical submitted rows lack real public LB scores

- Area: `submissions`

- Detail: 44 rows are now labeled via status_clean/risk_reason.

- Recommendation: Use safe_for_default_ensemble and status_clean instead of raw is_submitted for candidate selection.

### 3. [HIGH] Large CV-LB gaps exist

- Area: `validation`

- Detail: 8 scored rows have |public-local| >= 0.03.

- Recommendation: Treat these as validation failures; compare their feature sets and fold strategy before ensembling them.

## Key Stats

- `git_status_by_code`: `{"M": 26, "??": 1159}`

- `git_status_entries`: `1185`

- `largest_cv_lb_gaps`: `[{"id": "287", "file": "submission_id_287_post285_base282_fixedtemp_balanced_cv0.378934_unsubmitted.csv", "local": "0.378934", "public": "0.6698787318", "gap": 0.290945}, {"id": "9", "file": "submission_id_009_score0.6258.csv", "local": "0.6258", "public": "0.8208847151", "gap": 0.195085}, {"id": "74", "file": "submission_id_074_score0.5986.csv", "local": "0.5986", "public": "0.7436431908", "gap": 0.145043}, {"id": "73", "file": "submission_id_073_score0.6022.csv", "local": "0.6022", "public": "0.7454473391", "gap": 0.143247}, {"id": "329", "file": "submission_id_329_post328_combo4_cap080_budget050_practical_cv0.516718_unsubmitted.csv", "local": "0.516718", "public": "0.6017908577", "gap": 0.085073}, {"id": "178", "file": "submission_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638_unsubmitted.csv", "local": "0.538638", "public": "0.6014113009", "gap": 0.062773}, {"id": "10", "file": "submission_id_010_score0.5972.csv", "local": "0.5972", "public": "0.6301928255", "gap":`

- `logged_files_missing`: `0`

- `make_targets`: `["KAGGLE_DATASET_DIR", "LINT_PATHS", "artifact-contract-audit", "audit", "build-kaggle-dataset", "cleanup-manifest", "dacon-submit", "data-source-coverage-audit", "docs-audit", "ensemble", "format", "help", "install", "jupyter", "kaggle-check-base-dataset-files", "kaggle-init-code", "kaggle-init-notebook", "kaggle-kernel-status-upstream-rebuild", "kaggle-pull-candidate-parity-output", "kaggle-pull-output", "kaggle-pull-sequence-anchor-probe-output", "kaggle-pull-sequence-source-output", "kaggle-pull-strict-output", "kaggle-pull-upstream-rebuild-output", "kaggle-push-candidate-parity-runner", "kaggle-push-code", "kaggle-push-notebook", "kaggle-push-only", "kaggle-push-sequence-anchor-probe-runner", "kaggle-push-sequence-source-runner", "kaggle-push-strict-runner", "kaggle-push-upstream-rebuild-runner", "kaggle-review-upstream-rebuild-output", "kaggle-upstream-prelaunch-local", "kaggle-wait-and-pull", "lint", "normalize-submissions", "public-local-gap-diagnosis", "require-kaggle-remote-a`

- `missing_markdown_links`: `[]`

- `missing_scripts_referenced_by_makefile`: `[]`

- `referenced_missing_make_targets`: `[]`

- `size_checkpoints`: `0B`

- `size_data`: `126.3MB`

- `size_docs`: `513.1KB`

- `size_lightning_logs`: `0B`

- `size_models`: `0B`

- `size_notebooks`: `4.7MB`

- `size_scratch`: `0B`

- `size_scripts`: `3.4MB`

- `size_src`: `195.7KB`

- `size_submissions`: `17.9MB`

- `submission_rows`: `341`

- `submitted_missing_public_or_zero`: `44`

- `submitted_rows`: `67`

- `tracked_generated_files`: `0`
