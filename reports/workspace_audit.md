# Competition Workspace Audit

## Executive Summary

Found **4 issues**: 0 critical, 2 high, 2 medium.

## Issues

### 1. [HIGH] Historical submitted rows lack real public LB scores

- Area: `submissions`

- Detail: 44 rows are now labeled via status_clean/risk_reason.

- Recommendation: Use safe_for_default_ensemble and status_clean instead of raw is_submitted for candidate selection.

### 2. [HIGH] Large CV-LB gaps exist

- Area: `validation`

- Detail: 5 scored rows have |public-local| >= 0.03.

- Recommendation: Treat these as validation failures; compare their feature sets and fold strategy before ensembling them.

### 3. [MEDIUM] Model directory is very large

- Area: `artifacts`

- Detail: models/ is 9.0GB.

- Recommendation: Keep only champion artifacts locally; move old runs to external storage or ignored archives.

### 4. [MEDIUM] Submissions directory is bloated

- Area: `artifacts`

- Detail: submissions/ is 1.5GB.

- Recommendation: Retain top submissions plus metadata; move embeddings/OOFs to an experiments artifact directory.

## Key Stats

- `git_status_by_code`: `{}`

- `git_status_entries`: `0`

- `largest_cv_lb_gaps`: `[{"id": "9", "file": "submission_id_009_score0.6258.csv", "local": "0.6258", "public": "0.8208847151", "gap": 0.195085}, {"id": "74", "file": "submission_id_074_score0.5986.csv", "local": "0.5986", "public": "0.7436431908", "gap": 0.145043}, {"id": "73", "file": "submission_id_073_score0.6022.csv", "local": "0.6022", "public": "0.7454473391", "gap": 0.143247}, {"id": "10", "file": "submission_id_010_score0.5972.csv", "local": "0.5972", "public": "0.6301928255", "gap": 0.032993}, {"id": "16", "file": "submission_id_016_score0.5875.csv", "local": "0.5875", "public": "0.6184828015", "gap": 0.030983}, {"id": "11", "file": "submission_id_011_score0.5909.csv", "local": "0.5909", "public": "0.6208484881", "gap": 0.029948}, {"id": "7", "file": "submission_id_007_score0.5921.csv", "local": "0.5921", "public": "0.6166592444", "gap": 0.024559}, {"id": "76", "file": "submission_id_076_score0.5996.csv", "local": "0.599611", "public": "0.6241517536", "gap": 0.024541}]`

- `logged_files_missing`: `0`

- `make_targets`: `["KAGGLE_DATASET_DIR", "LINT_PATHS", "audit", "build-kaggle-dataset", "cleanup-manifest", "dacon-submit", "ensemble", "format", "help", "install", "jupyter", "kaggle-init-code", "kaggle-init-notebook", "kaggle-pull-output", "kaggle-push-code", "kaggle-push-notebook", "kaggle-push-only", "kaggle-wait-and-pull", "lint", "normalize-submissions", "run-local", "test"]`

- `missing_markdown_links`: `[]`

- `missing_scripts_referenced_by_makefile`: `[]`

- `referenced_missing_make_targets`: `[]`

- `size_checkpoints`: `0B`

- `size_data`: `2.8GB`

- `size_docs`: `181.4KB`

- `size_lightning_logs`: `0B`

- `size_models`: `9.0GB`

- `size_notebooks`: `4.3MB`

- `size_scratch`: `0B`

- `size_scripts`: `440.1KB`

- `size_src`: `307.8KB`

- `size_submissions`: `1.5GB`

- `submission_rows`: `93`

- `submitted_missing_public_or_zero`: `44`

- `submitted_rows`: `63`

- `tracked_generated_files`: `0`
