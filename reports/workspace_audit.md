# Competition Workspace Audit

## Executive Summary

Found **2 issues**: 0 critical, 2 high, 0 medium.

## Issues

### 1. [HIGH] Historical submitted rows lack real public LB scores

- Area: `submissions`

- Detail: 44 rows are now labeled via status_clean/risk_reason.

- Recommendation: Use safe_for_default_ensemble and status_clean instead of raw is_submitted for candidate selection.

### 2. [HIGH] Large CV-LB gaps exist

- Area: `validation`

- Detail: 5 scored rows have |public-local| >= 0.03.

- Recommendation: Treat these as validation failures; compare their feature sets and fold strategy before ensembling them.

## Key Stats

- `git_status_by_code`: `{"M": 1}`

- `git_status_entries`: `1`

- `largest_cv_lb_gaps`: `[{"id": "9", "file": "submission_id_009_score0.6258.csv", "local": "0.6258", "public": "0.8208847151", "gap": 0.195085}, {"id": "74", "file": "submission_id_074_score0.5986.csv", "local": "0.5986", "public": "0.7436431908", "gap": 0.145043}, {"id": "73", "file": "submission_id_073_score0.6022.csv", "local": "0.6022", "public": "0.7454473391", "gap": 0.143247}, {"id": "10", "file": "submission_id_010_score0.5972.csv", "local": "0.5972", "public": "0.6301928255", "gap": 0.032993}, {"id": "16", "file": "submission_id_016_score0.5875.csv", "local": "0.5875", "public": "0.6184828015", "gap": 0.030983}, {"id": "11", "file": "submission_id_011_score0.5909.csv", "local": "0.5909", "public": "0.6208484881", "gap": 0.029948}, {"id": "7", "file": "submission_id_007_score0.5921.csv", "local": "0.5921", "public": "0.6166592444", "gap": 0.024559}, {"id": "76", "file": "submission_id_076_score0.5996.csv", "local": "0.599611", "public": "0.6241517536", "gap": 0.024541}]`

- `logged_files_missing`: `0`

- `make_targets`: `["KAGGLE_DATASET_DIR", "LINT_PATHS", "audit", "build-kaggle-dataset", "cleanup-manifest", "dacon-submit", "docs-audit", "ensemble", "format", "help", "install", "jupyter", "kaggle-init-code", "kaggle-init-notebook", "kaggle-pull-output", "kaggle-push-code", "kaggle-push-notebook", "kaggle-push-only", "kaggle-wait-and-pull", "lint", "normalize-submissions", "run-local", "structure-audit", "submissions-audit", "test"]`

- `missing_markdown_links`: `[]`

- `missing_scripts_referenced_by_makefile`: `[]`

- `referenced_missing_make_targets`: `[]`

- `size_checkpoints`: `0B`

- `size_data`: `123.2MB`

- `size_docs`: `181.6KB`

- `size_lightning_logs`: `0B`

- `size_models`: `0B`

- `size_notebooks`: `4.3MB`

- `size_scratch`: `0B`

- `size_scripts`: `471.2KB`

- `size_src`: `156.5KB`

- `size_submissions`: `6.5MB`

- `submission_rows`: `93`

- `submitted_missing_public_or_zero`: `44`

- `submitted_rows`: `63`

- `tracked_generated_files`: `0`
