# Post-ID341 Q3/S3 semantic-routine metadata smoke

Created: `2026-06-23 18:07:40 UTC+09:00`

Verify with bounded raw samples that Q3 semantic novelty and S3 ordered state-transition features are structurally computable before any model or artifact work.

This is report-only and bounded-sample only. It creates no predictions, CSVs, Kaggle runs, or DACON submissions.

## Decision

- Metadata smoke passed: `True`
- Target focus: `['Q3', 'S3']`
- Q3 ready sources: `['mUsageStats', 'mWifi', 'mBle', 'mGps', 'mAmbience']`
- S3 ready sources: `['mScreenStatus', 'mActivity', 'mACStatus', 'mLight', 'wLight', 'wPedo']`
- Public 0.58 evidence: `False`
- Next local allowed step: `resource_gated_fold_internal_semantic_routine_model_smoke_only`
- Requires resource gate before model smoke: `True`
- Create submission artifact: `False`
- Submit now: `False`
- Next action: Metadata/cardinality smoke passed. The next step may only be a no-artifact resource-gated fold-internal model smoke; do not create ID342 or any submission artifacts.

## Sample policy

- max_sample_rows_per_file: `4096`
- batch_size: `1024`
- full_file_load_allowed: `False`
- model_training_allowed: `False`
- raw_identifier_keys_not_emitted: `address`, `app_name`, `bssid`, `latitude`, `longitude`

## Checks

- design_ready: `True`
- all_sources_readable: `True`
- bounded_sample_only: `True`
- q3_metadata_ready: `True`
- s3_state_machine_metadata_ready: `True`
- privacy_no_raw_identifier_examples: `True`
- no_artifact_actions: `True`

## Source summaries

| source | sample rows | non-null share | temporal coverage | nested tokens | scalar uniques | ready |
|---|---:|---:|---|---:|---:|---|
| mUsageStats | 4096 | 1.0000 | True | 144 | 0 | True |
| mWifi | 4096 | 1.0000 | True | 8256 | 0 | True |
| mBle | 4096 | 1.0000 | True | 64315 | 0 | True |
| mGps | 4096 | 1.0000 | True | 343 | 0 | True |
| mAmbience | 4096 | 1.0000 | True | 267 | 0 | True |
| mScreenStatus | 4096 | 1.0000 | True | 0 | 2 | True |
| mActivity | 4096 | 1.0000 | True | 0 | 5 | True |
| mACStatus | 4096 | 1.0000 | True | 0 | 2 | True |
| mLight | 4096 | 1.0000 | True | 0 | 657 | True |
| wLight | 4096 | 1.0000 | True | 0 | 264 | True |
| wPedo | 4096 | 1.0000 | True | 0 | 94 | True |
| wHr | 4096 | 1.0000 | False | 0 | 0 | False |

