# Post-ID342 heavy runtime budget

Created: `2026-06-24 10:26:14 UTC+09:00`

This report uses file sizes and parquet footer metadata only. It reads no raw row payloads, creates no token cache, fits no embeddings, runs no Kaggle job, and submits nothing.

## Decision

- Runtime budget ready: `True`
- Metadata-only read: `True`
- Raw payload rows read: `0`
- Total raw file size MiB: `122.832`
- Total raw rows from metadata: `6122179`
- All modalities single row group: `True`
- Recommended scanner batch rows after approval: `65536`
- Recommended local day batch after approval: `16`
- Recommended Kaggle day batch after approval: `64`
- Run Kaggle now: `False`
- DACON submit now: `False`
- Required phrase: `Self-supervised subject-state heavy compute no-submit 실행 승인`
- Next action: Use scanner/batch streaming after exact heavy approval; do not row-group read whole modality files.

## Parquet Budget

```json
{
  "expected_modality_count": 12,
  "largest_file_modality": "mGps",
  "largest_file_size_mib": 68.59204006195068,
  "metadata_read_ok_count": 12,
  "one_row_group_modalities": [
    "mACStatus",
    "mActivity",
    "mAmbience",
    "mBle",
    "mGps",
    "mLight",
    "mScreenStatus",
    "mUsageStats",
    "mWifi",
    "wHr",
    "wLight",
    "wPedo"
  ],
  "one_row_group_modality_count": 12,
  "total_compressed_column_bytes": 128758035,
  "total_file_size_bytes": 128798903,
  "total_file_size_mib": 122.8322057723999,
  "total_row_groups": 12,
  "total_rows": 6122179,
  "total_uncompressed_column_bytes": 422974635,
  "total_uncompressed_column_mib": 403.3800458908081
}
```

## Token Budget

```json
{
  "assumed_day_records": 700,
  "day_grid_hourly_slots": 24,
  "estimated_projected_tensor_fp16_mib": 45.80078125,
  "estimated_projected_tensor_fp32_mib": 91.6015625,
  "estimated_total_tokens": 375200,
  "evening_grid_half_hour_slots": 18,
  "float16_bytes": 2,
  "float32_bytes": 4,
  "kaggle_batch_tokens_after_approval": 34304,
  "kaggle_projected_batch_fp32_mib": 8.375,
  "local_smoke_batch_tokens_after_approval": 8576,
  "local_smoke_projected_batch_fp32_mib": 2.09375,
  "modalities": 12,
  "projection_dim": 64,
  "recommended_kaggle_day_batch_after_approval": 64,
  "recommended_local_day_batch_for_smoke_after_approval": 16,
  "recommended_scanner_batch_rows": 65536,
  "sleep_transition_event_slots": 32,
  "source": "DACON/ETRI data page describes about 700 days of public lifelog records.",
  "tokens_per_day_record": 536
}
```

## Runtime Plan

| stage | allowed now | after exact approval | plan | reason |
|---|---:|---:|---|---|
| `metadata_gate` | `True` | `True` | Use parquet footer metadata and file sizes only. | This is the current report-only stage and reads no row payload. |
| `payload_streaming_after_approval` | `False` | `True` | Use pyarrow.dataset or polars scan batches around 65536 rows; avoid whole-file reads. | All modalities have one row group, so row-group reads would load full modality files. |
| `token_cache_after_approval` | `False` | `True` | Write compact day/evening/sleep-transition tokens only after local no-submit smoke proves no CSV/log side effects. | The token tensor budget is small, but cache creation is still execution work. |
| `embedding_fit_after_approval` | `False` | `True` | Prefer Kaggle no-submit execution; local Mac only for tiny smoke batches. | Local memory shows compressor pressure and the goal explicitly prefers careful resource use. |

## Raw Metadata Rows

| modality | file MiB | rows | row groups | compressed MiB | uncompressed MiB | payload rows read |
|---|---:|---:|---:|---:|---:|---:|
| `mACStatus` | `4.328` | `939896` | `1` | `4.325` | `7.168` | `0` |
| `mActivity` | `4.427` | `961062` | `1` | `4.425` | `7.344` | `0` |
| `mAmbience` | `14.155` | `476577` | `1` | `14.152` | `123.619` | `0` |
| `mBle` | `3.878` | `21830` | `1` | `3.875` | `9.492` | `0` |
| `mGps` | `68.592` | `800611` | `1` | `68.588` | `188.183` | `0` |
| `mLight` | `0.646` | `96258` | `1` | `0.643` | `0.893` | `0` |
| `mScreenStatus` | `4.359` | `939653` | `1` | `4.356` | `7.217` | `0` |
| `mUsageStats` | `0.955` | `45197` | `1` | `0.952` | `1.552` | `0` |
| `mWifi` | `3.799` | `76336` | `1` | `3.796` | `22.446` | `0` |
| `wHr` | `9.894` | `382918` | `1` | `9.892` | `22.543` | `0` |
| `wLight` | `3.251` | `633741` | `1` | `3.248` | `5.089` | `0` |
| `wPedo` | `4.548` | `748100` | `1` | `4.542` | `7.835` | `0` |

## Local Resource Snapshot

```json
{
  "logical_cpu": 12,
  "total_mem_bytes": 25769803776,
  "vm_compressor_bytes_estimate": 6175653888,
  "vm_compressor_pages": 376932,
  "vm_free_bytes_estimate": 100450304,
  "vm_free_pages": 6131,
  "vm_page_size": 16384
}
```

## Prerequisites

```json
{
  "external_priority_ready": true,
  "implementation_feasibility_ready": true,
  "input_reports_parse": true,
  "metadata_read_ok": true,
  "payload_rows_read_now_zero": true,
  "raw_payload_rows_read_still_zero": true,
  "raw_schema_contract_ready": true,
  "success_thresholds_ready": true
}
```

## Forbidden Actions Now

```json
{
  "create_predictions": false,
  "create_runner_files": false,
  "create_submission_artifact": false,
  "fit_embedding": false,
  "modify_submission_log": false,
  "pull_kaggle_output": false,
  "push_kaggle": false,
  "read_raw_parquet_payloads": false,
  "run_kaggle": false,
  "submit_dacon": false,
  "train_model": false,
  "write_feature_matrix": false,
  "write_token_cache": false
}
```
