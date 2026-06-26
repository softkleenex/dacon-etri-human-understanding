# Post-ID342 self-supervised subject-state no-submit smoke

Created: `2026-06-24 13:48:09 UTC+09:00`

This runner reads bounded raw payload batches and performs a tiny label-free masked reconstruction smoke. It creates no submission CSV, base prediction artifact, submission-log row, or DACON submission.

## Decision

- Mode: `local_smoke`
- Smoke pass: `True`
- Payload rows read: `24576`
- Label-free: `True`
- Persistent model artifact created: `False`
- Create submission artifact: `False`
- Modify submission log: `False`
- DACON submit: `False`
- Next action: Local/remote no-submit smoke passed; review reports before any larger embedding run.

## Token Summary

```json
{
  "feature_dim": 3,
  "subject_date_hour_modality_tokens_seen": 2048,
  "token_count": 2048
}
```

## Masked Reconstruction Smoke

```json
{
  "epochs": 1,
  "feature_dim": 3,
  "final_masked_mse": 0.9779338240623474,
  "losses": [
    0.9779338240623474
  ],
  "persistent_model_artifact_created": false,
  "token_count": 2048,
  "trained": true
}
```

## Modality Rows

| modality | rows read | batches | subjects | dates | payload max | error |
|---|---:|---:|---:|---:|---:|---|
| `mACStatus` | `2048` | `2` | `1` | `2` | `1` | `None` |
| `mActivity` | `2048` | `2` | `1` | `2` | `1` | `None` |
| `mAmbience` | `2048` | `2` | `1` | `4` | `10` | `None` |
| `mBle` | `2048` | `2` | `1` | `42` | `172` | `None` |
| `mGps` | `2048` | `2` | `1` | `2` | `19` | `None` |
| `mLight` | `2048` | `2` | `1` | `15` | `1` | `None` |
| `mScreenStatus` | `2048` | `2` | `1` | `2` | `1` | `None` |
| `mUsageStats` | `2048` | `2` | `1` | `25` | `15` | `None` |
| `mWifi` | `2048` | `2` | `1` | `19` | `128` | `None` |
| `wHr` | `2048` | `2` | `1` | `5` | `60` | `None` |
| `wLight` | `2048` | `2` | `1` | `4` | `1` | `None` |
| `wPedo` | `2048` | `2` | `1` | `3` | `1` | `None` |

## Output Snapshot Before Write

```json
{
  "file_count": 2,
  "files": [
    "/Volumes/samsd/workspace_v2/dacon/dacon-etri-human-understanding/reports/post_id342_self_supervised_subject_state_local_smoke_20260624.json",
    "/Volumes/samsd/workspace_v2/dacon/dacon-etri-human-understanding/reports/post_id342_self_supervised_subject_state_local_smoke_20260624.md"
  ],
  "output_dir": "/Volumes/samsd/workspace_v2/dacon/dacon-etri-human-understanding/reports",
  "suspicious_submission_like_files": []
}
```
