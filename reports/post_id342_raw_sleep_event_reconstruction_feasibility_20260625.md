# Post-ID342 raw sleep event reconstruction feasibility

Created: `2026-06-25 02:02:04 UTC+09:00`

This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.

## Decision

- Raw files scanned: `12`
- Direct sleep event columns: `0`
- Direct event route available: `False`
- Proxy modalities available: `7`
- Processed S4/S3 eligible rows: `0`
- Run Kaggle now: `False`
- DACON submit now: `False`
- Next action: The public raw files do not expose direct sleep-stage, SOL, or WASO event fields. Existing processed HR/light/screen proxies have also failed the S4/S3 gate. Do not run another processed-column overlay; either design a genuinely sequence-level HR/light event model with a new local gate, or prioritize validation/public-gap redesign.

## Raw Schema

| file | rows | direct event columns | proxy columns | role |
|---|---:|---:|---:|---|
| `ch2025_mACStatus.parquet` | `939896` | `0` | `2` | charging state; proxy for bedtime context |
| `ch2025_mActivity.parquet` | `961062` | `0` | `2` | phone activity state; proxy for pre-sleep movement |
| `ch2025_mAmbience.parquet` | `476577` | `0` | `1` | other raw modality |
| `ch2025_mBle.parquet` | `21830` | `0` | `1` | other raw modality |
| `ch2025_mGps.parquet` | `800611` | `0` | `1` | other raw modality |
| `ch2025_mLight.parquet` | `96258` | `0` | `2` | phone light sequence; proxy for environment exposure |
| `ch2025_mScreenStatus.parquet` | `939653` | `0` | `2` | phone screen on/off; proxy for pre-sleep usage |
| `ch2025_mUsageStats.parquet` | `45197` | `0` | `1` | other raw modality |
| `ch2025_mWifi.parquet` | `76336` | `0` | `1` | other raw modality |
| `ch2025_wHr.parquet` | `382918` | `0` | `2` | wearable heart rate sequence; proxy for sleep depth/awake state only |
| `ch2025_wLight.parquet` | `633741` | `0` | `2` | wearable light sequence; proxy for darkness/light exposure |
| `ch2025_wPedo.parquet` | `748100` | `0` | `5` | wearable steps; proxy for rest/activity |
