# Post-ID342 raw subject-state feature gate

Created: `2026-06-24 15:15:50 UTC+09:00`

This is report-only. It builds raw daily/window features in memory and creates no feature matrix, prediction, submission artifact, Kaggle action, or DACON submission.

## Decision

- Reviewed candidates: `90`
- Local gate pass count: `0`
- Best candidate: `raw_subject_state_daily_windows_s3_lr_c0_02_w0.01`
- Best target: `S3`
- Best mean/max all-target delta vs ID342: `-0.000011461` / `0.000043369`
- Best projected public proxy: `0.596071791`
- Best gap to Public 0.55: `+0.046071791`
- Bounded scan: `False`
- Kaggle full scan recommended: `False`
- Create feature matrix artifact: `False`
- Create submission artifact: `False`
- DACON submit now: `False`
- Next action: No row passed the strict ID342/Public-0.55 gate; keep this as negative evidence, not a candidate route.

## Feature Frame

- Raw rows read: `6122179`
- Rows after train-key filter: `3896255`
- Mean modality train-key coverage: `0.959074`
- Feature columns: `969`
- Domain composite columns: `9`
- Mean missing rate: `0.092512`

## Top Rows

| rank | pass | candidate | target | weight | wins | mean all delta | max all delta | proxy | gap 0.55 |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `False` | `raw_subject_state_daily_windows_s3_lr_c0_02_w0.01` | `S3` | 0.01 | 3 | -0.000011 | +0.000043 | 0.596072 | +0.046072 |
| 2 | `False` | `raw_subject_state_daily_windows_s3_lr_c0_02_w0.025` | `S3` | 0.025 | 3 | -0.000008 | +0.000148 | 0.596075 | +0.046075 |
| 3 | `False` | `raw_subject_state_daily_windows_s3_lr_c0_02_w0.005` | `S3` | 0.005 | 3 | -0.000007 | +0.000019 | 0.596076 | +0.046076 |
| 4 | `False` | `raw_subject_state_daily_windows_s3_lr_c0_20_w0.01` | `S3` | 0.01 | 3 | -0.000007 | +0.000065 | 0.596076 | +0.046076 |
| 5 | `False` | `raw_subject_state_daily_windows_s3_lr_c0_20_w0.005` | `S3` | 0.005 | 3 | -0.000005 | +0.000029 | 0.596078 | +0.046078 |
| 6 | `False` | `raw_subject_state_daily_windows_s3_lr_c0_08_w0.005` | `S3` | 0.005 | 3 | -0.000003 | +0.000036 | 0.596080 | +0.046080 |
| 7 | `False` | `raw_subject_state_daily_windows_s3_lr_c0_08_w0.01` | `S3` | 0.01 | 3 | -0.000002 | +0.000074 | 0.596081 | +0.046081 |
| 8 | `False` | `raw_subject_state_daily_windows_s4_lr_c0_20_w0.005` | `S4` | 0.005 | 2 | -0.000001 | +0.000021 | 0.596082 | +0.046082 |
| 9 | `False` | `raw_subject_state_daily_windows_s4_lr_c0_02_w0.005` | `S4` | 0.005 | 3 | -0.000001 | +0.000017 | 0.596082 | +0.046082 |
| 10 | `False` | `raw_subject_state_daily_windows_s4_lr_c0_02_w0.01` | `S4` | 0.01 | 2 | -0.000001 | +0.000035 | 0.596083 | +0.046083 |
| 11 | `False` | `raw_subject_state_daily_windows_s4_lr_c0_20_w0.01` | `S4` | 0.01 | 2 | +0.000000 | +0.000044 | 0.596083 | +0.046083 |
| 12 | `False` | `raw_subject_state_daily_windows_s4_lr_c0_08_w0.005` | `S4` | 0.005 | 2 | +0.000001 | +0.000025 | 0.596084 | +0.046084 |
| 13 | `False` | `raw_subject_state_daily_windows_s4_lr_c0_08_w0.01` | `S4` | 0.01 | 2 | +0.000004 | +0.000052 | 0.596087 | +0.046087 |
| 14 | `False` | `raw_subject_state_daily_windows_s3_lr_c0_20_w0.025` | `S3` | 0.025 | 3 | +0.000008 | +0.000210 | 0.596092 | +0.046092 |
| 15 | `False` | `raw_subject_state_daily_windows_s4_lr_c0_02_w0.025` | `S4` | 0.025 | 2 | +0.000010 | +0.000098 | 0.596093 | +0.046093 |
| 16 | `False` | `raw_subject_state_daily_windows_q2_lr_c0_20_w0.005` | `Q2` | 0.005 | 1 | +0.000013 | +0.000030 | 0.596097 | +0.046097 |
| 17 | `False` | `raw_subject_state_daily_windows_q2_lr_c0_08_w0.005` | `Q2` | 0.005 | 1 | +0.000016 | +0.000033 | 0.596099 | +0.046099 |
| 18 | `False` | `raw_subject_state_daily_windows_s3_lr_c0_08_w0.025` | `S3` | 0.025 | 3 | +0.000017 | +0.000197 | 0.596101 | +0.046101 |
| 19 | `False` | `raw_subject_state_daily_windows_s4_lr_c0_20_w0.025` | `S4` | 0.025 | 2 | +0.000019 | +0.000125 | 0.596102 | +0.046102 |
| 20 | `False` | `raw_subject_state_daily_windows_q2_lr_c0_02_w0.005` | `Q2` | 0.005 | 0 | +0.000020 | +0.000031 | 0.596104 | +0.046104 |
| 21 | `False` | `raw_subject_state_daily_windows_s4_lr_c0_08_w0.025` | `S4` | 0.025 | 2 | +0.000025 | +0.000143 | 0.596108 | +0.046108 |
| 22 | `False` | `raw_subject_state_daily_windows_q3_lr_c0_02_w0.005` | `Q3` | 0.005 | 1 | +0.000025 | +0.000041 | 0.596109 | +0.046109 |
| 23 | `False` | `raw_subject_state_daily_windows_q2_lr_c0_20_w0.01` | `Q2` | 0.01 | 1 | +0.000030 | +0.000063 | 0.596113 | +0.046113 |
| 24 | `False` | `raw_subject_state_daily_windows_q2_lr_c0_08_w0.01` | `Q2` | 0.01 | 1 | +0.000034 | +0.000068 | 0.596117 | +0.046117 |
| 25 | `False` | `raw_subject_state_daily_windows_q3_lr_c0_08_w0.005` | `Q3` | 0.005 | 1 | +0.000034 | +0.000053 | 0.596117 | +0.046117 |

## Modality Coverage

- `mACStatus`: rows `939896`, matched `604047`, covered train keys `450`, error `None`
- `mActivity`: rows `961062`, matched `617483`, covered train keys `450`, error `None`
- `mAmbience`: rows `476577`, matched `306124`, covered train keys `450`, error `None`
- `mBle`: rows `21830`, matched `13690`, covered train keys `416`, error `None`
- `mGps`: rows `800611`, matched `513225`, covered train keys `425`, error `None`
- `mLight`: rows `96258`, matched `61819`, covered train keys `450`, error `None`
- `mScreenStatus`: rows `939653`, matched `603493`, covered train keys `450`, error `None`
- `mUsageStats`: rows `45197`, matched `28698`, covered train keys `440`, error `None`
- `mWifi`: rows `76336`, matched `48869`, covered train keys `439`, error `None`
- `wHr`: rows `382918`, matched `239607`, covered train keys `391`, error `None`
- `wLight`: rows `633741`, matched `399932`, covered train keys `414`, error `None`
- `wPedo`: rows `748100`, matched `459268`, covered train keys `404`, error `None`

## Guardrails

- no_feature_matrix_created: `True`
- no_predictions_created: `True`
- no_submission_created: `True`
- no_base_prediction_artifact_created: `True`
- no_notebook_history_artifact_created: `True`
- no_submission_log_row_created: `True`
- no_kaggle_action_made_by_this_script: `True`
- no_dacon_submission_made: `True`
