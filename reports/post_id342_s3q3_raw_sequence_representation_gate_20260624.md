# Post-ID342 S3/Q3 raw sequence representation gate

Created: `2026-06-24 20:44:04 UTC+09:00`

This is report-only. It keeps random-convolution sequence features in memory and creates no feature matrix, prediction, submission artifact, Kaggle action, or DACON submission.

## Decision

- Reviewed candidates: `240`
- Local gate pass count: `0`
- Passing targets: `[]`
- Final gate pass: `False`
- Best candidate: `raw_sequence_representation_q3_usage_routine_q3_lr_c0_20_seed45602_w0.005`
- Best target: `Q3`
- Best mean/max all-target delta vs ID342: `-0.000000019` / `0.000033135`
- Best projected public proxy: `0.596083233`
- Best gap to Public 0.55: `+0.046083233`
- Artifact review allowed now: `False`
- Create feature matrix artifact: `False`
- Create submission artifact: `False`
- DACON submit now: `False`
- Next action: No strict S3/Q3 sequence-representation row passed; keep this as negative evidence, not a candidate.

## Sequence Frame

- Sequence rows: `809`
- Train/test rows: `450` / `250`
- Raw sequence feature columns: `1496`
- Value/total channels: `34` / `68`
- Bins: `44`
- Kernels: `256`
- Seeds: `[45502, 45602]`

## Families

| family | skipped | channels | bins | read |
|---|---:|---:|---:|---|
| all_channels | `False` | 68 | 44 | All raw half-hour channels and missingness channels. |
| q3_usage_routine | `False` | 40 | 32 | Q3 stress/fatigue is screened through app use, phone state, mobility, and context routine. |
| s3_sleep_boundary | `False` | 32 | 28 | S3 sleep-onset-latency is screened through evening-to-morning light, motion, HR, and phone state. |
| usage_evening | `False` | 38 | 24 | Evening usage, location/proximity, and ambience may capture recovery load and routine disruption. |
| physio_light_sleep | `False` | 30 | 28 | Physiology, light, movement, and charging/screen context across the pre-sleep and wake boundary. |

## Top Rows

| rank | pass | candidate | target | wins | mean all delta | max all delta | proxy | gap 0.55 |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `False` | `raw_sequence_representation_q3_usage_routine_q3_lr_c0_20_seed45602_w0.005` | `Q3` | 2 | -0.000000 | +0.000033 | 0.596083 | +0.046083 |
| 2 | `False` | `raw_sequence_representation_q3_usage_routine_q3_lr_c0_20_seed45602_w0.01` | `Q3` | 2 | +0.000002 | +0.000068 | 0.596085 | +0.046085 |
| 3 | `False` | `raw_sequence_representation_usage_evening_q3_lr_c0_20_seed45602_w0.005` | `Q3` | 2 | +0.000006 | +0.000040 | 0.596089 | +0.046089 |
| 4 | `False` | `raw_sequence_representation_q3_usage_routine_q3_lr_c0_08_seed45602_w0.005` | `Q3` | 2 | +0.000007 | +0.000033 | 0.596091 | +0.046091 |
| 5 | `False` | `raw_sequence_representation_physio_light_sleep_s3_lr_c0_20_seed45602_w0.005` | `S3` | 3 | +0.000009 | +0.000031 | 0.596093 | +0.046093 |
| 6 | `False` | `raw_sequence_representation_usage_evening_q3_lr_c0_08_seed45602_w0.005` | `Q3` | 2 | +0.000010 | +0.000049 | 0.596094 | +0.046094 |
| 7 | `False` | `raw_sequence_representation_q3_usage_routine_q3_lr_c0_08_seed45502_w0.005` | `Q3` | 1 | +0.000011 | +0.000047 | 0.596094 | +0.046094 |
| 8 | `False` | `raw_sequence_representation_q3_usage_routine_q3_lr_c0_20_seed45502_w0.005` | `Q3` | 1 | +0.000011 | +0.000045 | 0.596094 | +0.046094 |
| 9 | `False` | `raw_sequence_representation_usage_evening_q3_lr_c0_20_seed45602_w0.01` | `Q3` | 2 | +0.000013 | +0.000082 | 0.596096 | +0.046096 |
| 10 | `False` | `raw_sequence_representation_physio_light_sleep_s3_lr_c0_08_seed45602_w0.005` | `S3` | 1 | +0.000016 | +0.000037 | 0.596099 | +0.046099 |
| 11 | `False` | `raw_sequence_representation_q3_usage_routine_q3_lr_c0_08_seed45602_w0.01` | `Q3` | 2 | +0.000017 | +0.000068 | 0.596100 | +0.046100 |
| 12 | `False` | `raw_sequence_representation_q3_usage_routine_q3_lr_c0_02_seed45502_w0.005` | `Q3` | 1 | +0.000017 | +0.000055 | 0.596100 | +0.046100 |
| 13 | `False` | `raw_sequence_representation_usage_evening_q3_lr_c0_02_seed45602_w0.005` | `Q3` | 1 | +0.000020 | +0.000063 | 0.596103 | +0.046103 |
| 14 | `False` | `raw_sequence_representation_q3_usage_routine_q3_lr_c0_02_seed45602_w0.005` | `Q3` | 1 | +0.000020 | +0.000046 | 0.596104 | +0.046104 |
| 15 | `False` | `raw_sequence_representation_q3_usage_routine_q3_lr_c0_20_seed45602_w0.025` | `Q3` | 2 | +0.000021 | +0.000183 | 0.596104 | +0.046104 |
| 16 | `False` | `raw_sequence_representation_physio_light_sleep_s3_lr_c0_20_seed45602_w0.01` | `S3` | 1 | +0.000021 | +0.000065 | 0.596104 | +0.046104 |
| 17 | `False` | `raw_sequence_representation_usage_evening_q3_lr_c0_08_seed45602_w0.01` | `Q3` | 2 | +0.000022 | +0.000099 | 0.596105 | +0.046105 |
| 18 | `False` | `raw_sequence_representation_s3_sleep_boundary_q3_lr_c0_20_seed45602_w0.005` | `Q3` | 1 | +0.000022 | +0.000063 | 0.596106 | +0.046106 |
| 19 | `False` | `raw_sequence_representation_q3_usage_routine_q3_lr_c0_08_seed45502_w0.01` | `Q3` | 1 | +0.000023 | +0.000097 | 0.596106 | +0.046106 |
| 20 | `False` | `raw_sequence_representation_q3_usage_routine_q3_lr_c0_20_seed45502_w0.01` | `Q3` | 1 | +0.000025 | +0.000092 | 0.596108 | +0.046108 |
| 21 | `False` | `raw_sequence_representation_physio_light_sleep_s3_lr_c0_02_seed45602_w0.005` | `S3` | 0 | +0.000027 | +0.000045 | 0.596110 | +0.046110 |
| 22 | `False` | `raw_sequence_representation_usage_evening_s3_lr_c0_02_seed45602_w0.005` | `S3` | 0 | +0.000029 | +0.000068 | 0.596112 | +0.046112 |
| 23 | `False` | `raw_sequence_representation_s3_sleep_boundary_q3_lr_c0_08_seed45602_w0.005` | `Q3` | 1 | +0.000029 | +0.000065 | 0.596112 | +0.046112 |
| 24 | `False` | `raw_sequence_representation_usage_evening_s3_lr_c0_08_seed45602_w0.005` | `S3` | 0 | +0.000029 | +0.000071 | 0.596112 | +0.046112 |
| 25 | `False` | `raw_sequence_representation_usage_evening_s3_lr_c0_20_seed45602_w0.005` | `S3` | 1 | +0.000030 | +0.000076 | 0.596114 | +0.046114 |

## Guardrails

- no_feature_matrix_created: `True`
- no_predictions_created: `True`
- no_submission_created: `True`
- no_base_prediction_artifact_created: `True`
- no_notebook_history_artifact_created: `True`
- no_submission_log_row_created: `True`
- no_kaggle_action_made_by_this_script: `True`
- no_dacon_submission_made: `True`
