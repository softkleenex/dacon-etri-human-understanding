# Post-ID342 final-day independent signal sweep

Created: `2026-06-25 10:26:39 UTC+09:00`

This is report-only. It creates no feature matrix artifacts, predictions, submission artifacts, Kaggle runs, or DACON submissions.

## Decision

- Reviewed candidates: `40`
- Local gate passes: `4`
- Review gate passes: `0`
- DACON gate passes: `0`
- Best candidate: `s4_sleep_solar_joint_s4_lr_c0_10_w0.05`
- Best target/family: `S4` / `s4_sleep_solar_joint`
- Best wins: `4`
- Best mean/max delta: `-0.002013473` / `+0.000266583`
- Best projected Public proxy: `0.595795613`
- Best gap to Public 0.55: `+0.045795613`
- Gate pass: `False`
- Kaggle now: `False`
- DACON submit now: `False`
- Next action: Promote nothing unless a future row clears the Public-0.58 review proxy. Current final-day local signals remain too small for artifact creation, Kaggle escalation, or DACON submission.

## Feature Families

| family | targets | features | note |
|---|---|---:|---|
| `s3_phone_gps_temporal_deviation` | `['S3']` | 208 | Per-subject lag1, previous-3-row mean, and current-minus-previous-3-row deviation. |
| `s4_sleep_solar_joint` | `['S4']` | 216 | Processed sleep fragmentation/onset features crossed with solar/calendar priors. |
| `q3s3s4_compact_joint_state` | `['Q3', 'S3', 'S4']` | 216 | Compact union of phone/GPS temporal and sleep/solar state columns. |

## Top Rows

| rank | local | review | dacon | family | target | config | wins | mean delta | max delta | proxy | gap 0.55 | max failed corr |
|---:|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `True` | `False` | `False` | `s4_sleep_solar_joint` | `S4` | `s4_sleep_solar_joint_s4_lr_c0_10_w0.05` | 4 | -0.002013473 | +0.000266583 | 0.595795613 | +0.045795613 | +0.189500 |
| 2 | `True` | `False` | `False` | `s4_sleep_solar_joint` | `S4` | `s4_sleep_solar_joint_s4_lr_c0_10_w0.025` | 4 | -0.001124684 | +0.000047178 | 0.595922583 | +0.045922583 | +0.189500 |
| 3 | `True` | `False` | `False` | `s4_sleep_solar_joint` | `S4` | `s4_sleep_solar_joint_s4_lr_c0_10_w0.01` | 5 | -0.000479024 | -0.000002176 | 0.596014820 | +0.046014820 | +0.189500 |
| 4 | `True` | `False` | `False` | `s4_sleep_solar_joint` | `S4` | `s4_sleep_solar_joint_s4_lr_c0_10_w0.005` | 5 | -0.000244448 | -0.000004631 | 0.596048331 | +0.046048331 | +0.189500 |
| 5 | `False` | `False` | `False` | `s4_sleep_solar_joint` | `S4` | `s4_sleep_solar_joint_s4_lr_c0_03_w0.05` | 3 | -0.001749202 | +0.000484301 | 0.595833366 | +0.045833366 | +0.241333 |
| 6 | `False` | `False` | `False` | `s4_sleep_solar_joint` | `S4` | `s4_sleep_solar_joint_s4_lr_c0_03_w0.025` | 4 | -0.000968181 | +0.000173944 | 0.595944940 | +0.045944940 | +0.241333 |
| 7 | `False` | `False` | `False` | `s4_sleep_solar_joint` | `S4` | `s4_sleep_solar_joint_s4_lr_c0_03_w0.01` | 4 | -0.000410423 | +0.000052845 | 0.596024620 | +0.046024620 | +0.241333 |
| 8 | `False` | `False` | `False` | `s4_sleep_solar_joint` | `S4` | `s4_sleep_solar_joint_s4_lr_c0_03_w0.005` | 4 | -0.000209133 | +0.000023601 | 0.596053376 | +0.046053376 | +0.241333 |
| 9 | `False` | `False` | `False` | `s3_phone_gps_temporal_deviation` | `S3` | `s3_phone_gps_temporal_deviation_s3_lr_c0_10_w0.025` | 3 | -0.000086673 | +0.001370024 | 0.596070870 | +0.046070870 | -0.021653 |
| 10 | `False` | `False` | `False` | `s3_phone_gps_temporal_deviation` | `S3` | `s3_phone_gps_temporal_deviation_s3_lr_c0_10_w0.01` | 3 | -0.000079049 | +0.000530870 | 0.596071959 | +0.046071959 | -0.021653 |
| 11 | `False` | `False` | `False` | `s3_phone_gps_temporal_deviation` | `S3` | `s3_phone_gps_temporal_deviation_s3_lr_c0_10_w0.005` | 3 | -0.000047257 | +0.000262553 | 0.596076501 | +0.046076501 | -0.021653 |
| 12 | `False` | `False` | `False` | `q3s3s4_compact_joint_state` | `S3` | `q3s3s4_compact_joint_state_s3_lr_c0_03_w0.01` | 3 | -0.000025748 | +0.000335489 | 0.596079574 | +0.046079574 | +0.054979 |
| 13 | `False` | `False` | `False` | `q3s3s4_compact_joint_state` | `S3` | `q3s3s4_compact_joint_state_s3_lr_c0_03_w0.005` | 3 | -0.000020620 | +0.000164255 | 0.596080306 | +0.046080306 | +0.054979 |
| 14 | `False` | `False` | `False` | `q3s3s4_compact_joint_state` | `S4` | `q3s3s4_compact_joint_state_s4_lr_c0_10_w0.005` | 3 | +0.000009456 | +0.000118385 | 0.596084603 | +0.046084603 | +0.090062 |
| 15 | `False` | `False` | `False` | `q3s3s4_compact_joint_state` | `S3` | `q3s3s4_compact_joint_state_s3_lr_c0_10_w0.005` | 3 | +0.000016328 | +0.000212639 | 0.596085585 | +0.046085585 | +0.037235 |
| 16 | `False` | `False` | `False` | `s3_phone_gps_temporal_deviation` | `S3` | `s3_phone_gps_temporal_deviation_s3_lr_c0_03_w0.005` | 2 | +0.000026194 | +0.000268449 | 0.596086994 | +0.046086994 | -0.024685 |
| 17 | `False` | `False` | `False` | `q3s3s4_compact_joint_state` | `S4` | `q3s3s4_compact_joint_state_s4_lr_c0_10_w0.01` | 3 | +0.000031277 | +0.000245162 | 0.596087720 | +0.046087720 | +0.090062 |
| 18 | `False` | `False` | `False` | `q3s3s4_compact_joint_state` | `S3` | `q3s3s4_compact_joint_state_s3_lr_c0_03_w0.025` | 3 | +0.000042997 | +0.000890407 | 0.596089394 | +0.046089394 | +0.054979 |
| 19 | `False` | `False` | `False` | `q3s3s4_compact_joint_state` | `S3` | `q3s3s4_compact_joint_state_s3_lr_c0_10_w0.01` | 3 | +0.000050071 | +0.000435109 | 0.596090405 | +0.046090405 | +0.037235 |
| 20 | `False` | `False` | `False` | `q3s3s4_compact_joint_state` | `S4` | `q3s3s4_compact_joint_state_s4_lr_c0_03_w0.005` | 2 | +0.000054258 | +0.000154524 | 0.596091003 | +0.046091003 | +0.145166 |
| 21 | `False` | `False` | `False` | `s3_phone_gps_temporal_deviation` | `S3` | `s3_phone_gps_temporal_deviation_s3_lr_c0_03_w0.01` | 2 | +0.000065475 | +0.000541109 | 0.596092605 | +0.046092605 | -0.024685 |
| 22 | `False` | `False` | `False` | `q3s3s4_compact_joint_state` | `S4` | `q3s3s4_compact_joint_state_s4_lr_c0_03_w0.01` | 2 | +0.000116314 | +0.000314762 | 0.596099868 | +0.046099868 | +0.145166 |
| 23 | `False` | `False` | `False` | `q3s3s4_compact_joint_state` | `S4` | `q3s3s4_compact_joint_state_s4_lr_c0_10_w0.025` | 3 | +0.000169104 | +0.000675146 | 0.596107410 | +0.046107410 | +0.090062 |
| 24 | `False` | `False` | `False` | `s3_phone_gps_temporal_deviation` | `S3` | `s3_phone_gps_temporal_deviation_s3_lr_c0_10_w0.05` | 2 | +0.000169380 | +0.002880587 | 0.596107449 | +0.046107449 | -0.021653 |
| 25 | `False` | `False` | `False` | `q3s3s4_compact_joint_state` | `S3` | `q3s3s4_compact_joint_state_s3_lr_c0_10_w0.025` | 2 | +0.000246336 | +0.001160049 | 0.596118443 | +0.046118443 | +0.037235 |
