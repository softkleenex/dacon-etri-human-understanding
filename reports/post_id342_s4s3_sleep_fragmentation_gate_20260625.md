# Post-ID342 S4/S3 sleep fragmentation gate

Created: `2026-06-25 02:01:57 UTC+09:00`

This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.

## Feature Families

| family | features | mean missing rate | sample |
|---|---:|---:|---|
| `s4_waso_fragmentation_core` | 12 | 0.215741 | `wHr_sleep_duration_awake`, `wHr_sleep_efficiency_proxy`, `wHr_sleep_fragmentation_index`, `wHr_deep_sleep_ratio`, `wHr_slp_hr_awake_ratio`, `slp_hr_awake_ratio` |
| `s3_onset_context` | 48 | 0.055185 | `ch2025_mScreenStatus_m_screen_use_sum`, `ch2025_mScreenStatus_m_screen_use_mean`, `ch2025_mScreenStatus_is_unlock_sum`, `ch2025_mScreenStatus_bedtime_screen_use_sum`, `ch2025_mScreenStatus_screen_first_hour_min`, `ch2025_mScreenStatus_screen_last_hour_max` |
| `sleep_hr_awake_light` | 29 | 0.406513 | `ch2025_wHr_hr_rmssd_mean`, `ch2025_wHr_hr_rmssd_max`, `wHr_sleep_hr_mean`, `wHr_sleep_hr_std`, `wHr_slp_hr_deep_ratio`, `wHr_slp_hr_awake_ratio` |
| `s4s3_sleep_union` | 80 | 0.191944 | `wHr_sleep_duration_awake`, `wHr_sleep_efficiency_proxy`, `wHr_sleep_fragmentation_index`, `wHr_deep_sleep_ratio`, `wHr_slp_hr_awake_ratio`, `slp_hr_awake_ratio` |

## Top Rows

| family | target | config | wins | mean delta | max delta | eligible |
|---|---|---|---:|---:|---:|---|
| `sleep_hr_awake_light` | `S4` | `sleep_hr_awake_light_s4_hgb_l7_w0.05` | 3 | -0.000272711 | 0.002148118 | `False` |
| `sleep_hr_awake_light` | `S4` | `sleep_hr_awake_light_s4_hgb_l7_w0.02` | 3 | -0.000191166 | 0.000776679 | `False` |
| `sleep_hr_awake_light` | `S4` | `sleep_hr_awake_light_s4_hgb_l7_w0.08` | 3 | -0.000124721 | 0.003752956 | `False` |
| `sleep_hr_awake_light` | `S4` | `sleep_hr_awake_light_s4_hgb_l7_w0.01` | 3 | -0.000109791 | 0.000374123 | `False` |
| `sleep_hr_awake_light` | `S4` | `sleep_hr_awake_light_s4_hgb_l7_w0.005` | 3 | -0.000058501 | 0.000184204 | `False` |
| `sleep_hr_awake_light` | `S4` | `sleep_hr_awake_light_s4_lr_c0_10_w0.005` | 3 | -0.000000151 | 0.000212861 | `False` |
| `s3_onset_context` | `S4` | `s3_onset_context_s4_hgb_l7_w0.005` | 3 | 0.000002174 | 0.000216028 | `False` |
| `sleep_hr_awake_light` | `S4` | `sleep_hr_awake_light_s4_lr_c0_03_w0.005` | 2 | 0.000003582 | 0.000195908 | `False` |
| `sleep_hr_awake_light` | `S4` | `sleep_hr_awake_light_s4_lr_c0_10_w0.01` | 3 | 0.000006737 | 0.000430600 | `False` |
| `s3_onset_context` | `S4` | `s3_onset_context_s4_hgb_l7_w0.01` | 3 | 0.000008282 | 0.000434519 | `False` |
| `sleep_hr_awake_light` | `S4` | `sleep_hr_awake_light_s4_lr_c0_03_w0.01` | 2 | 0.000013693 | 0.000396433 | `False` |
| `s4s3_sleep_union` | `S4` | `s4s3_sleep_union_s4_hgb_l7_w0.005` | 2 | 0.000016282 | 0.000256845 | `False` |
| `s4_waso_fragmentation_core` | `S4` | `s4_waso_fragmentation_core_s4_lr_c0_03_w0.005` | 2 | 0.000019162 | 0.000195185 | `False` |
| `s4_waso_fragmentation_core` | `S4` | `s4_waso_fragmentation_core_s4_lr_c0_10_w0.005` | 2 | 0.000020137 | 0.000198231 | `False` |
| `s3_onset_context` | `S4` | `s3_onset_context_s4_hgb_l7_w0.02` | 3 | 0.000032176 | 0.000878865 | `False` |
| `s4s3_sleep_union` | `S4` | `s4s3_sleep_union_s4_hgb_l7_w0.01` | 2 | 0.000036616 | 0.000516265 | `False` |

## Target Summary

| target | domain | eligible | best wins | best mean delta | best max delta |
|---|---|---:|---:|---:|---:|
| `S4` | objective WASO / wakefulness after sleep onset guideline | 0 | 3 | -0.000272711 | 0.002148118 |
| `S3` | objective sleep onset latency guideline | 0 | 2 | 0.000075639 | 0.000378615 |

## Decision

- Eligible rows: `0`
- Selected targets: `[]`
- Projected Public proxy: `0.596083252`
- Gap to Public 0.55: `0.046083252`
- Kaggle now: `False`
- DACON submit now: `False`
- Next action: If S4/S3 remain unselected, processed WASO/SOL proxy features are closed and the next route needs raw sleep-stage event reconstruction or validation-gap redesign, not another processed-column overlay.
