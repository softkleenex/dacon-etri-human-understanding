# Post-ID342 processed sleep temporal-deviation gate

Created: `2026-06-24 08:22:08 UTC+09:00`

This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.

## Feature Families

| family | features | mean missing rate | sample |
|---|---:|---:|---|
| `sleep_temporal_past_only` | 144 | 0.184460 | `ch2025_mScreenStatus_bedtime_screen_use_sum__lag1`, `ch2025_mScreenStatus_bedtime_screen_use_sum__roll3_mean_past`, `ch2025_mScreenStatus_bedtime_screen_use_sum__roll3_std_past`, `act_presleep_active__lag1`, `act_presleep_active__roll3_mean_past` |
| `sleep_temporal_deviation` | 96 | 0.431968 | `ch2025_mScreenStatus_bedtime_screen_use_sum__delta_lag1`, `ch2025_mScreenStatus_bedtime_screen_use_sum__dev_roll3_mean`, `act_presleep_active__delta_lag1`, `act_presleep_active__dev_roll3_mean`, `screen_presleep_on__delta_lag1` |
| `sleep_temporal_all` | 240 | 0.283463 | `ch2025_mScreenStatus_bedtime_screen_use_sum__lag1`, `ch2025_mScreenStatus_bedtime_screen_use_sum__roll3_mean_past`, `ch2025_mScreenStatus_bedtime_screen_use_sum__roll3_std_past`, `act_presleep_active__lag1`, `act_presleep_active__roll3_mean_past` |

## Top Rows

| family | target | config | wins | mean delta | max delta | eligible |
|---|---|---|---:|---:|---:|---|
| `sleep_temporal_deviation` | `Q3` | `sleep_temporal_deviation_Q3_lr_c0_10_w0.005` | 2 | 0.000111587 | 0.000361616 | `False` |
| `sleep_temporal_deviation` | `Q3` | `sleep_temporal_deviation_Q3_lr_c0_03_w0.005` | 2 | 0.000137908 | 0.000384650 | `False` |
| `sleep_temporal_past_only` | `Q3` | `sleep_temporal_past_only_Q3_hgb_l7_w0.005` | 1 | 0.000147279 | 0.000347905 | `False` |
| `sleep_temporal_all` | `Q3` | `sleep_temporal_all_Q3_hgb_l7_w0.005` | 0 | 0.000176774 | 0.000375756 | `False` |
| `sleep_temporal_past_only` | `S3` | `sleep_temporal_past_only_S3_lr_c0_10_w0.005` | 0 | 0.000184003 | 0.000345841 | `False` |
| `sleep_temporal_deviation` | `S3` | `sleep_temporal_deviation_S3_hgb_l7_w0.005` | 1 | 0.000187780 | 0.000415913 | `False` |
| `sleep_temporal_past_only` | `S3` | `sleep_temporal_past_only_S3_lr_c0_03_w0.005` | 0 | 0.000209934 | 0.000323800 | `False` |
| `sleep_temporal_deviation` | `Q3` | `sleep_temporal_deviation_Q3_hgb_l7_w0.005` | 0 | 0.000214560 | 0.000460664 | `False` |
| `sleep_temporal_past_only` | `S3` | `sleep_temporal_past_only_S3_hgb_l7_w0.005` | 0 | 0.000221372 | 0.000318377 | `False` |
| `sleep_temporal_deviation` | `Q3` | `sleep_temporal_deviation_Q3_lr_c0_10_w0.01` | 2 | 0.000231905 | 0.000729865 | `False` |
| `sleep_temporal_all` | `S3` | `sleep_temporal_all_S3_hgb_l7_w0.005` | 0 | 0.000269651 | 0.000347100 | `False` |
| `sleep_temporal_deviation` | `Q3` | `sleep_temporal_deviation_Q3_lr_c0_03_w0.01` | 2 | 0.000283261 | 0.000775648 | `False` |

## Decision

- Eligible rows: `0`
- Selected targets: `[]`
- Projected Public proxy: `0.596083252`
- Gap to Public 0.55: `0.046083252`
- Kaggle now: `False`
- DACON submit now: `False`
- Next action: Past-only and deviation-from-personal-sleep-baseline features do not open Q3/S3 against ID342. Keep this as negative domain evidence and do not escalate to Kaggle, artifacts, or DACON.
