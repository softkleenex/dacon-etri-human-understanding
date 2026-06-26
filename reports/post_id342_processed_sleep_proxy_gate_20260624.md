# Post-ID342 processed sleep-proxy gate

Created: `2026-06-24 08:16:58 UTC+09:00`

This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.

## Feature Families

| family | features | mean missing rate | sample |
|---|---:|---:|---|
| `sleep_proxy` | 48 | 0.317500 | `ch2025_mScreenStatus_bedtime_screen_use_sum`, `act_presleep_active`, `screen_presleep_on`, `screen_presleep_ratio`, `ac_presleep_charging`, `wHr_sleep_duration_deep` |
| `sleep_proxy_no_ids` | 47 | 0.324255 | `act_presleep_active`, `screen_presleep_on`, `screen_presleep_ratio`, `ac_presleep_charging`, `wHr_sleep_duration_deep`, `wHr_sleep_duration_light` |
| `core_sleep` | 42 | 0.353968 | `wHr_sleep_duration_deep`, `wHr_sleep_duration_light`, `wHr_sleep_duration_awake`, `wHr_sleep_efficiency_proxy`, `wHr_sleep_fragmentation_index`, `wHr_sleep_hr_mean` |

## Top Rows

| family | target | config | wins | mean delta | max delta | eligible |
|---|---|---|---:|---:|---:|---|
| `sleep_proxy` | `S3` | `sleep_proxy_S3_hgb_l7_w0.005` | 2 | 0.000019110 | 0.000173356 | `False` |
| `sleep_proxy` | `S3` | `sleep_proxy_S3_hgb_l7_w0.01` | 2 | 0.000047211 | 0.000358659 | `False` |
| `sleep_proxy_no_ids` | `S3` | `sleep_proxy_no_ids_S3_hgb_l7_w0.005` | 2 | 0.000053473 | 0.000200904 | `False` |
| `core_sleep` | `S3` | `core_sleep_S3_hgb_l7_w0.005` | 2 | 0.000083803 | 0.000237993 | `False` |
| `sleep_proxy_no_ids` | `S3` | `sleep_proxy_no_ids_S3_hgb_l7_w0.01` | 2 | 0.000115896 | 0.000412640 | `False` |
| `sleep_proxy` | `S3` | `sleep_proxy_S3_hgb_l7_w0.02` | 2 | 0.000129807 | 0.000764172 | `False` |
| `core_sleep` | `S3` | `core_sleep_S3_lr_c0_10_w0.005` | 0 | 0.000146749 | 0.000260141 | `False` |
| `core_sleep` | `S3` | `core_sleep_S3_lr_c0_03_w0.005` | 0 | 0.000175818 | 0.000346566 | `False` |
| `core_sleep` | `S3` | `core_sleep_S3_hgb_l7_w0.01` | 2 | 0.000176735 | 0.000481965 | `False` |
| `sleep_proxy_no_ids` | `Q3` | `sleep_proxy_no_ids_Q3_hgb_l7_w0.005` | 1 | 0.000180767 | 0.000324742 | `False` |
| `sleep_proxy` | `Q3` | `sleep_proxy_Q3_lr_c0_10_w0.005` | 1 | 0.000199954 | 0.000342044 | `False` |
| `sleep_proxy_no_ids` | `Q3` | `sleep_proxy_no_ids_Q3_lr_c0_03_w0.005` | 1 | 0.000201802 | 0.000367382 | `False` |

## Decision

- Eligible rows: `0`
- Selected targets: `[]`
- Projected Public proxy: `0.596083252`
- Gap to Public 0.55: `0.046083252`
- Kaggle now: `False`
- DACON submit now: `False`
- Next action: Processed sleep-proxy columns are present and domain-relevant, but do not open Q3/S3 against ID342 under the stable primary lenses. Keep as negative evidence; do not create artifacts, Kaggle runs, or DACON submissions from this lane.
