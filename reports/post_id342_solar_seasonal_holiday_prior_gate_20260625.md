# Post-ID342 solar/seasonal/holiday prior gate

Created: `2026-06-25 06:39:38 UTC+09:00`

This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.

## Feature Families

| family | features | mean missing rate | sample |
|---|---:|---:|---|
| `solar_photoperiod` | 13 | 0.000000 | `solar_daylight_hours`, `solar_night_hours`, `solar_declination_sin`, `solar_declination_cos`, `solar_daylight_change_1d`, `solar_sunrise_minutes` |
| `korean_holiday_calendar` | 11 | 0.000000 | `weekday`, `is_weekend`, `weekday_sin`, `weekday_cos`, `is_korean_public_holiday`, `days_to_nearest_korean_holiday_abs` |
| `solar_holiday_interaction` | 7 | 0.000000 | `solar_daylight_x_weekend`, `solar_daylight_x_holiday_near7`, `solar_night_x_holiday_near7`, `autumn_x_weekend`, `autumn_x_holiday_near7`, `subject_solar_daylight_delta` |
| `solar_holiday_all` | 28 | 0.000000 | `autumn_x_holiday_near7`, `autumn_x_weekend`, `days_to_nearest_korean_holiday_abs`, `days_to_nearest_korean_holiday_signed`, `is_korean_public_holiday`, `is_weekend` |

## Top Rows

| family | target | config | wins | mean delta | max delta | eligible |
|---|---|---|---:|---:|---:|---|
| `solar_holiday_interaction` | `S4` | `solar_holiday_interaction_S4_lr_c0_08_w0.05` | 3 | -0.002238508 | 0.001238751 | `False` |
| `solar_holiday_interaction` | `S4` | `solar_holiday_interaction_S4_lr_c0_02_w0.05` | 3 | -0.001902758 | 0.001497327 | `False` |
| `solar_holiday_interaction` | `S4` | `solar_holiday_interaction_S4_lr_c0_08_w0.02` | 3 | -0.000999811 | 0.000429214 | `False` |
| `solar_holiday_interaction` | `S4` | `solar_holiday_interaction_S4_lr_c0_02_w0.02` | 3 | -0.000854928 | 0.000537954 | `False` |
| `korean_holiday_calendar` | `S4` | `korean_holiday_calendar_S4_hgb_l5_w0.05` | 3 | -0.000836628 | 0.001371497 | `False` |
| `solar_holiday_all` | `S4` | `solar_holiday_all_S4_lr_c0_02_w0.05` | 3 | -0.000769288 | 0.001462371 | `False` |
| `solar_photoperiod` | `S4` | `solar_photoperiod_S4_hgb_l5_w0.05` | 3 | -0.000601448 | 0.001717095 | `False` |
| `solar_holiday_all` | `S4` | `solar_holiday_all_S4_hgb_l5_w0.05` | 3 | -0.000590944 | 0.001610774 | `False` |
| `solar_holiday_interaction` | `S4` | `solar_holiday_interaction_S4_lr_c0_08_w0.01` | 3 | -0.000518019 | 0.000203222 | `False` |
| `solar_holiday_interaction` | `S4` | `solar_holiday_interaction_S4_lr_c0_02_w0.01` | 3 | -0.000443742 | 0.000258503 | `False` |
| `korean_holiday_calendar` | `S4` | `korean_holiday_calendar_S4_hgb_l5_w0.02` | 3 | -0.000413978 | 0.000477291 | `False` |
| `solar_holiday_all` | `S4` | `solar_holiday_all_S4_lr_c0_02_w0.02` | 3 | -0.000383523 | 0.000521329 | `False` |
| `solar_photoperiod` | `S4` | `solar_photoperiod_S4_hgb_l5_w0.02` | 3 | -0.000316465 | 0.000612670 | `False` |
| `solar_holiday_all` | `S4` | `solar_holiday_all_S4_hgb_l5_w0.02` | 3 | -0.000312584 | 0.000571826 | `False` |
| `solar_holiday_interaction` | `S4` | `solar_holiday_interaction_S4_lr_c0_08_w0.005` | 3 | -0.000263611 | 0.000098730 | `False` |
| `solar_holiday_all` | `S4` | `solar_holiday_all_S4_lr_c0_08_w0.05` | 3 | -0.000233514 | 0.001459288 | `False` |

## Decision

- Eligible rows: `0`
- Selected targets: `[]`
- Projected Public proxy: `0.596083252`
- Gap to Public 0.55: `0.046083252`
- Gate pass: `False`
- Kaggle now: `False`
- DACON submit now: `False`
- Next action: Treat solar/seasonal/holiday priors as report-only domain evidence. Do not create artifacts, Kaggle runs, or DACON submissions unless a future independent gate produces a Public-0.55 plausible row.
