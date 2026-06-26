# Post-ID342 AirKorea 2024 context gate

Created: `2026-06-25 17:48:58 UTC+09:00`

This is report-only. It creates no feature matrix artifacts, predictions, submission artifacts, Kaggle runs, or DACON submissions.

## Decision

- AirKorea cache available: `True`
- Reviewed candidates: `216`
- Local gate passes: `0`
- Review gate passes: `0`
- DACON gate passes: `0`
- Best candidate: `airkorea_selected_station_s4_hgb_l7_w0.025`
- Best target/family: `S4` / `airkorea_selected_station`
- Best projected Public proxy: `0.596056065`
- Gate pass: `False`
- Kaggle now: `False`
- DACON submit now: `False`
- Next action: Promote nothing unless a row clears the Public-0.58 review proxy and a separate artifact contract. This gate is external-source evidence only.

## AirKorea Cache

```json
{
  "airkorea_cache": "data/external/weather/airkorea_2024/airkorea_2024_daejeon_hourly.csv",
  "cache_rows": 114192,
  "columns": [
    "region",
    "network",
    "station_code",
    "station_name",
    "datetime",
    "so2",
    "co",
    "o3",
    "no2",
    "pm10",
    "pm25",
    "address",
    "source_file"
  ],
  "competition_window_rows": 54600,
  "competition_window_station_count": 13,
  "competition_window_unique_hours": 4200,
  "pollutants": [
    "so2",
    "co",
    "o3",
    "no2",
    "pm10",
    "pm25"
  ],
  "secret_values_stored": false,
  "source_note": "Historical 2024 AirKorea hourly Daejeon rows extracted from the user-provided archive.",
  "station_count": 13,
  "station_names": [
    "관평동",
    "구성동",
    "노은동",
    "대성동",
    "대흥동1",
    "둔산동",
    "문창동",
    "문평동",
    "상대동(대전)",
    "성남동1",
    "월평동",
    "읍내동",
    "정림동"
  ],
  "time_max": "2025-01-01 00:00:00",
  "time_min": "2024-01-01 01:00:00"
}
```

## Feature Families

| family | features | mean missing rate | sample |
|---|---:|---:|---|
| `airkorea_particulate_burden` | 1056 | 0.032727 | `air_day_12_23_allmean_pm10_mean`, `air_day_12_23_allmean_pm10_max`, `air_day_12_23_allmean_pm10_min`, `air_day_12_23_allmean_pm10_std`, `air_day_12_23_allmean_pm10_p95`, `air_day_12_23_allmean_pm10_range`, `air_day_12_23_allmean_pm10_hours_ge_50p0`, `air_day_12_23_allmean_pm10_hours_ge_80p0` |
| `airkorea_oxidant_traffic` | 1584 | 0.027385 | `air_day_12_23_allmean_co_mean`, `air_day_12_23_allmean_co_max`, `air_day_12_23_allmean_co_min`, `air_day_12_23_allmean_co_std`, `air_day_12_23_allmean_co_p95`, `air_day_12_23_allmean_co_range`, `air_day_12_23_allmean_co_hours_ge_0p5`, `air_day_12_23_allmean_co_hours_ge_0p8` |
| `airkorea_selected_station` | 1728 | 0.042870 | `air_day_12_23_dunsan_so2_mean`, `air_day_12_23_dunsan_so2_max`, `air_day_12_23_dunsan_so2_min`, `air_day_12_23_dunsan_so2_std`, `air_day_12_23_dunsan_so2_p95`, `air_day_12_23_dunsan_so2_range`, `air_day_12_23_dunsan_so2_hours_ge_0p003`, `air_day_12_23_dunsan_so2_hours_ge_0p005` |
| `airkorea_dunsan_station` | 576 | 0.044815 | `air_day_12_23_dunsan_so2_mean`, `air_day_12_23_dunsan_so2_max`, `air_day_12_23_dunsan_so2_min`, `air_day_12_23_dunsan_so2_std`, `air_day_12_23_dunsan_so2_p95`, `air_day_12_23_dunsan_so2_range`, `air_day_12_23_dunsan_so2_hours_ge_0p003`, `air_day_12_23_dunsan_so2_hours_ge_0p005` |
| `airkorea_station_dispersion` | 864 | 0.014815 | `air_day_12_23_allmax_so2_mean`, `air_day_12_23_allmax_so2_max`, `air_day_12_23_allmax_so2_min`, `air_day_12_23_allmax_so2_std`, `air_day_12_23_allmax_so2_p95`, `air_day_12_23_allmax_so2_range`, `air_day_12_23_allmax_so2_hours_ge_0p003`, `air_day_12_23_allmax_so2_hours_ge_0p005` |
| `airkorea_all` | 3168 | 0.030118 | `air_day_12_23_allmean_so2_mean`, `air_day_12_23_allmean_so2_max`, `air_day_12_23_allmean_so2_min`, `air_day_12_23_allmean_so2_std`, `air_day_12_23_allmean_so2_p95`, `air_day_12_23_allmean_so2_range`, `air_day_12_23_allmean_so2_hours_ge_0p003`, `air_day_12_23_allmean_so2_hours_ge_0p005` |

## Top Rows

| rank | local | review | dacon | family | target | config | wins | mean delta | max delta | proxy | gap 0.55 | max failed corr |
|---:|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `False` | `False` | `False` | `airkorea_selected_station` | `S4` | `airkorea_selected_station_s4_hgb_l7_w0.025` | 3 | -0.000190312 | +0.000819150 | 0.596056065 | +0.046056065 | +0.262872 |
| 2 | `False` | `False` | `False` | `airkorea_selected_station` | `S4` | `airkorea_selected_station_s4_hgb_l7_w0.05` | 3 | -0.000150190 | +0.001816842 | 0.596061796 | +0.046061796 | +0.262872 |
| 3 | `False` | `False` | `False` | `airkorea_selected_station` | `S4` | `airkorea_selected_station_s4_hgb_l7_w0.01` | 3 | -0.000105165 | +0.000305531 | 0.596068228 | +0.046068228 | +0.262872 |
| 4 | `False` | `False` | `False` | `airkorea_selected_station` | `S4` | `airkorea_selected_station_s4_hgb_l7_w0.005` | 3 | -0.000057551 | +0.000149012 | 0.596075030 | +0.046075030 | +0.262872 |
| 5 | `False` | `False` | `False` | `airkorea_station_dispersion` | `Q3` | `airkorea_station_dispersion_q3_lr_c0_10_w0.005` | 2 | +0.000009678 | +0.000329002 | 0.596084634 | +0.046084634 | +0.385689 |
| 6 | `False` | `False` | `False` | `airkorea_station_dispersion` | `Q3` | `airkorea_station_dispersion_q3_lr_c0_10_w0.01` | 2 | +0.000028881 | +0.000667871 | 0.596087378 | +0.046087378 | +0.385689 |
| 7 | `False` | `False` | `False` | `airkorea_station_dispersion` | `S4` | `airkorea_station_dispersion_s4_lr_c0_10_w0.005` | 2 | +0.000031551 | +0.000164607 | 0.596087759 | +0.046087759 | +0.263022 |
| 8 | `False` | `False` | `False` | `airkorea_particulate_burden` | `S4` | `airkorea_particulate_burden_s4_hgb_l7_w0.005` | 2 | +0.000032056 | +0.000170256 | 0.596087831 | +0.046087831 | +0.262427 |
| 9 | `False` | `False` | `False` | `airkorea_all` | `S4` | `airkorea_all_s4_hgb_l7_w0.005` | 2 | +0.000032056 | +0.000170256 | 0.596087831 | +0.046087831 | +0.262427 |
| 10 | `False` | `False` | `False` | `airkorea_station_dispersion` | `S4` | `airkorea_station_dispersion_s4_lr_c0_03_w0.005` | 1 | +0.000036475 | +0.000168710 | 0.596088463 | +0.046088463 | +0.287252 |
| 11 | `False` | `False` | `False` | `airkorea_particulate_burden` | `S4` | `airkorea_particulate_burden_s4_lr_c0_03_w0.005` | 1 | +0.000038381 | +0.000161439 | 0.596088735 | +0.046088735 | +0.296245 |
| 12 | `False` | `False` | `False` | `airkorea_all` | `S4` | `airkorea_all_s4_lr_c0_03_w0.005` | 1 | +0.000038381 | +0.000161439 | 0.596088735 | +0.046088735 | +0.296245 |
| 13 | `False` | `False` | `False` | `airkorea_particulate_burden` | `S4` | `airkorea_particulate_burden_s4_lr_c0_10_w0.005` | 1 | +0.000041635 | +0.000178872 | 0.596089200 | +0.046089200 | +0.269794 |
| 14 | `False` | `False` | `False` | `airkorea_all` | `S4` | `airkorea_all_s4_lr_c0_10_w0.005` | 1 | +0.000041635 | +0.000178872 | 0.596089200 | +0.046089200 | +0.269794 |
| 15 | `False` | `False` | `False` | `airkorea_station_dispersion` | `Q3` | `airkorea_station_dispersion_q3_lr_c0_03_w0.005` | 2 | +0.000058932 | +0.000396452 | 0.596091671 | +0.046091671 | +0.433098 |
| 16 | `False` | `False` | `False` | `airkorea_station_dispersion` | `S4` | `airkorea_station_dispersion_s4_lr_c0_10_w0.01` | 2 | +0.000072099 | +0.000336893 | 0.596093552 | +0.046093552 | +0.263022 |
| 17 | `False` | `False` | `False` | `airkorea_particulate_burden` | `S4` | `airkorea_particulate_burden_s4_hgb_l7_w0.01` | 2 | +0.000072443 | +0.000347468 | 0.596093601 | +0.046093601 | +0.262427 |
| 18 | `False` | `False` | `False` | `airkorea_all` | `S4` | `airkorea_all_s4_hgb_l7_w0.01` | 2 | +0.000072443 | +0.000347468 | 0.596093601 | +0.046093601 | +0.262427 |
| 19 | `False` | `False` | `False` | `airkorea_station_dispersion` | `S4` | `airkorea_station_dispersion_s4_lr_c0_03_w0.01` | 1 | +0.000080353 | +0.000343862 | 0.596094731 | +0.046094731 | +0.287252 |
| 20 | `False` | `False` | `False` | `airkorea_particulate_burden` | `S4` | `airkorea_particulate_burden_s4_lr_c0_03_w0.01` | 1 | +0.000084009 | +0.000329359 | 0.596095253 | +0.046095253 | +0.296245 |
| 21 | `False` | `False` | `False` | `airkorea_all` | `S4` | `airkorea_all_s4_lr_c0_03_w0.01` | 1 | +0.000084009 | +0.000329359 | 0.596095253 | +0.046095253 | +0.296245 |
| 22 | `False` | `False` | `False` | `airkorea_selected_station` | `S4` | `airkorea_selected_station_s4_lr_c0_03_w0.005` | 1 | +0.000084538 | +0.000186469 | 0.596095329 | +0.046095329 | +0.277596 |
| 23 | `False` | `False` | `False` | `airkorea_particulate_burden` | `S4` | `airkorea_particulate_burden_s4_lr_c0_10_w0.01` | 1 | +0.000091519 | +0.000368940 | 0.596096326 | +0.046096326 | +0.269794 |
| 24 | `False` | `False` | `False` | `airkorea_all` | `S4` | `airkorea_all_s4_lr_c0_10_w0.01` | 1 | +0.000091519 | +0.000368940 | 0.596096326 | +0.046096326 | +0.269794 |
