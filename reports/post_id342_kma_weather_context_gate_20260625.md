# Post-ID342 KMA weather context gate

Created: `2026-06-25 16:07:44 UTC+09:00`

This is report-only. It creates no feature matrix artifacts, predictions, submission artifacts, Kaggle runs, or DACON submissions.

## Decision

- Weather cache available: `True`
- Reviewed candidates: `144`
- Local gate passes: `0`
- Review gate passes: `0`
- DACON gate passes: `0`
- Best candidate: `kma_weather_precip_pressure_s4_lr_c0_10_w0.05`
- Best target/family: `S4` / `kma_weather_precip_pressure`
- Best projected Public proxy: `0.596006236`
- Gate pass: `False`
- Kaggle now: `False`
- DACON submit now: `False`
- Next action: Promote nothing unless a row clears the Public-0.58 review proxy and a separate artifact contract. This gate is weather-source evidence only.

## Weather Cache

```json
{
  "cache_rows": 4200,
  "columns": [
    "tm",
    "rnum",
    "stnId",
    "stnNm",
    "ta",
    "taQcflg",
    "rn",
    "rnQcflg",
    "ws",
    "wsQcflg",
    "wd",
    "wdQcflg",
    "hm",
    "hmQcflg",
    "pv",
    "td",
    "pa",
    "paQcflg",
    "ps",
    "psQcflg",
    "ss",
    "ssQcflg",
    "icsr",
    "dsnw",
    "hr3Fhsc",
    "dc10Tca",
    "dc10LmcsCa",
    "clfmAbbrCd",
    "lcsCh",
    "vs",
    "gndSttCd",
    "dmstMtphNo",
    "ts",
    "tsQcflg",
    "m005Te",
    "m01Te",
    "m02Te",
    "m03Te"
  ],
  "secret_values_stored": false,
  "station_ids": [
    "133"
  ],
  "station_names": [
    "\ub300\uc804"
  ],
  "time_max": "2024-11-22 23:00:00",
  "time_min": "2024-06-01 00:00:00",
  "weather_cache": "data/external/weather/kma_asos_hourly_20240601_20241122.csv"
}
```

## Feature Families

| family | features | mean missing rate | sample |
|---|---:|---:|---|
| `kma_weather_thermal_load` | 156 | 0.015271 | `kma_day_12_23_temp_mean`, `kma_day_12_23_temp_max`, `kma_day_12_23_temp_min`, `kma_day_12_23_temp_std`, `kma_day_12_23_temp_range`, `kma_day_12_23_hot_hours_ge26`, `kma_day_12_23_hot_hours_ge28`, `kma_day_12_23_humidity_mean` |
| `kma_weather_precip_pressure` | 108 | 0.014815 | `kma_day_12_23_rain_sum`, `kma_day_12_23_rain_max`, `kma_day_12_23_rain_hours`, `kma_day_12_23_wind_mean`, `kma_day_12_23_wind_max`, `kma_day_12_23_pa_mean`, `kma_day_12_23_pa_range`, `kma_day_12_23_ps_mean` |
| `kma_weather_solar_ground` | 36 | 0.016790 | `kma_day_12_23_ss_sum`, `kma_day_12_23_icsr_sum`, `kma_day_12_23_ground_temp_mean`, `kma_night_00_11_ss_sum`, `kma_night_00_11_icsr_sum`, `kma_night_00_11_ground_temp_mean`, `kma_sleep_core_22_06_ss_sum`, `kma_sleep_core_22_06_icsr_sum` |
| `kma_weather_all` | 300 | 0.015052 | `kma_day_12_23_temp_mean`, `kma_day_12_23_temp_max`, `kma_day_12_23_temp_min`, `kma_day_12_23_temp_std`, `kma_day_12_23_temp_range`, `kma_day_12_23_hot_hours_ge26`, `kma_day_12_23_hot_hours_ge28`, `kma_day_12_23_cool_hours_lt15` |

## Top Rows

| rank | local | review | dacon | family | target | config | wins | mean delta | max delta | proxy | gap 0.55 | max failed corr |
|---:|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `False` | `False` | `False` | `kma_weather_precip_pressure` | `S4` | `kma_weather_precip_pressure_s4_lr_c0_10_w0.05` | 2 | -0.000539108 | +0.000515499 | 0.596006236 | +0.046006236 | +0.269367 |
| 2 | `False` | `False` | `False` | `kma_weather_precip_pressure` | `S4` | `kma_weather_precip_pressure_s4_lr_c0_10_w0.025` | 3 | -0.000376671 | +0.000159935 | 0.596029442 | +0.046029442 | +0.269367 |
| 3 | `False` | `False` | `False` | `kma_weather_thermal_load` | `S4` | `kma_weather_thermal_load_s4_lr_c0_10_w0.05` | 2 | -0.000328673 | +0.000734800 | 0.596036299 | +0.046036299 | +0.266624 |
| 4 | `False` | `False` | `False` | `kma_weather_thermal_load` | `S4` | `kma_weather_thermal_load_s4_lr_c0_10_w0.025` | 2 | -0.000275953 | +0.000281916 | 0.596043830 | +0.046043830 | +0.266624 |
| 5 | `False` | `False` | `False` | `kma_weather_precip_pressure` | `S4` | `kma_weather_precip_pressure_s4_lr_c0_10_w0.01` | 4 | -0.000177572 | +0.000039647 | 0.596057884 | +0.046057884 | +0.269367 |
| 6 | `False` | `False` | `False` | `kma_weather_precip_pressure` | `S4` | `kma_weather_precip_pressure_s4_lr_c0_03_w0.025` | 2 | -0.000170864 | +0.000436599 | 0.596058843 | +0.046058843 | +0.293991 |
| 7 | `False` | `False` | `False` | `kma_weather_precip_pressure` | `S4` | `kma_weather_precip_pressure_s4_lr_c0_03_w0.05` | 2 | -0.000155438 | +0.001040278 | 0.596061047 | +0.046061047 | +0.293991 |
| 8 | `False` | `False` | `False` | `kma_weather_thermal_load` | `S4` | `kma_weather_thermal_load_s4_lr_c0_10_w0.01` | 3 | -0.000138365 | +0.000091444 | 0.596063486 | +0.046063486 | +0.266624 |
| 9 | `False` | `False` | `False` | `kma_weather_all` | `S4` | `kma_weather_all_s4_lr_c0_10_w0.025` | 3 | -0.000129753 | +0.000519270 | 0.596064716 | +0.046064716 | +0.222839 |
| 10 | `False` | `False` | `False` | `kma_weather_precip_pressure` | `S4` | `kma_weather_precip_pressure_s4_lr_c0_10_w0.005` | 4 | -0.000093380 | +0.000015691 | 0.596069912 | +0.046069912 | +0.269367 |
| 11 | `False` | `False` | `False` | `kma_weather_precip_pressure` | `S4` | `kma_weather_precip_pressure_s4_lr_c0_03_w0.01` | 2 | -0.000091700 | +0.000153878 | 0.596070152 | +0.046070152 | +0.293991 |
| 12 | `False` | `False` | `False` | `kma_weather_all` | `S4` | `kma_weather_all_s4_lr_c0_10_w0.01` | 3 | -0.000085914 | +0.000176207 | 0.596070979 | +0.046070979 | +0.222839 |
| 13 | `False` | `False` | `False` | `kma_weather_thermal_load` | `S4` | `kma_weather_thermal_load_s4_lr_c0_10_w0.005` | 3 | -0.000073957 | +0.000042094 | 0.596072687 | +0.046072687 | +0.266624 |
| 14 | `False` | `False` | `False` | `kma_weather_precip_pressure` | `S4` | `kma_weather_precip_pressure_s4_lr_c0_03_w0.005` | 2 | -0.000049834 | +0.000073414 | 0.596076133 | +0.046076133 | +0.293991 |
| 15 | `False` | `False` | `False` | `kma_weather_all` | `S4` | `kma_weather_all_s4_lr_c0_10_w0.005` | 3 | -0.000048755 | +0.000082744 | 0.596076287 | +0.046076287 | +0.222839 |
| 16 | `False` | `False` | `False` | `kma_weather_solar_ground` | `S4` | `kma_weather_solar_ground_s4_lr_c0_03_w0.01` | 3 | -0.000007387 | +0.000305114 | 0.596082197 | +0.046082197 | +0.286531 |
| 17 | `False` | `False` | `False` | `kma_weather_solar_ground` | `S4` | `kma_weather_solar_ground_s4_lr_c0_03_w0.005` | 3 | -0.000006651 | +0.000149704 | 0.596082302 | +0.046082302 | +0.286531 |
| 18 | `False` | `False` | `False` | `kma_weather_thermal_load` | `S4` | `kma_weather_thermal_load_s4_lr_c0_03_w0.005` | 2 | -0.000004834 | +0.000116214 | 0.596082561 | +0.046082561 | +0.289558 |
| 19 | `False` | `False` | `False` | `kma_weather_thermal_load` | `S4` | `kma_weather_thermal_load_s4_lr_c0_03_w0.01` | 2 | -0.000002193 | +0.000238047 | 0.596082939 | +0.046082939 | +0.289558 |
| 20 | `False` | `False` | `False` | `kma_weather_all` | `S4` | `kma_weather_all_s4_lr_c0_03_w0.005` | 1 | -0.000000749 | +0.000061083 | 0.596083145 | +0.046083145 | +0.247187 |
| 21 | `False` | `False` | `False` | `kma_weather_solar_ground` | `S4` | `kma_weather_solar_ground_s4_lr_c0_10_w0.005` | 3 | +0.000001756 | +0.000134191 | 0.596083503 | +0.046083503 | +0.270436 |
| 22 | `False` | `False` | `False` | `kma_weather_all` | `S4` | `kma_weather_all_s4_lr_c0_03_w0.01` | 1 | +0.000007291 | +0.000130488 | 0.596084293 | +0.046084293 | +0.247187 |
| 23 | `False` | `False` | `False` | `kma_weather_solar_ground` | `S4` | `kma_weather_solar_ground_s4_lr_c0_10_w0.01` | 3 | +0.000009762 | +0.000274629 | 0.596084646 | +0.046084646 | +0.270436 |
| 24 | `False` | `False` | `False` | `kma_weather_solar_ground` | `S4` | `kma_weather_solar_ground_s4_hgb_l7_w0.005` | 3 | +0.000011342 | +0.000150641 | 0.596084872 | +0.046084872 | +0.259270 |
