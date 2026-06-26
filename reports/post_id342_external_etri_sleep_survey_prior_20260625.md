# Post-ID342 external ETRI sleep/survey prior

Created: `2026-06-25 17:58:05 UTC+09:00`

This is report-only. It creates no feature matrix, predictions, submission artifacts, Kaggle runs, or DACON submissions.

## Decision

- Paired 2020 PM-to-next-sleep rows/users: `523` / `22`
- Q2 fatigue sleep-coupling supported: `True`
- Q3 stress sleep-coupling supported: `False`
- Direct 2026 feature route available: `False`
- External prior gate pass: `False`
- Projected Public proxy: `0.596083252`
- Gap to Public 0.55: `+0.046083252`
- Run Kaggle now: `False`
- DACON submit now: `False`
- Next action: Use the external ETRI sleep/survey files as domain prior only. Q2 fatigue has a coherent next-sleep coupling, but Q3 stress does not support a simple sleep-quality proxy and no direct 2026 feature route is opened. Do not create artifacts, Kaggle runs, or DACON submissions from this prior.

## Paired 2020 Rates

```json
{
  "q2_positive": 0.5659655831739961,
  "q3_positive": 0.5372848948374761,
  "s3_sol_0_30m": 0.6252390057361377,
  "s3_sol_15_30m": 0.4168260038240918,
  "s4_waso_lt20m": 0.14531548757170173,
  "sleep_score_positive": 0.502868068833652
}
```

## Positive Minus Negative Deltas

| label | duration-to-sleep | wakeupduration | sleep score | S3 0-30m | S3 15-30m | S4 WASO<20m |
|---|---:|---:|---:|---:|---:|---:|
| `q2_positive` | -407.064 | -745.873 | -0.118 | +0.061719 | +0.074875 | +0.038814 |
| `q3_positive` | +58.624 | +58.421 | -1.519 | -0.059160 | -0.016367 | -0.029484 |

## Top Correlations

| x | y | n | Spearman |
|---|---|---:|---:|
| `pmEmotion` | `breathing_disturbances_intensity` | 523 | +0.135291 |
| `pmStress` | `snoringepisodecount` | 523 | -0.118704 |
| `pmEmotion` | `hr_average` | 523 | +0.108579 |
| `pmFatigue` | `hr_average` | 523 | -0.094716 |
| `pmFatigue` | `snoringepisodecount` | 523 | -0.085496 |

## Current 2026 Schema Matches

| family | columns | missing rate | examples |
|---|---:|---:|---|
| `sleep_hr_fragmentation` | 29 | 0.420000 | `slp_hr_awake_ratio`, `slp_hr_deep_ratio`, `slp_hr_early_mean`, `slp_hr_late_mean`, `slp_hr_light_ratio`, `slp_hr_max`, `slp_hr_mean`, `slp_hr_median` |
| `pre_sleep_settling` | 17 | 0.162353 | `ac_presleep_charging`, `act_presleep_active`, `screen_presleep_on`, `screen_presleep_ratio`, `slp_act_active_ratio`, `slp_act_n_records`, `slp_act_still_ratio`, `slp_pedo_active` |
| `usage_stress_proxy` | 31 | 0.022151 | `ch2025_mScreenStatus_bedtime_screen_use_sum`, `ch2025_mScreenStatus_m_screen_use_mean`, `ch2025_mScreenStatus_m_screen_use_sum`, `ch2025_mScreenStatus_screen_first_hour_min`, `ch2025_mScreenStatus_screen_last_hour_max`, `ch2025_mUsageStats_app_camera_sum`, `ch2025_mUsageStats_app_chrome_sum`, `ch2025_mUsageStats_app_gallery_sum` |
| `social_routine_proxy` | 39 | 0.207806 | `ble_n_scans`, `ble_n_unique`, `ch2025_mAmbience_ambience_changes_mean`, `ch2025_mAmbience_ambience_changes_sum`, `ch2025_mAmbience_snd_music_max`, `ch2025_mAmbience_snd_music_mean`, `ch2025_mAmbience_snd_outdoor_max`, `ch2025_mAmbience_snd_outdoor_mean` |

## Source Files

```json
{
  "sleep_2019_2018": {
    "columns": [
      "userId",
      "startDt",
      "endDt",
      "device",
      "sleep_score",
      "total_sleep_time",
      "time_in_bed",
      "waso",
      "wakeupcount",
      "aal",
      "movement_index",
      "fragmentation_index",
      "sleep_frag_index"
    ],
    "exists": true,
    "mean_missing_rate": 0.205685618729097,
    "path": "data/external/etri_lifelog/user_sleep_2019_2018.csv",
    "shape": [
      736,
      13
    ]
  },
  "sleep_2020": {
    "columns": [
      "userId",
      "timezone",
      "date",
      "startDt",
      "endDt",
      "lastUpdate",
      "wakeupduration",
      "lightsleepduration",
      "deepsleepduration",
      "wakeupcount",
      "durationtosleep",
      "remsleepduration",
      "durationtowakeup",
      "hr_average",
      "hr_min",
      "hr_max",
      "rr_average",
      "rr_min",
      "rr_max",
      "breathing_disturbances_intensity",
      "snoring",
      "snoringepisodecount",
      "sleep_score"
    ],
    "exists": true,
    "mean_missing_rate": 0.0,
    "path": "data/external/etri_lifelog/user_sleep_2020.csv",
    "shape": [
      615,
      23
    ]
  },
  "survey_2019_2018": {
    "columns": [
      "userId",
      "amPm",
      "inputDt",
      "sleep",
      "sleepProblem",
      "dream",
      "amCondition",
      "amEmotion",
      "pmEmotion",
      "pmStress",
      "ifUnusual",
      "breakfast",
      "lunch",
      "dinner",
      "lateSnack",
      "amCaffeine",
      "amCaffAmount",
      "pmCaffeine",
      "pmCaffAmount",
      "alcohol",
      "aAmount"
    ],
    "exists": true,
    "mean_missing_rate": 0.5495645729108765,
    "path": "data/external/etri_lifelog/user_survey_2019_2018.csv",
    "shape": [
      1285,
      21
    ]
  },
  "survey_2020": {
    "columns": [
      "userId",
      "date",
      "amPm",
      "startInput",
      "endInput",
      "sleep",
      "sleepProblem",
      "dream",
      "amCondition",
      "amEmotion",
      "pmEmotion",
      "pmStress",
      "pmFatigue",
      "caffeine",
      "cAmount(ml)",
      "alcohol",
      "aAmount(ml)"
    ],
    "exists": true,
    "mean_missing_rate": 0.4327594452415112,
    "path": "data/external/etri_lifelog/user_survey_2020.csv",
    "shape": [
      1230,
      17
    ]
  }
}
```
