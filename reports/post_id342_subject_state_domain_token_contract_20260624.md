# Post-ID342 subject-state domain/token contract

Created: `2026-06-24 09:22:11 UTC+09:00`

This report freezes domain semantics for the only remaining post-ID342 heavy route. It does not read raw parquet payloads, fit embeddings, create features, run Kaggle, or submit to DACON.

## Decision

- Domain/token contract ready: `True`
- Primary score targets: `['Q3', 'S3']`
- Secondary sanity target: `Q2`
- Create runner files now: `False`
- Run Kaggle now: `False`
- DACON submit now: `False`
- Required phrase: `Self-supervised subject-state heavy compute no-submit 실행 승인`
- Next action: Use this contract to implement the no-submit runner only after exact heavy approval.

## Modality Summary

```json
{
  "cross_target_overlap_modalities": [
    "mActivity",
    "mAmbience",
    "mLight",
    "mScreenStatus",
    "mWifi",
    "wHr",
    "wLight",
    "wPedo"
  ],
  "modality_count": 12,
  "q2_modalities": [
    "mGps",
    "mUsageStats",
    "wPedo"
  ],
  "q3_modalities": [
    "mActivity",
    "mAmbience",
    "mBle",
    "mGps",
    "mLight",
    "mScreenStatus",
    "mUsageStats",
    "mWifi",
    "wHr",
    "wLight",
    "wPedo"
  ],
  "s3_modalities": [
    "mACStatus",
    "mActivity",
    "mAmbience",
    "mLight",
    "mScreenStatus",
    "mWifi",
    "wHr",
    "wLight",
    "wPedo"
  ]
}
```

## Target Token Views

| target | meaning | must capture | danger |
|---|---|---|---|
| `Q3` | low stress before sleep relative to the participant baseline | subject-relative arousal/load deviation; evening recovery and routine break; phone/social/place novelty; wearable HR and activity recovery | A global stress prior can look locally good but miss participant-specific thresholds. |
| `S3` | sleep onset latency adheres to sleep-health guidance | screen-off to still/dark transition; charging and phone parking routine; light/activity/HR settling order; night interruption before stable sleep onset | Daily aggregates flatten the transition ordering that likely determines SOL. |
| `Q2` | low physical fatigue before sleep relative to the participant baseline | daytime physical load; late recovery from high-load day; sleep-transition sanity interactions | Use Q2 only as sanity support; it must not distract from Q3/S3. |

## Modality Contracts

| modality | domain | primary targets | token role | privacy guard |
|---|---|---|---|---|
| `mUsageStats` | `phone_app_load` | `['Q2', 'Q3']` | app-session burden, late-evening stimulation, routine break, social/media load | Hash or bucket app names; do not expose raw app-name identifiers in model reports. |
| `mScreenStatus` | `sleep_boundary_phone_use` | `['S3', 'Q3']` | screen-off settling, last use before sleep, night interruption proxy | Use binary state transitions only. |
| `mACStatus` | `charge_bedside_routine` | `['S3', 'S1']` | bedside charging habit and nightly phone parking proxy | Use charging binary state only. |
| `mActivity` | `phone_activity_state` | `['S3', 'Q3']` | stillness onset, activity fragmentation, commute/activity stress proxy | Use coarse activity labels only. |
| `wHr` | `wearable_recovery_arousal` | `['Q3', 'S3', 'S4']` | physiological arousal, recovery slope, nocturnal fragmentation proxy | Normalize inside fold by subject; do not fit subject baselines on validation labels. |
| `wPedo` | `wearable_exertion_inactivity` | `['Q2', 'Q3', 'S3']` | daytime load, evening inactivity, sleep-preparation settling | Use aggregate movement states only. |
| `mLight` | `phone_light_context` | `['S3', 'Q3']` | darkness onset, light exposure, pre-sleep environment proxy | Use scalar light values only. |
| `wLight` | `watch_light_context` | `['S3', 'Q3']` | body-worn light exposure and dark-state confirmation | Use scalar light values only. |
| `mGps` | `mobility_place_routine` | `['Q2', 'Q3']` | mobility load, place regularity, commute/stress context | Never expose raw coordinates; derive coarse, subject-relative tokens only. |
| `mWifi` | `place_stability` | `['Q3', 'S3']` | home/work/place stability and routine deviation | Hash BSSID and report only counts/entropy. |
| `mBle` | `near_device_social_context` | `['Q3']` | near-device density, social/place context, routine deviation | Hash device IDs and use only counts/classes/RSSI summaries. |
| `mAmbience` | `ambient_sound_context` | `['Q3', 'S3']` | noise/speech/music/silence context and sleep environment proxy | Use label categories and probabilities only; no audio payloads. |

## Tokenization Windows

| name | relative hours | purpose | binning |
|---|---|---|---|
| `day_grid` | 00:00-24:00 by clock day | daily load and routine stability for Q2/Q3 | 24 hourly bins with missingness mask |
| `anchored_evening_grid` | 18:00-03:00 anchored to lifelog/sleep date | pre-sleep load, stress recovery, and transition onset | 30 or 60 minute bins with modality masks |
| `sleep_transition_grid` | last active/light/screen state through first sustained still/dark state | S3 sleep-onset-latency state ordering | event tokens plus coarse time deltas |
| `subject_baseline_delta` | fold-internal subject history only | Q3/Q2 participant-relative labels | deviation from fold-fit subject medians or rolling unlabeled norms |

## Objective Ablations After Approval

| stage | name | why |
|---:|---|---|
| 1 | `masked_single_modality_reconstruction` | Sanity-check token quality and missingness masks before cross-modal learning. |
| 2 | `cross_modal_masked_reconstruction` | Primary MoCA-style objective for phone/watch correlations and missing channels. |
| 3 | `latent_masking_with_modality_specific_encoders` | CroSSL-style ablation for robust missing-modality aggregation. |
| 4 | `subject_baseline_dynamics_prediction` | Personalized stress SSL ablation for Q3 participant thresholds. |
| 5 | `fold_internal_downstream_smoke` | Only after label-free embeddings are frozen; compare Q3/S3 against ID342 stable lenses. |

## Primary Sources

| source | url | contract use |
|---|---|---|
| Human Understanding AI Paper Challenge 2024 - Dataset Design | https://arxiv.org/abs/2403.16509 | Use participant/date-aware validation and account for noisy, missing, real-world phone and wearable streams. |
| Understanding Human Daily Experience Through Continuous Sensing: ETRI Lifelog Dataset 2024 | https://arxiv.org/html/2508.03698v1 | Treat Q3 as within-subject pre-sleep stress and S3 as sleep-onset-latency adherence; keep all timestamps in KST and align by subject/date. |
| MoCA: Multi-modal Cross-masked Autoencoder for Digital Health Measurements | https://arxiv.org/abs/2506.02260 | Use cross-modality masking so missing phone/watch channels are part of the objective. |
| CroSSL: Cross-modal Self-Supervised Learning for Time-series Signals | https://arxiv.org/abs/2307.16847 | Use modality-specific encoders plus a cross-modal aggregator instead of concatenating all streams first. |
| Personalization of Stress Mobile Sensing using Self-Supervised Learning | https://arxiv.org/abs/2308.02731 | Evaluate Q3 with explicit subject personalization, not only pooled validation loss. |
| Personalized Prediction of Recurrent Stress Events Using Self-supervised Learning | https://arxiv.org/abs/2307.03337 | Keep a subject-baseline dynamics objective for stress-state transfer as an ablation. |

## Forbidden Actions Now

```json
{
  "create_predictions": false,
  "create_submission_artifact": false,
  "fit_embedding": false,
  "modify_submission_log": false,
  "push_kaggle": false,
  "read_raw_parquet_payloads": false,
  "run_kaggle": false,
  "submit_dacon": false,
  "train_downstream_model": false,
  "write_feature_matrix": false,
  "write_token_cache": false
}
```
