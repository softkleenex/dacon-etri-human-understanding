# Post-ID342 materially new source hypothesis

Created: `2026-06-25 22:27:44 UTC+09:00`

This is report-only. It creates no feature matrix, predictions, submission artifact, Kaggle run, or DACON submission.

## Decision

- Hypothesis packet ready: `True`
- Top route: `q3_app_name_stress_lexicon_v1`
- Score candidate now: `False`
- Create feature matrix now: `False`
- Run Kaggle now: `False`
- DACON submit now: `False`
- Next action: If paper-deadline work is stable, implement only the bounded dry-run for q3_app_name_stress_lexicon_v1; stop before artifacts unless the target gate clears.

## Score State

- Current public anchor: `ID342` / `0.5960832519`
- Gap to Public 0.55: `+0.0460832519`

## Proposed Routes

| priority | route | targets | why materially new |
|---:|---|---|---|
| 1 | `q3_app_name_stress_lexicon_v1` | `Q3, Q2` | Existing usage features keep seven coarse categories and Markov/session summaries; processed matrices contain no raw app-name lexicon or stress-specific lexical aggregates. |
| 2 | `s3_s4_home_return_sleep_boundary_v1` | `S3, S4, Q3` | Existing GPS/proximity features count tokens and activity windows, but do not explicitly infer a subject-specific night-home anchor, last-home-arrival time, or home-stability deviation. |
| 3 | `q_target_rank_quota_constraint_v1` | `Q1, Q2, Q3` | Subject-prior calibration failed; this route is only distinct if it uses within-subject rank/order constraints plus independent raw-signal scores, not a direct push toward train subject means. |

## Top Route Gate

```json
{
  "gate": {
    "dacon_public_proxy_max": 0.55,
    "failed_frontier_corr_abs_max": 0.2,
    "local_targets": [
      "Q3"
    ],
    "mean_delta_target_vs_id342_max": -0.002,
    "must_beat_id342_on_primary_splits": "at least 4/5",
    "review_public_proxy_max": 0.58
  },
  "minimum_dryrun": [
    "Predeclare a small Korean/English app-name lexicon into stress/work/finance/social/media/routine buckets.",
    "Emit only aggregate counts, time shares, late-evening bursts, and entropy; do not persist raw app names.",
    "Join mScreenStatus and mAmbience only for late-screen-on and speech/noise context."
  ],
  "route": "q3_app_name_stress_lexicon_v1",
  "stop_conditions": [
    "If the lexicon route only improves Q2 or only produces calibration movement, close it.",
    "If raw names are needed in artifacts, stop and redesign into non-secret aggregates.",
    "If gains vanish under subject-hole or LOSO splits, do not create a submission CSV."
  ]
}
```

## Processed Novelty Token Hits

```json
{
  "home_anchor": [],
  "raw_app_name_lexicon": [],
  "subject_quota": []
}
```

## Target Distribution Summary

```json
{
  "target_means": {
    "Q1": 0.4955555555555556,
    "Q2": 0.5622222222222222,
    "Q3": 0.6,
    "S1": 0.6822222222222222,
    "S2": 0.6511111111111111,
    "S3": 0.6622222222222223,
    "S4": 0.56
  },
  "test_date_max": "2024-11-19",
  "test_date_min": "2024-07-06",
  "test_rows": 250,
  "test_subjects": 10,
  "train_date_max": "2024-11-14",
  "train_date_min": "2024-06-03",
  "train_rows": 450,
  "train_subjects": 10
}
```

## Closed Route Status

```json
{
  "all_reports_parse": true,
  "closed_route_count": 32,
  "current_public_anchor_id": 342,
  "current_public_score": 0.5960832519,
  "dacon_submit_now": false,
  "external_priority_ready": true,
  "registered_routes_closed": true,
  "route_count": 32,
  "run_kaggle_now": false,
  "same_family_heavy_rerun_allowed": false
}
```

## Do Not Repeat

- alpha/temperature/subject-prior calibration without independent signal
- weather, air-quality, solar, and holiday date priors
- same-family missingness or subject-state heavy reruns without a changed local gate
- image-only or generic sequence-summary reruns
- tiny S3/S4 micro-overlays near Public 0.596
