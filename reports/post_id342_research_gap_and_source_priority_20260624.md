# Post-ID342 research gap and source priority

Created: `2026-06-24 03:11:23 UTC+09:00`

This is report-only. It creates no predictions, submission artifacts, Kaggle runs, or DACON submissions.

## Anchor read

- Current public anchor: ID342 Public `0.5960832519`, local `0.534000`
- Delta vs ID178 Public: `-0.0053280490`
- Delta vs ID337 Public: `-0.0045373767`
- Gap to Public 0.55 target: `+0.0460832519`

## Target bottleneck

- ID342 Q3/S3/S4 local target losses: `0.5632259042854845`, `0.48994468343497904`, `0.5916290514500427`
- Q3 lag-1 agreement: `0.6522727272727272`; top same-day correlation: `{'target': 'Q2', 'pearson_corr': 0.3401292217410808, 'joint_positive_rate': 0.42, 'Q2_positive_given_Q3_positive': 0.7, 'Q2_positive_given_Q3_negative': 0.35555555555555557}`
- S3 lag-1 agreement: `0.6568181818181819`; top same-day correlation: `{'target': 'S2', 'pearson_corr': 0.39402250407091105, 'joint_positive_rate': 0.52, 'S2_positive_given_S3_positive': 0.785234899328859, 'S2_positive_given_S3_negative': 0.3881578947368421}`

## Closed evidence

- Local Q3/S3 lanes closed: `True`; closed lanes: `8`
- Clean backbone closed: `True`; next lane: `stronger_raw_or_sequence_source_model`
- Phase-2 block A selected targets: `['Q1']`; final gate pass: `False`
- Phase-2 block B pass: `False`; usage temporal pass: `False`

## Source priority

| rank | route | status | focus | Kaggle fit | submit allowed |
|---:|---|---|---|---|---|
| 1 | `raw_temporal_q3s3_sequence_image` | `open_next` | `Q3,S3` | `good_cpu_first_no_submit` | `False` |
| 2 | `paper_baseline_time_zone_rebuild_id342_aware` | `open_supporting` | `Q3,S3,Q1,S4` | `local_or_kaggle_cpu` | `False` |
| 3 | `self_supervised_subject_state_embedding` | `heavy_deferred` | `Q2,Q3,S3` | `kaggle_cpu_or_gpu_later` | `False` |
| 4 | `material_feature_micro_overlays` | `deprioritized` | `S3` | `not_primary` | `False` |

## Decision

- Raw temporal Q3/S3 sequence-image primary: `True`
- Kaggle no-submit recommended: `True`
- Create submission artifact now: `False`
- DACON submit now: `False`
- Next action: Implement and run a report-only ID342-aware raw temporal Q3/S3 sequence-image diagnostic. If local gates pass, push it through Kaggle no-submit. Do not create or submit a DACON artifact from the existing material feature overlays or clean ID221/ID226 replay.
