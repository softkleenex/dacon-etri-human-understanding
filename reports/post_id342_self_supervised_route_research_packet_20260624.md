# Post-ID342 self-supervised route research packet

Created: `2026-06-24 09:04:23 UTC+09:00`

This is report-only. It reads existing reports only and creates no raw-stream cache, embedding, feature matrix, model, prediction, submission artifact, Kaggle run, or DACON submission.

## Decision

- Current anchor: `ID342` / `0.5960832519`
- Gap to Public 0.55: `+0.0460832519`
- Post-ID342 closed routes: `11/12`
- Heavy route is only non-closed route: `True`
- Heavy compute allowed now: `False`
- Dedicated heavy runner required: `True`
- Existing upstream runner sufficient: `False`
- Required phrase: `Self-supervised subject-state heavy compute no-submit 실행 승인`
- Kaggle upstream phrase is not enough: `True`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- DACON submit now: `False`
- Next action: Do not execute heavy compute yet. If the exact heavy approval phrase is provided later, refresh the heavy-compute contract first, then run only no-submit label-free stages.

## Domain Signal Map

| target | meaning | primary modalities | heavy-route hypothesis |
|---|---|---|---|
| `Q3` | Stress before sleep; in the ETRI 2024 target definition, 1 means below the participant average. | mUsageStats, mScreenStatus, mActivity, wHr, mAmbience, mGps | Learn label-free day/evening state tokens that combine app usage, phone rest, movement, HR, ambient context, and location routine before fold-internal Q3 testing. |
| `S3` | Sleep onset latency guideline adherence; 1 means the sleep record meets the guideline. | mScreenStatus, mACStatus, mLight, wLight, mActivity, wPedo, wHr | Represent the transition into sleep as a synchronized phone-wearable context state rather than hand-built SOL or single-channel boundary summaries. |
| `Q2` | Pre-sleep affect/fatigue-style subjective state depending on the competition-year target spec. | mUsageStats, mActivity, mGps, mAmbience, wHr | Use Q2 as a secondary sanity target: it should improve only if the representation learns real day-state structure and does not overfit public-feedback geometry. |

## Success Gates

| gate | required before execution/artifact | criterion | blocks |
|---|---:|---|---|
| `approval` | `True` | Exact phrase required: Self-supervised subject-state heavy compute no-submit 실행 승인 | full raw read, embedding fit, Kaggle run |
| `label_free_pretraining` | `True` | No target labels, no OOF predictions, no public-score feedback, and no donor submission predictions in embedding selection. | OOF recombination, ID287/ID329 direction tuning, post-public target swap |
| `fold_internal_smoke` | `True` | Q3/S3 must beat ID342 on stable near-anchor lenses and must not win only under failed-frontier-inclusive views. | local-frontier-only gain, subject-slice donor selector trap |
| `public_gap_budget` | `True` | Projected public proxy must show material movement beyond ID342, not another 0.595-0.596 micro-change. | same-family Kaggle rerun, tiny calibration/temperature/alpha edit |
| `artifact_contract` | `True` | A separate future artifact contract must name the prediction artifact, log row, notebook/source, and no-submit review. | submission CSV, submission_log.csv edit, DACON submission |

## Research Sources

| source | url | takeaway |
|---|---|---|
| ETRI Human Understanding AI Paper Challenge 2024 dataset design | https://arxiv.org/html/2403.16509v1 | Train/validation/test cohorts differ, validation/test participants are relabeled, and Q3/S3 are participant-relative or guideline-derived targets. |
| DACON 4th ETRI Human Understanding data page | https://dacon.io/en/competitions/official/236468/data | The public data description emphasizes 12 lifelog streams from smartphone, smartwatch, sleep sensor, and self-recording app sources. |
| PixleepFlow | https://arxiv.org/html/2502.17469v2 | Pixel/image-style multimodal lifelog representations are motivated by sensor synchronization, sleep/stress target coupling, and interpretable channel importance. |

## Forbidden Actions Now

```json
{
  "create_submission_artifact": false,
  "dacon_submit_now": false,
  "fit_embedding_now": false,
  "modify_submission_log": false,
  "push_kaggle_now": false,
  "read_full_raw_streams_now": false,
  "run_kaggle_now": false,
  "train_downstream_model_now": false,
  "write_feature_matrix_now": false,
  "write_oof_or_test_predictions_now": false
}
```
