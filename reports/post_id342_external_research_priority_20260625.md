# Post-ID342 external research priority refresh

Created: `2026-06-25 18:00:55 UTC+09:00`

This report updates research priority only. It performs no raw payload read, embedding fit, feature creation, model training, Kaggle action, submission-log edit, or DACON submission.

## Decision

- Research priority ready: `True`
- Top priority route: `materially_new_external_source_hypothesis`
- Same-family heavy rerun allowed: `False`
- MIS-LSTM Kaggle allowed now: `False`
- Heavy compute allowed now: `False`
- Create runner files now: `False`
- Run Kaggle now: `False`
- DACON submit now: `False`
- Next action: Keep execution closed. The validation-weighted pseudo-public audit and the first external solar/holiday and KMA ASOS weather gates are now closed, the approved KMA living-weather and AirKorea information APIs have been smoke-tested and do not provide a 2024 historical feature source, and the separate downloaded AirKorea 2024 context gate also opens no reviewable row. The historical ETRI sleep/survey prior supports Q2 fatigue coupling but opens no direct Q3/S3/S4 feature route. The cross-source split audit finds no reviewable target, the focused S3 phone/GPS interaction remains far from review thresholds, and the final-day independent sweep opens no Public-0.58 review row; next, define a materially different Q3/S3/S4 source hypothesis before any Kaggle, artifact, or DACON action.

## Score State

```json
{
  "best_public_anchor_id": 342,
  "best_public_score": 0.5960832519,
  "closed_route_count": 32,
  "gap_to_public_055": 0.046083251899999955,
  "route_count": 32,
  "target_public_score": 0.55
}
```

## Recommended Next Route

```json
{
  "dacon_candidate_threshold_public_proxy": 0.55,
  "name": "materially_new_external_source_hypothesis",
  "priority_components": [
    "new source mechanism rather than another weather/air-quality date prior",
    "avoid simple Q3-from-sleep coupling because external ETRI prior did not support it",
    "target-specific Q3/S3/S4 local gate before artifacts",
    "validation-weighted pseudo-public and stable-shape review",
    "failed-frontier exclusion and leave-one-out stress checks"
  ],
  "review_threshold_public_proxy": 0.58
}
```

## Research Implications

| finding | evidence | action |
|---|---|---|
| The large public gap is representation-sized, not calibration-sized. | ID342 is still +0.0460832519 from Public 0.55 after 32 registered routes closed. | Do not spend DACON on alpha, temperature, target-swap, or image-only variants. |
| Missingness remains important, but the first missingness-heavy execution did not transfer. | The post-ID342 missingness subject-state no-submit route scanned all raw rows and passed 0 stable rows. | Do not rerun AIM/CroSSL-style missingness unless the architecture and local gate materially change. |
| Temporal block structure has been locally tested and is not sufficient by itself. | The local MIS-LSTM-style block proxy found only S4 and failed Q3/S3. | Do not launch a Kaggle runner from this route; repair validation or change the representation materially. |
| Validation weighting is useful as a guardrail but does not create a candidate route. | The validation-weighted pseudo-public audit found 0 credible Public-0.55 rows and 0 credible Public-0.58 review rows. | Use it to reject future shelf rows; do not rerun it as another candidate scan. |
| The first external date-derived source is not stable enough. | Solar/seasonal/Korean-holiday priors move S4 locally in 3/5 lenses but pass 0 stable rows. | Do not promote the external-calendar prior; require a different source mechanism. |
| Historical KMA weather is available but not strong enough. | The ASOS weather gate finds a best S4 pressure/precipitation row near Public 0.596006 with 0 review passes. | Keep KMA weather as negative external-source evidence unless a new source changes the mechanism. |
| The approved non-ASOS public-data APIs do not cover the 2024 feature window. | The public-data smoke verifies KMA living-weather current forecasts and AirKorea current/recent endpoints, but KMA living-weather rejects 2024 history and AirKorea station 3MONTH returns recent 2026 data. | Do not build features from these approved endpoints; require a separate historical/statistics source. |
| The separate downloaded AirKorea 2024 historical source is valid but too weak. | The archive audit finds 13 Daejeon stations and all 4,200 expected timestamps, but the ID342 AirKorea gate has 0 local/review/DACON passes and best proxy 0.596056065. | Do not promote air-quality features; require a materially different Q3/S3/S4 source mechanism. |
| Historical ETRI sleep/survey priors do not rescue Q3. | The 2020 PM-to-next-sleep audit pairs 523 rows. Lower fatigue aligns with shorter sleep latency and lower WASO, but lower stress does not align with better sleep outcomes and no direct 2026 feature route is available. | Use this as a warning against simple Q3-from-sleep coupling; do not promote or launch from it. |
| Cross-source weak-signal alignment is too small to promote. | S3 has a phone/GPS stable micro-alignment, but its best proxy remains near 0.596; S4 conflicts on LOSO. | Use the S3 alignment as design hint only; do not create artifacts or launch Kaggle from it. |
| Focused S3 phone/GPS interactions improve local stability but not public distance. | The best S3 phone/GPS concat row wins 5/5 splits, but still projects only to 0.595849549. | Do not submit the micro-signal; require a materially larger S3/S4 source before promotion. |
| The final-day independent sweep found a local S4 signal but no reviewable candidate. | The best S4 sleep-solar row wins 4/5 with mean delta -0.002013473, but projects only to 0.595795613. | Keep submission closed unless a future row clears the Public-0.58 review proxy. |
| Image-style layouts are useful as priors but not sufficient. | PixleepFlow supports synchronized sensor images, but local ID342 image/CNN smokes did not pass. | Use image layout only with block-sequence aggregation and discrete-event handling. |

## Method Candidates

| priority | route | fit | why | risk |
|---:|---|---|---|---|
| 1 | `materially_new_external_source_hypothesis` | highest_after_validation_audit_closed | The registered OOF shelf, raw sleep/event routes, block-sequence proxy, and validation-weighted audit all fail to open even a credible Public-0.58 review row. | The bar is high: small local overlays and same-family heavy reruns have repeatedly compressed near Public 0.596. |
| 3 | `solar_seasonal_holiday_prior` | completed_closed | External date-derived Seoul/Korea photoperiod, season, and holiday proximity features produced only unstable S4 movement and no eligible Q3/S3/S4 row. | The S4 mean delta is tempting but loses subject/generalization stability. |
| 4 | `airkorea_2024_historical_gate` | completed_closed | The downloaded 2024 AirKorea archive provides a true historical external source with full Daejeon hourly coverage, but its ID342 gate opens no local or Public-0.58 review row. | The small S4 movement is date-structured and failed frontier-correlation checks. |
| 4 | `approved_public_data_api_feasibility` | completed_closed | KMA living-weather and AirKorea information endpoints are reachable, but the smoke does not expose a date-bounded 2024 historical source beyond KMA ASOS. | Recent/realtime air-quality data cannot be backfilled into the 2024 train/test window without a separate source. |
| 4 | `external_etri_sleep_survey_prior` | completed_closed | Historical ETRI sleep/survey files provide direct PM fatigue/stress and sleep metrics, but the audit supports Q2 fatigue coupling only and opens no direct Q3/S3/S4 2026 feature route. | Directly concatenating historical users or survey/sleep labels would not match the 2026 feature contract. |
| 4 | `kma_asos_weather_context` | completed_closed | Historical Daejeon ASOS hourly weather provides true external observation data, but the first ID342 gate finds only a small unstable S4 signal and no Public-0.58 review row. | Weather is strongly date-structured and can mimic failed frontier/date movement if overweighted. |
| 4 | `validation_weighted_pseudo_public_split` | completed_closed | The audit evaluated target-weighted full/subject-hole/LOSO/test-neighbor lenses and found no credible Public-0.55 or Public-0.58 stable-shape row. | Failed-frontier rows still look locally attractive, so this lens must stay paired with shape penalties. |
| 5 | `mis_lstm_multichannel_block_sequence_v2` | high_but_local_and_validation_gates_failed | The first local block-sequence proxy found only S4. A second attempt must change the representation, not simply rerun the same block aggregates. | Do not launch Kaggle from the current local proxy, which failed Q3/S3 and projected only 0.5959. |
| 6 | `aim_cross_modal_masked_subject_state_v2` | high_but_already_failed_once | AIM/CroSSL still matches incomplete lifelog data, but the post-ID342 subject-state and missingness runs did not open a supervised gate. | Same-family missingness/subject-state reruns are closed unless the local feasibility story changes. |
| 7 | `time_mae_semantic_window_subject_state` | high | The dataset is label-scarce and long-horizon; semantic sub-series windows are more relevant than point loss. | Can collapse into another generic sequence summary unless cross-modal missingness is retained. |
| 8 | `crossl_latent_modality_dropout` | high | The raw streams have natural missing modalities and different sampling frequencies. | May be too small to close a 0.046 public gap unless paired with AIM-style inherited masking. |
| 9 | `tnc_subject_temporal_neighborhood` | medium | Q3/S3 are state-transition targets, so local temporal smoothness is plausible. | High leakage risk if neighborhoods are not fold- and date-safe; not a standalone first route. |
| 10 | `ts2vec_hierarchical_context` | medium | Hierarchical timestamp/subsequence embeddings are broadly useful for irregular lifelog windows. | Less directly matched to missing modality and cross-device behavior than AIM/CroSSL. |
| 11 | `pixleepflow_image_layout_only` | low_as_standalone | Image conversion is domain-relevant, but local sequence-image and tiny CNN gates did not open Q3/S3. | Do not launch another image-only or random-conv route from current evidence. |

## External Sources

| source | type | URL | implication |
|---|---|---|---|
| Understanding Human Daily Experience Through Continuous Sensing: ETRI Lifelog Dataset 2024 | dataset_paper | https://arxiv.org/html/2508.03698v1 | The remaining route should model day-state and sleep-transition context jointly. |
| The 4th ETRI Human Understanding AI Paper Challenge data page | competition_page | https://dacon.io/en/competitions/official/236468/data | A future runner should preserve all 12 modalities and their missingness patterns. |
| PixleepFlow: A Pixel-Based Lifelog Framework for Predicting Sleep Quality and Stress Level | etri_competition_method | https://arxiv.org/html/2502.17469v2 | Image layout is useful as a representation prior, but local ID342 smokes say it is not enough alone. |
| MIS-LSTM: Multichannel Image-Sequence LSTM for Sleep Quality and Stress Prediction | etri_competition_method | https://arxiv.org/abs/2509.11232 | A future image/sequence attempt should not replay stacked single images; it needs block-level temporal aggregation plus discrete-event handling and uncertainty checks. |
| Multi-modal Time Series Analysis: A Tutorial and Survey | survey | https://arxiv.org/html/2503.13709v1 | The next design should use cross-modal alignment/fusion with explicit robustness checks. |
| CroSSL: Cross-modal Self-Supervised Learning for Time-series through Latent Masking | method_paper | https://arxiv.org/abs/2307.16847 | This is a strong template for phone/watch/sleep-context modality dropout and fusion. |
| LSM-2: Learning from Incomplete Wearable Sensor Data | method_paper | https://arxiv.org/html/2506.05321v1 | Missingness should be a first-class token, not a preprocessing nuisance to fill away. |
| TimeMAE: Self-Supervised Representations of Time Series with Decoupled Masked Autoencoders | method_paper | https://arxiv.org/html/2303.00320v4 | Use 30-minute to multi-hour windows as semantic units instead of minute-level reconstruction only. |
| Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding | method_paper | https://arxiv.org/abs/2106.00750 | Subject-day neighborhoods are plausible, but validation must prevent date/subject leakage. |
| TS2Vec: Towards Universal Representation of Time Series | method_paper | https://arxiv.org/abs/2106.10466 | Use as a secondary objective only after missingness-aware cross-modal masking is fixed. |

## Prerequisites

```json
{
  "airkorea_2024_historical_archive_ready": true,
  "airkorea_2024_historical_gate_already_failed": true,
  "approved_non_asos_public_data_api_historical_route_closed": true,
  "approved_public_data_api_smoke_checked": true,
  "cross_source_split_stability_route_already_failed": true,
  "domain_contract_ready": true,
  "execution_closed_now": true,
  "external_etri_sleep_survey_prior_already_failed": true,
  "final_day_independent_signal_route_already_failed": true,
  "goal_not_achieved": true,
  "id342_anchor_current": true,
  "input_reports_parse": true,
  "kma_weather_context_route_already_failed": true,
  "mis_lstm_block_route_already_failed": true,
  "missingness_route_already_failed": true,
  "public_gap_penalty_closed": true,
  "raw_payload_not_read": true,
  "raw_schema_contract_ready": true,
  "registered_routes_closed": true,
  "s3_phone_gps_interaction_route_already_failed": true,
  "sequence_event_route_already_failed": true,
  "solar_seasonal_holiday_route_already_failed": true,
  "success_thresholds_ready": true,
  "validation_weighted_route_already_failed": true
}
```

## Forbidden Actions Now

```json
{
  "create_predictions": false,
  "create_submission_artifact": false,
  "fit_embedding": false,
  "modify_submission_log": false,
  "pull_kaggle_output": false,
  "push_kaggle": false,
  "read_raw_parquet_payloads": false,
  "run_kaggle": false,
  "submit_dacon": false,
  "train_model": false,
  "write_feature_matrix": false,
  "write_token_cache": false
}
```
