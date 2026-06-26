# Post-ID341 Q3/S3 remaining route inventory

Created: `2026-06-23 18:15:19 UTC+09:00`

This is report-only. It creates no predictions, submission CSV, notebook artifact, Kaggle run, or DACON submission.

## Decision

- Inventory ready: `True`
- Q3/S3 still uncovered: `True`
- GPS topology model lane closed: `True`
- External Public 0.5917 direct route closed: `True`
- Local no-approval score route exists: `False`
- Only score-relevant open route: `explicit_approved_clean_upstream_id221_to_id226`
- Requires explicit Kaggle approval: `True`
- Approval phrase: `Kaggle upstream rebuild no-submit 실행 승인`
- Manual probe if spending DACON submission: `None`
- Manual probe family status: `consumed_after_id341_id340_id337_public_micro_probe_slate`
- Create submission artifact: `False`
- Submit now: `False`
- Next action: Do not create a local ID342 from closed Q3/S3 lanes. The next score-relevant action is the explicit-approved no-submit Kaggle upstream rebuild; otherwise continue with report-only design/tests.

## Checks

```json
{
  "all_inputs_ready": true,
  "all_raw_sources_readable": true,
  "q3s3_local_lanes_closed": true,
  "fold_nested_sibling_bridge_closed": true,
  "gps_path_topology_model_lane_closed": true,
  "external_public05917_direct_route_closed": true,
  "local_no_approval_score_route_count": 0,
  "open_score_route_count": 1
}
```

## Closed local routes

| route | status | targets | artifact allowed | reason |
|---|---|---|---|---|
| `physio_light_onset` | `closed` | `Q3,S3` | `False` | No selected Q3/S3 target from wearable/light/movement/onset features. |
| `semantic_routine` | `closed` | `Q3,S3` | `False` | State/routine lane was structurally computable but selected no Q3/S3 target. |
| `calendar_sleep_prior` | `closed` | `Q3,S3` | `False` | Calendar, subject-progress, demographics, and materialized sleep-window summaries selected no target. |
| `subject_drift` | `closed` | `Q3,S3` | `False` | Subject residual drift is too weak for Q3/S3 and fails ID178/Public-0.58 promotion requirements. |
| `event_cadence` | `closed` | `Q3,S3` | `False` | Raw timestamp cadence, silence gaps, and inter-event intervals selected no Q3/S3 target. |
| `place_social_novelty` | `closed` | `Q3,S3` | `False` | GPS/WiFi/BLE place-social novelty selected no Q3/S3 target and has no reviewable low-movement row. |
| `routine_stability_place_entropy` | `closed` | `Q3,S3` | `False` | Routine-stability/place-entropy selected no Q3/S3 target and has no proxy-below-anchor row. |
| `sensor_adherence_dropout` | `closed` | `Q3,S3` | `False` | Sensor-adherence/dropout selected no Q3/S3 target and is closed as a standalone local lane. |
| `cross_modal_disagreement_scout` | `closed_scout` | `Q3,S3` | `False` | Phone/watch cross-modal disagreement selected no Q3/S3 target; kept as optional scout evidence without changing the authoritative 8-lane prelaunch contract. |
| `q2_only_local_fallback` | `closed_for_q3s3_score_route` | `Q2` | `False` | The local fallback leaves Q3/S3 uncovered and cannot justify a new artifact. |
| `fold_nested_sibling_bridge` | `closed_for_local_artifact` | `Q3,S3` | `False` | Leakage-safe Q3<-Q2 and S3<-S2 nested sibling predictions selected no Q3/S3 target and provide no Public-0.58 evidence. |
| `gps_path_topology_sequence_v1` | `closed_after_model_smoke` | `Q3,S3` | `False` | The raw GPS path-topology feature matrix is valid, but fold-internal model smoke selected no Q3/S3 target and opened no Public-0.58 score route. |
| `external_public05917_direct_replay` | `closed_as_direct_route` | `Q3,S3` | `False` | The external Public-0.5917 notebook remains useful as validation philosophy, but it lacks a reproducible direct route and still leaves Q3/S3 uncovered. |

## Open or blocked routes

| route | status | score relevant | targets | reason |
|---|---|---|---|---|
| `explicit_approved_clean_upstream_id221_to_id226` | `open_but_requires_explicit_approval` | `True` | `Q2,Q3,S3` | Historical ID226-style upstream signal covers Q2/Q3/S3, while local post-ID341 lanes are closed. This is the only route that can move the missing targets without spending a DACON submission. |
| `local_summary_or_combo_cascade` | `closed` | `False` | `Q3,S3` | ID287 and ID329 showed that local-frontier/full-OOF or same-family combo candidates can look better locally while staying near or worse than the ID178 public anchor. |
| `external_direct_row_concatenation` | `blocked` | `False` | `Q3,S3` | External ETRI data is allowed as prior/context only; direct train-row concatenation remains blocked. |
| `manual_id341_public_sanity_probe` | `closed_after_id341_id340_id337_micro_probe_results` | `False` | `Q2` | The ID341/340/337 sanity-probe family has already been spent and remained near 0.60, so it should not be repeated. |

## Raw modality coverage

| modality | rows | columns | purpose |
|---|---:|---:|---|
| `mACStatus` | 939896 | 3 | charging/sleep-boundary state |
| `mActivity` | 961062 | 3 | phone activity state |
| `mAmbience` | 476577 | 3 | ambient semantic context |
| `mBle` | 21830 | 5 | nearby-device/social novelty |
| `mGps` | 800611 | 6 | place movement and route novelty |
| `mLight` | 96258 | 3 | phone-light sleep-boundary context |
| `mScreenStatus` | 939653 | 3 | screen-off/on transitions |
| `mUsageStats` | 45197 | 4 | app-use routine and category load |
| `mWifi` | 76336 | 4 | place/social novelty |
| `wHr` | 382918 | 3 | wearable physiology |
| `wLight` | 633741 | 3 | wearable light exposure |
| `wPedo` | 748100 | 9 | movement and step rhythm |

