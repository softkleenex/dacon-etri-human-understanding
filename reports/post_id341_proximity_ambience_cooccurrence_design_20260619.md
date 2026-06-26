# Post-ID341 proximity/ambience co-occurrence design

Created: `2026-06-23 18:15:53 UTC+09:00`

This is report-only. It creates no features, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, model, or DACON submission.

## Decision

- Design ready: `True`
- Source family: `proximity_ambience_cooccurrence_v1`
- Target focus: `Q3,S3`
- Public `0.58` evidence: `False`
- Score route opened: `False`
- Feature matrix allowed now: `False`
- Train model now: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next local allowed step: `proximity_ambience_cooccurrence_metadata_smoke_no_features`
- Next action: Run only the report-only metadata/co-occurrence smoke next; do not write a feature matrix, train, create ID342, run Kaggle, or submit.

## Context status

- all_context_reports_present: `True`
- all_context_reports_action_safe: `True`
- no_approval_score_route_exists_false: `True`
- only_score_relevant_open_route: `explicit_approved_clean_upstream_id221_to_id226`
- prelaunch_ready_for_explicit_approval: `True`
- q3_s3_still_uncovered: `True`
- all_local_q3s3_lanes_closed: `True`
- usage_validation_stop_passed: `True`
- usage_stop_opened_no_score_route: `True`
- local_source_stack_closed: `True`

## Raw schema

| source | rows | columns | ready |
|---|---:|---|---|
| `mWifi` | `76336` | `subject_id,timestamp,bssid,rssi` | `True` |
| `mBle` | `21830` | `subject_id,timestamp,address,device_class,rssi` | `True` |
| `mAmbience` | `476577` | `subject_id,timestamp,element` | `True` |
| `mScreenStatus` | `939653` | `subject_id,timestamp,m_screen_use` | `True` |

## Blueprint

### proximity_signature_stability

- Sources: `mWifi,mBle`
- Target hypothesis: S3/Q3 may react to social/place density and late-day context shifts rather than raw GPS paths.
  - count-only WiFi/BLE scan richness by evening and late windows
  - hashed-never-persisted proximity signature repeat rate within subject/day
  - near-device burst density from RSSI thresholds, without storing raw network identifier values

### speech_crowd_inside_outside_context

- Sources: `mAmbience`
- Target hypothesis: S3 and Q3 can plausibly depend on social stimulation and environment recovery pressure.
  - speech/conversation/crowd scores as social context indicators
  - inside/outside and vehicle/music scores as coarse environment states
  - late-evening ambience volatility instead of daily aggregate ambience totals

### cross_sensor_cooccurrence

- Sources: `mWifi,mBle,mAmbience,mScreenStatus`
- Target hypothesis: This is materially different from closed GPS topology, usage Markov, and ambience-only lanes.
  - 10-minute-bin co-occurrence of proximity scans with speech/crowd ambience
  - screen-on persistence when proximity/social ambience is active
  - subject-relative novelty of proximity+ambience state sequences

## Material difference

- vs_gps_path_topology: Uses no coordinate/path topology; proximity IDs are only ephemeral count/signature inputs.
- vs_usage_semantic_burst_markov: Uses WiFi/BLE/social ambience co-occurrence, not app-category Markov chains.
- vs_phase3_ambience_context: Requires cross-sensor proximity+ambience timing, not ambience-only daily context.
- vs_event_cadence_or_dropout: Uses context co-occurrence intensity, not missingness or generic event cadence alone.

## Guards

- Do not persist raw WiFi/BLE identifiers or exact proximity token values in reports or feature matrices.
- Use predeclared ambience groups and RSSI thresholds; no target-guided vocabulary selection.
- Any later model smoke must fit preprocessing fold-internally and may only report split losses.
- No submission artifact, base prediction artifact, notebook/history artifact, or submission-log row is authorized.
