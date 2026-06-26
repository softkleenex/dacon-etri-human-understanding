# Post-ID341 Q3/S3 semantic-routine v2 design

Created: `2026-06-23 18:07:39 UTC+09:00`

Define a materially different no-submit Q3/S3 source hypothesis after the physio-light-onset lane failed: semantic novelty, routine disruption, and ordered sleep-onset state transitions.

This is report-only. It defines the next Q3/S3 hypothesis without creating predictions, CSVs, Kaggle runs, or DACON submissions.

## Decision

- Design ready: `True`
- Target focus: `['Q3', 'S3']`
- Selected targets: `[]`
- Public 0.58 evidence: `False`
- Materially different from physio-light-onset: `True`
- External direct concat allowed: `False`
- Next local allowed step: `metadata_cardinality_and_state_transition_smoke_only`
- Create submission artifact: `False`
- Submit now: `False`
- Next action: Do not create ID342. If continuing locally, run only a small metadata/cardinality smoke for the semantic routine/state-machine features. Move to heavy feature extraction only with explicit Kaggle approval or after a separate no-artifact local resource check.

## Checks

- closed_physio_light_onset_lane_confirmed: `True`
- source_matrix_ready: `True`
- data_sources_ready: `True`
- materially_different_from_physio_light_onset: `True`
- external_direct_concat_blocked: `True`
- no_artifact_actions: `True`

## Raw schema coverage

- All required files present: `True`
- All required schemas readable: `True`
- Total raw rows across required files: `6122179`

## Feature blueprint

### Q3 — `semantic_app_social_place_novelty`

- Sources: `['mUsageStats', 'mWifi', 'mBle', 'mGps', 'mAmbience']`
- Why distinct: The closed physio-light-onset lane used physiological/light/onset aggregates. This family treats stress as social/context novelty and routine disruption, matching the paper hint that Q3 benefits from mUsageStats and wPedo-style daily-window context.
- Fail-fast rule: Report-only metadata smoke first; then fold-internal no-artifact model smoke only if feature coverage is complete. Stop unless Q3 beats ID341 in at least 4/5 primary splits with low joint-risk warnings.
- Feature shapes:
  - subject-relative evening app-category entropy and late-night category switch count
  - wifi/ble neighbor novelty count versus the subject's recent same-week baseline
  - gps place-cluster novelty and evening route-fragmentation proxy without storing raw coordinates
  - ambience state-change rate around the evening-to-late boundary
  - cross-modal novelty burst: app switch + new proximity device + place transition within a short window

### S3 — `sleep_onset_state_machine`

- Sources: `['mScreenStatus', 'mActivity', 'mACStatus', 'mLight', 'wLight', 'wPedo', 'wHr']`
- Why distinct: The closed lane summarized independent sensor levels. This one models ordered transitions and sustained states, which is closer to the sleep-onset-latency label mechanism.
- Fail-fast rule: Run metadata/cardinality smoke first. Stop unless S3 produces a low-movement fold-internal improvement; do not use Q1/S4 sleep-quality movement as a proxy for S3.
- Feature shapes:
  - first sustained quiet-dark-screen-off run after 20:00 and its duration
  - screen-off to charging-on latency and charging-state stability before sleep
  - last movement then first dark interval transition gap
  - state-machine transition counts among active/screen/charging/light buckets from 20:00 to 03:00
  - subject-relative onset regularity: deviation from each subject's typical quiet-window timing

