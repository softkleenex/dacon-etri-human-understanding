# Post-ID341 new source blueprint

Created: `2026-06-18 04:26:23 UTC+09:00`

This is report-only. It trains no model, creates no submission artifact, and makes no DACON submission.

## Why this exists

The post-ID341 transfer audit blocked the current candidate set: local CV is a weak selector and no current row passes the Public `0.58` gate. The next useful work must be a genuinely new source family or a clean upstream rebuild, not more OOF recombination.

## Data coverage read

- Train labels: `[450, 10]`; sample/test rows: `[250, 10]`
- Processed train/test: `[450, 242]` / `[250, 242]`
- Processed feature columns: `232`
- Train sleep dates: `2024-06-04` to `2024-11-15`
- Test sleep dates: `2024-07-07` to `2024-11-20`

Processed feature group counts:

- `screen`: 17
- `activity`: 27
- `pedo`: 27
- `heart_rate`: 39
- `light`: 36
- `place`: 28
- `usage`: 27
- `demographic_calendar`: 12

Target prevalence in train:

- `Q1`: 0.495556
- `Q2`: 0.562222
- `Q3`: 0.600000
- `S1`: 0.682222
- `S2`: 0.651111
- `S3`: 0.662222
- `S4`: 0.560000

Raw sensor metadata:

| file | rows | role |
|---|---:|---|
| `data/raw/ch2025_data_items/ch2025_mACStatus.parquet` | 939896 | charge/bedside proxy |
| `data/raw/ch2025_data_items/ch2025_mActivity.parquet` | 961062 | stillness/activity transitions |
| `data/raw/ch2025_data_items/ch2025_mAmbience.parquet` | 476577 | ambient sound context |
| `data/raw/ch2025_data_items/ch2025_mBle.parquet` | 21830 | near-device/social-place context |
| `data/raw/ch2025_data_items/ch2025_mGps.parquet` | 800611 | mobility/place regularity |
| `data/raw/ch2025_data_items/ch2025_mLight.parquet` | 96258 | phone ambient light/darkness |
| `data/raw/ch2025_data_items/ch2025_mScreenStatus.parquet` | 939653 | screen on/off sleep-boundary proxy |
| `data/raw/ch2025_data_items/ch2025_mUsageStats.parquet` | 45197 | app-use behavior and evening load |
| `data/raw/ch2025_data_items/ch2025_mWifi.parquet` | 76336 | place/home-stability proxy |
| `data/raw/ch2025_data_items/ch2025_wHr.parquet` | 382918 | night HR/recovery/fragmentation proxy |
| `data/raw/ch2025_data_items/ch2025_wLight.parquet` | 633741 | watch light/darkness |
| `data/raw/ch2025_data_items/ch2025_wPedo.parquet` | 748100 | steps/inactivity/exertion |

## Prioritized new source families

### 1. `sleep_boundary_transition_v1`

Thesis: Infer sleep onset/wake boundary and sleep interruption structure from cross-modal transitions, rather than reusing daily aggregate OOF artifacts.

Targets: `S1, S2, S3, S4, Q1`

Why new: Processed features contain many daily aggregates and some slp_* summaries, but this lane explicitly models event boundaries and transitions with fold-internal per-subject baselines.

Candidate features:
- last screen-on before dark/still interval
- first screen-on after long still interval
- longest nightly still+dark+low-step segment
- charge-start/charge-end alignment with inferred sleep window
- HR dip timing and night HR recovery slope
- within-subject deviation of inferred bedtime/waketime/duration
- night interruption count from screen/activity/HR spikes

Validation contract:
- fit transition thresholds inside training folds only
- compare against ID178/ID341 on frozen ID340 split manifest
- no artifact unless it beats ID341 proxy and passes post-ID341 transfer audit

Compute: Kaggle CPU recommended after local metadata smoke; no GPU required

### 2. `evening_load_recovery_v1`

Thesis: Represent the day as load followed by recovery: daytime exertion/mobility/social/app load, then evening screen/light/HR recovery before sleep.

Targets: `Q2, Q3, S2, S4`

Why new: External/paper guidance says subjective fatigue/stress are daily-experience outcomes; this lane tests a causal-ish load→recovery representation instead of target-wise prior shrinkage.

Candidate features:
- daytime step and movement intensity percentiles
- evening-to-day load ratio
- late app/session entropy and dopamine-app buckets
- ambient speech/music/silence evening mix
- night HR minus day HR recovery delta
- post-heavy-day next-sleep interaction terms

Validation contract:
- group-safe feature normalization by subject inside folds
- single-family model first, no stacking
- slice check on high-activity and high-screen evenings

Compute: local light scan possible; Kaggle for repeated folds

### 3. `routine_stability_place_entropy_v1`

Thesis: Capture circadian regularity and place/social entropy across adjacent days; public split may reward stable subject-specific routine rather than local OOF frontier moves.

Targets: `Q1, Q2, Q3, S3`

Why new: Uses unlabeled temporal continuity and raw place/context logs; avoids direct external-target prevalence transfer.

Candidate features:
- home-stay consistency from Wi-Fi/GPS proxies
- BLE/Wi-Fi novelty and transition entropy
- weekday-adjusted app/screen routine distance
- rolling 3/7-day deviations from subject routine
- previous-day and next-day smoothness features available at inference without labels

Validation contract:
- chronological and subject-hole lenses both required
- no use of test-label-derived anchors
- must improve transfer proxy without moving far from ID178/ID341 unless source is independently strong

Compute: Kaggle CPU if raw event expansion becomes memory-heavy

### 4. `external_prior_regularizer_v2`

Thesis: Use older ETRI data only to regularize feature transformations and monotonic priors, not as labeled rows.

Targets: `Q2, S2`

Why new: Prior external attempts showed direct prevalence mismatch, especially S1/S3/S4; keep it as weak regularizer only.

Candidate features:
- sleep-score range sanity checks
- survey scale behavior priors
- feature bucket boundaries for duration/wake/activity only

Validation contract:
- never concatenate external rows into current train
- document every external-derived transform in submission notes
- drop if frozen split gate selects base for all targets again

Compute: local only

## Decision

- Top new source family: `sleep_boundary_transition_v1`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Next action: Implement a report-only Kaggle/local smoke for sleep_boundary_transition_v1 feature generation and frozen-split validation. Do not create a submission artifact unless it beats ID341 proxy under the post-ID341 transfer audit.

