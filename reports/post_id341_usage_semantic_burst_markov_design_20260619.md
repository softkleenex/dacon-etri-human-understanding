# Post-ID341 usage semantic burst/Markov design

Created: `2026-06-23 18:15:20 UTC+09:00`

This is report-only. It creates no feature matrix, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, model training job, or DACON submission.

## Decision

- Design ready: `True`
- Source family: `usage_semantic_burst_markov_v1`
- Target focus: `['Q2', 'Q3']`
- Public `0.58` evidence: `False`
- Feature matrix allowed now: `False`
- Train model now: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next local allowed step: `usage_semantic_burst_markov_metadata_cardinality_smoke_no_features`
- Next action: Do not create ID342. The next no-approval step may only be a metadata/cardinality smoke for session state counts and Markov state coverage; no feature parquet, model, Kaggle run, or DACON submission.

## Raw metadata

- Usage rows / total app events / unique app names: `45197` / `158426` / `586`
- Usage train/test coverage: `0.9778` / `1.0000`
- Screen rows / state-change rate: `939653` / `0.050733`
- Screen train/test coverage: `1.0000` / `1.0000`

## Blueprint

### `app_burst_sessionization`

- Target focus: `['Q2', 'Q3']`
- Sources: `['mUsageStats', 'mScreenStatus']`
- Measurements:
  - sessionized app bursts from usage-stat timestamp deltas, not daily total usage
  - late-evening and post-midnight reactivation bursts coupled with screen-on persistence
  - subject-relative deviation from recent same-subject burst timing and duration
- Anti-repeat rule: The failed phase3 usage-temporal lane used aggregate daily category totals. This design promotes only ordered burst/session shape and keeps raw app names out of model features.

### `semantic_app_category_markov_chain`

- Target focus: `['Q2', 'Q3']`
- Sources: `['mUsageStats']`
- Measurements:
  - category transition entropy and self-transition streaks within evening windows
  - rare transition motifs such as communication→finance/shop or media/search→system_home loops
  - Markov surprise versus each subject's rolling baseline transition matrix
- Leakage guard: Use only fixed hand-written app categories; no target-guided app vocabulary or raw app one-hot.

### `ambience_guarded_context_burst`

- Target focus: `['Q3']`
- Sources: `['mUsageStats', 'mAmbience']`
- Measurements:
  - speech/music/vehicle ambience flags around high-intensity app bursts
  - context disagreement: quiet ambience but high app-switching, or noisy ambience plus late reactivation
- Reopen guard: mAmbience is a guard/context source only because the ambience closure found full-OOF traps and no nested positive signal.

## Boundary

This design does not create ID342 and does not authorize a submission. Any future feature matrix or model smoke requires a separate no-artifact contract after metadata/cardinality coverage is checked.

