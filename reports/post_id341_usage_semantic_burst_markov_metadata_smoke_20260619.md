# Post-ID341 usage semantic burst/Markov metadata smoke

Created: `2026-06-23 18:15:21 UTC+09:00`

This is report-only. It creates no feature matrix, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, model training job, or DACON submission.

## Decision

- Metadata smoke passed: `True`
- Source family: `usage_semantic_burst_markov_v1`
- Target focus: `['Q2', 'Q3']`
- Public `0.58` evidence: `False`
- Feature matrix allowed now: `False`
- Train model now: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next local allowed step: `report_only_usage_semantic_burst_markov_feature_spec_no_matrix`
- Next action: A future no-approval step may only write a report-only feature spec with exact predeclared states and leakage guards. Do not generate a feature matrix, train a model, create ID342, run Kaggle, or submit.

## Coverage/cardinality

- Usage train/test coverage: `0.9778` / `1.0000`
- Dominant app-category states used: `7` / `7`
- Valid Markov transitions / unique pairs / pair coverage: `44372` / `48` / `0.9796`
- Evening transitions / evening unique pairs: `12945` / `47`
- Sessions / multi-category session rate: `4686` / `0.6135`
- Evening burst days / late reactivation days: `679` / `680`
- Screen train/test coverage: `1.0000` / `1.0000`
- Usage rows with screen match within 5min: `0.9985`

## Top transition pairs

- `other→other`: `15509`
- `system_home→system_home`: `2925`
- `other→system_home`: `2918`
- `system_home→other`: `2876`
- `communication→communication`: `2386`
- `routine_health→routine_health`: `2281`
- `communication→other`: `1880`
- `other→communication`: `1876`
- `media_search→media_search`: `1759`
- `other→media_search`: `1072`
- `media_search→other`: `1039`
- `communication→system_home`: `1001`

## Boundary

Passing this metadata smoke does not create ID342 and does not authorize feature generation. It only allows a later report-only feature spec with exact predeclared states, leakage guards, and no model.

