# Circadian HR/light phase-shape metadata probe — post ID341

Created: `2026-06-19 09:47:52 UTC+09:00`

This is report-only. It creates no predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, model training job, or DACON submission.

## Decision

- Metadata probe passed: `True`
- Source family: `circadian_hr_light_phase_shape_v1`
- Target focus: `Q3, S3, S2`
- Strong phase-shape train/test rates: `0.5622` / `0.6120`
- Weak phase-shape train/test rates: `0.7489` / `0.8280`
- Next local allowed step: `report_only_feature_spec_no_model`
- Train model now: `False`
- Public `0.58` evidence: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: Circadian HR/light phase-shape metadata is feasible enough for a design spec, but this is not score evidence. Do not train or create ID342; only a report-only feature-spec step is allowed unless the exact Kaggle no-submit approval phrase is provided.

## Source coverage

| source | rows | subjects | subject-days | train coverage | test coverage | train >=18h | test >=18h | purpose |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `wHr` | 382918 | 10 | 636 | 0.8689 | 0.9800 | 0.0778 | 0.1160 | wearable autonomic/circadian recovery curve |
| `wLight` | 633741 | 10 | 664 | 0.9200 | 1.0000 | 0.7133 | 0.8120 | wearable light exposure phase and amplitude |
| `mLight` | 96258 | 10 | 700 | 1.0000 | 1.0000 | 0.9489 | 0.9480 | phone light context and dark-window continuity |
| `wPedo` | 748100 | 10 | 653 | 0.8978 | 0.9960 | 0.6778 | 0.7720 | movement phase and rest/activity rhythm |

## Phase feasibility

| split | expected days | strong days | weak days | strict days | strong rate | weak rate | light+pedo no-HR rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `test` | 250 | 153 | 207 | 142 | 0.6120 | 0.8280 | 0.7200 |
| `train` | 450 | 253 | 337 | 239 | 0.5622 | 0.7489 | 0.6533 |

## Heart-rate payload

- Nonempty HR payload rate: `1.0000`
- Mean/median readings per HR row: `52.211` / `59.000`

## Boundary

This report can justify only a report-only feature-spec step. It is not Public 0.58 evidence, does not open a submission artifact, and does not authorize Kaggle or DACON actions.

