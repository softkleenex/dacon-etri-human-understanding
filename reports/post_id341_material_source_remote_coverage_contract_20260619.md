# Post-ID341 material source remote coverage contract

Created: `2026-06-23 18:17:39 UTC+09:00`

This is report-only. It creates no model, prediction, submission CSV, Kaggle run, or DACON submission.

## Decision

- Coverage contract pass: `True`
- All queue items covered: `True`
- Only score-relevant open route: `explicit_approved_clean_upstream_id221_to_id226`
- Requires explicit Kaggle approval: `True`
- Approval phrase: `Kaggle upstream rebuild no-submit 실행 승인`
- Create submission artifact: `False`
- Submit now: `False`
- Next action: Coverage is complete; wait for exact Kaggle no-submit approval or continue report-only source design.

## Coverage rows

| source | mode | covered | reason |
|---|---|:---:|---|
| `approved_clean_upstream_id221_to_id226` | `explicit_approval_remote_score_route` | `True` | Score-relevant route is open only behind the exact Kaggle no-submit approval phrase. |
| `gps_path_topology_sequence_v1` | `local_closed_evidence_only` | `True` | Local validation/model-smoke evidence closed the lane; remote runner coverage is intentionally not required. |
| `circadian_hr_light_phase_shape_v1` | `local_closed_evidence_only` | `True` | Local validation/model-smoke evidence closed the lane; remote runner coverage is intentionally not required. |
| `usage_semantic_burst_markov_v1` | `remote_required_report_only_gate` | `True` | This source family must be regenerated/staged by the no-submit runner as report-only evidence. |
| `proximity_ambience_cooccurrence_v1` | `local_closed_evidence_only` | `True` | Local validation/model-smoke evidence closed the lane; remote runner coverage is intentionally not required. |
| `phone_wearable_discordance_v1` | `local_closed_evidence_only` | `True` | Local validation/model-smoke evidence closed the lane; remote runner coverage is intentionally not required. |
| `self_supervised_subject_state_embedding_v1` | `remote_required_report_only_gate` | `True` | This source family must be regenerated/staged by the no-submit runner as report-only evidence. |
| `sleep_proxy_threshold_residual_v1` | `local_closed_evidence_only` | `True` | Local validation/model-smoke evidence closed the lane; remote runner coverage is intentionally not required. |

