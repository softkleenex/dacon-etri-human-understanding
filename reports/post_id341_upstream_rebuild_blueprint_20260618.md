# Post-ID341 upstream rebuild blueprint

Created: `2026-06-23 20:00:14 UTC+09:00`

This is report-only. It trains no model, creates no submission artifact, and makes no DACON submission.

## Read

Freeze the report-only Kaggle upstream rebuild contract after local OOF artifact lanes closed.

## Required file checks

| path | exists |
|---|---|
| `dataset-metadata.json` | `True` |
| `pyproject.toml` | `True` |
| `Makefile` | `True` |
| `reports/post_id341_family_priority_20260618.json` | `True` |
| `reports/post_id341_a1_qfree_lane_closure_20260618.json` | `True` |
| `reports/b1_pair_blend_source_dependency_audit_20260618.json` | `True` |
| `reports/id340_validation_split_manifest_20260618.json` | `True` |

## Rebuild phases

| phase | compute | purpose | must output |
|---|---|---|---|
| `environment_and_payload_smoke` | Kaggle CPU, no GPU/TPU | Unpack project payload, verify data/raw + artifacts/base_predictions + frozen reports exist. | `payload_manifest.json, environment_versions.json` |
| `frozen_split_reconstruction` | CPU | Load id340_validation_split_manifest and verify fold row counts, subject coverage, and target prevalence. | `frozen_split_audit.json, frozen_split_audit.md` |
| `upstream_source_model_rebuild` | CPU first; GPU disabled unless a later DL-specific parity note exists | Rebuild model-level approximations below IDs 221/226/227/228/237 inside frozen folds. Do not select/finalize using outer validation rows. | `source_rebuild_oof.parquet, source_rebuild_metrics.json, source_rebuild_metrics.md` |
| `a1_b1_replay_from_rebuilt_sources` | CPU | Replay A1 qfree and B1 pair/logit selectors using rebuilt OOF sources only. | `a1_b1_replay_metrics.json, a1_b1_replay_metrics.md` |
| `transfer_gate_only_no_artifact` | CPU | Compare any rebuilt-source candidate against ID178/287/329/335/341 geometry and ID341 proxy. | `transfer_gate_post_rebuild.json, transfer_gate_post_rebuild.md` |

## Source records

| layer | ids |
|---|---|
| `upstream_model_layer` | `221, 226, 227, 228, 237` |
| `a1_replay_layer` | `238, 239, 241` |
| `b1_blocked_layer` | `254, 255, 256, 257, 258, 266` |
| `public_anchor_layer` | `178, 287, 329, 335, 341` |

## Hard gates

- No DACON submission from the Kaggle runner.
- No generated submission CSV or submission_log row from the Kaggle runner.
- No full-OOF calibration or target selector may use the same rows for fit/select/score.
- A rebuilt candidate must beat ID341 proxy 0.6038297835 before local artifact review.
- Any promoted artifact after pullback requires unique ID > 341 and the full artifact contract.

## Decision

- Blueprint ready: `True`
- Create submission artifact: **no**
- Submit now: **no**
- Push Kaggle now: **no**
- Planned notebook dir: `notebooks/kaggle_upstream_rebuild_runner`
- Planned slug: `softkleenex/dacon-etri-upstream-rebuild-post-id341`

Create a report-only Kaggle runner scaffold for this blueprint, validate pre-flight locally, then ask before any kernel push/run if a long rebuild will be launched.

