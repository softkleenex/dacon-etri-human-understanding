# Post-ID341 self-supervised subject-state embedding design

Created: `2026-06-23 18:17:39 UTC+09:00`

This is report-only. It reads metadata, not full raw streams, and creates no embedding, feature matrix, prediction file, submission artifact, Kaggle run, or DACON submission.

## Decision

- Design ready: `True`
- Source family: `self_supervised_subject_state_embedding_v1`
- Target focus: `Q2,Q3,S3`
- Materially new but heavy: `True`
- Local no-approval allowed: `False`
- Requires explicit heavy-compute approval: `True`
- Current score-relevant route: `explicit_approved_clean_upstream_id221_to_id226`
- Public `0.58` evidence: `False`
- Score route opened: `False`
- Feature matrix allowed now: `False`
- Train model now: `False`
- Create submission artifact: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Submit now: `False`
- Next local allowed step: `none_without_new_explicit_heavy_compute_contract`
- Next action: Keep this as a design-only route. If pursued later, first write a separate heavy-compute contract; otherwise the only score-relevant action remains the explicit-approved no-submit upstream rebuild.

## Metadata feasibility

- Raw modalities present: `12/12`
- Raw rows available: `6122179`
- Raw size MB: `122.832`
- Streams have subject/timestamp keys: `True`
- Official train/test rows: `450` / `250`
- Processed train/test rows: `450` / `250`

## Design stages

| stage | allowed now | output | guard |
|---|---|---|---|
| `metadata_only_current_step` | `True` | this report only | No full raw read, no embedding fit, no feature matrix, no model. |
| `approved_label_free_tokenization` | `False` | bounded event-token schema report only | Requires a new explicit heavy-compute contract; fixed vocabularies only, no target-guided tokens. |
| `approved_self_supervised_pretraining` | `False` | temporary embeddings under ignored build/artifact area only | Masked modality/time-bin reconstruction only; no target labels, public scores, or submission files. |
| `fold_internal_downstream_smoke` | `False` | report-only subject-hole fold losses | Outer validation rows must not influence embedding selection, target calibration, or blending. |
| `artifact_contract_review` | `False` | none unless Public-0.58 gates pass later | A separate decision note and exact user approval are required before any submission artifact. |

## Closed-lane distinctness

| closed lane | overlap risk | distinction | do not repeat |
|---|---|---|---|
| `usage_semantic_burst_markov_v1` | `medium` | Embedding would use synchronized multimodal day-state tokens, not app-state Markov counts alone. | Do not turn this into another app category/session aggregate scan. |
| `gps_path_topology_sequence_v1` | `medium` | GPS is only one channel in the state token; topology/path features are not the primary representation. | Do not replay route topology or dwell-segment ordering. |
| `circadian_hr_light_phase_shape_v1` | `medium` | Phase-shape used hand-built HR/light curves; embedding would learn cross-modal co-occurrence states. | Do not rerun circadian curve summaries with tiny model changes. |
| `proximity_ambience_cooccurrence_v1` | `medium` | Co-occurrence was fixed WiFi/BLE/ambience windows; embedding would tokenize all sensors with masked reconstruction. | Do not produce another proximity/ambience-only matrix. |
| `phone_wearable_discordance_v1` | `medium` | Discordance used fixed phone-vs-wearable rules; embedding would learn broader latent day states. | Do not build another deterministic discordance feature matrix. |
| `local_oof_combo_or_second_stage_calibration` | `critical` | This design is source-level and label-free before downstream folds, not a recombination of existing OOF predictions. | No full-OOF calibration, public-anchor blend, or ID287/ID329-style cascade. |

## Safety contract

```json
{
  "read_full_raw_streams_now": false,
  "fit_embedding_now": false,
  "write_feature_matrix_now": false,
  "train_model_now": false,
  "write_oof_or_test_predictions_now": false,
  "create_submission_artifact": false,
  "modify_submission_log": false,
  "push_kaggle_now": false,
  "run_kaggle_now": false,
  "submit_now": false,
  "requires_new_heavy_compute_contract_before_execution": true,
  "requires_new_artifact_contract_before_any_submission_file": true,
  "current_approval_phrase_only_applies_to_upstream_rebuild": true
}
```
