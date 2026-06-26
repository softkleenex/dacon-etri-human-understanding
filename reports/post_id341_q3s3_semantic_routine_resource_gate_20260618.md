# Post-ID341 Q3/S3 semantic-routine resource gate

Created: `2026-06-23 18:07:40 UTC+09:00`

Decide whether the semantic-routine v2 lane may run a small local fold-internal model smoke without creating artifacts or overloading local memory.

This is report-only. It creates no features, predictions, CSVs, Kaggle runs, or DACON submissions.

## Decision

- Resource gate passed: `True`
- Local no-artifact model smoke allowed: `True`
- Target focus: `['Q3', 'S3']`
- Public 0.58 evidence: `False`
- Next local allowed step: `fold_internal_semantic_routine_model_smoke_only`
- Kaggle heavy execution recommended: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Next action: Local no-artifact fold-internal model smoke is allowed only under the listed streaming/width/grid constraints. Do not create ID342 or any submission artifact.

## Resource budget

- Total raw size: `122.832 MiB`
- Local soft limit: `256.0 MiB`
- Largest source: `mGps`

## Constraints

- stream_raw_sources_one_at_a_time: `True`
- build_daily_700_row_feature_frame_only: `True`
- do_not_export_raw_identifiers_or_coordinates: `True`
- do_not_one_hot_raw_app_wifi_ble_ids: `True`
- max_engineered_feature_columns: `64`
- max_model_smoke_targets: `2`
- max_model_specs: `2`
- max_blend_weights: `4`
- expected_daily_rows: `700`

## Checks

- metadata_smoke_ready: `True`
- raw_size_within_local_soft_limit: `True`
- streaming_constraints_required: `True`
- feature_width_limited: `True`
- model_grid_limited: `True`
- no_artifact_actions: `True`

