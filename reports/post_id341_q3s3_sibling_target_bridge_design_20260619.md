# Post-ID341 Q3/S3 sibling-target bridge design

Created: `2026-06-19 00:50:08 UTC+09:00`

This is report-only. It trains no model and creates no predictions, submission artifacts, Kaggle runs, or DACON submissions.

## Decision

- Sibling bridge design ready: `True`
- Material sibling structure found: `True`
- Requires fold-nested rebuild: `True`
- Recommended route: `explicit_approved_clean_upstream_id221_to_id226`
- Requires explicit Kaggle approval: `True`
- Approval phrase: `Kaggle upstream rebuild no-submit 실행 승인`
- Public 0.58 evidence: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: Carry Q3<-Q2 and S3<-S2 sibling bridges only as fold-nested upstream design requirements. Do not create ID342 or blend sibling predictions locally from existing artifacts.

## Bridge rows

### q3_from_q2_subjective_state_sibling
- Target: `Q3`
- Sibling target: `Q2`
- Label correlation: `0.340129`
- Allowed use: `future_upstream_fold_nested_design_only`
- Blocked shortcut: Do not blend current Q2 predictions or Q2 labels into Q3 outside a fold-nested rebuild.

### s3_from_s2_objective_sleep_sibling
- Target: `S3`
- Sibling target: `S2`
- Label correlation: `0.394023`
- Allowed use: `future_upstream_fold_nested_design_only`
- Blocked shortcut: Do not blend current S2 predictions or S2 labels into S3 outside a fold-nested rebuild.

## Leakage controls

- Sibling target labels may not be used as direct features for the same row.
- Any sibling-target prediction feature must be generated out-of-fold for train rows and by a separately fitted test-time sibling model.
- Current full-OOF/local artifact recombinations are not sufficient evidence for ID342.
- A sibling bridge can only be evaluated inside the explicit-approved upstream rebuild or a future no-artifact fold-nested local smoke.

## Checks

- all_inputs_ready: `True`
- anatomy_ready: `True`
- q3_sibling_is_q2: `True`
- s3_sibling_is_s2: `True`
- q3_corr_is_material: `True`
- s3_corr_is_material: `True`
- local_q3s3_lanes_closed: `True`
- q3_s3_still_uncovered: `True`
- q2_context_only: `True`
- no_approval_score_route_closed: `True`
- operator_handoff_ready: `True`
- operator_requires_exact_approval: `True`

