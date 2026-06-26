# Post-A1 sequence pivot priority

Created: `2026-06-18 02:06:12 UTC+09:00`

This is report-only. No CSV, notebook, base predictions, submission-log row, or DACON submission was created.

## Sequence-summary anchor replay

| rank | base | base local | best local | gain | modified targets | max test Δ vs base |
|---:|---:|---:|---:|---:|---|---:|
| 1 | 190 | 0.537956 | 0.536326 | +0.001629 | `['Q1', 'Q2', 'Q3']` | 0.121314 |
| 2 | 191 | 0.538009 | 0.536429 | +0.001581 | `['Q1', 'Q2', 'Q3']` | 0.121314 |
| 3 | 179 | 0.538290 | 0.536647 | +0.001643 | `['Q1', 'Q2', 'Q3', 'S1']` | 0.121314 |
| 4 | 178 | 0.538638 | 0.536786 | +0.001852 | `['Q1', 'Q2', 'Q3', 'S1']` | 0.121314 |

## ID178 target improvements

| target | best | improvement | max test Δ vs ID178 base |
|---|---|---:|---:|
| `Q1` | `blend_lr_c0_3_0.1` | +0.002872 | 0.094315 |
| `Q2` | `blend_lr_c0_3_0.15` | +0.005991 | 0.121314 |
| `Q3` | `blend_lr_c0_1_0.1` | +0.002544 | 0.083883 |
| `S1` | `blend_lr_c0_3_0.05` | +0.001556 | 0.042166 |
| `S2` | `base` | +0.000000 | 0.000000 |
| `S3` | `base` | +0.000000 | 0.000000 |
| `S4` | `base` | +0.000000 | 0.000000 |

## Combined candidate estimate

- Name: `id178_q123_sequence_summary_plus_id337_s1_safety_estimate`
- Estimated local: `0.536437`
- Modified targets: `['Q1', 'Q2', 'Q3', 'S1']`
- Caution: This is only an additive estimate from existing diagnostics. It is not a generated artifact and has no test-geometry/projection gate yet.

## Recommendations

### 1. `sequence_summary_anchor_gate`

- Read: Sequence-summary embeddings survive on ID178/179/190/191 after A1 failed. This is now the best non-calibration modeling lane, but it needs a movement/projection gate before artifact promotion.
- Next: Build a report-only geometry gate for an ID178 Q1/Q2/Q3 sequence-summary overlay and the ID178 Q1/Q2/Q3 + ID337-S1 combined estimate before creating ID339.

### 2. `id337_controlled_public_probe_hold`

- Read: ID337 remains the safest exact public probe, but it is not validation repair and needs exact approval.
- Next: Keep as explicit-approval-only; do not submit automatically.

### 3. `a1_q240_free_deprioritized`

- Read: The q240-free A1 stack does not preserve enough signal after outer-fold source replay. Deprioritize this lane before any Kaggle-heavy model-level rebuild.
- Next: Do not spend heavy Kaggle/model-level compute on this exact A1 q240-free stack.

## Decision

After A1 q240-free collapsed, sequence-summary embeddings are the strongest surviving non-calibration signal. They improve all four checked anchors, but broad target movement means a geometry gate is required before promoting any ID339 artifact.

