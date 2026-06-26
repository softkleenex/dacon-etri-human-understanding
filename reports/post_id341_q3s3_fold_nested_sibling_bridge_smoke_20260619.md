# Post-ID341 Q3/S3 fold-nested sibling bridge smoke

Created: `2026-06-23 18:15:19 UTC+09:00`

This is report-only. It creates no full-test predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.

## Leakage control

- Q3 uses only fold-nested predictions of Q2, not Q2 labels from the same row.
- S3 uses only fold-nested predictions of S2, not S2 labels from the same row.
- Inner OOF sibling predictions are used for outer training rows; sibling models fitted on outer training rows predict outer validation rows.

## Feature input

- Aligned train rows: `450`
- Feature columns used: `494`
- Mean missing rate: `0.0221`

## Selected target configs

| target | sibling | candidate | wins | mean Δ341 | max Δ341 |
|---|---|---|---:|---:|---:|
| _(none)_ |  |  |  |  |  |

## Top rows by target

### `Q3` from `Q2`

| rank | candidate | wins | mean Δ341 | max Δ341 |
|---:|---|---:|---:|---:|
| 1 | `blend_id178_nested_q3_from_q2_lr_c0_02_w0.005` | 0 | +0.000189 | +0.000328 |
| 2 | `blend_id341_nested_q3_from_q2_lr_c0_02_w0.005` | 0 | +0.000189 | +0.000328 |
| 3 | `blend_id178_nested_q3_from_q2_lr_c0_08_w0.005` | 0 | +0.000208 | +0.000368 |
| 4 | `blend_id341_nested_q3_from_q2_lr_c0_08_w0.005` | 0 | +0.000208 | +0.000368 |
| 5 | `blend_id178_nested_q3_from_q2_lr_c0_20_w0.005` | 0 | +0.000225 | +0.000399 |

### `S3` from `S2`

| rank | candidate | wins | mean Δ341 | max Δ341 |
|---:|---|---:|---:|---:|
| 1 | `blend_id178_nested_s3_from_s2_lr_c0_02_w0.005` | 0 | +0.000493 | +0.000822 |
| 2 | `blend_id341_nested_s3_from_s2_lr_c0_02_w0.005` | 0 | +0.000493 | +0.000822 |
| 3 | `blend_id178_nested_s3_from_s2_lr_c0_08_w0.005` | 0 | +0.000593 | +0.001003 |
| 4 | `blend_id341_nested_s3_from_s2_lr_c0_08_w0.005` | 0 | +0.000593 | +0.001003 |
| 5 | `blend_id178_nested_s3_from_s2_lr_c0_20_w0.005` | 0 | +0.000599 | +0.001006 |

## Decision

- Fold-nested sibling smoke completed: `True`
- Selected targets: `[]`
- Q3 signal found: `False`
- S3 signal found: `False`
- All Q3/S3 targets covered: `False`
- Best proxy: `0.601600`
- Beats ID178 public anchor proxy: `False`
- Public `0.58` evidence: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: If both Q3 and S3 are selected, feed this only into a no-artifact transfer/joint-risk gate. Otherwise keep sibling bridges as upstream design context and do not create ID342.

## Checks

- sibling_design_ready: `True`
- negative_map_closed: `True`
- features_aligned: `True`
- anchors_loaded: `True`

