# Q3/S3 cross-modal disagreement smoke — post ID341

Created: `2026-06-18 23:47:17 UTC+09:00`

This is report-only. It uses materialized processed features to test disagreement between phone/watch signals around the sleep boundary and creates no submission artifacts.

## Decision

- Model smoke completed: `True`
- Selected targets: `[]`
- Q3 signal found: `False`
- S3 signal found: `False`
- All Q3/S3 targets covered: `False`
- Public 0.58 evidence: `False`
- Cross-modal artifact allowed: `False`
- Negative-map extension recommended: `True`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next local allowed step: `extend_negative_map_as_closed_scout`

## Feature coverage

- Rows: `700` train/test = `450/250`
- Raw feature columns: `42`
- Feature columns incl. missing indicators: `84`
- Ratio columns used: `23`
- Continuous columns used: `9`
- Train any non-null row share: `1.000`
- Test any non-null row share: `1.000`
- Train missing mean: `0.1469`
- Test missing mean: `0.1371`

## Top rows by target

### Q3

| candidate | wins | mean Δ vs ID341 | max Δ vs ID341 |
|---|---:|---:|---:|
| `blend_id178_cross_modal_disagreement_lr_c0_02_w0.005` | 0 | +0.000173 | +0.000459 |
| `blend_id341_cross_modal_disagreement_lr_c0_02_w0.005` | 0 | +0.000173 | +0.000459 |
| `blend_id178_cross_modal_disagreement_lr_c0_08_w0.005` | 0 | +0.000231 | +0.000493 |
| `blend_id341_cross_modal_disagreement_lr_c0_08_w0.005` | 0 | +0.000231 | +0.000493 |
| `blend_id178_cross_modal_disagreement_lr_c0_20_w0.005` | 0 | +0.000277 | +0.000523 |
| `blend_id341_cross_modal_disagreement_lr_c0_20_w0.005` | 0 | +0.000277 | +0.000523 |

### S3

| candidate | wins | mean Δ vs ID341 | max Δ vs ID341 |
|---|---:|---:|---:|
| `blend_id178_cross_modal_disagreement_lr_c0_20_w0.005` | 1 | +0.000109 | +0.000220 |
| `blend_id341_cross_modal_disagreement_lr_c0_20_w0.005` | 1 | +0.000109 | +0.000220 |
| `blend_id178_cross_modal_disagreement_lr_c0_08_w0.005` | 1 | +0.000112 | +0.000220 |
| `blend_id341_cross_modal_disagreement_lr_c0_08_w0.005` | 1 | +0.000112 | +0.000220 |
| `blend_id178_cross_modal_disagreement_lr_c0_02_w0.005` | 1 | +0.000164 | +0.000294 |
| `blend_id341_cross_modal_disagreement_lr_c0_02_w0.005` | 1 | +0.000164 | +0.000294 |

## Next action

If selected targets exist, keep this as report-only context and run only a combined no-artifact geometry gate. If no selected targets exist, add it to the Q3/S3 negative map as a closed scout.

