# Q3/S3 event-cadence smoke — post ID341

Created: `2026-06-23 18:11:26 UTC+09:00`

This is report-only. It uses raw timestamp cadence/silence-gap features and creates no submission artifacts.

## Decision

- Model smoke completed: `True`
- Selected targets: `[]`
- Q3 signal found: `False`
- S3 signal found: `False`
- All Q3/S3 targets covered: `False`
- Public 0.58 evidence: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Next local allowed step: `close_event_cadence_lane`

## Feature coverage

- Rows: `700` train/test = `450/250`
- Raw feature columns: `336`
- Feature columns incl. missing indicators: `672`
- Train any non-null row share: `1.000`
- Test any non-null row share: `1.000`
- Train missing mean: `0.0518`
- Test missing mean: `0.0421`

## Top rows by target

### Q3

| candidate | wins | mean Δ vs ID341 | max Δ vs ID341 |
|---|---:|---:|---:|
| `blend_id178_event_cadence_lr_c0_08_w0.005` | 2 | +0.000213 | +0.000526 |
| `blend_id341_event_cadence_lr_c0_08_w0.005` | 2 | +0.000213 | +0.000526 |
| `blend_id178_event_cadence_lr_c0_20_w0.005` | 2 | +0.000254 | +0.000652 |
| `blend_id341_event_cadence_lr_c0_20_w0.005` | 2 | +0.000254 | +0.000652 |
| `blend_id178_event_cadence_lr_c0_08_w0.01` | 2 | +0.000440 | +0.001061 |
| `blend_id341_event_cadence_lr_c0_08_w0.01` | 2 | +0.000440 | +0.001061 |

### S3

| candidate | wins | mean Δ vs ID341 | max Δ vs ID341 |
|---|---:|---:|---:|
| `blend_id178_event_cadence_lr_c0_20_w0.005` | 1 | +0.000026 | +0.000151 |
| `blend_id341_event_cadence_lr_c0_20_w0.005` | 1 | +0.000026 | +0.000151 |
| `blend_id178_event_cadence_lr_c0_20_w0.01` | 1 | +0.000071 | +0.000334 |
| `blend_id341_event_cadence_lr_c0_20_w0.01` | 1 | +0.000071 | +0.000334 |
| `blend_id178_event_cadence_lr_c0_08_w0.005` | 1 | +0.000114 | +0.000229 |
| `blend_id341_event_cadence_lr_c0_08_w0.005` | 1 | +0.000114 | +0.000229 |

## Next action

If selected targets exist, keep event-cadence as report-only context and run only a combined no-artifact geometry gate. If no selected targets exist, close this local lane and return to the explicit-approved upstream rebuild path.
