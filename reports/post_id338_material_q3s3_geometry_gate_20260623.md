# ID338 material Q3/S3 geometry gate

Created: `2026-06-23 23:51:11 UTC+09:00`

This is report-only. It reconstructs the selected overlay in memory and writes no full-test predictions, submission CSV, base artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.

## Movement vs ID338

- Mean absolute movement, all targets: `0.00011334`
- Max absolute movement, all targets: `0.00365242`

| target | mean abs movement | max abs movement | mean shift | source | candidate |
|---|---:|---:|---:|---|---|
| `S3` | 0.00079336 | 0.00365242 | +0.00001734 | `phone_wearable_discordance` | `blend_id338_phone_wearable_discordance_lr_c0_20_w0.005` |

## Joint Deltas vs ID338

| metric | delta |
|---|---:|
| `target_mean_l1_to_train` | -0.00000248 |
| `target_mean_max_abs_to_train` | +0.00000000 |
| `pair11_l1_to_train` | +0.00000349 |
| `temporal_volatility_ratio_to_train` | -0.00016171 |
| `prediction_entropy` | +0.00004130 |
| `prediction_sharpness` | -0.00003428 |
| `extreme_rate_05_95` | +0.00000000 |
| `extreme_rate_01_99` | +0.00000000 |

## Decision

- Geometry gate completed: `True`
- Geometry gate pass: `True`
- Risk flags: `[]`
- Public `0.58` evidence: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Next local allowed step: `promotion_contract_review_for_id338_s3_micro_overlay`
- Next action: Gate is distribution-safe but too small for automatic promotion. A separate promotion contract would be required before creating CSV/base/notebook/log artifacts.

