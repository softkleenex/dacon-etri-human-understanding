# ID340 sequence-summary split probe

Created: `2026-06-18 03:28:44 UTC+09:00`

This is report-only. It trains sequence-summary source models inside the frozen ID340 splits and tests fixed blends against ID178; it creates no submission CSV, base artifact, or DACON submission.

- Sequence rows: `809`
- Total channels: `68`
- Summary features: `2108`

## Best by split

| split | best | kind | loss | ID178 loss | Δ vs ID178 | pass |
|---|---|---|---:|---:|---:|---|
| `subject_hole_5` | `blend_id178_sequence_summary_lr_c0_3_w0.03` | `id178_fixed_blend` | 0.538346 | 0.538638 | -0.000292 | `True` |
| `subject_timeseries_5` | `blend_id178_sequence_summary_lr_c0_3_w0.01` | `id178_fixed_blend` | 0.539783 | 0.539747 | +0.000036 | `False` |
| `actual_test_band_holdout` | `blend_id178_sequence_summary_lr_c0_3_w0.07` | `id178_fixed_blend` | 0.539538 | 0.541604 | -0.002066 | `True` |
| `test_boundary_neighbor2_holdout` | `blend_id178_sequence_summary_lr_c0_3_w0.03` | `id178_fixed_blend` | 0.585580 | 0.586290 | -0.000709 | `True` |
| `test_boundary_neighbor5_holdout` | `blend_id178_sequence_summary_lr_c0_1_w0.03` | `id178_fixed_blend` | 0.557744 | 0.557986 | -0.000242 | `True` |
| `loso_10` | `blend_id178_sequence_summary_lr_c0_3_w0.03` | `id178_fixed_blend` | 0.538270 | 0.538638 | -0.000368 | `True` |

## Primary gate

- `subject_hole_5` beats ID178 reference: `True`
- `actual_test_band_holdout` beats ID178 reference: `True`
- `test_boundary_neighbor2_holdout` beats ID178 reference: `True`
- `test_boundary_neighbor5_holdout` beats ID178 reference: `True`
- `loso_10` beats ID178 reference: `True`

Stable best counts on primary splits:

- `blend_id178_sequence_summary_lr_c0_3_w0.03`: 3
- `blend_id178_sequence_summary_lr_c0_1_w0.03`: 1
- `blend_id178_sequence_summary_lr_c0_3_w0.07`: 1

## Top rows per split

### `subject_hole_5`

| rank | candidate | kind | mean loss | Δ vs ID178 |
|---:|---|---|---:|---:|
| 1 | `blend_id178_sequence_summary_lr_c0_3_w0.03` | `id178_fixed_blend` | 0.538346 | -0.000292 |
| 2 | `blend_id178_sequence_summary_lr_c0_1_w0.03` | `id178_fixed_blend` | 0.538394 | -0.000244 |
| 3 | `blend_id178_sequence_summary_lr_c0_3_w0.01` | `id178_fixed_blend` | 0.538442 | -0.000196 |
| 4 | `blend_id178_sequence_summary_lr_c0_1_w0.01` | `id178_fixed_blend` | 0.538478 | -0.000159 |
| 5 | `blend_id178_sequence_summary_lr_c0_1_w0.05` | `id178_fixed_blend` | 0.538580 | -0.000058 |
| 6 | `blend_id178_sequence_summary_lr_c0_3_w0.05` | `id178_fixed_blend` | 0.538587 | -0.000050 |
| 7 | `blend_id178_sequence_summary_hgb_l7_w0.01` | `id178_fixed_blend` | 0.538837 | +0.000199 |
| 8 | `blend_id178_sequence_summary_lr_c0_1_w0.07` | `id178_fixed_blend` | 0.539006 | +0.000369 |

### `subject_timeseries_5`

| rank | candidate | kind | mean loss | Δ vs ID178 |
|---:|---|---|---:|---:|
| 1 | `blend_id178_sequence_summary_lr_c0_3_w0.01` | `id178_fixed_blend` | 0.539783 | +0.000036 |
| 2 | `blend_id178_sequence_summary_lr_c0_1_w0.01` | `id178_fixed_blend` | 0.539817 | +0.000069 |
| 3 | `blend_id178_sequence_summary_hgb_l7_w0.01` | `id178_fixed_blend` | 0.539913 | +0.000165 |
| 4 | `blend_id178_sequence_summary_lr_c0_3_w0.03` | `id178_fixed_blend` | 0.540115 | +0.000368 |
| 5 | `blend_id178_sequence_summary_lr_c0_1_w0.03` | `id178_fixed_blend` | 0.540177 | +0.000429 |
| 6 | `blend_id178_sequence_summary_hgb_l7_w0.03` | `id178_fixed_blend` | 0.540349 | +0.000602 |
| 7 | `blend_id178_sequence_summary_lr_c0_3_w0.05` | `id178_fixed_blend` | 0.540769 | +0.001021 |
| 8 | `blend_id178_sequence_summary_lr_c0_1_w0.05` | `id178_fixed_blend` | 0.540809 | +0.001062 |

### `actual_test_band_holdout`

| rank | candidate | kind | mean loss | Δ vs ID178 |
|---:|---|---|---:|---:|
| 1 | `blend_id178_sequence_summary_lr_c0_3_w0.07` | `id178_fixed_blend` | 0.539538 | -0.002066 |
| 2 | `blend_id178_sequence_summary_lr_c0_3_w0.05` | `id178_fixed_blend` | 0.539628 | -0.001975 |
| 3 | `blend_id178_sequence_summary_lr_c0_3_w0.1` | `id178_fixed_blend` | 0.540008 | -0.001596 |
| 4 | `blend_id178_sequence_summary_lr_c0_3_w0.03` | `id178_fixed_blend` | 0.540083 | -0.001521 |
| 5 | `blend_id178_sequence_summary_lr_c0_1_w0.05` | `id178_fixed_blend` | 0.540322 | -0.001282 |
| 6 | `blend_id178_sequence_summary_lr_c0_1_w0.07` | `id178_fixed_blend` | 0.540409 | -0.001195 |
| 7 | `blend_id178_sequence_summary_lr_c0_1_w0.03` | `id178_fixed_blend` | 0.540545 | -0.001058 |
| 8 | `blend_id178_sequence_summary_lr_c0_3_w0.01` | `id178_fixed_blend` | 0.540965 | -0.000639 |

### `test_boundary_neighbor2_holdout`

| rank | candidate | kind | mean loss | Δ vs ID178 |
|---:|---|---|---:|---:|
| 1 | `blend_id178_sequence_summary_lr_c0_3_w0.03` | `id178_fixed_blend` | 0.585580 | -0.000709 |
| 2 | `blend_id178_sequence_summary_lr_c0_3_w0.05` | `id178_fixed_blend` | 0.585601 | -0.000689 |
| 3 | `blend_id178_sequence_summary_lr_c0_1_w0.03` | `id178_fixed_blend` | 0.585707 | -0.000583 |
| 4 | `blend_id178_sequence_summary_lr_c0_1_w0.05` | `id178_fixed_blend` | 0.585739 | -0.000551 |
| 5 | `blend_id178_sequence_summary_lr_c0_3_w0.01` | `id178_fixed_blend` | 0.585945 | -0.000345 |
| 6 | `blend_id178_sequence_summary_lr_c0_3_w0.07` | `id178_fixed_blend` | 0.585969 | -0.000320 |
| 7 | `blend_id178_sequence_summary_lr_c0_1_w0.01` | `id178_fixed_blend` | 0.586003 | -0.000287 |
| 8 | `blend_id178_sequence_summary_lr_c0_1_w0.07` | `id178_fixed_blend` | 0.586064 | -0.000226 |

### `test_boundary_neighbor5_holdout`

| rank | candidate | kind | mean loss | Δ vs ID178 |
|---:|---|---|---:|---:|
| 1 | `blend_id178_sequence_summary_lr_c0_1_w0.03` | `id178_fixed_blend` | 0.557744 | -0.000242 |
| 2 | `blend_id178_sequence_summary_lr_c0_1_w0.01` | `id178_fixed_blend` | 0.557808 | -0.000178 |
| 3 | `blend_id178_sequence_summary_lr_c0_3_w0.03` | `id178_fixed_blend` | 0.557814 | -0.000172 |
| 4 | `blend_id178_sequence_summary_lr_c0_3_w0.01` | `id178_fixed_blend` | 0.557820 | -0.000166 |
| 5 | `blend_id178_sequence_summary_hgb_l7_w0.01` | `id178_fixed_blend` | 0.557938 | -0.000049 |
| 6 | `blend_id178_sequence_summary_hgb_l7_w0.03` | `id178_fixed_blend` | 0.557957 | -0.000029 |
| 7 | `blend_id178_sequence_summary_lr_c0_1_w0.05` | `id178_fixed_blend` | 0.558013 | +0.000027 |
| 8 | `blend_id178_sequence_summary_hgb_l7_w0.05` | `id178_fixed_blend` | 0.558115 | +0.000129 |

### `loso_10`

| rank | candidate | kind | mean loss | Δ vs ID178 |
|---:|---|---|---:|---:|
| 1 | `blend_id178_sequence_summary_lr_c0_3_w0.03` | `id178_fixed_blend` | 0.538270 | -0.000368 |
| 2 | `blend_id178_sequence_summary_lr_c0_3_w0.01` | `id178_fixed_blend` | 0.538363 | -0.000274 |
| 3 | `blend_id178_sequence_summary_lr_c0_1_w0.01` | `id178_fixed_blend` | 0.538463 | -0.000175 |
| 4 | `blend_id178_sequence_summary_lr_c0_1_w0.03` | `id178_fixed_blend` | 0.538496 | -0.000142 |
| 5 | `blend_id178_sequence_summary_lr_c0_3_w0.05` | `id178_fixed_blend` | 0.538671 | +0.000033 |
| 6 | `blend_id178_sequence_summary_hgb_l7_w0.01` | `id178_fixed_blend` | 0.538764 | +0.000126 |
| 7 | `blend_id178_sequence_summary_lr_c0_1_w0.05` | `id178_fixed_blend` | 0.538946 | +0.000308 |
| 8 | `blend_id178_sequence_summary_hgb_l7_w0.03` | `id178_fixed_blend` | 0.539160 | +0.000522 |

## Decision

- Promote submission candidate: **no**
- Create ID340: **no**
- Any sequence candidate beats ID178 on all primary splits: `True`
- Reason: This is still a report-only split probe. Even if a fixed blend improves one split, ID340 requires stable gains across the frozen primary lenses plus public-risk geometry before artifact creation.

