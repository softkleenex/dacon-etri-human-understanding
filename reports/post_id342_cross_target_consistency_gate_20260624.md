# Post-ID342 cross-target consistency gate

Created: `2026-06-24 04:04:11 UTC+09:00`

This is report-only. It creates no submission artifacts, base predictions, Kaggle runs, or DACON submissions.

## OOF Summary

- Candidate loss: `0.534113`
- ID342 loss: `0.534000`
- Delta vs ID342: `+0.000113`
- Delta vs ID178: `-0.004525`
- Delta vs ID221: `+0.001122`
- Primary wins vs ID342: `0`
- Primary mean/max delta vs ID342: `+0.000032` / `+0.000089`
- Projected public proxy: `0.596195808`
- Gap to Public 0.55: `+0.046195808`

## Split Gate

| split | loss | delta vs ID342 | delta vs ID178 | delta vs ID221 |
|---|---:|---:|---:|---:|
| subject_hole_5 | 0.533918 | +0.000070 | -0.004573 | +0.001092 |
| actual_test_band_holdout | 0.537755 | +0.000000 | -0.003849 | +0.000096 |
| test_boundary_neighbor2_holdout | 0.577991 | +0.000000 | -0.008299 | +0.003783 |
| test_boundary_neighbor5_holdout | 0.554516 | +0.000000 | -0.003470 | -0.000265 |
| loso_10 | 0.537830 | +0.000089 | -0.004635 | +0.001159 |

## Selection Summary

| target | top selections | full-train/test selected |
|---|---|---|
| `Q1` | `base:18` | `base` |
| `Q2` | `base:18` | `base` |
| `Q3` | `base:13, lr_c0.2_w0.05:3, lr_c0.2_w0.03:1, lr_c0.2_w0.01:1` | `lr_c0.2_w0.01` |
| `S1` | `base:18` | `base` |
| `S2` | `base:18` | `base` |
| `S3` | `base:18` | `base` |
| `S4` | `base:18` | `base` |

## Decision

- Gate pass: `False`
- Failed checks: `primary_wins_vs_id342,primary_mean_delta_material,primary_worst_delta_not_positive,public_proxy_low_enough,public_055_gap_low_enough`
- Artifact review allowed: `False`
- Create submission artifact: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Submit now: `False`
- Reason: Cross-target consistency does not clear the strict ID342/Public-0.55 gate.
