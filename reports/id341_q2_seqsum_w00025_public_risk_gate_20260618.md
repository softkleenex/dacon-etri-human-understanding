# ID341 Q2 sequence-summary w0.0025 public-risk gate

Created: `2026-06-18 03:53:21 UTC+09:00`

Candidate: `submissions/submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv`

## Summary

- Base: ID178 confirmed public anchor
- Change: Q2-only sequence-summary logistic source, fixed weight `0.0025`; all other targets unchanged from ID178
- Local Subject-Hole CV: `0.538584` vs ID178 `0.538638`
- Conservative public-risk proxy: `0.603830`
- Proxy Δ vs ID340: `-0.000232`
- Proxy Δ vs ID337: `-0.001192`
- Proxy Δ vs ID339: `-0.005717`
- Public `0.58` plausible: `False`

## Movement / projection

- Mean Δ vs ID178: `0.000124`
- Max Δ vs ID178: `0.002025`
- Mean Δ vs ID329: `0.019359`
- Max Δ vs ID329: `0.079192`
- Failed-direction projection: `-0.000085`
- Risk tier: `reviewable_low_movement`

## Split gate

| split | primary | loss | ID178 | Δ vs ID178 | pass |
|---|---|---:|---:|---:|---|
| `subject_hole_5` | `True` | 0.538584 | 0.538638 | -0.000054 | `True` |
| `subject_timeseries_5` | `False` | 0.539712 | 0.539747 | -0.000035 | `True` |
| `actual_test_band_holdout` | `True` | 0.541477 | 0.541604 | -0.000126 | `True` |
| `test_boundary_neighbor2_holdout` | `True` | 0.586181 | 0.586290 | -0.000109 | `True` |
| `test_boundary_neighbor5_holdout` | `True` | 0.557922 | 0.557986 | -0.000064 | `True` |
| `loso_10` | `True` | 0.538607 | 0.538638 | -0.000031 | `True` |

## Decision

- DACON submission: **no**
- Public 0.58 plausible: **no**
- Controlled low-movement probe candidate: **yes, explicit approval only**

ID341 is safer/lower-movement than ID340 by the conservative proxy, but its proxy remains around 0.604 and it is not Public 0.58 evidence.
