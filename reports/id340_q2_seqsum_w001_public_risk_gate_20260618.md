# ID340 Q2 sequence-summary w0.01 public-risk gate

Created: `2026-06-18 03:42:03 UTC+09:00`

Candidate: `submissions/submission_id_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426_unsubmitted.csv`

## Summary

- Base: ID178 confirmed public anchor
- Change: Q2-only sequence-summary logistic source, fixed weight `0.01`; all other targets unchanged from ID178
- Local Subject-Hole CV: `0.538426` vs ID178 `0.538638`
- Conservative public-risk proxy: `0.604062`
- Proxy Δ vs ID337: `-0.000960`
- Proxy Δ vs ID339: `-0.005485`
- Public `0.58` plausible: `False`

## Movement / projection

- Mean Δ vs ID178: `0.000494`
- Max Δ vs ID178: `0.008102`
- Mean Δ vs ID329: `0.019345`
- Max Δ vs ID329: `0.079192`
- Failed-direction projection: `-0.000339`
- Risk tier: `reviewable_low_movement`

## Split gate

| split | primary | loss | ID178 | Δ vs ID178 | pass |
|---|---|---:|---:|---:|---|
| `subject_hole_5` | `True` | 0.538426 | 0.538638 | -0.000212 | `True` |
| `subject_timeseries_5` | `False` | 0.539612 | 0.539747 | -0.000135 | `True` |
| `actual_test_band_holdout` | `True` | 0.541105 | 0.541604 | -0.000498 | `True` |
| `test_boundary_neighbor2_holdout` | `True` | 0.585859 | 0.586290 | -0.000430 | `True` |
| `test_boundary_neighbor5_holdout` | `True` | 0.557736 | 0.557986 | -0.000250 | `True` |
| `loso_10` | `True` | 0.538520 | 0.538638 | -0.000117 | `True` |

## Decision

- DACON submission: **no**
- Public 0.58 plausible: **no**
- Controlled low-movement probe candidate: **yes, explicit approval only**

ID340 is safer by proxy than ID337/ID339 and moves only Q2 at very low weight, but the proxy remains around 0.604, so it is not Public 0.58 evidence.
