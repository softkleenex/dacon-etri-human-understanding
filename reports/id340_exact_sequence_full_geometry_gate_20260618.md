# ID340 exact sequence-summary full geometry gate

Created: `2026-06-18 03:35:22 UTC+09:00`

This is report-only. It trains the exact source on the full train set and creates candidate test predictions in memory only. No CSV, base prediction artifact, notebook/history artifact, submission-log row, or DACON submission was created.

## Candidate design

- Name: `blend_id178_sequence_summary_lr_c0_3_w0.03`
- Base: `ID178`
- Source model: `{'name': 'lr_c0_3', 'kind': 'lr', 'c_value': 0.3, 'max_depth': None, 'max_leaf_nodes': None}`
- Fixed blend weight: `0.03`
- Modified targets: `['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']`

## Geometry

- Mean Δ vs ID178: `0.009405`
- Max Δ vs ID178: `0.028566`
- Mean Δ vs ID329: `0.020999`
- Max Δ vs ID329: `0.084887`
- Failed-direction projection: `-0.004776`
- Conservative public-risk proxy: `0.606723`
- Proxy Δ vs ID337: `+0.001701`
- Proxy Δ vs ID339: `-0.002824`
- Naive ID178-gap projected public: `0.601119`
- Risk tier: `reviewable_low_movement`

## Target movement vs ID178 test

| target | mean Δ | max Δ | source test mean | source test std |
|---|---:|---:|---:|---:|
| `Q1` | 0.009603 | 0.028306 | 0.511355 | 0.401708 |
| `Q2` | 0.010375 | 0.024305 | 0.535649 | 0.396321 |
| `Q3` | 0.010243 | 0.026011 | 0.601977 | 0.389821 |
| `S1` | 0.007941 | 0.026654 | 0.609064 | 0.394453 |
| `S2` | 0.009408 | 0.027638 | 0.618400 | 0.398256 |
| `S3` | 0.008953 | 0.028566 | 0.632494 | 0.365551 |
| `S4` | 0.009315 | 0.021408 | 0.612589 | 0.374687 |

## Split-probe context

| split | primary | candidate loss | ID178 loss | Δ vs ID178 | pass |
|---|---|---:|---:|---:|---|
| `subject_hole_5` | `True` | 0.538346 | 0.538638 | -0.000292 | `True` |
| `subject_timeseries_5` | `False` | 0.540115 | 0.539747 | +0.000368 | `False` |
| `actual_test_band_holdout` | `True` | 0.540083 | 0.541604 | -0.001521 | `True` |
| `test_boundary_neighbor2_holdout` | `True` | 0.585580 | 0.586290 | -0.000709 | `True` |
| `test_boundary_neighbor5_holdout` | `True` | 0.557814 | 0.557986 | -0.000172 | `True` |
| `loso_10` | `True` | 0.538270 | 0.538638 | -0.000368 | `True` |

## Loss context warning

These losses blend ID178 OOF with full-train in-sample source predictions. They are diagnostic only and must not be used as local CV evidence.

## Decision

- Passes full geometry gate for artifact review: `False`
- Public `0.58` plausible: `False`
- Promote submission candidate now: **no**
- Create ID340 artifact now: **no**
- Submit now: **no**

The exact design is safe enough for artifact-review if movement/proxy are below ID337/ID339, but it remains report-only here. Public 0.58 requires stronger evidence than this proxy gate.

