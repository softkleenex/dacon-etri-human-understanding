# ID340 sequence-summary candidate decision

Created: `2026-06-18 03:29:41 UTC+09:00`

Source probe: `reports/id340_sequence_summary_split_probe_20260618.json`

## Candidate design

- Name: `blend_id178_sequence_summary_lr_c0_3_w0.03`
- Base: `ID178`
- Source: sequence-summary split-trained logistic regression, `C=0.3`
- Fixed blend weight: `0.03`
- Artifact status: **not created**

## Split read

| split | primary | candidate loss | ID178 loss | Δ vs ID178 | pass |
|---|---|---:|---:|---:|---|
| `subject_hole_5` | `True` | 0.538346 | 0.538638 | -0.000292 | `True` |
| `subject_timeseries_5` | `False` | 0.540115 | 0.539747 | +0.000368 | `False` |
| `actual_test_band_holdout` | `True` | 0.540083 | 0.541604 | -0.001521 | `True` |
| `test_boundary_neighbor2_holdout` | `True` | 0.585580 | 0.586290 | -0.000709 | `True` |
| `test_boundary_neighbor5_holdout` | `True` | 0.557814 | 0.557986 | -0.000172 | `True` |
| `loso_10` | `True` | 0.538270 | 0.538638 | -0.000368 | `True` |

## Decision

- Primary gate pass: `True`
- Create ID340 now: **no**
- Submit now: **no**
- Promote full artifact contract now: **no**

This is the first credible ID340 construction lead after the transfer gates, but it is still only a report-only split probe. The gains are small and `subject_timeseries_5` is slightly worse than ID178, so the next required step is a full-train/test prediction geometry gate for this exact design before any CSV/notebook/log/base-pred artifact exists.
