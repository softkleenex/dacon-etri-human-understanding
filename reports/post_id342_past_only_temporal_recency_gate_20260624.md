# Post-ID342 past-only temporal recency gate

Created: `2026-06-24 03:59:39 UTC+09:00`

This is report-only. It creates no submission artifacts, base predictions, Kaggle runs, or DACON submissions.

## OOF Summary

- Candidate loss: `0.534430`
- ID342 loss: `0.534000`
- Delta vs ID342: `+0.000430`
- Delta vs ID178: `-0.004207`
- Delta vs ID221: `+0.001440`
- Primary wins vs ID342: `0`
- Primary mean/max delta vs ID342: `+0.000130` / `+0.000425`
- Projected public proxy: `0.596513661`
- Gap to Public 0.55: `+0.046513661`

## Split Gate

| split | loss | delta vs ID342 | delta vs ID178 | delta vs ID221 |
|---|---:|---:|---:|---:|
| subject_hole_5 | 0.534075 | +0.000227 | -0.004416 | +0.001249 |
| actual_test_band_holdout | 0.537755 | +0.000000 | -0.003849 | +0.000096 |
| test_boundary_neighbor2_holdout | 0.577991 | +0.000000 | -0.008299 | +0.003783 |
| test_boundary_neighbor5_holdout | 0.554516 | +0.000000 | -0.003470 | -0.000265 |
| loso_10 | 0.538167 | +0.000425 | -0.004298 | +0.001496 |

## Selection Summary

| target | top selections | full-train/test selected |
|---|---|---|
| `Q1` | `base:17, past_k1_hl7_w0.01:1` | `base` |
| `Q2` | `base:13, past_k1_hl7_w0.01:3, past_k1_hl7_w0.03:2` | `past_k1_hl7_w0.01` |
| `Q3` | `base:17, past_k1_hl7_w0.01:1` | `base` |
| `S1` | `base:18` | `base` |
| `S2` | `base:9, past_k1_hl7_w0.03:4, past_k1_hl7_w0.01:3, past_k3_hl7_w0.03:1, past_k3_hl21_w0.03:1` | `past_k1_hl7_w0.01` |
| `S3` | `base:13, past_k1_hl7_w0.01:4, past_k1_hl7_w0.03:1` | `past_k1_hl7_w0.01` |
| `S4` | `base:16, past_k1_hl7_w0.03:1, past_k1_hl7_w0.01:1` | `base` |

## Decision

- Gate pass: `False`
- Failed checks: `primary_wins_vs_id342,primary_mean_delta_material,primary_worst_delta_not_positive,public_proxy_low_enough,public_055_gap_low_enough`
- Artifact review allowed: `False`
- Create submission artifact: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Submit now: `False`
- Reason: Past-only temporal recency does not clear the strict ID342/Public-0.55 gate.
