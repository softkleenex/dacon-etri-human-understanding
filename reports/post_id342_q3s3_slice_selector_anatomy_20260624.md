# Post-ID342 Q3/S3 slice-selector anatomy

Created: `2026-06-24 05:22:04 UTC+09:00`

This is report-only. It creates no predictions, submission artifacts, Kaggle runs, or DACON submissions.

## Selector Gate

| target | wins | mean delta | max delta | eligible |
|---|---:|---:|---:|---|
| `Q3` | 0 | 0.000063533 | 0.000317663 | `False` |
| `S3` | 0 | 0.000103967 | 0.000519836 | `False` |

## Stable Subject Leads

| target | subject | rows | best stable | delta vs ID342 | ID342 loss |
|---|---|---:|---:|---:|---:|
| `Q3` | `id03` | 33 | 178 | -0.015256353 | 0.552447 |
| `Q3` | `id02` | 48 | 338 | -0.008197936 | 0.478649 |
| `Q3` | `id08` | 56 | 178 | -0.007030515 | 0.513758 |
| `Q3` | `id10` | 33 | 339 | -0.005241182 | 0.544513 |
| `Q3` | `id04` | 57 | 339 | -0.000622164 | 0.540843 |
| `S3` | `id05` | 44 | 178 | -0.005737671 | 0.391154 |
| `S3` | `id07` | 49 | 178 | -0.005421402 | 0.469276 |

## Decision

- Stable selector selected targets: `[]`
- Stable subject lead rows: `7`
- Failed-frontier subject lead rows: `20`
- Projected Public proxy: `0.596083252`
- Gap to Public 0.55: `0.046083252`
- Kaggle now: `False`
- DACON submit now: `False`
- Next action: Stable public-scored donors help isolated subject/target slices but fail the fold-internal selector. Do not promote a selector or target-swap artifact; use this only to guide future representation work.
