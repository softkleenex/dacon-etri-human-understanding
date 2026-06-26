# Candidate parity check — post ID339

Created: `2026-06-18 02:58:51 UTC+09:00`

This is report-only. It creates no submission CSV, no submission-log row, no base prediction artifact, and no DACON submission.

## Environment

- `python`: `3.12.13 (main, Mar  4 2026, 09:23:07) [GCC 11.4.0]`
- `platform`: `Linux-6.12.90+-x86_64-with-glibc2.35`
- `machine`: `x86_64`
- `numpy`: `2.4.6`
- `pandas`: `2.3.3`
- `sklearn`: `1.6.1`
- `scipy`: `1.16.3`

## Candidate parity

| id | status | ref local | rebuilt local | Δ local | OOF max Δ | test max Δ | min corr | file |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 337 | `fail` | 0.538066 | 0.538066 | +4.215e-07 | 6.264e-06 | 1.495e-06 | 1.000000000 | `submission_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.csv` |
| 339 | `fail` | 0.536441 | 0.536441 | -2.337e-07 | 1.663e-05 | 2.249e-05 | 1.000000000 | `submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv` |

## Decision

- All candidates pass review gate: `False`
- DACON submission: **no**

Run this same diagnostic on Kaggle CPU before treating ID337 or ID339 as an environment-independent public probe candidate.

