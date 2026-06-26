# Candidate parity check — post ID339

Created: `2026-06-18 03:04:28 UTC+09:00`

This is report-only. It creates no submission CSV, no submission-log row, no base prediction artifact, and no DACON submission.

## Environment

- `python`: `3.14.3 (v3.14.3:323c59a5e34, Feb  3 2026, 11:41:37) [Clang 16.0.0 (clang-1600.0.26.6)]`
- `platform`: `macOS-26.5.1-arm64-arm-64bit-Mach-O`
- `machine`: `arm64`
- `numpy`: `2.4.4`
- `pandas`: `3.0.2`
- `sklearn`: `1.8.0`
- `scipy`: `1.17.1`

## Candidate parity

| id | status | ref local | rebuilt local | Δ local | OOF max Δ | test max Δ | min corr | file |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 337 | `review_pass` / `cross_platform_review_pass` | 0.538066 | 0.538066 | +3.966e-07 | 1.110e-16 | 1.110e-16 | 1.000000000 | `submission_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.csv` |
| 339 | `review_pass` / `cross_platform_review_pass` | 0.536441 | 0.536441 | -2.810e-07 | 1.110e-16 | 1.110e-16 | 1.000000000 | `submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv` |

## Decision

- All candidates pass strict review gate: `True`
- All candidates pass cross-platform gate: `True`
- DACON submission: **no**

Use the strict review gate for same-environment rebuilds and the cross-platform gate for Kaggle CPU or other architecture/package-version parity checks. Passing parity does not imply public-LB gain.

