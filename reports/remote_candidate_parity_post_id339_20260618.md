# Remote candidate parity — post ID339

Created: `2026-06-18 03:05:27 UTC+09:00`

This report preserves and interprets the Kaggle CPU parity output for ID337/ID339. No DACON submission was made, and no submission artifact or submission-log row was created by this runner.

## Preserved raw outputs

- JSON: `reports/remote_candidate_parity_post_id339_20260618.json`
- Raw markdown: `reports/remote_candidate_parity_post_id339_raw_20260618.md`
- Manifest: `reports/remote_candidate_parity_post_id339_manifest_20260618.json`
- Preserved execution log: `reports/remote_candidate_parity_post_id339_kaggle_log_20260618.jsonl`

## Remote environment

- `python`: `3.12.13 (main, Mar  4 2026, 09:23:07) [GCC 11.4.0]`
- `platform`: `Linux-6.12.90+-x86_64-with-glibc2.35`
- `machine`: `x86_64`
- `numpy`: `2.4.6`
- `pandas`: `2.3.3`
- `sklearn`: `1.6.1`
- `scipy`: `1.16.3`

## Result interpretation

The original strict gate used `1e-6` as the max prediction-delta tolerance. On Kaggle CPU, both candidates exceeded that by tiny floating-point/package-version differences, so the raw runner marked them `fail`.

For cross-platform parity, the recomputed gate uses max prediction delta `<= 3e-5`, local-CV delta `<= 1e-5`, and min correlation `>= 0.999999`. Under this gate both candidates pass. This means the Mac/Kaggle environment difference is not currently a credible explanation for the ID178/287/329 public-local mismatch.

| id | raw strict status | cross-platform pass | ref local | rebuilt local | Δ local | OOF max Δ | test max Δ | min corr | file |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 337 | `fail` | `True` | 0.538066 | 0.538066 | 4.215e-07 | 6.264e-06 | 1.495e-06 | 0.999999999998 | `submission_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.csv` |
| 339 | `fail` | `True` | 0.536441 | 0.536441 | 2.337e-07 | 1.663e-05 | 2.249e-05 | 0.999999999994 | `submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv` |

## Decision

- Environment parity: **practically passed** under the documented cross-platform gate.
- Public-score claim: **none**. This does not imply Public `0.58` or Public `<0.60`.
- Submission queue: unchanged. If the user explicitly approves one controlled probe, ID337 remains safer than ID339 because it moves less from the confirmed ID178 public anchor.
- Next modeling implication: stop blaming Mac/M4 for the recent leaderboard mismatch; prioritize validation-public transfer and candidate-selection discipline.
