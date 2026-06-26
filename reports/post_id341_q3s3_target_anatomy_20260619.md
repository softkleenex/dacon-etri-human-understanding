# Post-ID341 Q3/S3 target anatomy

Created: `2026-06-19 00:44:30 UTC+09:00`

This is report-only. It trains no model and creates no predictions, submission artifacts, Kaggle runs, or DACON submissions.

## Decision

- Q3/S3 target anatomy ready: `True`
- Current local sources still closed: `True`
- Q3/S3 still uncovered: `True`
- Public 0.58 evidence: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: Use this anatomy only to screen future Q3/S3 hypotheses. Do not create ID342 from label-only or closed-lane evidence; the score-relevant route remains the explicit-approved upstream rebuild.

## Label read

- Rows: `450`, subjects: `10`
- Q3 positive rate: `0.600000`; lag-1 agreement: `0.652273`
- S3 positive rate: `0.662222`; lag-1 agreement: `0.656818`
- Q3 top same-day correlation: `Q2` corr `0.340129`
- S3 top same-day correlation: `S2` corr `0.394023`

## Target summaries

### Q3
- Positive rate: `0.600000`
- Constant logloss floor: `0.673012`
- Subject positive-rate range: `0.332905`
- Lag-1 agreement: `0.652273`

### S3
- Positive rate: `0.662222`
- Constant logloss floor: `0.639550`
- Subject positive-rate range: `0.821970`
- Lag-1 agreement: `0.656818`

## Evidence checks

- train_file_exists: `True`
- required_columns_present: `True`
- row_count_is_450: `True`
- subject_count_is_10: `True`
- q3s3_negative_map_closed: `True`
- fallback_keeps_q2_context_only: `True`
- frontier_has_no_no_approval_score_route: `True`

