# ID178 sequence target rollback diagnostic

Created: `2026-06-17 21:59:13 UTC+09:00`

This is report-only. No submission CSV, notebook, base artifact, submission-log row, or DACON submission was created.

## Variant ranking

| rank | variant | OOF loss | stress mean | min stress improvement | max Δ178 | failed projection | verdict |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `rollback_s1_s4` | 0.540347 | 0.573438 | +0.001436 | 0.119120 | +0.0039 | reviewable_hypothesis_only |
| 2 | `rollback_s1` | 0.539474 | 0.574341 | +0.001850 | 0.111949 | +0.0024 | reviewable_hypothesis_only |
| 3 | `rollback_s4` | 0.539511 | 0.575349 | +0.000724 | 0.119120 | +0.0015 | reviewable_hypothesis_only |
| 4 | `id178_keep_all` | 0.538638 | 0.576252 | +0.001138 | 0.000000 | +0.0000 | confirmed_anchor_reference |

## Interpretation

The strongest diagnostic variant is `rollback_s1_s4`. It is more stress-consistent than the original ID178 overlay, but it is still a same-family sequence rollback, not a Public 0.58-proven candidate.

If this is ever promoted, it must receive a new immutable submission ID with matching CSV, notebook, submission-log row, metadata JSON, and base OOF/test artifacts. Do not reuse the ID178 filename.

