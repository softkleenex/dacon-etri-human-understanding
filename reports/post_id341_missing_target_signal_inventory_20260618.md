# Post-ID341 missing-target signal inventory

Created: `2026-06-23 18:07:16 UTC+09:00`

This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.

## Decision

- Focus targets: `Q2, Q3, S3`
- Focus targets missing locally: `Q2, Q3, S3`
- Focus targets with upstream backbone signal: `Q2, Q3, S3`
- Secondary watch-only targets: `S1, S2`
- Local Q2/Q3/S3 promotable: `False`
- External direct concat allowed: `False`
- Prelaunch ready for explicit approval: `True`
- Requires explicit user approval: `True`
- Create submission artifact: `False`
- Submit now: `False`

Do not create a local ID342. Q2/Q3/S3 have no current local selected rows; the only meaningful Q2/Q3/S3 surface is the report-first clean ID221→ID226 upstream rebuild, gated by explicit Kaggle approval.

## Target inventory

| target | local status | local rows | best local mean Δ341 | ID221 Δ178 | ID226 Δ178 | ID226 fold wins | action |
|---|---|---:|---:|---:|---:|---:|---|
| Q2 | `missing` | 0 |  | -0.006508 | -0.064994 | 23/23 | `route_to_clean_id221_then_id226_no_submit_upstream_rebuild` |
| Q3 | `missing` | 0 |  | -0.004859 | -0.051075 | 23/23 | `route_to_clean_id221_then_id226_no_submit_upstream_rebuild` |
| S3 | `missing` | 0 |  | -0.001404 | -0.025778 | 23/23 | `route_to_clean_id221_then_id226_no_submit_upstream_rebuild` |
| S1 | `watch_only` | 2 | -0.001482 | -0.004513 | -0.038218 | 23/23 | `do_not_promote_watch_only_signal_without_all_primary_confirmation` |
| S2 | `watch_only` | 2 | -0.002460 | -0.000676 | -0.052389 | 23/23 | `do_not_promote_watch_only_signal_without_all_primary_confirmation` |

## Evidence flags

- Data/source coverage ready: `True`
- All major sources present: `True`
- Local source stack closed for artifact: `True`
- Any local lane promotable: `False`

Interpretation: Q2/Q3/S3 are not absent because files or external data are missing; they are absent because the local post-ID341 source gates selected no transfer-safe rows. Historical upstream artifacts contain the missing target surface, but they must be rebuilt cleanly before artifact review.

