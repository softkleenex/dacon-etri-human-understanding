# Post-ID341 next lane decision

Created: `2026-06-18 03:55:59 UTC+09:00`

This is report-only. It creates no CSV, notebook, base predictions, submission-log row, or DACON submission.

## Current control artifact

- ID: `341`
- File: `submissions/submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv`
- Local CV: `0.538584`
- Conservative proxy: `0.603830`
- Read: Lowest-risk sequence-summary micro-control so far, but almost an ID178 clone and not Public 0.58 evidence.

## Submission decision

- Submit now: **no**
- Public `0.58`-plausible candidate exists: `False`
- Any DACON submission still requires explicit user approval for the exact CSV.
- If forced probe: ID341 is now the lowest-movement sequence-summary sanity probe; ID337 remains the S1-signal probe. Neither is a Public 0.58 claim.

## Modeling decision

- Next primary lane: `A1_q240_free_source_layer_rebuild`
- Next primary step: Rebuild or simulate ID238-like source choice and conservative ID238 calibration inside frozen outer folds before replaying cap0.30 bridge selection.
- Why: Selector-level A1 signal survives, but fold-internal calibration shows ID239/241-style full-OOF calibration collapses badly unless rebuilt more cleanly.

Blocked/avoid lanes:

- same-family full-OOF/local-CV cascades
- direct use of ID239/241 q240-free artifacts as clean sources
- Mac/M4 environment blame loop for current public-local gap
