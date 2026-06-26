# Random-conv sequence robustness diagnostic — post ID329

Created: `2026-06-17 21:59:11 UTC+09:00`

This is report-only. No submission CSV was created, no submission-log row was appended, and no DACON submission was made.

## Strategy summary

| strategy | folds | base mean | raw/base mean | selected blend mean | blend improvement |
|---|---:|---:|---:|---:|---:|
| subject_hole_5 | 5 | 0.545253 | 2.067284 | 0.541937 | +0.003316 |
| leave_one_subject_out_10 | 10 | 0.545253 | 2.389745 | 0.544115 | +0.001138 |
| test_boundary_neighbor_holdout | 1 | 0.645241 | 2.450830 | 0.642704 | +0.002537 |

## Verdict

Sequence signal shows some robustness, but this diagnostic still does not create a Public 0.58-plausible candidate.

## Interpretation

- If the selected ID178 sequence overlay only helps under the public-interleaved lens but not under subject-hole/LOSO/boundary stress, it should not be expanded into another public probe.
- A robust sequence branch should show target-level improvements outside the original selection lens before any new artifact is promoted.

