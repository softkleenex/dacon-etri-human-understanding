# Target-specific demographic gate — post ID335

Created: `2026-06-17 22:19:32 UTC+09:00`

This is report-only. It creates no model predictions, submission CSV, notebook, submission-log row, or DACON submission.

## Gate summary

| target | decision | subject-hole Δ demo | LOSO Δ demo | reason |
|---|---|---:|---:|---|
| Q1 | `reject_for_now` | -0.008620 | +0.003786 | demo does not win under LOSO stress |
| Q2 | `allow_research` | -0.005608 | -0.007770 | demo wins under both subject-hole and LOSO stress |
| Q3 | `watch_loso_only` | +0.001294 | -0.000742 | demo wins under LOSO but not subject-hole |
| S1 | `reject_for_now` | -0.002527 | +0.010705 | demo does not win under LOSO stress |
| S2 | `allow_research` | -0.020235 | -0.005097 | demo wins under both subject-hole and LOSO stress |
| S3 | `reject_for_now` | -0.013500 | +0.017640 | demo does not win under LOSO stress |
| S4 | `allow_research` | -0.021335 | -0.015023 | demo wins under both subject-hole and LOSO stress |

## Interpretation

- Only `allow_research` targets may be used in a future target-specific demographic branch.
- This gate is not promotion evidence by itself; any future demographic artifact still needs fold-safe OOF/test predictions and ID178/ID329/ID287 anchor geometry.
- Global demographic overlays remain frozen because LOSO did not support them.

