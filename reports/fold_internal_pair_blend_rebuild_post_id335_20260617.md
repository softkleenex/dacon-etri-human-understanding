# Fold-internal pair/logit blend rebuild — post ID335

Created: `2026-06-17 23:04:04 UTC+09:00`

This is report-only. Pair/logit blend choices are learned inside each outer fold, but source artifacts are still existing OOF products.

## Results

| split | scenario | overall | robust | worst target | worst subject | source ids |
|---|---|---:|---:|---|---|---|
| `subject_hole_fold_internal` | `direct_no_252_cap015` | 0.446845 | 0.470270 | S4 `0.518466` | id10 `0.529188` | `[178, 254, 255, 256, 257, 258, 266]` |
| `subject_hole_fold_internal` | `direct_no_252_cap022` | 0.433848 | 0.457077 | S4 `0.503963` | id10 `0.516895` | `[178, 254, 255, 256, 257, 258, 266]` |
| `subject_hole_fold_internal` | `with_ultra252_cap022` | 0.424104 | 0.447935 | S4 `0.497309` | id10 `0.508142` | `[178, 252, 255, 257, 258]` |
| `subject_hole_fold_internal` | `with_ultra252_cap026` | 0.402530 | 0.425994 | S4 `0.471500` | id10 `0.488390` | `[178, 252, 255, 257, 258]` |
| `subject_hole_fold_internal` | `full_upstream_cap026` | 0.402530 | 0.425994 | S4 `0.471500` | id10 `0.488390` | `[178, 252, 254, 255, 256, 257, 258, 266]` |
| `subject_hole_fold_internal` | `full_upstream_cap030` | 0.399920 | 0.423760 | S3 `0.476569` | id10 `0.480614` | `[178, 252, 254, 255, 256, 257, 258, 266]` |
| `loso_fold_internal` | `direct_no_252_cap015` | 0.446523 | 0.469812 | S4 `0.517582` | id10 `0.529456` | `[178, 254, 255, 256, 257, 258, 266]` |
| `loso_fold_internal` | `direct_no_252_cap022` | 0.434096 | 0.457385 | S4 `0.503963` | id10 `0.517281` | `[178, 254, 255, 256, 257, 258, 266]` |
| `loso_fold_internal` | `with_ultra252_cap022` | 0.426157 | 0.450137 | S4 `0.501545` | id10 `0.506763` | `[178, 252, 255, 257, 258]` |
| `loso_fold_internal` | `with_ultra252_cap026` | 0.402860 | 0.426094 | S3 `0.471585` | id10 `0.486486` | `[178, 252, 255, 257, 258]` |
| `loso_fold_internal` | `full_upstream_cap026` | 0.402860 | 0.426094 | S3 `0.471585` | id10 `0.486486` | `[178, 252, 254, 255, 256, 257, 258, 266]` |
| `loso_fold_internal` | `full_upstream_cap030` | 0.400168 | 0.423989 | S3 `0.477011` | id10 `0.480951` | `[178, 252, 254, 255, 256, 257, 258, 266]` |

## Best rows

| split | best scenario | overall | robust | note |
|---|---|---:|---:|---|
| `subject_hole_fold_internal` | `full_upstream_cap030` | 0.399920 | 0.423760 | uses direct ID252 |
| `loso_fold_internal` | `full_upstream_cap030` | 0.400168 | 0.423989 | uses direct ID252 |

## Decision

Fold-internal pair/logit selection is a useful stress test, but it still reuses existing OOF source artifacts. Even if it scores well locally, it cannot authorize a CSV or DACON submission without rebuilding the source artifacts and calibration choices inside true model-level outer folds.
