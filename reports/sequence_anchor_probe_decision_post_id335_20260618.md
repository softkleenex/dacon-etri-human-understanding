# Sequence anchor probe decision post-ID335

The post-ID335 S1 random-conv seed `nk512_seed45502` was replayed on ID178, ID179, ID190, and ID191 through the report-only Kaggle anchor probe.

## Anchor replay summary

| base | base mean | S1 best | S1 improvement | S1-only estimated mean | targetwise best mean |
|---:|---:|---|---:|---:|---:|
| 178 | 0.538638 | `blend_lr_c0_1_0.15` | 0.006820 | 0.537664 | 0.537628 |
| 179 | 0.538290 | `blend_lr_c0_1_0.15` | 0.005051 | 0.537568 | 0.537542 |
| 190 | 0.537956 | `blend_lr_c0_1_0.1` | 0.001385 | 0.537758 | 0.537723 |
| 191 | 0.538009 | `blend_lr_c0_1_0.1` | 0.002306 | 0.537680 | 0.537654 |

## Interpretation

- The signal is real enough to survive stronger anchors: S1 improves on all four replay bases.
- The cleanest public-anchor candidate is `ID178 + S1-only seed45502/lr_c0_1/weight0.15`, estimated local `0.537664`.
- The best local same-family candidate is `ID179 + S1-only seed45502/lr_c0_1/weight0.15`, estimated local `0.537568`.
- This is still not Public `0.58` evidence. It is a source-generation success and a candidate-construction lead.

## Decision

Do not auto-submit and do not promote yet. If we choose to promote, create one uniquely named S1-only artifact with a matched CSV/notebook/submission-log/report/base OOF/base test contract, then run movement/projection gates before asking for DACON submission approval.
