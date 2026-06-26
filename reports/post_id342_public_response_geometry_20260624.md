# Post-ID342 public-response geometry

Created: `2026-06-24 03:37:53 UTC+09:00`

This is report-only. It creates no submission artifacts, base predictions, Kaggle runs, or DACON submissions.

## Public Read

- Slate public order: `[342, 339, 338]`
- Slate local order: `[342, 339, 338]`
- Local/public order agrees in 2026-06-23 slate: `True`
- ID342 public gain vs ID337: `-0.004537377`
- ID342 public gain vs ID178: `-0.005328049`
- ID342 gap to Public 0.55: `+0.046083252`
- ID342 public-minus-local gap: `+0.062083246`
- ID342 gap compression vs ID178 gap: `-0.000690317`

## Candidate Rows

| id | local | public | gap | mean Δ178 | max Δ178 | target-mean L1 | entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 178 | 0.538638 | 0.601411301 | +0.062774 | 0.000000 | 0.000000 | 0.014594 | 0.562109 |
| 337 | 0.538066 | 0.600620629 | +0.062554 | 0.001336 | 0.037585 | 0.014337 | 0.562563 |
| 338 | 0.537861 | 0.599214518 | +0.061354 | 0.004660 | 0.073989 | 0.014399 | 0.563224 |
| 339 | 0.536441 | 0.598518779 | +0.062078 | 0.012602 | 0.097051 | 0.014192 | 0.566071 |
| 340 | 0.538426 | 0.601379175 | +0.062953 | 0.000494 | 0.008102 | 0.014623 | 0.562234 |
| 342 | 0.534000 | 0.596083252 | +0.062083 | 0.018463 | 0.093911 | 0.013728 | 0.566769 |

## ID342 Movement

- Dominant movement targets vs ID178: `['S4', 'Q1', 'Q2']`
- Best local-loss targets vs ID339: `['S4', 'Q1', 'Q3']`
- ID342 vs ID339 public/local deltas: `-0.002435527` / `-0.002440713`
- ID342 vs ID338 public/local deltas: `-0.003131266` / `-0.003860888`

## Decision

- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next research route: `id342_preserving_feature_similarity_source_decomposition`
- Next action: Do not spend DACON or Kaggle on the failed Q3/S3 raw/sequence branch. Decompose the ID221 feature-similarity source that transferred through ID342, then test only low-movement ID342-preserving variants with the same public/local order and joint-risk constraints.
