# Response to Reviewer

Thank you for the constructive review and the positive recommendation. We
revised the manuscript as follows. The revised PDF has not been uploaded to
EDAS automatically.

## 1. Public-leaderboard adaptation

We now state in the Introduction, Abstract, Results, and Limitations that the
anchor and operating policy were developed after observing earlier public
scores. The reported improvement is therefore leaderboard-adapted comparative
competition evidence, not an estimate on an untouched independent test set.
We also avoid attributing the public improvement to feature similarity alone.

## 2. Choice of alpha

Alpha `0.50` was chosen once as a conservative midpoint before ID342 was
publicly scored; it was not selected through a leaderboard sweep. We added the
following primary-OOF sensitivity analysis:

| alpha | Mean log loss | Folds improved over ID178 |
|---:|---:|---:|
| 0.00 | 0.538638 | 0/5 |
| 0.25 | 0.535813 | 5/5 |
| 0.50 | 0.534000 | 5/5 |
| 0.75 | 0.533074 | 4/5 |
| 1.00 | 0.532990 | 4/5 |

The full-OOF mean favors a larger alpha, while the midpoint retains improvement
in every fold and halves the source's movement away from the scored anchor. We
explicitly note that this is still selection-conditioned rather than
independent validation.

## 3. Similarity-grid size and fold dependence

The grid evaluated four feature families, three neighbor counts, three temporal
penalties, and six blend weights: `216` configurations per target and `1,512`
target/configuration evaluations in total. Selection was non-nested. Although
the selected setting improved the aggregate loss for all seven targets, it
improved only `21/35` target-fold losses relative to ID218. We added both the
search size and this evidence of material fold dependence to the manuscript.

## 4. Transductive preprocessing

We clarify that combined train/test imputation and standardization are
transductive and require the test batch in advance. The same method can operate
sequentially by fitting preprocessing only on historical training covariates.
In a train-only preprocessing check, ID221 primary-OOF loss changed from
`0.532990` to `0.533015`, a difference of `+0.000025`. This small local
difference is not independent deployment evidence, so the revised paper also
warns that future sequential distribution shifts may produce a larger gap.

## 5. Retrospective rank agreement

We strengthened the qualification around Table IV. The two exclusions use
known public outcomes; consequently, perfect rank agreement is a retrospective
diagnostic and neither independently validates the promotion rule nor provides
prospective validation.
