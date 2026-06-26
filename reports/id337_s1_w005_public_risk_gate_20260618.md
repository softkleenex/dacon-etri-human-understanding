# ID337 S1 weight-0.05 public-risk gate

Candidate: `submission_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.csv`

## Summary

- Base: ID178 confirmed public anchor
- Change: S1 only, random-conv `nk512 seed45502`, `lr_c0_1`, weight `0.05`
- Local CV: `0.538066` vs ID178 `0.538638`
- Local gain: `0.000572`
- Naive ID178-gap projected public: `0.600839`

## Movement / projection

- Mean Δ vs ID178: `0.001336`
- Max Δ vs ID178: `0.037585`
- Failed-direction projection: `-0.000554`
- Rerank tier: `reviewable_low_movement`
- Conservative public-risk proxy: `0.605022`

## Identity check

- Non-S1 OOF exact vs ID178: `True`
- Non-S1 test exact vs ID178: `True`
- S1 test max Δ: `0.037585`
- S1 test mean Δ: `0.009352`

## Decision

ID337 is the movement-safe version of the ID336 S1 probe. It is cleaner by proxy than ID336 but has less local gain. It is not Public `0.58` evidence and no DACON submission was made.
