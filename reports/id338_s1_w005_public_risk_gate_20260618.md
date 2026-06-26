# ID338 S1 weight-0.05 public-risk gate

Candidate: `submission_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.csv`

## Summary

- Base: ID179 sequence anchor
- Change: S1 only, random-conv `nk512 seed45502`, `lr_c0_1`, weight `0.05`
- Local CV: `0.537861` vs ID179 `0.538290`
- Local gain vs base: `0.000429`
- Naive ID178-gap projected public: `0.600634`

## Movement / projection

- Mean Δ vs ID178: `0.004660`
- Max Δ vs ID178: `0.073989`
- Mean Δ vs ID179: `0.001316`
- Max Δ vs ID179: `0.035699`
- Failed-direction projection: `+0.000549`
- Rerank tier: `reviewable_low_movement`
- Conservative public-risk proxy: `0.606833`

## Identity check

- Non-S1 OOF exact vs ID179: `True`
- Non-S1 test exact vs ID179: `True`
- S1 test max Δ vs ID179: `0.035699`
- S1 test mean Δ vs ID179: `0.009210`

## ID337 comparison

- Local advantage vs ID337: `0.000205`
- Proxy disadvantage vs ID337: `0.001811`

## Decision

ID338 is a stronger-local but less-conservative sibling of ID337. It remains unsubmitted and is not Public `0.58` evidence.
