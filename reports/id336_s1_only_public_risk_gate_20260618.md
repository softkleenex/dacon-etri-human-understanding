# ID336 S1-only public-risk gate

Candidate: `submission_id_336_id178_s1_seed45502_randomconv_overlay_cv0.537663_unsubmitted.csv`

## Summary

- Base: ID178 confirmed public anchor
- Change: S1 only, random-conv `nk512 seed45502`, `lr_c0_1`, weight `0.15`
- Local CV: `0.537663` vs ID178 `0.538638`
- Local gain: `0.000974`
- S1 gain: `0.006820`
- Naive ID178-gap projected public: `0.600436`

## Movement / projection

- Family: `sequence_temporal_family`
- Mean Δ vs ID178: `0.004008`
- Max Δ vs ID178: `0.112756`
- Failed-direction projection: `-0.001662`
- Rerank tier: `reviewable_low_movement`
- Conservative public-risk proxy: `0.607945`

## Identity check

All non-S1 targets are exactly identical to ID178 in both OOF and test predictions. S1 test max Δ is `0.112756` and mean Δ is `0.028056`.

## Decision

ID336 is a reviewable controlled probe, not an automatic DACON submission. It is cleaner than broad post-ID287 combo candidates because it changes only S1 and has non-positive projection onto the ID287 failed direction. It is still not Public `0.58` evidence.
