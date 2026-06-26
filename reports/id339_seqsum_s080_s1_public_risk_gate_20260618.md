# ID339 sequence-summary scale-0.80 + S1 safety public-risk gate

Candidate: `submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv`

## Summary

- Base: ID178 confirmed public anchor
- Change: Q1/Q2/Q3 sequence-summary overlay at scale `0.80`; S1 from ID337; S2/S3/S4 unchanged from ID178
- Local CV: `0.536441` vs ID178 `0.538638`
- Local gain vs ID178: `0.002198`
- Naive ID178-gap projected public: `0.599214`

## Movement / projection

- Mean Δ vs ID178: `0.012602`
- Max Δ vs ID178: `0.097051`
- Mean Δ vs ID329: `0.023091`
- Max Δ vs ID329: `0.108981`
- Failed-direction projection: `-0.012001`
- Rerank tier: `reviewable_low_movement`
- Conservative public-risk proxy: `0.609547`

## Identity check

- S2/S3/S4 OOF exact vs ID178: `True`
- S2/S3/S4 test exact vs ID178: `True`
- S1 OOF exact vs ID337: `True`
- S1 test exact vs ID337: `True`

## ID337 comparison

- Local gain vs ID337: `0.001626`
- Proxy disadvantage vs ID337: `0.004525`

## Decision

ID339 is a valid full-contract generated-unsubmitted candidate, but its conservative proxy is worse than ID337 despite better local CV. Treat it as a reviewable candidate-construction lead, not the next automatic DACON submission.

No DACON submission was made. This is not Public `0.58` evidence.
