# Post-ID341 A1 qfree lane closure

Created: `2026-06-18 04:03:14 UTC+09:00`

This is report-only. It creates no CSV, notebook/history artifact, base prediction artifact, submission-log row, or DACON submission.

## Findings

- Full q240-free source replay survives: `False`
- Target salvage that survived replay: `['S4']`
- Direct ID235 S4 geometry pass: `False` (`reject_high_public_risk`)
- ID178-anchor S4 scan has artifact-review variant: `False`
- Best ID178-anchor S4 variant: `id178_s4_id239_prob_w0.0025`, local `0.538492`, proxy `0.603926` (+0.000096 vs ID341)

## Decision

- Close local A1 qfree artifact lane: **yes**
- Create artifact now: **no**
- Submit now: **no**
- Public 0.58-plausible candidate exists: **no**
- Best current control candidate: `ID341`
- Next local lane: new_source_family_or_validation_design, not more A1 qfree artifact recombination
- Next Kaggle lane if compute is spent: true upstream model-level rebuild of IDs 221/226/227/228/237 before replaying A1; do not use existing OOF artifacts as clean sources

A1 q240-free has one real S4 signal, but direct ID235 S4 salvage is high public-risk and ID178-anchored S4 micro variants fail to beat the ID341 proxy. Further local artifact recombination would preserve the same validation failure mode.
