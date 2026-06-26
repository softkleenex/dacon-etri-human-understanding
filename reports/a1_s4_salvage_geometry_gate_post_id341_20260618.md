# A1 S4 salvage geometry gate — post ID341

Created: `2026-06-18 04:00:42 UTC+09:00`

This is report-only. It creates no CSV, notebook/history artifact, base prediction artifact, submission-log row, or DACON submission.

## Candidate design

- Name: `id235_s4_id239_prob_a055_salvage`
- Base: `ID235`
- Source: `ID239`
- Modified target: `S4` via `prob` alpha `0.55`

## Local context

- Base ID235 local: `0.482255`
- Candidate full-OOF local: `0.465443`
- Δ vs base: `-0.016812`
- Robust score: `0.487666`

## Public-risk geometry

- Mean Δ vs ID178: `0.119472`
- Max Δ vs ID178: `0.574855`
- Mean Δ vs ID329: `0.112790`
- Max Δ vs ID329: `0.529546`
- Failed-direction projection: `+0.521897`
- Risk tier: `reject_high_public_risk`

## Reference distances

| reference | mean abs Δ | max abs Δ | corr |
|---|---:|---:|---:|
| `id178` | 0.119472 | 0.574855 | 0.819209 |
| `id287` | 0.109332 | 0.572388 | 0.862388 |
| `id329` | 0.112790 | 0.529546 | 0.846610 |
| `id335` | 0.119006 | 0.574855 | 0.820840 |

## Decision

- Artifact review pass: `False`
- Create artifact now: **no**
- Submit now: **no**
- Public 0.58 plausible: **no**

S4 survives the q240-free source replay, but this gate must reject the candidate if it is too far from the confirmed ID178 public anchor or aligned with the failed ID287 direction.

