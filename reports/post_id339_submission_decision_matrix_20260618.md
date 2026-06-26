# Post-ID339 submission decision matrix

Created: `2026-06-18 02:27:29 UTC+09:00`

No DACON submission was made. This matrix ranks candidates for possible explicit user-approved probing, not automatic submission.

## Public context

| id | local | public | role |
|---:|---:|---:|---|
| `id178` | `0.538638` | `0.6014113009` | best recent confirmed public anchor |
| `id329` | `0.516718` | `0.6017908577` | same-family local gain failed to improve public |
| `id287` | `0.378934` | `0.6698787318` | catastrophic public collapse |

## Candidate comparison

| id | local | proxy | max Δ178 | proj failed | tier | read | file |
|---:|---:|---:|---:|---:|---|---|---|
| 337 | 0.538066 | 0.605022 | 0.037585 | -0.000554 | `reviewable_low_movement` | S1-only low-movement safety probe | `submission_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.csv` |
| 339 | 0.536441 | 0.609547 | 0.097051 | -0.012001 | `reviewable_low_movement` | Q123 sequence-summary scale0.80 + ID337 S1 local-gain probe | `submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv` |

## Decision

- Submit now: **no**
- If the user explicitly wants one controlled public probe: **ID337**
- File: `submission_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.csv`

ID337 has the lower conservative public-risk proxy and smaller movement. ID339 is better locally but moves four targets and has a worse proxy, so it should not jump the queue without explicit user preference for higher-local/higher-risk probing.

ID339 remains useful because it tells us the sequence-summary signal survives local construction, but after ID287/ID329 the selection rule is not “lowest local CV wins.”
