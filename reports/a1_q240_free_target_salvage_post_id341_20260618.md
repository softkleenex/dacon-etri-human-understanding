# A1 q240-free target salvage review — post ID341

Created: `2026-06-18 03:58:59 UTC+09:00`

This is report-only. It creates no CSV, notebook/history artifact, base prediction artifact, submission-log row, or DACON submission.

## Target classification

| target | class | mean replay Δ vs base | max regret vs selector | SH replay/base/selector | LOSO replay/base/selector |
|---|---|---:|---:|---|---|
| `S4` | `stable_salvage_candidate` | -0.106317 | +0.012991 | 0.409179/0.513875/0.396188 | 0.405938/0.513875/0.396188 |
| `Q3` | `selector_only_trap` | +0.023373 | +0.115390 | 0.540587/0.513626/0.425197 | 0.533411/0.513626/0.425197 |
| `S1` | `selector_only_trap` | +0.023525 | +0.103261 | 0.466811/0.428421/0.363550 | 0.437080/0.428421/0.363550 |
| `Q1` | `selector_only_trap` | +0.026367 | +0.099255 | 0.529160/0.493404/0.429906 | 0.510382/0.493404/0.429906 |
| `Q2` | `selector_only_trap` | +0.033916 | +0.114188 | 0.517558/0.488122/0.412329 | 0.526517/0.488122/0.412329 |
| `S2` | `selector_only_trap` | +0.037659 | +0.103282 | 0.510905/0.473224/0.407622 | 0.510861/0.473224/0.408797 |
| `S3` | `reject` | +0.000000 | +0.000000 | 0.465115/0.465115/0.465115 | 0.465115/0.465115/0.465115 |

## Hybrid sanity check

| split | base | full replay | selector-only qfree | stable-target hybrid | weak-watch hybrid |
|---|---:|---:|---:|---:|---:|
| `subject_hole_source_replay` | 0.482255 | 0.491331 | 0.414272 | 0.467299 (-0.014957) | 0.467299 (-0.014957) |
| `loso_source_replay` | 0.482255 | 0.484186 | 0.414440 | 0.466836 (-0.015420) | 0.466836 (-0.015420) |

## Decision

- Stable salvage targets: `['S4']`
- Weak watch targets: `[]`
- Selector-only trap targets: `['Q1', 'Q2', 'Q3', 'S1', 'S2']`
- Stable salvage artifact review pass: `True`
- Create artifact now: **no**
- Submit now: **no**

If no stable salvage target passes the hybrid threshold, stop local A1 q240-free work and pivot to new source families. If stable targets exist, require a full prediction-geometry gate before any artifact.

