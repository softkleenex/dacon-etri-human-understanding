# ID178 sequence alpha scan — post ID334

Created: `2026-06-17 22:11:28 UTC+09:00`

This is report-only. No submission CSV, notebook, base artifact, submission-log row, or DACON submission was created.

## Anchors

| label | local OOF | stress mean | min stress improvement | max Δ178 | max Δ173 | failed projection | alphas |
|---|---:|---:|---:|---:|---:|---:|---|
| `id173_base` | 0.545253 | 0.578583 | +0.000000 | 0.137158 | 0.000000 | +0.0365 | `{'Q1': 0.0, 'Q2': 0.0, 'Q3': 0.0, 'S1': 0.0, 'S2': 0.0, 'S3': 0.0, 'S4': 0.0}` |
| `id178_keep_all` | 0.538638 | 0.576252 | +0.001138 | 0.000000 | 0.137158 | +0.0000 | `{'Q1': 1.0, 'Q2': 1.0, 'Q3': 1.0, 'S1': 1.0, 'S2': 1.0, 'S3': 0.0, 'S4': 1.0}` |
| `id334_rollback_s1_s4` | 0.540347 | 0.573438 | +0.001436 | 0.119120 | 0.137158 | +0.0039 | `{'Q1': 1.0, 'Q2': 1.0, 'Q3': 1.0, 'S1': 0.0, 'S2': 1.0, 'S3': 0.0, 'S4': 0.0}` |

## Selected grid rows

- `best_stress_positive`: local `0.540395`, stress `0.573432`, min stress improvement `+0.001543`, max Δ178 `0.119120`, failed projection `+0.0068`, alphas `{'Q1': 1.0, 'Q2': 1.0, 'Q3': 0.75, 'S1': 0.0, 'S2': 1.0, 'S3': 0.0, 'S4': 0.0}`
- `best_local_among_positive`: local `0.538638`, stress `0.576252`, min stress improvement `+0.001138`, max Δ178 `0.000000`, failed projection `+0.0000`, alphas `{'Q1': 1.0, 'Q2': 1.0, 'Q3': 1.0, 'S1': 1.0, 'S2': 1.0, 'S3': 0.0, 'S4': 1.0}`
- `best_low_movement_positive`: local `0.538638`, stress `0.576252`, min stress improvement `+0.001138`, max Δ178 `0.000000`, failed projection `+0.0000`, alphas `{'Q1': 1.0, 'Q2': 1.0, 'Q3': 1.0, 'S1': 1.0, 'S2': 1.0, 'S3': 0.0, 'S4': 1.0}`

## Verdict

No alpha row is promoted. The scan is useful for understanding ID178's sequence signal, but it does not create a Public 0.58-plausible or submission-ready candidate.

Do not submit any alpha-scan row directly. If a row is promoted later, it needs a new immutable ID and a full artifact contract.

