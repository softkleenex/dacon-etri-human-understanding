# ID178 sequence alpha targeted refinement — post ID334

Created: `2026-06-17 22:13:40 UTC+09:00`

This is report-only. It scans a small neighborhood around ID334 and does not create any submission artifact.

## Anchor vs selected rows

| row | local OOF | stress mean | min stress improvement | max Δ178 | failed projection | alphas |
|---|---:|---:|---:|---:|---:|---|
| `id334_anchor` | 0.540347 | 0.573438 | +0.001436 | 0.119120 | +0.0039 | `{'Q1': 1.0, 'Q2': 1.0, 'Q3': 1.0, 'S1': 0.0, 'S2': 1.0, 'S3': 0.0, 'S4': 0.0}` |
| `best_stress` | 0.540182 | 0.573385 | +0.001943 | 0.111949 | +0.0070 | `{'Q1': 0.875, 'Q2': 0.875, 'Q3': 0.875, 'S1': 0.0, 'S2': 1.0, 'S3': 0.0, 'S4': 0.125}` |
| `best_local_stress_guarded` | 0.539690 | 0.573622 | +0.001896 | 0.111949 | +0.0032 | `{'Q1': 1.0, 'Q2': 1.0, 'Q3': 1.0, 'S1': 0.0, 'S2': 1.0, 'S3': 0.0, 'S4': 0.5}` |
| `best_local_geometry_guarded` | 0.539690 | 0.573622 | +0.001896 | 0.111949 | +0.0032 | `{'Q1': 1.0, 'Q2': 1.0, 'Q3': 1.0, 'S1': 0.0, 'S2': 1.0, 'S3': 0.0, 'S4': 0.5}` |

## Verdict

A tiny geometry-guarded alpha refinement improves ID334 local OOF while keeping stress essentially flat. It is still same-family and should only be promoted as a generated_unsubmitted artifact, never submitted directly.

