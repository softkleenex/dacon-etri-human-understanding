# Sequence-summary geometry gate — post A1

Created: `2026-06-18 02:08:00 UTC+09:00`

This is report-only. No CSV, notebook, base prediction artifact, submission-log row, or DACON submission was created.

## Candidate gate

| rank | candidate | local | robust | max Δ178 | mean Δ178 | failed proj | worst target | modified | read |
|---:|---|---:|---:|---:|---:|---:|---|---|---|
| 1 | `id178_q123_sequence_summary_plus_id337_s1_safety_gate` | 0.536437 | 0.561965 | 0.121314 | 0.015419 | -0.014863 | S4 `0.602700` | `['Q1', 'Q2', 'Q3', 'S1']` | public-anchor Q1/Q2/Q3 sequence-summary plus ID337 S1 low-movement random-conv |
| 2 | `id178_q123s1_sequence_summary_gate` | 0.536786 | 0.562501 | 0.121314 | 0.015681 | -0.015549 | S4 `0.602700` | `['Q1', 'Q2', 'Q3', 'S1']` | public-anchor sequence-summary on Q1/Q2/Q3/S1 |
| 3 | `id178_q123_sequence_summary_gate` | 0.537008 | 0.562416 | 0.121314 | 0.014083 | -0.014309 | S4 `0.602700` | `['Q1', 'Q2', 'Q3']` | public-anchor Q1/Q2/Q3 sequence-summary only |

## Target movement vs ID178

### `id178_q123_sequence_summary_plus_id337_s1_safety_gate`

| target | max Δ178 | loss |
|---|---:|---:|
| `Q1` | 0.094315 | 0.549044 |
| `Q2` | 0.121314 | 0.550689 |
| `Q3` | 0.083883 | 0.564546 |
| `S1` | 0.037585 | 0.467631 |
| `S2` | 0.000000 | 0.529556 |
| `S3` | 0.000000 | 0.490893 |
| `S4` | 0.000000 | 0.602700 |

### `id178_q123s1_sequence_summary_gate`

| target | max Δ178 | loss |
|---|---:|---:|
| `Q1` | 0.094315 | 0.549044 |
| `Q2` | 0.121314 | 0.550689 |
| `Q3` | 0.083883 | 0.564546 |
| `S1` | 0.042166 | 0.470074 |
| `S2` | 0.000000 | 0.529556 |
| `S3` | 0.000000 | 0.490893 |
| `S4` | 0.000000 | 0.602700 |

### `id178_q123_sequence_summary_gate`

| target | max Δ178 | loss |
|---|---:|---:|
| `Q1` | 0.094315 | 0.549044 |
| `Q2` | 0.121314 | 0.550689 |
| `Q3` | 0.083883 | 0.564546 |
| `S1` | 0.000000 | 0.471630 |
| `S2` | 0.000000 | 0.529556 |
| `S3` | 0.000000 | 0.490893 |
| `S4` | 0.000000 | 0.602700 |

## Decision

Sequence-summary gives a real local improvement over ID178, but Q2 movement reaches about 0.12 and the branch changes multiple targets. Treat this as a stronger candidate-construction lead, not an automatic ID339 promotion.

