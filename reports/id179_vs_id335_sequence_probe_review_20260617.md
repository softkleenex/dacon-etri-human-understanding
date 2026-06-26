# ID179 vs ID335 sequence probe review

Created: `2026-06-17 22:31:20 UTC+09:00`

This is report-only. No submission CSV, notebook, base artifact, submission-log row, or DACON submission was created.

## Candidate comparison

| id | overall | robust | worst subject | worst date | worst target | max Δ178 | max Δ335 | failed projection |
|---:|---:|---:|---|---|---|---:|---:|---:|
| 178 | 0.538638 | 0.564192 | id10 `0.633714` | q4 `0.564958` | S4 `0.602700` | 0.000000 | 0.111949 | +0.0000 |
| 179 | 0.538290 | 0.563975 | id10 `0.633590` | q4 `0.565095` | S4 `0.602685` | 0.051824 | 0.093766 | +0.0011 |
| 334 | 0.540347 | 0.566401 | id10 `0.634386` | q4 `0.567058` | S4 `0.608814` | 0.119120 | 0.059560 | +0.0039 |
| 335 | 0.539690 | 0.565340 | id10 `0.633887` | q4 `0.566569` | S4 `0.604213` | 0.111949 | 0.000000 | +0.0032 |

## Verdict

ID179 is locally stronger and much closer to ID178 than ID335, so it remains the cleanest same-family manual-review probe. However, it is still non-strict and OOF-selected; it is not Public 0.58-plausible and should not be submitted without explicit approval.

