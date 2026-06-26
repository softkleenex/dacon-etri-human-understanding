# Evening-load/recovery smoke — post ID341

Created: `2026-06-18 04:53:10 UTC+09:00`

This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, or DACON submission.

## Feature coverage

- Rows: `700` total / `450` train / `250` test
- Feature columns: `148`
- Mean missing rate: train `0.0947`, test `0.0885`
- Available base columns: `48`; missing base columns: `4`

## Selected target configs

| target | candidate | wins | mean Δ | max Δ |
|---|---|---:|---:|---:|
| `S4` | `S4_id178_lr_c0_2_w0.05` | 4 | -0.000696 | +0.000654 |

## Scale scan

| scale | primary wins | all primary | local | proxy | Δ341 mean/max | failed proj | reviewable |
|---:|---:|---|---:|---:|---:|---:|---|
| 0.05 | 4 | `False` | 0.538579 | 0.601507 | 0.000056/0.001344 | -0.000093 | `False` |
| 0.10 | 4 | `False` | 0.538575 | 0.601553 | 0.000112/0.002689 | -0.000102 | `False` |
| 0.15 | 4 | `False` | 0.538571 | 0.601613 | 0.000168/0.004033 | -0.000110 | `False` |
| 0.20 | 4 | `False` | 0.538567 | 0.601673 | 0.000224/0.005378 | -0.000119 | `False` |
| 0.30 | 4 | `False` | 0.538560 | 0.601792 | 0.000335/0.008067 | -0.000136 | `False` |
| 0.40 | 4 | `False` | 0.538553 | 0.601912 | 0.000447/0.010756 | -0.000153 | `False` |
| 0.50 | 4 | `False` | 0.538547 | 0.602031 | 0.000559/0.013444 | -0.000170 | `False` |
| 0.65 | 4 | `False` | 0.538539 | 0.602211 | 0.000727/0.017478 | -0.000196 | `False` |
| 0.80 | 4 | `False` | 0.538533 | 0.602390 | 0.000894/0.021511 | -0.000221 | `False` |
| 1.00 | 4 | `False` | 0.538527 | 0.602629 | 0.001118/0.026889 | -0.000255 | `False` |

## Decision

- Reviewable low-movement candidate: `False`
- Best proxy: `0.601507` at scale `0.05`
- Public `0.58` plausible: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Next action: If evening-load has a reviewable row, combine it with the sleep-boundary scale-0.05 lead in a no-artifact combined gate. Otherwise keep it as rejected context and try routine-stability/place entropy.

