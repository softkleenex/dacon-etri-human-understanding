# Sequence-summary cap sensitivity — post A1

Created: `2026-06-18 02:09:57 UTC+09:00`

This is report-only. No CSV, notebook, base prediction artifact, submission-log row, or DACON submission was created.

## Scale sweep

| scale | local | robust | max Δ178 | mean Δ178 | failed proj | worst target |
|---:|---:|---:|---:|---:|---:|---|
| 0.40 | 0.536914 | 0.562587 | 0.048526 | 0.006969 | -0.006278 | S4 `0.602700` |
| 0.50 | 0.536735 | 0.562395 | 0.060657 | 0.008377 | -0.007709 | S4 `0.602700` |
| 0.60 | 0.536597 | 0.562240 | 0.072788 | 0.009786 | -0.009139 | S4 `0.602700` |
| 0.70 | 0.536500 | 0.562120 | 0.084920 | 0.011194 | -0.010570 | S4 `0.602700` |
| 0.75 | 0.536465 | 0.562073 | 0.090985 | 0.011898 | -0.011286 | S4 `0.602700` |
| 0.80 | 0.536441 | 0.562034 | 0.097051 | 0.012602 | -0.012001 | S4 `0.602700` |
| 0.85 | 0.536426 | 0.562004 | 0.103117 | 0.013306 | -0.012717 | S4 `0.602700` |
| 0.90 | 0.536420 | 0.561983 | 0.109183 | 0.014010 | -0.013432 | S4 `0.602700` |
| 0.95 | 0.536424 | 0.561970 | 0.115248 | 0.014714 | -0.014148 | S4 `0.602700` |
| 1.00 | 0.536437 | 0.561965 | 0.121314 | 0.015419 | -0.014863 | S4 `0.602700` |

## Best by cap

| cap | variant | local | max Δ178 |
|---|---|---:|---:|
| `cap010` | `id178_q123_summary_scale0.80_plus_id337_s1` | 0.536441 | 0.097051 |
| `cap011` | `id178_q123_summary_scale0.90_plus_id337_s1` | 0.536420 | 0.109183 |
| `cap012` | `id178_q123_summary_scale0.90_plus_id337_s1` | 0.536420 | 0.109183 |

## Decision

Shrinking Q1/Q2/Q3 sequence-summary weights can bring max movement below 0.10 while keeping a meaningful local gain over ID337. This is now a cleaner candidate-construction target, but still needs explicit promotion and artifact-contract creation before any submission discussion.

