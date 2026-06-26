# ID179 S1 weight sensitivity

Created: `2026-06-18 01:13:16 UTC+09:00`

Base ID: `179`; target: `S1`; seed/kernels/model: `45502` / `512` / `lr_c0_1`.

## Ranked by local CV

| weight | local CV | local gain | S1 gain | mean Δ178 | max Δ178 | failed proj | proxy | naive public |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 0.537568 | 0.000722 | 0.005051 | 0.006802 | 0.144825 | -0.000598 | 0.609734 | 0.600341 |
| 0.12 | 0.537591 | 0.000699 | 0.004890 | 0.006079 | 0.123406 | -0.000254 | 0.608770 | 0.600364 |
| 0.18 | 0.537595 | 0.000695 | 0.004866 | 0.007541 | 0.166245 | -0.000942 | 0.610702 | 0.600368 |
| 0.10 | 0.537636 | 0.000654 | 0.004578 | 0.005636 | 0.109126 | -0.000024 | 0.608138 | 0.600409 |
| 0.20 | 0.537639 | 0.000651 | 0.004559 | 0.008038 | 0.180524 | -0.001171 | 0.611349 | 0.600412 |
| 0.07 | 0.537750 | 0.000540 | 0.003779 | 0.005031 | 0.087707 | +0.000320 | 0.607268 | 0.600523 |
| 0.05 | 0.537861 | 0.000429 | 0.003003 | 0.004660 | 0.073989 | +0.000549 | 0.606833 | 0.600634 |

## Picks

- Best local: weight `0.15`, local `0.537568`.
- Best proxy: weight `0.05`, proxy `0.606833`.
- Balanced choice: weight `0.05`, local `0.537861`, proxy `0.606833`.

## Decision

This is report-only. It does not create a DACON submission or new artifact ID. Any promoted candidate from this sweep must use a new immutable artifact ID unless that exact artifact already exists.
