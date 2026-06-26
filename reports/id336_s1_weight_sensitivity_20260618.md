# ID336 S1 weight sensitivity

Created: `2026-06-18 01:03:56 UTC+09:00`

Base ID: `178`; target: `S1`; seed/kernels/model: `45502` / `512` / `lr_c0_1`.

## Ranked by local CV

| weight | local CV | local gain | S1 gain | mean Δ178 | max Δ178 | failed proj | proxy | naive public |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 0.537663 | 0.000974 | 0.006820 | 0.004008 | 0.112756 | -0.001662 | 0.607945 | 0.600436 |
| 0.18 | 0.537671 | 0.000967 | 0.006769 | 0.004810 | 0.135307 | -0.001994 | 0.608971 | 0.600444 |
| 0.20 | 0.537704 | 0.000934 | 0.006537 | 0.005344 | 0.150342 | -0.002215 | 0.609655 | 0.600477 |
| 0.12 | 0.537710 | 0.000928 | 0.006495 | 0.003206 | 0.090205 | -0.001329 | 0.606921 | 0.600483 |
| 0.10 | 0.537773 | 0.000864 | 0.006050 | 0.002672 | 0.075171 | -0.001108 | 0.606339 | 0.600546 |
| 0.07 | 0.537924 | 0.000714 | 0.004998 | 0.001870 | 0.052620 | -0.000775 | 0.605546 | 0.600697 |
| 0.05 | 0.538066 | 0.000571 | 0.003999 | 0.001336 | 0.037585 | -0.000554 | 0.605022 | 0.600839 |

## Picks

- Best local: weight `0.15`, local `0.537663`.
- Best proxy: weight `0.05`, proxy `0.605022`.
- Balanced choice: weight `0.05`, local `0.538066`, proxy `0.605022`.

## Decision

This is report-only. It does not create a DACON submission or new artifact ID. ID336 remains the existing weight-0.15 generated-unsubmitted artifact unless a separate promotion is explicitly created.
