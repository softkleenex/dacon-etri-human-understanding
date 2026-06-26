# Recent S1 probe stress review

Created: `2026-06-18 01:21:16 UTC+09:00`

This is report-only. Existing OOF/test artifacts were scanned; no model was trained and no DACON submission was made.

## Candidate comparison

| rank | id | local | robust | gain vs 178 | robust gain | max Δ178 | failed proj | proxy | worst target | read |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 178 | 0.538638 | 0.564192 | +0.000000 | +0.000000 | 0.000000 | +0.000000 | 0.603753 | S4 `0.602700` | reference/watch |
| 2 | 337 | 0.538066 | 0.563741 | +0.000571 | +0.000451 | 0.037585 | -0.000554 | 0.605022 | S4 `0.602700` | safety-first S1 probe |
| 3 | 179 | 0.538290 | 0.563975 | +0.000348 | +0.000218 | 0.051824 | +0.001123 | 0.606255 | S4 `0.602685` | same-family anchor reference |
| 4 | 338 | 0.537861 | 0.563600 | +0.000777 | +0.000592 | 0.073989 | +0.000549 | 0.606833 | S4 `0.602685` | local-gain sibling; higher public-anchor movement |
| 5 | 336 | 0.537663 | 0.563379 | +0.000974 | +0.000813 | 0.112756 | -0.001662 | 0.607945 | S4 `0.602700` | local-best ID178 S1 weight; movement higher than ID337 |
| 6 | 335 | 0.539690 | 0.565340 | -0.001052 | -0.001147 | 0.111949 | +0.003189 | 0.609846 | S4 `0.604213` | reference/watch |
| 7 | 334 | 0.540347 | 0.566401 | -0.001710 | -0.002208 | 0.119120 | +0.003943 | 0.610986 | S4 `0.608814` | reference/watch |

## S1 target losses

| id | S1 loss | Δ S1 vs 178 | Δ S1 vs 337 |
|---:|---:|---:|---:|
| 178 | 0.471630 | +0.000000 | +0.003999 |
| 337 | 0.467631 | -0.003999 | +0.000000 |
| 179 | 0.469714 | -0.001916 | +0.002083 |
| 338 | 0.466711 | -0.004920 | -0.000920 |
| 336 | 0.464810 | -0.006820 | -0.002821 |
| 335 | 0.477483 | +0.005852 | +0.009852 |
| 334 | 0.477483 | +0.005852 | +0.009852 |

## Decision

ID337 remains the safety-first S1 probe: it has the best conservative proxy and lowest movement from ID178. ID338 is locally stronger by 0.000206, but its proxy is worse by 0.001811. Neither candidate repairs validation or proves Public 0.58; both remain explicit-approval-only controlled probes.

