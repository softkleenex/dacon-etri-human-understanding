# B1 pair/logit source dependency audit

Created: `2026-06-18 01:30:05 UTC+09:00`

This is report-only. It audits metadata dependencies only; no model was trained and no DACON submission was made.

## B1 source records

| id | risk | local | direct upstream ids | method |
|---:|---|---:|---|---|
| 254 | `high_non_nested` | 0.471985 | `[178, 247, 248, 252]` | Confirmed-public-anchor target-wise delta-capped bridge |
| 255 | `high_non_nested` | 0.469816 | `[178, 242, 243, 247, 248, 249, 250, 251, 252]` | Confirmed-public-anchor target-wise delta-capped bridge |
| 256 | `high_non_nested` | 0.464121 | `[178, 242, 243, 247, 248, 249, 250, 251, 252]` | Confirmed-public-anchor target-wise delta-capped bridge |
| 257 | `high_non_nested` | 0.433079 | `[178, 242, 243, 247, 248, 249, 250, 251, 252]` | Confirmed-public-anchor target-wise delta-capped bridge |
| 258 | `high_non_nested` | 0.457647 | `[178, 247, 248, 249, 252]` | Confirmed-public-anchor target-wise delta-capped bridge |
| 266 | `high_non_nested` | 0.457894 | `[255, 258]` | Target-wise full-OOF pair probability/logit blend probe |

## Dependency closure risk counts

- `anchor_reference`: 3
- `extreme_non_nested`: 9
- `high_non_nested`: 20

## Critical finding

The B1 lane excludes direct ID252 at the selector layer, but its source artifacts are not dependency-clean. IDs 254/255/256/257/258/266 trace back through high/extreme full-OOF bridge or calibration artifacts, including the ID242/250/251/252 chain. A true rebuild must start below these source artifacts, not from them.

## Dependency edges

| source | upstream | upstream risk |
|---:|---:|---|
| 168 | 167 | `high_non_nested` |
| 169 | 168 | `high_non_nested` |
| 172 | 163 | `high_non_nested` |
| 173 | 169 | `high_non_nested` |
| 178 | 173 | `anchor_reference` |
| 202 | 172 | `anchor_reference` |
| 202 | 200 | `high_non_nested` |
| 218 | 172 | `anchor_reference` |
| 218 | 215 | `high_non_nested` |
| 221 | 202 | `high_non_nested` |
| 221 | 218 | `high_non_nested` |
| 226 | 172 | `anchor_reference` |
| 226 | 221 | `high_non_nested` |
| 227 | 226 | `high_non_nested` |
| 228 | 226 | `high_non_nested` |
| 235 | 226 | `high_non_nested` |
| 238 | 221 | `high_non_nested` |
| 238 | 226 | `high_non_nested` |
| 238 | 227 | `extreme_non_nested` |
| 238 | 228 | `extreme_non_nested` |
| 239 | 238 | `high_non_nested` |
| 242 | 238 | `high_non_nested` |
| 243 | 238 | `high_non_nested` |
| 247 | 235 | `extreme_non_nested` |
| 247 | 238 | `high_non_nested` |
| 247 | 239 | `extreme_non_nested` |
| 247 | 242 | `extreme_non_nested` |
| 248 | 235 | `extreme_non_nested` |
| 248 | 238 | `high_non_nested` |
| 248 | 239 | `extreme_non_nested` |
| 248 | 242 | `extreme_non_nested` |
| 249 | 235 | `extreme_non_nested` |
| 249 | 238 | `high_non_nested` |
| 249 | 239 | `extreme_non_nested` |
| 249 | 242 | `extreme_non_nested` |
| 250 | 242 | `extreme_non_nested` |
| 251 | 242 | `extreme_non_nested` |
| 252 | 249 | `high_non_nested` |
| 254 | 178 | `anchor_reference` |
| 254 | 247 | `high_non_nested` |
| 254 | 248 | `high_non_nested` |
| 254 | 252 | `extreme_non_nested` |
| 255 | 178 | `anchor_reference` |
| 255 | 242 | `extreme_non_nested` |
| 255 | 243 | `extreme_non_nested` |
| 255 | 247 | `high_non_nested` |
| 255 | 248 | `high_non_nested` |
| 255 | 249 | `high_non_nested` |
| 255 | 250 | `extreme_non_nested` |
| 255 | 251 | `extreme_non_nested` |
| 255 | 252 | `extreme_non_nested` |
| 256 | 178 | `anchor_reference` |
| 256 | 242 | `extreme_non_nested` |
| 256 | 243 | `extreme_non_nested` |
| 256 | 247 | `high_non_nested` |
| 256 | 248 | `high_non_nested` |
| 256 | 249 | `high_non_nested` |
| 256 | 250 | `extreme_non_nested` |
| 256 | 251 | `extreme_non_nested` |
| 256 | 252 | `extreme_non_nested` |
| 257 | 178 | `anchor_reference` |
| 257 | 242 | `extreme_non_nested` |
| 257 | 243 | `extreme_non_nested` |
| 257 | 247 | `high_non_nested` |
| 257 | 248 | `high_non_nested` |
| 257 | 249 | `high_non_nested` |
| 257 | 250 | `extreme_non_nested` |
| 257 | 251 | `extreme_non_nested` |
| 257 | 252 | `extreme_non_nested` |
| 258 | 178 | `anchor_reference` |
| 258 | 247 | `high_non_nested` |
| 258 | 248 | `high_non_nested` |
| 258 | 249 | `high_non_nested` |
| 258 | 252 | `extreme_non_nested` |
| 266 | 255 | `high_non_nested` |
| 266 | 258 | `high_non_nested` |
