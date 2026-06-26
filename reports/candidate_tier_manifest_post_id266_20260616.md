# Candidate tier manifest after ID266

ID264 is now the preferred first public probe: it is the best refined cap0.12 direct-lite candidate and avoids direct stress IDs. ID260 is marginally sharper at cap0.12 but has a worse source-risk story. ID266 is the best clean sharper second probe if the first result suggests the ID178 delta cap can be relaxed.

## New candidates
- ID264: Refined direct-lite cap0.12 pair blend. Local 0.464758, robustness 0.494762, max delta 0.119995. Preferred first probe after replacing ID263.
- ID265: Refined direct-lite cap0.14 pair blend. Local 0.458478, robustness 0.488017, max delta 0.139926. Intermediate clean sharper card.
- ID266: Refined direct-lite cap0.15 pair blend. Local 0.457894, robustness 0.487400, max delta 0.149809. Best clean sharper second-probe card.

## Ranked candidates

| rank | id | local CV | robustness | ID178-gap estimate | max delta vs ID178 | risk note |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 264 | 0.464758 | 0.494762 | 0.527532 | 0.119995 | preferred first public probe: refined direct-lite cap0.12; no direct stress IDs; slightly lower local than ID263 |
| 2 | 260 | 0.463684 | 0.493703 | 0.526458 | 0.119513 | slightly sharper cap0.12 score-pressure card, but directly uses riskier source choices from ID259/248/258/253 |
| 3 | 266 | 0.457894 | 0.487400 | 0.520668 | 0.149809 | best clean sharper second probe: cap0.15; nearly ID258 local with lower max delta by only repairing Q2 over ID258 |
| 4 | 265 | 0.458478 | 0.488017 | 0.521251 | 0.139926 | intermediate cap0.14 clean sharper bridge; smoother step from ID264 to ID266 |
| 5 | 263 | 0.464978 | 0.495013 | 0.527751 | 0.119513 | coarse direct-lite cap0.12 predecessor; superseded by ID264 |
| 6 | 255 | 0.469816 | 0.500061 | 0.532590 | 0.115233 | simplest conservative fallback: fine cap0.12 anchor bridge, no pair recombination |
| 7 | 258 | 0.457647 | 0.487146 | 0.520420 | 0.159452 | clean cap0.16 bridge; lower local than ID266 but slightly larger max delta |
| 8 | 253 | 0.477131 | 0.507486 | 0.539904 | 0.106724 | original cap0.12 public-anchor bridge; clean fallback with smallest nonzero max delta |
| 9 | 261 | 0.441919 | 0.471868 | 0.504693 | 0.159583 | sharp pair-blend stress card: local 0.441919, but leans on ID252 and cap0.16 |
| 10 | 262 | 0.440031 | 0.469165 | 0.502805 | 0.179554 | sharpest pair-blend stress card: local 0.440031, cap0.18, high movement/stress-source risk |
| 11 | 259 | 0.443567 | 0.472753 | 0.506340 | 0.178005 | clean cap0.18 bridge; high movement stress card |
| 12 | 257 | 0.433079 | 0.462022 | 0.495853 | 0.176195 | fine cap0.18 bridge; high-risk stress card |
| 13 | 247 | 0.415523 | 0.443859 | 0.478297 | 0.548540 | old local frontier bridge source; direct test movement too large for first probe |
| 14 | 248 | 0.391735 | 0.421628 | 0.454509 | 0.538331 | mid-risk cap0.30 source; direct test movement too large for first probe |
| 15 | 252 | 0.274420 | 0.315901 | 0.337193 | 0.807354 | second-stage stress source; ingredient only, not direct probe |

## Practical interpretation

- ID264 supersedes ID263 as the first-probe recommendation.
- ID266 is important because it gets almost all of ID258 local strength while reducing max movement from `0.159452` to `0.149809`.
- ID260 still has the lowest cap0.12 local, but its source path is less clean than ID264.
- ID261/ID262 remain stress cards only until leaderboard feedback says the larger cap/stress-source direction is safe.

No DACON submission was made.
