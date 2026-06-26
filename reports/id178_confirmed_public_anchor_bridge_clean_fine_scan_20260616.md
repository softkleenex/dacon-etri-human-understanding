# ID178 confirmed-public anchor bridge clean fine scan

This scan excludes the raw ultra-frontier IDs 250/251/242/243 and uses only frontier IDs 247/248/249/252.

Anchor public score: `0.6014113009`
Frontier IDs: `[247, 248, 249, 252]`

| cap | local CV | max delta vs ID178 | changed targets | ID178-gap estimate |
|---:|---:|---:|---:|---:|
| 0.060 | 0.505657 | 0.056962 | 7 | 0.568430 |
| 0.080 | 0.495498 | 0.076369 | 7 | 0.558271 |
| 0.100 | 0.486210 | 0.093524 | 7 | 0.548983 |
| 0.120 | 0.477131 | 0.106724 | 7 | 0.539904 |
| 0.140 | 0.472369 | 0.139880 | 7 | 0.535142 |
| 0.160 | 0.457647 | 0.159452 | 7 | 0.520420 |
| 0.180 | 0.443567 | 0.178005 | 7 | 0.506340 |

Promoted cards:
- ID258: cap `0.16`, local `0.457647`, clean sharper card.
- ID259: cap `0.18`, local `0.443567`, clean high-risk card.

Caution: ID252 is still a full-OOF second-stage frontier source, but this clean scan avoids the most extreme ID250/251/242/243 sources.
