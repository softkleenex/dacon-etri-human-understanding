# ID178 confirmed-public anchor bridge fine scan

This fine scan expands frontier IDs and alpha granularity while keeping target-wise max test delta versus ID178 capped.

Anchor public score: `0.6014113009`
Frontier IDs: `[247, 248, 249, 250, 251, 252, 242, 243]`

| cap | local CV | max delta vs ID178 | changed targets | ID178-gap estimate |
|---:|---:|---:|---:|---:|
| 0.060 | 0.501990 | 0.057803 | 7 | 0.564763 |
| 0.080 | 0.490143 | 0.079073 | 7 | 0.552916 |
| 0.100 | 0.480188 | 0.095378 | 7 | 0.542961 |
| 0.120 | 0.469816 | 0.115233 | 7 | 0.532590 |
| 0.140 | 0.464121 | 0.131771 | 7 | 0.526894 |
| 0.160 | 0.449141 | 0.158120 | 7 | 0.511914 |
| 0.180 | 0.433079 | 0.176195 | 7 | 0.495853 |
| 0.200 | 0.431574 | 0.199091 | 7 | 0.494348 |

Promoted cards:
- ID255: cap `0.12`, local `0.469816`, sharpened conservative card.
- ID256: cap `0.14`, local `0.464121`, balanced sharper card.
- ID257: cap `0.18`, local `0.433079`, high-risk stress card.

Caution: the fine scan draws from ultra-frontier IDs 250/251 for some targets. The ID178 delta cap controls test movement but does not make this nested or public-proof.
