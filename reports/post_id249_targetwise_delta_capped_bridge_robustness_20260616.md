# Post-ID249 target-wise delta-capped bridge robustness

| id | local | robustness | id178-gap estimate | max delta vs base | note |
|---:|---:|---:|---:|---:|---|
| 235 | 0.482255 | 0.511518 | 0.545029 |  | balanced existing card |
| 242 | 0.357651 | 0.387740 | 0.420424 |  | q240 local minimum stress test |
| 243 | 0.360420 | 0.390285 | 0.423193 |  | q240 all stress-test alternative |
| 244 | 0.437262 | 0.464193 | 0.500036 | 0.148047 | cap≈0.15 uniform bridge; conservative useful probe |
| 245 | 0.415035 | 0.443276 | 0.477808 | 0.246745 | uniform alpha0.375; lower local but larger movement |
| 246 | 0.438034 | 0.464926 | 0.500807 | 0.148047 | ID243-based conservative alternative |
| 247 | 0.415523 | 0.443859 | 0.478297 | 0.198676 | target-wise cap0.20; best safer local-vs-delta tradeoff |
| 248 | 0.391735 | 0.421628 | 0.454509 | 0.298791 | target-wise cap0.30; strong mid-risk local card |
| 249 | 0.374465 | 0.405255 | 0.437238 | 0.399938 | target-wise cap0.40; aggressive frontier bridge |

Ranking by robustness: `[242, 243, 249, 248, 245, 247, 244, 246, 235]`.
Risk-aware recommendation: `[247, 244, 235, 248, 246, 245, 249, 243, 242]`.

ID247 improves the previous safer bridge region: it is near ID245 local CV while capping per-target test movement below 0.20. ID248 is the most interesting mid-risk local improvement, reaching 0.391735 with cap below 0.30. ID249 approaches the q240 frontier at 0.374465 but moves predictions up to 0.40 from ID235 and should be treated as a high-risk probe.
