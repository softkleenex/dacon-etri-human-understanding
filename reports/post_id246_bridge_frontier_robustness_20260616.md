# Post-ID246 bridge frontier robustness

External `[Public_0.5917] LGBM+XGB` notebook was useful as strategy guidance, not as a directly reproducible artifact. Its strongest transferable idea is anchor-preserving, subject-aware, stability-controlled blending.

| id | local/overall | robustness | id178-gap estimate | max delta vs ID235/base | note |
|---:|---:|---:|---:|---:|---|
| 235 | 0.482255 | 0.511518 | 0.545029 |  | balanced existing card |
| 238 | 0.462365 | 0.493658 | 0.525138 |  | aggressive recombination |
| 242 | 0.357651 | 0.387740 | 0.420424 |  | raw local minimum, severe overfit risk |
| 243 | 0.360420 | 0.390285 | 0.423193 |  | q240 all alternative |
| 244 | 0.437262 | 0.464193 | 0.500036 | 0.148047 | recommended new bridge; max_delta≈0.148 |
| 245 | 0.415035 | 0.443276 | 0.477808 | 0.246745 | higher-risk bridge; max_delta≈0.247 |
| 246 | 0.438034 | 0.464926 | 0.500807 | 0.148047 | ID243 alternative bridge; less S4-isotonic |

Ranking by robustness: `[242, 243, 245, 244, 246, 238, 235]`.
Risk-aware recommendation: `[244, 235, 246, 245, 238, 243, 242]`.

Recommendation: ID244 is the next sanest public probe candidate if the user approves a DACON submission. ID245 is a bolder second probe, while ID242/243 remain stress tests rather than proof.
