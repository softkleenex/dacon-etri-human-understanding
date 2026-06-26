# Public-gap adjusted candidate ranking after ID188

This is a heuristic risk-control ranking, not a leaderboard prediction. It uses ID178 as the warning case: local `0.538638` scored Public `0.601411`.

## Rank by heuristic risk score

| rank | id | local_cv | heuristic_score | added_gap | tier | max_delta_vs_179 | note |
|---:|---:|---:|---:|---:|---|---:|---|
| 1 | 179 | 0.538290 | 0.573290 | 0.035000 | low_delta | 0.000000 | heuristic_not_claim |
| 2 | 181 | 0.526430 | 0.579971 | 0.053541 | moderate_delta | 0.059985 | heuristic_not_claim |
| 3 | 182 | 0.522909 | 0.581107 | 0.058198 | moderate_delta | 0.079994 | heuristic_not_claim |
| 4 | 180 | 0.529579 | 0.582037 | 0.052458 | moderate_delta | 0.059056 | heuristic_not_claim |
| 5 | 177 | 0.540607 | 0.585607 | 0.045000 | moderate_delta | 0.075416 | heuristic_not_claim |
| 6 | 183 | 0.519611 | 0.597458 | 0.077847 | high_delta | 0.099984 | heuristic_not_claim |
| 7 | 184 | 0.516499 | 0.598998 | 0.082499 | high_delta | 0.119969 | heuristic_not_claim |
| 8 | 178 | 0.538638 | 0.601411 | 0.062774 | moderate_delta | 0.051824 | actual_public_observed |
| 9 | 185 | 0.512540 | 0.601881 | 0.089341 | high_delta | 0.149976 | heuristic_not_claim |
| 10 | 176 | 0.542270 | 0.602270 | 0.060000 | high_delta | 0.138706 | heuristic_not_claim |
| 11 | 175 | 0.543336 | 0.603336 | 0.060000 | high_delta | 0.138706 | heuristic_not_claim |
| 12 | 174 | 0.543431 | 0.603431 | 0.060000 | high_delta | 0.138706 | heuristic_not_claim |
| 13 | 173 | 0.545253 | 0.605253 | 0.060000 | high_delta | 0.139421 | heuristic_not_claim |
| 14 | 186 | 0.506162 | 0.627174 | 0.121012 | extreme_delta | 0.199968 | heuristic_not_claim |
| 15 | 187 | 0.501633 | 0.633931 | 0.132298 | extreme_delta | 0.249960 | heuristic_not_claim |
| 16 | 188 | 0.498094 | 0.641673 | 0.143578 | extreme_delta | 0.299952 | heuristic_not_claim |

## Interpretation

- ID179 is the safest post-ID178 anchor-like candidate, but its local score is not ambitious.
- ID180-ID182 are moderate-delta counterprobes.
- ID183-ID188 are high/extreme-risk rungs; their lower local scores are increasingly driven by ID172 full-OOF calibration.
- Do not submit any candidate without explicit approval.
