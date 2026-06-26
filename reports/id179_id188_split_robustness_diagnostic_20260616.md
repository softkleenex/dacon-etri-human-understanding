# ID179-ID188 split robustness diagnostic

Lower is better. Robustness score penalizes worst subject, worst date quartile, worst target, and late-date degradation. This is not a public LB prediction.

| id | overall | robust_score | worst_subject | worst_subj_loss | worst_date_bin | worst_date_loss | late-early | worst_target | worst_target_loss |
|---:|---:|---:|---|---:|---|---:|---:|---|---:|
| 179 | 0.538290 | 0.571757 | id10 | 0.633590 | q3 | 0.569956 | 0.064347 | S4 | 0.602685 |
| 180 | 0.529579 | 0.562892 | id10 | 0.625831 | q3 | 0.559096 | 0.062198 | S4 | 0.594836 |
| 181 | 0.526430 | 0.559530 | id10 | 0.622749 | q3 | 0.554727 | 0.061346 | S4 | 0.591671 |
| 182 | 0.522909 | 0.555910 | id10 | 0.619611 | q3 | 0.550169 | 0.060437 | S4 | 0.588410 |
| 183 | 0.519611 | 0.552531 | id10 | 0.616701 | q3 | 0.545877 | 0.059577 | S4 | 0.585438 |
| 184 | 0.516499 | 0.549354 | id10 | 0.614014 | q3 | 0.541799 | 0.058753 | S4 | 0.582688 |
| 185 | 0.512540 | 0.545381 | id10 | 0.610831 | q3 | 0.536823 | 0.057989 | S4 | 0.578954 |
| 186 | 0.506162 | 0.538933 | id10 | 0.605606 | q3 | 0.528204 | 0.056115 | S4 | 0.573861 |
| 187 | 0.501633 | 0.534604 | id10 | 0.602643 | q3 | 0.522600 | 0.055380 | S4 | 0.570184 |
| 188 | 0.498094 | 0.531377 | id10 | 0.600778 | q3 | 0.518168 | 0.054655 | S4 | 0.568118 |

## Rank by robustness score
1. ID188 robust=0.531377, overall=0.498094
2. ID187 robust=0.534604, overall=0.501633
3. ID186 robust=0.538933, overall=0.506162
4. ID185 robust=0.545381, overall=0.512540
5. ID184 robust=0.549354, overall=0.516499
6. ID183 robust=0.552531, overall=0.519611
7. ID182 robust=0.555910, overall=0.522909
8. ID181 robust=0.559530, overall=0.526430
9. ID180 robust=0.562892, overall=0.529579
10. ID179 robust=0.571757, overall=0.538290
