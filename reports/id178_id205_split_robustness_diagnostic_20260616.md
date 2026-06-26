# ID178-ID205 split robustness diagnostic

Lower is better. Robustness score penalizes worst subject, worst date quartile, worst target, and late-date degradation. This is not a public LB prediction.

| id | overall | robust_score | worst_subject | worst_subj_loss | worst_date_bin | worst_date_loss | late-early | worst_target | worst_target_loss |
|---:|---:|---:|---|---:|---|---:|---:|---|---:|
| 178 | 0.538638 | 0.571307 | id10 | 0.633714 | q4 | 0.564958 | 0.065989 | S4 | 0.602700 |
| 193 | 0.536429 | 0.569090 | id10 | 0.628175 | q4 | 0.564927 | 0.068231 | S4 | 0.602685 |
| 198 | 0.536380 | 0.568934 | id10 | 0.627613 | q4 | 0.564776 | 0.068350 | S4 | 0.602685 |
| 200 | 0.536190 | 0.568848 | id10 | 0.627898 | q4 | 0.564598 | 0.068174 | S4 | 0.602656 |
| 203 | 0.535980 | 0.568318 | id10 | 0.626543 | q4 | 0.564471 | 0.068624 | S4 | 0.601188 |
| 194 | 0.525179 | 0.557310 | id10 | 0.618311 | q4 | 0.550505 | 0.061130 | S4 | 0.591671 |
| 199 | 0.525173 | 0.557241 | id10 | 0.617975 | q4 | 0.550468 | 0.061279 | S4 | 0.591671 |
| 201 | 0.524999 | 0.557148 | id10 | 0.618189 | q4 | 0.550305 | 0.061075 | S4 | 0.591609 |
| 204 | 0.524761 | 0.556611 | id10 | 0.617038 | q4 | 0.550125 | 0.061453 | S4 | 0.589941 |
| 195 | 0.521836 | 0.553822 | id10 | 0.615486 | q4 | 0.546180 | 0.058943 | S4 | 0.588410 |
| 202 | 0.521667 | 0.553676 | id10 | 0.615403 | q4 | 0.546004 | 0.058874 | S4 | 0.588340 |
| 205 | 0.521424 | 0.553144 | id10 | 0.614319 | q4 | 0.545809 | 0.059227 | S4 | 0.586639 |
| 196 | 0.518710 | 0.550576 | id10 | 0.612916 | q4 | 0.542107 | 0.056840 | S4 | 0.585438 |
| 197 | 0.515748 | 0.547507 | id10 | 0.610521 | q4 | 0.538219 | 0.054789 | S4 | 0.582688 |
| 189 | 0.492595 | 0.525866 | id10 | 0.599849 | q2 | 0.510464 | 0.032476 | S4 | 0.567760 |
| 172 | 0.492762 | 0.526318 | id10 | 0.601349 | q2 | 0.509990 | 0.032761 | S4 | 0.568928 |

## Rank by robustness score
1. ID189 robust=0.525866, overall=0.492595
2. ID172 robust=0.526318, overall=0.492762
3. ID197 robust=0.547507, overall=0.515748
4. ID196 robust=0.550576, overall=0.518710
5. ID205 robust=0.553144, overall=0.521424
6. ID202 robust=0.553676, overall=0.521667
7. ID195 robust=0.553822, overall=0.521836
8. ID204 robust=0.556611, overall=0.524761
9. ID201 robust=0.557148, overall=0.524999
10. ID199 robust=0.557241, overall=0.525173
11. ID194 robust=0.557310, overall=0.525179
12. ID203 robust=0.568318, overall=0.535980
13. ID200 robust=0.568848, overall=0.536190
14. ID198 robust=0.568934, overall=0.536380
15. ID193 robust=0.569090, overall=0.536429
16. ID178 robust=0.571307, overall=0.538638

## Interpretation

- ID203 is compared against ID200 to check whether recurrence/GAF S4 repair adds obvious split fragility.
- ID204 and ID205 are compared against ID201/ID202 at the same delta06/delta08 bridge tiers.
- ID178 remains included as the known submitted public-gap failure reference.
