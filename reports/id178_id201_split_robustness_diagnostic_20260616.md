# ID178-ID201 split robustness diagnostic

Lower is better. Robustness score penalizes worst subject, worst date quartile, worst target, and late-date degradation. This is not a public LB prediction.

| id | overall | robust_score | worst_subject | worst_subj_loss | worst_date_bin | worst_date_loss | late-early | worst_target | worst_target_loss |
|---:|---:|---:|---|---:|---|---:|---:|---|---:|
| 178 | 0.538638 | 0.571307 | id10 | 0.633714 | q4 | 0.564958 | 0.065989 | S4 | 0.602700 |
| 193 | 0.536429 | 0.569090 | id10 | 0.628175 | q4 | 0.564927 | 0.068231 | S4 | 0.602685 |
| 198 | 0.536380 | 0.568934 | id10 | 0.627613 | q4 | 0.564776 | 0.068350 | S4 | 0.602685 |
| 200 | 0.536190 | 0.568848 | id10 | 0.627898 | q4 | 0.564598 | 0.068174 | S4 | 0.602656 |
| 194 | 0.525179 | 0.557310 | id10 | 0.618311 | q4 | 0.550505 | 0.061130 | S4 | 0.591671 |
| 199 | 0.525173 | 0.557241 | id10 | 0.617975 | q4 | 0.550468 | 0.061279 | S4 | 0.591671 |
| 201 | 0.524999 | 0.557148 | id10 | 0.618189 | q4 | 0.550305 | 0.061075 | S4 | 0.591609 |
| 195 | 0.521836 | 0.553822 | id10 | 0.615486 | q4 | 0.546180 | 0.058943 | S4 | 0.588410 |
| 196 | 0.518710 | 0.550576 | id10 | 0.612916 | q4 | 0.542107 | 0.056840 | S4 | 0.585438 |
| 197 | 0.515748 | 0.547507 | id10 | 0.610521 | q4 | 0.538219 | 0.054789 | S4 | 0.582688 |
| 189 | 0.492595 | 0.525866 | id10 | 0.599849 | q2 | 0.510464 | 0.032476 | S4 | 0.567760 |
| 172 | 0.492762 | 0.526318 | id10 | 0.601349 | q2 | 0.509990 | 0.032761 | S4 | 0.568928 |

## Rank by robustness score
1. ID189 robust=0.525866, overall=0.492595
2. ID172 robust=0.526318, overall=0.492762
3. ID197 robust=0.547507, overall=0.515748
4. ID196 robust=0.550576, overall=0.518710
5. ID195 robust=0.553822, overall=0.521836
6. ID201 robust=0.557148, overall=0.524999
7. ID199 robust=0.557241, overall=0.525173
8. ID194 robust=0.557310, overall=0.525179
9. ID200 robust=0.568848, overall=0.536190
10. ID198 robust=0.568934, overall=0.536380
11. ID193 robust=0.569090, overall=0.536429
12. ID178 robust=0.571307, overall=0.538638

## Interpretation

- ID200 is compared against ID193/ID198 to see whether the cleaner SAX/PAA gains add obvious split fragility.
- ID201 is compared against ID194/ID199 at the same moderate-low ID172 bridge tier.
- ID178 remains included as the known submitted public-gap failure reference.
