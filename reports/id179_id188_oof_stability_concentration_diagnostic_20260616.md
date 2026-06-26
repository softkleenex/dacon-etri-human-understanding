# ID179-ID188 OOF stability/concentration diagnostic

Positive improvement means lower per-cell logloss than ID179. This is not public proof; it is a concentration/fragility check.

| id | mean_loss | stab_adj | mean_imp | share_improved | top10_pos_share | subjects+ | targets+ | caution |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 179 | 0.538290 | 0.579290 | 0.000000 | 0.0000 | 0.0000 | 0 | 0 | anchor |
| 180 | 0.529579 | 0.537673 | 0.008711 | 0.5990 | 0.5119 | 10 | 7 | broad |
| 181 | 0.526430 | 0.533910 | 0.011860 | 0.5990 | 0.4996 | 10 | 7 | broad |
| 182 | 0.522909 | 0.530197 | 0.015381 | 0.5990 | 0.4958 | 10 | 7 | broad |
| 183 | 0.519611 | 0.526723 | 0.018679 | 0.5990 | 0.4922 | 10 | 7 | broad |
| 184 | 0.516499 | 0.523446 | 0.021790 | 0.5990 | 0.4889 | 10 | 7 | broad |
| 185 | 0.512540 | 0.519280 | 0.025750 | 0.5990 | 0.4848 | 10 | 7 | broad |
| 186 | 0.506162 | 0.512525 | 0.032128 | 0.5990 | 0.4773 | 10 | 7 | broad |
| 187 | 0.501633 | 0.507770 | 0.036657 | 0.5990 | 0.4727 | 10 | 7 | broad |
| 188 | 0.498094 | 0.504045 | 0.040196 | 0.5990 | 0.4690 | 10 | 7 | broad |

Interpretation: lower local rungs improve OOF broadly, but increasingly inherit non-nested ID172 calibration. Use this with the public-gap adjusted ranking before any submission.
