# Post-ID287 anti-failure anchor blend scan

ID287 failed public badly, so this scan tests tiny negative blends from confirmed anchor ID178 away from ID287. Negative weights are deliberately small; this is a public-risk diagnostic, not a local frontier search.

| variant | kind | w/cap | local CV | Δ local vs ID178 | max Δ ID178 | mean Δ ID178 |
|---|---|---:|---:|---:|---:|---:|
| fixed_weight | prob_pair | -0.005 | 0.539620 | +0.000983 | 0.001492 | 0.000577 |
| targetwise_min_local_damage | targetwise | 0.005 | 0.539620 | +0.000983 | 0.001492 | 0.000577 |
| targetwise_min_local_damage | targetwise | 0.010 | 0.539620 | +0.000983 | 0.001492 | 0.000577 |
| targetwise_min_local_damage | targetwise | 0.015 | 0.539620 | +0.000983 | 0.001492 | 0.000577 |
| targetwise_min_local_damage | targetwise | 0.020 | 0.539620 | +0.000983 | 0.001492 | 0.000577 |
| targetwise_min_local_damage | targetwise | 0.030 | 0.539620 | +0.000983 | 0.001492 | 0.000577 |
| targetwise_min_local_damage | targetwise | 0.040 | 0.539620 | +0.000983 | 0.001492 | 0.000577 |
| targetwise_min_local_damage | targetwise | 0.050 | 0.539620 | +0.000983 | 0.001492 | 0.000577 |
| fixed_weight | logit_pair | -0.005 | 0.539838 | +0.001200 | 0.001737 | 0.000737 |
| fixed_weight | prob_pair | -0.010 | 0.540605 | +0.001968 | 0.002985 | 0.001154 |
| fixed_weight | logit_pair | -0.010 | 0.541043 | +0.002405 | 0.003492 | 0.001476 |
| fixed_weight | prob_pair | -0.015 | 0.541593 | +0.002955 | 0.004477 | 0.001730 |
| fixed_weight | logit_pair | -0.015 | 0.542253 | +0.003615 | 0.005264 | 0.002217 |
| fixed_weight | prob_pair | -0.020 | 0.542584 | +0.003946 | 0.005969 | 0.002307 |
| fixed_weight | logit_pair | -0.020 | 0.543468 | +0.004830 | 0.007054 | 0.002959 |
| fixed_weight | prob_pair | -0.030 | 0.544573 | +0.005935 | 0.008954 | 0.003461 |
| fixed_weight | logit_pair | -0.030 | 0.545912 | +0.007275 | 0.010686 | 0.004447 |
| fixed_weight | prob_pair | -0.040 | 0.546572 | +0.007935 | 0.011938 | 0.004614 |

Promotion reports created for fixed logit -0.02, fixed probability -0.02, and targetwise cap0.02. No DACON submission was made.
