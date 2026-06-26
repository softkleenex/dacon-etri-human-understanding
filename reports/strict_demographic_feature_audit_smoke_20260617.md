# Strict demographic/static-feature audit

Created: `2026-06-17 01:49:06` KST

This is a report-only diagnostic. It creates no submission file and performs no DACON submission.

## Candidate feature sets

| candidate | model | n features |
|---|---|---:|
| `compact_plus_demo__extratrees_d5_l8` | `extratrees_d5_l8` | 17 |
| `base_plus_demo__extratrees_d5_l8` | `extratrees_d5_l8` | 289 |
| `compact_plus_demo__logreg_l2_c020` | `logreg_l2_c020` | 17 |
| `base_no_demo__extratrees_d5_l8` | `extratrees_d5_l8` | 279 |
| `demo_only__extratrees_d5_l8` | `extratrees_d5_l8` | 10 |
| `demo_only__logreg_l2_c020` | `logreg_l2_c020` | 10 |
| `compact_covariates__extratrees_d5_l8` | `extratrees_d5_l8` | 7 |
| `compact_covariates__logreg_l2_c020` | `logreg_l2_c020` | 7 |
| `base_no_demo__logreg_l2_c020` | `logreg_l2_c020` | 279 |
| `base_plus_demo__logreg_l2_c020` | `logreg_l2_c020` | 289 |

## subject_hole_temporal

| rank | candidate | overall | Q1 | Q2 | Q3 | S1 | S2 | S3 | S4 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `compact_plus_demo__extratrees_d5_l8` | 0.617693 | 0.648480 | 0.664584 | 0.682165 | 0.583783 | 0.565933 | 0.537926 | 0.640978 |
| 2 | `base_plus_demo__extratrees_d5_l8` | 0.619591 | 0.651231 | 0.665713 | 0.661133 | 0.576841 | 0.579407 | 0.550770 | 0.652041 |
| 3 | `compact_plus_demo__logreg_l2_c020` | 0.623858 | 0.659085 | 0.676586 | 0.681027 | 0.590932 | 0.577036 | 0.532147 | 0.650194 |
| 4 | `base_no_demo__extratrees_d5_l8` | 0.624720 | 0.659920 | 0.664744 | 0.662652 | 0.582429 | 0.588120 | 0.554985 | 0.660189 |
| 5 | `demo_only__extratrees_d5_l8` | 0.626920 | 0.662834 | 0.685983 | 0.687883 | 0.579824 | 0.574906 | 0.545152 | 0.651858 |
| 6 | `demo_only__logreg_l2_c020` | 0.630142 | 0.659049 | 0.676271 | 0.677402 | 0.607423 | 0.582964 | 0.560561 | 0.647323 |
| 7 | `compact_covariates__extratrees_d5_l8` | 0.634858 | 0.665023 | 0.674415 | 0.666169 | 0.596209 | 0.609792 | 0.562474 | 0.669922 |
| 8 | `compact_covariates__logreg_l2_c020` | 0.639352 | 0.675629 | 0.691664 | 0.663646 | 0.603211 | 0.616520 | 0.545647 | 0.679148 |
| 9 | `base_no_demo__logreg_l2_c020` | 0.944938 | 1.006285 | 0.942982 | 1.057746 | 0.847631 | 0.896984 | 0.857066 | 1.005871 |
| 10 | `base_plus_demo__logreg_l2_c020` | 0.948634 | 1.000338 | 0.943783 | 1.069178 | 0.856604 | 0.899746 | 0.861161 | 1.009626 |

### Best by target

| target | candidate | loss |
|---|---|---:|
| Q1 | `compact_plus_demo__extratrees_d5_l8` | 0.648480 |
| Q2 | `compact_plus_demo__extratrees_d5_l8` | 0.664584 |
| Q3 | `base_plus_demo__extratrees_d5_l8` | 0.661133 |
| S1 | `base_plus_demo__extratrees_d5_l8` | 0.576841 |
| S2 | `compact_plus_demo__extratrees_d5_l8` | 0.565933 |
| S3 | `compact_plus_demo__logreg_l2_c020` | 0.532147 |
| S4 | `compact_plus_demo__extratrees_d5_l8` | 0.640978 |

## leave_one_subject_out_stress

| rank | candidate | overall | Q1 | Q2 | Q3 | S1 | S2 | S3 | S4 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `compact_covariates__logreg_l2_c020` | 0.669770 | 0.697381 | 0.718663 | 0.667734 | 0.644188 | 0.677496 | 0.565196 | 0.717731 |
| 2 | `base_no_demo__extratrees_d5_l8` | 0.670567 | 0.698583 | 0.696112 | 0.668971 | 0.627926 | 0.653355 | 0.633133 | 0.715891 |
| 3 | `base_plus_demo__extratrees_d5_l8` | 0.673768 | 0.700283 | 0.698138 | 0.667013 | 0.649328 | 0.655532 | 0.645511 | 0.700573 |
| 4 | `compact_covariates__extratrees_d5_l8` | 0.683952 | 0.719927 | 0.737638 | 0.668145 | 0.629553 | 0.686570 | 0.612796 | 0.733032 |
| 5 | `compact_plus_demo__extratrees_d5_l8` | 0.698296 | 0.711228 | 0.740932 | 0.676950 | 0.696153 | 0.683155 | 0.673449 | 0.706207 |
| 6 | `demo_only__extratrees_d5_l8` | 0.711685 | 0.700741 | 0.702815 | 0.677945 | 0.716560 | 0.709945 | 0.767270 | 0.706519 |
| 7 | `compact_plus_demo__logreg_l2_c020` | 0.731225 | 0.855677 | 0.733161 | 0.704441 | 0.797557 | 0.663090 | 0.582837 | 0.781817 |
| 8 | `demo_only__logreg_l2_c020` | 0.740336 | 0.715496 | 0.690529 | 0.687306 | 0.869766 | 0.722631 | 0.787446 | 0.709179 |
| 9 | `base_no_demo__logreg_l2_c020` | 1.425666 | 2.008450 | 1.541761 | 1.447116 | 1.024983 | 1.131233 | 1.318796 | 1.507324 |
| 10 | `base_plus_demo__logreg_l2_c020` | 1.426604 | 1.961107 | 1.550337 | 1.445687 | 1.055876 | 1.155727 | 1.299195 | 1.518298 |

### Best by target

| target | candidate | loss |
|---|---|---:|
| Q1 | `compact_covariates__logreg_l2_c020` | 0.697381 |
| Q2 | `demo_only__logreg_l2_c020` | 0.690529 |
| Q3 | `base_plus_demo__extratrees_d5_l8` | 0.667013 |
| S1 | `base_no_demo__extratrees_d5_l8` | 0.627926 |
| S2 | `base_no_demo__extratrees_d5_l8` | 0.653355 |
| S3 | `compact_covariates__logreg_l2_c020` | 0.565196 |
| S4 | `base_plus_demo__extratrees_d5_l8` | 0.700573 |

## Interpretation

- Best subject-hole candidate is compact_plus_demo__extratrees_d5_l8 with overall 0.617693; this split can reward subject-static signals because each subject appears in both train and validation holes.
- Best LOSO stress candidate is compact_covariates__logreg_l2_c020 with overall 0.669770; large degradation here means demographics are not a robust standalone cross-subject signal.
- No submission artifact is promoted by this audit. Use target-level wins only as hypotheses for a future strict branch.
