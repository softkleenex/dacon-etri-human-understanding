# Strict demographic/static-feature audit

Created: `2026-06-17 10:16:32` KST

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
| 1 | `compact_plus_demo__extratrees_d5_l8` | 0.616853 | 0.647368 | 0.661295 | 0.681876 | 0.581815 | 0.567935 | 0.539022 | 0.638659 |
| 2 | `base_plus_demo__extratrees_d5_l8` | 0.617823 | 0.651485 | 0.665902 | 0.660668 | 0.574995 | 0.577811 | 0.545104 | 0.648799 |
| 3 | `compact_plus_demo__logreg_l2_c020` | 0.623858 | 0.659085 | 0.676586 | 0.681027 | 0.590932 | 0.577036 | 0.532147 | 0.650194 |
| 4 | `base_no_demo__extratrees_d5_l8` | 0.623895 | 0.655988 | 0.666903 | 0.659374 | 0.577522 | 0.588170 | 0.559314 | 0.659993 |
| 5 | `demo_only__extratrees_d5_l8` | 0.627340 | 0.662828 | 0.688266 | 0.688159 | 0.579042 | 0.574930 | 0.546367 | 0.651788 |
| 6 | `demo_only__logreg_l2_c020` | 0.630142 | 0.659049 | 0.676271 | 0.677402 | 0.607423 | 0.582964 | 0.560561 | 0.647323 |
| 7 | `compact_covariates__extratrees_d5_l8` | 0.634759 | 0.663924 | 0.675545 | 0.665747 | 0.593109 | 0.607545 | 0.564712 | 0.672733 |
| 8 | `compact_covariates__logreg_l2_c020` | 0.639352 | 0.675629 | 0.691664 | 0.663646 | 0.603211 | 0.616520 | 0.545647 | 0.679148 |
| 9 | `base_no_demo__logreg_l2_c020` | 0.944938 | 1.006285 | 0.942982 | 1.057746 | 0.847631 | 0.896984 | 0.857066 | 1.005871 |
| 10 | `base_plus_demo__logreg_l2_c020` | 0.948634 | 1.000338 | 0.943783 | 1.069178 | 0.856604 | 0.899746 | 0.861161 | 1.009626 |

### Best by target

| target | candidate | loss |
|---|---|---:|
| Q1 | `compact_plus_demo__extratrees_d5_l8` | 0.647368 |
| Q2 | `compact_plus_demo__extratrees_d5_l8` | 0.661295 |
| Q3 | `base_no_demo__extratrees_d5_l8` | 0.659374 |
| S1 | `base_plus_demo__extratrees_d5_l8` | 0.574995 |
| S2 | `compact_plus_demo__extratrees_d5_l8` | 0.567935 |
| S3 | `compact_plus_demo__logreg_l2_c020` | 0.532147 |
| S4 | `compact_plus_demo__extratrees_d5_l8` | 0.638659 |

## leave_one_subject_out_stress

| rank | candidate | overall | Q1 | Q2 | Q3 | S1 | S2 | S3 | S4 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `compact_covariates__logreg_l2_c020` | 0.669770 | 0.697381 | 0.718663 | 0.667734 | 0.644188 | 0.677496 | 0.565196 | 0.717731 |
| 2 | `base_no_demo__extratrees_d5_l8` | 0.670115 | 0.694845 | 0.698299 | 0.667725 | 0.627932 | 0.657326 | 0.633986 | 0.710689 |
| 3 | `base_plus_demo__extratrees_d5_l8` | 0.672271 | 0.703407 | 0.700234 | 0.666984 | 0.638637 | 0.652229 | 0.648741 | 0.695666 |
| 4 | `compact_covariates__extratrees_d5_l8` | 0.684416 | 0.716480 | 0.738481 | 0.668035 | 0.630571 | 0.684371 | 0.615003 | 0.737969 |
| 5 | `compact_plus_demo__extratrees_d5_l8` | 0.697267 | 0.708069 | 0.732021 | 0.683735 | 0.689112 | 0.680164 | 0.678336 | 0.709429 |
| 6 | `demo_only__extratrees_d5_l8` | 0.715896 | 0.698631 | 0.709766 | 0.679756 | 0.724588 | 0.716808 | 0.775321 | 0.706401 |
| 7 | `compact_plus_demo__logreg_l2_c020` | 0.731225 | 0.855677 | 0.733161 | 0.704441 | 0.797557 | 0.663090 | 0.582837 | 0.781817 |
| 8 | `demo_only__logreg_l2_c020` | 0.740336 | 0.715496 | 0.690529 | 0.687306 | 0.869766 | 0.722631 | 0.787446 | 0.709179 |
| 9 | `base_no_demo__logreg_l2_c020` | 1.425666 | 2.008450 | 1.541761 | 1.447116 | 1.024983 | 1.131233 | 1.318796 | 1.507324 |
| 10 | `base_plus_demo__logreg_l2_c020` | 1.426604 | 1.961107 | 1.550337 | 1.445687 | 1.055876 | 1.155727 | 1.299195 | 1.518298 |

### Best by target

| target | candidate | loss |
|---|---|---:|
| Q1 | `base_no_demo__extratrees_d5_l8` | 0.694845 |
| Q2 | `demo_only__logreg_l2_c020` | 0.690529 |
| Q3 | `base_plus_demo__extratrees_d5_l8` | 0.666984 |
| S1 | `base_no_demo__extratrees_d5_l8` | 0.627932 |
| S2 | `base_plus_demo__extratrees_d5_l8` | 0.652229 |
| S3 | `compact_covariates__logreg_l2_c020` | 0.565196 |
| S4 | `base_plus_demo__extratrees_d5_l8` | 0.695666 |

## Interpretation

- Best subject-hole candidate is compact_plus_demo__extratrees_d5_l8 with overall 0.616853; this split can reward subject-static signals because each subject appears in both train and validation holes.
- Best LOSO stress candidate is compact_covariates__logreg_l2_c020 with overall 0.669770; large degradation here means demographics are not a robust standalone cross-subject signal.
- No submission artifact is promoted by this audit. Use target-level wins only as hypotheses for a future strict branch.
