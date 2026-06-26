# ID340 sequence subset geometry scan

Created: `2026-06-18 03:38:15 UTC+09:00`

This is report-only. It scans smaller target subsets/weights for the same sequence-summary source and creates no CSV, base artifact, notebook/history artifact, submission-log row, or DACON submission.

- ID337 proxy: `0.605022`
- ID339 proxy: `0.609547`

## Top variants

| rank | variant | targets | primary pass | full geometry pass | proxy | Δ proxy vs 337 | max Δ178 | failed proj | subject-hole |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | `q2_only_w0.01` | `['Q2']` | `True` | `True` | 0.604062 | -0.000960 | 0.008102 | -0.000339 | 0.538426 |
| 2 | `q23_w0.01` | `['Q2', 'Q3']` | `True` | `True` | 0.604184 | -0.000838 | 0.008670 | -0.000616 | 0.538442 |
| 3 | `q123_w0.01` | `['Q1', 'Q2', 'Q3']` | `True` | `True` | 0.604307 | -0.000715 | 0.009435 | -0.001217 | 0.538350 |
| 4 | `q2_only_w0.02` | `['Q2']` | `True` | `True` | 0.604384 | -0.000638 | 0.016204 | -0.000678 | 0.538227 |
| 5 | `q123_s1_w0.01` | `['Q1', 'Q2', 'Q3', 'S1']` | `True` | `True` | 0.604393 | -0.000630 | 0.009435 | -0.001440 | 0.538288 |
| 6 | `q123_s1_s4_w0.01` | `['Q1', 'Q2', 'Q3', 'S1', 'S4']` | `True` | `True` | 0.604491 | -0.000532 | 0.009435 | -0.001263 | 0.538277 |
| 7 | `q23_w0.02` | `['Q2', 'Q3']` | `True` | `True` | 0.604627 | -0.000396 | 0.017341 | -0.001231 | 0.538268 |
| 8 | `all_w0.01` | `['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']` | `True` | `True` | 0.604692 | -0.000330 | 0.009522 | -0.001592 | 0.538442 |
| 9 | `q2_only_w0.03` | `['Q2']` | `True` | `True` | 0.604732 | -0.000290 | 0.024305 | -0.001017 | 0.538039 |
| 10 | `q123_w0.02` | `['Q1', 'Q2', 'Q3']` | `True` | `True` | 0.604875 | -0.000147 | 0.018871 | -0.002435 | 0.538101 |
| 11 | `q123_s1_w0.02` | `['Q1', 'Q2', 'Q3', 'S1']` | `True` | `False` | 0.605058 | +0.000036 | 0.018871 | -0.002881 | 0.538000 |
| 12 | `q23_w0.03` | `['Q2', 'Q3']` | `True` | `False` | 0.605082 | +0.000059 | 0.026011 | -0.001847 | 0.538116 |
| 13 | `q123_s1_s4_w0.02` | `['Q1', 'Q2', 'Q3', 'S1', 'S4']` | `True` | `False` | 0.605258 | +0.000236 | 0.018871 | -0.002526 | 0.537987 |
| 14 | `q2_only_w0.05` | `['Q2']` | `True` | `False` | 0.605444 | +0.000422 | 0.040509 | -0.001694 | 0.537698 |
| 15 | `q123_w0.03` | `['Q1', 'Q2', 'Q3']` | `True` | `False` | 0.605457 | +0.000435 | 0.028306 | -0.003652 | 0.537890 |
| 16 | `all_w0.02` | `['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']` | `True` | `False` | 0.605682 | +0.000660 | 0.019044 | -0.003184 | 0.538348 |
| 17 | `q123_s1_w0.03` | `['Q1', 'Q2', 'Q3', 'S1']` | `True` | `False` | 0.605743 | +0.000721 | 0.028306 | -0.004321 | 0.537767 |
| 18 | `q123_s1_s4_w0.03` | `['Q1', 'Q2', 'Q3', 'S1', 'S4']` | `True` | `False` | 0.606049 | +0.001027 | 0.028306 | -0.003789 | 0.537759 |
| 19 | `q23_w0.05` | `['Q2', 'Q3']` | `True` | `False` | 0.606056 | +0.001033 | 0.043351 | -0.003078 | 0.537876 |
| 20 | `q2_only_w0.07` | `['Q2']` | `True` | `False` | 0.606174 | +0.001151 | 0.056712 | -0.002372 | 0.537399 |

## Best variant split rows

Best variant: `q2_only_w0.01`

| split | primary | loss | ID178 | Δ vs ID178 | pass |
|---|---|---:|---:|---:|---|
| `subject_hole_5` | `True` | 0.538426 | 0.538638 | -0.000212 | `True` |
| `subject_timeseries_5` | `False` | 0.539612 | 0.539747 | -0.000135 | `True` |
| `actual_test_band_holdout` | `True` | 0.541105 | 0.541604 | -0.000498 | `True` |
| `test_boundary_neighbor2_holdout` | `True` | 0.585859 | 0.586290 | -0.000430 | `True` |
| `test_boundary_neighbor5_holdout` | `True` | 0.557736 | 0.557986 | -0.000250 | `True` |
| `loso_10` | `True` | 0.538520 | 0.538638 | -0.000117 | `True` |

## Decision

- Has variant for artifact review: `True`
- Promote submission candidate now: **no**
- Create ID340 artifact now: **no**
- Submit now: **no**

A variant can move to artifact-review only if it passes all primary split lenses and improves the conservative proxy versus ID337/ID339. This scan still creates no artifact.

