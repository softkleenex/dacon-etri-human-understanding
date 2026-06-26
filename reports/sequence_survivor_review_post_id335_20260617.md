# Sequence survivor review — post ID335

Created: `2026-06-17 23:16:29 UTC+09:00`

This is report-only. No CSV, notebook, base artifact, submission-log row, or DACON submission was created.

## Candidate comparison

| rank | id | local | robust | gain vs 178 | robust gain | max Δ178 | proj failed | worst target | read |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 179 | 0.538290 | 0.563975 | +0.000348 | +0.000218 | 0.051824 | +0.0011 | S4 `0.602685` | cleanest low-movement same-family review anchor |
| 2 | 178 | 0.538638 | 0.564192 | +0.000000 | +0.000000 | 0.000000 | +0.0000 | S4 `0.602700` | reference/watch-only |
| 3 | 177 | 0.540607 | 0.566010 | -0.001969 | -0.001818 | 0.115165 | +0.0012 | S4 `0.603270` | reference/watch-only |
| 4 | 198 | 0.536380 | 0.562109 | +0.002258 | +0.002083 | 0.121839 | -0.0102 | S4 `0.602685` | watch-only: more movement than ID179 for tiny gain |
| 5 | 193 | 0.536429 | 0.562220 | +0.002209 | +0.001972 | 0.121839 | -0.0105 | S4 `0.602685` | watch-only: more movement than ID179 for tiny gain |
| 6 | 335 | 0.539690 | 0.565340 | -0.001052 | -0.001147 | 0.111949 | +0.0032 | S4 `0.604213` | reference/watch-only |
| 7 | 191 | 0.538009 | 0.564015 | +0.000628 | +0.000177 | 0.121839 | +0.0024 | S4 `0.602685` | watch-only: more movement than ID179 for tiny gain |
| 8 | 334 | 0.540347 | 0.566401 | -0.001710 | -0.002208 | 0.119120 | +0.0039 | S4 `0.608814` | reference/watch-only |
| 9 | 190 | 0.537956 | 0.564171 | +0.000682 | +0.000021 | 0.127403 | +0.0023 | S4 `0.602700` | watch-only: more movement than ID179 for tiny gain |
| 10 | 206 | 0.535764 | 0.561409 | +0.002874 | +0.002783 | 0.136414 | -0.0141 | S4 `0.600718` | watch-only: more movement than ID179 for tiny gain |
| 11 | 203 | 0.535980 | 0.561523 | +0.002658 | +0.002669 | 0.136414 | -0.0127 | S4 `0.601188` | watch-only: more movement than ID179 for tiny gain |
| 12 | 200 | 0.536190 | 0.561988 | +0.002448 | +0.002204 | 0.136414 | -0.0129 | S4 `0.602656` | watch-only: more movement than ID179 for tiny gain |
| 13 | 180 | 0.529579 | 0.555022 | +0.009059 | +0.009171 | 0.072340 | +0.0348 | S4 `0.594836` | reject: failed-direction projection too high |

## Target deltas versus ID178

Negative is locally better than ID178.

| id | Q1 | Q2 | Q3 | S1 | S2 | S3 | S4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 179 | -0.000026 | +0.000000 | -0.000443 | -0.001916 | -0.000034 | +0.000000 | -0.000015 |
| 178 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 177 | +0.002600 | +0.002579 | +0.002645 | +0.000585 | +0.004806 | +0.000000 | +0.000570 |
| 198 | -0.002658 | -0.005991 | -0.002886 | -0.003880 | -0.000034 | -0.000340 | -0.000015 |
| 193 | -0.002658 | -0.005991 | -0.002886 | -0.003880 | -0.000034 | +0.000000 | -0.000015 |
| 335 | +0.000000 | +0.000000 | +0.000000 | +0.005852 | +0.000000 | +0.000000 | +0.001513 |
| 191 | -0.000026 | +0.000000 | -0.000443 | -0.003880 | -0.000034 | +0.000000 | -0.000015 |
| 334 | +0.000000 | +0.000000 | +0.000000 | +0.005852 | +0.000000 | +0.000000 | +0.006114 |
| 190 | +0.000000 | +0.000000 | +0.000000 | -0.004773 | +0.000000 | +0.000000 | +0.000000 |
| 206 | -0.003658 | -0.005991 | -0.004213 | -0.003893 | -0.000040 | -0.000340 | -0.001982 |
| 203 | -0.003658 | -0.005991 | -0.003175 | -0.003893 | -0.000034 | -0.000340 | -0.001511 |
| 200 | -0.003658 | -0.005991 | -0.003175 | -0.003893 | -0.000034 | -0.000340 | -0.000043 |
| 180 | -0.010791 | -0.011516 | -0.010393 | -0.008733 | -0.010438 | -0.003678 | -0.007864 |

## Recommendation

ID179 remains the cleanest low-movement same-family review anchor after excluding the ID242 chain. Several sequence descendants improve local CV slightly more, but they move much farther from the confirmed ID178 public anchor and still use non-strict/OOF-selected sequence overlays. This review does not authorize a DACON submission.
