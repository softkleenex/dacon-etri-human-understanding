# Post-ID285 temperature-only fixed scan

This scan uses fixed logit temperature/shift transforms only. It avoids isotonic and quantile calibrators, so the remaining risk is target-wise full-OOF selection rather than fitted calibration leakage.

## Top balanced candidates under local<0.44 and max Δ vs ID178<=0.30

| base | local | score proxy | stability proxy | max Δ base | max Δ ID178 | changed | methods |
|---:|---:|---:|---:|---:|---:|---:|---|
| 282 | 0.378934 | 0.441707 | 0.493669 | 0.049781 | 0.298460 | 6 | `Q1=temp_t0.85_s0.00,Q2=temp_t0.80_s0.00,Q3=temp_t0.80_s0.00,S1=temp_t0.80_s0.00,S2=temp_t0.80_s0.00,S3=base,S4=temp_t0.85_s0.00` |
| 280 | 0.395368 | 0.458141 | 0.496510 | 0.041869 | 0.259624 | 6 | `Q1=temp_t0.85_s0.00,Q2=temp_t0.85_s-0.04,Q3=temp_t0.85_s0.00,S1=temp_t0.85_s0.00,S2=temp_t0.85_s0.00,S3=base,S4=temp_t0.85_s0.00` |
| 281 | 0.381827 | 0.444600 | 0.496937 | 0.071693 | 0.299534 | 6 | `Q1=temp_t0.80_s-0.04,Q2=temp_t0.75_s-0.06,Q3=temp_t0.75_s-0.04,S1=temp_t0.75_s-0.04,S2=temp_t0.75_s-0.04,S3=base,S4=temp_t0.80_s-0.04` |
| 281 | 0.383885 | 0.446658 | 0.498995 | 0.057850 | 0.299534 | 6 | `Q1=temp_t0.80_s-0.04,Q2=temp_t0.80_s-0.06,Q3=temp_t0.80_s-0.06,S1=temp_t0.80_s-0.06,S2=temp_t0.80_s-0.06,S3=base,S4=temp_t0.80_s-0.04` |
| 279 | 0.398269 | 0.461042 | 0.499088 | 0.069100 | 0.258700 | 6 | `Q1=temp_t0.80_s-0.04,Q2=temp_t0.75_s0.00,Q3=temp_t0.75_s0.00,S1=temp_t0.75_s-0.04,S2=temp_t0.75_s0.00,S3=base,S4=temp_t0.80_s-0.04` |
| 279 | 0.399454 | 0.462227 | 0.499624 | 0.057850 | 0.256848 | 6 | `Q1=temp_t0.80_s-0.04,Q2=temp_t0.80_s-0.06,Q3=temp_t0.80_s-0.06,S1=temp_t0.80_s-0.06,S2=temp_t0.80_s-0.06,S3=base,S4=temp_t0.80_s-0.04` |
| 280 | 0.391441 | 0.454214 | 0.500553 | 0.057850 | 0.282396 | 6 | `Q1=temp_t0.80_s-0.04,Q2=temp_t0.80_s-0.06,Q3=temp_t0.80_s-0.06,S1=temp_t0.80_s-0.06,S2=temp_t0.80_s-0.06,S3=base,S4=temp_t0.80_s-0.06` |
| 279 | 0.413433 | 0.476206 | 0.500697 | 0.011481 | 0.219973 | 1 | `Q1=base,Q2=temp_t0.95_s0.00,Q3=base,S1=base,S2=base,S3=base,S4=base` |
| 281 | 0.399583 | 0.462356 | 0.500796 | 0.000000 | 0.259827 | 0 | `Q1=base,Q2=base,Q3=base,S1=base,S2=base,S3=base,S4=base` |
| 279 | 0.399412 | 0.462185 | 0.501072 | 0.057850 | 0.261107 | 6 | `Q1=temp_t0.80_s-0.04,Q2=temp_t0.80_s-0.06,Q3=temp_t0.80_s-0.06,S1=temp_t0.80_s-0.06,S2=temp_t0.80_s-0.06,S3=base,S4=temp_t0.80_s-0.06` |
| 280 | 0.386714 | 0.449488 | 0.501181 | 0.088958 | 0.297697 | 6 | `Q1=temp_t0.70_s0.00,Q2=temp_t0.70_s-0.08,Q3=temp_t0.70_s-0.06,S1=temp_t0.75_s-0.12,S2=temp_t0.70_s-0.06,S3=base,S4=temp_t0.75_s-0.08` |
| 278 | 0.386397 | 0.449170 | 0.501450 | 0.023569 | 0.299373 | 3 | `Q1=temp_t0.90_s0.00,Q2=base,Q3=base,S1=base,S2=temp_t0.95_s0.04,S3=base,S4=temp_t0.90_s0.00` |

## Top by stability-veto proxy

| base | local | score proxy | stability proxy | max Δ base | max Δ ID178 | changed | methods |
|---:|---:|---:|---:|---:|---:|---:|---|
| 282 | 0.378934 | 0.441707 | 0.493669 | 0.049781 | 0.298460 | 6 | `Q1=temp_t0.85_s0.00,Q2=temp_t0.80_s0.00,Q3=temp_t0.80_s0.00,S1=temp_t0.80_s0.00,S2=temp_t0.80_s0.00,S3=base,S4=temp_t0.85_s0.00` |
| 280 | 0.395368 | 0.458141 | 0.496510 | 0.041869 | 0.259624 | 6 | `Q1=temp_t0.85_s0.00,Q2=temp_t0.85_s-0.04,Q3=temp_t0.85_s0.00,S1=temp_t0.85_s0.00,S2=temp_t0.85_s0.00,S3=base,S4=temp_t0.85_s0.00` |
| 281 | 0.381827 | 0.444600 | 0.496937 | 0.071693 | 0.299534 | 6 | `Q1=temp_t0.80_s-0.04,Q2=temp_t0.75_s-0.06,Q3=temp_t0.75_s-0.04,S1=temp_t0.75_s-0.04,S2=temp_t0.75_s-0.04,S3=base,S4=temp_t0.80_s-0.04` |
| 278 | 0.372773 | 0.435547 | 0.498594 | 0.057850 | 0.330135 | 6 | `Q1=temp_t0.80_s-0.06,Q2=temp_t0.80_s-0.06,Q3=temp_t0.80_s-0.06,S1=temp_t0.80_s-0.06,S2=temp_t0.80_s-0.06,S3=base,S4=temp_t0.80_s-0.06` |
| 281 | 0.383885 | 0.446658 | 0.498995 | 0.057850 | 0.299534 | 6 | `Q1=temp_t0.80_s-0.04,Q2=temp_t0.80_s-0.06,Q3=temp_t0.80_s-0.06,S1=temp_t0.80_s-0.06,S2=temp_t0.80_s-0.06,S3=base,S4=temp_t0.80_s-0.04` |
| 279 | 0.398269 | 0.461042 | 0.499088 | 0.069100 | 0.258700 | 6 | `Q1=temp_t0.80_s-0.04,Q2=temp_t0.75_s0.00,Q3=temp_t0.75_s0.00,S1=temp_t0.75_s-0.04,S2=temp_t0.75_s0.00,S3=base,S4=temp_t0.80_s-0.04` |
| 279 | 0.399454 | 0.462227 | 0.499624 | 0.057850 | 0.256848 | 6 | `Q1=temp_t0.80_s-0.04,Q2=temp_t0.80_s-0.06,Q3=temp_t0.80_s-0.06,S1=temp_t0.80_s-0.06,S2=temp_t0.80_s-0.06,S3=base,S4=temp_t0.80_s-0.04` |
| 282 | 0.377266 | 0.440039 | 0.499678 | 0.057850 | 0.320397 | 6 | `Q1=temp_t0.80_s-0.04,Q2=temp_t0.80_s-0.06,Q3=temp_t0.80_s-0.06,S1=temp_t0.80_s-0.06,S2=temp_t0.80_s-0.06,S3=base,S4=temp_t0.80_s-0.06` |
| 281 | 0.383839 | 0.446612 | 0.500401 | 0.057850 | 0.303683 | 6 | `Q1=temp_t0.80_s-0.04,Q2=temp_t0.80_s-0.06,Q3=temp_t0.80_s-0.06,S1=temp_t0.80_s-0.06,S2=temp_t0.80_s-0.06,S3=base,S4=temp_t0.80_s-0.06` |
| 280 | 0.391441 | 0.454214 | 0.500553 | 0.057850 | 0.282396 | 6 | `Q1=temp_t0.80_s-0.04,Q2=temp_t0.80_s-0.06,Q3=temp_t0.80_s-0.06,S1=temp_t0.80_s-0.06,S2=temp_t0.80_s-0.06,S3=base,S4=temp_t0.80_s-0.06` |
| 279 | 0.413433 | 0.476206 | 0.500697 | 0.011481 | 0.219973 | 1 | `Q1=base,Q2=temp_t0.95_s0.00,Q3=base,S1=base,S2=base,S3=base,S4=base` |
| 281 | 0.399583 | 0.462356 | 0.500796 | 0.000000 | 0.259827 | 0 | `Q1=base,Q2=base,Q3=base,S1=base,S2=base,S3=base,S4=base` |

## Top by raw local

| base | local | score proxy | stability proxy | max Δ base | max Δ ID178 | changed | methods |
|---:|---:|---:|---:|---:|---:|---:|---|
| 278 | 0.362956 | 0.425729 | 0.520116 | 0.132675 | 0.401007 | 6 | `Q1=temp_t0.70_s-0.06,Q2=temp_t0.60_s-0.18,Q3=temp_t0.60_s-0.18,S1=temp_t0.60_s-0.18,S2=temp_t0.60_s-0.18,S3=base,S4=temp_t0.70_s-0.18` |
| 278 | 0.364245 | 0.427018 | 0.511356 | 0.118892 | 0.380170 | 6 | `Q1=temp_t0.70_s-0.06,Q2=temp_t0.60_s-0.06,Q3=temp_t0.60_s-0.06,S1=temp_t0.70_s-0.18,S2=temp_t0.60_s-0.06,S3=base,S4=temp_t0.70_s-0.18` |
| 278 | 0.366459 | 0.429232 | 0.503742 | 0.094151 | 0.362884 | 6 | `Q1=temp_t0.70_s-0.06,Q2=temp_t0.70_s-0.12,Q3=temp_t0.70_s-0.12,S1=temp_t0.70_s-0.12,S2=temp_t0.70_s-0.12,S3=base,S4=temp_t0.70_s-0.12` |
| 278 | 0.366847 | 0.429621 | 0.502680 | 0.094149 | 0.358739 | 6 | `Q1=temp_t0.70_s-0.06,Q2=temp_t0.70_s-0.12,Q3=temp_t0.70_s-0.08,S1=temp_t0.70_s-0.08,S2=temp_t0.70_s-0.08,S3=base,S4=temp_t0.70_s-0.08` |
| 282 | 0.367879 | 0.430652 | 0.519060 | 0.132676 | 0.383921 | 6 | `Q1=temp_t0.70_s-0.06,Q2=temp_t0.60_s-0.18,Q3=temp_t0.60_s-0.18,S1=temp_t0.60_s-0.18,S2=temp_t0.60_s-0.18,S3=base,S4=temp_t0.70_s-0.18` |
| 278 | 0.368136 | 0.430910 | 0.501679 | 0.079876 | 0.352199 | 6 | `Q1=temp_t0.70_s0.00,Q2=temp_t0.70_s0.00,Q3=temp_t0.70_s0.00,S1=temp_t0.75_s-0.12,S2=temp_t0.70_s0.00,S3=base,S4=temp_t0.75_s-0.12` |
| 282 | 0.369141 | 0.431914 | 0.511724 | 0.118891 | 0.367232 | 6 | `Q1=temp_t0.70_s-0.06,Q2=temp_t0.60_s-0.06,Q3=temp_t0.60_s-0.06,S1=temp_t0.70_s-0.18,S2=temp_t0.60_s-0.06,S3=base,S4=temp_t0.70_s-0.18` |
| 282 | 0.369370 | 0.432143 | 0.509293 | 0.118888 | 0.359634 | 6 | `Q1=temp_t0.70_s-0.06,Q2=temp_t0.60_s-0.06,Q3=temp_t0.60_s-0.04,S1=temp_t0.70_s-0.18,S2=temp_t0.60_s-0.04,S3=base,S4=temp_t0.70_s-0.12` |
| 282 | 0.371063 | 0.433836 | 0.506110 | 0.094152 | 0.356496 | 6 | `Q1=temp_t0.70_s-0.06,Q2=temp_t0.70_s-0.12,Q3=temp_t0.70_s-0.12,S1=temp_t0.70_s-0.12,S2=temp_t0.70_s-0.12,S3=base,S4=temp_t0.70_s-0.12` |
| 282 | 0.372753 | 0.435526 | 0.503312 | 0.079889 | 0.343676 | 6 | `Q1=temp_t0.70_s0.00,Q2=temp_t0.70_s0.00,Q3=temp_t0.70_s0.00,S1=temp_t0.75_s-0.12,S2=temp_t0.70_s0.00,S3=base,S4=temp_t0.75_s-0.12` |
| 278 | 0.372773 | 0.435547 | 0.498594 | 0.057850 | 0.330135 | 6 | `Q1=temp_t0.80_s-0.06,Q2=temp_t0.80_s-0.06,Q3=temp_t0.80_s-0.06,S1=temp_t0.80_s-0.06,S2=temp_t0.80_s-0.06,S3=base,S4=temp_t0.80_s-0.06` |
| 281 | 0.375014 | 0.437787 | 0.524529 | 0.132677 | 0.379162 | 6 | `Q1=temp_t0.70_s-0.04,Q2=temp_t0.60_s-0.18,Q3=temp_t0.60_s-0.18,S1=temp_t0.70_s-0.18,S2=temp_t0.60_s-0.18,S3=base,S4=temp_t0.60_s-0.18` |

Practical read: prioritize balanced fixed-transform cards before another isotonic/quantile stress probe. No DACON submission was made.
