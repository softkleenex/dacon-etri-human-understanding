# Post-ID282 second-stage calibration scan

Full-OOF calibration diagnostic on selected bridge/frontier bases. Lower local is better, but calibration is high-risk and non-nested.

| base | variant | local | max Δ vs base | max Δ vs ID178 | mean Δ vs ID178 | methods | report |
|---:|---|---:|---:|---:|---:|---|---|
| 272 | raw_best | 0.349101 | 0.719710 | 0.727165 | 0.144967 | `Q1=q120,Q2=q160,Q3=q160,S1=q120,S2=q160,S3=q160,S4=q160` | `reports/post282_second_stage_calibration_scan_base272_20260616.json` |
| 272 | base_delta_guard | 0.371170 | 0.552767 | 0.559886 | 0.140310 | `Q1=q80,Q2=isotonic,Q3=q80,S1=q80,S2=isotonic,S3=q160,S4=q160` | `reports/post282_second_stage_calibration_scan_base272_20260616.json` |
| 272 | id178_delta_guard | 0.360855 | 0.596566 | 0.559886 | 0.142581 | `Q1=q160,Q2=q160,Q3=q80,S1=q80,S2=isotonic,S3=q160,S4=q160` | `reports/post282_second_stage_calibration_scan_base272_20260616.json` |
| 272 | temp_quantile_guard | 0.440206 | 0.071721 | 0.200699 | 0.076267 | `Q1=temp_t0.85_s0.00,Q2=temp_t0.75_s-0.06,Q3=temp_t0.85_s0.00,S1=base,S2=temp_t0.85_s0.00,S3=base,S4=temp_t0.85_s0.00` | `reports/post282_second_stage_calibration_scan_base272_20260616.json` |
| 272 | ultra_guard | 0.349137 | 0.719710 | 0.727165 | 0.144749 | `Q1=q160,Q2=q160,Q3=q160,S1=q120,S2=q160,S3=q160,S4=q160` | `reports/post282_second_stage_calibration_scan_base272_20260616.json` |
| 279 | raw_best | 0.321823 | 0.562570 | 0.632785 | 0.162517 | `Q1=q160,Q2=q160,Q3=q160,S1=q120,S2=q160,S3=q160,S4=q160` | `reports/post282_second_stage_calibration_scan_base279_20260616.json` |
| 279 | base_delta_guard | 0.340560 | 0.514863 | 0.528044 | 0.162316 | `Q1=q160,Q2=isotonic,Q3=isotonic,S1=temp_t0.75_s-0.06,S2=isotonic,S3=q160,S4=q120` | `reports/post282_second_stage_calibration_scan_base279_20260616.json` |
| 279 | id178_delta_guard | 0.340791 | 0.514863 | 0.528044 | 0.162295 | `Q1=q160,Q2=isotonic,Q3=isotonic,S1=temp_t0.75_s0.00,S2=isotonic,S3=q160,S4=q120` | `reports/post282_second_stage_calibration_scan_base279_20260616.json` |
| 279 | temp_quantile_guard | 0.393508 | 0.214225 | 0.429140 | 0.110083 | `Q1=temp_t0.85_s0.00,Q2=isotonic,Q3=temp_t0.75_s0.00,S1=temp_t0.85_s0.00,S2=temp_t0.85_s0.00,S3=base,S4=temp_t0.85_s0.00` | `reports/post282_second_stage_calibration_scan_base279_20260616.json` |
| 279 | ultra_guard | 0.321823 | 0.562570 | 0.632785 | 0.162517 | `Q1=q160,Q2=q160,Q3=q160,S1=q120,S2=q160,S3=q160,S4=q160` | `reports/post282_second_stage_calibration_scan_base279_20260616.json` |
| 280 | raw_best | 0.317301 | 0.566117 | 0.663693 | 0.157792 | `Q1=q160,Q2=q80,Q3=q120,S1=q120,S2=q160,S3=q160,S4=q160` | `reports/post282_second_stage_calibration_scan_base280_20260616.json` |
| 280 | base_delta_guard | 0.338976 | 0.514863 | 0.559886 | 0.166106 | `Q1=isotonic,Q2=isotonic,Q3=isotonic,S1=temp_t0.75_s-0.12,S2=isotonic,S3=q160,S4=q160` | `reports/post282_second_stage_calibration_scan_base280_20260616.json` |
| 280 | id178_delta_guard | 0.332094 | 0.545262 | 0.559886 | 0.163613 | `Q1=q160,Q2=isotonic,Q3=isotonic,S1=temp_t0.75_s-0.06,S2=isotonic,S3=q160,S4=q160` | `reports/post282_second_stage_calibration_scan_base280_20260616.json` |
| 280 | temp_quantile_guard | 0.385005 | 0.213604 | 0.409299 | 0.115276 | `Q1=temp_t0.85_s0.00,Q2=isotonic,Q3=temp_t0.75_s0.00,S1=temp_t0.85_s0.00,S2=temp_t0.85_s0.00,S3=base,S4=temp_t0.85_s0.00` | `reports/post282_second_stage_calibration_scan_base280_20260616.json` |
| 280 | ultra_guard | 0.317588 | 0.566117 | 0.624500 | 0.159675 | `Q1=q160,Q2=q80,Q3=q120,S1=q80,S2=q160,S3=q160,S4=q160` | `reports/post282_second_stage_calibration_scan_base280_20260616.json` |
| 281 | raw_best | 0.314728 | 0.550697 | 0.712020 | 0.159707 | `Q1=q160,Q2=q160,Q3=q160,S1=q120,S2=q160,S3=q160,S4=q160` | `reports/post282_second_stage_calibration_scan_base281_20260616.json` |
| 281 | base_delta_guard | 0.330673 | 0.514863 | 0.506112 | 0.172668 | `Q1=isotonic,Q2=isotonic,Q3=isotonic,S1=temp_t0.75_s-0.12,S2=isotonic,S3=q160,S4=q160` | `reports/post282_second_stage_calibration_scan_base281_20260616.json` |
| 281 | id178_delta_guard | 0.328091 | 0.514863 | 0.506112 | 0.168415 | `Q1=q160,Q2=isotonic,Q3=isotonic,S1=temp_t0.75_s0.00,S2=isotonic,S3=q160,S4=q160` | `reports/post282_second_stage_calibration_scan_base281_20260616.json` |
| 281 | temp_quantile_guard | 0.369790 | 0.224558 | 0.431277 | 0.128343 | `Q1=temp_t0.85_s0.00,Q2=isotonic,Q3=isotonic,S1=temp_t0.85_s0.00,S2=temp_t0.85_s0.00,S3=base,S4=temp_t0.85_s0.00` | `reports/post282_second_stage_calibration_scan_base281_20260616.json` |
| 281 | ultra_guard | 0.316119 | 0.550697 | 0.712020 | 0.165710 | `Q1=q160,Q2=q160,Q3=isotonic,S1=q120,S2=q160,S3=q160,S4=q160` | `reports/post282_second_stage_calibration_scan_base281_20260616.json` |
| 282 | raw_best | 0.311538 | 0.655404 | 0.712557 | 0.163303 | `Q1=q160,Q2=q120,Q3=q160,S1=q160,S2=q160,S3=q160,S4=q160` | `reports/post282_second_stage_calibration_scan_base282_20260616.json` |
| 282 | base_delta_guard | 0.326877 | 0.514863 | 0.557236 | 0.178777 | `Q1=isotonic,Q2=isotonic,Q3=isotonic,S1=isotonic,S2=isotonic,S3=q160,S4=isotonic` | `reports/post282_second_stage_calibration_scan_base282_20260616.json` |
| 282 | id178_delta_guard | 0.326229 | 0.514863 | 0.506112 | 0.175651 | `Q1=isotonic,Q2=isotonic,Q3=isotonic,S1=temp_t0.75_s0.00,S2=isotonic,S3=q160,S4=q120` | `reports/post282_second_stage_calibration_scan_base282_20260616.json` |
| 282 | temp_quantile_guard | 0.362262 | 0.226788 | 0.464481 | 0.133807 | `Q1=temp_t0.85_s0.00,Q2=isotonic,Q3=isotonic,S1=temp_t0.85_s0.00,S2=temp_t0.85_s0.00,S3=base,S4=temp_t0.85_s0.00` | `reports/post282_second_stage_calibration_scan_base282_20260616.json` |
| 282 | ultra_guard | 0.312813 | 0.655404 | 0.712557 | 0.174796 | `Q1=q160,Q2=isotonic,Q3=isotonic,S1=q160,S2=q160,S3=q160,S4=q160` | `reports/post282_second_stage_calibration_scan_base282_20260616.json` |
| 278 | raw_best | 0.305709 | 0.659114 | 0.707355 | 0.182480 | `Q1=q120,Q2=q80,Q3=isotonic,S1=isotonic,S2=q120,S3=q160,S4=q160` | `reports/post282_second_stage_calibration_scan_base278_20260616.json` |
| 278 | base_delta_guard | 0.322067 | 0.604040 | 0.705028 | 0.180818 | `Q1=isotonic,Q2=isotonic,Q3=isotonic,S1=temp_t0.75_s-0.06,S2=isotonic,S3=q160,S4=q80` | `reports/post282_second_stage_calibration_scan_base278_20260616.json` |
| 278 | id178_delta_guard | 0.315449 | 0.612005 | 0.705028 | 0.180584 | `Q1=q120,Q2=isotonic,Q3=isotonic,S1=temp_t0.75_s0.00,S2=isotonic,S3=q160,S4=q120` | `reports/post282_second_stage_calibration_scan_base278_20260616.json` |
| 278 | temp_quantile_guard | 0.356330 | 0.218347 | 0.474428 | 0.139552 | `Q1=temp_t0.85_s0.00,Q2=isotonic,Q3=isotonic,S1=temp_t0.85_s0.00,S2=temp_t0.75_s0.00,S3=base,S4=temp_t0.85_s0.00` | `reports/post282_second_stage_calibration_scan_base278_20260616.json` |
| 278 | ultra_guard | 0.307635 | 0.659114 | 0.705028 | 0.186884 | `Q1=q120,Q2=isotonic,Q3=isotonic,S1=isotonic,S2=isotonic,S3=q160,S4=q160` | `reports/post282_second_stage_calibration_scan_base278_20260616.json` |

Practical read:
- Raw/isotonic/large-quantile variants can sharply lower local but are extreme full-OOF calibration probes.
- Guarded variants trade local for smaller movement versus the base and ID178.
- No DACON submission was made.
