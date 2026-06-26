# Post-ID288 fixed-temperature robustness

IDs 286-288 use fixed logit temperature/shift transforms only. This removes fitted isotonic/quantile calibration from the promoted cards, though target-wise OOF selection is still non-nested.

Confirmed ID178 gap reference: local `0.538638` -> Public `0.6014113009`, gap `+0.062773`.

| role | id | base | local CV | score proxy | stability-veto proxy | max Δ ID178 | max Δ base | methods |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| balanced_fixed_temp_score_pressure | 287 | 282 | 0.378934 | 0.441707 | 0.493668 | 0.298460 | 0.049781 | `Q1=temp_t0.85_s0.00,Q2=temp_t0.80_s0.00,Q3=temp_t0.80_s0.00,S1=temp_t0.80_s0.00,S2=temp_t0.80_s0.00,S3=base,S4=temp_t0.85_s0.00` |
| conservative_fixed_temp_guard | 286 | 280 | 0.395368 | 0.458141 | 0.496510 | 0.259624 | 0.041869 | `Q1=temp_t0.85_s0.00,Q2=temp_t0.85_s-0.04,Q3=temp_t0.85_s0.00,S1=temp_t0.85_s0.00,S2=temp_t0.85_s0.00,S3=base,S4=temp_t0.85_s0.00` |
| frontier_fixed_temp_stress | 288 | 278 | 0.372773 | 0.435546 | 0.498594 | 0.330135 | 0.057850 | `Q1=temp_t0.80_s-0.06,Q2=temp_t0.80_s-0.06,Q3=temp_t0.80_s-0.06,S1=temp_t0.80_s-0.06,S2=temp_t0.80_s-0.06,S3=base,S4=temp_t0.80_s-0.06` |

Recommendation: ID287 is the best fixed-temp balance, ID286 is the safer bridge, and ID288 is a frontier stress card. No DACON submission was made.
