# Post-ID287 failure confirmed-anchor bridge scan

ID287 scored Public `0.6698787318`, so this scan re-centers on confirmed Public anchor ID178 and permits only small target-wise test deltas.

| cap | local CV | max Δ vs ID178 | changed targets | mean alpha | choices |
|---:|---:|---:|---:|---:|---|
| 0.020 | 0.524448 | 0.019170 | 7 | 0.0693 | `Q1=id287:logit:a0.060,Q2=id287:logit:a0.060,Q3=id287:logit:a0.060,S1=id287:logit:a0.060,S2=id287:logit:a0.060,S3=id288:prob:a0.125,S4=id287:logit:a0.060` |
| 0.030 | 0.517586 | 0.029698 | 7 | 0.1571 | `Q1=id280:logit:a0.125,Q2=id272:logit:a0.200,Q3=id280:logit:a0.125,S1=id280:logit:a0.125,S2=id272:logit:a0.200,S3=id288:prob:a0.200,S4=id280:logit:a0.125` |
| 0.040 | 0.509874 | 0.039811 | 7 | 0.1429 | `Q1=id287:logit:a0.125,Q2=id287:logit:a0.125,Q3=id287:logit:a0.125,S1=id287:logit:a0.125,S2=id287:logit:a0.125,S3=id288:prob:a0.250,S4=id287:logit:a0.125` |
| 0.050 | 0.504593 | 0.049197 | 7 | 0.1786 | `Q1=id287:logit:a0.150,Q2=id287:logit:a0.150,Q3=id281:logit:a0.200,S1=id287:logit:a0.150,S2=id281:logit:a0.200,S3=id288:prob:a0.250,S4=id287:logit:a0.150` |
| 0.060 | 0.498891 | 0.059485 | 7 | 0.2429 | `Q1=id280:logit:a0.250,Q2=id280:logit:a0.250,Q3=id280:logit:a0.250,S1=id280:logit:a0.250,S2=id280:logit:a0.250,S3=id279:logit:a0.200,S4=id280:logit:a0.250` |
| 0.080 | 0.484546 | 0.079090 | 7 | 0.2500 | `Q1=id287:logit:a0.250,Q2=id287:logit:a0.250,Q3=id287:logit:a0.250,S1=id287:logit:a0.250,S2=id287:logit:a0.250,S3=id279:logit:a0.250,S4=id287:logit:a0.250` |
| 0.100 | 0.481592 | 0.089148 | 7 | 0.2500 | `Q1=id288:logit:a0.250,Q2=id288:logit:a0.250,Q3=id288:logit:a0.250,S1=id288:logit:a0.250,S2=id288:logit:a0.250,S3=id279:logit:a0.250,S4=id288:logit:a0.250` |
| 0.120 | 0.481281 | 0.115181 | 7 | 0.2429 | `Q1=id288:logit:a0.250,Q2=id288:logit:a0.250,Q3=id288:logit:a0.250,S1=id288:logit:a0.250,S2=id288:logit:a0.250,S3=id272:logit:a0.200,S4=id288:logit:a0.250` |
| 0.150 | 0.480979 | 0.129138 | 7 | 0.2500 | `Q1=id288:logit:a0.250,Q2=id288:logit:a0.250,Q3=id288:logit:a0.250,S1=id288:logit:a0.250,S2=id288:logit:a0.250,S3=id288:logit:a0.250,S4=id288:logit:a0.250` |

Interpretation: after the ID287 public failure, prefer the smallest cap that gives meaningful local improvement over ID178, not the lowest local row. No DACON submission was made.
