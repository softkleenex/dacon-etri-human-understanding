# Post-ID287 failure clean anchor bridge scan

This scan excludes submitted-failed ID287 and frontier-stress ID288, then moves from confirmed Public anchor ID178 toward cleaner bridge candidates only.

| cap | local CV | max Δ vs ID178 | changed targets | mean alpha | choices |
|---:|---:|---:|---:|---:|---|
| 0.020 | 0.524653 | 0.019576 | 7 | 0.0864 | `Q1=id280:logit:a0.080,Q2=id281:logit:a0.080,Q3=id281:logit:a0.080,S1=id281:logit:a0.080,S2=id281:logit:a0.080,S3=id272:prob:a0.125,S4=id280:logit:a0.080` |
| 0.030 | 0.517635 | 0.029698 | 7 | 0.1571 | `Q1=id280:logit:a0.125,Q2=id272:logit:a0.200,Q3=id280:logit:a0.125,S1=id280:logit:a0.125,S2=id272:logit:a0.200,S3=id272:prob:a0.200,S4=id280:logit:a0.125` |
| 0.040 | 0.510463 | 0.039871 | 7 | 0.1643 | `Q1=id281:logit:a0.150,Q2=id286:logit:a0.150,Q3=id286:logit:a0.150,S1=id286:logit:a0.150,S2=id286:logit:a0.150,S3=id272:prob:a0.250,S4=id281:logit:a0.150` |
| 0.050 | 0.504921 | 0.049218 | 7 | 0.2071 | `Q1=id280:logit:a0.200,Q2=id281:logit:a0.200,Q3=id281:logit:a0.200,S1=id281:logit:a0.200,S2=id281:logit:a0.200,S3=id272:prob:a0.250,S4=id280:logit:a0.200` |
| 0.060 | 0.498891 | 0.059485 | 7 | 0.2429 | `Q1=id280:logit:a0.250,Q2=id280:logit:a0.250,Q3=id280:logit:a0.250,S1=id280:logit:a0.250,S2=id280:logit:a0.250,S3=id279:logit:a0.200,S4=id280:logit:a0.250` |
| 0.080 | 0.492260 | 0.068694 | 7 | 0.2500 | `Q1=id286:logit:a0.250,Q2=id286:logit:a0.250,Q3=id286:logit:a0.250,S1=id286:logit:a0.250,S2=id286:logit:a0.250,S3=id279:logit:a0.250,S4=id286:logit:a0.250` |
| 0.100 | 0.492260 | 0.068694 | 7 | 0.2500 | `Q1=id286:logit:a0.250,Q2=id286:logit:a0.250,Q3=id286:logit:a0.250,S1=id286:logit:a0.250,S2=id286:logit:a0.250,S3=id279:logit:a0.250,S4=id286:logit:a0.250` |
| 0.120 | 0.491948 | 0.115181 | 7 | 0.2429 | `Q1=id286:logit:a0.250,Q2=id286:logit:a0.250,Q3=id286:logit:a0.250,S1=id286:logit:a0.250,S2=id286:logit:a0.250,S3=id272:logit:a0.200,S4=id286:logit:a0.250` |
| 0.150 | 0.491744 | 0.125149 | 7 | 0.2500 | `Q1=id286:logit:a0.250,Q2=id286:logit:a0.250,Q3=id286:logit:a0.250,S1=id286:logit:a0.250,S2=id286:logit:a0.250,S3=id272:logit:a0.250,S4=id286:logit:a0.250` |

Recommendation: cap0.03/cap0.05 are the only reasonable post-failure probes; larger caps chase local and are not justified after ID287. No DACON submission was made.
