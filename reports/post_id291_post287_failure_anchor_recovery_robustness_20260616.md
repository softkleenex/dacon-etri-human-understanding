# Post-ID291 anchor recovery robustness

ID287 scored Public `0.6698787318`, so IDs 289-291 are clean recovery probes from confirmed Public anchor ID178. They exclude ID287 and ID288 from frontier choices.

| role | id | cap | local CV | optimistic ID178-gap proxy | post-failure risk proxy | max Δ ID178 | mean Δ ID178 | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| post_failure_safest_probe | 289 | 0.030 | 0.517635 | 0.580408 | 0.580408 | 0.029698 | 0.012847 | safest next public-information probe |
| post_failure_balanced_probe | 290 | 0.050 | 0.504921 | 0.567694 | 0.583507 | 0.049218 | 0.020598 | balanced but riskier; local below 0.51 |
| post_failure_score_pressure_probe | 291 | 0.060 | 0.498891 | 0.561664 | 0.586086 | 0.059485 | 0.024229 | score pressure; do not submit before safer feedback |

Recommendation: if we submit again immediately, ID289 is the correct recovery probe. ID290 is the only score-pressure alternative; ID291 should wait. No DACON submission was made.
