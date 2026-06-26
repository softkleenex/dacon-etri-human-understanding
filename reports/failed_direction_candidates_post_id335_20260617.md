# Failed-direction candidate audit — post ID335

Created: `2026-06-17 22:36:20 UTC+09:00`

This is report-only. No submission CSV, notebook, base artifact, submission-log row, or DACON submission was created.

## Candidate decisions

| id | decision | overall | robust | gain vs ID179 | mean Δ179 | max Δ178 | proj failed | gain/Δ179 | reason |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 295 | `diagnostic_antivector_only` | 0.548583 | 0.574349 | -0.010293 | 0.007758 | 0.014923 | -0.0500 | -1.327 | Negative failed-vector projection is informative, but it does not improve local CV versus ID179. |
| 296 | `diagnostic_antivector_only` | 0.558807 | 0.584792 | -0.020517 | 0.012888 | 0.029846 | -0.1000 | -1.592 | Negative failed-vector projection is informative, but it does not improve local CV versus ID179. |
| 297 | `diagnostic_antivector_only` | 0.550861 | 0.576578 | -0.012571 | 0.009149 | 0.018164 | -0.0614 | -1.374 | Negative failed-vector projection is informative, but it does not improve local CV versus ID179. |
| 304 | `freeze_full_oof_efficiency_trap` | 0.532349 | 0.558144 | +0.005941 | 0.009615 | 0.029765 | -0.0000 | +0.618 | Large local gain comes from non-nested/full-OOF recombination. ID329 already showed this family can preserve public plateau despite much better local CV. |
| 305 | `freeze_full_oof_efficiency_trap` | 0.529570 | 0.555462 | +0.008720 | 0.013646 | 0.048772 | +0.0000 | +0.639 | Large local gain comes from non-nested/full-OOF recombination. ID329 already showed this family can preserve public plateau despite much better local CV. |
| 306 | `freeze_full_oof_efficiency_trap` | 0.526367 | 0.552229 | +0.011923 | 0.018291 | 0.074735 | +0.0000 | +0.652 | Large local gain comes from non-nested/full-OOF recombination. ID329 already showed this family can preserve public plateau despite much better local CV. |
| 307 | `diagnostic_only_not_promoted` | 0.537792 | 0.563569 | +0.000498 | 0.007243 | 0.026555 | -0.0170 | +0.069 | Low movement is interesting, but non-nested full-OOF selection is not enough after ID329. |
| 310 | `freeze_full_oof_efficiency_trap` | 0.521885 | 0.547040 | +0.016405 | 0.026535 | 0.114878 | +0.0007 | +0.618 | Large local gain comes from non-nested/full-OOF recombination. ID329 already showed this family can preserve public plateau despite much better local CV. |
| 311 | `freeze_full_oof_efficiency_trap` | 0.524195 | 0.549724 | +0.014095 | 0.019538 | 0.091284 | -0.0000 | +0.721 | Large local gain comes from non-nested/full-OOF recombination. ID329 already showed this family can preserve public plateau despite much better local CV. |

## Interpretation

The top failed-direction/orthogonal rows look attractive because they buy local CV with tiny test movement, but their gain is non-nested/full-OOF recombination. ID329 already demonstrated that this evidence does not convert into public improvement. Keep these rows diagnostic-only unless a fold-nested validation path is built.

