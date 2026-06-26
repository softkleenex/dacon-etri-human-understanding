# Public-anchor geometry diagnostic after ID329

Created: `2026-06-17 01:19:51` KST

This diagnostic uses actual public-scored rows and compares available submission CSV predictions against ID178, ID287, and ID329. Placeholder public scores of `0.0` and the invalid `13.322254` row are excluded.

Valid public rows: `23`; rows with comparable submission geometry: `23`.

## Anchor facts

| id | role | local CV | public LB |
|---:|---|---:|---:|
| 178 | best confirmed public anchor | 0.538638 | 0.6014113009 |
| 287 | failed public direction | 0.378934 | 0.6698787318 |
| 329 | guarded practical non-improvement | 0.516718 | 0.6017908577 |

## Public-scored rows with geometry

| id | local | public | gap | mean Δ178 | max Δ178 | proj failed | Δ public vs ID178 | file |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 0.619900 | 0.6362389122 | +0.016339 | 0.110265 | 0.510146 | -0.2106 | +0.0348276113 | `submission_baseline_score0.6199.csv` |
| 3 | 0.694400 | 0.6927104216 | -0.001690 | 0.110265 | 0.510146 | -0.2106 | +0.0912991207 | `submission_kaggle_compute_fe_optuna_v1_score0.6944.csv` |
| 4 | 0.669000 | 0.6606887692 | -0.008311 | 0.110265 | 0.510146 | -0.2106 | +0.0592774683 | `submission_kaggle_dl_lstm_attention_v1_score0.6690.csv` |
| 5 | 0.627700 | 0.6360617792 | +0.008362 | 0.110265 | 0.510146 | -0.2106 | +0.0346504783 | `submission_id_005_score0.6277.csv` |
| 6 | 0.592500 | 0.6116754269 | +0.019175 | 0.110265 | 0.510146 | -0.2106 | +0.0102641260 | `submission_id_006_score0.5925.csv` |
| 7 | 0.592100 | 0.6166592444 | +0.024559 | 0.110265 | 0.510146 | -0.2106 | +0.0152479435 | `submission_id_007_score0.5921.csv` |
| 8 | 0.591400 | 0.6151360527 | +0.023736 | 0.110265 | 0.510146 | -0.2106 | +0.0137247518 | `submission_id_008_score0.5914.csv` |
| 9 | 0.625800 | 0.8208847151 | +0.195085 | 0.110265 | 0.510146 | -0.2106 | +0.2194734142 | `submission_id_009_score0.6258.csv` |
| 10 | 0.597200 | 0.6301928255 | +0.032993 | 0.110265 | 0.510146 | -0.2106 | +0.0287815246 | `submission_id_010_score0.5972.csv` |
| 11 | 0.590900 | 0.6208484881 | +0.029948 | 0.110265 | 0.510146 | -0.2106 | +0.0194371872 | `submission_id_011_score0.5909.csv` |
| 12 | 0.622700 | 0.6268431303 | +0.004143 | 0.109509 | 0.480331 | -0.1717 | +0.0254318294 | `submission_optimal_ensemble_final.csv` |
| 13 | 0.619800 | 0.6362064833 | +0.016406 | 0.110265 | 0.510146 | -0.2106 | +0.0347951824 | `submission_id_013_score0.6198.csv` |
| 14 | 0.618200 | 0.6293399255 | +0.011140 | 0.110265 | 0.510146 | -0.2106 | +0.0279286246 | `submission_id_014_score0.6182.csv` |
| 15 | 0.615400 | 0.6260445149 | +0.010645 | 0.110265 | 0.510146 | -0.2106 | +0.0246332140 | `submission_id_015_score0.6154.csv` |
| 16 | 0.587500 | 0.6184828015 | +0.030983 | 0.110265 | 0.510146 | -0.2106 | +0.0170715006 | `submission_id_016_score0.5875.csv` |
| 17 | 0.611700 | 0.6238825362 | +0.012183 | 0.110265 | 0.510146 | -0.2106 | +0.0224712353 | `submission_id_017_score0.6117.csv` |
| 73 | 0.602200 | 0.7454473391 | +0.143247 | 0.182808 | 0.844816 | -0.2917 | +0.1440360382 | `submission_id_073_score0.6022.csv` |
| 74 | 0.598600 | 0.7436431908 | +0.145043 | 0.224748 | 0.718185 | -0.4163 | +0.1422318899 | `submission_id_074_score0.5986.csv` |
| 76 | 0.599611 | 0.6241517536 | +0.024541 | 0.122637 | 0.486291 | -0.2300 | +0.0227404527 | `submission_id_076_score0.5996.csv` |
| 93 | 0.591466 | 0.6166556393 | +0.025190 | 0.109509 | 0.480331 | -0.1717 | +0.0152443384 | `submission_id_085_score0.591466.csv` |
| 178 | 0.538638 | 0.6014113009 | +0.062773 | 0.000000 | 0.000000 | +0.0000 | +0.0000000000 | `submission_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638_unsubmitted.csv` |
| 287 | 0.378934 | 0.6698787318 | +0.290945 | 0.115353 | 0.298460 | +1.0000 | +0.0684674309 | `submission_id_287_post285_base282_fixedtemp_balanced_cv0.378934_unsubmitted.csv` |
| 329 | 0.516718 | 0.6017908577 | +0.085073 | 0.019372 | 0.079192 | +0.0499 | +0.0003795568 | `submission_id_329_post328_combo4_cap080_budget050_practical_cv0.516718_unsubmitted.csv` |

## Correlations with public LB

| feature | Pearson | Spearman |
|---|---:|---:|
| `local_cv` | 0.15642020505949714 | 0.6215415019762845 |
| `gap` | 0.6102771480689735 | 0.009881422924901188 |
| `mean_abs_delta_vs_id178` | 0.5422877631645541 | 0.6014685625537782 |
| `max_abs_delta_vs_id178` | 0.4654372430351836 | 0.5003521085592301 |
| `rmse_vs_id178` | 0.5518505969177508 | 0.6014685625537782 |
| `mean_abs_delta_vs_id329` | 0.5500430804485775 | 0.4689711400781633 |
| `projection_to_id287_failed_direction` | -0.07794573782613298 | -0.41783178403494353 |
| `projection_to_id329_direction` | 0.5123759209845148 | 0.4596730753430325 |
| `prediction_mean` | -0.4624403375371552 | 0.008716935689185192 |
| `prediction_std` | 0.0308280912110229 | -0.038935646078360524 |
| `prediction_min` | -0.1224072337139112 | 0.029637581343229653 |
| `prediction_max` | -0.42399166458578996 | -0.4197186402045587 |

## Interpretation

- The only recent rows with complete base artifacts are ID178, ID287, and ID329; therefore geometry diagnostics are informative but not a validated model.
- ID329 confirms that lower local CV plus low ID287 projection is insufficient: Public stayed near ID178.
- A redesigned validation target must explain both ID287 collapse and ID329 non-improvement before generating more same-family candidates.

## Validation redesign implication

Do not optimize another local frontier or same-family constrained combo. The next validation proxy must be evaluated by whether it ranks ID178/ID329 near their observed Public `~0.601` and penalizes ID287 toward its observed collapse, before any new candidate is promoted.
