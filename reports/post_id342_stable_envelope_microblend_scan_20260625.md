# Post-ID342 stable-envelope micro-blend scan

Created: `2026-06-25 00:44:42 UTC+09:00`

This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.

## Decision

- Sources scanned: `246`
- Blend rows scanned: `3198`
- Anchor projected public: `0.596318745`
- Stable-envelope rows: `3122`
- Public-0.55 envelope rows: `0`
- Gate pass count: `1124`
- Best envelope source/alpha: `ID250` / `0.08`
- Best envelope projected public: `0.563921703`
- Best envelope local delta vs ID342: `-0.032397041`
- Best gate source: `285`
- DACON submit now: `False`
- Next action: No submission artifact should be created unless a row both improves ID342 inside the stable envelope and clears legacy-source risk. If no gate row exists, continue with validation/source redesign.

## Top Envelope Rows

| source | alpha | local | delta local | projected public | gap 0.55 | mean move | max move | legacy | file |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `250` | `0.08` | `0.501603` | `-0.032397` | `0.563922` | `+0.013922` | `0.018610` | `0.069715` | `True` | `base_oof_id_250_id242_iso_non_s3_s4base_ultra_frontier_cv0.261184.csv` |
| `251` | `0.08` | `0.501860` | `-0.032140` | `0.564179` | `+0.014179` | `0.018442` | `0.069376` | `True` | `base_oof_id_251_id242_quantile_non_s3_s4q100_ultra_frontier_cv0.263237.csv` |
| `252` | `0.08` | `0.504075` | `-0.029925` | `0.566394` | `+0.016394` | `0.017425` | `0.069715` | `True` | `base_oof_id_252_id249_iso_mixed_second_stage_frontier_cv0.274420.csv` |
| `285` | `0.12` | `0.507569` | `-0.026431` | `0.569888` | `+0.019888` | `0.017237` | `0.059301` | `False` | `base_oof_id_285_post282_base278_tempguard_calibration_cv0.356330.csv` |
| `250` | `0.06` | `0.509367` | `-0.024633` | `0.571686` | `+0.021686` | `0.013958` | `0.052286` | `True` | `base_oof_id_250_id242_iso_non_s3_s4base_ultra_frontier_cv0.261184.csv` |
| `251` | `0.06` | `0.509563` | `-0.024437` | `0.571882` | `+0.021882` | `0.013832` | `0.052032` | `True` | `base_oof_id_251_id242_quantile_non_s3_s4q100_ultra_frontier_cv0.263237.csv` |
| `288` | `0.12` | `0.510577` | `-0.023423` | `0.572896` | `+0.022896` | `0.015079` | `0.046763` | `True` | `base_oof_id_288_post285_base278_fixedtemp_frontier_cv0.372773.csv` |
| `252` | `0.06` | `0.511283` | `-0.022717` | `0.573602` | `+0.023602` | `0.013069` | `0.052286` | `True` | `base_oof_id_252_id249_iso_mixed_second_stage_frontier_cv0.274420.csv` |
| `287` | `0.12` | `0.511737` | `-0.022263` | `0.574055` | `+0.024055` | `0.014329` | `0.043528` | `False` | `base_oof_id_287_post285_base282_fixedtemp_balanced_cv0.378934.csv` |
| `285` | `0.1` | `0.511842` | `-0.022158` | `0.574161` | `+0.024161` | `0.014364` | `0.049418` | `False` | `base_oof_id_285_post282_base278_tempguard_calibration_cv0.356330.csv` |
| `284` | `0.12` | `0.512542` | `-0.021458` | `0.574861` | `+0.024861` | `0.014455` | `0.053269` | `False` | `base_oof_id_284_post282_base280_tempguard_calibration_cv0.385005.csv` |
| `278` | `0.12` | `0.513361` | `-0.020639` | `0.575680` | `+0.025680` | `0.012654` | `0.041278` | `False` | `base_oof_id_278_post274_refined_cap300_p005_blend_cv0.389120.csv` |
| `282` | `0.12` | `0.514113` | `-0.019887` | `0.576432` | `+0.026432` | `0.012217` | `0.040640` | `False` | `base_oof_id_282_post278_gapaware_cap280_sp005_dp100_blend_cv0.393400.csv` |
| `288` | `0.1` | `0.514378` | `-0.019622` | `0.576697` | `+0.026697` | `0.012566` | `0.038970` | `True` | `base_oof_id_288_post285_base278_fixedtemp_frontier_cv0.372773.csv` |
| `286` | `0.12` | `0.514425` | `-0.019575` | `0.576744` | `+0.026744` | `0.012848` | `0.038543` | `False` | `base_oof_id_286_post285_base280_fixedtemp_guard_cv0.395368.csv` |
| `247` | `0.12` | `0.515002` | `-0.018998` | `0.577321` | `+0.027321` | `0.013771` | `0.064675` | `True` | `base_oof_id_247_id235_frontier_twise_cap020_logit_bridge_cv0.415523.csv` |
| `277` | `0.12` | `0.515027` | `-0.018973` | `0.577346` | `+0.027346` | `0.011944` | `0.040640` | `False` | `base_oof_id_277_post274_refined_cap280_p010_blend_cv0.399355.csv` |
| `281` | `0.12` | `0.515134` | `-0.018866` | `0.577452` | `+0.027452` | `0.011679` | `0.038782` | `False` | `base_oof_id_281_post278_gapaware_cap260_sp005_dp100_blend_cv0.399583.csv` |
| `287` | `0.1` | `0.515356` | `-0.018644` | `0.577675` | `+0.027675` | `0.011941` | `0.036273` | `False` | `base_oof_id_287_post285_base282_fixedtemp_balanced_cv0.378934.csv` |
| `284` | `0.1` | `0.516032` | `-0.017968` | `0.578351` | `+0.028351` | `0.012046` | `0.044391` | `False` | `base_oof_id_284_post282_base280_tempguard_calibration_cv0.385005.csv` |
| `285` | `0.08` | `0.516166` | `-0.017834` | `0.578485` | `+0.028485` | `0.011491` | `0.039534` | `False` | `base_oof_id_285_post282_base278_tempguard_calibration_cv0.356330.csv` |
| `280` | `0.12` | `0.516278` | `-0.017722` | `0.578597` | `+0.028597` | `0.011078` | `0.036417` | `False` | `base_oof_id_280_post278_gapaware_cap240_sp005_dp100_blend_cv0.406704.csv` |
| `278` | `0.1` | `0.516719` | `-0.017281` | `0.579038` | `+0.029038` | `0.010545` | `0.034398` | `False` | `base_oof_id_278_post274_refined_cap300_p005_blend_cv0.389120.csv` |
| `276` | `0.12` | `0.516871` | `-0.017129` | `0.579190` | `+0.029190` | `0.011005` | `0.037230` | `False` | `base_oof_id_276_post274_refined_cap260_p010_blend_cv0.411217.csv` |
| `250` | `0.04` | `0.517341` | `-0.016659` | `0.579660` | `+0.029660` | `0.009305` | `0.034857` | `True` | `base_oof_id_250_id242_iso_non_s3_s4base_ultra_frontier_cv0.261184.csv` |

## Top Nonlegacy Envelope Rows

| source | alpha | local | delta local | projected public | gap 0.55 | mean move | max move | legacy | file |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `285` | `0.12` | `0.507569` | `-0.026431` | `0.569888` | `+0.019888` | `0.017237` | `0.059301` | `False` | `base_oof_id_285_post282_base278_tempguard_calibration_cv0.356330.csv` |
| `287` | `0.12` | `0.511737` | `-0.022263` | `0.574055` | `+0.024055` | `0.014329` | `0.043528` | `False` | `base_oof_id_287_post285_base282_fixedtemp_balanced_cv0.378934.csv` |
| `285` | `0.1` | `0.511842` | `-0.022158` | `0.574161` | `+0.024161` | `0.014364` | `0.049418` | `False` | `base_oof_id_285_post282_base278_tempguard_calibration_cv0.356330.csv` |
| `284` | `0.12` | `0.512542` | `-0.021458` | `0.574861` | `+0.024861` | `0.014455` | `0.053269` | `False` | `base_oof_id_284_post282_base280_tempguard_calibration_cv0.385005.csv` |
| `278` | `0.12` | `0.513361` | `-0.020639` | `0.575680` | `+0.025680` | `0.012654` | `0.041278` | `False` | `base_oof_id_278_post274_refined_cap300_p005_blend_cv0.389120.csv` |
| `282` | `0.12` | `0.514113` | `-0.019887` | `0.576432` | `+0.026432` | `0.012217` | `0.040640` | `False` | `base_oof_id_282_post278_gapaware_cap280_sp005_dp100_blend_cv0.393400.csv` |
| `286` | `0.12` | `0.514425` | `-0.019575` | `0.576744` | `+0.026744` | `0.012848` | `0.038543` | `False` | `base_oof_id_286_post285_base280_fixedtemp_guard_cv0.395368.csv` |
| `277` | `0.12` | `0.515027` | `-0.018973` | `0.577346` | `+0.027346` | `0.011944` | `0.040640` | `False` | `base_oof_id_277_post274_refined_cap280_p010_blend_cv0.399355.csv` |
| `281` | `0.12` | `0.515134` | `-0.018866` | `0.577452` | `+0.027452` | `0.011679` | `0.038782` | `False` | `base_oof_id_281_post278_gapaware_cap260_sp005_dp100_blend_cv0.399583.csv` |
| `287` | `0.1` | `0.515356` | `-0.018644` | `0.577675` | `+0.027675` | `0.011941` | `0.036273` | `False` | `base_oof_id_287_post285_base282_fixedtemp_balanced_cv0.378934.csv` |
| `284` | `0.1` | `0.516032` | `-0.017968` | `0.578351` | `+0.028351` | `0.012046` | `0.044391` | `False` | `base_oof_id_284_post282_base280_tempguard_calibration_cv0.385005.csv` |
| `285` | `0.08` | `0.516166` | `-0.017834` | `0.578485` | `+0.028485` | `0.011491` | `0.039534` | `False` | `base_oof_id_285_post282_base278_tempguard_calibration_cv0.356330.csv` |
| `280` | `0.12` | `0.516278` | `-0.017722` | `0.578597` | `+0.028597` | `0.011078` | `0.036417` | `False` | `base_oof_id_280_post278_gapaware_cap240_sp005_dp100_blend_cv0.406704.csv` |
| `278` | `0.1` | `0.516719` | `-0.017281` | `0.579038` | `+0.029038` | `0.010545` | `0.034398` | `False` | `base_oof_id_278_post274_refined_cap300_p005_blend_cv0.389120.csv` |
| `276` | `0.12` | `0.516871` | `-0.017129` | `0.579190` | `+0.029190` | `0.011005` | `0.037230` | `False` | `base_oof_id_276_post274_refined_cap260_p010_blend_cv0.411217.csv` |
| `282` | `0.1` | `0.517353` | `-0.016647` | `0.579671` | `+0.029671` | `0.010181` | `0.033867` | `False` | `base_oof_id_282_post278_gapaware_cap280_sp005_dp100_blend_cv0.393400.csv` |
| `279` | `0.12` | `0.517450` | `-0.016550` | `0.579769` | `+0.029769` | `0.010468` | `0.034012` | `False` | `base_oof_id_279_post278_gapaware_cap220_sp005_dp100_blend_cv0.414169.csv` |
| `286` | `0.1` | `0.517617` | `-0.016383` | `0.579936` | `+0.029936` | `0.010707` | `0.032119` | `False` | `base_oof_id_286_post285_base280_fixedtemp_guard_cv0.395368.csv` |
| `275` | `0.12` | `0.517778` | `-0.016222` | `0.580097` | `+0.030097` | `0.010281` | `0.033054` | `False` | `base_oof_id_275_post274_refined_cap220_p005_blend_cv0.416609.csv` |
| `277` | `0.1` | `0.518120` | `-0.015880` | `0.580439` | `+0.030439` | `0.009953` | `0.033867` | `False` | `base_oof_id_277_post274_refined_cap280_p010_blend_cv0.399355.csv` |
| `281` | `0.1` | `0.518211` | `-0.015789` | `0.580530` | `+0.030530` | `0.009733` | `0.032318` | `False` | `base_oof_id_281_post278_gapaware_cap260_sp005_dp100_blend_cv0.399583.csv` |
| `287` | `0.08` | `0.519011` | `-0.014989` | `0.581330` | `+0.031330` | `0.009553` | `0.029019` | `False` | `base_oof_id_287_post285_base282_fixedtemp_balanced_cv0.378934.csv` |
| `280` | `0.1` | `0.519173` | `-0.014827` | `0.581491` | `+0.031491` | `0.009232` | `0.030347` | `False` | `base_oof_id_280_post278_gapaware_cap240_sp005_dp100_blend_cv0.406704.csv` |
| `284` | `0.08` | `0.519555` | `-0.014445` | `0.581874` | `+0.031874` | `0.009637` | `0.035513` | `False` | `base_oof_id_284_post282_base280_tempguard_calibration_cv0.385005.csv` |
| `276` | `0.1` | `0.519668` | `-0.014332` | `0.581987` | `+0.031987` | `0.009170` | `0.031025` | `False` | `base_oof_id_276_post274_refined_cap260_p010_blend_cv0.411217.csv` |
