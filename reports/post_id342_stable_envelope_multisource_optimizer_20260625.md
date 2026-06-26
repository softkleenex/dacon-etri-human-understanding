# Post-ID342 stable-envelope multi-source optimizer

Created: `2026-06-25 00:51:23 UTC+09:00`

This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.

## Decision

- Candidate sources: `18`
- Scanned nonlegacy gate sources: `101`
- Candidate source families: `5`
- Beam states retained: `16153`
- Anchor projected public: `0.596318745`
- Best projected public: `0.566362799`
- Best local delta vs ID342: `-0.029955946`
- Best gap to Public 0.55: `+0.016362799`
- Best total source weight: `0.140`
- Best active source count: `2`
- Public-0.55 envelope rows in report: `0`
- Artifact review required: `False`
- DACON submit now: `False`
- Next action: If this report still misses Public 0.55 under the stable-gap projection, keep DACON closed and treat the result as validation-compression evidence. If it passes 0.55, perform a separate artifact-contract review before creating any CSV.

## Top Rows

| local | delta local | projected public | gap 0.55 | total weight | active sources | mean move | max move | weights |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `0.504044` | `-0.029956` | `0.566363` | `+0.016363` | `0.140` | `2` | `0.019484` | `0.065893` | ID285:0.125, ID278:0.015 |
| `0.504075` | `-0.029925` | `0.566393` | `+0.016393` | `0.140` | `3` | `0.019467` | `0.065850` | ID285:0.125, ID278:0.010, ID282:0.005 |
| `0.504105` | `-0.029895` | `0.566424` | `+0.016424` | `0.140` | `3` | `0.019449` | `0.065808` | ID285:0.125, ID278:0.005, ID282:0.010 |
| `0.504111` | `-0.029889` | `0.566430` | `+0.016430` | `0.140` | `3` | `0.019493` | `0.065364` | ID285:0.120, ID287:0.015, ID282:0.005 |
| `0.504111` | `-0.029889` | `0.566430` | `+0.016430` | `0.140` | `3` | `0.019456` | `0.065925` | ID285:0.125, ID278:0.010, ID277:0.005 |
| `0.504116` | `-0.029884` | `0.566435` | `+0.016435` | `0.140` | `3` | `0.019445` | `0.065804` | ID285:0.125, ID278:0.010, ID281:0.005 |
| `0.504118` | `-0.029882` | `0.566437` | `+0.016437` | `0.140` | `4` | `0.019482` | `0.065883` | ID285:0.125, ID278:0.005, ID282:0.005, ID286:0.005 |
| `0.504123` | `-0.029877` | `0.566441` | `+0.016441` | `0.140` | `3` | `0.019485` | `0.065918` | ID285:0.125, ID287:0.005, ID281:0.010 |
| `0.504127` | `-0.029873` | `0.566446` | `+0.016446` | `0.140` | `4` | `0.019482` | `0.065913` | ID285:0.125, ID287:0.005, ID282:0.005, ID280:0.005 |
| `0.504136` | `-0.029864` | `0.566455` | `+0.016455` | `0.140` | `2` | `0.019431` | `0.065765` | ID285:0.125, ID282:0.015 |
| `0.504142` | `-0.029858` | `0.566461` | `+0.016461` | `0.140` | `4` | `0.019439` | `0.065882` | ID285:0.125, ID278:0.005, ID282:0.005, ID277:0.005 |
| `0.504144` | `-0.029856` | `0.566463` | `+0.016463` | `0.140` | `4` | `0.019474` | `0.065903` | ID285:0.125, ID287:0.005, ID278:0.005, ID279:0.005 |
| `0.504146` | `-0.029854` | `0.566465` | `+0.016465` | `0.140` | `3` | `0.019432` | `0.065203` | ID285:0.120, ID287:0.010, ID278:0.010 |
| `0.504147` | `-0.029853` | `0.566466` | `+0.016466` | `0.140` | `4` | `0.019428` | `0.065761` | ID285:0.125, ID278:0.005, ID282:0.005, ID281:0.005 |
| `0.504148` | `-0.029852` | `0.566467` | `+0.016467` | `0.140` | `3` | `0.019483` | `0.065438` | ID285:0.120, ID287:0.015, ID277:0.005 |
| `0.504149` | `-0.029851` | `0.566468` | `+0.016468` | `0.140` | `3` | `0.019465` | `0.065841` | ID285:0.125, ID282:0.010, ID286:0.005 |
| `0.504151` | `-0.029849` | `0.566469` | `+0.016469` | `0.140` | `4` | `0.019478` | `0.065986` | ID285:0.125, ID287:0.005, ID282:0.005, ID276:0.005 |
| `0.504153` | `-0.029847` | `0.566472` | `+0.016472` | `0.140` | `3` | `0.019472` | `0.065317` | ID285:0.120, ID287:0.015, ID281:0.005 |
| `0.504155` | `-0.029845` | `0.566474` | `+0.016474` | `0.140` | `4` | `0.019472` | `0.065958` | ID285:0.125, ID278:0.005, ID286:0.005, ID277:0.005 |
| `0.504156` | `-0.029844` | `0.566475` | `+0.016475` | `0.140` | `3` | `0.019490` | `0.065886` | ID285:0.125, ID284:0.005, ID281:0.010 |
| `0.504157` | `-0.029843` | `0.566476` | `+0.016476` | `0.140` | `4` | `0.019468` | `0.065897` | ID285:0.125, ID287:0.005, ID278:0.005, ID275:0.005 |
| `0.504160` | `-0.029840` | `0.566478` | `+0.016478` | `0.140` | `4` | `0.019461` | `0.065837` | ID285:0.125, ID278:0.005, ID286:0.005, ID281:0.005 |
| `0.504161` | `-0.029839` | `0.566479` | `+0.016479` | `0.140` | `4` | `0.019487` | `0.065882` | ID285:0.125, ID284:0.005, ID282:0.005, ID280:0.005 |
| `0.504163` | `-0.029837` | `0.566481` | `+0.016481` | `0.140` | `3` | `0.019421` | `0.065752` | ID285:0.125, ID278:0.010, ID280:0.005 |
| `0.504164` | `-0.029836` | `0.566483` | `+0.016483` | `0.140` | `4` | `0.019472` | `0.065988` | ID285:0.125, ID287:0.005, ID277:0.005, ID280:0.005 |

## Candidate Sources

| source | family | single alpha | single projected public | file |
|---:|---|---:|---:|---|
| `ID285` | `post282_tempguard` | `0.120` | `0.569888` | `base_oof_id_285_post282_base278_tempguard_calibration_cv0.356330.csv` |
| `ID287` | `post285_fixedtemp` | `0.140` | `0.570470` | `base_oof_id_287_post285_base282_fixedtemp_balanced_cv0.378934.csv` |
| `ID284` | `post282_tempguard` | `0.140` | `0.571404` | `base_oof_id_284_post282_base280_tempguard_calibration_cv0.385005.csv` |
| `ID278` | `post274_refined` | `0.140` | `0.572353` | `base_oof_id_278_post274_refined_cap300_p005_blend_cv0.389120.csv` |
| `ID282` | `post278_gapaware` | `0.140` | `0.573221` | `base_oof_id_282_post278_gapaware_cap280_sp005_dp100_blend_cv0.393400.csv` |
| `ID286` | `post285_fixedtemp` | `0.140` | `0.573579` | `base_oof_id_286_post285_base280_fixedtemp_guard_cv0.395368.csv` |
| `ID277` | `post274_refined` | `0.140` | `0.574279` | `base_oof_id_277_post274_refined_cap280_p010_blend_cv0.399355.csv` |
| `ID281` | `post278_gapaware` | `0.140` | `0.574401` | `base_oof_id_281_post278_gapaware_cap260_sp005_dp100_blend_cv0.399583.csv` |
| `ID280` | `post278_gapaware` | `0.140` | `0.575724` | `base_oof_id_280_post278_gapaware_cap240_sp005_dp100_blend_cv0.406704.csv` |
| `ID276` | `post274_refined` | `0.140` | `0.576414` | `base_oof_id_276_post274_refined_cap260_p010_blend_cv0.411217.csv` |
| `ID279` | `post278_gapaware` | `0.140` | `0.577082` | `base_oof_id_279_post278_gapaware_cap220_sp005_dp100_blend_cv0.414169.csv` |
| `ID275` | `post274_refined` | `0.140` | `0.577463` | `base_oof_id_275_post274_refined_cap220_p005_blend_cv0.416609.csv` |
| `ID283` | `post282_tempguard` | `0.140` | `0.581318` | `base_oof_id_283_post282_base272_tempguard_calibration_cv0.440206.csv` |
| `ID171` | `other` | `0.120` | `0.587389` | `base_oof_id_171_id163_q40_extreme_public_probe_cv0.493975.csv` |
| `ID172` | `other` | `0.120` | `0.587469` | `base_oof_id_172_id163_targetwise_extreme_public_probe_cv0.492762.csv` |
| `ID231` | `other` | `0.120` | `0.587547` | `base_oof_id_231_id226_outer_supported_q2_temp085_probe_cv0.491287.csv` |
| `ID188` | `other` | `0.140` | `0.588767` | `base_oof_id_188_id179_targetwise_delta30_id172_extreme_probe_cv0.498094.csv` |
| `ID225` | `other` | `0.140` | `0.588881` | `base_oof_id_225_id221_id172_cap030_convex_bridge_probe_cv0.496691.csv` |
