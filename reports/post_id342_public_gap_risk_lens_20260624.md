# Post-ID342 public-gap risk lens

Created: `2026-06-25 00:57:07 UTC+09:00`

This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.

## Decision

- OOF files scanned: `247`
- Scored rows: `8`
- Stable gap median: `0.062318739`
- Required local for Public 0.55: `0.487681261`
- Stable shape-envelope rows: `11`
- Public-0.55 shape-lens pass count: `0`
- Best shape-envelope candidate: `ID342` / `0.596318745`
- Best shape-envelope gap to 0.55: `+0.046318745`
- DACON submit now: `False`
- Next action: The stable-shape lens leaves no Public-0.55 shelf candidate. The next experiment should search for new signal inside the stable movement envelope or redesign validation around public-gap compression, not amplify local frontier blends.

## Stable Shape Envelope

```json
{
  "extreme_probability_rate_p95": 0.0018253968253968255,
  "max_abs_delta_vs_id342_p95": 0.12327747882405193,
  "max_abs_mean_rate_shift_p95": 0.005794333085877401,
  "max_target_mean_abs_delta_vs_id342_p95": 0.03721482827964871,
  "mean_abs_delta_vs_id342_p95": 0.01949450630919881
}
```

## Top Shape-Envelope Rows

| id | local | projected public | gap 0.55 | mean move | max move | lens | file |
|---:|---:|---:|---:|---:|---:|---|---|
| `342` | `0.534000` | `0.596319` | `+0.046319` | `0.000000` | `0.000000` | `stable_public_shape_envelope` | `base_oof_id_342_id178_id221_featuresim_shrink050_cv0.534000.csv` |
| `198` | `0.536380` | `0.598699` | `+0.048699` | `0.018455` | `0.120810` | `stable_public_shape_envelope` | `base_oof_id_198_id193_s3_smartphone_light_compact_overlay_cv0.536380.csv` |
| `193` | `0.536429` | `0.598747` | `+0.048747` | `0.018275` | `0.120810` | `stable_public_shape_envelope` | `base_oof_id_193_id191_sequence_summary_embedding_overlay_cv0.536429.csv` |
| `344` | `0.537839` | `0.600158` | `+0.050158` | `0.018135` | `0.120810` | `stable_public_shape_envelope` | `base_oof_id_344_id338_s3_phone_wearable_w0020_cv0.537839.csv` |
| `343` | `0.537854` | `0.600173` | `+0.050173` | `0.018043` | `0.120810` | `stable_public_shape_envelope` | `base_oof_id_343_id338_s3_phone_wearable_micro_w0005_cv0.537854.csv` |
| `338` | `0.537861` | `0.600180` | `+0.050180` | `0.018041` | `0.120810` | `stable_public_shape_envelope` | `base_oof_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861.csv` |
| `191` | `0.538009` | `0.600328` | `+0.050328` | `0.019079` | `0.120810` | `stable_public_shape_envelope` | `base_oof_id_191_id190_tempered_id177_remote_s1_anchor_blend_cv0.538009.csv` |
| `337` | `0.538066` | `0.600385` | `+0.050385` | `0.019403` | `0.123277` | `stable_public_shape_envelope` | `base_oof_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066.csv` |
| `179` | `0.538290` | `0.600609` | `+0.050609` | `0.017891` | `0.120810` | `stable_public_shape_envelope` | `base_oof_id_179_id178_tempered_id177_random_conv_anchor_blend_cv0.538290.csv` |
| `340` | `0.538426` | `0.600745` | `+0.050745` | `0.019204` | `0.123277` | `stable_public_shape_envelope` | `base_oof_id_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426.csv` |
| `341` | `0.538584` | `0.600902` | `+0.050902` | `0.019444` | `0.123277` | `stable_public_shape_envelope` | `base_oof_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584.csv` |

## Public Gap Correlations

| feature | pearson gap | spearman gap | pearson public | spearman public |
|---|---:|---:|---:|---:|
| `local_loss` | `-0.999` | `-0.286` | `-0.991` | `-0.167` |
| `mean_abs_delta_vs_id342` | `0.985` | `0.833` | `0.991` | `0.976` |
| `max_abs_delta_vs_id342` | `0.907` | `0.824` | `0.927` | `0.875` |
| `max_target_mean_abs_delta_vs_id342` | `0.957` | `0.824` | `0.971` | `0.875` |
| `max_target_abs_delta_vs_id342` | `0.907` | `0.824` | `0.927` | `0.875` |
| `prob_entropy` | `-0.987` | `-0.381` | `-0.995` | `-0.500` |
| `max_abs_mean_rate_shift` | `0.962` | `0.325` | `0.939` | `0.374` |
| `mean_abs_mean_rate_shift` | `0.987` | `0.310` | `0.970` | `0.238` |
| `extreme_probability_rate` | `0.994` | `0.325` | `0.997` | `0.325` |
