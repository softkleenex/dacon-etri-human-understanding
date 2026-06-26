# ID338 S3 phone/wearable weight geometry scan

Created: `2026-06-24 00:06:40 UTC+09:00`

This is report-only. It writes no full-test predictions, submission CSV, base artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.

## Weight Rows

| weight | candidate | wins | mean Δ338 | max Δ338 | mean move | max move | mean all move | flags |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0.0025 | `blend_id338_phone_wearable_discordance_lr_c0_20_w0.0025` | 4 | -0.000032 | 0.000029 | 0.000397 | 0.001826 | 0.000057 | `mean_delta_not_below_gate` |
| 0.0050 | `blend_id338_phone_wearable_discordance_lr_c0_20_w0.005` | 4 | -0.000062 | 0.000058 | 0.000793 | 0.003652 | 0.000113 | `` |
| 0.0100 | `blend_id338_phone_wearable_discordance_lr_c0_20_w0.01` | 4 | -0.000115 | 0.000123 | 0.001587 | 0.007305 | 0.000227 | `` |
| 0.0200 | `blend_id338_phone_wearable_discordance_lr_c0_20_w0.02` | 4 | -0.000192 | 0.000267 | 0.003173 | 0.014610 | 0.000453 | `` |
| 0.0350 | `blend_id338_phone_wearable_discordance_lr_c0_20_w0.035` | 3 | -0.000240 | 0.000525 | 0.005554 | 0.025567 | 0.000793 | `mean_all_delta_vs_id338_gt_0.00075,max_all_delta_vs_id338_gt_0.020,primary_wins_lt_4` |
| 0.0500 | `blend_id338_phone_wearable_discordance_lr_c0_20_w0.05` |  |  |  | 0.007934 | 0.036524 | 0.001133 | `mean_all_delta_vs_id338_gt_0.00075,max_all_delta_vs_id338_gt_0.020,primary_wins_lt_4,mean_delta_not_below_gate,max_delta_above_gate` |

## Decision

- Scan completed: `True`
- Selected candidate: `blend_id338_phone_wearable_discordance_lr_c0_20_w0.02`
- Selected weight: `0.02`
- Public `0.58` evidence: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Next local allowed step: `promotion_contract_review_for_bolder_id338_s3_micro_overlay`
- Next action: Selected weight is still no-submit; create an artifact only through a separate promotion script and audit.

## Selected Config

```json
{
  "target": "S3",
  "candidate": "blend_id338_phone_wearable_discordance_lr_c0_20_w0.02",
  "source_group": "phone_wearable_discordance",
  "model_spec": {
    "name": "lr_c0_20",
    "kind": "lr",
    "c_value": 0.2
  },
  "weight": 0.02,
  "probe_summary": {
    "target": "S3",
    "source_group": "phone_wearable_discordance",
    "candidate": "blend_id338_phone_wearable_discordance_lr_c0_20_w0.02",
    "model_spec": {
      "name": "lr_c0_20",
      "kind": "lr",
      "c_value": 0.2
    },
    "weight": 0.02,
    "primary_wins_vs_id338": 4,
    "mean_delta_vs_id338": -0.00019225793876567198,
    "min_delta_vs_id338": -0.0009907543372417171,
    "max_delta_vs_id338": 0.0002671799229903571,
    "mean_target_movement_vs_id338": 0.0022915331203010986,
    "max_target_movement_vs_id338": 0.01780473128556559,
    "mean_failed_projection": 0.0005894365562974696,
    "split_deltas_vs_id338": {
      "subject_hole_5": -0.00015472321083154217,
      "actual_test_band_holdout": 0.0002671799229903571,
      "test_boundary_neighbor2_holdout": -5.573848120843383e-05,
      "test_boundary_neighbor5_holdout": -0.0009907543372417171,
      "loso_10": -2.7253587537023893e-05
    },
    "split_mean_movements_vs_id338": {
      "subject_hole_5": 0.0034647657293875103,
      "actual_test_band_holdout": 0.0007571744999328849,
      "test_boundary_neighbor2_holdout": 0.0005584853308317663,
      "test_boundary_neighbor5_holdout": 0.001521401877571041,
      "loso_10": 0.005155838163782291
    }
  },
  "source_prediction_mean": 0.6404835688565824,
  "source_prediction_min": 0.015441504156825905,
  "source_prediction_max": 0.9984312266745693,
  "target_mean_shift_vs_id338": 6.93692857970385e-05,
  "mean_abs_target_vs_id338": 0.003173453438307452,
  "max_abs_target_vs_id338": 0.014609685987716725,
  "mean_abs_all_targets_vs_id338": 0.0004533504911867788,
  "max_abs_all_targets_vs_id338": 0.014609685987716725,
  "joint_deltas_vs_id338": {
    "target_mean_l1_to_train": -9.909897971005499e-06,
    "target_mean_max_abs_to_train": 0.0,
    "pair11_l1_to_train": 1.3961522788877995e-05,
    "temporal_volatility_ratio_to_train": -0.0006205424186233599,
    "prediction_entropy": 0.00015998319530741156,
    "prediction_sharpness": -0.00013358913670780126,
    "extreme_rate_05_95": 0.0,
    "extreme_rate_01_99": 0.0
  },
  "risk_flags": [],
  "geometry_pass": true
}
```

