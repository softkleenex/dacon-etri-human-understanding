# S3 hard-wall review post ID252

Best observed S3 local loss remains `0.465115`.

Evidence:
- all base OOF artifacts: no S3 candidate below the current anchor;
- S3 isotonic/quantile/temperature calibration: no improvement;
- strict hourly sequence overlay on ID247/ID249: no improvement;
- processed HR/BLE/screen/demo compact side models: no blend improved over base.

Best processed compact checks:

| candidate | raw S3 loss | best blend weight | best blend loss |
|---|---:|---:|---:|
| `s3_bio_context__lr_c003` | 0.601809 | `w0` | 0.465115 |
| `s3_bio_context__lr_c01` | 0.628627 | `w0` | 0.465115 |
| `s3_bio_context__hgb_l7` | 0.554852 | `w0` | 0.465115 |
| `s3_bio_context__hgb_l15` | 0.562177 | `w0` | 0.465115 |
| `s3_bio_context__et_d3` | 0.555590 | `w0` | 0.465115 |
| `s3_bio_context__rf_d3` | 0.574798 | `w0` | 0.465115 |
| `s3_hr_sleep_only__lr_c003` | 0.650127 | `w0` | 0.465115 |
| `s3_hr_sleep_only__lr_c01` | 0.671364 | `w0` | 0.465115 |
| `s3_hr_sleep_only__hgb_l7` | 0.619993 | `w0` | 0.465115 |
| `s3_hr_sleep_only__hgb_l15` | 0.626284 | `w0` | 0.465115 |
| `s3_hr_sleep_only__et_d3` | 0.624931 | `w0` | 0.465115 |
| `s3_hr_sleep_only__rf_d3` | 0.622422 | `w0` | 0.465115 |

Decision: preserve the current S3 anchor unless a genuinely different representation appears. Next S3 work should go to Kaggle as a raw multichannel sequence/ROCKET or CNN embedding experiment; local compact feature families are saturated.
