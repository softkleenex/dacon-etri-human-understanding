# Post-ID342 S3 phone/GPS interaction gate

Created: `2026-06-25 09:38:16 UTC+09:00`

This is report-only. It creates no feature matrix artifacts, predictions, submission artifacts, Kaggle runs, or DACON submissions.

## Decision

- Reviewed candidates: `60`
- Local gate pass count: `0`
- Best candidate: `s3_phone_gps_selected_concat_s3_lr_c0_20_w0.05`
- Best source: `s3_phone_gps_selected_concat`
- Best wins: `5`
- Best target mean/max delta vs ID342: `-0.001635922` / `-0.000068993`
- Best projected Public proxy: `0.595849549`
- Best gap to Public 0.55: `+0.045849549`
- Best interaction candidate: `s3_phone_gps_interaction_s3_lr_c0_20_w0.05`
- Interaction gate pass: `False`
- Run Kaggle now: `False`
- Submit now: `False`
- Reason: Focused phone/GPS S3 interaction is still far from the Public-0.55/review threshold. Treat it as design evidence only, not as a candidate route.

## Feature Frame

- Selected phone columns: `37`
- Selected GPS columns: `15`
- Interaction columns: `230`

## Top Rows

| rank | pass | source | candidate | wins | mean all Δ342 | max all Δ342 | target mean Δ342 | target max Δ342 | proxy | gap 0.55 |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `False` | `s3_phone_gps_selected_concat` | `s3_phone_gps_selected_concat_s3_lr_c0_20_w0.05` | 5 | -0.000233703 | -0.000009856 | -0.001635922 | -0.000068993 | 0.595849549 | +0.045849549 |
| 2 | `False` | `s3_phone_gps_selected_concat` | `s3_phone_gps_selected_concat_s3_lr_c0_20_w0.025` | 5 | -0.000132452 | -0.000013441 | -0.000927165 | -0.000094089 | 0.595950800 | +0.045950800 |
| 3 | `False` | `s3_phone_gps_selected_concat` | `s3_phone_gps_selected_concat_s3_lr_c0_08_w0.05` | 4 | -0.000123801 | +0.000156965 | -0.000866604 | +0.001098758 | 0.595959451 | +0.045959451 |
| 4 | `False` | `s3_phone_gps_interaction` | `s3_phone_gps_interaction_s3_lr_c0_20_w0.05` | 2 | -0.000113077 | +0.000264752 | -0.000791537 | +0.001853267 | 0.595970175 | +0.045970175 |
| 5 | `False` | `s3_phone_gps_interaction` | `s3_phone_gps_interaction_s3_lr_c0_08_w0.05` | 2 | -0.000086201 | +0.000220178 | -0.000603404 | +0.001541246 | 0.595997051 | +0.045997051 |
| 6 | `False` | `s3_phone_gps_interaction` | `s3_phone_gps_interaction_s3_lr_c0_20_w0.025` | 3 | -0.000078071 | +0.000085520 | -0.000546499 | +0.000598642 | 0.596005181 | +0.046005181 |
| 7 | `False` | `s3_phone_gps_selected_concat` | `s3_phone_gps_selected_concat_s3_lr_c0_08_w0.025` | 4 | -0.000077303 | +0.000070592 | -0.000541120 | +0.000494141 | 0.596005949 | +0.046005949 |
| 8 | `False` | `s3_phone_gps_interaction` | `s3_phone_gps_interaction_s3_lr_c0_08_w0.025` | 3 | -0.000061332 | +0.000069500 | -0.000429323 | +0.000486503 | 0.596021920 | +0.046021920 |
| 9 | `False` | `s3_phone_gps_selected_concat` | `s3_phone_gps_selected_concat_s3_lr_c0_20_w0.01` | 5 | -0.000057007 | -0.000007446 | -0.000399052 | -0.000052125 | 0.596026244 | +0.046026244 |
| 10 | `False` | `s3_phone_gps_interaction` | `s3_phone_gps_interaction_s3_lr_c0_20_w0.01` | 3 | -0.000036747 | +0.000021629 | -0.000257232 | +0.000151400 | 0.596046504 | +0.046046504 |
| 11 | `False` | `s3_phone_gps_selected_concat` | `s3_phone_gps_selected_concat_s3_lr_c0_08_w0.01` | 4 | -0.000034923 | +0.000026317 | -0.000244463 | +0.000184219 | 0.596048329 | +0.046048329 |
| 12 | `False` | `s3_phone_gps_selected_concat` | `s3_phone_gps_selected_concat_s3_lr_c0_20_w0.005` | 5 | -0.000029204 | -0.000004071 | -0.000204428 | -0.000028494 | 0.596054048 | +0.046054048 |
| 13 | `False` | `s3_phone_gps_interaction` | `s3_phone_gps_interaction_s3_lr_c0_08_w0.01` | 3 | -0.000029199 | +0.000016967 | -0.000204391 | +0.000118767 | 0.596054053 | +0.046054053 |
| 14 | `False` | `s3_phone_selected` | `s3_phone_selected_s3_lr_c0_20_w0.025` | 3 | -0.000019757 | +0.000094572 | -0.000138298 | +0.000662007 | 0.596063495 | +0.046063495 |
| 15 | `False` | `s3_phone_gps_interaction` | `s3_phone_gps_interaction_s3_lr_c0_20_w0.005` | 4 | -0.000019331 | +0.000008571 | -0.000135316 | +0.000059994 | 0.596063921 | +0.046063921 |
| 16 | `False` | `s3_phone_gps_selected_concat` | `s3_phone_gps_selected_concat_s3_lr_c0_08_w0.005` | 4 | -0.000018160 | +0.000012836 | -0.000127122 | +0.000089854 | 0.596065092 | +0.046065092 |
| 17 | `False` | `s3_phone_gps_interaction` | `s3_phone_gps_interaction_s3_lr_c0_08_w0.005` | 3 | -0.000015407 | +0.000006561 | -0.000107849 | +0.000045924 | 0.596067845 | +0.046067845 |
| 18 | `False` | `s3_phone_selected` | `s3_phone_selected_s3_lr_c0_20_w0.05` | 3 | -0.000014091 | +0.000203228 | -0.000098636 | +0.001422593 | 0.596069161 | +0.046069161 |
| 19 | `False` | `s3_phone_selected` | `s3_phone_selected_s3_lr_c0_20_w0.01` | 3 | -0.000011127 | +0.000036126 | -0.000077890 | +0.000252885 | 0.596072125 | +0.046072125 |
| 20 | `False` | `s3_phone_selected` | `s3_phone_selected_s3_lr_c0_20_w0.005` | 3 | -0.000006118 | +0.000017778 | -0.000042828 | +0.000124448 | 0.596077134 | +0.046077134 |
| 21 | `False` | `s3_phone_gps_interaction` | `s3_phone_gps_interaction_s3_lr_c0_02_w0.005` | 2 | -0.000001202 | +0.000023257 | -0.000008413 | +0.000162801 | 0.596082050 | +0.046082050 |
| 22 | `False` | `s3_phone_gps_interaction` | `s3_phone_gps_interaction_s3_lr_c0_02_w0.01` | 2 | -0.000001001 | +0.000047246 | -0.000007005 | +0.000330719 | 0.596082251 | +0.046082251 |
| 23 | `False` | `s3_gps_selected` | `s3_gps_selected_s3_lr_c0_20_w0.005` | 4 | -0.000000488 | +0.000029294 | -0.000003413 | +0.000205059 | 0.596082764 | +0.046082764 |
| 24 | `False` | `s3_gps_selected` | `s3_gps_selected_s3_lr_c0_20_w0.01` | 3 | +0.000000317 | +0.000059039 | +0.000002217 | +0.000413273 | 0.596083569 | +0.046083569 |
| 25 | `False` | `s3_phone_selected` | `s3_phone_selected_s3_lr_c0_08_w0.005` | 1 | +0.000002010 | +0.000032017 | +0.000014073 | +0.000224122 | 0.596085262 | +0.046085262 |
