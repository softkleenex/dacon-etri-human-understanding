# Post-ID252 ultra-frontier second-stage robustness

S3 finding: no existing candidate or S3 calibration improved below `0.465115`; ID250-ID252 gains come from non-S3 second-stage calibration.

| id | local | robustness | id178-gap estimate | max delta vs immediate base | note |
|---:|---:|---:|---:|---:|---|
| 242 | 0.357651 | 0.387740 | 0.420424 | 0.621692 | q240 stress reference |
| 243 | 0.360420 | 0.390285 | 0.423193 | 0.621692 | q240 all stress reference |
| 247 | 0.415523 | 0.443859 | 0.478297 | 0.198676 | best safer cap0.20 bridge |
| 248 | 0.391735 | 0.421628 | 0.454509 | 0.298791 | mid-risk cap0.30 bridge |
| 249 | 0.374465 | 0.405255 | 0.437238 | 0.399938 | high-risk cap0.40 bridge |
| 250 | 0.261184 | 0.303454 | 0.323957 | 0.249999 | raw local frontier; isotonic on already extreme ID242, severe public-gap risk |
| 251 | 0.263237 | 0.305248 | 0.326010 | 0.247992 | quantile-only ultra frontier alternative |
| 252 | 0.274420 | 0.315901 | 0.337193 | 0.306845 | ID249 mixed second-stage; slightly less discrete than ID250/251 |

Ranking by robustness: `[250, 251, 252, 242, 243, 249, 248, 247]`.
Risk-aware recommendation keeps ID247/ID248 ahead of ID250 despite the much lower raw local score.
ID250 is the new raw local minimum at 0.261184, but it is a severe full-OOF second-stage calibration stress test. ID251 confirms the low frontier with quantile-only calibration. ID252 is a less discrete mixed variant from ID249. For an actual public probe, ID247/ID248 remain more informative than directly submitting ID250.
