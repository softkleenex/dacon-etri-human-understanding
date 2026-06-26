# Candidate tier manifest after ID188

This manifest separates score ambition from public-gap risk. It is not a submission command.

| tier | ids | interpretation |
|---|---|---|
| anchor_safe_reference | 179 | Closest post-ID178 anchor. It reduces high-C random-conv exposure while barely changing predictions. Safest reference but not score-ambitious. |
| moderate_counterprobe | 181, 182, 180 | Best compromise zone: meaningful local improvement, moderate movement from ID179, broad OOF improvement. Still non-nested because ID172 calibration is involved. |
| bold_counterprobe | 183, 184, 185 | Lower local scores with high delta from ID179. Use only if deliberately probing leaderboard response to tempered ID172 calibration. |
| extreme_counterprobe_only | 186, 187, 188, 172 | Very exposed to ID172 full-OOF calibration. Useful for mapping the local-score frontier, not for safe score claims. |
| failed_public_reference | 178 | Actual DACON Public 0.601411 despite local 0.538638; use as the calibration warning case. |

## Recommended next actions
- First pull/analyze Kaggle v23 random-conv sweep when it finishes; prioritize cleaner non-extreme improvements.
- If a DACON probe is explicitly requested and risk appetite is low, prefer ID179 or ID181 over local-minimum ID188.
- If a DACON probe is explicitly requested and risk appetite is moderate, ID182 is the current compromise candidate.
- If mapping leaderboard response to extreme calibration is the goal, ID183-ID185 are bold probes; ID186-ID188 are extreme counterprobes only.
- Never submit any candidate without explicit user approval.

## Key warning

ID178 scored Public `0.6014113009` from local `0.538638`, so local-minimum candidates such as ID188 must not be treated as public-ready without a real DACON result.
