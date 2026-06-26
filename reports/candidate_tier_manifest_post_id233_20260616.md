# Candidate tier manifest after ID233

Best confirmed Public remains ID178 `0.6014113009`. Public 0.55 is not proven.

## Recommendation

If probing for Public 0.55 without jumping to broad all-target calibration, ID232 is the least reckless new score-margin card. ID233 is the sharper but riskier alternative. DACON submission still requires explicit user approval.

## Candidate rows

- ID221: local `0.532990`, robustness `0.564811`, ID178-gap estimate `0.595763` — cleanest non-ID172 local branch; no public 0.55 margin
- ID224: local `0.509110`, robustness `0.539895`, ID178-gap estimate `0.571883` — least reckless ID221-ID172 bridge; still non-nested extreme source
- ID226: local `0.491771`, robustness `0.524870`, ID178-gap estimate `0.554544` — targetwise convex local frontier; very exposed to ID172 except partial S4 clean weight
- ID228: local `0.482245`, robustness `0.513849`, ID178-gap estimate `0.545018` — all-target isotonic except S3; strong margin but broad full-OOF calibration risk
- ID230: local `0.472703`, robustness `0.504400`, ID178-gap estimate `0.535476` — S4 q80 all-target frontier; stronger local but very high S4 calibration risk
- ID231: local `0.491287`, robustness `0.524343`, ID178-gap estimate `0.554060` — outer-supported Q2 temp only; safer repair but no public 0.55 nominal margin
- ID232: local `0.486685`, robustness `0.516793`, ID178-gap estimate `0.549458` — new Q2 temp + S4 isotonic partial candidate; less broad than ID228 and has thin ID178-gap margin
- ID233: local `0.484413`, robustness `0.513300`, ID178-gap estimate `0.547187` — new Q2 temp + S4 q40 partial candidate; better local/robustness than ID232 but larger S4 movement risk

## New partial calibration cards

- ID232 is the preferred new probe if we want target-margin while avoiding ID228 broad calibration exposure.
- ID233 is retained as a higher-risk S4-q40 score-margin card.