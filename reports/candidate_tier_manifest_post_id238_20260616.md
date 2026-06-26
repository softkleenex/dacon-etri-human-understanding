# Candidate tier manifest after ID238

Best confirmed Public remains ID178 `0.6014113009`. Public 0.55 is not proven.

## Recommendation

ID235 remains the balanced next Public-0.55 probe. ID238 is now the strongest local/aggressive probe, but its full-OOF oracle recombination makes it riskier than ID235/237. DACON submission requires explicit user approval.

## Candidate rows

- ID221: local `0.532990`, robustness `0.564811`, ID178-gap estimate `0.595763` — clean non-ID172 branch; no public 0.55 margin
- ID226: local `0.491771`, robustness `0.524870`, ID178-gap estimate `0.554544` — targetwise bridge frontier; local just above 0.55-gap line
- ID227: local `0.464409`, robustness `0.495627`, ID178-gap estimate `0.527182` — previous lowest local; all-target isotonic plus S4 q160, ultra-risk
- ID230: local `0.472703`, robustness `0.504400`, ID178-gap estimate `0.535476` — all-target isotonic plus S4 q80; strong local, high S4 risk
- ID235: local `0.482255`, robustness `0.511518`, ID178-gap estimate `0.545029` — best balanced mid-risk score-margin card from temp-grid plus S4 q40
- ID236: local `0.474985`, robustness `0.505914`, ID178-gap estimate `0.537758` — temp-grid plus S4 q80; lower local, high S4 risk
- ID237: local `0.466691`, robustness `0.497142`, ID178-gap estimate `0.529464` — temp-grid plus S4 q160; cleaner non-S4 low-local frontier
- ID238: local `0.462365`, robustness `0.493658`, ID178-gap estimate `0.525138` — new local frontier; full-OOF targetwise clean-anchor/ultra-calibration logit oracle, highest risk

## New interpretation

- ID238 improves local frontier from ID227 `0.464409` to `0.462365` by logit-blending ID221 clean anchors with ID227 ultra calibration target-wise.
- ID238 is not safer validation evidence; it is an aggressive leaderboard-response card.