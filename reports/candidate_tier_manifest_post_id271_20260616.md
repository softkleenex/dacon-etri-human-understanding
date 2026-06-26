# Candidate tier manifest after ID271

Keep ID264/ID267 as first-feedback probes, ID268 as the balanced Public-0.55 chase card, and hold ID270/ID271 as stress follow-ups. ID271 is locally tempting but too dependent on ID257-style full-OOF oracle behavior to lead the next submission.

## New candidates
- ID270: Risk-penalized cap0.16/p0.05 pair blend. Local 0.440505, robustness 0.470154, max delta 0.159931. Strong but stress-heavy.
- ID271: Risk-penalized cap0.18/p0.10 pair blend. Local 0.435576, robustness 0.466134, max delta 0.176195. Lowest local but high ID257 inheritance.

## Ranked candidates

| rank | id | local CV | robustness | ID178-gap estimate | max delta vs ID178 | risk note |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 267 | 0.461960 | 0.492124 | 0.524733 | 0.119969 | score-targeting cap0.12 candidate; lower than ID264/260 but borrows limited target-wise signal from ID257/261 stress sources |
| 2 | 264 | 0.464758 | 0.494762 | 0.527532 | 0.119995 | cleaner cap0.12 first-probe baseline; no direct stress-source exposure |
| 3 | 260 | 0.463684 | 0.493703 | 0.526458 | 0.119513 | slightly sharper old cap0.12 card, but source path is less clean than ID264 and less locally strong than ID267 |
| 4 | 268 | 0.448976 | 0.478923 | 0.511750 | 0.149985 | best balanced sharper card; cap0.15 with local 0.448976 and controlled ID257 exposure |
| 5 | 266 | 0.457894 | 0.487400 | 0.520668 | 0.149809 | clean sharper card; cap0.15, almost ID258 local with lower max delta, no direct stress-source exposure |
| 6 | 265 | 0.458478 | 0.488017 | 0.521251 | 0.139926 | intermediate clean cap0.14 card |
| 7 | 269 | 0.443780 | 0.473962 | 0.506553 | 0.159906 | lower-local cap0.16 stress card; strong local but heavier ID257 exposure |
| 8 | 270 | 0.440505 | 0.470154 | 0.503278 | 0.159931 | new strongest cap0.16 stress card by local; lower local than ID269/261, but target choices lean on high-risk sources |
| 9 | 261 | 0.441919 | 0.471868 | 0.504693 | 0.159583 | sharp cap0.16 stress card using ID252-style source; not first probe |
| 10 | 262 | 0.440031 | 0.469165 | 0.502805 | 0.179554 | sharp prior cap0.18 stress card; not first probe |
| 11 | 271 | 0.435576 | 0.466134 | 0.498350 | 0.176195 | new lowest local controlled pair-blend stress card; heavily inherits ID257 and should not be first feedback |
| 12 | 257 | 0.433079 | 0.462022 | 0.495853 | 0.176195 | single fine cap0.18 stress source; useful as ingredient but risky direct probe |
| 13 | 258 | 0.457647 | 0.487146 | 0.520420 | 0.159452 | clean cap0.16 bridge; lower local than ID266 but larger movement |
| 14 | 255 | 0.469816 | 0.500061 | 0.532590 | 0.115233 | simplest conservative fallback; fine cap0.12 anchor bridge |
| 15 | 253 | 0.477131 | 0.507486 | 0.539904 | 0.106724 | original cap0.12 public-anchor fallback |
| 16 | 259 | 0.443567 | 0.472753 | 0.506340 | 0.178005 | clean cap0.18 stress card; high movement |
| 17 | 252 | 0.274420 | 0.315901 | 0.337193 | 0.807354 | second-stage stress source only |

## Practical interpretation

- ID267 is still the sharp cap0.12 score-targeting probe, while ID264 is the cleaner cap0.12 probe.
- ID268 remains the most useful balanced Public-0.55 chase card because it lowers local below 0.45 without jumping straight to the ID257-heavy region.
- ID270 is a respectable stress follow-up: similar cap to ID269/261 but slightly better local/robustness.
- ID271 should not seduce us just because it has local `0.435576`; it is closer to an ID257 stress inheritance test than a trustworthy generalization candidate.
- No DACON submission was made.
