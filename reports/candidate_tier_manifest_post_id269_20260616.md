# Candidate tier manifest after ID269

ID267 is now the strongest cap0.12 score-targeting probe, but ID264 remains the cleaner cap0.12 option. ID268 is the best balanced sharper card for chasing Public 0.55 after first feedback. ID269 is a stress card: useful if the public gap of this family is benign, but not the first submission.

## New candidates
- ID267: Risk-penalized cap0.12 pair blend. Local 0.461960, robustness 0.492124, max delta 0.119969. Better score-targeting first probe than ID264 if limited stress-source exposure is acceptable.
- ID268: Risk-penalized cap0.15 pair blend. Local 0.448976, robustness 0.478923, max delta 0.149985. Best balanced sharper card.
- ID269: Risk-penalized cap0.16 pair blend. Local 0.443780, robustness 0.473962, max delta 0.159906. Lower-local stress card.

## Ranked candidates

| rank | id | local CV | robustness | ID178-gap estimate | max delta vs ID178 | risk note |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 267 | 0.461960 | 0.492124 | 0.524733 | 0.119969 | new score-targeting cap0.12 candidate; lower than ID264/260 but borrows limited target-wise signal from ID257/261 stress sources |
| 2 | 264 | 0.464758 | 0.494762 | 0.527532 | 0.119995 | cleaner cap0.12 first-probe baseline; no direct stress-source exposure |
| 3 | 260 | 0.463684 | 0.493703 | 0.526458 | 0.119513 | slightly sharper old cap0.12 card, but source path is less clean than ID264 and less locally strong than ID267 |
| 4 | 268 | 0.448976 | 0.478923 | 0.511750 | 0.149985 | best balanced sharper card; cap0.15 with local 0.448976 and controlled ID257 exposure |
| 5 | 266 | 0.457894 | 0.487400 | 0.520668 | 0.149809 | clean sharper card; cap0.15, almost ID258 local with lower max delta, no direct stress-source exposure |
| 6 | 265 | 0.458478 | 0.488017 | 0.521251 | 0.139926 | intermediate clean cap0.14 card |
| 7 | 269 | 0.443780 | 0.473962 | 0.506553 | 0.159906 | lower-local cap0.16 stress card; strong local but heavier ID257 exposure |
| 8 | 261 | 0.441919 | 0.471868 | 0.504693 | 0.159583 | sharp cap0.16 stress card using ID252-style source; not first probe |
| 9 | 262 | 0.440031 | 0.469165 | 0.502805 | 0.179554 | sharpest prior cap0.18 stress card; not first probe |
| 10 | 257 | 0.433079 | 0.462022 | 0.495853 | 0.176195 | single fine cap0.18 stress source; useful as ingredient but risky direct probe |
| 11 | 258 | 0.457647 | 0.487146 | 0.520420 | 0.159452 | clean cap0.16 bridge; lower local than ID266 but larger movement |
| 12 | 255 | 0.469816 | 0.500061 | 0.532590 | 0.115233 | simplest conservative fallback; fine cap0.12 anchor bridge |
| 13 | 253 | 0.477131 | 0.507486 | 0.539904 | 0.106724 | original cap0.12 public-anchor fallback |
| 14 | 259 | 0.443567 | 0.472753 | 0.506340 | 0.178005 | clean cap0.18 stress card; high movement |
| 15 | 252 | 0.274420 | 0.315901 | 0.337193 | 0.807354 | second-stage stress source only |

## Practical interpretation

- ID267 is the new cap0.12 local leader, but it is not as clean as ID264.
- ID268 is the most interesting Public-0.55 chase card: it moves local below 0.45 while keeping max delta near 0.15.
- ID269 gets close to the prior stress-card local range with a smaller cap than ID262, but it remains stress only.
- No DACON submission was made.
