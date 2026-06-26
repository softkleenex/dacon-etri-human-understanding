# Candidate tier manifest after ID263

ID263 is now the preferred first public probe if we want a balanced risk story: it nearly matches ID260 but excludes direct stress IDs from the pair-blend scan. ID260 remains the sharper score-pressure candidate. ID255 is the simplest conservative fallback. ID261/ID262 remain stress cards only.

## New candidate
- ID263: Direct-lite cap0.12 pair blend from IDs 178/253/255/258 only. Local 0.464978, robustness 0.495013, max delta 0.119513. It nearly matches ID260 while avoiding direct stress IDs in the pair search.

## Ranked candidates

| rank | id | local CV | robustness | ID178-gap estimate | max delta vs ID178 | risk note |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 263 | 0.464978 | 0.495013 | 0.527751 | 0.119513 | best safer first-probe challenger: direct-lite cap0.12 pair blend excluding direct stress IDs 252/257/259/260/261/262; only 0.001294 worse than ID260 local |
| 2 | 260 | 0.463684 | 0.493703 | 0.526458 | 0.119513 | slightly sharper score-pressure candidate: cap0.12 pair blend, but directly uses ID259/248/258/253 ingredients target-wise |
| 3 | 255 | 0.469816 | 0.500061 | 0.532590 | 0.115233 | simplest conservative first-probe fallback: fine cap0.12 anchor bridge, no pair-blend recombination |
| 4 | 258 | 0.457647 | 0.487146 | 0.520420 | 0.159452 | best clean sharper bridge excluding IDs 250/251/242/243, but larger cap0.16 movement |
| 5 | 253 | 0.477131 | 0.507486 | 0.539904 | 0.106724 | original cap0.12 public-anchor bridge; clean fallback with smallest nonzero max delta |
| 6 | 261 | 0.441919 | 0.471868 | 0.504693 | 0.159583 | sharp pair-blend second/stress card: local 0.441919, cap0.16, but leans on ID252 second-stage source |
| 7 | 262 | 0.440031 | 0.469165 | 0.502805 | 0.179554 | sharpest pair-blend stress card: local 0.440031, cap0.18, high movement/stress-source risk |
| 8 | 259 | 0.443567 | 0.472753 | 0.506340 | 0.178005 | clean cap0.18 bridge; high movement stress card |
| 9 | 257 | 0.433079 | 0.462022 | 0.495853 | 0.176195 | fine cap0.18 bridge; high-risk stress card |
| 10 | 247 | 0.415523 | 0.443859 | 0.478297 | 0.548540 | old local frontier bridge source; direct test movement too large for first probe |
| 11 | 248 | 0.391735 | 0.421628 | 0.454509 | 0.538331 | mid-risk cap0.30 source; direct test movement too large for first probe |
| 12 | 252 | 0.274420 | 0.315901 | 0.337193 | 0.807354 | second-stage stress source; ingredient only, not direct probe |

## Practical interpretation

- ID263 is the cleanest score-targeting first probe produced after the ID260 pair-blend search.
- ID260 is only `0.001294` better locally, so the extra risk may not be worth it as the first public check.
- If we want maximum caution, ID255 remains valid, but ID263 gives more public-0.55 margin with a still-controlled ID178 delta.

No DACON submission was made.
