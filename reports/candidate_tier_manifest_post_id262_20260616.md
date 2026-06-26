# Candidate tier manifest after ID262

For a single next public probe, ID260 is now the best score-targeting candidate: it improves local CV over ID255 while keeping max test movement essentially in the cap0.12 band. If the priority is structural simplicity over score pressure, use ID255 first instead. ID261/ID262 are not first probes; they are stress cards after ID260/ID255/ID258 evidence.

## New candidates
- ID260: Cap0.12 target-wise pair blend, local 0.463684, robustness 0.493703, max delta 0.119513 vs ID178. New best first-probe challenger if full-OOF pair risk is acceptable.
- ID261: Cap0.16 pair blend, local 0.441919, robustness 0.471868, max delta 0.159583. Second/stress card because it leans on ID252.
- ID262: Cap0.18 pair blend, local 0.440031, robustness 0.469165, max delta 0.179554. Sharp stress card, not first probe.

## Ranked candidates

| rank | id | local CV | robustness | ID178-gap estimate | max delta vs ID178 | risk note |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 260 | 0.463684 | 0.493703 | 0.526458 | 0.119513 | best first public-probe challenger: cap0.12 pair blend, low local, near-ID255 movement, but target-wise full-OOF recombination |
| 2 | 255 | 0.469816 | 0.500061 | 0.532590 | 0.115233 | cleaner conservative first-probe alternative: simpler fine cap0.12 anchor bridge despite small ultra-frontier source risk |
| 3 | 258 | 0.457647 | 0.487146 | 0.520420 | 0.159452 | best clean sharper bridge: excludes IDs 250/251/242/243, but uses larger cap0.16 movement |
| 4 | 253 | 0.477131 | 0.507486 | 0.539904 | 0.106724 | original cap0.12 public-anchor bridge; clean fallback with smallest nonzero max delta |
| 5 | 261 | 0.441919 | 0.471868 | 0.504693 | 0.159583 | sharp pair-blend second/stress card: local 0.441919, cap0.16, but leans on ID252 second-stage source |
| 6 | 262 | 0.440031 | 0.469165 | 0.502805 | 0.179554 | sharpest pair-blend stress card in this family: local 0.440031, cap0.18, but high movement and stress sources |
| 7 | 259 | 0.443567 | 0.472753 | 0.506340 | 0.178005 | clean cap0.18 bridge; avoids most extreme sources but high movement stress card |
| 8 | 257 | 0.433079 | 0.462022 | 0.495853 | 0.176195 | fine cap0.18 bridge; high-risk stress card |
| 9 | 247 | 0.415523 | 0.443859 | 0.478297 | 0.548540 | best safer cap0.20 frontier bridge by old local ranking, but too far from ID178 on test deltas |
| 10 | 248 | 0.391735 | 0.421628 | 0.454509 | 0.538331 | mid-risk cap0.30 frontier bridge; test movement too large for first probe |
| 11 | 252 | 0.274420 | 0.315901 | 0.337193 | 0.807354 | ID249 mixed second-stage source; useful as ingredient/stress source, not direct probe |

## Practical interpretation

- ID260 is the best single score-targeting probe candidate after ID262, but it must be treated as high-risk because target choices came from full-OOF pair recombination.
- ID255 is the cleaner conservative candidate if we want to probe with the simplest cap0.12 movement from the confirmed-public ID178 anchor.
- ID258 remains the clean sharper bridge excluding the most extreme IDs, useful if ID260 looks too recombinational but ID255 is too timid.
- ID261 and ID262 have attractive local numbers but should wait until at least one safer public-anchor/pair-blend candidate has leaderboard feedback.

No DACON submission was made.
