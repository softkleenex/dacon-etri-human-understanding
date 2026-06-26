# Post-ID262 public-anchor / pair-blend robustness

Lower robustness score is better. Pair this with source-risk and ID178 test-delta before choosing a public probe.

| id | local | robustness | ID178-gap est | max delta vs ID178 | mean delta vs ID178 | worst target | worst subject | worst date quartile |
|---:|---:|---:|---:|---:|---:|---|---:|---|
| 252 | 0.274420 | 0.315901 | 0.337193 | 0.807354 | 0.222174 | S3=0.465115 | 0.360637 | q4_late=0.297992 |
| 248 | 0.391735 | 0.421628 | 0.454509 | 0.538331 | 0.127223 | S3=0.465115 | 0.481982 | q4_late=0.409864 |
| 247 | 0.415523 | 0.443859 | 0.478297 | 0.548540 | 0.117523 | S3=0.465115 | 0.509363 | q4_late=0.433307 |
| 257 | 0.433079 | 0.462022 | 0.495853 | 0.176195 | 0.071503 | S4=0.503963 | 0.514418 | q4_late=0.453453 |
| 262 | 0.440031 | 0.469165 | 0.502805 | 0.179554 | 0.065694 | S4=0.503963 | 0.525330 | q4_late=0.460590 |
| 261 | 0.441919 | 0.471868 | 0.504693 | 0.159583 | 0.064392 | S4=0.510411 | 0.528367 | q4_late=0.463340 |
| 259 | 0.443567 | 0.472753 | 0.506340 | 0.178005 | 0.062329 | S4=0.503963 | 0.529734 | q4_late=0.464841 |
| 258 | 0.457647 | 0.487146 | 0.520420 | 0.159452 | 0.052985 | S4=0.520488 | 0.543692 | q4_late=0.479235 |
| 260 | 0.463684 | 0.493703 | 0.526458 | 0.119513 | 0.049071 | S4=0.531135 | 0.549316 | q4_late=0.485789 |
| 256 | 0.464121 | 0.495002 | 0.526894 | 0.131771 | 0.048946 | S4=0.534634 | 0.550170 | q4_late=0.488711 |
| 255 | 0.469816 | 0.500061 | 0.532590 | 0.115233 | 0.044999 | S4=0.535453 | 0.555812 | q4_late=0.493491 |
| 254 | 0.471985 | 0.502061 | 0.534758 | 0.146734 | 0.043987 | S4=0.529523 | 0.560945 | q4_late=0.495643 |
| 253 | 0.477131 | 0.507486 | 0.539904 | 0.106724 | 0.038948 | S4=0.538548 | 0.565300 | q4_late=0.500917 |
| 178 | 0.538638 | 0.571839 | 0.601411 | 0.000000 | 0.000000 | S4=0.602700 | 0.633714 | q4_late=0.566578 |

Practical read:
- ID260 is the new best first-probe challenger if we accept high-risk target-wise pair recombination: local `0.463684`, robustness `0.493703`, max delta vs ID178 `0.119513`.
- ID255 remains the cleaner conservative alternative: local `0.469816`, robustness `0.500061`, max delta `0.115233`, and a simpler cap0.12 anchor bridge.
- ID258 is still the best clean sharper bridge excluding the most extreme IDs 250/251/242/243, but it moves farther from ID178 than ID255/260.
- ID261 and ID262 are stress cards: local is lower, but they lean on ID252/ID259-style full-OOF frontier sources and larger ID178 movement.
- No DACON submission was made.
