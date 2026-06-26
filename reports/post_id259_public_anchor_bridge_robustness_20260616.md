# Post-ID259 public-anchor bridge robustness

Lower robustness score is better. Pair this with source-risk and ID178 test-delta before choosing a public probe.

| id | local | robustness | ID178-gap est | max delta vs ID178 | worst target | worst subject | worst date quartile |
|---:|---:|---:|---:|---:|---|---:|---|
| 252 | 0.274420 | 0.315901 | 0.337193 | 0.807354 | S3=0.465115 | 0.360637 | q4_late=0.297992 |
| 248 | 0.391735 | 0.421628 | 0.454509 | 0.538331 | S3=0.465115 | 0.481982 | q4_late=0.409864 |
| 247 | 0.415523 | 0.443859 | 0.478297 | 0.548540 | S3=0.465115 | 0.509363 | q4_late=0.433307 |
| 257 | 0.433079 | 0.462022 | 0.495853 | 0.176195 | S4=0.503963 | 0.514418 | q4_late=0.453453 |
| 259 | 0.443567 | 0.472753 | 0.506340 | 0.178005 | S4=0.503963 | 0.529734 | q4_late=0.464841 |
| 258 | 0.457647 | 0.487146 | 0.520420 | 0.159452 | S4=0.520488 | 0.543692 | q4_late=0.479235 |
| 256 | 0.464121 | 0.495002 | 0.526894 | 0.131771 | S4=0.534634 | 0.550170 | q4_late=0.488711 |
| 255 | 0.469816 | 0.500061 | 0.532590 | 0.115233 | S4=0.535453 | 0.555812 | q4_late=0.493491 |
| 254 | 0.471985 | 0.502061 | 0.534758 | 0.146734 | S4=0.529523 | 0.560945 | q4_late=0.495643 |
| 253 | 0.477131 | 0.507486 | 0.539904 | 0.106724 | S4=0.538548 | 0.565300 | q4_late=0.500917 |
| 178 | 0.538638 | 0.571839 | 0.601411 | 0.000000 | S4=0.602700 | 0.633714 | q4_late=0.566578 |

Practical read:
- ID255 remains the first public-probe candidate if the goal is controlled movement from ID178.
- ID258 is the cleaner sharper alternative: lower local than ID256 with no ID250/251/242/243 sources, but a larger cap than ID255.
- ID259/257 are stress cards, not first probes.
