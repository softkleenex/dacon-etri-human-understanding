# ID178 confirmed-public anchor bridge scan

This scan moves from confirmed Public anchor ID178 toward stronger local frontier artifacts under target-wise max test-delta caps.

Anchor public score: `0.6014113009`
Frontier IDs: `[247, 248, 252]`

| cap | local CV | max delta vs ID178 | changed targets | ID178-gap estimate |
|---:|---:|---:|---:|---:|
| 0.030 | 0.525702 | 0.028057 | 7 | 0.588475 |
| 0.050 | 0.517452 | 0.047700 | 7 | 0.580226 |
| 0.080 | 0.500179 | 0.078580 | 7 | 0.562953 |
| 0.100 | 0.491342 | 0.098225 | 7 | 0.554116 |
| 0.120 | 0.477131 | 0.106724 | 7 | 0.539904 |
| 0.150 | 0.471985 | 0.146734 | 7 | 0.534758 |
| 0.200 | 0.443234 | 0.190923 | 7 | 0.506008 |
| 0.250 | 0.433257 | 0.242206 | 7 | 0.496030 |
| 0.300 | 0.403685 | 0.284808 | 7 | 0.466458 |

Recommended promotions from this scan:
- Conservative probe: cap `0.120` (ID253), local `0.477131`, max delta `0.106724`.
- Medium probe: cap `0.150` (ID254), local `0.471985`, max delta `0.146734`.

Caution: this is full-OOF target-wise selection and not strict Subject-Hole proof. It is safer than raw ultra-frontier candidates only because movement is capped from a public-confirmed anchor.
