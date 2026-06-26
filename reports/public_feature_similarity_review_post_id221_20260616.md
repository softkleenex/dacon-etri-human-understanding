# Public feature-similarity review after ID221

ID221 is the current best cleaner/non-ID172 branch: local `0.532990`, robustness `0.564811`.
It improves ID218 by overall `-0.001113` and robustness `-0.000822`.

Target movement versus ID218:
- Q1: `-0.003953`
- Q2: `-0.000518`
- Q3: `-0.000537`
- S1: `-0.000128`
- S2: `-0.000636`
- S3: `-0.001056`
- S4: `-0.000964`

External Public 0.5917 notebook verdict:
- Claimed Public `0.5917` and local OOF `0.514`, with a large claimed gap `0.0777`.
- Useful ideas: conservative anchor blending, subject-hole validation, stability feature selection, and public-gap risk ranking.
- Not plug-and-play: missing anchor artifacts and full executable feature construction.

Decision: keep ID221 as the preferred clean local branch, not as public proof. Confirmed best Public remains ID178 at `0.6014113009`.
