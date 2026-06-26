# Post-ID341 Public/Local Gap Diagnosis

## Decision

- Create submission artifact: `False`
- Submit now: `False`
- Public 0.58 local CV target under recent floor: `0.517445`
- Public 0.58 local CV target under recent stress: `0.494927`
- Next action: Do not promote local CV-only candidates. The next score-relevant move remains the explicit-approved no-submit Kaggle/Linux upstream rebuild, followed by a separate Public-0.58 artifact-contract review only if pulled reports are safe, beat the current confirmed public anchor band, and show min_proxy <= 0.58; ID178-only improvement above 0.58 is a lead, not ID342.

## Stress references

- Reference gaps: `{'id178_gap': 0.062773301, 'id329_gap': 0.085072858, 'id337_gap': 0.062554629, 'id340_gap': 0.062953175}`
- Recent floor gap: `0.062554629`
- Recent stress gap: `0.085072858`

## Watch candidates by recent proxy

| id | local_cv | floor_proxy | stress_proxy | pass_floor | pass_stress |
|---:|---:|---:|---:|:---:|:---:|
| 342 | 0.534000 | 0.596555 | 0.619073 | no | no |
| 339 | 0.536441 | 0.598996 | 0.621514 | no | no |
| 336 | 0.537663 | 0.600218 | 0.622736 | no | no |
| 338 | 0.537861 | 0.600416 | 0.622934 | no | no |
| 335 | 0.539690 | 0.602245 | 0.624763 | no | no |
| 334 | 0.540347 | 0.602902 | 0.625420 | no | no |

## Interpretation

Recent public evidence says local CV alone is not a promotion signal. The current watch candidates only look promising under older optimistic gaps, not under the recent confirmed-control transfer floor. Therefore this report does not authorize a DACON submission or a new local ID342 artifact.

The Mac/M4 suspicion is tracked separately, but the existing parity evidence and the current dependency snapshot do not make it the primary explanation for the public-local gap.
