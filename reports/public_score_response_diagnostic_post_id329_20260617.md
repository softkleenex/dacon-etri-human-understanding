# Public score response diagnostic post-ID329

- DACON submission id: `1465639`
- Submitted at: `2026-06-17 00:45:07`
- File: `submission_id_329_post328_combo4_cap080_budget050_practical_cv0.516718_unsubmitted.csv`
- Local CV: `0.516718`
- Public LB: `0.6017908577`
- Gap: `+0.085073`

| id | role | local CV | public LB | gap | delta vs ID178 public |
|---:|---|---:|---:|---:|---:|
| 178 | confirmed_public_anchor | 0.538638 | 0.6014113009 | +0.062774 | +0.0000000000 |
| 287 | confirmed_public_failure | 0.378934 | 0.6698787318 | +0.290945 | +0.0684674309 |
| 329 | guarded_practical_probe | 0.516718 | 0.6017908577 | +0.085073 | +0.0003795568 |

## Interpretation

ID329 did not collapse like ID287, so the failed-direction guard helped safety. But it did not improve over ID178; it is slightly worse by `+0.0003795568` on Public.

Do not submit ID330/ID331 immediately. They are same-family bolder moves, and ID329 says that this local-improvement direction is not paying on Public.
