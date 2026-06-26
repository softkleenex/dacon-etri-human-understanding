# Post-ID285 second-stage calibration robustness

IDs 283-285 were promoted from the post-ID282 second-stage calibration scan. Lower local is better, but every card here is full-OOF second-stage calibration and therefore high risk.

Confirmed ID178 gap reference: local `0.538638` -> Public `0.6014113009`, gap `+0.062773`.

The stability-veto proxy intentionally penalizes large test movement more heavily than local-score gain; it is a ranking aid, not a leaderboard estimate.

| role | id | base | local CV | score-pressure proxy | stability-veto proxy | max Δ vs ID178 | max Δ vs base | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| conservative_base272_temp_guard | 283 | 272 | 0.440206 | 0.502979 | 0.523259 | 0.200699 | 0.071721 | most conservative calibrated card; public-transfer lead among IDs 283-285 |
| frontier_base278_temp_guard | 285 | 278 | 0.356330 | 0.419103 | 0.578461 | 0.474428 | 0.218347 | frontier stress only; local is lowest but public-gap risk is large |
| mid_base280_temp_guard | 284 | 280 | 0.385005 | 0.447778 | 0.579899 | 0.409299 | 0.213604 | score-pressure card; much stronger local, materially more movement |

Recommendation: keep ID283 as the calibrated public-transfer lead, ID284 as the next score-pressure probe, and ID285 as a map of how far the local frontier can be pushed. None is Public 0.55 proof without DACON feedback.

No DACON submission was made.
