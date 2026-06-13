# Submissions Integrity Audit

Found **1 findings**. Safe default ensemble rows: **14**.

## Status Counts

- `submitted_placeholder_score`: 33
- `generated_unsubmitted`: 29
- `scored`: 14
- `submitted_unscored`: 11
- `cv_lb_gap_risk`: 5
- `invalid_public_score`: 1

## Findings

### 1. [MEDIUM] Risk-labelled rows exist

- Area: `log`
- Detail: {'submitted_placeholder_score': 33, 'submitted_unscored': 11, 'cv_lb_gap_risk': 5, 'invalid_public_score': 1}
- Recommendation: Keep rows for history, but exclude them from default ensembling and score claims.
