# Submissions Integrity Audit

Found **1 findings**. Safe default ensemble rows: **15**.

## Status Counts

- `generated_unsubmitted`: 272
- `submitted_placeholder_score`: 33
- `cv_lb_gap_risk`: 16
- `scored`: 15
- `submitted_unscored`: 12
- `invalid_public_score`: 1

## Findings

### 1. [MEDIUM] Risk-labelled rows exist

- Area: `log`
- Detail: {'submitted_placeholder_score': 33, 'cv_lb_gap_risk': 16, 'submitted_unscored': 12, 'invalid_public_score': 1}
- Recommendation: Keep rows for history, but exclude them from default ensembling and score claims.
