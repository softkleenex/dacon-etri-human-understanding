# Candidate outer-selector diagnostic — post ID335

Created: `2026-06-17 22:38:50 UTC+09:00`

This is report-only. It nests only the choice among existing OOF artifacts; candidate artifacts themselves may still be non-nested.

## Split summary

| split | overall | robust | worst target | worst subject | note |
|---|---:|---:|---|---|---|
| `subject_hole_choice_nested` | 0.521786 | 0.547322 | S4 `0.579557` | id10 `0.623402` | selection-nested only |
| `loso_choice_nested` | 0.522195 | 0.547759 | S4 `0.579557` | id10 `0.623767` | selection-nested only |

## Full-train selected targets

| target | selected id | train loss |
|---|---:|---:|
| Q1 | 310 | 0.529879 |
| Q2 | 311 | 0.543787 |
| Q3 | 310 | 0.542047 |
| S1 | 310 | 0.451908 |
| S2 | 310 | 0.519440 |
| S3 | 310 | 0.483970 |
| S4 | 310 | 0.579557 |

## Verdict

Choice-nested selection still prefers non-nested failed-direction artifacts in several folds/targets, but this does not repair the underlying artifact-generation leakage. Treat it as a map for a future true fold-nested rebuild, not as a submission gate.

