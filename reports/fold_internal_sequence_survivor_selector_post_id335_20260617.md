# Fold-internal sequence survivor selector — post ID335

Created: `2026-06-17 23:20:31 UTC+09:00`

This is report-only. It nests only the target-wise choice among existing OOF artifacts; source artifacts remain non-nested.

## Results

| split | scenario | overall | robust | worst target | worst subject |
|---|---|---:|---:|---|---|
| `subject_hole_fold_internal` | `id179_anchor_only` | 0.538638 | 0.564192 | S4 `0.602700` | id10 `0.633714` |
| `subject_hole_fold_internal` | `sequence_local_only` | 0.537268 | 0.562957 | S4 `0.604577` | id10 `0.625078` |
| `subject_hole_fold_internal` | `sequence_cap008_penalty` | 0.538638 | 0.564192 | S4 `0.602700` | id10 `0.633714` |
| `subject_hole_fold_internal` | `sequence_cap012_penalty` | 0.538638 | 0.564192 | S4 `0.602700` | id10 `0.633714` |
| `subject_hole_fold_internal` | `sequence_cap014_penalty` | 0.538638 | 0.564192 | S4 `0.602700` | id10 `0.633714` |
| `subject_hole_fold_internal` | `sequence_with_id180_stress` | 0.538638 | 0.564192 | S4 `0.602700` | id10 `0.633714` |
| `loso_fold_internal` | `id179_anchor_only` | 0.538638 | 0.564192 | S4 `0.602700` | id10 `0.633714` |
| `loso_fold_internal` | `sequence_local_only` | 0.536920 | 0.562563 | S4 `0.601977` | id10 `0.627105` |
| `loso_fold_internal` | `sequence_cap008_penalty` | 0.538638 | 0.564192 | S4 `0.602700` | id10 `0.633714` |
| `loso_fold_internal` | `sequence_cap012_penalty` | 0.538638 | 0.564192 | S4 `0.602700` | id10 `0.633714` |
| `loso_fold_internal` | `sequence_cap014_penalty` | 0.538638 | 0.564192 | S4 `0.602700` | id10 `0.633714` |
| `loso_fold_internal` | `sequence_with_id180_stress` | 0.538638 | 0.564192 | S4 `0.602700` | id10 `0.633714` |

## Best rows and selected IDs

| split | best scenario | overall | robust | selected IDs |
|---|---|---:|---:|---|
| `subject_hole_fold_internal` | `sequence_local_only` | 0.537268 | 0.562957 | `[178, 179, 190, 193, 198, 200, 203, 206]` |
| `loso_fold_internal` | `sequence_local_only` | 0.536920 | 0.562563 | `[178, 179, 190, 191, 193, 198, 200, 203, 206]` |

## Choice counts for best rows

### subject_hole_fold_internal

- `Q1`: `{178: 1, 193: 1, 200: 3}`
- `Q2`: `{193: 5}`
- `Q3`: `{206: 5}`
- `S1`: `{190: 5}`
- `S2`: `{206: 3, 179: 1, 178: 1}`
- `S3`: `{198: 4, 178: 1}`
- `S4`: `{206: 3, 178: 1, 203: 1}`

### loso_fold_internal

- `Q1`: `{200: 10}`
- `Q2`: `{193: 10}`
- `Q3`: `{193: 3, 206: 7}`
- `S1`: `{190: 9, 191: 1}`
- `S2`: `{178: 4, 206: 5, 179: 1}`
- `S3`: `{198: 8, 178: 2}`
- `S4`: `{206: 8, 203: 2}`

## Decision

Fold-internal target selection can test whether sequence descendants have stable target-wise signal, but it still selects among pre-existing non-nested OOF artifacts and uses unlabeled test-geometry risk controls. Treat this as a map for future source generation, not as a submission gate.
