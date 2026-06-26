# A1 q240-free nested source replay — post ID338

Created: `2026-06-18 01:53:58 UTC+09:00`

This is report-only. No CSV, notebook, base artifact, submission-log row, or DACON submission was created.

## Results

| split | replay overall | replay robust | selector-only qfree | delta vs selector-only | worst target | worst subject | changed |
|---|---:|---:|---:|---:|---|---|---:|
| `subject_hole_source_replay` | 0.491331 | 0.511750 | 0.414272 | +0.077058 | Q3 `0.540587` | id10 `0.588129` | 0.914 |
| `loso_source_replay` | 0.484186 | 0.507550 | 0.414440 | +0.069746 | Q3 `0.533411` | id10 `0.593861` | 0.914 |

## Base reference

ID235 base OOF is `0.482255` / robust `0.502595`.

## Predominant choices

### Pair source choices

#### `subject_hole_source_replay`
- `Q1`: `logit_pair:221:227:w0.660` x2, `logit_pair:221:227:w0.710` x1
- `Q2`: `logit_pair:221:227:w0.840` x2, `logit_pair:221:227:w0.896` x1
- `Q3`: `logit_pair:221:227:w0.830` x1, `logit_pair:221:227:w0.788` x1
- `S1`: `logit_pair:221:227:w0.720` x2, `logit_pair:221:227:w0.810` x2
- `S2`: `logit_pair:221:227:w0.810` x1, `logit_pair:221:227:w0.940` x1
- `S3`: `single:226:226:w0.000` x3, `logit_pair:226:227:w0.500` x1
- `S4`: `logit_pair:227:228:w0.100` x1, `logit_pair:228:237:w0.896` x1

Bridge choices:
- `Q1`: `id241_q160_s4iso_replay:logit:a0.450` x5
- `Q2`: `id241_q160_s4iso_replay:logit:a0.463` x3, `id241_q160_s4iso_replay:logit:a0.525` x1
- `Q3`: `id241_q160_s4iso_replay:logit:a0.450` x2, `id241_q160_s4iso_replay:prob:a0.513` x1
- `S1`: `id241_q160_s4iso_replay:logit:a0.450` x3, `id241_q160_s4iso_replay:logit:a0.475` x1
- `S2`: `id241_q160_s4iso_replay:logit:a0.450` x2, `id241_q160_s4iso_replay:logit:a0.425` x1
- `S3`: `base_235:base:a0.000` x3, `id238_replay:logit:a0.025` x1
- `S4`: `id239_iso_replay:prob:a0.550` x5

#### `loso_source_replay`
- `Q1`: `logit_pair:221:227:w0.700` x2, `logit_pair:221:227:w0.620` x1
- `Q2`: `logit_pair:221:227:w0.810` x2, `logit_pair:221:227:w0.880` x1
- `Q3`: `logit_pair:221:227:w0.760` x2, `logit_pair:221:227:w0.773` x2
- `S1`: `logit_pair:221:227:w0.730` x3, `logit_pair:221:227:w0.773` x1
- `S2`: `logit_pair:221:227:w0.840` x2, `logit_pair:221:227:w0.830` x2
- `S3`: `single:226:226:w0.000` x8, `prob_pair:226:227:w0.460` x1
- `S4`: `logit_pair:227:228:w0.087` x5, `logit_pair:228:237:w0.896` x2

Bridge choices:
- `Q1`: `id241_q160_s4iso_replay:logit:a0.425` x4, `id241_q160_s4iso_replay:logit:a0.450` x4
- `Q2`: `id241_q160_s4iso_replay:logit:a0.463` x3, `id241_q160_s4iso_replay:prob:a0.463` x2
- `Q3`: `id241_q160_s4iso_replay:prob:a0.475` x4, `id241_q160_s4iso_replay:logit:a0.450` x2
- `S1`: `id241_q160_s4iso_replay:logit:a0.400` x3, `id241_q160_s4iso_replay:logit:a0.475` x3
- `S2`: `id241_q160_s4iso_replay:logit:a0.475` x5, `id241_q160_s4iso_replay:logit:a0.450` x3
- `S3`: `base_235:base:a0.000` x6, `id238_replay:prob:a0.075` x2
- `S4`: `id239_iso_replay:prob:a0.550` x10

## Decision

The q240-free A1 stack does not preserve enough signal after outer-fold source replay. Deprioritize this lane before any Kaggle-heavy model-level rebuild.

- Source replay survives: `False`
- Public 0.58 plausible: `False`
- Create submission artifact: `False`
