# Sequence source-generation plan — post ID335

Created: `2026-06-17 23:25:21 UTC+09:00`

This is report-only. It creates no CSV, notebook, base artifact, submission-log row, Kaggle push, or DACON submission.

## Target consensus from local-only fold-internal selector

| target | winner | support | family | priority | ranked counts |
|---|---:|---:|---|---|---|
| Q1 | 200 | 86.67% | `sax_paa_sequence_pattern` | `high` | `[(200, 13), (178, 1), (193, 1)]` |
| Q2 | 193 | 100.00% | `sequence_summary_embedding` | `high` | `[(193, 15)]` |
| Q3 | 206 | 80.00% | `wavelet_spectral_sequence` | `high` | `[(206, 12), (193, 3)]` |
| S1 | 190 | 93.33% | `random_conv_sequence_refresh` | `high` | `[(190, 14), (191, 1)]` |
| S2 | 206 | 53.33% | `wavelet_spectral_sequence` | `medium` | `[(206, 8), (178, 5), (179, 2)]` |
| S3 | 198 | 80.00% | `smartphone_light_compact` | `high` | `[(198, 12), (178, 3)]` |
| S4 | 206 | 73.33% | `wavelet_spectral_sequence` | `high` | `[(206, 11), (203, 3), (178, 1)]` |

## Report-only command plan

Do not use `--promote` in this phase. Pull/review reports before creating any matched artifact.

### S1_random_conv_refresh

- Targets: `['S1']`
- Priority: `high`
- Note: Run on Kaggle if raw sequence expansion is heavy; report-only, no promotion.

Commands:

```bash
.venv/bin/python scripts/diagnose_public_random_conv_sequence.py --base-id 173 --n-kernels 512 --seed 45501 --output reports/id173_random_conv_sequence_post_id335_s1_nk512_seed45501_20260617.json
```

```bash
.venv/bin/python scripts/diagnose_public_random_conv_sequence.py --base-id 173 --n-kernels 512 --seed 45502 --output reports/id173_random_conv_sequence_post_id335_s1_nk512_seed45502_20260617.json
```

```bash
.venv/bin/python scripts/diagnose_public_random_conv_sequence.py --base-id 173 --n-kernels 768 --seed 45601 --output reports/id173_random_conv_sequence_post_id335_s1_nk768_seed45601_20260617.json
```

```bash
.venv/bin/python scripts/diagnose_public_random_conv_sequence.py --base-id 173 --n-kernels 768 --seed 45602 --output reports/id173_random_conv_sequence_post_id335_s1_nk768_seed45602_20260617.json
```

```bash
.venv/bin/python scripts/diagnose_public_random_conv_sequence.py --base-id 173 --n-kernels 1024 --seed 45701 --output reports/id173_random_conv_sequence_post_id335_s1_nk1024_seed45701_20260617.json
```

```bash
.venv/bin/python scripts/diagnose_public_random_conv_sequence.py --base-id 173 --n-kernels 1024 --seed 45702 --output reports/id173_random_conv_sequence_post_id335_s1_nk1024_seed45702_20260617.json
```

### Q2_sequence_summary_embedding

- Targets: `['Q2', 'Q3']`
- Priority: `high`
- Note: Q2 is the strongest consensus target for ID193-like source generation.

Commands:

```bash
.venv/bin/python scripts/diagnose_public_sequence_summary_embedding.py --base-id 191 --output reports/id191_sequence_summary_embedding_post_id335_20260617.json
```

### S3_smartphone_light_compact

- Targets: `['S3']`
- Priority: `high`
- Note: S3 selects ID198 in 12/15 local-only folds; diagnostic-only unless nested source improves.

Commands:

```bash
.venv/bin/python scripts/run_public_smartphone_light_compact_overlay_experiment.py --base-id 193 --output reports/id193_smartphone_light_compact_post_id335_20260617.json
```

### Q1_sax_paa_sequence_pattern

- Targets: `['Q1']`
- Priority: `high`
- Note: Q1 consensus is ID200, but support is less clean in subject-hole than LOSO.

Commands:

```bash
.venv/bin/python scripts/run_public_sax_paa_sequence_overlay_experiment.py --base-id 198 --output reports/id198_sax_paa_sequence_post_id335_20260617.json
```

### Q3_S2_S4_wavelet_spectral

- Targets: `['Q3', 'S2', 'S4']`
- Priority: `medium`
- Note: Wavelet/spectral dominates Q3/S4 and appears in S2, but movement risk kept caps at ID178.

Commands:

```bash
.venv/bin/python scripts/run_public_wavelet_spectral_sequence_overlay_experiment.py --base-id 203 --output reports/id203_wavelet_spectral_sequence_post_id335_20260617.json
```

### S4_recurrence_gaf_alternative

- Targets: `['S4']`
- Priority: `low`
- Note: Only an S4 alternative behind ID206; run after wavelet if needed.

Commands:

```bash
.venv/bin/python scripts/run_public_recurrence_gaf_sequence_overlay_experiment.py --base-id 200 --output reports/id200_recurrence_gaf_sequence_post_id335_20260617.json
```

## Decision

The sequence branch has target-wise local signal, but risk-controlled selectors revert to ID178. The next useful work is report-only source generation for the consensus target families, preferably on Kaggle for heavier raw-sequence jobs, followed by strict review before any artifact promotion.
