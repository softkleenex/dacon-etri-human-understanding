# Phase-3 ambience context closure — post ID341

Created: `2026-06-18 08:35:41 UTC+09:00`

This is report-only. It reads existing ambience diagnostics only; no model is trained and no predictions or submission artifacts are created.

## Aggregate read

- Ambience reports read: `7`
- Full-OOF positive rows: `2`
- Nested positive rows: `0`
- S4 full-OOF trap rows: `2`

Best full-OOF ambience row:

- `reports/strict_s4_ambience_context_diagnostic_id118.json` `S4` base ID `118`
- Full-OOF Δ vs base: `-0.002880`
- Nested Δ vs base: `+0.003467`

## Rows

| report | target | base | full-OOF Δ | nested Δ | nested improves | best full candidate |
|---|---|---:|---:|---:|---|---|
| `reports/strict_q2_ambience_context_diagnostic_id118.json` | `Q2` | 118 | +0.000000 | +0.000000 | `False` | `id118_base` |
| `reports/strict_q2_ambience_context_diagnostic_id167.json` | `Q2` | 167 | +0.000000 | +0.000000 | `False` | `id167_base` |
| `reports/strict_q3_ambience_context_diagnostic_id118.json` | `Q3` | 118 | +0.000000 | +0.000000 | `False` | `id118_base` |
| `reports/strict_q3_ambience_context_diagnostic_id167.json` | `Q3` | 167 | +0.000000 | +0.000000 | `False` | `id167_base` |
| `reports/strict_s4_ambience_context_confirm_diagnostic_id118.json` | `S4` | 118 | -0.001997 | +0.000642 | `False` | `blend_base0.90_amb_focused_abs_rel__lgbm_l15_lr015_n220_l50_confirm_0.10` |
| `reports/strict_s4_ambience_context_diagnostic_id118.json` | `S4` | 118 | -0.002880 | +0.003467 | `False` | `blend_base0.75_amb_focused_abs_rel__lgbm_l15_lr015_n220_l50_0.25` |
| `reports/strict_s4_ambience_context_diagnostic_id167.json` | `S4` | 167 | +0.000000 | +0.003239 | `False` | `id167_base` |

## Decision

- Ambience reopen now: `False`
- Historical full-OOF signal exists: `True`
- Nested positive signal exists: `False`
- S4 full-OOF is transfer trap: `True`
- Public 0.58 reviewable: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: Do not create ID342 from semantic ambience context. Reopen mAmbience only if a future upstream model-level rebuild can nest source generation before selector choice; do not replay broad/full-OOF S4 ambience overlays.

