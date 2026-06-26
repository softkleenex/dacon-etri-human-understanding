# Post-ID271 ultra-gated pair-blend scan

Includes ID247/248/249/252 ultra-frontier ingredients with high source-risk penalties. Lower local is better, but this ladder is much riskier than ID268.

| cap | raw local | raw max delta | raw source risk | p0.005 | p0.010 | p0.020 | p0.040 | p0.080 | report |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.18 | 0.430847 | 0.179934 | 2.545 | 0.433773 | 0.439028 | 0.442767 | 0.445365 | 0.447563 | `reports/post271_ultra_gated_pair_blend_cap180_diagnostic_20260616.json` |
| 0.20 | 0.421660 | 0.199406 | 3.171 | 0.423144 | 0.432315 | 0.440931 | 0.441686 | 0.447563 | `reports/post271_ultra_gated_pair_blend_cap200_diagnostic_20260616.json` |
| 0.24 | 0.406522 | 0.239897 | 3.416 | 0.409898 | 0.417823 | 0.427336 | 0.429861 | 0.435714 | `reports/post271_ultra_gated_pair_blend_cap240_diagnostic_20260616.json` |
| 0.28 | 0.392644 | 0.279773 | 3.755 | 0.397954 | 0.399355 | 0.405585 | 0.416819 | 0.424916 | `reports/post271_ultra_gated_pair_blend_cap280_diagnostic_20260616.json` |
| 0.30 | 0.385287 | 0.299664 | 3.855 | 0.389971 | 0.399004 | 0.403490 | 0.405936 | 0.415101 | `reports/post271_ultra_gated_pair_blend_cap300_diagnostic_20260616.json` |

Practical read:
- Raw variants chase local aggressively using frontier ingredients.
- Penalty >=0.04 collapses back toward safer post-ID266/ID257 family, which is useful as a sanity check.
- No DACON submission was made.
