# Post-ID278 public-gap-aware shrinkage scan

This scan blends stable anchors and frontier cards while penalizing source risk and test movement versus ID178. The public-gap proxy is diagnostic, not a leaderboard claim.

| cap | raw local | raw max delta | proxy local | proxy max delta | proxy score | proxy source risk | report |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.16 | 0.439710 | 0.159949 | 0.440550 | 0.159994 | 0.516057 | 2.147 | `reports/post278_public_gap_aware_shrinkage_cap160_diagnostic_20260616.json` |
| 0.18 | 0.430318 | 0.179992 | 0.430926 | 0.179994 | 0.513269 | 2.714 | `reports/post278_public_gap_aware_shrinkage_cap180_diagnostic_20260616.json` |
| 0.20 | 0.421450 | 0.199977 | 0.422025 | 0.199948 | 0.511011 | 3.245 | `reports/post278_public_gap_aware_shrinkage_cap200_diagnostic_20260616.json` |
| 0.22 | 0.413697 | 0.219940 | 0.413979 | 0.219676 | 0.509393 | 3.741 | `reports/post278_public_gap_aware_shrinkage_cap220_diagnostic_20260616.json` |
| 0.24 | 0.406187 | 0.239943 | 0.406514 | 0.239676 | 0.508393 | 4.234 | `reports/post278_public_gap_aware_shrinkage_cap240_diagnostic_20260616.json` |
| 0.26 | 0.399185 | 0.259962 | 0.399392 | 0.259827 | 0.507723 | 4.718 | `reports/post278_public_gap_aware_shrinkage_cap260_diagnostic_20260616.json` |
| 0.28 | 0.392918 | 0.279994 | 0.393210 | 0.279793 | 0.507719 | 5.155 | `reports/post278_public_gap_aware_shrinkage_cap280_diagnostic_20260616.json` |

Practical read:
- Raw-local winners still chase frontier behavior.
- Proxy winners intentionally sacrifice some local score to reduce movement/source risk.
- Promote only candidates that add a new point on the risk/local frontier.
- No DACON submission was made.
