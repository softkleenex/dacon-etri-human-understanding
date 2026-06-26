# ID178 direct-lite 4-way convex grid scan

Clean source pool `178/253/255/258`; direct stress IDs excluded. Weights use simplex step `0.025`. Lower local is better; caps are max test-delta vs ID178.

| cap | local CV | max delta vs ID178 | diagnostic |
|---:|---:|---:|---|
| 0.10 | 0.473312 | 0.099986 | `reports/id178_direct_lite_convex4_grid_cap01_diagnostic_20260616.json` |
| 0.12 | 0.464806 | 0.119975 | `reports/id178_direct_lite_convex4_grid_cap012_diagnostic_20260616.json` |
| 0.14 | 0.458474 | 0.139905 | `reports/id178_direct_lite_convex4_grid_cap014_diagnostic_20260616.json` |
| 0.15 | 0.457888 | 0.149960 | `reports/id178_direct_lite_convex4_grid_cap015_diagnostic_20260616.json` |
| 0.16 | 0.457647 | 0.159452 | `reports/id178_direct_lite_convex4_grid_cap016_diagnostic_20260616.json` |

Practical read:
- Compare cap0.12 against ID264 and ID260.
- Compare cap0.15/0.16 against ID266 and ID258.
- This remains full-OOF target-wise selection, not public proof.
- No DACON submission was made.
