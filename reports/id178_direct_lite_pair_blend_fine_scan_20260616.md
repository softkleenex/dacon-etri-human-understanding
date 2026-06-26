# ID178 direct-lite pair blend fine scan

Direct stress IDs `252/257/259/260/261/262` were excluded. Weights use `0.01` resolution. Lower local is better; caps are max test-delta vs confirmed-public ID178.

| cap | local CV | max delta vs ID178 | diagnostic |
|---:|---:|---:|---|
| 0.10 | 0.473243 | 0.099896 | `reports/id178_direct_lite_pair_blend_fine_cap01_diagnostic_20260616.json` |
| 0.11 | 0.468144 | 0.109950 | `reports/id178_direct_lite_pair_blend_fine_cap011_diagnostic_20260616.json` |
| 0.12 | 0.464758 | 0.119995 | `reports/id178_direct_lite_pair_blend_fine_cap012_diagnostic_20260616.json` |
| 0.13 | 0.461543 | 0.129969 | `reports/id178_direct_lite_pair_blend_fine_cap013_diagnostic_20260616.json` |
| 0.14 | 0.458478 | 0.139926 | `reports/id178_direct_lite_pair_blend_fine_cap014_diagnostic_20260616.json` |
| 0.15 | 0.457894 | 0.149809 | `reports/id178_direct_lite_pair_blend_fine_cap015_diagnostic_20260616.json` |
| 0.16 | 0.457647 | 0.159452 | `reports/id178_direct_lite_pair_blend_fine_cap016_diagnostic_20260616.json` |

Practical read:
- Fine weights improve the direct-lite cap ladder without introducing direct stress IDs.
- cap0.12 is comparable to ID263; cap0.14/0.15 are sharper second-probe candidates.
- This remains full-OOF target-wise selection, not public proof.
- No DACON submission was made.
