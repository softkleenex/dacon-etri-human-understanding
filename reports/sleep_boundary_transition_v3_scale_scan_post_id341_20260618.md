# Sleep-boundary transition v3 scale scan — post ID341

Created: `2026-06-18 04:45:23 UTC+09:00`

This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, or DACON submission.

## Selected v2 targets

| target | base candidate | base weight |
|---|---|---:|
| `Q1` | `Q1_id178_hgb_l5_l2_2_w0.05` | 0.0500 |
| `S1` | `S1_id178_lr_c0_2_w0.03` | 0.0300 |
| `S2` | `S2_id178_lr_c0_05_w0.1` | 0.1000 |
| `S4` | `S4_id178_lr_c0_05_w0.1` | 0.1000 |

## Scale scan

| scale | primary wins | all primary | local | proxy | Δ341 mean/max | failed proj | reviewable |
|---:|---:|---|---:|---:|---:|---:|---|
| 0.05 | 5 | `True` | 0.538511 | 0.601697 | 0.000371/0.004759 | -0.000176 | `True` |
| 0.10 | 5 | `True` | 0.538442 | 0.601961 | 0.000743/0.009518 | -0.000266 | `True` |
| 0.15 | 5 | `True` | 0.538374 | 0.602225 | 0.001114/0.014277 | -0.000357 | `True` |
| 0.20 | 5 | `True` | 0.538310 | 0.602488 | 0.001486/0.019036 | -0.000448 | `True` |
| 0.30 | 5 | `True` | 0.538187 | 0.603016 | 0.002229/0.028554 | -0.000630 | `True` |
| 0.40 | 5 | `True` | 0.538074 | 0.603543 | 0.002971/0.038073 | -0.000811 | `True` |
| 0.50 | 5 | `True` | 0.537970 | 0.604071 | 0.003714/0.047591 | -0.000993 | `False` |
| 0.65 | 5 | `True` | 0.537831 | 0.604862 | 0.004829/0.061868 | -0.001265 | `False` |
| 0.80 | 5 | `True` | 0.537712 | 0.605653 | 0.005943/0.076145 | -0.001537 | `False` |
| 1.00 | 5 | `True` | 0.537585 | 0.606708 | 0.007429/0.095181 | -0.001901 | `False` |

## Decision

- Reviewable low-movement candidate: `True`
- Best reviewable scale: `0.05`
- Best reviewable proxy: `0.601697`
- Best reviewable max Δ vs ID341: `0.004759`
- Best proxy scale: `0.05`
- Best proxy: `0.601697`
- Public `0.58` plausible: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Next action: If a reviewable row exists, run a final no-artifact prepromotion gate with exact target weights and stricter geometry. Do not create ID342 or submit unless that prepromotion gate remains clean.

