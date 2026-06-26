# External ETRI prior smoke — post ID341

Created: `2026-06-18 06:54:14 UTC+09:00`

This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.

## Purpose

Estimate whether older ETRI 2019/2020 summary CSVs are useful as weak external priors around ID341 without concatenating external rows into current training.

## Coverage

- External long target rows: `6135`
- External users: `42`
- Targets available: `['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']`
- Candidate overlays scanned: `50`

## Target prevalence comparison

| target | external n | external mean | current train mean |
|---|---:|---:|---:|
| `Q1` | 937 | 0.593383 | 0.495556 |
| `Q2` | 615 | 0.560976 | 0.562222 |
| `Q3` | 899 | 0.462736 | 0.600000 |
| `S1` | 1023 | 0.144673 | 0.682222 |
| `S2` | 1023 | 0.696970 | 0.651111 |
| `S3` | 615 | 0.609756 | 0.662222 |
| `S4` | 1023 | 0.311828 | 0.560000 |

## Best scanned rows

| name | targets | weight | primary wins | local | proxy | Δ341 mean/max | failed proj | review | promote |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `Q1` | `Q1` | 0.005 | 3 | 0.538599 | 0.601571 | 0.000151/0.002872 | -0.000499 | `False` | `False` |
| `S4` | `S4` | 0.005 | 3 | 0.538599 | 0.601605 | 0.000236/0.003149 | -0.000665 | `False` | `False` |
| `Q1` | `Q1` | 0.010 | 3 | 0.538618 | 0.601708 | 0.000301/0.005743 | -0.000914 | `False` | `False` |
| `S4` | `S4` | 0.010 | 3 | 0.538618 | 0.601776 | 0.000471/0.006299 | -0.001245 | `False` | `False` |
| `Q1` | `Q1` | 0.020 | 3 | 0.538663 | 0.601982 | 0.000603/0.011486 | -0.001743 | `False` | `False` |
| `S4` | `S4` | 0.020 | 3 | 0.538661 | 0.602119 | 0.000942/0.012597 | -0.002406 | `False` | `False` |
| `Q1` | `Q1` | 0.050 | 3 | 0.538846 | 0.602805 | 0.001507/0.028715 | -0.004231 | `False` | `False` |
| `S4` | `S4` | 0.050 | 3 | 0.538850 | 0.603148 | 0.002356/0.031493 | -0.005888 | `False` | `False` |
| `S2` | `S2` | 0.005 | 2 | 0.538607 | 0.601564 | 0.000125/0.002894 | -0.000426 | `False` | `False` |
| `S2` | `S2` | 0.010 | 2 | 0.538632 | 0.601694 | 0.000250/0.005788 | -0.000767 | `False` | `False` |
| `S2` | `S2` | 0.020 | 2 | 0.538684 | 0.601954 | 0.000500/0.011577 | -0.001449 | `False` | `False` |
| `S2` | `S2` | 0.050 | 2 | 0.538862 | 0.602735 | 0.001251/0.028942 | -0.003495 | `False` | `False` |
| `S4` | `S4` | 0.100 | 2 | 0.539339 | 0.604862 | 0.004711/0.062986 | -0.011691 | `False` | `False` |
| `Q2` | `Q2` | 0.005 | 1 | 0.538617 | 0.601570 | 0.000124/0.002624 | -0.000337 | `False` | `False` |
| `Q_all` | `Q1,Q2,Q3` | 0.005 | 1 | 0.538664 | 0.601675 | 0.000459/0.003003 | -0.001290 | `False` | `False` |
| `S1` | `S1` | 0.005 | 1 | 0.538592 | 0.601686 | 0.000383/0.004285 | -0.000692 | `False` | `False` |
| `Q2` | `Q2` | 0.010 | 1 | 0.538650 | 0.601687 | 0.000248/0.005247 | -0.000590 | `False` | `False` |
| `S_sleep_external` | `S1,S2,S3,S4` | 0.005 | 1 | 0.538653 | 0.601839 | 0.000894/0.004285 | -0.001630 | `False` | `False` |
| `Q_all` | `Q1,Q2,Q3` | 0.010 | 1 | 0.538750 | 0.601897 | 0.000919/0.006006 | -0.002495 | `False` | `False` |
| `Q2` | `Q2` | 0.020 | 1 | 0.538721 | 0.601925 | 0.000496/0.010494 | -0.001094 | `False` | `False` |
| `S1` | `S1` | 0.010 | 1 | 0.538618 | 0.601938 | 0.000767/0.008570 | -0.001300 | `False` | `False` |
| `all_external` | `Q1,Q2,Q3,S1,S2,S3,S4` | 0.005 | 1 | 0.538734 | 0.601966 | 0.001353/0.004285 | -0.002836 | `False` | `False` |
| `S_sleep_external` | `S1,S2,S3,S4` | 0.010 | 1 | 0.538746 | 0.602244 | 0.001788/0.008570 | -0.003176 | `False` | `False` |
| `Q_all` | `Q1,Q2,Q3` | 0.020 | 1 | 0.538935 | 0.602356 | 0.001837/0.012012 | -0.004905 | `False` | `False` |
| `S1` | `S1` | 0.020 | 1 | 0.538714 | 0.602442 | 0.001534/0.017140 | -0.002514 | `False` | `False` |
| `all_external` | `Q1,Q2,Q3,S1,S2,S3,S4` | 0.010 | 1 | 0.538912 | 0.602508 | 0.002707/0.008570 | -0.005586 | `False` | `False` |
| `Q2` | `Q2` | 0.050 | 1 | 0.538951 | 0.602643 | 0.001240/0.026236 | -0.002609 | `False` | `False` |
| `S_sleep_external` | `S1,S2,S3,S4` | 0.020 | 1 | 0.538990 | 0.603055 | 0.003576/0.017140 | -0.006267 | `False` | `False` |
| `all_external` | `Q1,Q2,Q3,S1,S2,S3,S4` | 0.020 | 1 | 0.539342 | 0.603593 | 0.005414/0.017140 | -0.011088 | `False` | `False` |
| `Q_all` | `Q1,Q2,Q3` | 0.050 | 1 | 0.539590 | 0.603758 | 0.004593/0.030030 | -0.012136 | `False` | `False` |

## Reviewable rows

| name | targets | weight | primary wins | local | proxy | Δ341 mean/max | failed proj | review | promote |
|---|---|---:|---:|---:|---:|---:|---:|---|---|

## Decision

- External-prior reviewable low-movement candidate exists: `False`
- External-prior Public 0.58 promotable candidate exists: `False`
- Best proxy: `0.601564`
- Best proxy row: `S2`
- Create submission artifact: `False`
- Submit now: `False`
- Next action: Use external ETRI data only as weak-prior/reference evidence unless a future source passes the combined transfer gate. Do not concatenate older external rows into current training without a separate leak-free proof.

