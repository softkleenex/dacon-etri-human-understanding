# Combined new-source gate — post ID341

Created: `2026-06-18 04:56:51 UTC+09:00`

This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push, or DACON submission.

## Best rows

- `best_by_proxy`: sleep scale `0.000`, evening scale `0.025`, wins `4`, local `0.538581`, proxy `0.601490`, max Δ341 `0.000672`, failed proj `-0.000089`
- `best_reviewable_low_movement`: sleep scale `0.025`, evening scale `0.000`, wins `5`, local `0.538547`, proxy `0.601565`, max Δ341 `0.002380`, failed proj `-0.000130`
- `best_below_id178_proxy`: None

## Top 20 proxy rows

| rank | sleep scale | evening scale | wins | all primary | local | proxy | max Δ341 | failed proj | below ID178 | public 0.58 |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|
| 1 | 0.000 | 0.025 | 4 | `False` | 0.538581 | 0.601490 | 0.000672 | -0.000089 | `False` | `False` |
| 2 | 0.000 | 0.050 | 4 | `False` | 0.538579 | 0.601507 | 0.001344 | -0.000093 | `False` | `False` |
| 3 | 0.000 | 0.075 | 4 | `False` | 0.538577 | 0.601523 | 0.002017 | -0.000098 | `False` | `False` |
| 4 | 0.000 | 0.100 | 4 | `False` | 0.538575 | 0.601553 | 0.002689 | -0.000102 | `False` | `False` |
| 5 | 0.025 | 0.000 | 5 | `True` | 0.538547 | 0.601565 | 0.002380 | -0.000130 | `False` | `False` |
| 6 | 0.025 | 0.025 | 5 | `True` | 0.538545 | 0.601569 | 0.002380 | -0.000134 | `False` | `False` |
| 7 | 0.025 | 0.050 | 5 | `True` | 0.538543 | 0.601574 | 0.002380 | -0.000139 | `False` | `False` |
| 8 | 0.025 | 0.075 | 5 | `True` | 0.538541 | 0.601598 | 0.002936 | -0.000143 | `False` | `False` |
| 9 | 0.000 | 0.150 | 4 | `False` | 0.538571 | 0.601613 | 0.004033 | -0.000110 | `False` | `False` |
| 10 | 0.025 | 0.100 | 5 | `True` | 0.538539 | 0.601627 | 0.003608 | -0.000147 | `False` | `False` |
| 11 | 0.000 | 0.200 | 4 | `False` | 0.538567 | 0.601673 | 0.005378 | -0.000119 | `False` | `False` |
| 12 | 0.025 | 0.150 | 5 | `True` | 0.538535 | 0.601685 | 0.004952 | -0.000156 | `False` | `False` |
| 13 | 0.050 | 0.000 | 5 | `True` | 0.538511 | 0.601697 | 0.004759 | -0.000176 | `False` | `False` |
| 14 | 0.050 | 0.025 | 5 | `True` | 0.538509 | 0.601700 | 0.004759 | -0.000180 | `False` | `False` |
| 15 | 0.050 | 0.050 | 5 | `True` | 0.538507 | 0.601704 | 0.004759 | -0.000184 | `False` | `False` |
| 16 | 0.050 | 0.075 | 5 | `True` | 0.538505 | 0.601709 | 0.004759 | -0.000188 | `False` | `False` |
| 17 | 0.050 | 0.100 | 5 | `True` | 0.538503 | 0.601714 | 0.004759 | -0.000193 | `False` | `False` |
| 18 | 0.025 | 0.200 | 5 | `True` | 0.538531 | 0.601745 | 0.006297 | -0.000164 | `False` | `False` |
| 19 | 0.050 | 0.150 | 5 | `True` | 0.538499 | 0.601763 | 0.005871 | -0.000201 | `False` | `False` |
| 20 | 0.000 | 0.300 | 4 | `False` | 0.538560 | 0.601792 | 0.008067 | -0.000136 | `False` | `False` |

## Decision

- Reviewable low-movement candidate exists: `True`
- Proxy-below-ID178 candidate exists: `False`
- Public `0.58` plausible: `False`
- Best proxy: `0.601490`
- Create submission artifact: `False`
- Submit now: `False`
- Next action: If no combined row falls below ID178, keep both as source leads and move to routine-stability/place entropy. If a row falls below ID178 but not Public 0.58, require another independent source before artifact promotion.

