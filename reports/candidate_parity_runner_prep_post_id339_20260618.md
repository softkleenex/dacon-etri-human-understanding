# Candidate parity runner prep — post ID339

Created: `2026-06-18 02:46:57 UTC+09:00`

## Prepared runner

- Notebook: `notebooks/kaggle_candidate_parity_runner/Candidate_Parity_Post_ID339_Runner.ipynb`
- Kernel metadata: `notebooks/kaggle_candidate_parity_runner/kernel-metadata.json`
- Kaggle slug: `softkleenex/dacon-etri-candidate-parity-post-id339`
- GPU/TPU: disabled
- Output type: JSON/MD parity diagnostics only

## Local parity smoke test

| id | status | Δ local | OOF max Δ | test max Δ |
|---:|---|---:|---:|---:|
| 337 | `review_pass` | +3.966e-07 | 1.110e-16 | 1.110e-16 |
| 339 | `review_pass` | -2.810e-07 | 1.110e-16 | 1.110e-16 |

## Preflight

- Command: `.venv/bin/python scripts/pre_flight_check.py --notebook_dir notebooks/kaggle_candidate_parity_runner --research`
- Passed: `True`

## Decision

The local parity diagnostic passes for ID337 and ID339. The next validation step is environment-independent Kaggle CPU parity, not another broad model sweep.

No Kaggle push and no DACON submission were made.
