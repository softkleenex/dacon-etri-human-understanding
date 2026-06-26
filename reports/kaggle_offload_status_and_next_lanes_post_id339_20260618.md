# Kaggle offload status and next lanes — post ID339

Created: `2026-06-18 02:28:32 UTC+09:00`

## CLI status

- Project CLI: `.venv/bin/kaggle` works
- Version: `2.1.1` (Kaggle warns latest is `2.2.2`; no upgrade performed)
- Bare `kaggle` is not on the current shell PATH, but Makefile uses `.venv/bin/kaggle`.

## Pulled remote outputs

- Sequence-source runner: COMPLETE, `6` jobs pulled and inspected from `build/kaggle_outputs/sequence_source_post_id335`
- Sequence-anchor probe runner: COMPLETE, `4` jobs pulled and inspected from `build/kaggle_outputs/sequence_anchor_probe_post_id335`

The transient `build/` pull directory was removed after manifest inspection to keep `make structure-audit` clean; outputs can be re-pulled with the Make targets.

## Submission decision linkage

- Decision matrix: `reports/post_id339_submission_decision_matrix_20260618.md`
- If the user explicitly forces one controlled public probe, matrix recommends **ID337**, not ID339.
- No DACON submission was made.

## Next Kaggle lanes

| priority | lane | compute | purpose |
|---:|---|---|---|
| 1 | `candidate_parity_post_id339` | Kaggle CPU, no GPU/TPU required | Reproduce ID337/ID339 candidate construction and metrics remotely before treating either as a trusted public probe. |
| 2 | `dl_environment_parity_before_any_mac_dl_promotion` | Kaggle CPU/GPU depending on model, but report-only | If any TensorFlow/Keras/PyTorch-MPS local artifact is considered, compare same seed/data slice against Kaggle/CPU reference first. |
| 3 | `new_model_heavy_training` | Kaggle only after parity/selection gate | Only after validation/selection gates stop contradicting public behavior. |

## Decision

Existing outputs were pulled and inspected successfully, then transient build outputs were removed to keep the repository structure clean. The next useful Kaggle job should be a deliberately scoped parity runner, not another broad model sweep, and should be prepared as report-only before any push.

