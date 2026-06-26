# Mac M4 ↔ Kaggle compute parity policy

Created: `2026-06-18 02:24:03 UTC+09:00`

## Local environment observed

- Machine: Apple M4 Pro / arm64
- macOS: `26.5.1` build `25F80`
- Project Python: `3.14.3`
- TensorFlow import: unavailable in the current venv
- PyTorch: `2.11.0`, MPS available
- Kaggle credentials: `~/.kaggle/kaggle.json` exists
- Kaggle CLI: bare `kaggle` is not on PATH, but project CLI `.venv/bin/kaggle` works (`2.1.1`)

## Read

The recent ID339 path is not directly explained by Mac TensorFlow/Metal behavior: it uses sequence-summary and sklearn-style diagnostics, not TensorFlow GPU training. The public/local mismatch is still primarily a validation/selection-transfer problem.

However, Mac-local DL artifacts are now treated as environment-risky. Any TensorFlow, Keras, PyTorch-MPS, or GPU-dependent embedding artifact must pass a Kaggle/CPU parity check before it can become a submission candidate.

## Parity gate

Minimum checks before promoting a Mac-local DL artifact:

1. Same seed and same data slice locally and on Kaggle or CPU reference.
2. Record package versions and visible devices.
3. Compare OOF/test prediction correlation, max absolute delta, mean absolute delta, and per-target log-loss delta.
4. Deterministic code should have correlation `>= 0.999` or an explainably tiny max delta.
5. Stochastic training should compare repeated-seed distributions, not a single lucky run.

## Operational decision

Do not use Mac-local DL outputs as promoted submission candidates unless their metadata/decision note links a parity report. Use `.venv/bin/kaggle` or the existing Makefile targets for offloaded runs; optional PATH/CLI-version cleanup can be handled separately. No package installation, Kaggle push, or DACON submission was made here.
