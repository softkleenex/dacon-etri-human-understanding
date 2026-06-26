# S3 Kaggle random-conv runner plan

Prepared `notebooks/kaggle_strict_runner/Strict_Remote_Experiment_Runner.ipynb` as an S3 hard-wall raw-sequence sweep runner.

Anchor: ID247. Jobs: `6` sequential random-conv diagnostics with 512/768/1024 kernels.

Local smoke passed:
- S3 base loss: `0.465115`
- S3 smoke best: `base` / `0.465115`
- channels/features: `68` / `64`

Preflight passed for the Kaggle strict runner. The runner creates reports only; any useful S3 report must be promoted locally under the normal artifact contract.

Next commands:
```bash
make kaggle-push-code
make kaggle-push-strict-runner
make kaggle-pull-strict-output
```
