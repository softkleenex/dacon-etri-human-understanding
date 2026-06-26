# Kaggle upstream rebuild v2 failure

Created: `2026-06-23 KST`

This is a no-submit failure record. It creates no submission CSV, no base
prediction artifact, no notebook-history artifact, no submission-log row, and
no DACON submission.

## Status

- Kernel: `softkleenex/dacon-etri-upstream-rebuild-post-id341`
- Status: `KernelWorkerStatus.ERROR`
- Dataset sources:
  - `softkleenex/dacon-etri-human-base`
  - `softkleenex/dacon-etri-upstream-patch-post-id341`

## Read

The first Kaggle failure was a stale-input packaging problem. That part is now
fixed: runner v2 passed the packaged-project required-path check with
`Missing required paths: []` and generated many remote no-submit reports.

The current failure is different. The runner stopped at the final upstream
blueprint assertion because
`remote_post_id341_upstream_rebuild_blueprint_20260618.json` reported
`decision.blueprint_ready=False`.

This means the current run is not something to wait on. It is a
blueprint/report-contract debug state, not a Kaggle CLI/auth/slug failure.

## Evidence

- Log: `build/kaggle_outputs/upstream_rebuild_post_id341_v2_logs/dacon-etri-upstream-rebuild-post-id341.log`
- Remote package check: `Missing required paths: []`
- Failing cell: upstream rebuild blueprint assertion
- Failing key: `decision.blueprint_ready`
- Failing value: `False`
- Remote missing required files: `dataset-metadata.json`
- Pulled remote blueprint:
  `build/kaggle_outputs/upstream_rebuild_post_id341_v2_direct/project/reports/remote_post_id341_upstream_rebuild_blueprint_20260618.json`

Root cause: Kaggle consumed `dataset-metadata.json` as dataset metadata and did
not expose it as a normal input file, so the staged working project lacked that
file even though base+patch overlay and remote report execution worked.

Prepared fix: the no-submit runner now synthesizes a minimal runtime
`dataset-metadata.json` after overlay when Kaggle input files omit it. The
source self-test cleanup was also narrowed to `build/kaggle_dataset`, so
`build/kaggle_outputs` logs and direct pulls are preserved.

## Decision

- Wait for current Kaggle run: **no**
- Create ID342/submission artifact: **no**
- Edit `submission_log.csv`: **no**
- Submit to DACON: **no**
- Next action: push the refreshed patch dataset and runner only under explicit
  no-submit Kaggle approval, then confirm the remote blueprint gate passes
  before any further report phase.
