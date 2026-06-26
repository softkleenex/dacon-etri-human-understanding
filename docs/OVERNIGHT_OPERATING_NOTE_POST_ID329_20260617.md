# Overnight operating note after ID329

Created: `2026-06-17 01:54:22` KST

You do not need to do anything before sleeping.

## Allowed while user is away

- Run local or Kaggle CLI report-only diagnostics.
- Package/push Kaggle datasets or private kernels for non-submission audits.
- Pull Kaggle outputs and write reports.
- Run docs/submissions/artifact audits and tests.

## Forbidden without explicit approval

- DACON submission.
- Deleting, archiving, or mass-moving files.
- Promoting a candidate into the submission contract as next public probe.
- Claiming Public 0.58 without a decision note and anchor-risk evidence.

## Public 0.58 critical-expectation gate

- **minimum_condition**: A candidate must not be same-family with ID329 constrained-combo unless validation explains ID329 public non-improvement.
- **validation_condition**: Strict fold-safe CV must improve a trusted non-public-guided baseline, not only full-OOF recombination.
- **anchor_condition**: Candidate must be close to ID178/ID329 public anchor geometry or have a documented orthogonal rationale; avoid ID287 failed vector.
- **risk_proxy_condition**: Conservative public-anchor rerank/proxy should be materially below the ID178/ID329 plateau, target <= 0.590 before considering Public 0.58 plausible.
- **artifact_condition**: Unique CSV/notebook/log/report/base predictions must match before review.

## Current status

- ID178 Public 0.6014113009 remains best confirmed recent anchor.
- ID329 Public 0.6017908577 showed guarded local improvement did not convert to Public improvement.
- Existing rerank found no immediate low-risk Public <0.60 candidate.
- Demographic smoke and Kaggle full audits found subject-hole gains but LOSO stress weakness; use as target-specific hypothesis only.

## Next non-destructive tasks

- Build a target-specific demographic gate only if it is evaluated fold-safely and compared against ID178/ID329/ID287 anchor geometry.
- Build a candidate decision template requiring Public 0.58 plausibility evidence before any future submission.
