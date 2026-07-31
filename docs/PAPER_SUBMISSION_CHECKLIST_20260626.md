# Paper Submission Checklist 2026-07-31

Status: updated deadline checklist for the technical draft in
`docs/PAPER_SUBMISSION_DRAFT_20260626.md`. This document creates no prediction,
submission CSV, Kaggle run, submission-log row, or DACON submission.

Working IEEE source draft:
`docs/paper_submission_20260626/paper.tex`.

Lineage/runbook package:
`docs/paper_submission_20260626/id342_lineage_manifest.md` and
`docs/paper_submission_20260626/reproducibility_runbook.md`.

EDAS handoff:
`docs/paper_submission_20260626/edas_submission_handoff.md`.

EDAS copy-paste metadata:
`docs/paper_submission_20260626/edas_metadata.md`.

## Current Deadline Position

- Paper deadline date: `2026-07-31 23:59`.
- Primary source: DACON talkboard notice updated `2026-07-17 09:10`.
- Corroborating source: ICTC workshop list and EDAS link.
- Official ICTC paper-submission page says full and short papers must both be
  at least `2` pages.
- DACON rules for this challenge require the IWETRIAI paper to use the Standard
  IEEE conference template as a `4`-page full paper.
- Algorithm leaderboard close recorded separately as `2026-07-23`.
- Treat the safe operational deadline as before Korean business-day end on
  `2026-07-31`; the official notice states `23:59`.
- Review submission completed: EDAS paper `1571325954`, status
  `Active (has manuscript)`, four-page PDF uploaded.

## Must Finish Before Submission

The initial review submission items below are complete. Retain them as the
camera-ready recheck list if the paper is accepted.

- Confirm the single corresponding-author metadata: `Shangjae Lee`,
  `Kyungpook National University`, `dltkdwo31@knu.ac.kr`.
- Rebuild the verified Standard IEEE conference `4`-page full paper after any
  camera-ready edit. Current source draft:
  `docs/paper_submission_20260626/paper.tex`; verified PDF:
  `build/paper_submission_20260626/compile/paper.pdf`.
- Verify whether the IWETRIAI workshop track is visible in EDAS at submission
  time.
- Use the ICTC official EDAS submission link recorded in
  `docs/paper_submission_20260626/edas_submission_handoff.md`.
- Attach or cite only reproducible local records, not private API keys,
  ignored raw caches, or untracked large artifacts.

## Four-Page Draft Budget

- Page 1: problem, target definitions, data sources, and the conservative
  public-anchor framing.
- Page 2: model lineage, ID342 blend construction, and validation-risk-control
  protocol.
- Page 3: public/local result table, failed-frontier analysis, and negative
  evidence from external and raw-source gates.
- Page 4: limitations, reproducibility pointers, security/data provenance, and
  conclusion.

## Safe Technical Claims

- Current confirmed public anchor: ID342, Public `0.5960832519`, local
  `0.534000`, DACON submission `1467937`.
- Local `0.534000` is a non-nested model-selection score and must not be
  described as an unbiased generalization estimate.
- The 2026-06-26 emergency Q3 app-lexicon slate did not beat ID342 publicly.
- Public `0.55` remains unproven; do not imply prize-range performance.
- The method is an anchor-controlled ensemble and validation-risk-control
  workflow, not a solved Q3/S3/S4 representation breakthrough.
- Negative evidence from KMA ASOS, AirKorea 2024, historical ETRI priors,
  raw sequence gates, missingness/subject-state gates, and registered material
  fusion should be presented as an explicit part of the methodology.

## Do Not Do For Paper Deadline

- Do not spend DACON submissions solely to improve the paper story.
- Do not run long Kaggle/heavy jobs unless a short local gate first opens a
  materially new signal.
- Do not create a new submission artifact unless notebook, CSV, decision
  report, and submission-log lineage can be made one-to-one immediately.
- Do not include secrets, service keys, raw external cache paths with private
  provenance, or Kaggle/DACON credentials.

## Last Verification Commands

```bash
make paper-submission-metadata
make paper-submission-length-audit
make paper-submission-source-archive
make post-id342-deadline-operating-policy
make paper-submission-package-audit
make docs-audit
git diff --check
```
