# Paper Submission Final Handoff

Status: updated handoff for the 2026-07-31 paper-submission path. This is
non-score-bearing and creates no predictions, feature matrices, submission CSVs,
Kaggle runs, submission-log rows, or DACON submissions.

## Current Decision

- Paper submission deadline date: `2026-07-31 23:59`.
- Deadline source: DACON notice updated `2026-07-17 09:10`.
- Algorithm close date: `2026-07-23`.
- Paper package audit: use `reports/paper_submission_package_audit_20260625.md`.
- Source ZIP: `build/paper_submission_20260626/paper_submission_20260626_source.zip`.
- EDAS submission link: `https://edas.info/newPaper.php?c=35149`.
- Working source: `docs/paper_submission_20260626/paper.tex`.
- Verified PDF: `build/paper_submission_20260626/compile/paper.pdf` (`4` pages,
  IEEE conference Letter format, Tectonic `0.16.9`).
- Metadata to copy: `docs/paper_submission_20260626/edas_metadata.md`.
- Compile note: `docs/paper_submission_20260626/overleaf_compile_note.md`.
- EDAS paper ID: `1571325954`.
- EDAS status: `Active (has manuscript)`.
- Review manuscript: uploaded and accepted by EDAS as a four-page Letter-size
  PDF; reference section starts on page `4.3`, PDF version is `1.5`.

## Score Boundary

- Current confirmed public anchor: ID342, Public `0.5960832519`, local
  `0.534000`, DACON submission id `1467937`.
- Final emergency slate IDs `346`, `350`, and `347` did not beat ID342
  publicly.
- Public `0.55` is not achieved.
- The paper should present a conservative, reproducible risk-control workflow
  and negative-evidence accounting, not prize-range performance.

## Submit Policy

- Kaggle run now: `False`.
- DACON submit now: `False`.
- Do not spend DACON slots solely to improve the paper story.
- Reopen score-bearing work only if a new candidate clears a fresh local,
  stable-shape, and failed-frontier gate and if the competition accepts the
  relevant post-close artifact path.

## Human Fill-Ins

- The paper is configured for one corresponding author, `Shangjae Lee`, at
  Kyungpook National University using `dltkdwo31@knu.ac.kr`.
- EDAS track selection and paper metadata confirmation.
- Final rebuild after author insertion and EDAS PDF-checker result.

These review-submission actions are complete. `Final manuscript` remains a
future camera-ready action only if requested after the review decision.
- EDAS paper ID and final submission timestamp.

## Verification Commands

```bash
make paper-submission-package-audit
make post-id342-deadline-operating-policy
make docs-audit
```
