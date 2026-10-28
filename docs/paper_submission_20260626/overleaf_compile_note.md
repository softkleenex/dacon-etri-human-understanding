# Overleaf Compile Note

Status: human compile note for the paper source package. This document creates
no predictions, feature matrices, submission CSVs, Kaggle runs, submission-log
rows, or DACON submissions.

## Source Package

- Source ZIP: `build/paper_submission_20260626/paper_submission_20260626_source.zip`
- Main file: `paper.tex`
- Bibliography: `references.bib`
- Expected format: IEEE conference, four pages maximum

## Overleaf Steps

1. Run `make paper-submission-package-audit` before uploading the source ZIP.
2. Upload the ZIP to a new Overleaf project.
3. Set `paper.tex` as the main file if Overleaf does not detect it.
4. Use pdfLaTeX unless the EDAS checker requires a different compiler.
5. Recompile from scratch after bibliography changes.
6. Confirm the compiled PDF is four pages or fewer.
7. Confirm the bibliography appears and there are no unresolved citations.
8. Confirm the author block matches the EDAS profile for `Shangjae Lee`.
9. Upload only when an organizer instruction opens a revision or camera-ready
   action; do not overwrite the recorded review submission by default.

## Local Boundary

Tectonic `0.16.9` compiles the source to
`build/paper_submission_20260626/compile/paper.pdf`, and the Make target copies
the same bytes to `paper.pdf`. The immutable EDAS review snapshot is
`review_manuscript_submitted_20260717.pdf`. Any future uploaded revision still
requires the EDAS PDF checker.
