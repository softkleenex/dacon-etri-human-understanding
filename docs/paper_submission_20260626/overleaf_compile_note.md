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
9. Download the PDF and upload it through EDAS.

## Local Boundary

Tectonic `0.16.9` compiled the source locally on `2026-07-17`. The verified
output is `build/paper_submission_20260626/compile/paper.pdf`: IEEE conference
Letter format, exactly four pages, with no overfull boxes or unresolved
citations. EDAS PDF-checker compliance must still be verified after the author
block is replaced.
