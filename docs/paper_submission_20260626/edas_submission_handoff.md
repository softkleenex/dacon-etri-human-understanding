# EDAS Submission Handoff

Status: human handoff checklist for the ICTC/EDAS paper upload. This document
does not create predictions, feature matrices, submission artifacts, Kaggle
runs, submission-log rows, or DACON submissions.

## Official Links

- ICTC 2026 paper submission page:
  `https://ictc.org/authors_paperSubmission`
- ICTC EDAS submission link observed from the official ICTC page:
  `https://edas.info/newPaper.php?c=35149`
- DACON rules page:
  `https://dacon.io/en/competitions/official/236690/overview/rules`
- DACON paper-deadline talkboard notice:
  `https://dacon.io/en/competitions/official/236690/talkboard/416793`
- EDAS author instructions:
  `https://edas.info/doc/authors.html`

## Required Human Actions

1. Log in to EDAS or create an EDAS account.
2. Open the ICTC submission link above.
3. Confirm the IWETRIAI / ETRI Human Understanding workshop track is visible.
4. Register the paper title and abstract from `paper.tex`.
5. Add all authors, affiliations, and email addresses.
6. Upload the compiled IEEE-format four-page PDF.
7. Check the EDAS PDF checker output and fix any page/font/template warnings.
8. Save the EDAS paper ID and submission timestamp outside this repo, then
   optionally mirror non-secret metadata into a future report.

## Draft Values To Copy Carefully

- Title:
  `Conservative Multimodal Lifelog Modeling with Public-Anchor Risk Control for Sleep and Stress State Prediction`
- Working source:
  `docs/paper_submission_20260626/paper.tex`
- EDAS metadata:
  `docs/paper_submission_20260626/edas_metadata.md`
- Bibliography:
  `docs/paper_submission_20260626/references.bib`
- Model-description draft:
  `docs/paper_submission_20260626/model_description_draft.md`
- Lineage manifest:
  `docs/paper_submission_20260626/id342_lineage_manifest.md`

## Risk Notes

- Do not claim Public `0.55` or prize-range performance.
- Do not submit ID343/ID344 or any other low-evidence DACON file to improve
  the paper story.
- Do not expose API keys, DACON credentials, Kaggle credentials, or EDAS
  credentials in any repo file.
- Local `pdflatex` is currently unavailable in this workspace, so final PDF
  compilation must happen in another TeX environment unless TeX is installed.
