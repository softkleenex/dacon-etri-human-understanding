# EDAS Metadata

Created: `2026-07-31 15:23:45 UTC+09:00`

This is copy-paste metadata for EDAS. It creates no predictions, feature matrices, submission CSVs, Kaggle runs, or DACON submissions.

## Title

Anchor-Constrained Feature-Similarity Ensembling for Multimodal Lifelog State Prediction

## Abstract

Predicting daily sleep and stress states from lifelogs is difficult when labels are scarce, observations are participant-dependent, and validation does not match the hidden test distribution. We present an anchor-constrained ensemble for seven binary targets in the 5th ETRI Human Understanding AI Paper Challenge. A stable sequence-based probability model is augmented with a same-participant nearest-neighbor signal computed from four sensor feature families. Target-specific similarity predictions are first blended conservatively into a secondary model, then shrunk halfway toward the stable public anchor. On 450 labeled person-days from 10 participants, the resulting model obtains a local mean log loss of 0.534000 and a DACON public score of 0.5960832519, improving the previous anchor by 0.0053280490. More aggressive models achieved much lower local loss but transferred poorly. Ablations and three later public probes show that small local gains are not sufficient evidence near the anchor. The main result is therefore a practical validation and risk-control procedure for multimodal lifelog prediction under small-sample distribution shift.

## EDAS Form Fields

- Topic: `Artificial Intelligence and Machine Learning`
- Author count: `1`
- Author name: `Shangjae Lee`
- Affiliation: `Kyungpook National University`
- Email: `dltkdwo31@knu.ac.kr`
- Corresponding author: `True`
- EDAS paper ID: `1571325954`
- Status: `Active (has manuscript)`
- Review manuscript uploaded: `True`

## Keywords 1-6

- Keyword 1: `lifelog`
- Keyword 2: `multimodal sensing`
- Keyword 3: `sleep prediction`
- Keyword 4: `stress prediction`
- Keyword 5: `nearest neighbors`
- Keyword 6: `ensemble learning`

Combined: lifelog, multimodal sensing, sleep prediction, stress prediction, nearest neighbors, ensemble learning

## Links

- EDAS: `https://edas.info/newPaper.php?c=35149`
- ICTC: `https://ictc.org/authors_paperSubmission`
- DACON rules: `https://dacon.io/en/competitions/official/236690/overview/rules`

## Counts

- Abstract words: `160`
- Keywords: `6`

## Human Fill-Ins

- None.
