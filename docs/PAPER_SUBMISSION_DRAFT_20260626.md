# ETRI Human Understanding Challenge 2026 Paper Draft

Status: updated technical draft from verified local records. Author,
affiliation, and final EDAS metadata still need human review before EDAS/ICTC
submission. The canonical IEEE source compiles locally to exactly four pages.

IEEE package draft: `docs/paper_submission_20260626/paper.tex`.

## Title

Anchor-Constrained Feature-Similarity Ensembling for Multimodal Lifelog State
Prediction

## Abstract

Predicting daily sleep and stress states from lifelogs is difficult when labels
are scarce, observations are participant-dependent, and validation does not
match the hidden test distribution. We present an anchor-constrained ensemble
for seven binary targets in the 5th ETRI Human Understanding AI Paper
Challenge. A stable sequence-based probability model is augmented with a
same-participant nearest-neighbor signal computed from four sensor feature
families. Target-specific similarity predictions are first blended
conservatively into a secondary model, then shrunk halfway toward the stable
public anchor. On 450 labeled person-days from 10 participants, the resulting
model obtains a local mean log loss of 0.534000 and a DACON public score of
0.5960832519, improving the previous anchor by 0.0053280490. More aggressive
models achieved much lower local loss but transferred poorly. The result is a
practical validation and risk-control procedure for small-sample multimodal
lifelog prediction; the local score remains a non-nested model-selection
estimate.

## Contributions

This draft should be framed around three conservative contributions:

- A multimodal lifelog classification pipeline that combines public-anchor
  probability control with target-specific source screening.
- A validation protocol that explicitly models public-transfer risk using
  failed-frontier submissions, subject-hole splits, LOSO-style stress checks,
  and test-neighbor holdouts.
- A reproducible negative-evidence record showing which plausible external,
  temporal, and subject-state feature families did not transfer under the
  current ID342 anchor.

## Task And Targets

The challenge target consists of seven binary outcomes:

- Q1: subjective sleep quality relative to each participant's average.
- Q2: subjective physical fatigue before sleep, where 1 means lower fatigue.
- Q3: subjective stress before sleep, where 1 means lower stress.
- S1: total sleep time guideline adherence.
- S2: sleep efficiency guideline adherence.
- S3: sleep onset latency guideline adherence.
- S4: wake after sleep onset guideline adherence.

All targets use 1 for a positive or healthy state and 0 for a negative or
unhealthy state. The metric is treated locally as multi-target log loss over
the seven binary outcomes.

## Data Sources

The local system uses processed competition features derived from smartphone,
wearable, proximity, location, light, screen, activity, app-usage, and sleep
proxy streams. In addition, the following external or auxiliary sources were
audited:

- KMA ASOS hourly weather for Daejeon, 2024-06-01 through 2024-11-22.
- Downloaded AirKorea 2024 hourly air-quality archive for 13 Daejeon stations.
- Historical ETRI lifelog survey/sleep CSVs from 2018-2020.
- Prior ETRI lifelog papers and competition method papers.

External sources were used only after report-only gates. No external source
was promoted into a final submission artifact unless it passed the local and
transfer-risk gates; none did.

## Validation Design

Validation is deliberately stricter than full out-of-fold score selection. The
candidate review stack uses multiple lenses:

- subject-hole splits,
- leave-one-subject-out stress checks,
- test-boundary neighbor holdouts,
- target-specific local loss deltas,
- public-gap proxy estimates,
- failed-frontier correlation checks against known bad public submissions,
- submission-log lineage checks tying every notebook, CSV, decision report, and
base prediction artifact together.

This design was introduced because very low local CV rows repeatedly failed on
the public leaderboard. The main failure mode was overconfident or
public-frontier-shaped movement that improved local validation while worsening
public score.

## Method Summary

The final confirmed system should be described as a conservative post-hoc
ensemble, not as a single end-to-end model. ID342 is a convex shrink blend
between a confirmed public anchor lineage and a stricter feature-similarity
source. Each later candidate was reviewed as a small target-specific movement
from ID342. Promotion required both local improvement and stable behavior under
near-anchor validation lenses; candidates that aligned with known failed public
directions were rejected even when local CV improved.

The late-stage experiments therefore act as an error-analysis and risk-control
layer. Weather, pollution, historical priors, raw sequence gates, and
subject-state representations were tested as possible independent signals for
Q3/S3/S4, but none cleared the review gate.

## Final Confirmed Submission

Current public anchor:

- ID: 342
- File:
  `submissions/submission_id_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.csv`
- Local CV: 0.534000
- Public score: 0.5960832519
- DACON submission id: 1467937
- Submitted at: 2026-06-23 23:30:37 KST

ID342 blends ID178 and ID221 through a feature-similarity shrink operation with
alpha 0.50. The purpose is not to chase the lowest local loss, but to keep an
independent strict-feature signal while limiting movement away from a known
public anchor.

The 2026-06-26 emergency Q3 app-name lexicon slate produced local improvements
but all submitted variants scored worse than ID342 publicly. This strengthens
the paper's validation-risk narrative.

## Confirmed Public Results

| ID | Description | Local CV | Public | Note |
| --- | --- | ---: | ---: | --- |
| 342 | ID178/ID221 feature-similarity shrink alpha 0.50 | 0.534000 | 0.5960832519 | Current confirmed public anchor |
| 339 | ID178 Q1/Q2/Q3 sequence-summary plus ID337 S1 | 0.536441 | 0.5985187787 | Useful but weaker than ID342 |
| 338 | ID179 S1 random-convolution safety overlay | 0.537861 | 0.5992145177 | Useful but weaker than ID342 |
| 329 | Guarded practical probe | 0.516718 | 0.6017908577 | Local-CV optimism warning |
| 287 | Failed local frontier | 0.378934 | 0.6698787318 | Strongest failed-frontier warning |

The main empirical lesson is that local CV and public leaderboard order can
diverge sharply unless failed-frontier submissions are treated as explicit
negative controls.

## Negative Evidence

Several plausible routes were closed before this draft:

- KMA ASOS weather: valid historical source, but only a tiny unstable S4
  signal near Public 0.596006 and no review gate pass.
- AirKorea 2024: valid Daejeon historical air-quality source with complete
  competition-window coverage, but 0 local, review, or DACON gates passed. The
  best proxy was 0.596056065.
- Historical ETRI survey/sleep prior: Q2 fatigue-to-sleep coupling was
  coherent, but Q3 stress did not support a simple sleep-quality proxy and no
  direct 2026 feature route opened.
- Missingness-aware and self-supervised subject-state runs: strong coverage
  but zero stable downstream passes.
- MIS-LSTM-style block sequence proxy: selected only S4 locally and failed
  Q3/S3 transfer requirements.
- Registered material-source fusion: no local gate pass and best public proxy
  remained near 0.596.
- Stable-envelope recombination and target reallocation improved projections
  only to about Public 0.566, still far from the 0.55 target.

## Reproducibility Pointers

Core reports:

- `reports/dacon_three_slot_submission_result_20260623.md`
- `reports/post_id342_remaining_route_synthesis_20260625.md`
- `reports/post_id342_external_research_priority_20260625.md`
- `reports/post_id342_deadline_operating_policy_20260625.md`
- `reports/post_id342_airkorea_2024_context_gate_20260625.md`
- `reports/post_id342_external_etri_sleep_survey_prior_20260625.md`

Paper package:

- `docs/paper_submission_20260626/paper.tex`
- `docs/paper_submission_20260626/references.bib`
- `docs/paper_submission_20260626/README.md`

Core commands:

```bash
make docs-audit
make post-id342-remaining-route-synthesis
make post-id342-external-research-priority
make post-id342-deadline-operating-policy
```

Security notes:

- API keys are not stored in tracked files.
- Raw weather and AirKorea caches stay under ignored `data/external/weather/`.
- No DACON submission should be run from this draft.

## Deadline Notes

Verified/updated public sources:

- DACON overview search result lists algorithm close as 2026-07-23.
- DACON talkboard notice updated on 2026-07-17 lists paper submission deadline
  as 2026-07-31 23:59 and links to EDAS `https://edas.info/newPaper.php?c=35149`.
- ICTC official paper-submission page states that both full and short papers
  must be at least 2 pages.
- DACON rules for this challenge require the IWETRIAI submission to use the
  Standard IEEE conference template as a 4-page full paper.

Because the paper deadline has been extended, the deadline-aware policy is to
use the remaining time for paper quality, reproducibility packaging, and final
author/EDAS preparation. It does not reopen low-evidence DACON submissions.

## Limitations

The confirmed public score remains far from the Public 0.55 target. The system
does not solve the Q3/S3/S4 representation gap. The strongest late-stage lesson
is methodological: local CV alone is not a reliable promotion criterion for
this challenge. Stable shape, failed-frontier avoidance, and target-specific
domain gates are necessary to avoid spending submissions on misleading local
gains.
