# Post-ID341 family priority

Created: `2026-06-18 04:08:39 UTC+09:00`

This is report-only. It creates no CSV, notebook/history artifact, base prediction artifact, submission-log row, or DACON submission.

## Public anchor context

- ID178 Public: `0.6014113009`
- ID287 Public: `0.6698787318`
- ID329 Public: `0.6017908577`
- Read: Recent public evidence says local CV below ~0.54 is not enough; transfer proxy and anchor movement dominate candidate selection.

## Current control candidate

- ID: `341`
- File: `submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv`
- Local CV: `0.538584`
- Proxy: `0.603830`
- Role: lowest-risk ID178-neighborhood control/probe, explicit approval only

## Lane priority

| lane | status | read | next action |
|---|---|---|---|
| `sequence_micro_control` | `hold_as_control` | ID341 beats ID340/337/339 proxy but remains around 0.604; not a Public 0.58 path by itself. | Do not keep shrinking sequence micro weights unless a new source lowers proxy below ID341 and changes more than numerical noise. |
| `demographic_static_target_overlay` | `closed_for_artifact` | promoted_targets=[], proxy_delta_vs_id341=+0.000000 | Keep demographic/static features as analysis context only; no local artifact recombination from this branch. |
| `A1_q240_free_bridge` | `closed_for_local_artifact` | S4 survives target replay, but ID178-anchor S4 best proxy=0.603926, delta_vs_ID341=+0.000096 | Only revisit with true upstream model-level Kaggle rebuild; do not use existing OOF artifacts as clean sources. |
| `B1_pair_logit_bridge` | `blocked_until_source_rebuild` | Selector signal is contaminated by high/extreme non-nested dependencies including ID242/252 chain. | Requires rebuilding below source IDs 254/255/256/257/258/266, not more local blending. |
| `raw_tabular_source_rebuild` | `deprioritized` | Raw/tabular source failed all primary ID340 split lenses against ID178. | Do not spend local cycles unless feature representation changes materially. |
| `true_upstream_kaggle_rebuild` | `next_heavy_candidate` | If compute is spent, it must rebuild upstream model-level sources/folds, not replay existing OOF artifacts. | Prepare a report-only Kaggle runner blueprint for upstream source rebuild with no DACON submission path. |

## Decision

- Submit now: **no**
- Create artifact now: **no**
- Public 0.58-plausible candidate exists: **no**
- Forced probe fallback: ID341 for lowest-movement sanity probe; ID337 only if specifically probing S1 sequence signal.
- Next recommended work: Prepare/report a Kaggle upstream model-level rebuild blueprint, or design a genuinely new feature source; stop local artifact recombination lanes that reuse existing OOF shortcuts.
