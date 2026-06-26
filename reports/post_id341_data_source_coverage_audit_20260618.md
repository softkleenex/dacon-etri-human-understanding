# Post-ID341 Data Source Coverage Audit

This is report-only: no model training, no submission CSV, no Kaggle run, and no DACON submission.

## Decision

- Data/source coverage ready: `True`
- All major sources present: `True`
- Evidence reports present: `True`
- Runner references required checks: `True`
- External direct concat allowed: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Next action: Data/source coverage is accounted for locally. Do not concatenate external rows or create ID342; the next score-relevant step remains explicit-approved no-submit Kaggle/Linux upstream rebuild.

## Source groups

| group | all present | missing count |
|---|:---:|---:|
| `official` | `True` | 0 |
| `raw_modalities` | `True` | 0 |
| `external_etri` | `True` | 0 |
| `reference_docs` | `True` | 0 |
| `evidence_reports` | `True` | 0 |

## Raw modality rows

| modality | rows | columns |
|---|---:|---:|
| `mACStatus` | 939896 | 3 |
| `mActivity` | 961062 | 3 |
| `mAmbience` | 476577 | 3 |
| `mBle` | 21830 | 5 |
| `mGps` | 800611 | 6 |
| `mLight` | 96258 | 3 |
| `mScreenStatus` | 939653 | 3 |
| `mUsageStats` | 45197 | 4 |
| `mWifi` | 76336 | 4 |
| `wHr` | 382918 | 3 |
| `wLight` | 633741 | 3 |
| `wPedo` | 748100 | 9 |

## External ETRI files

| file | rows |
|---|---:|
| `user_info_2019_2018_updated` | 50 |
| `user_info_2020` | 22 |
| `user_sleep_2019_2018` | 736 |
| `user_sleep_2020` | 615 |
| `user_survey_2019_2018` | 1285 |
| `user_survey_2020` | 1230 |

## Interpretation

The current issue is not that major local data sources are missing. They are present and already routed into either local closure reports or the no-submit Kaggle upstream runner. The blocker remains validation/public transfer: none of the local watch candidates passes the recent Public `0.58` proxy gate.
