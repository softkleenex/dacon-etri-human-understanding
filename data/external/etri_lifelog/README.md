# ETRI Lifelog External Data

This directory contains same-institution but non-current-period ETRI lifelog data. It is potentially useful, but it is not equivalent to the current DACON challenge data.

## Current contents

`etri_lifelog/` currently keeps lightweight summary CSVs:

- `user_info_2019_2018_updated.csv`, `user_info_2020.csv`
- `user_sleep_2019_2018.csv`, `user_sleep_2020.csv`
- `user_survey_2019_2018.csv`, `user_survey_2020.csv`

Bulky extracted sensor dumps were moved to `artifacts/external_data_archive/` so they cannot silently affect new experiments.

## Comparison with current competition data

Current competition data in `data/raw/` has 10 anonymized subjects (`id01`–`id10`), 2024 dates, official sensor parquet files, and binary targets `Q1`–`Q3`, `S1`–`S4`.

External data uses older 2019–2020 participants (`user01` etc. or numeric IDs), different date ranges, different devices/schemas, and survey/sleep-score columns rather than the exact competition targets. There is no direct row-level join key to the current subjects.

## Usefulness verdict

Use as auxiliary reference only:

- Good for priors: sleep-score ranges, survey scale behavior, broad sleep/condition relationships from older ETRI cohorts.
- Good for feature ideas: duration, wake count, heart-rate, caffeine/alcohol, AM/PM survey separation.
- Risky for direct training: subject IDs, dates, target definitions, devices, and collection protocols differ from the current 2024 challenge data.
- Do not concatenate with current train rows unless a validation note proves it improves leak-free CV.

Any submission using external-derived features must mention it in `submissions/submission_log.csv` notes.
