# AirKorea 2024 hourly archive audit

Created: `2026-06-25 17:29:09 UTC+09:00`

This is report-only. It creates a normalized ignored Daejeon air-quality cache, but no model feature matrix, predictions, submission artifacts, Kaggle runs, or DACON submissions.

## Decision

- Archive available: `True`
- Normalized Daejeon cache created: `True`
- Competition-window coverage OK: `True`
- Run Kaggle now: `False`
- DACON submit now: `False`
- Next action: Run a report-only ID342 gate over Daejeon AirKorea hourly/daily features. Do not create submission artifacts unless that gate clears the Public-0.58 review proxy and failed-frontier checks.

## Inputs

- Input zip: `data/external/weather/airkorea_2024/airkorea_2024_hourly_all_stations_xlsx.zip`
- Normalized Daejeon CSV: `data/external/weather/airkorea_2024/airkorea_2024_daejeon_hourly.csv`
- Zip size bytes: `345117708`

## Schema

- Expected columns: `['지역', '망', '측정소코드', '측정소명', '측정일시', 'SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM25', '주소']`

## Row Counts

- Total national hourly rows: `5819357`
- Daejeon hourly rows: `114192`
- Daejeon stations: `13`
- Daejeon time range: `2024-01-01 01:00:00` to `2025-01-01 00:00:00`

## Competition Window

- Window: `2024-06-01 00:00:00` to `2024-11-22 23:00:00`
- Expected hours: `4200`
- Daejeon rows: `54600`
- Unique timestamps: `4200`
- Unique stations: `13`

## Pollutant Summary

| pollutant | non-null | missing | mean | p05 | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| `so2` | 52506 | 2094 | 0.002480 | 0.002000 | 0.002400 | 0.003200 |
| `co` | 52351 | 2249 | 0.350040 | 0.210000 | 0.340000 | 0.540000 |
| `o3` | 52660 | 1940 | 0.029971 | 0.003400 | 0.027950 | 0.065700 |
| `no2` | 52280 | 2320 | 0.012506 | 0.004000 | 0.010000 | 0.029500 |
| `pm10` | 51840 | 2760 | 20.322897 | 5.000000 | 18.000000 | 42.000000 |
| `pm25` | 51121 | 3479 | 11.574050 | 1.000000 | 10.000000 | 27.000000 |

## Station Coverage

| station | name | rows | unique hours | coverage | address |
|---|---|---:|---:|---:|---|
| `525111` | `읍내동` | 4200 | 4200 | 1.000000 | 대전 대덕구 대전로 1331번길 75(태아산업(주)) |
| `525112` | `문평동` | 4200 | 4200 | 1.000000 | 대전 대덕구 문평동로 18번길 34(문평동 119안전센터) |
| `525121` | `문창동` | 4200 | 4200 | 1.000000 | 대전 중구 보문로 20번길 38(문창동 주민센터) |
| `525141` | `구성동` | 4200 | 4200 | 1.000000 | 대전 유성구 대학로 407 보건환경연구원 |
| `525142` | `노은동` | 4200 | 4200 | 1.000000 | 대전 유성구 노은동로 87번길 89(노은1동 주민센터) |
| `525143` | `상대동(대전)` | 4200 | 4200 | 1.000000 | 대전광역시 유성구 도안대로 398 |
| `525144` | `관평동` | 4200 | 4200 | 1.000000 | 대전광역시 유성구 테크노중앙로 88 |
| `525151` | `대흥동1` | 4200 | 4200 | 1.000000 | 대전 중구 중앙로 58 |
| `525161` | `성남동1` | 4200 | 4200 | 1.000000 | 대전 동구 계족로 368번길 70(성남동 주민센터) |
| `525162` | `대성동` | 4200 | 4200 | 1.000000 | 대전광역시 동구 동구청로 36 |
| `525171` | `정림동` | 4200 | 4200 | 1.000000 | 대전 서구 정림동로 10(정림동 주민센터) |
| `525172` | `둔산동` | 4200 | 4200 | 1.000000 | 대전 서구 둔산서로 84(근로자 종합복지회관) |
| `525173` | `월평동` | 4200 | 4200 | 1.000000 | 대전 서구 월평동160-5 |

## Monthly Files

| month | file | rows | Daejeon rows | time min | time max |
|---:|---|---:|---:|---|---|
| 1 | `2024/2024년 1월.xlsx` | 488064 | 9672 | `2024-01-01 01:00:00` | `2024-02-01 00:00:00` |
| 2 | `2024/2024년 2월.xlsx` | 458986 | 9048 | `2024-02-01 01:00:00` | `2024-03-01 00:00:00` |
| 3 | `2024/2024년 3월.xlsx` | 491304 | 9672 | `2024-03-01 01:00:00` | `2024-04-01 00:00:00` |
| 4 | `2024/2024년 4월.xlsx` | 476263 | 9360 | `2024-04-01 01:00:00` | `2024-05-01 00:00:00` |
| 5 | `2024/2024년 5월.xlsx` | 493045 | 9672 | `2024-05-01 01:00:00` | `2024-06-01 00:00:00` |
| 6 | `2024/2024년 6월.xlsx` | 478007 | 9360 | `2024-06-01 01:00:00` | `2024-07-01 00:00:00` |
| 7 | `2024/2024년 7월.xlsx` | 494016 | 9672 | `2024-07-01 01:00:00` | `2024-08-01 00:00:00` |
| 8 | `2024/2024년 8월.xlsx` | 494016 | 9672 | `2024-08-01 01:00:00` | `2024-09-01 00:00:00` |
| 9 | `2024/2024년 9월.xlsx` | 478080 | 9360 | `2024-09-01 01:00:00` | `2024-10-01 00:00:00` |
| 10 | `2024/2024년 10월.xlsx` | 494016 | 9672 | `2024-10-01 01:00:00` | `2024-11-01 00:00:00` |
| 11 | `2024/2024년 11월.xlsx` | 478800 | 9360 | `2024-11-01 01:00:00` | `2024-12-01 00:00:00` |
| 12 | `2024/2024년 12월.xlsx` | 494760 | 9672 | `2024-12-01 01:00:00` | `2025-01-01 00:00:00` |
