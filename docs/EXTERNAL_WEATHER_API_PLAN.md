# External Weather API Plan

This note records the approved public-data API boundary without storing API
keys. Keep real keys only in ignored local environment files or shell
environment variables.

## Secret Handling

- Never commit public-data API keys.
- Use `KMA_ASOS_SERVICE_KEY` for the KMA ASOS hourly observation API.
- Use `KMA_LIFE_INDEX_SERVICE_KEY` only for living-weather index probes.
- Use `AIRKOREA_SERVICE_KEY` only for AirKorea probes.
- `PUBLIC_DATA_SERVICE_KEY` may be used as a fallback when the same public-data
  service key is approved for multiple data.go.kr APIs.
- API response caches belong under ignored paths such as
  `data/external/weather/`.

## Approved Sources

### KMA ASOS Hourly Observations

- Endpoint: `https://apis.data.go.kr/1360000/AsosHourlyInfoService`
- Function: `/getWthrDataList`
- Use: first-priority external source because it provides historical hourly
  observations.
- Date window for this competition: `2024-06-01` through `2024-11-22`.
- Initial station policy: fixed Daejeon ASOS station `133`, matching the ETRI
  / Daejeon prior and avoiding GPS-derived nearest-station leakage.
- Candidate variables: temperature, humidity, precipitation, wind, pressure,
  sunshine/solar radiation when available.

### KMA Living Weather Index

- Endpoint: `https://apis.data.go.kr/1360000/LivingWthrIdxServiceV5`
- Functions: `/getUVIdxV5`, `/getAirDiffusionIdxV5`
- Use: research-only until proven historical lookup is possible.
- Current risk: the service description is forecast-style, with values from
  publication time to about 75-78 hours ahead. If 2024 historical lookup is not
  supported, it must not be used for train/test feature generation.
- 2026-06-25 smoke result:
  `reports/public_data_weather_air_api_smoke_20260625.md/.json` successfully
  called current UV and air-diffusion endpoints, but a `2024060106` UV probe
  returned result code `99` with the message that only the most recent one day
  of data is provided. Treat this service as forecast/report-only for this
  competition.

### AirKorea Air Pollution

- Endpoint: `https://apis.data.go.kr/B552584/ArpltnInforInqireSvc`
- Use: secondary research source for PM2.5, PM10, ozone, and air-quality
  context.
- Current risk: daily traffic is smaller and the listed license includes
  attribution/no-derivatives terms. Keep it report-only unless the competition
  and data license implications are explicitly reviewed.
- 2026-06-25 smoke result:
  `reports/public_data_weather_air_api_smoke_20260625.md/.json` successfully
  called all approved AirKorea information endpoints attempted in the smoke:
  province realtime measurement, station measurement, fine-dust forecast,
  weekly forecast, and bad-or-worse CAI station list. The station measurement
  endpoint exposes recent `3MONTH` data only; the smoke returned `totalCount`
  `2190` for the requested Daejeon station with page-1 times from
  `2026-05-14 22:00` through `2026-06-25 16:00`. This confirms the API works,
  but the approved `ArpltnInforInqireSvc` endpoints do not provide a
  date-bounded 2024 historical train/test feature source.
- 2026-06-25 downloaded historical archive:
  the user-provided `2024.zip` was moved to ignored
  `data/external/weather/airkorea_2024/airkorea_2024_hourly_all_stations_xlsx.zip`.
  `scripts/audit_airkorea_2024_hourly_zip.py` keeps the XLSX files compressed,
  extracts only ignored
  `data/external/weather/airkorea_2024/airkorea_2024_daejeon_hourly.csv`, and
  writes `reports/airkorea_2024_hourly_zip_audit_20260625.md/.json`.
  The audit confirms the archive is a valid 2024 historical hourly source with
  columns `region/network/station/time/SO2/CO/O3/NO2/PM10/PM25/address`, 13
  Daejeon stations, and full `2024-06-01 00:00` through
  `2024-11-22 23:00` competition-window timestamp coverage.
- 2026-06-25 ID342 gate result:
  `reports/post_id342_airkorea_2024_context_gate_20260625.md/.json` reviewed
  `216` compact target/family/model/weight overlays over hourly/daily Daejeon
  pollution features. It passed `0` local gates, `0` Public-0.58 review gates,
  and `0` DACON gates. The best row was
  `airkorea_selected_station_s4_hgb_l7_w0.025`, but it only projected Public
  `0.596056065`, still `+0.046056065` from Public `0.55`, with wins `3/5`,
  max delta `+0.000819150`, and failed-frontier correlation `+0.262872`.
  Treat downloaded AirKorea 2024 as valid negative external-source evidence,
  not as an artifact, Kaggle, or DACON route.

## API Smoke Verification

Use `make smoke-public-data-weather-air-api` after setting the relevant
environment variables. The smoke script:

- reads keys from `KMA_ASOS_SERVICE_KEY`, `KMA_LIFE_INDEX_SERVICE_KEY`,
  `AIRKOREA_SERVICE_KEY`, or `PUBLIC_DATA_SERVICE_KEY`;
- keeps only summary metadata by default and does not retain raw API payload
  caches;
- can retain troubleshooting payloads under ignored `data/external/weather/`
  only when explicitly run with `--keep-raw-cache`;
- records `service_key_value_stored=false`; and
- creates only `reports/public_data_weather_air_api_smoke_20260625.json/.md`.

## Execution Contract

1. Fetch external weather data only through scripts that read keys from
   environment variables.
2. Write raw/cache outputs only under ignored `data/external/weather/`.
3. Run local report-only gates before any artifact, Kaggle, or DACON action.
4. Do not create submission CSVs, base prediction artifacts, notebook history
   artifacts, or submission-log rows from external weather results unless a
   separate artifact gate passes.
5. Keep downloaded AirKorea XLSX archives and normalized CSV caches ignored;
   commit only the audit/gate scripts and concise reports.
