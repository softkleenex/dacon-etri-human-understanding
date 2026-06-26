# Public-data weather/air API smoke

Created: `2026-06-25 17:14:29 UTC+09:00`

This is report-only. It creates no feature matrix, predictions, submission artifacts, Kaggle runs, or DACON submissions.

## Decision

- KMA ASOS 2024 historical usable: `True`
- KMA living-weather 2024 historical usable from smoke: `False`
- AirKorea realtime available: `True`
- AirKorea station recent 3-month available: `True`
- AirKorea 2024 train-window usable from approved API: `False`
- Raw API cache retained: `False`
- Kaggle now: `False`
- DACON submit now: `False`
- Next action: Use KMA ASOS as the only confirmed 2024 historical source from this smoke. Treat KMA living-weather and AirKorea ArpltnInforInqireSvc as report-only unless a separate historical/statistics source is approved and passes a local ID342 gate.

## Service Summary

| service | calls | successes | historical probes with items | all success |
|---|---:|---:|---:|---|
| `kma_asos` | 1 | 1 | 1 | `True` |
| `kma_life` | 3 | 2 | 0 | `False` |
| `airkorea` | 5 | 5 | 1 | `True` |

## Calls

| service | function | historical | success | result | items | time keys | raw cache |
|---|---|---|---|---|---:|---|---|
| `kma_asos` | `/getWthrDataList` | `True` | `True` | NORMAL_SERVICE | 24 | `tm` | `False` |
| `kma_life` | `/getUVIdxV5` | `False` | `True` | NORMAL_SERVICE | 1 | `date`, `h0` | `False` |
| `kma_life` | `/getAirDiffusionIdxV5` | `False` | `True` | NORMAL_SERVICE | 1 | `date` | `False` |
| `kma_life` | `/getUVIdxV5` | `True` | `False` | 최근 1일 간의 자료만 제공합니다. | 0 |  | `False` |
| `airkorea` | `/getCtprvnRltmMesureDnsty` | `False` | `True` | NORMAL_CODE | 15 | `dataTime` | `False` |
| `airkorea` | `/getMsrstnAcctoRltmMesureDnsty` | `True` | `True` | NORMAL_CODE | 1000 | `dataTime` | `False` |
| `airkorea` | `/getMinuDustFrcstDspth` | `False` | `True` | NORMAL_CODE | 4 | `dataTime`, `informData` | `False` |
| `airkorea` | `/getMinuDustWeekFrcstDspth` | `False` | `True` | NORMAL_CODE | 0 |  | `False` |
| `airkorea` | `/getUnityAirEnvrnIdexSnstiveAboveMsrstnList` | `False` | `True` | NORMAL_CODE | 0 |  | `False` |

## Notes

- The service key value is not written to reports or optional cache metadata.
- Raw API payload caches are not retained by default; use `--keep-raw-cache` only for troubleshooting.
- AirKorea `ArpltnInforInqireSvc` smoke checks realtime, forecast, and recent station-query behavior; it does not prove 2024 historical train/test coverage.
- Any future feature gate must still be ID342 report-only first.
