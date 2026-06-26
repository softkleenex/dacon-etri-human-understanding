#!/usr/bin/env python3
"""Smoke-test approved public-data weather and air-quality APIs.

This script verifies the three user-approved public-data service families with
real network calls while keeping service keys only in environment variables:

- KMA ASOS hourly observations
- KMA Living Weather Index V5
- AirKorea air-pollution information

It writes a tracked summary report and, by default, does not retain raw API
payload caches. It does not create model features, predictions, submission
artifacts, Kaggle runs, or DACON submissions.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

try:
    import certifi
except ImportError:  # pragma: no cover - certifi is expected in the project venv.
    certifi = None

ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "public_data_weather_air_api_smoke_20260625.json"
DEFAULT_CACHE_DIR = ROOT / "data" / "external" / "weather" / "api_smoke_raw_cache"

KMA_ASOS_ENDPOINT = "https://apis.data.go.kr/1360000/AsosHourlyInfoService"
KMA_LIFE_ENDPOINT = "https://apis.data.go.kr/1360000/LivingWthrIdxServiceV5"
AIRKOREA_ENDPOINT = "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc"

SECRET_ENV_BY_SERVICE = {
    "kma_asos": ("KMA_ASOS_SERVICE_KEY", "PUBLIC_DATA_SERVICE_KEY"),
    "kma_life": ("KMA_LIFE_INDEX_SERVICE_KEY", "PUBLIC_DATA_SERVICE_KEY"),
    "airkorea": ("AIRKOREA_SERVICE_KEY", "PUBLIC_DATA_SERVICE_KEY"),
}


@dataclass(frozen=True)
class ApiCallSpec:
    service: str
    function: str
    description: str
    params: dict[str, Any]
    historical_probe: bool = False


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def service_key_from_env(service: str) -> tuple[str | None, str | None]:
    for env_name in SECRET_ENV_BY_SERVICE[service]:
        value = os.environ.get(env_name)
        if value:
            return env_name, value
    return None, None


def ssl_context(allow_insecure_ssl: bool) -> ssl.SSLContext:
    if allow_insecure_ssl:
        return ssl._create_unverified_context()  # noqa: S323
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def endpoint_for_service(service: str) -> str:
    if service == "kma_asos":
        return KMA_ASOS_ENDPOINT
    if service == "kma_life":
        return KMA_LIFE_ENDPOINT
    if service == "airkorea":
        return AIRKOREA_ENDPOINT
    raise ValueError(f"Unknown service: {service}")


def build_url(endpoint: str, function: str, service_key: str, params: dict[str, Any]) -> str:
    encoded = urlencode(params)
    return f"{endpoint}{function}?serviceKey={service_key}&{encoded}"


def parse_payload(raw: bytes) -> tuple[Any, str | None]:
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return text[:1000], f"JSONDecodeError: {exc}"


def nested_get(payload: Any, path: list[str]) -> Any:
    current = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def normalize_items(payload: Any) -> list[dict[str, Any]]:
    items = nested_get(payload, ["response", "body", "items", "item"])
    if isinstance(items, dict):
        return [items]
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    body_items = nested_get(payload, ["response", "body", "items"])
    if isinstance(body_items, list):
        return [item for item in body_items if isinstance(item, dict)]
    if isinstance(body_items, dict) and any(not isinstance(value, (str, int, float, type(None))) for value in body_items.values()):
        flat: list[dict[str, Any]] = []
        for value in body_items.values():
            if isinstance(value, dict):
                flat.append(value)
            elif isinstance(value, list):
                flat.extend(item for item in value if isinstance(item, dict))
        return flat
    return []


def header_status(payload: Any) -> dict[str, Any]:
    header = nested_get(payload, ["response", "header"])
    if isinstance(header, dict):
        return {
            "result_code": header.get("resultCode"),
            "result_msg": header.get("resultMsg"),
        }
    return {"result_code": None, "result_msg": None}


def body_count(payload: Any) -> int | None:
    value = nested_get(payload, ["response", "body", "totalCount"])
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def item_time_values(items: list[dict[str, Any]]) -> dict[str, Any]:
    time_keys = [
        "tm",
        "date",
        "dataTime",
        "informData",
        "informDate",
        "issueDate",
        "time",
        "h0",
        "today",
    ]
    observed: dict[str, list[str]] = {}
    for item in items:
        for key in time_keys:
            value = item.get(key)
            if value not in (None, ""):
                observed.setdefault(key, []).append(str(value))
    return {
        key: {
            "sample": values[:3],
            "min": min(values),
            "max": max(values),
        }
        for key, values in observed.items()
    }


def item_columns(items: list[dict[str, Any]]) -> list[str]:
    columns: set[str] = set()
    for item in items[:20]:
        columns.update(str(key) for key in item)
    return sorted(columns)


def redacted_cache_name(spec: ApiCallSpec) -> str:
    function = spec.function.strip("/").replace("/", "_")
    suffix = "historical" if spec.historical_probe else "current"
    return f"{spec.service}_{function}_{suffix}.json"


def call_api(spec: ApiCallSpec, args: argparse.Namespace) -> dict[str, Any]:
    key_env, service_key = service_key_from_env(spec.service)
    endpoint = endpoint_for_service(spec.service)
    cache_path = args.cache_dir / redacted_cache_name(spec)
    if not service_key or not key_env:
        return {
            "service": spec.service,
            "function": spec.function,
            "description": spec.description,
            "endpoint": endpoint,
            "params_without_secret": spec.params,
            "historical_probe": spec.historical_probe,
            "request_made": False,
            "success": False,
            "error": f"missing key env; set one of {SECRET_ENV_BY_SERVICE[spec.service]}",
            "service_key_value_stored": False,
            "raw_cache_written": False,
            "cache_path": None,
        }
    url = build_url(endpoint, spec.function, service_key, spec.params)
    started = datetime.now(KST)
    try:
        with urlopen(url, timeout=args.timeout_seconds, context=ssl_context(args.allow_insecure_ssl)) as response:
            status = int(response.status)
            raw = response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        return {
            "service": spec.service,
            "function": spec.function,
            "description": spec.description,
            "endpoint": endpoint,
            "params_without_secret": spec.params,
            "historical_probe": spec.historical_probe,
            "request_made": True,
            "success": False,
            "http_status": getattr(exc, "code", None),
            "error": f"{type(exc).__name__}: {exc}",
            "service_key_env": key_env,
            "service_key_value_stored": False,
            "raw_cache_written": False,
            "cache_path": None,
        }
    payload, parse_error = parse_payload(raw)
    cache_path_text = None
    raw_cache_written = False
    if args.keep_raw_cache:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_payload = {
            "created_at_kst": started.strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
            "service": spec.service,
            "function": spec.function,
            "endpoint": endpoint,
            "params_without_secret": spec.params,
            "historical_probe": spec.historical_probe,
            "service_key_env": key_env,
            "service_key_value_stored": False,
            "payload": payload,
        }
        cache_path.write_text(
            json.dumps(cache_payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        cache_path_text = rel(cache_path)
        raw_cache_written = True
    items = normalize_items(payload)
    header = header_status(payload)
    result_code = str(header.get("result_code") or "")
    api_success = parse_error is None and status == 200 and (not result_code or result_code in {"0", "00", "NORMAL_CODE"})
    return {
        "service": spec.service,
        "function": spec.function,
        "description": spec.description,
        "endpoint": endpoint,
        "params_without_secret": spec.params,
        "historical_probe": spec.historical_probe,
        "request_made": True,
        "success": bool(api_success),
        "http_status": status,
        "parse_error": parse_error,
        "result_code": header.get("result_code"),
        "result_msg": header.get("result_msg"),
        "total_count": body_count(payload),
        "item_count": int(len(items)),
        "item_columns": item_columns(items),
        "item_time_values": item_time_values(items),
        "cache_path": cache_path_text,
        "service_key_env": key_env,
        "service_key_value_stored": False,
        "raw_cache_written": raw_cache_written,
        "raw_bytes": int(len(raw)),
    }


def current_kma_life_time() -> str:
    now = datetime.now(KST)
    base_hour = (now.hour // 3) * 3
    return now.replace(hour=base_hour, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H")


def airkorea_search_date() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


def call_specs(args: argparse.Namespace) -> list[ApiCallSpec]:
    life_time = args.kma_life_time or current_kma_life_time()
    search_date = args.airkorea_search_date or airkorea_search_date()
    return [
        ApiCallSpec(
            service="kma_asos",
            function="/getWthrDataList",
            description="KMA ASOS hourly historical observation smoke for Daejeon station 133.",
            historical_probe=True,
            params={
                "pageNo": 1,
                "numOfRows": 24,
                "dataType": "JSON",
                "dataCd": "ASOS",
                "dateCd": "HR",
                "startDt": "20240601",
                "startHh": "00",
                "endDt": "20240601",
                "endHh": "23",
                "stnIds": "133",
            },
        ),
        ApiCallSpec(
            service="kma_life",
            function="/getUVIdxV5",
            description="KMA living-weather UV index forecast smoke for Daejeon area.",
            params={
                "pageNo": 1,
                "numOfRows": 10,
                "dataType": "JSON",
                "areaNo": args.kma_life_area_no,
                "time": life_time,
            },
        ),
        ApiCallSpec(
            service="kma_life",
            function="/getAirDiffusionIdxV5",
            description="KMA living-weather air-diffusion forecast smoke for Daejeon area.",
            params={
                "pageNo": 1,
                "numOfRows": 10,
                "dataType": "JSON",
                "areaNo": args.kma_life_area_no,
                "time": life_time,
            },
        ),
        ApiCallSpec(
            service="kma_life",
            function="/getUVIdxV5",
            description="KMA living-weather UV historical-window probe for 2024-06-01.",
            historical_probe=True,
            params={
                "pageNo": 1,
                "numOfRows": 10,
                "dataType": "JSON",
                "areaNo": args.kma_life_area_no,
                "time": "2024060106",
            },
        ),
        ApiCallSpec(
            service="airkorea",
            function="/getCtprvnRltmMesureDnsty",
            description="AirKorea province real-time measurement smoke for Daejeon.",
            params={
                "returnType": "json",
                "numOfRows": 100,
                "pageNo": 1,
                "sidoName": args.airkorea_sido_name,
                "ver": "1.0",
            },
        ),
        ApiCallSpec(
            service="airkorea",
            function="/getMsrstnAcctoRltmMesureDnsty",
            description="AirKorea station measurement smoke for a Daejeon station and largest exposed term.",
            historical_probe=True,
            params={
                "returnType": "json",
                "numOfRows": 1000,
                "pageNo": 1,
                "stationName": args.airkorea_station_name,
                "dataTerm": "3MONTH",
                "ver": "1.0",
            },
        ),
        ApiCallSpec(
            service="airkorea",
            function="/getMinuDustFrcstDspth",
            description="AirKorea fine-dust forecast-dispatch smoke.",
            params={
                "returnType": "json",
                "numOfRows": 10,
                "pageNo": 1,
                "searchDate": search_date,
                "informCode": "PM10",
            },
        ),
        ApiCallSpec(
            service="airkorea",
            function="/getMinuDustWeekFrcstDspth",
            description="AirKorea weekly fine-dust forecast-dispatch smoke.",
            params={
                "returnType": "json",
                "numOfRows": 10,
                "pageNo": 1,
                "searchDate": search_date,
            },
        ),
        ApiCallSpec(
            service="airkorea",
            function="/getUnityAirEnvrnIdexSnstiveAboveMsrstnList",
            description="AirKorea bad-or-worse CAI station-list smoke.",
            params={
                "returnType": "json",
                "numOfRows": 10,
                "pageNo": 1,
            },
        ),
    ]


def service_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_service: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_service.setdefault(row["service"], []).append(row)
    out: dict[str, Any] = {}
    for service, service_rows in by_service.items():
        successful = [row for row in service_rows if row.get("success")]
        historical = [row for row in service_rows if row.get("historical_probe")]
        historical_success = [row for row in historical if row.get("success") and int(row.get("item_count") or 0) > 0]
        out[service] = {
            "calls": int(len(service_rows)),
            "successes": int(len(successful)),
            "historical_probe_calls": int(len(historical)),
            "historical_probe_successes_with_items": int(len(historical_success)),
            "all_success": len(successful) == len(service_rows),
        }
    return out


def infer_competition_feature_usability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {(row["service"], row["function"], row.get("historical_probe", False)): row for row in rows}
    kma_asos = lookup.get(("kma_asos", "/getWthrDataList", True), {})
    kma_life_hist = lookup.get(("kma_life", "/getUVIdxV5", True), {})
    air_station = lookup.get(("airkorea", "/getMsrstnAcctoRltmMesureDnsty", True), {})
    air_realtime = lookup.get(("airkorea", "/getCtprvnRltmMesureDnsty", False), {})
    return {
        "kma_asos_historical_2024_usable": bool(kma_asos.get("success") and int(kma_asos.get("item_count") or 0) > 0),
        "kma_life_historical_2024_usable_from_smoke": bool(
            kma_life_hist.get("success") and int(kma_life_hist.get("item_count") or 0) > 0
        ),
        "airkorea_station_recent_3month_available": bool(
            air_station.get("success") and int(air_station.get("item_count") or 0) > 0
        ),
        "airkorea_realtime_available": bool(
            air_realtime.get("success") and int(air_realtime.get("item_count") or 0) > 0
        ),
        "airkorea_2024_training_window_usable_from_this_api": False,
        "airkorea_2024_reason": (
            "The approved ArpltnInforInqireSvc smoke covers real-time/forecast/recent station terms only. "
            "It does not expose a date-bounded 2024 historical query in these endpoints."
        ),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in call_specs(args):
        rows.append(call_api(spec, args))
        time.sleep(args.sleep_seconds)
    usability = infer_competition_feature_usability(rows)
    decision = {
        "all_three_service_families_requested": ["kma_asos", "kma_life", "airkorea"],
        "service_summary": service_summary(rows),
        "feature_usability": usability,
        "all_three_service_families_smoke_checked": True,
        "kma_asos_2024_historical_feature_source_available": usability["kma_asos_historical_2024_usable"],
        "kma_life_2024_historical_feature_source_available": usability[
            "kma_life_historical_2024_usable_from_smoke"
        ],
        "airkorea_api_access_working": usability["airkorea_realtime_available"]
        or usability["airkorea_station_recent_3month_available"],
        "airkorea_approved_info_api_2024_historical_feature_source_available": usability[
            "airkorea_2024_training_window_usable_from_this_api"
        ],
        "non_asos_approved_api_2024_historical_feature_source_available": bool(
            usability["kma_life_historical_2024_usable_from_smoke"]
            or usability["airkorea_2024_training_window_usable_from_this_api"]
        ),
        "create_model_features_now": False,
        "create_submission_artifact_now": False,
        "modify_submission_log_now": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "dacon_submit_now": False,
        "next_action": (
            "Use KMA ASOS as the only confirmed 2024 historical source from this smoke. Treat KMA living-weather "
            "and AirKorea ArpltnInforInqireSvc as report-only unless a separate historical/statistics source is "
            "approved and passes a local ID342 gate."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "public_data_weather_air_api_smoke",
        "purpose": (
            "Verify the three approved public-data API service families with real calls while keeping keys out of "
            "tracked files."
        ),
        "raw_cache_policy": {
            "keep_raw_cache": bool(args.keep_raw_cache),
            "cache_dir": rel(args.cache_dir) if args.keep_raw_cache else None,
            "default": "Raw API payload caches are not retained; the report keeps only summary metadata.",
        },
        "secret_handling": {
            "env_names": SECRET_ENV_BY_SERVICE,
            "service_key_values_stored": False,
            "cache_files_store_secret_values": False if args.keep_raw_cache else None,
        },
        "calls": rows,
        "decision": decision,
        "no_prediction_file_created": True,
        "no_feature_matrix_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def fmt_bool(value: Any) -> str:
    return "`True`" if bool(value) else "`False`"


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    usability = decision["feature_usability"]
    lines = [
        "# Public-data weather/air API smoke",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no feature matrix, predictions, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- KMA ASOS 2024 historical usable: {fmt_bool(usability['kma_asos_historical_2024_usable'])}",
        f"- KMA living-weather 2024 historical usable from smoke: {fmt_bool(usability['kma_life_historical_2024_usable_from_smoke'])}",
        f"- AirKorea realtime available: {fmt_bool(usability['airkorea_realtime_available'])}",
        f"- AirKorea station recent 3-month available: {fmt_bool(usability['airkorea_station_recent_3month_available'])}",
        f"- AirKorea 2024 train-window usable from approved API: {fmt_bool(usability['airkorea_2024_training_window_usable_from_this_api'])}",
        f"- Raw API cache retained: `{report['raw_cache_policy']['keep_raw_cache']}`",
        f"- Kaggle now: `{decision['run_kaggle_now']}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Service Summary",
        "",
        "| service | calls | successes | historical probes with items | all success |",
        "|---|---:|---:|---:|---|",
    ]
    for service, summary in decision["service_summary"].items():
        lines.append(
            f"| `{service}` | {summary['calls']} | {summary['successes']} | "
            f"{summary['historical_probe_successes_with_items']} | `{summary['all_success']}` |"
        )
    lines.extend(
        [
            "",
            "## Calls",
            "",
            "| service | function | historical | success | result | items | time keys | raw cache |",
            "|---|---|---|---|---|---:|---|---|",
        ]
    )
    for row in report["calls"]:
        time_keys = ", ".join(f"`{key}`" for key in row.get("item_time_values", {}))
        result = row.get("result_msg") or row.get("error") or row.get("parse_error") or ""
        if len(str(result)) > 90:
            result = str(result)[:87] + "..."
        lines.append(
            f"| `{row['service']}` | `{row['function']}` | `{row.get('historical_probe')}` | "
            f"`{row.get('success')}` | {result} | {row.get('item_count', 0)} | {time_keys} | "
            f"`{row.get('raw_cache_written', False)}` |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The service key value is not written to reports or optional cache metadata.",
            "- Raw API payload caches are not retained by default; use `--keep-raw-cache` only for troubleshooting.",
            "- AirKorea `ArpltnInforInqireSvc` smoke checks realtime, forecast, and recent station-query behavior; it does not prove 2024 historical train/test coverage.",
            "- Any future feature gate must still be ID342 report-only first.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--keep-raw-cache", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--allow-insecure-ssl", action="store_true")
    parser.add_argument("--kma-life-area-no", default="3017000000", help="Default Daejeon area code.")
    parser.add_argument("--kma-life-time", default=None, help="Override living-weather issue time YYYYMMDDHH.")
    parser.add_argument("--airkorea-sido-name", default="대전")
    parser.add_argument("--airkorea-station-name", default="둔산동")
    parser.add_argument("--airkorea-search-date", default=None, help="Override forecast search date YYYY-MM-DD.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output = args.output if args.output.is_absolute() else ROOT / args.output
    args.cache_dir = args.cache_dir if args.cache_dir.is_absolute() else ROOT / args.cache_dir
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    args.output.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")
    decision = report["decision"]
    print(f"Wrote {rel(args.output)}")
    print(f"Wrote {rel(args.output.with_suffix('.md'))}")
    print(json.dumps(decision["service_summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
