#!/usr/bin/env python3
"""Fetch KMA ASOS hourly observations into an ignored local cache.

The service key is read only from environment variables. It is never written to
the output CSV or metadata JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

try:
    import certifi
except ImportError:  # pragma: no cover - certifi is available in the project venv.
    certifi = None

ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9))
ENDPOINT = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
DEFAULT_OUTPUT = ROOT / "data" / "external" / "weather" / "kma_asos_hourly_20240601_20241122.csv"
DEFAULT_STATIONS = "133"
SECRET_ENV_NAMES = ("KMA_ASOS_SERVICE_KEY", "PUBLIC_DATA_SERVICE_KEY")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def service_key_from_env() -> tuple[str, str]:
    for name in SECRET_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            return name, value
    names = ", ".join(SECRET_ENV_NAMES)
    raise SystemExit(f"Missing KMA ASOS service key. Set one of: {names}")


def build_url(service_key: str, params: dict[str, Any]) -> str:
    query = urlencode(params)
    return f"{ENDPOINT}?serviceKey={service_key}&{query}"


def ssl_context(args: argparse.Namespace) -> ssl.SSLContext:
    if args.allow_insecure_ssl:
        return ssl._create_unverified_context()  # noqa: S323
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def normalize_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    response = payload.get("response", {})
    header = response.get("header", {})
    result_code = str(header.get("resultCode", ""))
    if result_code and result_code not in {"00", "0"}:
        raise RuntimeError(f"KMA API returned resultCode={result_code} resultMsg={header.get('resultMsg')}")
    body = response.get("body", {})
    items = body.get("items", {}).get("item", [])
    if isinstance(items, dict):
        return [items]
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    return []


def fetch_page(service_key: str, station_id: str, page_no: int, args: argparse.Namespace) -> tuple[list[dict[str, Any]], int]:
    params = {
        "pageNo": page_no,
        "numOfRows": args.num_rows,
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "HR",
        "startDt": args.start_date,
        "startHh": args.start_hour,
        "endDt": args.end_date,
        "endHh": args.end_hour,
        "stnIds": station_id,
    }
    url = build_url(service_key, params)
    try:
        with urlopen(url, timeout=args.timeout_seconds, context=ssl_context(args)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"KMA ASOS request failed for station {station_id} page {page_no}: {exc}") from exc
    response_body = payload.get("response", {}).get("body", {})
    total_count = int(response_body.get("totalCount", 0) or 0)
    return normalize_items(payload), total_count


def fetch_station(service_key: str, station_id: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page_no = 1
    total_count: int | None = None
    while True:
        page_rows, page_total = fetch_page(service_key, station_id, page_no, args)
        if total_count is None:
            total_count = page_total
        rows.extend(page_rows)
        if len(rows) >= (total_count or 0) or not page_rows:
            break
        page_no += 1
        time.sleep(args.sleep_seconds)
    return rows


def write_metadata(output: Path, args: argparse.Namespace, key_env_name: str, row_count: int, stations: list[str]) -> None:
    metadata = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "source": "KMA ASOS hourly observation API",
        "endpoint": ENDPOINT.rsplit("/", 1)[0],
        "function": "/getWthrDataList",
        "service_key_env": key_env_name,
        "service_key_value_stored": False,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "start_hour": args.start_hour,
        "end_hour": args.end_hour,
        "stations": stations,
        "row_count": int(row_count),
        "output_csv": rel(output),
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="20240601")
    parser.add_argument("--end-date", default="20241122")
    parser.add_argument("--start-hour", default="00")
    parser.add_argument("--end-hour", default="23")
    parser.add_argument("--stations", default=DEFAULT_STATIONS, help="Comma-separated ASOS station IDs.")
    parser.add_argument("--num-rows", type=int, default=999)
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-insecure-ssl",
        action="store_true",
        help="Disable TLS verification only for local public-data troubleshooting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    key_env_name, service_key = service_key_from_env()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    stations = [station.strip() for station in args.stations.split(",") if station.strip()]
    if not stations:
        raise SystemExit("At least one station ID is required.")
    all_rows: list[dict[str, Any]] = []
    for station_id in stations:
        station_rows = fetch_station(service_key, station_id, args)
        all_rows.extend(station_rows)
        print(f"Fetched station {station_id}: {len(station_rows)} rows")
    frame = pd.DataFrame(all_rows)
    if not frame.empty and "tm" in frame.columns:
        frame = frame.sort_values(["stnId", "tm"]).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    write_metadata(output, args, key_env_name, len(frame), stations)
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.metadata.json'))}")
    print("Service key was read from the environment and was not written to output files.")


if __name__ == "__main__":
    main()
