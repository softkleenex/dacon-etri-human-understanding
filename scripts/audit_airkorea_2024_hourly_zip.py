#!/usr/bin/env python3
"""Audit downloaded 2024 AirKorea hourly XLSX archive.

The input archive is a user-downloaded public historical air-quality dataset.
This script keeps the raw XLSX files compressed, extracts only a normalized
Daejeon hourly cache under ignored ``data/external/weather/``, and writes a
tracked report. It creates no model features, predictions, submission artifacts,
Kaggle runs, or DACON submissions.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
DEFAULT_ZIP = ROOT / "data" / "external" / "weather" / "airkorea_2024" / (
    "airkorea_2024_hourly_all_stations_xlsx.zip"
)
DEFAULT_DAEJEON_OUTPUT = ROOT / "data" / "external" / "weather" / "airkorea_2024" / (
    "airkorea_2024_daejeon_hourly.csv"
)
DEFAULT_OUTPUT = REPORTS_DIR / "airkorea_2024_hourly_zip_audit_20260625.json"

SPREADSHEET_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

EXPECTED_COLUMNS = ["지역", "망", "측정소코드", "측정소명", "측정일시", "SO2", "CO", "O3", "NO2", "PM10", "PM25", "주소"]
NORMALIZED_COLUMNS = [
    "region",
    "network",
    "station_code",
    "station_name",
    "datetime",
    "so2",
    "co",
    "o3",
    "no2",
    "pm10",
    "pm25",
    "address",
    "source_file",
]
POLLUTANTS = ["so2", "co", "o3", "no2", "pm10", "pm25"]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    value = 0
    for ch in letters:
        value = value * 26 + ord(ch.upper()) - 64
    return value - 1


def read_shared_strings(xlsx: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in xlsx.namelist():
        return []
    root = ET.fromstring(xlsx.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall("a:si", SPREADSHEET_NS):
        strings.append("".join(node.text or "" for node in item.findall(".//a:t", SPREADSHEET_NS)))
    return strings


def sheet_target(xlsx: zipfile.ZipFile) -> tuple[str, str]:
    workbook = ET.fromstring(xlsx.read("xl/workbook.xml"))
    rels = ET.fromstring(xlsx.read("xl/_rels/workbook.xml.rels"))
    relmap = {item.attrib["Id"]: item.attrib["Target"] for item in rels.findall("rel:Relationship", REL_NS)}
    first_sheet = workbook.find("a:sheets/a:sheet", SPREADSHEET_NS)
    if first_sheet is None:
        raise ValueError("Workbook has no worksheet.")
    sheet_name = first_sheet.attrib.get("name", "Sheet1")
    rel_id = first_sheet.attrib.get(f"{{{OFFICE_REL_NS}}}id")
    target = relmap.get(rel_id or "", "worksheets/sheet1.xml")
    if not target.startswith("xl/"):
        target = f"xl/{target}"
    return sheet_name, target


def cell_value(cell: ET.Element, shared_strings: list[str]) -> str | None:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//a:t", SPREADSHEET_NS))
    value_node = cell.find("a:v", SPREADSHEET_NS)
    if value_node is None:
        return None
    value = value_node.text
    if cell_type == "s" and value is not None:
        try:
            return shared_strings[int(value)]
        except (ValueError, IndexError):
            return value
    return value


def iter_sheet_rows(xlsx_bytes: bytes) -> tuple[str, str, list[list[str | None]]]:
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes)) as xlsx:
        shared_strings = read_shared_strings(xlsx)
        sheet_name, sheet_path = sheet_target(xlsx)
        dimension_ref = ""
        with xlsx.open(sheet_path) as stream:
            for _event, element in ET.iterparse(stream, events=("end",)):
                tag = element.tag.rsplit("}", 1)[-1]
                if tag == "dimension":
                    dimension_ref = element.attrib.get("ref", "")
                    element.clear()
                elif tag == "row":
                    max_idx = -1
                    values_by_idx: dict[int, str | None] = {}
                    for cell in element:
                        if not cell.tag.endswith("}c"):
                            continue
                        idx = column_index(cell.attrib.get("r", "A1"))
                        max_idx = max(max_idx, idx)
                        values_by_idx[idx] = cell_value(cell, shared_strings)
                    row = [values_by_idx.get(idx) for idx in range(max_idx + 1)]
                    yield sheet_name, dimension_ref, row
                    element.clear()


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"-", "NA", "nan", "NaN"}:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def parse_measurement_time(value: Any) -> datetime | None:
    if value is None:
        return None
    text = re.sub(r"\D", "", str(value))
    if len(text) < 10:
        return None
    date_text = text[:8]
    hour_text = text[8:10]
    try:
        base = datetime.strptime(date_text, "%Y%m%d")
        hour = int(hour_text)
    except ValueError:
        return None
    if hour == 24:
        return base + timedelta(days=1)
    if 0 <= hour <= 23:
        return base + timedelta(hours=hour)
    return None


def month_from_name(name: str) -> int | None:
    match = re.search(r"(\d{1,2})월", name)
    if not match:
        return None
    month = int(match.group(1))
    return month if 1 <= month <= 12 else None


def is_daejeon(row: list[str | None]) -> bool:
    region = str(row[0] or "")
    address = str(row[11] or "") if len(row) > 11 else ""
    return region.startswith("대전") or address.startswith("대전")


def normalize_daejeon_row(row: list[str | None], source_file: str) -> dict[str, Any] | None:
    if len(row) < len(EXPECTED_COLUMNS):
        row = [*row, *([None] * (len(EXPECTED_COLUMNS) - len(row)))]
    measured_at = parse_measurement_time(row[4])
    if measured_at is None:
        return None
    values = [parse_float(row[idx]) for idx in range(5, 11)]
    return {
        "region": row[0],
        "network": row[1],
        "station_code": row[2],
        "station_name": row[3],
        "datetime": measured_at.strftime("%Y-%m-%d %H:%M:%S"),
        "so2": values[0],
        "co": values[1],
        "o3": values[2],
        "no2": values[3],
        "pm10": values[4],
        "pm25": values[5],
        "address": row[11],
        "source_file": source_file,
    }


def numeric_summary(frame: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    out: dict[str, dict[str, float | int | None]] = {}
    for col in POLLUTANTS:
        values = pd.to_numeric(frame[col], errors="coerce")
        out[col] = {
            "non_null": int(values.notna().sum()),
            "missing": int(values.isna().sum()),
            "mean": None if values.notna().sum() == 0 else float(values.mean()),
            "p05": None if values.notna().sum() == 0 else float(values.quantile(0.05)),
            "p50": None if values.notna().sum() == 0 else float(values.quantile(0.50)),
            "p95": None if values.notna().sum() == 0 else float(values.quantile(0.95)),
        }
    return out


def station_coverage(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[dict[str, Any]]:
    window = frame[(frame["datetime"] >= start) & (frame["datetime"] <= end)].copy()
    expected_hours = int(((end - start).total_seconds() // 3600) + 1)
    rows: list[dict[str, Any]] = []
    for (station_code, station_name), group in window.groupby(["station_code", "station_name"], dropna=False):
        unique_hours = int(group["datetime"].nunique())
        rows.append(
            {
                "station_code": str(station_code),
                "station_name": str(station_name),
                "rows": int(len(group)),
                "unique_hours": unique_hours,
                "expected_hours": expected_hours,
                "coverage_ratio": float(unique_hours / expected_hours) if expected_hours else None,
                "address": str(group["address"].dropna().iloc[0]) if group["address"].notna().any() else "",
            }
        )
    return sorted(rows, key=lambda item: (-item["unique_hours"], item["station_code"]))


def audit_archive(args: argparse.Namespace) -> dict[str, Any]:
    zip_path = args.zip_path if args.zip_path.is_absolute() else ROOT / args.zip_path
    daejeon_output = args.daejeon_output if args.daejeon_output.is_absolute() else ROOT / args.daejeon_output
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing AirKorea archive: {rel(zip_path)}")

    monthly_rows: list[dict[str, Any]] = []
    total_data_rows = 0
    daejeon_rows = 0
    header_counter: Counter[tuple[str | None, ...]] = Counter()
    network_counter: Counter[str] = Counter()
    daejeon_station_counter: Counter[tuple[str, str]] = Counter()
    daejeon_time_min: datetime | None = None
    daejeon_time_max: datetime | None = None

    daejeon_output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as outer, daejeon_output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=NORMALIZED_COLUMNS)
        writer.writeheader()
        xlsx_names = sorted([name for name in outer.namelist() if name.lower().endswith(".xlsx")])
        for xlsx_name in xlsx_names:
            file_info = outer.getinfo(xlsx_name)
            month = month_from_name(xlsx_name)
            row_count = 0
            file_daejeon_rows = 0
            file_time_min: datetime | None = None
            file_time_max: datetime | None = None
            header: list[str | None] | None = None
            sheet_name = ""
            dimension_ref = ""
            xlsx_bytes = outer.read(xlsx_name)
            for row_sheet_name, row_dimension_ref, row in iter_sheet_rows(xlsx_bytes):
                sheet_name = row_sheet_name
                dimension_ref = row_dimension_ref
                if header is None:
                    header = row[: len(EXPECTED_COLUMNS)]
                    header_counter[tuple(header)] += 1
                    continue
                if len(row) < 5:
                    continue
                row_count += 1
                total_data_rows += 1
                measured_at = parse_measurement_time(row[4])
                if measured_at is not None:
                    file_time_min = measured_at if file_time_min is None else min(file_time_min, measured_at)
                    file_time_max = measured_at if file_time_max is None else max(file_time_max, measured_at)
                if is_daejeon(row):
                    normalized = normalize_daejeon_row(row, xlsx_name)
                    if normalized is None:
                        continue
                    writer.writerow(normalized)
                    file_daejeon_rows += 1
                    daejeon_rows += 1
                    measured = datetime.strptime(normalized["datetime"], "%Y-%m-%d %H:%M:%S")
                    daejeon_time_min = measured if daejeon_time_min is None else min(daejeon_time_min, measured)
                    daejeon_time_max = measured if daejeon_time_max is None else max(daejeon_time_max, measured)
                    network_counter[str(normalized["network"])] += 1
                    daejeon_station_counter[(str(normalized["station_code"]), str(normalized["station_name"]))] += 1
            monthly_rows.append(
                {
                    "source_file": xlsx_name,
                    "month": month,
                    "bytes": int(file_info.file_size),
                    "sheet_name": sheet_name,
                    "dimension": dimension_ref,
                    "header": header,
                    "data_rows": int(row_count),
                    "daejeon_rows": int(file_daejeon_rows),
                    "time_min": None if file_time_min is None else file_time_min.strftime("%Y-%m-%d %H:%M:%S"),
                    "time_max": None if file_time_max is None else file_time_max.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

    daejeon = pd.read_csv(daejeon_output, parse_dates=["datetime"])
    daejeon = daejeon.sort_values(["station_code", "datetime"]).reset_index(drop=True)
    daejeon.to_csv(daejeon_output, index=False)
    window_start = pd.Timestamp(args.window_start)
    window_end = pd.Timestamp(args.window_end)
    window = daejeon[(daejeon["datetime"] >= window_start) & (daejeon["datetime"] <= window_end)].copy()
    expected_hours = int(((window_end - window_start).total_seconds() // 3600) + 1)
    station_rows = station_coverage(daejeon, window_start, window_end)
    coverage_ok = bool(
        len(window) > 0
        and window["datetime"].nunique() == expected_hours
        and len(station_rows) >= 5
        and max((row["coverage_ratio"] or 0.0 for row in station_rows), default=0.0) >= 0.95
    )

    decision = {
        "airkorea_2024_zip_available": True,
        "zip_moved_to_ignored_external_weather": str(zip_path).startswith(str(ROOT / "data" / "external" / "weather")),
        "normalized_daejeon_hourly_cache_created": True,
        "competition_window_coverage_ok": coverage_ok,
        "create_feature_matrix_now": False,
        "create_submission_artifact_now": False,
        "modify_submission_log_now": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "dacon_submit_now": False,
        "next_action": (
            "Run a report-only ID342 gate over Daejeon AirKorea hourly/daily features. Do not create submission "
            "artifacts unless that gate clears the Public-0.58 review proxy and failed-frontier checks."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "airkorea_2024_hourly_zip_audit",
        "purpose": "Audit user-downloaded 2024 AirKorea hourly XLSX archive and normalize Daejeon rows.",
        "input_zip": rel(zip_path),
        "input_zip_size_bytes": int(zip_path.stat().st_size),
        "normalized_daejeon_csv": rel(daejeon_output),
        "monthly_files": monthly_rows,
        "schema": {
            "expected_columns": EXPECTED_COLUMNS,
            "headers_seen": {json.dumps(list(key), ensure_ascii=False): count for key, count in header_counter.items()},
        },
        "row_counts": {
            "total_national_hourly_rows": int(total_data_rows),
            "daejeon_hourly_rows": int(daejeon_rows),
            "daejeon_unique_stations": int(daejeon[["station_code", "station_name"]].drop_duplicates().shape[0]),
            "daejeon_unique_timestamps": int(daejeon["datetime"].nunique()),
            "daejeon_time_min": None if daejeon_time_min is None else daejeon_time_min.strftime("%Y-%m-%d %H:%M:%S"),
            "daejeon_time_max": None if daejeon_time_max is None else daejeon_time_max.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "competition_window": {
            "start": window_start.strftime("%Y-%m-%d %H:%M:%S"),
            "end": window_end.strftime("%Y-%m-%d %H:%M:%S"),
            "expected_hours": expected_hours,
            "daejeon_rows": int(len(window)),
            "daejeon_unique_timestamps": int(window["datetime"].nunique()),
            "daejeon_unique_stations": int(window[["station_code", "station_name"]].drop_duplicates().shape[0]),
            "pollutant_summary": numeric_summary(window),
            "station_coverage": station_rows,
        },
        "daejeon_network_counts": dict(network_counter),
        "daejeon_station_counts_top": [
            {"station_code": code, "station_name": name, "rows": int(rows)}
            for (code, name), rows in daejeon_station_counter.most_common(30)
        ],
        "decision": decision,
        "no_prediction_file_created": True,
        "no_feature_matrix_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    window = report["competition_window"]

    def fmt_stat(value: float | int | None) -> str:
        return "" if value is None else f"{float(value):.6f}"

    lines = [
        "# AirKorea 2024 hourly archive audit",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates a normalized ignored Daejeon air-quality cache, but no model feature matrix, predictions, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- Archive available: `{decision['airkorea_2024_zip_available']}`",
        f"- Normalized Daejeon cache created: `{decision['normalized_daejeon_hourly_cache_created']}`",
        f"- Competition-window coverage OK: `{decision['competition_window_coverage_ok']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Inputs",
        "",
        f"- Input zip: `{report['input_zip']}`",
        f"- Normalized Daejeon CSV: `{report['normalized_daejeon_csv']}`",
        f"- Zip size bytes: `{report['input_zip_size_bytes']}`",
        "",
        "## Schema",
        "",
        f"- Expected columns: `{report['schema']['expected_columns']}`",
        "",
        "## Row Counts",
        "",
        f"- Total national hourly rows: `{report['row_counts']['total_national_hourly_rows']}`",
        f"- Daejeon hourly rows: `{report['row_counts']['daejeon_hourly_rows']}`",
        f"- Daejeon stations: `{report['row_counts']['daejeon_unique_stations']}`",
        f"- Daejeon time range: `{report['row_counts']['daejeon_time_min']}` to `{report['row_counts']['daejeon_time_max']}`",
        "",
        "## Competition Window",
        "",
        f"- Window: `{window['start']}` to `{window['end']}`",
        f"- Expected hours: `{window['expected_hours']}`",
        f"- Daejeon rows: `{window['daejeon_rows']}`",
        f"- Unique timestamps: `{window['daejeon_unique_timestamps']}`",
        f"- Unique stations: `{window['daejeon_unique_stations']}`",
        "",
        "## Pollutant Summary",
        "",
        "| pollutant | non-null | missing | mean | p05 | p50 | p95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for pollutant, stats in window["pollutant_summary"].items():
        lines.append(
            f"| `{pollutant}` | {stats['non_null']} | {stats['missing']} | "
            f"{fmt_stat(stats['mean'])} | {fmt_stat(stats['p05'])} | "
            f"{fmt_stat(stats['p50'])} | {fmt_stat(stats['p95'])} |"
        )
    lines.extend(
        [
            "",
            "## Station Coverage",
            "",
            "| station | name | rows | unique hours | coverage | address |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in window["station_coverage"][:30]:
        lines.append(
            f"| `{row['station_code']}` | `{row['station_name']}` | {row['rows']} | "
            f"{row['unique_hours']} | {row['coverage_ratio']:.6f} | {row['address']} |"
        )
    lines.extend(
        [
            "",
            "## Monthly Files",
            "",
            "| month | file | rows | Daejeon rows | time min | time max |",
            "|---:|---|---:|---:|---|---|",
        ]
    )
    for row in sorted(report["monthly_files"], key=lambda item: item.get("month") or 99):
        lines.append(
            f"| {row['month']} | `{row['source_file']}` | {row['data_rows']} | {row['daejeon_rows']} | "
            f"`{row['time_min']}` | `{row['time_max']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--daejeon-output", type=Path, default=DEFAULT_DAEJEON_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window-start", default="2024-06-01 00:00:00")
    parser.add_argument("--window-end", default="2024-11-22 23:00:00")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = audit_archive(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(f"Wrote {report['normalized_daejeon_csv']}")
    print(
        "Daejeon rows/window/coverage: "
        f"{report['row_counts']['daejeon_hourly_rows']} / "
        f"{report['competition_window']['daejeon_rows']} / "
        f"{report['decision']['competition_window_coverage_ok']}"
    )


if __name__ == "__main__":
    main()
