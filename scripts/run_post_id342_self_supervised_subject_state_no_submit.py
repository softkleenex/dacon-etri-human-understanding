#!/usr/bin/env python3
"""No-submit post-ID342 subject-state self-supervised runner.

This script is the reusable local/Kaggle runner for the approved non-DACON
workflow. It reads raw lifelog rows in bounded streaming batches, builds a
small label-free subject-state token smoke, and trains a tiny masked
reconstruction model only to verify that the route is executable.

It never creates submission CSVs, base prediction artifacts, submission-log
rows, or DACON submissions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))

RAW_MODALITIES = {
    "mACStatus": ROOT / "data/raw/ch2025_data_items/ch2025_mACStatus.parquet",
    "mActivity": ROOT / "data/raw/ch2025_data_items/ch2025_mActivity.parquet",
    "mAmbience": ROOT / "data/raw/ch2025_data_items/ch2025_mAmbience.parquet",
    "mBle": ROOT / "data/raw/ch2025_data_items/ch2025_mBle.parquet",
    "mGps": ROOT / "data/raw/ch2025_data_items/ch2025_mGps.parquet",
    "mLight": ROOT / "data/raw/ch2025_data_items/ch2025_mLight.parquet",
    "mScreenStatus": ROOT / "data/raw/ch2025_data_items/ch2025_mScreenStatus.parquet",
    "mUsageStats": ROOT / "data/raw/ch2025_data_items/ch2025_mUsageStats.parquet",
    "mWifi": ROOT / "data/raw/ch2025_data_items/ch2025_mWifi.parquet",
    "wHr": ROOT / "data/raw/ch2025_data_items/ch2025_wHr.parquet",
    "wLight": ROOT / "data/raw/ch2025_data_items/ch2025_wLight.parquet",
    "wPedo": ROOT / "data/raw/ch2025_data_items/ch2025_wPedo.parquet",
}

DEFAULT_LOCAL_OUTPUT = REPORTS_DIR / "post_id342_self_supervised_subject_state_local_smoke_20260624.json"
REMOTE_OUTPUT_NAME = "remote_post_id342_self_supervised_subject_state_no_submit_smoke_20260624.json"

FORBIDDEN_OUTPUT_HINTS = [
    "submission",
    "submission_log",
    "base_oof",
    "base_test",
    "candidate",
]


@dataclass
class TokenAccumulator:
    count: int = 0
    value_sum: float = 0.0
    value_sq_sum: float = 0.0
    payload_count_sum: float = 0.0

    def add(self, value: float, payload_count: float) -> None:
        self.count += 1
        self.value_sum += value
        self.value_sq_sum += value * value
        self.payload_count_sum += payload_count

    def vector(self) -> list[float]:
        if self.count == 0:
            return [0.0, 0.0, 0.0]
        mean = self.value_sum / self.count
        var = max(self.value_sq_sum / self.count - mean * mean, 0.0)
        return [math.log1p(self.count), mean, math.sqrt(var + 1e-9) + math.log1p(self.payload_count_sum)]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def output_path_for_mode(mode: str, output: Path | None) -> Path:
    if output is not None:
        return output if output.is_absolute() else ROOT / output
    if mode == "remote_no_submit":
        output_dir = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working/kaggle_outputs"))
        return output_dir / REMOTE_OUTPUT_NAME
    return DEFAULT_LOCAL_OUTPUT


def list_payload_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, np.ndarray):
        return int(value.size)
    if isinstance(value, (list, tuple)):
        return len(value)
    return 1


def as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def numeric_from_nested(value: Any, modality: str) -> tuple[float, float]:
    if value is None:
        return 0.0, 0.0
    if not isinstance(value, (list, tuple, np.ndarray)):
        try:
            return float(value), 1.0
        except (TypeError, ValueError):
            return 0.0, 0.0
    value = as_sequence(value)
    if not value:
        return 0.0, 0.0
    if modality == "mUsageStats":
        total = 0.0
        count = 0
        for item in value:
            if isinstance(item, dict):
                total += float(item.get("total_time") or 0.0)
                count += 1
        return math.log1p(total), float(count)
    if modality in {"mBle", "mWifi"}:
        rssi_values = [float(item.get("rssi")) for item in value if isinstance(item, dict) and item.get("rssi") is not None]
        return (float(np.mean(rssi_values)) if rssi_values else 0.0), float(len(value))
    if modality == "mGps":
        speeds = [float(item.get("speed")) for item in value if isinstance(item, dict) and item.get("speed") is not None]
        return (float(np.mean(speeds)) if speeds else 0.0), float(len(value))
    if modality == "mAmbience":
        scores = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    scores.append(float(item[1]))
                except (TypeError, ValueError):
                    continue
        return (max(scores) if scores else 0.0), float(len(value))
    if modality == "wHr":
        values = [float(item) for item in value if item is not None]
        return (float(np.mean(values)) if values else 0.0), float(len(values))
    return float(len(value)), float(len(value))


def row_signal(row: pd.Series, modality: str) -> tuple[float, float]:
    if modality == "wPedo":
        numeric_cols = ["step", "step_frequency", "running_step", "walking_step", "distance", "speed", "burned_calories"]
        values = [float(row[col]) for col in numeric_cols if col in row and pd.notna(row[col])]
        return (float(np.mean(values)) if values else 0.0), float(len(values))
    value_columns = [col for col in row.index if col not in {"subject_id", "timestamp"}]
    if not value_columns:
        return 0.0, 0.0
    value = row[value_columns[0]]
    if isinstance(value, (list, tuple, np.ndarray)):
        return numeric_from_nested(value, modality)
    if pd.isna(value):
        return 0.0, 0.0
    return float(value), list_payload_count(value)


def update_nested_summary(summary: dict[str, Any], value: Any, modality: str) -> None:
    if value is None:
        return
    payload_count = list_payload_count(value)
    summary["nested_payload_count_sum"] += payload_count
    summary["nested_payload_count_max"] = max(summary["nested_payload_count_max"], payload_count)
    value = as_sequence(value)
    if modality == "mAmbience" and value:
        for item in value[:3]:
            if isinstance(item, (list, tuple)) and item:
                summary["top_labels"].update([str(item[0])])
    elif modality == "mUsageStats" and value:
        for item in value[:5]:
            if isinstance(item, dict):
                app_name = str(item.get("app_name", ""))
                summary["hashed_name_buckets"].update([hash(app_name) % 997])
    elif modality in {"mBle", "mWifi"} and value:
        classes = [str(item.get("device_class")) for item in value[:5] if isinstance(item, dict) and "device_class" in item]
        summary["top_labels"].update(classes)


def read_modality_stream(
    modality: str,
    path: Path,
    *,
    max_rows: int,
    batch_size: int,
    token_limit: int,
    tokens: dict[tuple[str, str, int, str], TokenAccumulator],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "modality": modality,
        "path": rel(path),
        "exists": path.exists(),
        "rows_read": 0,
        "batches_read": 0,
        "payload_rows_read": 0,
        "subject_count": 0,
        "date_count": 0,
        "timestamp_min": None,
        "timestamp_max": None,
        "nested_payload_count_sum": 0,
        "nested_payload_count_max": 0,
        "top_labels": Counter(),
        "hashed_name_buckets": Counter(),
        "error": None,
    }
    if not path.exists():
        summary["error"] = "missing"
        return summary
    subjects: set[str] = set()
    dates: set[str] = set()
    rows_remaining = max_rows
    try:
        scanner = ds.dataset(path, format="parquet").scanner(batch_size=batch_size)
        for batch in scanner.to_batches():
            if rows_remaining <= 0:
                break
            take = min(batch.num_rows, rows_remaining)
            if take <= 0:
                break
            frame = batch.slice(0, take).to_pandas()
            rows_remaining -= take
            summary["rows_read"] += len(frame)
            summary["payload_rows_read"] += len(frame)
            summary["batches_read"] += 1
            if "subject_id" in frame:
                subjects.update(str(value) for value in frame["subject_id"].dropna().unique())
            if "timestamp" in frame:
                timestamp = pd.to_datetime(frame["timestamp"], errors="coerce")
                if timestamp.notna().any():
                    current_min = timestamp.min()
                    current_max = timestamp.max()
                    min_value = summary["timestamp_min"]
                    max_value = summary["timestamp_max"]
                    summary["timestamp_min"] = str(current_min) if min_value is None else str(min(pd.Timestamp(min_value), current_min))
                    summary["timestamp_max"] = str(current_max) if max_value is None else str(max(pd.Timestamp(max_value), current_max))
                    dates.update(str(value.date()) for value in timestamp.dropna().unique())
            value_columns = [col for col in frame.columns if col not in {"subject_id", "timestamp"}]
            for _, row in frame.iterrows():
                timestamp = pd.to_datetime(row.get("timestamp"), errors="coerce")
                subject = row.get("subject_id")
                if pd.isna(timestamp) or subject is None:
                    continue
                value, payload_count = row_signal(row, modality)
                key = (str(subject), str(timestamp.date()), int(timestamp.hour), modality)
                if len(tokens) < token_limit or key in tokens:
                    tokens[key].add(float(value), float(payload_count))
                if value_columns:
                    update_nested_summary(summary, row[value_columns[0]], modality)
    except Exception as exc:  # noqa: BLE001
        summary["error"] = f"{type(exc).__name__}: {exc}"
    summary["subject_count"] = len(subjects)
    summary["date_count"] = len(dates)
    summary["top_labels"] = dict(summary["top_labels"].most_common(10))
    summary["hashed_name_buckets"] = dict(summary["hashed_name_buckets"].most_common(10))
    return summary


class TinyMaskedAutoencoder(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        hidden = max(4, min(16, in_dim * 2))
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def token_matrix(tokens: dict[tuple[str, str, int, str], TokenAccumulator], limit: int, seed: int) -> np.ndarray:
    keys = sorted(tokens)
    rng = random.Random(seed)
    if len(keys) > limit:
        keys = rng.sample(keys, limit)
    matrix = np.array([tokens[key].vector() for key in keys], dtype=np.float32)
    if matrix.size == 0:
        return matrix.reshape(0, 3)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True) + 1e-6
    return (matrix - mean) / std


def masked_reconstruction_smoke(matrix: np.ndarray, *, epochs: int, seed: int) -> dict[str, Any]:
    if matrix.shape[0] < 8:
        return {"trained": False, "reason": "not_enough_tokens", "token_count": int(matrix.shape[0])}
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = torch.tensor(matrix, dtype=torch.float32)
    model = TinyMaskedAutoencoder(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses: list[float] = []
    for _epoch in range(epochs):
        mask = torch.rand_like(x) < 0.25
        x_masked = x.clone()
        x_masked[mask] = 0.0
        pred = model(x_masked)
        loss = ((pred - x) ** 2)[mask].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {
        "trained": True,
        "persistent_model_artifact_created": False,
        "token_count": int(matrix.shape[0]),
        "feature_dim": int(matrix.shape[1]),
        "epochs": epochs,
        "losses": losses,
        "final_masked_mse": losses[-1] if losses else None,
    }


def output_snapshot(output: Path) -> dict[str, Any]:
    files = [output, output.with_suffix(".md")]
    suspicious = []
    for path in files:
        lower = path.name.lower()
        if path.suffix.lower() == ".csv" or any(hint in lower for hint in FORBIDDEN_OUTPUT_HINTS):
            suspicious.append(str(path))
    return {
        "output_dir": str(output.parent),
        "file_count": len(files),
        "files": [str(path) for path in files],
        "suspicious_submission_like_files": suspicious,
    }


def build_report(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.mode == "local_smoke":
        max_rows = args.max_rows_per_modality if args.max_rows_per_modality is not None else 32768
        epochs = args.epochs if args.epochs is not None else 2
    else:
        max_rows = args.max_rows_per_modality if args.max_rows_per_modality is not None else 200000
        epochs = args.epochs if args.epochs is not None else 3

    tokens: dict[tuple[str, str, int, str], TokenAccumulator] = defaultdict(TokenAccumulator)
    modality_rows = [
        read_modality_stream(
            modality,
            path,
            max_rows=max_rows,
            batch_size=args.batch_size,
            token_limit=args.token_limit,
            tokens=tokens,
        )
        for modality, path in RAW_MODALITIES.items()
    ]
    matrix = token_matrix(tokens, args.token_limit, seed)
    reconstruction = masked_reconstruction_smoke(matrix, epochs=epochs, seed=seed)
    snapshot = output_snapshot(output)
    smoke_pass = (
        all(row.get("error") is None for row in modality_rows)
        and all(row.get("rows_read", 0) > 0 for row in modality_rows)
        and reconstruction.get("trained") is True
        and not snapshot["suspicious_submission_like_files"]
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_self_supervised_subject_state_no_submit",
        "mode": args.mode,
        "purpose": "No-submit label-free raw-stream tokenization and masked reconstruction smoke.",
        "config": {
            "max_rows_per_modality": max_rows,
            "batch_size": args.batch_size,
            "token_limit": args.token_limit,
            "seed": seed,
            "epochs": epochs,
        },
        "modality_rows": modality_rows,
        "token_summary": {
            "token_count": int(matrix.shape[0]),
            "feature_dim": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
            "subject_date_hour_modality_tokens_seen": len(tokens),
        },
        "masked_reconstruction_smoke": reconstruction,
        "output_snapshot_before_write": snapshot,
        "decision": {
            "no_submit_runner_smoke_pass": smoke_pass,
            "payload_rows_read": int(sum(row.get("payload_rows_read", 0) for row in modality_rows)),
            "label_free": True,
            "persistent_model_artifact_created": False,
            "create_submission_artifact": False,
            "modify_submission_log": False,
            "dacon_submit": False,
            "next_action": (
                "Local/remote no-submit smoke passed; review reports before any larger embedding run."
                if smoke_pass
                else "Fix modality streaming or masked reconstruction before any Kaggle no-submit escalation."
            ),
        },
        "no_submission_created": True,
        "no_submission_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_dacon_submission_made": True,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision = report["decision"]
    lines = [
        "# Post-ID342 self-supervised subject-state no-submit smoke",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This runner reads bounded raw payload batches and performs a tiny label-free masked reconstruction smoke. It creates no submission CSV, base prediction artifact, submission-log row, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Mode: `{report['mode']}`",
        f"- Smoke pass: `{decision['no_submit_runner_smoke_pass']}`",
        f"- Payload rows read: `{decision['payload_rows_read']}`",
        f"- Label-free: `{decision['label_free']}`",
        f"- Persistent model artifact created: `{decision['persistent_model_artifact_created']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Modify submission log: `{decision['modify_submission_log']}`",
        f"- DACON submit: `{decision['dacon_submit']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Token Summary",
        "",
        "```json",
        json.dumps(report["token_summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Masked Reconstruction Smoke",
        "",
        "```json",
        json.dumps(report["masked_reconstruction_smoke"], indent=2, sort_keys=True),
        "```",
        "",
        "## Modality Rows",
        "",
        "| modality | rows read | batches | subjects | dates | payload max | error |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["modality_rows"]:
        lines.append(
            f"| `{row['modality']}` | `{row['rows_read']}` | `{row['batches_read']}` | "
            f"`{row['subject_count']}` | `{row['date_count']}` | `{row['nested_payload_count_max']}` | "
            f"`{row['error']}` |"
        )
    lines.extend(
        [
            "",
            "## Output Snapshot Before Write",
            "",
            "```json",
            json.dumps(report["output_snapshot_before_write"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["local_smoke", "remote_no_submit"], default="local_smoke")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-rows-per-modality", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--token-limit", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=45502)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = output_path_for_mode(args.mode, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(args, output)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, output)
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(f"Smoke pass: {report['decision']['no_submit_runner_smoke_pass']}")
    print(f"Create submission artifact: {report['decision']['create_submission_artifact']}")
    print(f"DACON submit: {report['decision']['dacon_submit']}")


if __name__ == "__main__":
    main()
