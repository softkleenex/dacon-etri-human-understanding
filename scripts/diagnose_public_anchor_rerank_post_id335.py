#!/usr/bin/env python3
"""Re-rank existing base-test artifacts after ID335.

This is a no-model, no-submission diagnostic. It refreshes the candidate map
using known public anchors ID178/ID329, failed direction ID287, and the current
internal no-submit reference ID335.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
BASE_DIR = ROOT / "artifacts" / "base_predictions"
LOG_PATH = ROOT / "submissions" / "submission_log.csv"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
KST = timezone(timedelta(hours=9))
PUBLIC_ID178 = 0.6014113009
PUBLIC_ID329 = 0.6017908577
PUBLIC_ID287 = 0.6698787318


def extract_id(path: Path) -> int | None:
    match = re.search(r"base_test_id_(\d+)_", path.name)
    return int(match.group(1)) if match else None


def find_base_test_files() -> dict[int, Path]:
    out: dict[int, Path] = {}
    for path in sorted(BASE_DIR.glob("base_test_id_*.csv")):
        candidate_id = extract_id(path)
        if candidate_id is not None and candidate_id not in out:
            out[candidate_id] = path
    return out


def load_log() -> pd.DataFrame:
    log = pd.read_csv(LOG_PATH)
    log["id_num"] = pd.to_numeric(log["id"], errors="coerce").astype("Int64")
    return log


def log_row(log: pd.DataFrame, candidate_id: int) -> dict[str, Any]:
    rows = log[log["id_num"].eq(candidate_id)]
    if rows.empty:
        return {}
    row = rows.iloc[-1].to_dict()
    for key in ["local_cv_score", "public_lb_score", "local_cv_score_num", "public_lb_score_num"]:
        row[key] = pd.to_numeric(row.get(key), errors="coerce")
        row[key] = None if pd.isna(row[key]) else float(row[key])
    return row


def family_for(candidate_id: int, submission_file: str, base_file: str) -> str:
    text = f"{submission_file} {base_file}".lower()
    if candidate_id in {173, 177, 178, 334, 335} or "sequence" in text or "random_conv" in text or "randomconv" in text:
        return "sequence_temporal_family"
    if "post328_combo" in text or "failed_direction" in text or "orthogonal" in text or "antivector" in text:
        return "failed_direction_combo_family"
    if "anchor" in text or "bridge" in text or "tempered" in text:
        return "anchor_bridge_family"
    if "calibration" in text or "isotonic" in text or "q40" in text or "q80" in text or "q160" in text:
        return "calibration_frontier_family"
    if "strict" in text or "compact" in text or "feature" in text:
        return "strict_feature_family"
    return "other_family"


def load_pred(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    return df[TARGETS].astype(float).to_numpy()


def projection(vec: np.ndarray, direction: np.ndarray) -> float:
    denom = float(np.dot(direction, direction))
    return float(np.dot(vec, direction) / denom) if denom > 0 else 0.0


def conservative_proxy(row: dict[str, Any]) -> float:
    """Screening heuristic only; not a calibrated public-score prediction."""
    base = min(PUBLIC_ID178, PUBLIC_ID329)
    movement = 0.22 * row["mean_delta_vs_178"] + 0.025 * row["max_delta_vs_178"]
    neutral = 0.08 * row["mean_delta_vs_329"] + 0.010 * row["max_delta_vs_329"]
    failed = 0.18 * max(0.0, row["projection_failed"])
    low_local_pressure = max(0.0, 0.538638 - float(row["local_cv"] or 0.538638))
    local_penalty = 0.25 * max(0.0, low_local_pressure - 0.02)
    return float(base + movement + neutral + failed + local_penalty)


def risk_tier(row: dict[str, Any]) -> str:
    candidate_id = int(row["id"])
    if candidate_id in {329, 330, 331, 332, 333}:
        return "same_family_freeze_after_id329"
    if candidate_id in {334, 335}:
        return "prepared_sequence_probe_reference"
    if row["public_lb"] is not None:
        return "known_public_reference"
    if row["projection_failed"] > 0.06 or row["max_delta_vs_178"] > 0.18 or row["max_delta_vs_329"] > 0.22:
        return "reject_high_public_risk"
    if row["max_delta_vs_178"] <= 0.12 and row["max_delta_vs_329"] <= 0.17 and row["projection_failed"] <= 0.02:
        return "reviewable_low_movement"
    return "diagnostic_only"


def row_for_candidate(
    candidate_id: int,
    path: Path,
    log: pd.DataFrame,
    arr178: np.ndarray,
    arr287: np.ndarray,
    arr329: np.ndarray,
    arr335: np.ndarray,
) -> dict[str, Any]:
    info = log_row(log, candidate_id)
    arr = load_pred(path)
    failed_direction = (arr287 - arr178).reshape(-1)
    neutral_direction = (arr329 - arr178).reshape(-1)
    current_direction = (arr335 - arr178).reshape(-1)
    vec178 = (arr - arr178).reshape(-1)
    local_cv = info.get("local_cv_score_num", info.get("local_cv_score"))
    public_lb = info.get("public_lb_score_num", info.get("public_lb_score"))
    public_lb = None if public_lb is None or pd.isna(public_lb) else float(public_lb)
    row = {
        "id": candidate_id,
        "submission_file": info.get("submission_file", ""),
        "base_test_file": str(path.relative_to(ROOT)),
        "family": family_for(candidate_id, str(info.get("submission_file", "")), path.name),
        "local_cv": None if local_cv is None or pd.isna(local_cv) else float(local_cv),
        "public_lb": public_lb,
        "status_clean": info.get("status_clean", ""),
        "mean_delta_vs_178": float(np.mean(np.abs(arr - arr178))),
        "max_delta_vs_178": float(np.max(np.abs(arr - arr178))),
        "rmse_vs_178": float(np.sqrt(np.mean((arr - arr178) ** 2))),
        "mean_delta_vs_329": float(np.mean(np.abs(arr - arr329))),
        "max_delta_vs_329": float(np.max(np.abs(arr - arr329))),
        "mean_delta_vs_335": float(np.mean(np.abs(arr - arr335))),
        "max_delta_vs_335": float(np.max(np.abs(arr - arr335))),
        "projection_failed": projection(vec178, failed_direction),
        "projection_329": projection(vec178, neutral_direction),
        "projection_335": projection(vec178, current_direction),
    }
    row["conservative_public_risk_proxy"] = conservative_proxy(row)
    row["risk_tier"] = risk_tier(row)
    return row


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Post-ID335 public-anchor rerank",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is a no-submission diagnostic. The proxy is a screening heuristic, not a leaderboard prediction.",
        "",
        "## Tier counts",
        "",
    ]
    for tier, count in report["tier_counts"].items():
        lines.append(f"- `{tier}`: {count}")
    lines.extend(
        [
            "",
            "## Top reviewable low-movement candidates",
            "",
            "| rank | id | local | proxy | family | mean Δ178 | max Δ178 | proj failed | status | file |",
            "|---:|---:|---:|---:|---|---:|---:|---:|---|---|",
        ]
    )
    for rank, row in enumerate(report["top_reviewable_low_movement"], start=1):
        lines.append(
            f"| {rank} | {row['id']} | {row['local_cv'] if row['local_cv'] is not None else ''} | "
            f"{row['conservative_public_risk_proxy']:.6f} | `{row['family']}` | "
            f"{row['mean_delta_vs_178']:.6f} | {row['max_delta_vs_178']:.6f} | "
            f"{row['projection_failed']:+.4f} | `{row['status_clean']}` | `{row['submission_file']}` |"
        )
    lines.extend(
        [
            "",
            "## Prepared sequence references",
            "",
            "| id | local | proxy | mean Δ178 | max Δ178 | proj failed | file |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["prepared_sequence_references"]:
        lines.append(
            f"| {row['id']} | {row['local_cv']:.6f} | {row['conservative_public_risk_proxy']:.6f} | "
            f"{row['mean_delta_vs_178']:.6f} | {row['max_delta_vs_178']:.6f} | "
            f"{row['projection_failed']:+.4f} | `{row['submission_file']}` |"
        )
    lines.extend(["", "## Recommendation", "", report["recommendation"]["next_action"], ""])
    return "\n".join(lines)


def run(output_json: Path) -> dict[str, Any]:
    log = load_log()
    files = find_base_test_files()
    required = [178, 287, 329, 335]
    missing = [candidate_id for candidate_id in required if candidate_id not in files]
    if missing:
        raise FileNotFoundError(f"Missing required base_test ids: {missing}")
    arr178 = load_pred(files[178])
    arr287 = load_pred(files[287])
    arr329 = load_pred(files[329])
    arr335 = load_pred(files[335])
    rows = [
        row_for_candidate(candidate_id, path, log, arr178, arr287, arr329, arr335)
        for candidate_id, path in sorted(files.items())
    ]
    tier_counts = dict(Counter(row["risk_tier"] for row in rows))
    family_counts = dict(Counter(row["family"] for row in rows))
    top_reviewable = sorted(
        [row for row in rows if row["risk_tier"] == "reviewable_low_movement"],
        key=lambda row: (row["conservative_public_risk_proxy"], row["max_delta_vs_178"]),
    )[:20]
    top_non_same_family = [row for row in top_reviewable if row["family"] not in {"failed_direction_combo_family"}][:12]
    prepared_refs = [row for row in rows if row["id"] in {334, 335}]
    known_public = [row for row in rows if row["id"] in {178, 287, 329}]
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "diagnostic": "post_id335_public_anchor_rerank",
        "purpose": "Refresh existing base-prediction candidate map after ID335; no new predictions and no DACON submission.",
        "n_base_test_ids": len(rows),
        "anchor_ids": {"public_good": [178, 329], "failed_direction": 287, "current_internal_reference": 335},
        "risk_proxy_warning": "The conservative_public_risk_proxy is a screening heuristic, not a validated public prediction.",
        "tier_counts": tier_counts,
        "family_counts": family_counts,
        "known_public_rows": known_public,
        "prepared_sequence_references": prepared_refs,
        "top_reviewable_low_movement": top_reviewable,
        "top_non_same_family_reviewable": top_non_same_family,
        "top_all_by_proxy": sorted(rows, key=lambda row: row["conservative_public_risk_proxy"])[:80],
        "recommendation": {
            "do_not_submit": True,
            "next_action": (
                "Do not submit from this rerank alone. If continuing locally, inspect top low-movement non-failed-family "
                "candidates against ID334/ID335 stress behavior, then require a decision note before any generated artifact or public probe."
            ),
        },
        "no_submission_created": True,
        "no_dacon_submission_made": True,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_json.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORTS_DIR / "post_id335_public_anchor_rerank_20260617.json",
    )
    args = parser.parse_args()
    report = run(args.output_json)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    print(
        json.dumps(
            {
                "n_base_test_ids": report["n_base_test_ids"],
                "tier_counts": report["tier_counts"],
                "top_reviewable_ids": [row["id"] for row in report["top_reviewable_low_movement"][:10]],
                "prepared_sequence_references": report["prepared_sequence_references"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
