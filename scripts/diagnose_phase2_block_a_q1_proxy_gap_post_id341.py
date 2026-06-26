#!/usr/bin/env python3
"""Quantify the Q1/proxy gap after phase-2 block A.

This report-only diagnostic reads the local block-A bridge smoke and measures
how far the Q1-only signal is from the ID178 public-anchor proxy and Public
0.58 review line. It trains no model and creates no predictions or submission
artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
DEFAULT_BLOCK_A_PATH = REPORTS_DIR / "phase2_block_a_q1_s4_sleep_residual_bridge_post_id341_20260618.json"
ID341_TRANSFER_PROXY = 0.6038297835
ID178_PUBLIC = 0.6014113009
PUBLIC_058_REVIEW_CUTOFF = 0.590


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_block_a(source_report: Path) -> dict[str, Any]:
    if not source_report.exists():
        raise FileNotFoundError(f"Missing block-A report: {source_report}")
    return json.loads(source_report.read_text(encoding="utf-8"))


def proxy_from_target_delta(target_delta: float, n_targets: int = len(TARGETS)) -> float:
    return float(ID341_TRANSFER_PROXY + target_delta / n_targets)


def q1_frontier_rows(block_a: dict[str, Any]) -> list[dict[str, Any]]:
    rows = block_a.get("top_scan_by_target", {}).get("Q1", [])
    out = []
    for row in rows:
        target_mean_delta = float(row["mean_delta_vs_id341"])
        out.append(
            {
                "candidate": row["candidate"],
                "primary_wins_vs_id341": row["primary_wins_vs_id341"],
                "target_mean_delta_vs_id341": target_mean_delta,
                "target_max_delta_vs_id341": float(row["max_delta_vs_id341"]),
                "target_min_delta_vs_id341": float(row["min_delta_vs_id341"]),
                "assembled_proxy_if_q1_only": proxy_from_target_delta(target_mean_delta),
                "proxy_gap_vs_id178_public": float(proxy_from_target_delta(target_mean_delta) - ID178_PUBLIC),
                "public_058_gap": float(proxy_from_target_delta(target_mean_delta) - PUBLIC_058_REVIEW_CUTOFF),
            }
        )
    return out


def s4_frontier_rows(block_a: dict[str, Any]) -> list[dict[str, Any]]:
    rows = block_a.get("top_scan_by_target", {}).get("S4", [])
    out = []
    for row in rows[:8]:
        target_mean_delta = float(row["mean_delta_vs_id341"])
        out.append(
            {
                "candidate": row["candidate"],
                "primary_wins_vs_id341": row["primary_wins_vs_id341"],
                "target_mean_delta_vs_id341": target_mean_delta,
                "target_max_delta_vs_id341": float(row["max_delta_vs_id341"]),
                "assembled_proxy_if_s4_only": proxy_from_target_delta(target_mean_delta),
                "eligible_under_block_a_rule": bool(
                    row["primary_wins_vs_id341"] >= 4
                    and row["mean_delta_vs_id341"] <= -0.00005
                    and row["max_delta_vs_id341"] <= 0.00050
                ),
            }
        )
    return out


def build_report(source_report: Path) -> dict[str, Any]:
    block_a = load_block_a(source_report)
    decision = block_a["decision"]
    q1_rows = q1_frontier_rows(block_a)
    s4_rows = s4_frontier_rows(block_a)
    best_q1 = min(q1_rows, key=lambda row: row["assembled_proxy_if_q1_only"]) if q1_rows else None
    required_mean_delta_to_beat_id178 = float(ID178_PUBLIC - ID341_TRANSFER_PROXY)
    required_mean_delta_to_public_058 = float(PUBLIC_058_REVIEW_CUTOFF - ID341_TRANSFER_PROXY)
    required_q1_target_delta_to_beat_id178 = float(required_mean_delta_to_beat_id178 * len(TARGETS))
    required_q1_target_delta_to_public_058 = float(required_mean_delta_to_public_058 * len(TARGETS))
    current_assembled_delta = float(decision["mean_primary_delta_vs_id341"])
    current_q1_target_delta = (
        float(block_a["selected_configs"]["Q1"]["mean_delta_vs_id341"]) if "Q1" in block_a["selected_configs"] else None
    )
    factor_to_id178 = (
        float(required_q1_target_delta_to_beat_id178 / current_q1_target_delta)
        if current_q1_target_delta and current_q1_target_delta < 0
        else None
    )
    factor_to_public_058 = (
        float(required_q1_target_delta_to_public_058 / current_q1_target_delta)
        if current_q1_target_delta and current_q1_target_delta < 0
        else None
    )
    safe_q1_rows = [
        row for row in q1_rows if row["primary_wins_vs_id341"] == 5 and row["target_max_delta_vs_id341"] <= 0.0
    ]
    best_safe_q1 = min(safe_q1_rows, key=lambda row: row["assembled_proxy_if_q1_only"]) if safe_q1_rows else None
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "phase2_block_a_q1_proxy_gap_post_id341",
        "purpose": "Measure whether the all-primary Q1 block-A bridge signal is large enough for ID178/Public-0.58 transfer.",
        "source_report": display_path(source_report),
        "source_decision": decision,
        "q1_frontier_rows": q1_rows,
        "s4_frontier_rows": s4_rows,
        "best_q1_proxy_row": best_q1,
        "best_safe_q1_proxy_row": best_safe_q1,
        "proxy_gap_math": {
            "id341_transfer_proxy": ID341_TRANSFER_PROXY,
            "id178_public": ID178_PUBLIC,
            "public_058_review_cutoff": PUBLIC_058_REVIEW_CUTOFF,
            "current_assembled_delta_vs_id341": current_assembled_delta,
            "current_projected_proxy": decision["projected_public_proxy_from_id341"],
            "current_gap_vs_id178_public": float(decision["projected_public_proxy_from_id341"] - ID178_PUBLIC),
            "required_mean_delta_to_beat_id178": required_mean_delta_to_beat_id178,
            "required_additional_mean_delta_to_beat_id178": float(
                required_mean_delta_to_beat_id178 - current_assembled_delta
            ),
            "required_q1_target_delta_to_beat_id178_if_q1_only": required_q1_target_delta_to_beat_id178,
            "current_q1_target_delta": current_q1_target_delta,
            "q1_signal_factor_needed_to_beat_id178_if_q1_only": factor_to_id178,
            "required_mean_delta_to_public_058": required_mean_delta_to_public_058,
            "required_q1_target_delta_to_public_058_if_q1_only": required_q1_target_delta_to_public_058,
            "q1_signal_factor_needed_to_public_058_if_q1_only": factor_to_public_058,
        },
        "decision": {
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "q1_all_primary_signal_confirmed": decision["all_primary_splits_beat_id341"] is True
            and decision["selected_targets"] == ["Q1"],
            "q1_signal_sufficient_for_id178_proxy": bool(decision["beats_id178_public_anchor_proxy"]),
            "q1_signal_sufficient_for_public_058": bool(decision["public_058_reviewable"]),
            "additional_independent_target_signal_required": True,
            "next_action": (
                "Do not promote Q1-only block A. Use it as the current strongest parity target, but reaching ID178/Public "
                "requires an additional independent target/source signal rather than merely enlarging this local Q1 overlay."
            ),
        },
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    math = report["proxy_gap_math"]
    decision = report["decision"]
    lines = [
        "# Phase-2 block A Q1 proxy gap — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It reads the existing block-A smoke and performs gap math only; no model is trained and no predictions or submission artifacts are created.",
        "",
        "## Gap math",
        "",
        f"- Current projected proxy: `{math['current_projected_proxy']:.6f}`",
        f"- Gap vs actual ID178 Public `{math['id178_public']:.10f}`: `{math['current_gap_vs_id178_public']:+.6f}`",
        f"- Required additional mean delta to beat ID178: `{math['required_additional_mean_delta_to_beat_id178']:+.6f}`",
        f"- Current Q1 target delta: `{math['current_q1_target_delta']:+.6f}`",
        f"- Q1-only signal factor needed to beat ID178: `{math['q1_signal_factor_needed_to_beat_id178_if_q1_only']:.2f}x`",
        f"- Q1-only signal factor needed for Public 0.58: `{math['q1_signal_factor_needed_to_public_058_if_q1_only']:.2f}x`",
        "",
        "## Q1 frontier rows",
        "",
        "| candidate | wins | target mean Δ | max Δ | proxy if Q1-only | gap vs ID178 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["q1_frontier_rows"][:10]:
        lines.append(
            f"| `{row['candidate']}` | {row['primary_wins_vs_id341']} | "
            f"{row['target_mean_delta_vs_id341']:+.6f} | {row['target_max_delta_vs_id341']:+.6f} | "
            f"{row['assembled_proxy_if_q1_only']:.6f} | {row['proxy_gap_vs_id178_public']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## S4 frontier check",
            "",
            "| candidate | wins | target mean Δ | max Δ | eligible |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in report["s4_frontier_rows"][:6]:
        lines.append(
            f"| `{row['candidate']}` | {row['primary_wins_vs_id341']} | "
            f"{row['target_mean_delta_vs_id341']:+.6f} | {row['target_max_delta_vs_id341']:+.6f} | "
            f"`{row['eligible_under_block_a_rule']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Q1 all-primary signal confirmed: `{decision['q1_all_primary_signal_confirmed']}`",
            f"- Q1 sufficient for ID178 proxy: `{decision['q1_signal_sufficient_for_id178_proxy']}`",
            f"- Q1 sufficient for Public 0.58: `{decision['q1_signal_sufficient_for_public_058']}`",
            f"- Additional independent target/source signal required: `{decision['additional_independent_target_signal_required']}`",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
            f"- Submit now: `{decision['submit_now']}`",
            f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
            f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
            f"- Next action: {decision['next_action']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "phase2_block_a_q1_proxy_gap_post_id341_20260618.json",
    )
    parser.add_argument(
        "--source-report",
        type=Path,
        default=DEFAULT_BLOCK_A_PATH,
        help="Block-A bridge report JSON to analyze. Defaults to the local post-ID341 block-A report.",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    source_report = args.source_report if args.source_report.is_absolute() else ROOT / args.source_report
    report = build_report(source_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
