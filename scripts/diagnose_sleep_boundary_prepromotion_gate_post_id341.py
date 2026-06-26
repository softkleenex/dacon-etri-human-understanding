#!/usr/bin/env python3
"""Prepromotion gate for the post-ID341 sleep-boundary v3 scale scan.

This report-only gate reads the v3 scale-scan evidence and decides whether the
best low-movement row is strong enough to become a full artifact candidate. It
creates no submission CSV, base prediction artifact, notebook/history artifact,
submission-log row, or DACON submission.
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
V3_PATH = REPORTS_DIR / "sleep_boundary_transition_v3_scale_scan_post_id341_20260618.json"
PUBLIC_ID178 = 0.6014113009
PUBLIC_ID329 = 0.6017908577
PUBLIC_058_GATE = 0.590
MAX_DELTA_VS_341_CEILING = 0.005
MEAN_DELTA_VS_341_CEILING = 0.0005
FAILED_PROJECTION_CEILING = 0.0
PROMOTION_PROXY_CEILING = PUBLIC_ID178


def load_v3(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing v3 report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def choose_best_row(v3: dict[str, Any]) -> dict[str, Any] | None:
    row = v3.get("best_reviewable_low_movement")
    if row is not None:
        return row
    rows = v3.get("scale_rows", [])
    if not rows:
        return None
    return sorted(rows, key=lambda item: item["test_geometry"]["proxy_public_risk"])[0]


def build_report(v3_path: Path = V3_PATH) -> dict[str, Any]:
    v3 = load_v3(v3_path)
    row = choose_best_row(v3)
    if row is None:
        raise ValueError("V3 report has no scale rows.")
    geom = row["test_geometry"]
    split_rows = row["split_rows"]
    all_split_wins = all(split["beats_id341"] for split in split_rows.values())
    gate_checks = {
        "all_splits_beat_id341": bool(all_split_wins),
        "max_delta_vs_341_within_ceiling": bool(geom["max_delta_vs_341"] <= MAX_DELTA_VS_341_CEILING),
        "mean_delta_vs_341_within_ceiling": bool(geom["mean_delta_vs_341"] <= MEAN_DELTA_VS_341_CEILING),
        "failed_projection_non_positive": bool(geom["projection_failed"] <= FAILED_PROJECTION_CEILING),
        "proxy_below_current_id341_proxy": bool(v3["decision"]["reviewable_low_movement_candidate_exists"]),
        "proxy_below_best_actual_public_anchor_id178": bool(geom["proxy_public_risk"] < PROMOTION_PROXY_CEILING),
        "public_058_plausible": bool(geom["proxy_public_risk"] <= PUBLIC_058_GATE),
    }
    promote = bool(
        gate_checks["all_splits_beat_id341"]
        and gate_checks["max_delta_vs_341_within_ceiling"]
        and gate_checks["mean_delta_vs_341_within_ceiling"]
        and gate_checks["failed_projection_non_positive"]
        and gate_checks["proxy_below_best_actual_public_anchor_id178"]
        and gate_checks["public_058_plausible"]
    )
    review_only = bool(
        gate_checks["all_splits_beat_id341"]
        and gate_checks["max_delta_vs_341_within_ceiling"]
        and gate_checks["mean_delta_vs_341_within_ceiling"]
        and gate_checks["failed_projection_non_positive"]
        and gate_checks["proxy_below_current_id341_proxy"]
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "sleep_boundary_prepromotion_gate_post_id341",
        "purpose": "Decide whether the best low-movement sleep-boundary v3 row is strong enough for artifact promotion.",
        "inputs": {"v3_scale_scan": str(v3_path.relative_to(ROOT))},
        "candidate_slug": "id341_sleep_boundary_v3_scale005_q1_s1_s2_s4_micro",
        "candidate_scale": row["scale"],
        "effective_weights": row["effective_weights"],
        "selected_target_configs": v3["selected_target_configs"],
        "split_rows": split_rows,
        "test_geometry": geom,
        "public_context": {
            "public_id178": PUBLIC_ID178,
            "public_id329": PUBLIC_ID329,
            "proxy_minus_public_id178": float(geom["proxy_public_risk"] - PUBLIC_ID178),
            "proxy_minus_public_id329": float(geom["proxy_public_risk"] - PUBLIC_ID329),
            "public_058_gate": PUBLIC_058_GATE,
        },
        "gate_checks": gate_checks,
        "decision": {
            "reviewable_low_movement_feature_lead": review_only,
            "promote_submission_artifact": promote,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "why": (
                "The scale-0.05 row is clean as a low-movement feature lead, but its conservative proxy remains above "
                "the best actual public anchor ID178 and far above Public 0.58. Keep it as a source lead; do not create ID342 yet."
            ),
            "next_action": (
                "Use sleep-boundary features inside the upstream/new-source lane, or search for an additional independent source "
                "that lowers the proxy below the ID178 public anchor before promotion."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    geom = report["test_geometry"]
    public_context = report["public_context"]
    lines = [
        "# Sleep-boundary prepromotion gate — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, or DACON submission.",
        "",
        f"Candidate slug: `{report['candidate_slug']}`",
        f"Scale: `{report['candidate_scale']}`",
        "",
        "## Effective target weights",
        "",
        "| target | effective weight | base config |",
        "|---|---:|---|",
    ]
    for target, weight in report["effective_weights"].items():
        config = report["selected_target_configs"][target]
        lines.append(f"| `{target}` | {weight:.6f} | `{config['candidate']}` |")
    lines.extend(
        [
            "",
            "## Split gate",
            "",
            "| split | loss | ID341 loss | Δ vs ID341 | pass |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for split_name, row in report["split_rows"].items():
        lines.append(
            f"| `{split_name}` | {row['candidate_loss']:.6f} | {row['id341_loss']:.6f} | "
            f"{row['delta_vs_id341']:+.6f} | `{row['beats_id341']}` |"
        )
    lines.extend(
        [
            "",
            "## Geometry and public context",
            "",
            f"- Proxy: `{geom['proxy_public_risk']:.6f}`",
            f"- Mean/max Δ vs ID341: `{geom['mean_delta_vs_341']:.6f}` / `{geom['max_delta_vs_341']:.6f}`",
            f"- Failed-direction projection: `{geom['projection_failed']:+.6f}`",
            f"- Proxy minus Public ID178: `{public_context['proxy_minus_public_id178']:+.6f}`",
            f"- Proxy minus Public ID329: `{public_context['proxy_minus_public_id329']:+.6f}`",
            "",
            "## Gate checks",
            "",
        ]
    )
    for key, value in report["gate_checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Reviewable feature lead: `{report['decision']['reviewable_low_movement_feature_lead']}`",
            f"- Promote submission artifact: `{report['decision']['promote_submission_artifact']}`",
            f"- Create submission artifact: `{report['decision']['create_submission_artifact']}`",
            f"- Submit now: `{report['decision']['submit_now']}`",
            f"- Why: {report['decision']['why']}",
            f"- Next action: {report['decision']['next_action']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "sleep_boundary_prepromotion_gate_post_id341_20260618.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
