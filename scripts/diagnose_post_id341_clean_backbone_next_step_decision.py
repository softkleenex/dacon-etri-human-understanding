#!/usr/bin/env python3
"""Decide the next clean-backbone step after ID221 smoke/probe results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))

DEFAULT_SMOKE = REPORTS_DIR / "post_id341_clean_id221_feature_similarity_smoke_20260618.json"
DEFAULT_INNER = REPORTS_DIR / "post_id341_clean_id221_inner_selection_probe_20260618.json"
DEFAULT_DESIGN = REPORTS_DIR / "post_id341_clean_backbone_rebuild_design_20260618.json"


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def build_report(smoke_path: Path, inner_path: Path, design_path: Path) -> dict[str, Any]:
    smoke = load(smoke_path)
    inner = load(inner_path)
    design = load(design_path)
    smoke_decision = smoke["decision"]
    inner_decision = inner["decision"]
    id221_improves_base = bool(
        smoke_decision.get("clean_id221_fixed_config_signal") or inner_decision.get("inner_selection_improves_id218")
    )
    id221_safe_vs_anchor = bool(
        smoke_decision.get("beats_id178_primary")
        and smoke_decision.get("beats_id341_primary")
        and inner_decision.get("beats_id178_primary")
        and inner_decision.get("beats_id341_primary")
    )
    failed_projection = float(inner["test_geometry_preview_no_artifact"]["projection_failed_id287"])
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_clean_backbone_next_step_decision",
        "purpose": "Convert clean ID221 smoke/probe evidence into the next no-submit clean-backbone action.",
        "inputs": {
            "clean_backbone_design": rel(design_path),
            "fixed_config_smoke": rel(smoke_path),
            "inner_selection_probe": rel(inner_path),
        },
        "evidence": {
            "fixed_config_delta_vs_id218": smoke["full_oof"]["delta_vs_id218"],
            "inner_selection_delta_vs_id218": inner["full_oof"]["delta_vs_id218"],
            "inner_selection_delta_vs_id178": inner["full_oof"]["delta_vs_id178"],
            "inner_selection_delta_vs_id341": inner["full_oof"]["delta_vs_id341"],
            "inner_selection_failed_projection_id287": failed_projection,
            "inner_selection_active_slots": inner["selection_summary"]["active_selection_count"],
            "inner_selection_total_slots": inner["selection_summary"]["total_selection_slots"],
            "design_ready": design["decision"]["clean_backbone_design_ready"],
        },
        "decision": {
            "spend_more_on_clean_id221_now": False,
            "treat_clean_id221_as_anchor_control": bool(id221_safe_vs_anchor and not id221_improves_base),
            "move_to_clean_id226_design_or_smoke": True,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_heavy_rebuild_now": False,
            "next_action": (
                "Do not promote or further tune ID221 as a standalone source. Use ID218/clean-ID221 behavior as a safe "
                "anchor/control and move the next report-only work to clean ID226 fold-internal convex/calibration smoke."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    ev = report["evidence"]
    decision = report["decision"]
    return "\n".join(
        [
            "# Post-ID341 clean-backbone next-step decision",
            "",
            f"Created: `{report['created_at_kst']}`",
            "",
            "This is report-only. It creates no submission artifact and makes no DACON submission.",
            "",
            "## Evidence",
            "",
            f"- Fixed-config clean ID221 Δ vs ID218: `{ev['fixed_config_delta_vs_id218']:+.6f}`",
            f"- Inner-selection clean ID221 Δ vs ID218: `{ev['inner_selection_delta_vs_id218']:+.6f}`",
            f"- Inner-selection clean ID221 Δ vs ID178/ID341: `{ev['inner_selection_delta_vs_id178']:+.6f}` / `{ev['inner_selection_delta_vs_id341']:+.6f}`",
            f"- Inner-selection failed-direction projection: `{ev['inner_selection_failed_projection_id287']:+.4f}`",
            f"- Active inner selections: `{ev['inner_selection_active_slots']}` / `{ev['inner_selection_total_slots']}`",
            "",
            "## Decision",
            "",
            f"- Spend more on clean ID221 now: `{decision['spend_more_on_clean_id221_now']}`",
            f"- Treat clean ID221 as anchor/control: `{decision['treat_clean_id221_as_anchor_control']}`",
            f"- Move to clean ID226 design/smoke: `{decision['move_to_clean_id226_design_or_smoke']}`",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
            f"- Submit now: `{decision['submit_now']}`",
            f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
            f"- Run heavy rebuild now: `{decision['run_heavy_rebuild_now']}`",
            f"- Next action: {decision['next_action']}",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-report", type=Path, default=DEFAULT_SMOKE)
    parser.add_argument("--inner-report", type=Path, default=DEFAULT_INNER)
    parser.add_argument("--design-report", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_clean_backbone_next_step_decision_20260618.json",
    )
    args = parser.parse_args()
    smoke_path = args.smoke_report if args.smoke_report.is_absolute() else ROOT / args.smoke_report
    inner_path = args.inner_report if args.inner_report.is_absolute() else ROOT / args.inner_report
    design_path = args.design_report if args.design_report.is_absolute() else ROOT / args.design_report
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(smoke_path, inner_path, design_path)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
