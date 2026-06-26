#!/usr/bin/env python3
"""Close or continue the post-ID341 clean-backbone lane."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))

DEFAULT_NEXT_STEP = REPORTS_DIR / "post_id341_clean_backbone_next_step_decision_20260618.json"
DEFAULT_ID226 = REPORTS_DIR / "post_id341_clean_id226_calibration_smoke_20260618.json"


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def build_report(next_step_path: Path, id226_path: Path) -> dict[str, Any]:
    next_step = load(next_step_path)
    id226 = load(id226_path)
    id221_as_control = bool(next_step["decision"]["treat_clean_id221_as_anchor_control"])
    id226_signal = bool(id226["decision"]["clean_id226_calibration_signal"])
    close_lane = bool(id221_as_control and not id226_signal)
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_clean_backbone_lane_closure",
        "purpose": "Decide whether to continue ID221/ID226 clean-backbone work after the fold-internal ID226 calibration smoke.",
        "inputs": {
            "clean_backbone_next_step": rel(next_step_path),
            "clean_id226_calibration_smoke": rel(id226_path),
        },
        "evidence": {
            "id221_treat_as_anchor_control": id221_as_control,
            "id226_delta_vs_id218": id226["full_oof"]["delta_vs_id218"],
            "id226_delta_vs_id178": id226["full_oof"]["delta_vs_id178"],
            "id226_delta_vs_id341": id226["full_oof"]["delta_vs_id341"],
            "id226_primary_beats_id178": id226["decision"]["beats_id178_primary"],
            "id226_primary_beats_id341": id226["decision"]["beats_id341_primary"],
            "id226_clean_calibration_signal": id226_signal,
            "id226_failed_projection_id287": id226["test_geometry_preview_no_artifact"]["projection_failed_id287"],
        },
        "decision": {
            "close_clean_backbone_lane_now": close_lane,
            "do_not_replay_id227_228_237": close_lane,
            "do_not_create_id342_from_clean_backbone": True,
            "next_lane": "stronger_raw_or_sequence_source_model",
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_heavy_rebuild_now": False,
            "next_action": (
                "Stop local ID221/ID226/ID227/ID228/ID237 clean-backbone escalation. The next score-relevant report-only "
                "work should be a stronger raw/sequence source model or another independent source family, then transfer-gated "
                "against ID178/287/329/341."
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
            "# Post-ID341 clean-backbone lane closure",
            "",
            f"Created: `{report['created_at_kst']}`",
            "",
            "This is report-only. It creates no submission artifact and makes no DACON submission.",
            "",
            "## Evidence",
            "",
            f"- ID221 as anchor/control: `{ev['id221_treat_as_anchor_control']}`",
            f"- Clean ID226 Δ vs ID218: `{ev['id226_delta_vs_id218']:+.6f}`",
            f"- Clean ID226 Δ vs ID178/ID341: `{ev['id226_delta_vs_id178']:+.6f}` / `{ev['id226_delta_vs_id341']:+.6f}`",
            f"- Clean ID226 primary beats ID178/ID341: `{ev['id226_primary_beats_id178']}` / `{ev['id226_primary_beats_id341']}`",
            f"- Clean ID226 failed-direction projection: `{ev['id226_failed_projection_id287']:+.4f}`",
            "",
            "## Decision",
            "",
            f"- Close clean-backbone lane now: `{decision['close_clean_backbone_lane_now']}`",
            f"- Do not replay ID227/228/237: `{decision['do_not_replay_id227_228_237']}`",
            f"- Do not create ID342 from clean backbone: `{decision['do_not_create_id342_from_clean_backbone']}`",
            f"- Next lane: `{decision['next_lane']}`",
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
    parser.add_argument("--next-step-report", type=Path, default=DEFAULT_NEXT_STEP)
    parser.add_argument("--id226-report", type=Path, default=DEFAULT_ID226)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_clean_backbone_lane_closure_20260618.json",
    )
    args = parser.parse_args()
    next_step_path = args.next_step_report if args.next_step_report.is_absolute() else ROOT / args.next_step_report
    id226_path = args.id226_report if args.id226_report.is_absolute() else ROOT / args.id226_report
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(next_step_path, id226_path)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
