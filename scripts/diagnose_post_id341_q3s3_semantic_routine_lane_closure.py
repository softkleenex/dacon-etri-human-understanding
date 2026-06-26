#!/usr/bin/env python3
"""Close the post-ID341 Q3/S3 semantic-routine local lane.

The semantic-routine v2 lane passed design, metadata, and resource gates, but
its fold-internal model smoke selected no Q3/S3 targets. This report-only
closure prevents the lane from being promoted into ID342 or any submission
artifact.
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
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_q3s3_semantic_routine_lane_closure_20260618.json"

INPUT_REPORTS = {
    "design": "post_id341_q3s3_semantic_routine_v2_design_20260618.json",
    "metadata_smoke": "post_id341_q3s3_semantic_routine_metadata_smoke_20260618.json",
    "resource_gate": "post_id341_q3s3_semantic_routine_resource_gate_20260618.json",
    "model_smoke": "post_id341_q3s3_semantic_routine_model_smoke_20260618.json",
    "local_smoke_freshness": "post_id341_local_smoke_freshness_gate_20260618.json",
}
UNCOVERED_TARGETS = ["Q3", "S3"]
PRIMARY_ROUTE = "explicit_approved_clean_upstream_id221_to_id226"


def rel(path: Path) -> str:
    path = path if path.is_absolute() else ROOT / path
    return str(path.relative_to(ROOT))


def load_report(name: str) -> dict[str, Any]:
    path = REPORTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required report: {rel(path)}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_report() -> dict[str, Any]:
    reports = {key: load_report(name) for key, name in INPUT_REPORTS.items()}
    design = reports["design"]["decision"]
    metadata = reports["metadata_smoke"]["decision"]
    resource = reports["resource_gate"]["decision"]
    model = reports["model_smoke"]["decision"]
    freshness = reports["local_smoke_freshness"]["decision"]

    gates_passed_before_model = bool(
        design.get("design_ready") is True
        and metadata.get("metadata_smoke_passed") is True
        and resource.get("resource_gate_passed") is True
        and resource.get("local_no_artifact_model_smoke_allowed") is True
    )
    model_closed = bool(
        model.get("model_smoke_completed") is True
        and model.get("selected_targets") == []
        and model.get("q3_signal_found") is False
        and model.get("s3_signal_found") is False
        and model.get("public_058_evidence") is False
        and model.get("create_submission_artifact") is False
        and model.get("submit_now") is False
    )
    prelaunch_ready = bool(
        freshness.get("local_runner_smoke_fresh") is True
        and freshness.get("requires_local_runner_smoke_rerun") is False
        and freshness.get("create_submission_artifact") is False
        and freshness.get("submit_now") is False
    )
    closure_passed = bool(gates_passed_before_model and model_closed)

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_q3s3_semantic_routine_lane_closure",
        "purpose": (
            "Close the local semantic-routine Q3/S3 lane after its fold-internal smoke selected no targets, while "
            "preserving the explicit-approved upstream rebuild as the score-relevant route."
        ),
        "inputs": {key: rel(REPORTS_DIR / name) for key, name in INPUT_REPORTS.items()},
        "checks": {
            "design_metadata_resource_gates_passed": gates_passed_before_model,
            "model_smoke_closed_without_selected_targets": model_closed,
            "prelaunch_ready_pending_safe": prelaunch_ready,
            "no_artifact_actions": True,
        },
        "decision": {
            "semantic_routine_lane_closure_passed": closure_passed,
            "q3_s3_still_uncovered": UNCOVERED_TARGETS,
            "local_semantic_routine_artifact_allowed": False,
            "public_058_evidence": False,
            "primary_next_score_relevant_route": PRIMARY_ROUTE,
            "requires_explicit_kaggle_approval": True,
            "prelaunch_ready_for_explicit_approval": prelaunch_ready,
            "execute_kaggle_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "next_action": (
                "Do not create ID342 from semantic-routine Q3/S3 features. The lane is structurally computable but "
                "failed fold-internal target selection. Continue only through the explicit-approved clean upstream "
                "ID221→ID226 no-submit rebuild, or design a materially different source."
            ),
        },
        "no_model_trained_now": True,
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# Post-ID341 Q3/S3 semantic-routine lane closure",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        report["purpose"],
        "",
        "This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Lane closure passed: `{decision['semantic_routine_lane_closure_passed']}`",
        f"- Q3/S3 still uncovered: `{decision['q3_s3_still_uncovered']}`",
        f"- Local semantic-routine artifact allowed: `{decision['local_semantic_routine_artifact_allowed']}`",
        f"- Public `0.58` evidence: `{decision['public_058_evidence']}`",
        f"- Primary next score-relevant route: `{decision['primary_next_score_relevant_route']}`",
        f"- Requires explicit Kaggle approval: `{decision['requires_explicit_kaggle_approval']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Checks",
        "",
    ]
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
