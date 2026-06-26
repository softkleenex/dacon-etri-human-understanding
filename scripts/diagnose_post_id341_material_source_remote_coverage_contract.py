#!/usr/bin/env python3
"""Audit material-source queue coverage in the post-ID341 Kaggle packet.

This is a report-only contract. It verifies that each material source-family in
`post_id341_material_new_source_queue_20260619` is either:

1. the explicit-approval score route,
2. required in the no-submit Kaggle runner/intake packet, or
3. deliberately local-only because the lane is already closed by local evidence.

It does not run Kaggle, train models, create predictions, create a submission
CSV, edit the submission log, or submit to DACON.
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

DEFAULT_QUEUE = REPORTS_DIR / "post_id341_material_new_source_queue_20260619.json"
DEFAULT_RUNNER_CONTRACT = REPORTS_DIR / "post_id341_kaggle_runner_intake_contract_20260618.json"
DEFAULT_EXECUTION_PACKET = REPORTS_DIR / "kaggle_upstream_execution_packet_post_id341_20260618.json"
DEFAULT_OUTPUT_INTAKE = REPORTS_DIR / "post_id341_kaggle_output_intake_gate_20260618.json"

APPROVAL_PHRASE = "Kaggle upstream rebuild no-submit 실행 승인"
SCORE_ROUTE = "explicit_approved_clean_upstream_id221_to_id226"

REMOTE_REQUIRED_REPORTS = {
    "usage_semantic_burst_markov_v1": "remote_post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop_20260619.json",
    "self_supervised_subject_state_embedding_v1": (
        "remote_post_id341_self_supervised_subject_state_embedding_design_20260619.json"
    ),
}

LOCAL_CLOSED_EVIDENCE = {
    "gps_path_topology_sequence_v1": [
        "reports/post_id341_gps_path_topology_model_smoke_20260619.json",
    ],
    "circadian_hr_light_phase_shape_v1": [
        "reports/post_id341_circadian_hr_light_phase_model_smoke_20260619.json",
    ],
    "proximity_ambience_cooccurrence_v1": [
        "reports/post_id341_proximity_ambience_cooccurrence_feature_matrix_validation_stop_20260619.json",
    ],
    "phone_wearable_discordance_v1": [
        "reports/post_id341_phone_wearable_discordance_feature_matrix_validation_stop_20260619.json",
    ],
    "sleep_proxy_threshold_residual_v1": [
        "reports/post_id341_sleep_proxy_threshold_residual_design_20260619.json",
    ],
}


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": safe_relative(path), "exists": False, "parse_ok": False, "decision": {}}
    try:
        payload = load_json(path)
    except Exception as exc:  # noqa: BLE001
        return {
            "path": safe_relative(path),
            "exists": True,
            "parse_ok": False,
            "decision": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "path": safe_relative(path),
        "exists": True,
        "parse_ok": True,
        "decision": payload.get("decision", {}),
    }


def local_evidence_status(paths: list[str]) -> list[dict[str, Any]]:
    return [file_status(ROOT / path) for path in paths]


def remote_report_present(report_name: str, runner: dict[str, Any], output_intake: dict[str, Any]) -> dict[str, Any]:
    generated = set(runner.get("runner_generated_reports", []))
    staged = set(runner.get("runner_staged_reports", []))
    expected = set(runner.get("intake_expected_reports", []))
    output_expected = {
        row.get("filename")
        for row in output_intake.get("expected_report_status", [])
        if isinstance(row, dict) and row.get("filename")
    }
    return {
        "remote_report": report_name,
        "runner_generates": report_name in generated,
        "runner_stages": report_name in staged,
        "runner_intake_expects": report_name in expected,
        "output_intake_expects": report_name in output_expected,
        "covered": bool(
            report_name in generated
            and report_name in staged
            and report_name in expected
            and report_name in output_expected
        ),
    }


def queue_item_coverage(item: dict[str, Any], runner: dict[str, Any], output_intake: dict[str, Any]) -> dict[str, Any]:
    name = item.get("name")
    status = item.get("status")
    common_safe = bool(
        item.get("local_no_approval_allowed") is False
        and item.get("train_model_now") is False
        and item.get("create_artifact_allowed") is False
    )
    if name == "approved_clean_upstream_id221_to_id226":
        clean_reports = [
            "remote_post_id341_clean_id221_feature_similarity_smoke_20260618.json",
            "remote_post_id341_clean_id221_inner_selection_probe_20260618.json",
            "remote_post_id341_clean_id226_calibration_smoke_20260618.json",
            "remote_post_id341_clean_backbone_lane_closure_20260618.json",
        ]
        remote_checks = [remote_report_present(report_name, runner, output_intake) for report_name in clean_reports]
        covered = bool(
            item.get("type") == "score_relevant_remote_no_submit"
            and status == "open_requires_exact_approval"
            and item.get("next_allowed_step") == "only_after_exact_approval_phrase"
            and item.get("score_relevant") is True
            and common_safe
            and all(row["covered"] for row in remote_checks)
        )
        return {
            "name": name,
            "coverage_mode": "explicit_approval_remote_score_route",
            "covered": covered,
            "remote_checks": remote_checks,
            "local_evidence": [],
            "reason": "Score-relevant route is open only behind the exact Kaggle no-submit approval phrase.",
        }
    if name in REMOTE_REQUIRED_REPORTS:
        remote_check = remote_report_present(REMOTE_REQUIRED_REPORTS[name], runner, output_intake)
        expected_status_ok = (
            status == "validated_stop_completed_no_model"
            if name == "usage_semantic_burst_markov_v1"
            else status == "design_gate_ready_heavy_deferred" and item.get("design_gate_ready") is True
        )
        covered = bool(expected_status_ok and common_safe and remote_check["covered"])
        return {
            "name": name,
            "coverage_mode": "remote_required_report_only_gate",
            "covered": covered,
            "remote_checks": [remote_check],
            "local_evidence": [],
            "reason": "This source family must be regenerated/staged by the no-submit runner as report-only evidence.",
        }
    if name in LOCAL_CLOSED_EVIDENCE:
        evidence = local_evidence_status(LOCAL_CLOSED_EVIDENCE[name])
        covered = bool(
            status
            in {
                "closed_after_feature_matrix_and_model_smoke",
                "closed_after_feature_matrix_and_model_smoke_no_signal",
                "validated_stop_completed_no_model",
                "design_backlog_recorded_closed_no_local_lane",
            }
            and item.get("next_allowed_step")
            in {
                "closed_no_artifact",
                "closed_stop_or_wait_for_explicit_kaggle_approval",
                "closed_design_backlog_only",
            }
            and common_safe
            and item.get("submission_artifact_allowed", False) is False
            and all(row["exists"] and row["parse_ok"] for row in evidence)
        )
        return {
            "name": name,
            "coverage_mode": "local_closed_evidence_only",
            "covered": covered,
            "remote_checks": [],
            "local_evidence": evidence,
            "reason": "Local validation/model-smoke evidence closed the lane; remote runner coverage is intentionally not required.",
        }
    return {
        "name": name,
        "coverage_mode": "unknown_queue_item",
        "covered": False,
        "remote_checks": [],
        "local_evidence": [],
        "reason": "Queue item is not covered by the remote/local coverage contract.",
    }


def build_report(
    queue_path: Path = DEFAULT_QUEUE,
    runner_contract_path: Path = DEFAULT_RUNNER_CONTRACT,
    execution_packet_path: Path = DEFAULT_EXECUTION_PACKET,
    output_intake_path: Path = DEFAULT_OUTPUT_INTAKE,
) -> dict[str, Any]:
    queue = load_json(queue_path)
    runner = load_json(runner_contract_path)
    packet = load_json(execution_packet_path)
    output_intake = load_json(output_intake_path)
    queue_items = list(queue.get("source_queue", []))
    coverage_rows = [queue_item_coverage(item, runner, output_intake) for item in queue_items]
    missing_names = sorted(
        set(REMOTE_REQUIRED_REPORTS)
        - {row["name"] for row in coverage_rows if row["coverage_mode"] != "unknown_queue_item"}
    )
    all_covered = all(row["covered"] for row in coverage_rows) and not missing_names
    queue_decision = queue.get("decision", {})
    runner_decision = runner.get("decision", {})
    packet_decision = packet.get("decision", {})
    output_decision = output_intake.get("decision", {})
    no_forbidden_actions = all(
        decision.get(key) is not True
        for decision in [queue_decision, runner_decision, packet_decision, output_decision]
        for key in ["create_submission_artifact", "submit_now", "push_kaggle_now", "run_kaggle_now"]
    )
    contract_pass = bool(
        all_covered
        and queue_decision.get("material_new_source_queue_ready") is True
        and queue_decision.get("local_no_approval_score_route_exists") is False
        and queue_decision.get("only_score_relevant_open_route") == SCORE_ROUTE
        and queue_decision.get("approval_phrase") == APPROVAL_PHRASE
        and runner_decision.get("runner_intake_contract_pass") is True
        and packet_decision.get("ready_for_explicit_approval") is True
        and output_decision.get("output_pending") is True
        and output_decision.get("artifact_contract_review_allowed") is False
        and no_forbidden_actions
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_material_source_remote_coverage_contract",
        "purpose": (
            "Verify every material source-family is covered by either the no-submit runner/intake packet, "
            "a local lane-closure evidence report, or the explicit-approval score route."
        ),
        "inputs": {
            "queue": safe_relative(queue_path),
            "runner_contract": safe_relative(runner_contract_path),
            "execution_packet": safe_relative(execution_packet_path),
            "output_intake": safe_relative(output_intake_path),
        },
        "coverage_rows": coverage_rows,
        "missing_remote_required_source_families": missing_names,
        "counts": {
            "queue_items": len(queue_items),
            "remote_required_source_families": len(REMOTE_REQUIRED_REPORTS),
            "local_closed_source_families": len(LOCAL_CLOSED_EVIDENCE),
            "covered_queue_items": sum(1 for row in coverage_rows if row["covered"]),
        },
        "decision": {
            "material_source_remote_coverage_contract_pass": contract_pass,
            "all_queue_items_covered": all_covered,
            "no_forbidden_actions": no_forbidden_actions,
            "only_score_relevant_open_route": queue_decision.get("only_score_relevant_open_route"),
            "requires_explicit_kaggle_approval": True,
            "approval_phrase": APPROVAL_PHRASE,
            "public_058_evidence": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Coverage is complete; wait for exact Kaggle no-submit approval or continue report-only source design."
                if contract_pass
                else "Fix queue/runner/output/prelaunch coverage gaps before any Kaggle approval is consumed."
            ),
        },
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    dec = report["decision"]
    lines = [
        "# Post-ID341 material source remote coverage contract",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no model, prediction, submission CSV, Kaggle run, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Coverage contract pass: `{dec['material_source_remote_coverage_contract_pass']}`",
        f"- All queue items covered: `{dec['all_queue_items_covered']}`",
        f"- Only score-relevant open route: `{dec['only_score_relevant_open_route']}`",
        f"- Requires explicit Kaggle approval: `{dec['requires_explicit_kaggle_approval']}`",
        f"- Approval phrase: `{dec['approval_phrase']}`",
        f"- Create submission artifact: `{dec['create_submission_artifact']}`",
        f"- Submit now: `{dec['submit_now']}`",
        f"- Next action: {dec['next_action']}",
        "",
        "## Coverage rows",
        "",
        "| source | mode | covered | reason |",
        "|---|---|:---:|---|",
    ]
    for row in report["coverage_rows"]:
        lines.append(f"| `{row['name']}` | `{row['coverage_mode']}` | `{row['covered']}` | {row['reason']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_material_source_remote_coverage_contract_20260619.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {safe_relative(output)}")
    print(f"Wrote {safe_relative(output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
