#!/usr/bin/env python3
"""Decide whether pulled remote evidence clears the Public 0.58 artifact-review threshold.

This is stricter than the generic remote next-action gate. It exists so a
Kaggle/Linux no-submit run can only open a Public-0.58-oriented artifact review
when the pulled reports provide both:

1. safe intake/summary/next-action evidence, and
2. a proxy at or below the Public 0.58 target.

It never creates predictions, CSVs, artifacts, Kaggle runs, or DACON submissions.
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

DEFAULT_REMOTE_SUMMARY = REPORTS_DIR / "post_id341_kaggle_remote_result_summary_20260618.json"
DEFAULT_NEXT_ACTION = REPORTS_DIR / "post_id341_remote_next_action_decision_20260618.json"
DEFAULT_PUBLIC_GAP = REPORTS_DIR / "post_id341_public_local_gap_diagnosis_20260618.json"
DEFAULT_DATA_COVERAGE = REPORTS_DIR / "post_id341_data_source_coverage_audit_20260618.json"
DEFAULT_PRELAUNCH = REPORTS_DIR / "post_id341_kaggle_prelaunch_readiness_gate_20260618.json"
PUBLIC_058_TARGET = 0.58
ID178_PUBLIC_ANCHOR = 0.6014113009


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def status_for(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": safe_relative(path), "exists": False, "parse_ok": False, "error": None, "decision": None}
    try:
        payload = load_json(path)
        return {
            "path": safe_relative(path),
            "exists": True,
            "parse_ok": True,
            "bytes": path.stat().st_size,
            "decision": payload.get("decision", {}),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "path": safe_relative(path),
            "exists": True,
            "parse_ok": False,
            "bytes": path.stat().st_size,
            "decision": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def numeric(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def best_proxy(summary: dict[str, Any]) -> dict[str, Any]:
    best = summary.get("best_proxy_row") or {}
    proxy = numeric(best.get("min_proxy"))
    return {
        "filename": best.get("filename"),
        "min_proxy": proxy,
        "delta_vs_public_058": None if proxy is None else proxy - PUBLIC_058_TARGET,
        "delta_vs_id178_public": None if proxy is None else proxy - ID178_PUBLIC_ANCHOR,
        "selected_targets": best.get("selected_targets"),
    }


def build_report(
    remote_summary_path: Path,
    next_action_path: Path,
    public_gap_path: Path,
    data_coverage_path: Path,
    prelaunch_path: Path,
) -> dict[str, Any]:
    summary_status = status_for(remote_summary_path)
    next_action_status = status_for(next_action_path)
    public_gap_status = status_for(public_gap_path)
    data_coverage_status = status_for(data_coverage_path)
    prelaunch_status = status_for(prelaunch_path)

    summary = load_json(remote_summary_path) if summary_status["parse_ok"] else {}
    next_action = load_json(next_action_path) if next_action_status["parse_ok"] else {}
    public_gap = load_json(public_gap_path) if public_gap_status["parse_ok"] else {}
    data_coverage = load_json(data_coverage_path) if data_coverage_status["parse_ok"] else {}
    prelaunch = load_json(prelaunch_path) if prelaunch_status["parse_ok"] else {}

    summary_decision = summary.get("decision", {})
    next_decision = next_action.get("decision", {})
    public_gap_decision = public_gap.get("decision", {})
    data_decision = data_coverage.get("decision", {})
    prelaunch_decision = prelaunch.get("decision", {})
    counts = summary.get("counts", {})
    proxy = best_proxy(summary)
    min_proxy = proxy["min_proxy"]

    output_pending = bool(summary_decision.get("output_pending") is True or next_decision.get("output_pending") is True)
    parse_or_missing_problem = bool(
        not summary_status["parse_ok"]
        or not next_action_status["parse_ok"]
        or not public_gap_status["parse_ok"]
        or not data_coverage_status["parse_ok"]
        or not prelaunch_status["parse_ok"]
        or counts.get("missing_reports", 0)
        or counts.get("parse_errors", 0)
    )
    forbidden_problem = bool(
        counts.get("forbidden_action_rows", 0)
        or summary_decision.get("create_submission_artifact") is True
        or summary_decision.get("submit_now") is True
        or next_decision.get("create_submission_artifact") is True
        or next_decision.get("submit_now") is True
    )
    public_gap_ready = public_gap_decision.get("public_local_gap_diagnosis_ready") is True
    data_coverage_ready = data_decision.get("data_source_coverage_ready") is True
    external_concat_blocked = data_decision.get("external_direct_concat_allowed") is False
    local_source_stack_closed = data_decision.get("local_source_stack_closed_for_submission_artifact") is True
    prelaunch_ready_or_pending_safe = (
        prelaunch_decision.get("prelaunch_ready_for_explicit_approval") is True
        and prelaunch_decision.get("execute_kaggle_now") is False
    )
    next_allows_review = next_decision.get("allow_separate_artifact_contract_review") is True
    summary_public_058 = summary_decision.get("public_058_evidence_present") is True
    summary_beats_id178 = summary_decision.get("beats_id178_public_anchor_evidence_present") is True
    proxy_le_public_058 = bool(min_proxy is not None and min_proxy <= PUBLIC_058_TARGET)
    proxy_lt_id178 = bool(min_proxy is not None and min_proxy < ID178_PUBLIC_ANCHOR)
    public_058_signal_consistency_problem = bool(summary_public_058 and not proxy_le_public_058)
    selected_targets = proxy.get("selected_targets")
    selected_targets_present = isinstance(selected_targets, list) and len(selected_targets) > 0

    public_058_artifact_review_allowed = bool(
        not output_pending
        and not parse_or_missing_problem
        and not forbidden_problem
        and not public_058_signal_consistency_problem
        and public_gap_ready
        and data_coverage_ready
        and external_concat_blocked
        and local_source_stack_closed
        and prelaunch_ready_or_pending_safe
        and next_allows_review
        and summary_public_058
        and summary_beats_id178
        and proxy_le_public_058
        and selected_targets_present
    )
    id178_improvement_only = bool(
        not output_pending
        and not parse_or_missing_problem
        and not forbidden_problem
        and not public_058_signal_consistency_problem
        and next_allows_review
        and summary_beats_id178
        and proxy_lt_id178
        and not proxy_le_public_058
        and selected_targets_present
    )

    if output_pending:
        next_step = "Wait for approved no-submit Kaggle outputs; no artifact review can open yet."
    elif public_058_artifact_review_allowed:
        next_step = (
            "Open a separate Public-0.58 artifact-contract review. This still creates no CSV until a later "
            "review names the exact artifact, notebook, base predictions, and submission-log row."
        )
    elif id178_improvement_only:
        next_step = (
            "Remote evidence may beat ID178 but does not clear Public 0.58; record it as an improvement lead, "
            "not a Public-0.58 artifact review."
        )
    elif public_058_signal_consistency_problem:
        next_step = "Do not open artifact review; remote Public-0.58 signal conflicts with the numeric proxy threshold."
    elif min_proxy is not None and not selected_targets_present:
        next_step = "Do not open artifact review; best proxy row lacks selected targets for a reproducible artifact."
    elif data_coverage_ready and external_concat_blocked and not local_source_stack_closed:
        next_step = "Do not open artifact review; local source stack is not closed for submission artifacts."
    else:
        next_step = "Do not open artifact review; close or redesign the remote lane after reviewing diagnostics."

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_remote_artifact_review_threshold",
        "purpose": "Apply the strict Public 0.58 artifact-review threshold to pulled no-submit remote evidence.",
        "inputs": {
            "remote_summary": summary_status,
            "remote_next_action": next_action_status,
            "public_local_gap": public_gap_status,
            "data_source_coverage": data_coverage_status,
            "prelaunch": prelaunch_status,
        },
        "thresholds": {
            "public_058_target": PUBLIC_058_TARGET,
            "id178_public_anchor": ID178_PUBLIC_ANCHOR,
        },
        "best_proxy": proxy,
        "evidence_flags": {
            "output_pending": output_pending,
            "parse_or_missing_problem": parse_or_missing_problem,
            "forbidden_problem": forbidden_problem,
            "public_gap_ready": public_gap_ready,
            "data_coverage_ready": data_coverage_ready,
            "external_concat_blocked": external_concat_blocked,
            "local_source_stack_closed": local_source_stack_closed,
            "prelaunch_ready_or_pending_safe": prelaunch_ready_or_pending_safe,
            "next_allows_review": next_allows_review,
            "summary_public_058_evidence_present": summary_public_058,
            "summary_beats_id178_evidence_present": summary_beats_id178,
            "proxy_le_public_058": proxy_le_public_058,
            "proxy_lt_id178_public": proxy_lt_id178,
            "selected_targets_present": selected_targets_present,
            "public_058_signal_consistency_problem": public_058_signal_consistency_problem,
            "id178_improvement_only": id178_improvement_only,
        },
        "decision": {
            "threshold_gate_ready": bool(
                summary_status["parse_ok"]
                and next_action_status["parse_ok"]
                and public_gap_status["parse_ok"]
                and data_coverage_status["parse_ok"]
                and prelaunch_status["parse_ok"]
            ),
            "output_pending": output_pending,
            "allow_public_058_artifact_contract_review": public_058_artifact_review_allowed,
            "record_id178_improvement_lead_only": id178_improvement_only,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": next_step,
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


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.9f}"
    return str(value)


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    proxy = report["best_proxy"]
    lines = [
        "# Post-ID341 remote artifact-review threshold",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It applies the strict Public `0.58` threshold to pulled no-submit remote evidence.",
        "",
        "## Decision",
        "",
        f"- Threshold gate ready: `{decision['threshold_gate_ready']}`",
        f"- Output pending: `{decision['output_pending']}`",
        f"- Allow Public 0.58 artifact-contract review: `{decision['allow_public_058_artifact_contract_review']}`",
        f"- Record ID178 improvement lead only: `{decision['record_id178_improvement_lead_only']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Best proxy",
        "",
        f"- Filename: `{proxy['filename']}`",
        f"- Min proxy: `{fmt(proxy['min_proxy'])}`",
        f"- Delta vs Public 0.58: `{fmt(proxy['delta_vs_public_058'])}`",
        f"- Delta vs ID178 Public: `{fmt(proxy['delta_vs_id178_public'])}`",
        "",
        "## Evidence flags",
        "",
    ]
    for key, value in report["evidence_flags"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-summary", type=Path, default=DEFAULT_REMOTE_SUMMARY)
    parser.add_argument("--remote-next-action", type=Path, default=DEFAULT_NEXT_ACTION)
    parser.add_argument("--public-gap", type=Path, default=DEFAULT_PUBLIC_GAP)
    parser.add_argument("--data-coverage", type=Path, default=DEFAULT_DATA_COVERAGE)
    parser.add_argument("--prelaunch", type=Path, default=DEFAULT_PRELAUNCH)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_remote_artifact_review_threshold_20260618.json",
    )
    args = parser.parse_args()
    paths = {
        "remote_summary_path": args.remote_summary if args.remote_summary.is_absolute() else ROOT / args.remote_summary,
        "next_action_path": args.remote_next_action
        if args.remote_next_action.is_absolute()
        else ROOT / args.remote_next_action,
        "public_gap_path": args.public_gap if args.public_gap.is_absolute() else ROOT / args.public_gap,
        "data_coverage_path": args.data_coverage if args.data_coverage.is_absolute() else ROOT / args.data_coverage,
        "prelaunch_path": args.prelaunch if args.prelaunch.is_absolute() else ROOT / args.prelaunch,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(**paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {safe_relative(output)}")
    print(f"Wrote {safe_relative(output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
