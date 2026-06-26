#!/usr/bin/env python3
"""Summarize pulled post-ID341 Kaggle no-submit remote reports.

This report-only post-run diagnostic is intended to run after:
1. `make kaggle-wait-and-pull-upstream-rebuild-output`, and
2. `scripts/diagnose_post_id341_kaggle_output_intake_gate.py`.

It extracts remote gate signals, proxy scores, selected targets, and forbidden
actions from the pulled report bundle. It never creates predictions, submission
CSVs, base prediction artifacts, notebook history, submission-log rows, Kaggle
runs, or DACON submissions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DEFAULT_PULL_DIR = ROOT / "build" / "kaggle_outputs" / "upstream_rebuild_post_id341"
DEFAULT_INTAKE_REPORT = REPORTS_DIR / "post_id341_kaggle_output_intake_gate_20260618.json"
INTAKE_SCRIPT = ROOT / "scripts" / "diagnose_post_id341_kaggle_output_intake_gate.py"
KST = timezone(timedelta(hours=9))

ID178_PUBLIC_ANCHOR = 0.6014113009
ID341_REFERENCE_PROXY = 0.6038297835
PUBLIC_058_TARGET = 0.58
FORBIDDEN_ACTION_KEYS = ["create_submission_artifact", "submit_now", "push_kaggle_now", "run_kaggle_now"]
REVIEW_SIGNAL_KEYS = [
    "public_058_reviewable",
    "public_058_plausible",
    "public_058_plausible_candidate_exists",
    "any_candidate_passes_combined_public_058_gate",
    "beats_id178_public_anchor_proxy",
    "proxy_below_id178_candidate_exists",
    "final_block_a_gate_pass",
    "final_block_b_gate_pass",
    "final_usage_temporal_gate_pass",
]
PROXY_KEYS = [
    "best_proxy",
    "projected_public_proxy_from_id341",
]


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_intake_module() -> Any:
    spec = importlib.util.spec_from_file_location("post_id341_kaggle_output_intake_gate", INTAKE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load intake script: {INTAKE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_first(root: Path, filename: str) -> Path | None:
    if not root.exists():
        return None
    matches = sorted(root.rglob(filename))
    return matches[0] if matches else None


def extract_proxy_values(decision: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in PROXY_KEYS:
        value = decision.get(key)
        if isinstance(value, int | float):
            values[key] = float(value)
    return values


def summarize_remote_report(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    decision = payload.get("decision", {})
    proxy_values = extract_proxy_values(decision)
    min_proxy = min(proxy_values.values()) if proxy_values else None
    review_signals = {key: decision.get(key) for key in REVIEW_SIGNAL_KEYS if decision.get(key) is True}
    forbidden_actions = {key: decision.get(key) for key in FORBIDDEN_ACTION_KEYS if decision.get(key) is True}
    selected_targets = decision.get("selected_targets")
    if selected_targets is None and decision.get("selected_target_count") == 0:
        selected_targets = []
    return {
        "filename": path.name,
        "path": safe_relative(path),
        "proxy_values": proxy_values,
        "min_proxy": min_proxy,
        "delta_vs_id178_public_anchor": None if min_proxy is None else min_proxy - ID178_PUBLIC_ANCHOR,
        "delta_vs_public_058_target": None if min_proxy is None else min_proxy - PUBLIC_058_TARGET,
        "review_signals": review_signals,
        "forbidden_actions": forbidden_actions,
        "selected_targets": selected_targets,
        "selected_target_count": decision.get("selected_target_count"),
        "next_action": decision.get("next_action"),
    }


def build_report(pull_dir: Path, intake_report_path: Path) -> dict[str, Any]:
    intake_module = load_intake_module()
    if intake_report_path.exists():
        intake_report = load_json(intake_report_path)
    else:
        intake_report = intake_module.build_report(pull_dir)
    intake_decision = intake_report.get("decision", {})
    output_pending = intake_decision.get("output_pending") is True
    remote_rows: list[dict[str, Any]] = []
    missing_reports: list[str] = []
    parse_errors: list[dict[str, str]] = []
    if pull_dir.exists():
        for filename in intake_module.EXPECTED_REMOTE_REPORTS:
            path = find_first(pull_dir, filename)
            if path is None:
                missing_reports.append(filename)
                continue
            try:
                remote_rows.append(summarize_remote_report(path))
            except Exception as exc:  # noqa: BLE001
                parse_errors.append({"filename": filename, "error": f"{type(exc).__name__}: {exc}"})
    rows_with_proxy = [row for row in remote_rows if row["min_proxy"] is not None]
    best_proxy_row = min(rows_with_proxy, key=lambda row: row["min_proxy"]) if rows_with_proxy else None
    review_signal_rows = [row for row in remote_rows if row["review_signals"]]
    forbidden_action_rows = [row for row in remote_rows if row["forbidden_actions"]]
    proxy_below_id178_rows = [
        row for row in rows_with_proxy if row["min_proxy"] is not None and row["min_proxy"] < ID178_PUBLIC_ANCHOR
    ]
    proxy_below_058_rows = [
        row for row in rows_with_proxy if row["min_proxy"] is not None and row["min_proxy"] <= PUBLIC_058_TARGET
    ]
    artifact_review_allowed = intake_decision.get("artifact_contract_review_allowed") is True
    public_058_evidence_present = bool(
        proxy_below_058_rows or any("public_058" in key for row in review_signal_rows for key in row["review_signals"])
    )
    beats_id178_evidence_present = bool(
        proxy_below_id178_rows
        or any(
            key in {"beats_id178_public_anchor_proxy", "proxy_below_id178_candidate_exists"}
            for row in review_signal_rows
            for key in row["review_signals"]
        )
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_kaggle_remote_result_summary",
        "purpose": (
            "Summarize pulled no-submit Kaggle remote reports against ID178 Public, ID341 proxy, and Public 0.58 "
            "review thresholds. This report never creates artifacts or submissions."
        ),
        "pull_dir": safe_relative(pull_dir),
        "intake_report": safe_relative(intake_report_path),
        "anchors": {
            "id178_public_anchor": ID178_PUBLIC_ANCHOR,
            "id341_reference_proxy": ID341_REFERENCE_PROXY,
            "public_058_target": PUBLIC_058_TARGET,
        },
        "intake_decision": intake_decision,
        "counts": {
            "expected_remote_reports": len(intake_module.EXPECTED_REMOTE_REPORTS),
            "remote_reports_found": len(remote_rows),
            "missing_reports": len(missing_reports),
            "parse_errors": len(parse_errors),
            "reports_with_proxy": len(rows_with_proxy),
            "review_signal_rows": len(review_signal_rows),
            "forbidden_action_rows": len(forbidden_action_rows),
            "proxy_below_id178_rows": len(proxy_below_id178_rows),
            "proxy_below_058_rows": len(proxy_below_058_rows),
        },
        "best_proxy_row": best_proxy_row,
        "review_signal_rows": review_signal_rows,
        "forbidden_action_rows": forbidden_action_rows,
        "proxy_below_id178_rows": proxy_below_id178_rows,
        "proxy_below_058_rows": proxy_below_058_rows,
        "missing_reports": missing_reports,
        "parse_errors": parse_errors,
        "remote_report_summaries": remote_rows,
        "decision": {
            "output_pending": output_pending,
            "artifact_contract_review_allowed_by_intake": artifact_review_allowed,
            "remote_summary_reviewable": bool(
                artifact_review_allowed and not forbidden_action_rows and not parse_errors
            ),
            "public_058_evidence_present": public_058_evidence_present,
            "beats_id178_public_anchor_evidence_present": beats_id178_evidence_present,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Outputs are pending; run approved Kaggle no-submit path before summarizing remote evidence."
                if output_pending
                else (
                    "Remote reports may enter a separate artifact-contract review; this summary still creates nothing."
                    if artifact_review_allowed
                    else "Do not create artifacts; intake gate or remote evidence is insufficient."
                )
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


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.9f}"
    return str(value)


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    best = report["best_proxy_row"] or {}
    lines = [
        "# Post-ID341 Kaggle remote result summary",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It summarizes pulled remote reports after the output-intake gate; it creates no predictions, artifacts, logs, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- Output pending: `{decision['output_pending']}`",
        f"- Artifact-contract review allowed by intake: `{decision['artifact_contract_review_allowed_by_intake']}`",
        f"- Remote summary reviewable: `{decision['remote_summary_reviewable']}`",
        f"- Public 0.58 evidence present: `{decision['public_058_evidence_present']}`",
        f"- Beats ID178 public-anchor evidence present: `{decision['beats_id178_public_anchor_evidence_present']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Counts",
        "",
    ]
    for key, value in report["counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Best proxy row",
            "",
            f"- Filename: `{best.get('filename')}`",
            f"- Min proxy: `{fmt(best.get('min_proxy'))}`",
            f"- Delta vs ID178 Public: `{fmt(best.get('delta_vs_id178_public_anchor'))}`",
            f"- Delta vs Public 0.58: `{fmt(best.get('delta_vs_public_058_target'))}`",
            f"- Selected targets: `{best.get('selected_targets')}`",
            "",
            "## Review signal rows",
            "",
        ]
    )
    if report["review_signal_rows"]:
        for row in report["review_signal_rows"]:
            lines.append(f"- `{row['filename']}` signals `{row['review_signals']}` min_proxy `{fmt(row['min_proxy'])}`")
    else:
        lines.append("- None")
    lines.extend(["", "## Forbidden action rows", ""])
    if report["forbidden_action_rows"]:
        for row in report["forbidden_action_rows"]:
            lines.append(f"- `{row['filename']}` forbidden `{row['forbidden_actions']}`")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pull-dir", type=Path, default=DEFAULT_PULL_DIR)
    parser.add_argument("--intake-report", type=Path, default=DEFAULT_INTAKE_REPORT)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_kaggle_remote_result_summary_20260618.json",
    )
    args = parser.parse_args()
    pull_dir = args.pull_dir if args.pull_dir.is_absolute() else ROOT / args.pull_dir
    intake_report = args.intake_report if args.intake_report.is_absolute() else ROOT / args.intake_report
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(pull_dir, intake_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {safe_relative(output)}")
    print(f"Wrote {safe_relative(output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
