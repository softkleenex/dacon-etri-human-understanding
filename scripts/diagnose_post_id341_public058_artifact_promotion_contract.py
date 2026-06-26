#!/usr/bin/env python3
"""Report-only promotion contract after a Public-0.58 remote threshold gate.

The remote threshold gate may, in a future approved no-submit run, allow a
separate artifact-contract review. That still must not create a submission CSV,
base predictions, notebook history, submission-log row, Kaggle run, or DACON
submission by itself. This diagnostic freezes that boundary and records the
minimum lineage fields required before any future promotion can be considered.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
SUBMISSIONS_DIR = ROOT / "submissions"
KST = timezone(timedelta(hours=9))

DEFAULT_THRESHOLD = REPORTS_DIR / "post_id341_remote_artifact_review_threshold_20260618.json"
DEFAULT_NEXT_ACTION = REPORTS_DIR / "post_id341_remote_next_action_decision_20260618.json"
DEFAULT_REMOTE_SUMMARY = REPORTS_DIR / "post_id341_kaggle_remote_result_summary_20260618.json"
DEFAULT_SUBMISSION_LOG = SUBMISSIONS_DIR / "submission_log.csv"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_public058_artifact_promotion_contract_20260619.json"

REQUIRED_LINEAGE_FIELDS = [
    "candidate_submission_id",
    "unique_submission_csv_path",
    "unique_notebook_history_path",
    "base_oof_prediction_path",
    "base_test_prediction_path",
    "source_remote_report_path",
    "source_threshold_report_path",
    "local_cv_score",
    "remote_min_proxy",
    "selected_targets",
    "prediction_hash",
    "submission_log_row",
    "exact_user_csv_approval",
]


def rel(path: Path) -> str:
    path = path if path.is_absolute() else ROOT / path
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def status_for(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "decision": {}}
    try:
        payload = load_json(path)
    except Exception as exc:  # noqa: BLE001
        return {
            "path": rel(path),
            "exists": True,
            "parse_ok": False,
            "bytes": path.stat().st_size,
            "decision": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "path": rel(path),
        "exists": True,
        "parse_ok": True,
        "bytes": path.stat().st_size,
        "decision": payload.get("decision", {}),
    }


def submission_log_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "rows": 0}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids: list[int] = []
    for row in rows:
        try:
            ids.append(int(row.get("id", "")))
        except ValueError:
            pass
    max_id = max(ids) if ids else None
    ids_are_unique = len(ids) == len(set(ids))
    return {
        "path": rel(path),
        "exists": True,
        "parse_ok": True,
        "rows": len(rows),
        "max_id": max_id,
        "has_id342_row": 342 in ids,
        "ids_are_unique": ids_are_unique,
        "ready_for_future_unique_id": bool(ids_are_unique and max_id is not None),
        "next_numeric_id_if_later_authorized": None if max_id is None else max_id + 1,
    }


def build_report(
    threshold_path: Path = DEFAULT_THRESHOLD,
    next_action_path: Path = DEFAULT_NEXT_ACTION,
    remote_summary_path: Path = DEFAULT_REMOTE_SUMMARY,
    submission_log_path: Path = DEFAULT_SUBMISSION_LOG,
) -> dict[str, Any]:
    threshold_status = status_for(threshold_path)
    next_action_status = status_for(next_action_path)
    remote_summary_status = status_for(remote_summary_path)
    log_status = submission_log_status(submission_log_path)

    threshold = load_json(threshold_path) if threshold_status["parse_ok"] else {}
    next_action = load_json(next_action_path) if next_action_status["parse_ok"] else {}
    remote_summary = load_json(remote_summary_path) if remote_summary_status["parse_ok"] else {}
    threshold_decision = threshold.get("decision", {})
    next_decision = next_action.get("decision", {})
    summary_decision = remote_summary.get("decision", {})
    best_proxy = threshold.get("best_proxy", {}) or {}

    threshold_gate_ready = threshold_decision.get("threshold_gate_ready") is True
    threshold_allows_review = threshold_decision.get("allow_public_058_artifact_contract_review") is True
    next_allows_review = next_decision.get("allow_separate_artifact_contract_review") is True
    output_pending = bool(
        threshold_decision.get("output_pending") is True or next_decision.get("output_pending") is True
    )
    remote_summary_reviewable = summary_decision.get("remote_summary_reviewable") is True
    summary_public_058 = summary_decision.get("public_058_evidence_present") is True
    summary_beats_id178 = summary_decision.get("beats_id178_public_anchor_evidence_present") is True
    log_clean = log_status.get("ready_for_future_unique_id") is True
    inputs_parse = bool(
        threshold_status["parse_ok"]
        and next_action_status["parse_ok"]
        and remote_summary_status["parse_ok"]
        and log_status["parse_ok"]
    )
    review_open = bool(
        inputs_parse
        and threshold_gate_ready
        and threshold_allows_review
        and next_allows_review
        and not output_pending
        and remote_summary_reviewable
        and summary_public_058
        and summary_beats_id178
        and log_clean
    )
    contract_ready = bool(inputs_parse and threshold_gate_ready and log_clean)

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_public058_artifact_promotion_contract",
        "report_only": True,
        "purpose": (
            "Separate Public-0.58 artifact-contract review from actual artifact promotion. "
            "A review may inspect lineage, but this diagnostic never creates CSVs, predictions, notebooks, "
            "submission-log rows, Kaggle runs, or DACON submissions."
        ),
        "inputs": {
            "threshold": threshold_status,
            "remote_next_action": next_action_status,
            "remote_summary": remote_summary_status,
            "submission_log": log_status,
        },
        "required_lineage_fields_before_future_promotion": REQUIRED_LINEAGE_FIELDS,
        "future_promotion_invariants": {
            "all_filenames_must_be_unique": True,
            "csv_notebook_base_oof_base_test_log_row_must_match": True,
            "candidate_id_must_be_greater_than_current_max_id": True,
            "exact_user_csv_approval_required": True,
            "dacon_submission_requires_separate_exact_approval": True,
            "public_058_review_is_not_submission_permission": True,
        },
        "best_proxy": {
            "filename": best_proxy.get("filename"),
            "min_proxy": best_proxy.get("min_proxy"),
            "selected_targets": best_proxy.get("selected_targets"),
        },
        "evidence_flags": {
            "inputs_parse": inputs_parse,
            "threshold_gate_ready": threshold_gate_ready,
            "threshold_allows_public058_review": threshold_allows_review,
            "next_allows_artifact_contract_review": next_allows_review,
            "output_pending": output_pending,
            "remote_summary_reviewable": remote_summary_reviewable,
            "summary_public_058_evidence_present": summary_public_058,
            "summary_beats_id178_public_anchor": summary_beats_id178,
            "submission_log_ready_for_future_unique_id": log_clean,
            "public058_artifact_contract_review_open": review_open,
        },
        "decision": {
            "promotion_contract_ready": contract_ready,
            "public058_artifact_contract_review_open": review_open,
            "artifact_promotion_allowed_now": False,
            "requires_separate_exact_candidate_approval": True,
            "requires_exact_csv_notebook_base_log_lineage": True,
            "create_submission_artifact": False,
            "create_base_prediction_artifacts": False,
            "create_notebook_history_artifact": False,
            "modify_submission_log": False,
            "create_id342_or_new_submission_id": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Public-0.58 artifact review is open, but promotion is still blocked until a separate exact "
                "candidate contract names every lineage artifact and receives explicit CSV approval."
                if review_open
                else "Do not promote artifacts. Keep waiting for approved no-submit remote evidence or repair gates."
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
    decision = report["decision"]
    flags = report["evidence_flags"]
    log_status = report["inputs"]["submission_log"]
    lines = [
        "# Post-ID341 Public-0.58 artifact promotion contract",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        (
            "This is report-only. A Public-0.58 artifact review, if opened later, is not permission to create a CSV, "
            "base predictions, notebook history, submission-log row, Kaggle run, or DACON submission."
        ),
        "",
        "## Decision",
        "",
        f"- Promotion contract ready: `{decision['promotion_contract_ready']}`",
        f"- Public-0.58 artifact-contract review open: `{decision['public058_artifact_contract_review_open']}`",
        f"- Artifact promotion allowed now: `{decision['artifact_promotion_allowed_now']}`",
        f"- Requires separate exact candidate approval: `{decision['requires_separate_exact_candidate_approval']}`",
        f"- Requires exact CSV/notebook/base/log lineage: `{decision['requires_exact_csv_notebook_base_log_lineage']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Create base prediction artifacts: `{decision['create_base_prediction_artifacts']}`",
        f"- Create notebook history artifact: `{decision['create_notebook_history_artifact']}`",
        f"- Modify submission log: `{decision['modify_submission_log']}`",
        f"- Create ID342/new submission id: `{decision['create_id342_or_new_submission_id']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Submission log invariant",
        "",
        f"- Rows: `{log_status.get('rows')}`",
        f"- Max id: `{log_status.get('max_id')}`",
        f"- Has ID342 row: `{log_status.get('has_id342_row')}`",
        f"- IDs are unique: `{log_status.get('ids_are_unique')}`",
        f"- Ready for future unique id: `{log_status.get('ready_for_future_unique_id')}`",
        f"- Next numeric id if later authorized: `{log_status.get('next_numeric_id_if_later_authorized')}`",
        "",
        "## Required lineage fields before any future promotion",
        "",
    ]
    lines.extend(f"- `{field}`" for field in report["required_lineage_fields_before_future_promotion"])
    lines.extend(["", "## Evidence flags", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in flags.items())
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=Path, default=DEFAULT_THRESHOLD)
    parser.add_argument("--remote-next-action", type=Path, default=DEFAULT_NEXT_ACTION)
    parser.add_argument("--remote-summary", type=Path, default=DEFAULT_REMOTE_SUMMARY)
    parser.add_argument("--submission-log", type=Path, default=DEFAULT_SUBMISSION_LOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(
        threshold_path=args.threshold if args.threshold.is_absolute() else ROOT / args.threshold,
        next_action_path=args.remote_next_action
        if args.remote_next_action.is_absolute()
        else ROOT / args.remote_next_action,
        remote_summary_path=args.remote_summary if args.remote_summary.is_absolute() else ROOT / args.remote_summary,
        submission_log_path=args.submission_log if args.submission_log.is_absolute() else ROOT / args.submission_log,
    )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    write_report(report, output)
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
