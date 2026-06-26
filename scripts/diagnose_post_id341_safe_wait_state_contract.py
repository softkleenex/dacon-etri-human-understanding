#!/usr/bin/env python3
"""Build the post-ID341 safe-wait state contract.

This report-only diagnostic is the top-level stoplight after the ID178/287/329
public evidence and the ID341/340/337 micro-probe results. It proves that the
current workspace is in a safe waiting state: no local no-approval score route
is open, the manual micro-probe family is consumed, Kaggle work still requires
the exact no-submit approval phrase, and all artifact/submission/log mutations
stay blocked.

It creates no predictions, submission CSVs, base prediction artifacts,
notebook/history artifacts, submission-log rows, Kaggle pushes/runs, or DACON
submissions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
SUBMISSIONS_DIR = ROOT / "submissions"
NOTEBOOKS_DIR = ROOT / "notebooks"
KST = timezone(timedelta(hours=9))
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_safe_wait_state_contract_20260619.json"

KAGGLE_APPROVAL_PHRASE = "Kaggle upstream rebuild no-submit 실행 승인"
HEAVY_APPROVAL_PHRASE = "Self-supervised subject-state heavy compute no-submit 실행 승인"
PRIMARY_SCORE_ROUTE = "explicit_approved_clean_upstream_id221_to_id226"
MANUAL_PROBE_ID = 341
MANUAL_PROBE_FILE = "submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv"
RECORDED_ID342 = 342
RECORDED_ID342_FILE = "submission_id_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.csv"
RECORDED_ID342_ARTIFACT_NAMES = {
    RECORDED_ID342_FILE,
    "submission_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.ipynb",
    "base_oof_id_342_id178_id221_featuresim_shrink050_cv0.534000.csv",
    "base_test_id_342_id178_id221_featuresim_shrink050_cv0.534000.csv",
    "decision_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.md",
    "experiment_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.json",
    "joint_risk_candidate_rerank_post_id342_20260623.json",
    "joint_risk_candidate_rerank_post_id342_20260623.md",
    "post_id342_transfer_gate_v2_joint_risk_20260623.json",
    "post_id342_transfer_gate_v2_joint_risk_20260623.md",
    "public_joint_distribution_risk_post_id342_20260623.json",
    "public_joint_distribution_risk_post_id342_20260623.md",
}
EXPECTED_COLUMNS = ["subject_id", "sleep_date", "lifelog_date", "Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

REPORT_PATHS = {
    "no_approval_frontier": "post_id341_no_approval_frontier_20260619.json",
    "manual_probe_decision": "post_id341_manual_submission_probe_decision_20260618.json",
    "operator_handoff": "post_id341_operator_handoff_contract_20260619.json",
    "prelaunch_freshness": "post_id341_prelaunch_freshness_contract_20260619.json",
    "public058_promotion": "post_id341_public058_artifact_promotion_contract_20260619.json",
    "self_supervised_heavy": "post_id341_self_supervised_heavy_compute_contract_20260619.json",
    "material_source_remote_coverage": "post_id341_material_source_remote_coverage_contract_20260619.json",
}


def rel(path: Path) -> str:
    path = path if path.is_absolute() else ROOT / path
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_report(name: str) -> dict[str, Any]:
    path = REPORTS_DIR / name
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "decision": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
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
        "payload": payload,
    }


def decision(payloads: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    return payloads[name].get("decision", {})


def flag(payloads: dict[str, dict[str, Any]], name: str, key: str, expected: Any = True) -> bool:
    return decision(payloads, name).get(key) is expected


def sha256_12(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def csv_shape_and_columns(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "shape": None, "columns_ok": False, "row_count_ok": False}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            header = []
            rows: list[list[str]] = []
        else:
            rows = list(reader)
    return {
        "path": rel(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256_12": sha256_12(path),
        "shape": [len(rows), len(header)],
        "columns": header,
        "columns_ok": header == EXPECTED_COLUMNS,
        "row_count_ok": len(rows) == 250,
        "has_blank_cells": any(any(cell == "" for cell in row) for row in rows),
        "duplicate_key_rows": len(rows) - len({tuple(row[:3]) for row in rows}),
    }


def submission_log_snapshot() -> dict[str, Any]:
    path = SUBMISSIONS_DIR / "submission_log.csv"
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "rows": 0, "max_id": None}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids: list[int] = []
    manual_rows: list[dict[str, str]] = []
    recorded_id342_rows: list[dict[str, str]] = []
    for row in rows:
        try:
            row_id = int(row.get("id", ""))
        except ValueError:
            continue
        ids.append(row_id)
        if row_id == RECORDED_ID342:
            recorded_id342_rows.append(row)
        if row.get("submission_file") == MANUAL_PROBE_FILE:
            manual_rows.append(row)
    max_id = max(ids) if ids else None
    ids_are_unique = len(ids) == len(set(ids))
    manual_row = manual_rows[-1] if manual_rows else {}
    public_value = str(manual_row.get("public_lb_score_num", "")).strip().lower()
    id342_row = recorded_id342_rows[-1] if recorded_id342_rows else {}
    id342_public_value = str(id342_row.get("public_lb_score_num", "")).strip().lower()
    recorded_id342_row_ok = bool(
        len(recorded_id342_rows) == 1
        and id342_row.get("submission_file") == RECORDED_ID342_FILE
        and str(id342_row.get("status_clean", "")).lower() == "generated_unsubmitted"
        and str(id342_row.get("is_submitted_bool", "")).lower() == "false"
        and id342_public_value in {"", "nan"}
        and str(id342_row.get("notebook_file", "")).endswith(".ipynb")
    )
    return {
        "path": rel(path),
        "exists": True,
        "parse_ok": True,
        "rows": len(rows),
        "max_id": max_id,
        "ids_are_unique": ids_are_unique,
        "recorded_id342_row_count": len(recorded_id342_rows),
        "recorded_id342_row_ok": recorded_id342_row_ok,
        "recorded_id342_file": id342_row.get("submission_file"),
        "post_manual_submission_ids": sorted(row_id for row_id in ids if row_id > MANUAL_PROBE_ID),
        "ready_for_existing_id342_state": bool(ids_are_unique and max_id == RECORDED_ID342 and recorded_id342_row_ok),
        "manual_probe_row_count": len(manual_rows),
        "manual_probe_row_ok": bool(
            len(manual_rows) == 1
            and str(manual_row.get("id")) == str(MANUAL_PROBE_ID)
            and public_value in {"", "nan"}
            and str(manual_row.get("notebook_file", "")).endswith(".ipynb")
        ),
        "manual_probe_fresh_for_submission": bool(str(manual_row.get("is_submitted_bool", "")).lower() == "false"),
        "manual_probe_already_submitted": bool(str(manual_row.get("is_submitted_bool", "")).lower() == "true"),
        "manual_probe_notebook_file": manual_row.get("notebook_file"),
        "next_numeric_id_if_later_authorized": None if max_id is None else max_id + 1,
    }


def is_controlled_id342_artifact(path: Path) -> bool:
    """Return True only for ID342-like generated competition artifacts.

    Raw/external data contains timestamps with the digits 342; those are not
    submission IDs. Keep this scoped to controlled artifact names.
    """

    name = path.name.lower()
    return (
        "id_342" in name
        or "id342" in name
        or name.startswith("submission_342")
        or name.startswith("submission_id_342")
        or name.startswith("experiment_342")
    )


def is_recorded_id342_artifact(path: Path) -> bool:
    return path.name in RECORDED_ID342_ARTIFACT_NAMES


def path_inventory(log_snapshot: dict[str, Any]) -> dict[str, Any]:
    submission_path = SUBMISSIONS_DIR / MANUAL_PROBE_FILE
    notebook_file = log_snapshot.get("manual_probe_notebook_file")
    notebook_path = (
        ROOT / notebook_file if isinstance(notebook_file, str) and notebook_file else NOTEBOOKS_DIR / "missing"
    )
    controlled_roots = [
        SUBMISSIONS_DIR,
        ROOT / "notebooks" / "history",
        ROOT / "artifacts" / "base_predictions",
        REPORTS_DIR,
    ]
    candidate_342_paths: list[str] = []
    for controlled_root in controlled_roots:
        if not controlled_root.exists():
            continue
        candidate_342_paths.extend(
            rel(path) for path in controlled_root.rglob("*") if path.is_file() and is_controlled_id342_artifact(path)
        )
    unexpected_342_paths = [path for path in candidate_342_paths if not is_recorded_id342_artifact(ROOT / path)]
    expected_342_paths = [
        SUBMISSIONS_DIR / RECORDED_ID342_FILE,
        ROOT / "notebooks" / "history" / "submission_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.ipynb",
        ROOT / "artifacts" / "base_predictions" / "base_oof_id_342_id178_id221_featuresim_shrink050_cv0.534000.csv",
        ROOT / "artifacts" / "base_predictions" / "base_test_id_342_id178_id221_featuresim_shrink050_cv0.534000.csv",
        REPORTS_DIR / "decision_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.md",
        REPORTS_DIR / "experiment_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.json",
    ]
    missing_expected_342_paths = [rel(path) for path in expected_342_paths if not path.exists()]
    return {
        "manual_probe_submission_csv": csv_shape_and_columns(submission_path),
        "manual_probe_notebook": {
            "path": rel(notebook_path),
            "exists": notebook_path.exists(),
            "bytes": notebook_path.stat().st_size if notebook_path.exists() else None,
            "sha256_12": sha256_12(notebook_path),
        },
        "id342_like_controlled_artifact_paths": sorted(candidate_342_paths),
        "unexpected_id342_like_controlled_artifact_paths": sorted(unexpected_342_paths),
        "missing_expected_id342_artifact_paths": missing_expected_342_paths,
        "id342_controlled_artifacts_accounted_for": len(unexpected_342_paths) == 0,
        "recorded_id342_core_artifacts_exist": len(missing_expected_342_paths) == 0,
    }


def action_flags_false(decision_row: dict[str, Any]) -> bool:
    action_keys = [
        "create_submission_artifact",
        "submit_now",
        "push_kaggle_now",
        "run_kaggle_now",
        "create_id342_or_new_submission_id",
        "modify_submission_log",
        "train_model_now",
        "feature_matrix_allowed_now",
        "artifact_promotion_allowed_now",
    ]
    return all(decision_row.get(key) is False for key in action_keys if key in decision_row)


def build_report() -> dict[str, Any]:
    payloads = {key: load_report(name) for key, name in REPORT_PATHS.items()}
    all_reports_ready = all(payload.get("exists") and payload.get("parse_ok") for payload in payloads.values())
    log = submission_log_snapshot()
    paths = path_inventory(log)

    source_decision_checks = {
        "no_approval_frontier_ready": flag(payloads, "no_approval_frontier", "no_approval_frontier_ready", True),
        "all_local_no_approval_score_routes_closed": flag(
            payloads, "no_approval_frontier", "all_local_no_approval_score_routes_closed", True
        ),
        "no_approval_score_route_exists_false": flag(
            payloads, "no_approval_frontier", "no_approval_score_route_exists", False
        ),
        "frontier_route_is_upstream_rebuild": decision(payloads, "no_approval_frontier").get(
            "only_score_relevant_open_route"
        )
        == PRIMARY_SCORE_ROUTE,
        "manual_probe_decision_ready": flag(
            payloads, "manual_probe_decision", "manual_submission_probe_decision_ready", True
        ),
        "manual_probe_is_id341": decision(payloads, "manual_probe_decision").get("lowest_risk_probe_id")
        == MANUAL_PROBE_ID,
        "manual_probe_file_matches": decision(payloads, "manual_probe_decision").get("lowest_risk_probe_file")
        == MANUAL_PROBE_FILE,
        "manual_probe_contract_closed_after_submission": flag(
            payloads, "manual_probe_decision", "lowest_risk_probe_artifact_contract_ready", False
        ),
        "manual_probe_not_public058": flag(
            payloads, "manual_probe_decision", "public_058_plausible_candidate_exists", False
        )
        and flag(payloads, "no_approval_frontier", "manual_probe_is_public_058_evidence", False),
        "operator_handoff_ready": flag(payloads, "operator_handoff", "operator_handoff_ready", True),
        "operator_safe_wait": flag(payloads, "operator_handoff", "safe_wait_state_without_user_action", True),
        "prelaunch_fresh": flag(payloads, "prelaunch_freshness", "prelaunch_freshness_contract_pass", True),
        "prelaunch_ready_for_approval": flag(
            payloads, "prelaunch_freshness", "prelaunch_ready_for_explicit_approval", True
        ),
        "public058_promotion_closed": flag(payloads, "public058_promotion", "promotion_contract_ready", True)
        and flag(payloads, "public058_promotion", "artifact_promotion_allowed_now", False)
        and flag(payloads, "public058_promotion", "create_id342_or_new_submission_id", False),
        "heavy_compute_closed_without_separate_approval": flag(
            payloads, "self_supervised_heavy", "heavy_compute_contract_ready", True
        )
        and flag(payloads, "self_supervised_heavy", "heavy_compute_allowed_now", False),
        "material_source_remote_coverage_ready": flag(
            payloads, "material_source_remote_coverage", "material_source_remote_coverage_contract_pass", True
        ),
    }

    artifact_checks = {
        "submission_log_ids_are_unique": bool(log.get("ids_are_unique")),
        "submission_log_ready_for_existing_id342_state": bool(log.get("ready_for_existing_id342_state")),
        "submission_log_has_single_recorded_id341_probe_row": bool(log.get("manual_probe_row_ok")),
        "manual_probe_is_not_fresh_for_resubmission": bool(not log.get("manual_probe_fresh_for_submission")),
        "recorded_id342_row_is_generated_unsubmitted": bool(log.get("recorded_id342_row_ok")),
        "manual_probe_csv_exists": bool(paths["manual_probe_submission_csv"].get("exists")),
        "manual_probe_csv_shape_ok": bool(
            paths["manual_probe_submission_csv"].get("shape") == [250, 10]
            and paths["manual_probe_submission_csv"].get("columns_ok")
            and paths["manual_probe_submission_csv"].get("row_count_ok")
            and not paths["manual_probe_submission_csv"].get("has_blank_cells")
            and paths["manual_probe_submission_csv"].get("duplicate_key_rows") == 0
        ),
        "manual_probe_notebook_exists": bool(paths["manual_probe_notebook"].get("exists")),
        "id342_controlled_artifacts_accounted_for": bool(paths["id342_controlled_artifacts_accounted_for"]),
        "recorded_id342_core_artifacts_exist": bool(paths["recorded_id342_core_artifacts_exist"]),
    }

    action_guard_checks = {
        key: action_flags_false(payload["decision"]) for key, payload in payloads.items() if payload.get("parse_ok")
    }
    action_guard_checks.update(
        {
            "kaggle_approval_phrase_exact": decision(payloads, "operator_handoff").get("approval_phrase")
            == KAGGLE_APPROVAL_PHRASE,
            "frontier_approval_phrase_exact": decision(payloads, "no_approval_frontier").get("approval_phrase")
            == KAGGLE_APPROVAL_PHRASE,
            "heavy_route_uses_separate_phrase": decision(payloads, "self_supervised_heavy").get("approval_phrase")
            == HEAVY_APPROVAL_PHRASE,
        }
    )

    safe_wait_state_ready = bool(
        all_reports_ready
        and all(source_decision_checks.values())
        and all(artifact_checks.values())
        and all(action_guard_checks.values())
    )

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_safe_wait_state_contract",
        "report_only": True,
        "purpose": (
            "Prove the current post-ID341 workspace state is safe to leave idle or continue report-only work: "
            "the ID341/340/337 micro-probe family is consumed, no Public-0.58 candidate is reviewable, "
            "the existing ID342 diagnostic artifact is not a score route, and any score-relevant execution still "
            "requires exact user approval."
        ),
        "inputs": {key: {k: v for k, v in payload.items() if k != "payload"} for key, payload in payloads.items()},
        "submission_log_snapshot": log,
        "artifact_inventory": paths,
        "source_decision_checks": source_decision_checks,
        "artifact_checks": artifact_checks,
        "action_guard_checks": action_guard_checks,
        "decision": {
            "safe_wait_state_contract_ready": bool(all_reports_ready),
            "safe_wait_state_ready": safe_wait_state_ready,
            "default_action": "do_not_submit",
            "score_relevant_open_route": PRIMARY_SCORE_ROUTE,
            "requires_exact_kaggle_approval_phrase": True,
            "kaggle_approval_phrase": KAGGLE_APPROVAL_PHRASE,
            "separate_heavy_compute_phrase_required": True,
            "heavy_compute_approval_phrase": HEAVY_APPROVAL_PHRASE,
            "manual_probe_if_user_spends_submission": None,
            "manual_probe_id": None,
            "manual_probe_sha256_12": paths["manual_probe_submission_csv"].get("sha256_12"),
            "manual_probe_is_public_058_evidence": False,
            "public_058_reviewable": False,
            "create_submission_artifact": False,
            "modify_submission_log": False,
            "create_id342_or_new_submission_id": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Do not submit or create a new submission ID. The ID341/337/340 micro-probes have already been "
                "spent and remain near 0.60, and existing ID342 is diagnostic-only. For score-relevant work, use "
                "only the registered upstream rebuild review chain."
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
    decision_row = report["decision"]
    csv_info = report["artifact_inventory"]["manual_probe_submission_csv"]
    lines = [
        "# Post-ID341 safe-wait state contract",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Safe-wait state ready: `{decision_row['safe_wait_state_ready']}`",
        f"- Default action: `{decision_row['default_action']}`",
        f"- Score-relevant open route: `{decision_row['score_relevant_open_route']}`",
        f"- Kaggle approval phrase: `{decision_row['kaggle_approval_phrase']}`",
        f"- Separate heavy-compute phrase required: `{decision_row['separate_heavy_compute_phrase_required']}`",
        f"- Manual probe if user spends submission: `{decision_row['manual_probe_if_user_spends_submission']}`",
        f"- Manual probe hash: `{decision_row['manual_probe_sha256_12']}`",
        f"- Manual probe is Public 0.58 evidence: `{decision_row['manual_probe_is_public_058_evidence']}`",
        f"- Public 0.58 reviewable: `{decision_row['public_058_reviewable']}`",
        f"- Create submission artifact: `{decision_row['create_submission_artifact']}`",
        f"- Modify submission log: `{decision_row['modify_submission_log']}`",
        f"- Create new submission ID: `{decision_row['create_id342_or_new_submission_id']}`",
        f"- Submit now: `{decision_row['submit_now']}`",
        f"- Push Kaggle now: `{decision_row['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision_row['run_kaggle_now']}`",
        f"- Next action: {decision_row['next_action']}",
        "",
        "## Manual probe artifact",
        "",
        f"- CSV: `{csv_info['path']}`",
        f"- Shape: `{csv_info.get('shape')}`",
        f"- Hash prefix: `{csv_info.get('sha256_12')}`",
        "",
        "## Source decision checks",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in report["source_decision_checks"].items())
    lines.extend(["", "## Artifact checks", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in report["artifact_checks"].items())
    lines.extend(["", "## Action guard checks", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in report["action_guard_checks"].items())
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
