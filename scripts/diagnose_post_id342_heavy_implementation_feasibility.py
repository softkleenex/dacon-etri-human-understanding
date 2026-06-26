#!/usr/bin/env python3
"""Post-ID342 heavy implementation feasibility audit.

This report-only diagnostic checks whether the missingness-aware cross-modal
subject-state heavy route can be implemented from the current repo/runtime once
the exact heavy approval phrase is provided. It records dependency availability,
raw file/schema readiness, and the after-approval Make/runner gaps.

It does not create runner files, read raw parquet payloads, fit embeddings,
write feature matrices, train models, create predictions, edit submission logs,
push/run/pull Kaggle, or submit to DACON.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_heavy_implementation_feasibility_20260624.json"
KST = timezone(timedelta(hours=9))

HEAVY_APPROVAL_PHRASE = "Self-supervised subject-state heavy compute no-submit 실행 승인"

REPORT_PATHS = {
    "external_research_priority": REPORTS_DIR / "post_id342_external_research_priority_20260624.json",
    "runner_spec": REPORTS_DIR / "post_id342_self_supervised_heavy_runner_spec_20260624.json",
    "domain_token_contract": REPORTS_DIR / "post_id342_subject_state_domain_token_contract_20260624.json",
    "raw_schema_contract": REPORTS_DIR / "post_id342_subject_state_raw_schema_contract_20260624.json",
    "kaggle_cli_health": REPORTS_DIR / "post_id342_kaggle_cli_health_20260624.json",
}

RAW_FILES = [
    "data/raw/ch2025_data_items/ch2025_mACStatus.parquet",
    "data/raw/ch2025_data_items/ch2025_mActivity.parquet",
    "data/raw/ch2025_data_items/ch2025_mAmbience.parquet",
    "data/raw/ch2025_data_items/ch2025_mBle.parquet",
    "data/raw/ch2025_data_items/ch2025_mGps.parquet",
    "data/raw/ch2025_data_items/ch2025_mLight.parquet",
    "data/raw/ch2025_data_items/ch2025_mScreenStatus.parquet",
    "data/raw/ch2025_data_items/ch2025_mUsageStats.parquet",
    "data/raw/ch2025_data_items/ch2025_mWifi.parquet",
    "data/raw/ch2025_data_items/ch2025_wHr.parquet",
    "data/raw/ch2025_data_items/ch2025_wLight.parquet",
    "data/raw/ch2025_data_items/ch2025_wPedo.parquet",
]

CORE_DEPENDENCIES = [
    {"distribution": "numpy", "module": "numpy", "purpose": "array/tensor staging"},
    {"distribution": "pandas", "module": "pandas", "purpose": "metadata and label table handling"},
    {"distribution": "pyarrow", "module": "pyarrow", "purpose": "parquet schema/payload reader"},
    {"distribution": "polars", "module": "polars", "purpose": "streaming tabular feature/token staging"},
    {"distribution": "torch", "module": "torch", "purpose": "masked encoder implementation"},
    {
        "distribution": "pytorch-lightning",
        "module": "pytorch_lightning",
        "purpose": "optional training loop structure",
    },
    {"distribution": "scikit-learn", "module": "sklearn", "purpose": "fold-internal downstream heads"},
    {"distribution": "scipy", "module": "scipy", "purpose": "sparse/statistical utilities"},
    {"distribution": "joblib", "module": "joblib", "purpose": "small parallel preprocessing tasks"},
    {"distribution": "kaggle", "module": "kaggle", "purpose": "future no-submit remote push/status/pull"},
    {"distribution": "nbformat", "module": "nbformat", "purpose": "future notebook materialization"},
]

AFTER_APPROVAL_TARGETS = [
    "post-id342-self-supervised-subject-state-local-smoke",
    "kaggle-push-self-supervised-subject-state-runner",
    "kaggle-wait-and-pull-self-supervised-subject-state-output",
    "post-id342-self-supervised-remote-review",
]

AFTER_APPROVAL_FILES = [
    "notebooks/kaggle_self_supervised_subject_state_runner/kernel-metadata.json",
    "notebooks/kaggle_self_supervised_subject_state_runner/Self_Supervised_Subject_State_Post_ID342.ipynb",
    "scripts/run_post_id342_self_supervised_subject_state_no_submit.py",
]

FORBIDDEN_ACTIONS_NOW = {
    "create_runner_files": False,
    "read_raw_parquet_payloads": False,
    "write_token_cache": False,
    "write_feature_matrix": False,
    "fit_embedding": False,
    "train_model": False,
    "create_predictions": False,
    "create_submission_artifact": False,
    "modify_submission_log": False,
    "push_kaggle": False,
    "run_kaggle": False,
    "pull_kaggle_output": False,
    "submit_dacon": False,
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "decision": {}, "error": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "path": rel(path),
            "exists": True,
            "parse_ok": False,
            "decision": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "path": rel(path),
        "exists": True,
        "parse_ok": True,
        "diagnostic": payload.get("diagnostic"),
        "decision": payload.get("decision", {}),
    }


def decision(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("decision", {})
    return value if isinstance(value, dict) else {}


def declared_dependencies() -> set[str]:
    pyproject = ROOT / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    match = re.search(r"dependencies\s*=\s*\[(.*?)\]", text, flags=re.S)
    if not match:
        return set()
    return {item.strip().strip("\"'") for item in match.group(1).split(",") if item.strip()}


def dependency_rows() -> list[dict[str, Any]]:
    declared = declared_dependencies()
    rows: list[dict[str, Any]] = []
    for dep in CORE_DEPENDENCIES:
        distribution = dep["distribution"]
        installed = True
        version = None
        error = None
        try:
            version = metadata.version(distribution)
        except Exception as exc:  # noqa: BLE001
            installed = False
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                **dep,
                "declared": distribution in declared,
                "installed": installed,
                "version": version,
                "error": error,
            }
        )
    return rows


def make_target_rows() -> list[dict[str, Any]]:
    makefile_text = (ROOT / "Makefile").read_text(encoding="utf-8")
    rows = []
    for target in AFTER_APPROVAL_TARGETS:
        exists = re.search(rf"^{re.escape(target)}\s*:", makefile_text, flags=re.M) is not None
        rows.append(
            {
                "target": target,
                "exists_now": exists,
                "required_now": False,
                "required_after_exact_approval": True,
                "status": "present" if exists else "after_approval_gap",
            }
        )
    return rows


def raw_file_rows() -> list[dict[str, Any]]:
    return [{"path": path, "exists": (ROOT / path).exists(), "payload_read": False} for path in RAW_FILES]


def after_approval_file_rows() -> list[dict[str, Any]]:
    return [
        {
            "path": path,
            "exists_now": (ROOT / path).exists(),
            "create_now": False,
            "create_after_exact_approval": True,
        }
        for path in AFTER_APPROVAL_FILES
    ]


def build_prerequisites(reports: dict[str, dict[str, Any]]) -> dict[str, bool]:
    external = decision(reports["external_research_priority"])
    runner = decision(reports["runner_spec"])
    domain = decision(reports["domain_token_contract"])
    schema = decision(reports["raw_schema_contract"])
    cli = decision(reports["kaggle_cli_health"])
    deps = dependency_rows()
    raw_files = raw_file_rows()
    return {
        "input_reports_parse": all(report.get("parse_ok") is True for report in reports.values()),
        "external_research_priority_ready": external.get("external_research_priority_ready") is True,
        "runner_spec_ready": runner.get("runner_spec_ready") is True,
        "domain_token_contract_ready": domain.get("domain_token_contract_ready", domain.get("domain_contract_ready"))
        is True,
        "raw_schema_contract_ready": schema.get("raw_schema_contract_ready") is True,
        "raw_payload_not_read": schema.get("raw_payload_rows_read") == 0,
        "kaggle_cli_usable": cli.get("kaggle_cli_health_ready") is True,
        "core_dependencies_installed": all(row["installed"] for row in deps),
        "required_raw_files_present": all(row["exists"] for row in raw_files),
    }


def implementation_tasks(make_targets: list[dict[str, Any]], files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "order": 1,
            "task": "materialize_runner_files",
            "allowed_now": False,
            "after_exact_approval": True,
            "evidence": [row["path"] for row in files if not row["exists_now"]],
            "guard": "Create only the dedicated no-submit runner files; no submission artifact path.",
        },
        {
            "order": 2,
            "task": "add_after_approval_make_targets",
            "allowed_now": False,
            "after_exact_approval": True,
            "evidence": [row["target"] for row in make_targets if not row["exists_now"]],
            "guard": "Targets must require explicit remote approval and produce report-only outputs.",
        },
        {
            "order": 3,
            "task": "local_no_network_runner_smoke",
            "allowed_now": False,
            "after_exact_approval": True,
            "evidence": ["post-id342-self-supervised-subject-state-local-smoke"],
            "guard": "Smoke must prove no CSV/log/artifact side effects before any Kaggle push.",
        },
        {
            "order": 4,
            "task": "remote_no_submit_run",
            "allowed_now": False,
            "after_exact_approval": True,
            "evidence": ["kaggle-push-self-supervised-subject-state-runner"],
            "guard": "Remote output must be reviewed against heavy success thresholds before artifact discussion.",
        },
    ]


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    reports = {key: read_json(path) for key, path in REPORT_PATHS.items()}
    prerequisites = build_prerequisites(reports)
    deps = dependency_rows()
    make_targets = make_target_rows()
    files = after_approval_file_rows()
    raw_files = raw_file_rows()
    after_approval_gaps = [row for row in make_targets if row["status"] == "after_approval_gap"] + [
        row for row in files if not row["exists_now"]
    ]
    feasibility_ready = all(prerequisites.values())
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_heavy_implementation_feasibility",
        "purpose": "Check implementation readiness for the approved-later missingness-aware heavy route.",
        "input_reports": reports,
        "prerequisites": prerequisites,
        "dependency_rows": deps,
        "raw_file_rows": raw_files,
        "after_approval_make_targets": make_targets,
        "after_approval_files": files,
        "implementation_tasks": implementation_tasks(make_targets, files),
        "decision": {
            "heavy_implementation_feasibility_ready": feasibility_ready,
            "runtime_dependencies_ready": prerequisites["core_dependencies_installed"],
            "raw_inputs_present": prerequisites["required_raw_files_present"],
            "after_approval_gap_count": len(after_approval_gaps),
            "after_approval_make_targets_missing": any(not row["exists_now"] for row in make_targets),
            "after_approval_runner_files_missing": any(not row["exists_now"] for row in files),
            "requires_exact_heavy_compute_approval": True,
            "heavy_approval_phrase": HEAVY_APPROVAL_PHRASE,
            "create_runner_files_now": False,
            "read_raw_parquet_payloads_now": False,
            "fit_embedding_now": False,
            "create_submission_artifact_now": False,
            "modify_submission_log_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "dacon_submit_now": False,
            "next_action": (
                "Implementation dependencies and raw inputs are ready, but runner files and no-submit Make targets "
                "should be created only after exact heavy approval."
                if feasibility_ready
                else "Refresh missing prerequisite reports or dependencies before requesting heavy execution."
            ),
        },
        "forbidden_actions_now": FORBIDDEN_ACTIONS_NOW,
        "report_only": True,
        "no_runner_files_created": True,
        "no_raw_payload_read": True,
        "no_embedding_fit": True,
        "no_feature_matrix_created": True,
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_submission_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_kaggle_output_pull_made": True,
        "no_dacon_submission_made": True,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision_payload = report["decision"]
    lines = [
        "# Post-ID342 heavy implementation feasibility",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This report checks implementation readiness only. It creates no runner files, reads no raw payloads, fits no embeddings, trains no model, runs no Kaggle job, and submits nothing to DACON.",
        "",
        "## Decision",
        "",
        f"- Feasibility ready: `{decision_payload['heavy_implementation_feasibility_ready']}`",
        f"- Runtime dependencies ready: `{decision_payload['runtime_dependencies_ready']}`",
        f"- Raw inputs present: `{decision_payload['raw_inputs_present']}`",
        f"- After-approval gap count: `{decision_payload['after_approval_gap_count']}`",
        f"- Make targets missing after approval: `{decision_payload['after_approval_make_targets_missing']}`",
        f"- Runner files missing after approval: `{decision_payload['after_approval_runner_files_missing']}`",
        f"- Create runner files now: `{decision_payload['create_runner_files_now']}`",
        f"- Run Kaggle now: `{decision_payload['run_kaggle_now']}`",
        f"- DACON submit now: `{decision_payload['dacon_submit_now']}`",
        f"- Required phrase: `{decision_payload['heavy_approval_phrase']}`",
        f"- Next action: {decision_payload['next_action']}",
        "",
        "## Dependencies",
        "",
        "| distribution | declared | installed | version | purpose |",
        "|---|---:|---:|---|---|",
    ]
    for row in report["dependency_rows"]:
        lines.append(
            f"| `{row['distribution']}` | `{row['declared']}` | `{row['installed']}` | "
            f"`{row['version']}` | {row['purpose']} |"
        )
    lines.extend(
        [
            "",
            "## After-Approval Make Targets",
            "",
            "| target | exists now | required now | required after exact approval | status |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in report["after_approval_make_targets"]:
        lines.append(
            f"| `{row['target']}` | `{row['exists_now']}` | `{row['required_now']}` | "
            f"`{row['required_after_exact_approval']}` | `{row['status']}` |"
        )
    lines.extend(
        [
            "",
            "## After-Approval Files",
            "",
            "| path | exists now | create now | create after exact approval |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report["after_approval_files"]:
        lines.append(
            f"| `{row['path']}` | `{row['exists_now']}` | `{row['create_now']}` | "
            f"`{row['create_after_exact_approval']}` |"
        )
    lines.extend(
        [
            "",
            "## Implementation Tasks",
            "",
            "| order | task | allowed now | after exact approval | guard |",
            "|---:|---|---:|---:|---|",
        ]
    )
    for row in report["implementation_tasks"]:
        lines.append(
            f"| {row['order']} | `{row['task']}` | `{row['allowed_now']}` | "
            f"`{row['after_exact_approval']}` | {row['guard']} |"
        )
    lines.extend(
        [
            "",
            "## Prerequisites",
            "",
            "```json",
            json.dumps(report["prerequisites"], indent=2, sort_keys=True),
            "```",
            "",
            "## Forbidden Actions Now",
            "",
            "```json",
            json.dumps(report["forbidden_actions_now"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, args.output)
    print(f"Wrote {rel(args.output)}")
    print(f"Wrote {rel(args.output.with_suffix('.md'))}")
    print(f"Feasibility ready: {report['decision']['heavy_implementation_feasibility_ready']}")
    print(f"Run Kaggle now: {report['decision']['run_kaggle_now']}")


if __name__ == "__main__":
    main()
