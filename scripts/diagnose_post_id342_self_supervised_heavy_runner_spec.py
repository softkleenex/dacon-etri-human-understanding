#!/usr/bin/env python3
"""Post-ID342 dedicated self-supervised heavy runner spec.

This report-only diagnostic defines the no-submit Kaggle runner that would be
needed after exact heavy-compute approval. It intentionally does not create the
runner notebook, read full raw streams, fit embeddings, write feature matrices,
push/run Kaggle, create submission artifacts, edit the submission log, or submit
to DACON.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_self_supervised_heavy_runner_spec_20260624.json"
KST = timezone(timedelta(hours=9))

SOURCE_FAMILY = "self_supervised_subject_state_embedding_v1"
TARGET_FOCUS = ["Q2", "Q3", "S3"]
HEAVY_APPROVAL_PHRASE = "Self-supervised subject-state heavy compute no-submit 실행 승인"
PROPOSED_NOTEBOOK_DIR = "notebooks/kaggle_self_supervised_subject_state_runner"
PROPOSED_KAGGLE_SLUG = "softkleenex/dacon-etri-self-supervised-subject-state-post-id342"
BASE_DATASET_SLUG = "softkleenex/dacon-etri-human-base"
PATCH_DATASET_SLUG = "softkleenex/dacon-etri-upstream-patch-post-id341"

REPORT_PATHS = {
    "heavy_contract": REPORTS_DIR / "post_id342_self_supervised_heavy_compute_contract_20260624.json",
    "research_packet": REPORTS_DIR / "post_id342_self_supervised_route_research_packet_20260624.json",
    "goal_state_audit": REPORTS_DIR / "post_id342_goal_state_audit_20260624.json",
    "remaining_route_synthesis": REPORTS_DIR / "post_id342_remaining_route_synthesis_20260624.json",
    "docs_records_audit": REPORTS_DIR / "docs_records_audit.json",
}

RESEARCH_SOURCES = [
    {
        "source": "ETRI Human Understanding AI Paper Challenge 2024 dataset design",
        "url": "https://arxiv.org/html/2403.16509v1",
        "runner_implication": "Use strict participant/date-aware folds and treat validation/test cohort shift as first-class risk.",
    },
    {
        "source": "MoCA: Multi-modal Cross-masked Autoencoder for Time Series in Digital Health",
        "url": "https://arxiv.org/html/2506.02260v3",
        "runner_implication": "Prefer cross-modal masking over single-channel reconstruction so missing sensor streams become part of the objective.",
    },
    {
        "source": "CroSSL: Cross-modal Self-Supervised Learning for Time-series Signals",
        "url": "https://arxiv.org/pdf/2307.16847",
        "runner_implication": "Keep modality-specific encoders and cross-modal latent masking separate from downstream target labels.",
    },
    {
        "source": "Personalization of Stress Mobile Sensing using Self-Supervised Learning",
        "url": "https://arxiv.org/abs/2308.02731",
        "runner_implication": "Evaluate subject-state transfer and personalization explicitly for Q3 instead of relying on pooled local CV.",
    },
    {
        "source": "PixleepFlow",
        "url": "https://arxiv.org/html/2502.17469v2",
        "runner_implication": "Represent synchronized lifelog channels as day/evening/sleep-transition state grids rather than isolated scalar proxies.",
    },
]

FORBIDDEN_ACTIONS_NOW = {
    "create_runner_notebook_now": False,
    "read_full_raw_streams_now": False,
    "fit_embedding_now": False,
    "write_feature_matrix_now": False,
    "train_downstream_model_now": False,
    "write_oof_or_test_predictions_now": False,
    "create_submission_artifact": False,
    "modify_submission_log": False,
    "push_kaggle_now": False,
    "run_kaggle_now": False,
    "submit_now": False,
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json_report(path: Path) -> dict[str, Any]:
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
    payload["path"] = rel(path)
    payload["exists"] = True
    payload["parse_ok"] = True
    return payload


def decision(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("decision", {})
    return value if isinstance(value, dict) else {}


def command_output(cmd: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=10)  # noqa: S603
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "command": cmd, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": result.returncode == 0,
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def local_resource_snapshot() -> dict[str, Any]:
    memsize = command_output(["sysctl", "-n", "hw.memsize"])
    total_bytes = int(memsize["stdout"]) if memsize.get("ok") and str(memsize.get("stdout", "")).isdigit() else None
    return {
        "total_memory_bytes": total_bytes,
        "total_memory_gib": None if total_bytes is None else round(total_bytes / (1024**3), 3),
        "resource_check_completed_before_this_spec": True,
        "local_execution_recommendation": "Do not run heavy embedding locally; use Kaggle only after exact heavy approval.",
    }


def prerequisite_status(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    heavy_decision = decision(reports["heavy_contract"])
    research_decision = decision(reports["research_packet"])
    goal_decision = decision(reports["goal_state_audit"])
    synthesis_decision = decision(reports["remaining_route_synthesis"])
    docs_findings = reports["docs_records_audit"].get("findings", [])
    return {
        "heavy_contract_ready": heavy_decision.get("post_id342_heavy_compute_contract_ready") is True,
        "heavy_contract_blocks_execution_now": heavy_decision.get("heavy_compute_allowed_now") is False
        and heavy_decision.get("run_kaggle_now") is False,
        "dedicated_runner_required": heavy_decision.get("dedicated_heavy_runner_required") is True,
        "upstream_runner_not_sufficient": heavy_decision.get(
            "existing_upstream_runner_is_sufficient_for_heavy_execution"
        )
        is False,
        "research_packet_ready": research_decision.get("research_packet_ready") is True,
        "goal_not_achieved": goal_decision.get("goal_achieved") is False
        and goal_decision.get("public_055_reached") is False,
        "post_id342_routes_closed_except_heavy": synthesis_decision.get("closed_route_count") == 11
        and synthesis_decision.get("route_count") == 12
        and synthesis_decision.get("missing_or_review_route_count") == 0,
        "docs_audit_clean": reports["docs_records_audit"].get("parse_ok") is True
        and isinstance(docs_findings, list)
        and not docs_findings,
    }


def runner_files_after_approval() -> list[dict[str, Any]]:
    return [
        {
            "path": f"{PROPOSED_NOTEBOOK_DIR}/kernel-metadata.json",
            "create_now": False,
            "after_exact_approval": True,
            "purpose": "Kaggle kernel metadata with GPU/TPU disabled unless separately justified.",
        },
        {
            "path": f"{PROPOSED_NOTEBOOK_DIR}/Self_Supervised_Subject_State_Post_ID342.ipynb",
            "create_now": False,
            "after_exact_approval": True,
            "purpose": "No-submit notebook that stages data, fits label-free embeddings, and writes review reports only.",
        },
        {
            "path": "scripts/run_post_id342_self_supervised_subject_state_no_submit.py",
            "create_now": False,
            "after_exact_approval": True,
            "purpose": "Reusable runner script invoked by the Kaggle notebook; should not produce submission CSVs.",
        },
        {
            "path": "reports/post_id342_self_supervised_remote_*",
            "create_now": False,
            "after_exact_approval": True,
            "purpose": "Remote report-only outputs for embedding fit, fold-internal downstream smoke, and stop decisions.",
        },
    ]


def data_contract() -> dict[str, Any]:
    return {
        "dataset_sources": [BASE_DATASET_SLUG, PATCH_DATASET_SLUG],
        "raw_modalities": [
            "mACStatus",
            "mActivity",
            "mAmbience",
            "mBle",
            "mGps",
            "mLight",
            "mScreenStatus",
            "mUsageStats",
            "mWifi",
            "wHr",
            "wLight",
            "wPedo",
        ],
        "row_keys": ["subject_id", "sleep_date", "lifelog_date"],
        "target_focus": TARGET_FOCUS,
        "label_free_pretraining_rows": "train+test lifelog days may be tokenized without target labels",
        "downstream_labels": "train labels only inside fold-internal downstream smoke",
        "forbidden_inputs": [
            "DACON public leaderboard scores",
            "submission predictions from ID287/ID329 donors",
            "target labels for embedding selection",
            "test target labels or same-row true sibling labels",
        ],
    }


def model_contract() -> dict[str, Any]:
    return {
        "preferred_objective": "cross_modal_masked_reconstruction",
        "secondary_objective": "time_order_or_day_segment_contrastive_loss",
        "architecture_scope": "small modality-specific encoders plus shared transformer/temporal aggregator",
        "token_granularity": [
            "day grid: 24 hourly bins",
            "evening grid: 18:00-03:00 finer bins",
            "sleep-transition grid: light/screen/charge/activity/HR state changes around sleep_date",
        ],
        "resource_cap": {
            "max_epochs_first_run": 12,
            "max_embedding_dim": 64,
            "max_transformer_layers": 2,
            "max_wall_clock_hours": 5,
            "device": "Kaggle CPU first unless a separate GPU note is written",
        },
        "reproducibility": {
            "fixed_seeds": [45502, 45503],
            "save_config_json": True,
            "save_no_submit_manifest": True,
        },
    }


def validation_contract() -> list[dict[str, Any]]:
    return [
        {
            "gate": "embedding_purity",
            "criterion": "Embedding fit uses no target labels, no OOF predictions, and no public-score feedback.",
            "stop_if_fail": True,
        },
        {
            "gate": "fold_internal_downstream",
            "criterion": "Q3/S3 downstream smoke must beat ID342 on stable near-anchor lenses.",
            "stop_if_fail": True,
        },
        {
            "gate": "failed_frontier_resistance",
            "criterion": "The gain must not appear only when ID287/ID329-like failed frontier geometry is included.",
            "stop_if_fail": True,
        },
        {
            "gate": "public_gap_budget",
            "criterion": "Projected public proxy must be materially below the 0.596 band before artifact review.",
            "stop_if_fail": True,
        },
        {
            "gate": "artifact_separation",
            "criterion": "No submission CSV, base prediction artifact, notebook history artifact, or log row is created by this runner.",
            "stop_if_fail": True,
        },
    ]


def approved_command_sequence() -> list[dict[str, Any]]:
    return [
        {
            "step": 1,
            "command": "make post-id342-self-supervised-heavy-contract",
            "allowed_now": True,
            "after_exact_approval": True,
            "why": "Refresh the contract and verify approval boundary.",
        },
        {
            "step": 2,
            "command": "create dedicated runner files under notebooks/kaggle_self_supervised_subject_state_runner/",
            "allowed_now": False,
            "after_exact_approval": True,
            "why": "The upstream rebuild runner is not a heavy embedding runner.",
        },
        {
            "step": 3,
            "command": "ALLOW_KAGGLE_REMOTE=true make kaggle-push-self-supervised-subject-state-runner",
            "allowed_now": False,
            "after_exact_approval": True,
            "why": "Push no-submit runner only after exact approval and local contracts pass.",
        },
        {
            "step": 4,
            "command": "ALLOW_KAGGLE_REMOTE=true make kaggle-wait-and-pull-self-supervised-subject-state-output",
            "allowed_now": False,
            "after_exact_approval": True,
            "why": "Pull reports only; do not create submissions.",
        },
        {
            "step": 5,
            "command": "make post-id342-self-supervised-remote-review",
            "allowed_now": False,
            "after_exact_approval": True,
            "why": "Review remote reports before any future artifact contract.",
        },
    ]


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    reports = {key: read_json_report(path) for key, path in REPORT_PATHS.items()}
    prereq = prerequisite_status(reports)
    all_prereq_ready = all(prereq.values())
    decision_payload = {
        "runner_spec_ready": all_prereq_ready,
        "source_family": SOURCE_FAMILY,
        "target_focus": TARGET_FOCUS,
        "proposed_notebook_dir": PROPOSED_NOTEBOOK_DIR,
        "proposed_kaggle_slug": PROPOSED_KAGGLE_SLUG,
        "requires_exact_heavy_compute_approval": True,
        "heavy_approval_phrase": HEAVY_APPROVAL_PHRASE,
        "create_runner_files_now": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "create_submission_artifact_now": False,
        "modify_submission_log_now": False,
        "dacon_submit_now": False,
        "dedicated_runner_required": True,
        "existing_upstream_runner_is_sufficient": False,
        "next_action": (
            "Spec is ready. Do not create or push the runner until the exact heavy approval phrase is provided."
            if all_prereq_ready
            else "Repair prerequisite reports before preparing the runner."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_self_supervised_heavy_runner_spec",
        "purpose": "Define the dedicated no-submit Kaggle heavy runner needed after exact approval.",
        "research_sources": RESEARCH_SOURCES,
        "input_reports": {
            key: {"path": report.get("path"), "exists": report.get("exists"), "parse_ok": report.get("parse_ok")}
            for key, report in reports.items()
        },
        "local_resource_snapshot": local_resource_snapshot(),
        "prerequisites": prereq,
        "runner_files_after_approval": runner_files_after_approval(),
        "data_contract": data_contract(),
        "model_contract": model_contract(),
        "validation_contract": validation_contract(),
        "approved_command_sequence": approved_command_sequence(),
        "decision": decision_payload,
        "forbidden_actions_now": FORBIDDEN_ACTIONS_NOW,
        "report_only": True,
        "no_runner_files_created": True,
        "no_full_raw_stream_read": True,
        "no_embedding_fit": True,
        "no_feature_matrix_created": True,
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_submission_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision_payload = report["decision"]
    lines = [
        "# Post-ID342 self-supervised heavy runner spec",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no runner notebook, raw cache, embedding, feature matrix, model, prediction, submission artifact, Kaggle run, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Runner spec ready: `{decision_payload['runner_spec_ready']}`",
        f"- Proposed notebook dir: `{decision_payload['proposed_notebook_dir']}`",
        f"- Proposed Kaggle slug: `{decision_payload['proposed_kaggle_slug']}`",
        f"- Create runner files now: `{decision_payload['create_runner_files_now']}`",
        f"- Push Kaggle now: `{decision_payload['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision_payload['run_kaggle_now']}`",
        f"- DACON submit now: `{decision_payload['dacon_submit_now']}`",
        f"- Required phrase: `{decision_payload['heavy_approval_phrase']}`",
        f"- Next action: {decision_payload['next_action']}",
        "",
        "## Runner Files After Approval",
        "",
        "| path | create now | after exact approval | purpose |",
        "|---|---:|---:|---|",
    ]
    for row in report["runner_files_after_approval"]:
        lines.append(
            f"| `{row['path']}` | `{row['create_now']}` | `{row['after_exact_approval']}` | {row['purpose']} |"
        )

    lines.extend(
        [
            "",
            "## Validation Gates",
            "",
            "| gate | criterion | stop if fail |",
            "|---|---|---:|",
        ]
    )
    for row in report["validation_contract"]:
        lines.append(f"| `{row['gate']}` | {row['criterion']} | `{row['stop_if_fail']}` |")

    lines.extend(
        [
            "",
            "## Approved Command Sequence",
            "",
            "| step | command | allowed now | after exact approval | why |",
            "|---:|---|---:|---:|---|",
        ]
    )
    for row in report["approved_command_sequence"]:
        lines.append(
            f"| {row['step']} | `{row['command']}` | `{row['allowed_now']}` | `{row['after_exact_approval']}` | {row['why']} |"
        )

    lines.extend(
        [
            "",
            "## Research Sources",
            "",
            "| source | url | runner implication |",
            "|---|---|---|",
        ]
    )
    for row in report["research_sources"]:
        lines.append(f"| {row['source']} | {row['url']} | {row['runner_implication']} |")

    lines.extend(
        [
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
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.output)
    print(f"Wrote {rel(args.output)}")
    print(f"Wrote {rel(args.output.with_suffix('.md'))}")
    print(f"Runner spec ready: {report['decision']['runner_spec_ready']}")
    print(f"Run Kaggle now: {report['decision']['run_kaggle_now']}")


if __name__ == "__main__":
    main()
