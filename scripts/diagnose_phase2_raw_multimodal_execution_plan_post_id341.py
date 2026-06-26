#!/usr/bin/env python3
"""Concrete phase-2 raw multimodal execution plan after ID341.

This report-only diagnostic converts the post-ID341 closure evidence into a
small Kaggle-CPU execution plan for the next raw-source smoke. It does not train
models, create predictions, create submission artifacts, push/run Kaggle, or
submit to DACON.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
PROCESSED_TRAIN = ROOT / "data" / "processed" / "df_train_raw.parquet"
PROCESSED_TEST = ROOT / "data" / "processed" / "df_test_raw.parquet"
KST = timezone(timedelta(hours=9))

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
REQUIRED_EVIDENCE = {
    "phase2_spec": "post_id341_phase2_raw_source_execution_spec_20260618.json",
    "transfer_gate_v2": "post_id341_transfer_gate_v2_joint_risk_20260618.json",
    "sleep_boundary_gate": "sleep_boundary_prepromotion_gate_post_id341_20260618.json",
    "combined_external_gate": "combined_sleep_evening_external_prior_gate_post_id341_20260618.json",
    "subject_residual_gate": "subject_residual_target_restricted_gate_post_id341_20260618.json",
    "place_social_smoke": "place_social_novelty_smoke_fixedlist_post_id341_20260618.json",
    "sensor_adherence_smoke": "sensor_adherence_dropout_smoke_post_id341_20260618.json",
    "split_manifest": "id340_validation_split_manifest_20260618.json",
}


def load_report(name: str) -> dict[str, Any]:
    path = REPORTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing evidence report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def evidence_status() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, name in REQUIRED_EVIDENCE.items():
        path = REPORTS_DIR / name
        out[key] = {
            "path": str(path.relative_to(ROOT)),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
        }
    return out


def raw_file_meta() -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for path in sorted(RAW_DIR.glob("*.parquet")):
        frame = pd.read_parquet(path)
        meta[path.stem.replace("ch2025_", "")] = {
            "path": str(path.relative_to(ROOT)),
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "subject_count": int(frame["subject_id"].nunique()) if "subject_id" in frame else None,
            "has_timestamp": "timestamp" in frame.columns,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
        }
    return meta


def processed_meta() -> dict[str, Any]:
    train = pd.read_parquet(PROCESSED_TRAIN)
    test = pd.read_parquet(PROCESSED_TEST)
    feature_cols = [
        col for col in train.columns if col not in {"subject_id", "date", "sleep_date", "lifelog_date", *TARGETS}
    ]
    token_groups = {
        "screen": ["screen"],
        "activity": ["activity", "act_"],
        "charge": ["charge", "charging", "ac_"],
        "light": ["light", "mlight", "wlight"],
        "heart_rate": ["hr", "whr"],
        "pedo": ["pedo", "step"],
        "usage": ["usage", "app"],
        "place_social": ["gps", "wifi", "ble"],
    }
    group_counts = {
        group: int(sum(any(token in col.lower() for token in tokens) for col in feature_cols))
        for group, tokens in token_groups.items()
    }
    return {
        "train_shape": [int(train.shape[0]), int(train.shape[1])],
        "test_shape": [int(test.shape[0]), int(test.shape[1])],
        "feature_columns": int(len(feature_cols)),
        "group_counts": group_counts,
        "train_subject_count": int(train["subject_id"].nunique()),
        "test_subject_count": int(test["subject_id"].nunique()),
        "train_date_min": str(pd.to_datetime(train["date"]).min().date()),
        "train_date_max": str(pd.to_datetime(train["date"]).max().date()),
        "test_date_min": str(pd.to_datetime(test["date"]).min().date()),
        "test_date_max": str(pd.to_datetime(test["date"]).max().date()),
    }


def evidence_read() -> dict[str, Any]:
    reports = {key: load_report(name) for key, name in REQUIRED_EVIDENCE.items() if key != "split_manifest"}
    sleep_decision = reports["sleep_boundary_gate"].get("decision", {})
    combined_decision = reports["combined_external_gate"].get("decision", {})
    residual_decision = reports["subject_residual_gate"].get("decision", {})
    place_decision = reports["place_social_smoke"].get("decision", {})
    sensor_decision = reports["sensor_adherence_smoke"].get("decision", {})
    transfer_decision = reports["transfer_gate_v2"].get("decision", {})
    return {
        "transfer_gate_closed": transfer_decision.get("any_candidate_passes_combined_public_058_gate") is False,
        "sleep_boundary_low_movement_lead": sleep_decision.get("reviewable_low_movement_feature_lead") is True,
        "sleep_boundary_promotable": sleep_decision.get("promote_submission_artifact") is True,
        "combined_weak_best_proxy": combined_decision.get("best_proxy"),
        "combined_weak_public_058": combined_decision.get("public_058_plausible"),
        "residual_selected_targets": residual_decision.get("selected_targets", []),
        "residual_primary_wins": residual_decision.get("primary_wins_vs_id341"),
        "residual_projected_proxy": residual_decision.get("projected_public_proxy_from_id341"),
        "residual_public_058": residual_decision.get("public_058_reviewable"),
        "place_selected_targets": place_decision.get("selected_targets", []),
        "sensor_selected_targets": sensor_decision.get("selected_targets", []),
        "phase2_ready": reports["phase2_spec"].get("decision", {}).get("phase2_spec_ready"),
    }


def execution_blocks(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "block": "A_q1_s4_sleep_residual_bridge",
            "priority": 1,
            "why": (
                "The only repeated local raw/source signals after ID341 are sleep-boundary structure plus residual "
                "Q1/S4 drift. This block tests their common raw cause instead of blending their OOF outputs."
            ),
            "target_focus": ["Q1", "S4"],
            "raw_inputs": ["mScreenStatus", "mActivity", "mACStatus", "mLight", "wLight", "wHr", "wPedo"],
            "feature_families": [
                "12h-36h relative-night bins for screen/activity/charge/light/hr/pedo",
                "longest quiet interval and interruption count from screen-on + active + step spikes",
                "HR dip timing, morning-minus-night HR, and light/dark transitions",
                "fold-internal subject residuals of inferred onset/wake/recovery features",
                "Q1/S4-only compact model heads; all other targets stay anchored during this block",
            ],
            "model_allowlist": [
                "LogisticRegression C in {0.03, 0.10}",
                "HistGradientBoosting max_leaf_nodes<=5 l2>=3",
                "ExtraTrees max_depth<=4 min_samples_leaf>=25 only for feature screening, not final promotion",
            ],
            "selection_gate": {
                "target_config_win_floor": 4,
                "assembled_primary_wins_floor": 5,
                "max_primary_delta_vs_id341_ceiling": 0.0,
                "projected_proxy_must_be_below": 0.6014113009,
                "public_058_review_cutoff": 0.590,
            },
            "expected_failure_mode": (
                "Current residual gate wins only 3/5 primary splits and fails subject_hole/loso; if this raw bridge "
                "does not fix those two lenses, stop immediately."
            ),
        },
        {
            "block": "B_sleep_boundary_s2_recovery_retry",
            "priority": 2,
            "why": (
                "S2 had residual smoke but failed the stricter target-restricted gate. Test whether raw recovery "
                "features explain S2 without borrowing Q1/S4 movement."
            ),
            "target_focus": ["S2"],
            "raw_inputs": ["wHr", "wPedo", "mActivity", "mLight", "wLight", "mScreenStatus"],
            "feature_families": [
                "night HR volatility and min-HR timing",
                "low-step and stillness segment duration",
                "morning recovery slope and light exposure",
                "fold-internal subject residuals of recovery features",
            ],
            "model_allowlist": ["LogisticRegression", "HistGradientBoosting shallow"],
            "selection_gate": {
                "target_config_win_floor": 4,
                "mean_delta_vs_id341_must_be_negative": True,
                "assembled_primary_wins_floor": 5,
                "projected_proxy_must_be_below": 0.6014113009,
            },
            "expected_failure_mode": "If S2 remains 2/5 or worse, close S2 residual/recovery locally.",
        },
        {
            "block": "C_env_social_routine_rebuild_only_if_A_or_B_passes",
            "priority": 3,
            "why": (
                "Place/social novelty and sensor-adherence standalone lanes selected no targets, so this should not "
                "consume compute unless a sleep/residual block creates a stronger anchor first."
            ),
            "target_focus": ["Q2", "Q3"],
            "raw_inputs": ["mWifi", "mBle", "mGps", "mUsageStats", "mAmbience"],
            "feature_families": [
                "home/routine entropy with hashed top-k buckets only",
                "app-category evening novelty, not direct app-name memorization",
                "ambience context ratios as weak modifiers",
            ],
            "model_allowlist": ["LogisticRegression with strong regularization"],
            "selection_gate": {
                "run_condition": "only after block A or B passes all-primary transfer pre-gate",
                "target_config_win_floor": 4,
                "no_high_joint_risk_warning": True,
            },
            "expected_failure_mode": "Do not widen local place/social scans; previous fixed-list parser smoke already failed.",
        },
    ]


def remote_runner_contract(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "compute": "Kaggle CPU only; GPU/TPU disabled",
        "max_runtime_plan": "single notebook revision, report-only, stop after block A if it fails all-primary gate",
        "allowed_outputs": [
            "reports/remote_phase2_raw_multimodal_execution_plan_post_id341_20260618.json/.md",
            "reports/remote_phase2_block_A_q1_s4_sleep_residual_bridge_20260618.json/.md",
            "reports/remote_phase2_block_B_s2_recovery_retry_20260618.json/.md only if block A has non-catastrophic transfer",
            "kaggle_outputs/phase2_raw_multimodal_manifest.json",
        ],
        "forbidden_outputs": [
            "submissions/*.csv",
            "artifacts/base_predictions/base_oof_id_*.csv",
            "artifacts/base_predictions/base_test_id_*.csv",
            "notebooks/history/submission_*.ipynb",
            "submission_log.csv row changes",
            "DACON submission",
        ],
        "stop_rules": [
            "Stop if block A does not beat ID341 on all primary splits.",
            "Stop if projected proxy remains above actual ID178 public 0.6014113009.",
            "Stop if any temporary source resembles ID287 high-movement/high-joint-risk geometry.",
            "Stop before creating ID342; artifact-contract review is separate even if gates pass.",
        ],
        "block_order": [block["block"] for block in sorted(blocks, key=lambda row: row["priority"])],
    }


def build_report() -> dict[str, Any]:
    statuses = evidence_status()
    missing = [row["path"] for row in statuses.values() if not row["exists"]]
    evidence = evidence_read()
    blocks = execution_blocks(evidence)
    contract = remote_runner_contract(blocks)
    plan_ready = bool(
        not missing
        and evidence.get("phase2_ready") is True
        and evidence.get("transfer_gate_closed") is True
        and evidence.get("sleep_boundary_low_movement_lead") is True
        and evidence.get("sleep_boundary_promotable") is False
        and evidence.get("combined_weak_public_058") is False
        and evidence.get("residual_public_058") is False
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "phase2_raw_multimodal_execution_plan_post_id341",
        "purpose": (
            "Translate local post-ID341 weak-source evidence into a small no-submit Kaggle CPU raw-source execution plan."
        ),
        "evidence_status": statuses,
        "missing_evidence": missing,
        "evidence_read": evidence,
        "processed_meta": processed_meta(),
        "raw_file_meta": raw_file_meta(),
        "execution_blocks": blocks,
        "remote_runner_contract": contract,
        "decision": {
            "phase2_raw_multimodal_plan_ready": plan_ready,
            "recommended_first_block": "A_q1_s4_sleep_residual_bridge",
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "run_heavy_rebuild_now": False,
            "next_action": (
                "Wire this plan into the no-submit Kaggle upstream runner only after explicit approval. Locally, do not "
                "create ID342 from residual/sleep/evening micro leads; use them only to implement block A."
            ),
        },
        "no_model_trained": True,
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
        "# Phase-2 raw multimodal execution plan — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It trains no model, creates no predictions or submission artifacts, and makes no Kaggle/DACON external action.",
        "",
        "## Evidence read",
        "",
        "```json",
        json.dumps(report["evidence_read"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Execution blocks",
        "",
    ]
    for block in report["execution_blocks"]:
        lines.extend(
            [
                f"### {block['priority']}. `{block['block']}`",
                "",
                block["why"],
                "",
                f"- Target focus: `{', '.join(block['target_focus'])}`",
                f"- Raw inputs: `{', '.join(block['raw_inputs'])}`",
                "",
                "Feature families:",
            ]
        )
        lines.extend(f"- {item}" for item in block["feature_families"])
        lines.extend(
            [
                "",
                "Selection gate:",
                "",
                "```json",
                json.dumps(block["selection_gate"], indent=2, ensure_ascii=False),
                "```",
                "",
                f"Expected failure mode: {block['expected_failure_mode']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Remote runner contract",
            "",
            "```json",
            json.dumps(report["remote_runner_contract"], indent=2, ensure_ascii=False),
            "```",
            "",
            "## Decision",
            "",
            f"- Phase-2 raw multimodal plan ready: `{decision['phase2_raw_multimodal_plan_ready']}`",
            f"- Recommended first block: `{decision['recommended_first_block']}`",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
            f"- Submit now: `{decision['submit_now']}`",
            f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
            f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
            f"- Run heavy rebuild now: `{decision['run_heavy_rebuild_now']}`",
            f"- Next action: {decision['next_action']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "phase2_raw_multimodal_execution_plan_post_id341_20260618.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
