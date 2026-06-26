#!/usr/bin/env python3
"""Phase-2 raw/sequence source execution spec after ID341.

Report-only diagnostic. It translates the current post-ID341 closure state into
a concrete no-submit contract for the next Kaggle CPU revision. It does not train
models, create predictions, create submission artifacts, push/run Kaggle, or
submit to DACON.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
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

REQUIRED_REPORTS = {
    "phase1_readiness": "post_id341_upstream_rebuild_phase1_readiness_20260618.json",
    "priority_matrix": "post_id341_upstream_source_priority_matrix_20260618.json",
    "clean_backbone_closure": "post_id341_clean_backbone_lane_closure_20260618.json",
    "transfer_gate_v2": "post_id341_transfer_gate_v2_joint_risk_20260618.json",
    "external_prior": "external_etri_prior_smoke_post_id341_20260618.json",
    "combined_external": "combined_sleep_evening_external_prior_gate_post_id341_20260618.json",
    "new_source_blueprint": "post_id341_new_source_blueprint_20260618.json",
    "split_manifest": "id340_validation_split_manifest_20260618.json",
}
OPTIONAL_LOCAL_SMOKE_REPORTS = {
    "subject_residual_sequence_drift": "subject_residual_sequence_drift_smoke_post_id341_20260618.json",
    "subject_residual_target_restricted_gate": "subject_residual_target_restricted_gate_post_id341_20260618.json",
    "phase2_raw_multimodal_execution_plan": "phase2_raw_multimodal_execution_plan_post_id341_20260618.json",
    "phase2_block_a_q1_s4_sleep_residual_bridge": "phase2_block_a_q1_s4_sleep_residual_bridge_post_id341_20260618.json",
    "phase2_block_a_q1_proxy_gap": "phase2_block_a_q1_proxy_gap_post_id341_20260618.json",
    "phase2_block_b_s2_sleep_recovery_retry": "phase2_block_b_s2_sleep_recovery_retry_post_id341_20260618.json",
    "phase3_usage_temporal_microstructure": "phase3_usage_temporal_microstructure_smoke_post_id341_20260618.json",
    "phase3_ambience_context_closure": "phase3_ambience_context_closure_post_id341_20260618.json",
    "post_id341_local_source_closure_synthesis": "post_id341_local_source_closure_synthesis_20260618.json",
}

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


def load_json_report(name: str) -> dict[str, Any]:
    path = REPORTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def report_status() -> dict[str, dict[str, Any]]:
    out = {}
    for key, name in REQUIRED_REPORTS.items():
        path = REPORTS_DIR / name
        out[key] = {
            "path": str(path.relative_to(ROOT)),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
        }
    return out


def optional_local_smoke_evidence() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, name in OPTIONAL_LOCAL_SMOKE_REPORTS.items():
        path = REPORTS_DIR / name
        if not path.exists():
            out[key] = {
                "path": str(path.relative_to(ROOT)),
                "exists": False,
                "summary": None,
            }
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        decision = report.get("decision", {})
        out[key] = {
            "path": str(path.relative_to(ROOT)),
            "exists": True,
            "summary": {
                "source_family_has_smoke_signal": decision.get("source_family_has_smoke_signal"),
                "selected_targets": decision.get("selected_targets"),
                "phase2_raw_multimodal_plan_ready": decision.get("phase2_raw_multimodal_plan_ready"),
                "recommended_first_block": decision.get("recommended_first_block"),
                "final_block_a_gate_pass": decision.get("final_block_a_gate_pass"),
                "q1_all_primary_signal_confirmed": decision.get("q1_all_primary_signal_confirmed"),
                "q1_signal_sufficient_for_id178_proxy": decision.get("q1_signal_sufficient_for_id178_proxy"),
                "additional_independent_target_signal_required": decision.get(
                    "additional_independent_target_signal_required"
                ),
                "s2_has_config_signal": decision.get("s2_has_config_signal"),
                "final_block_b_gate_pass": decision.get("final_block_b_gate_pass"),
                "usage_temporal_has_config_signal": decision.get("usage_temporal_has_config_signal"),
                "final_usage_temporal_gate_pass": decision.get("final_usage_temporal_gate_pass"),
                "ambience_reopen_now": decision.get("ambience_reopen_now"),
                "ambience_has_nested_positive_signal": decision.get("ambience_has_nested_positive_signal"),
                "local_source_stack_closed_for_submission_artifact": decision.get(
                    "local_source_stack_closed_for_submission_artifact"
                ),
                "any_local_lane_promotable": decision.get("any_local_lane_promotable"),
                "all_primary_splits_beat_id341": decision.get("all_primary_splits_beat_id341"),
                "primary_split_wins_vs_id341": decision.get("primary_split_wins_vs_id341"),
                "primary_wins_vs_id341": decision.get("primary_wins_vs_id341"),
                "mean_primary_delta_vs_id341": decision.get("mean_primary_delta_vs_id341"),
                "projected_public_proxy_from_id341": decision.get("projected_public_proxy_from_id341"),
                "public_058_reviewable": decision.get("public_058_reviewable"),
                "create_submission_artifact": decision.get("create_submission_artifact"),
                "submit_now": decision.get("submit_now"),
            },
        }
    return out


def parquet_meta(path: Path, include_target_prevalence: bool) -> dict[str, Any]:
    df = pd.read_parquet(path)
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    out = {
        "path": str(path.relative_to(ROOT)),
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "numeric_columns": int(len(numeric_cols)),
        "subject_count": int(df["subject_id"].nunique()) if "subject_id" in df else None,
        "date_min": str(pd.to_datetime(df["date"]).min().date()) if "date" in df else None,
        "date_max": str(pd.to_datetime(df["date"]).max().date()) if "date" in df else None,
        "target_columns_present": [target for target in TARGETS if target in df],
        "target_prevalence": (
            {target: float(df[target].mean()) for target in TARGETS if target in df}
            if include_target_prevalence
            else None
        ),
        "target_prevalence_note": (
            "training labels" if include_target_prevalence else "not reported for test placeholder target columns"
        ),
    }
    return out


def raw_file_meta() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(RAW_DIR.glob("*.parquet")):
        try:
            frame = pd.read_parquet(path)
            rows.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "rows": int(len(frame)),
                    "columns": list(frame.columns),
                    "subject_count": int(frame["subject_id"].nunique()) if "subject_id" in frame else None,
                    "has_timestamp": "timestamp" in frame.columns,
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"path": str(path.relative_to(ROOT)), "error": str(exc)})
    return rows


def split_manifest_summary() -> dict[str, Any]:
    data = load_json_report(REQUIRED_REPORTS["split_manifest"])
    folds = data.get("folds", [])
    counts = Counter(row["split_name"] for row in folds)
    return {
        "n_train_rows": data.get("n_train_rows"),
        "n_test_rows": data.get("n_test_rows"),
        "n_folds_total": len(folds),
        "split_counts": dict(counts),
        "primary_splits_required": [
            "subject_hole_5",
            "actual_test_band_holdout",
            "test_boundary_neighbor2_holdout",
            "test_boundary_neighbor5_holdout",
            "loso_10",
        ],
    }


def current_closure_read() -> dict[str, Any]:
    phase1 = load_json_report(REQUIRED_REPORTS["phase1_readiness"])
    priority = load_json_report(REQUIRED_REPORTS["priority_matrix"])
    clean_closure = load_json_report(REQUIRED_REPORTS["clean_backbone_closure"])
    transfer_v2 = load_json_report(REQUIRED_REPORTS["transfer_gate_v2"])
    combined_external = load_json_report(REQUIRED_REPORTS["combined_external"])
    blueprint = load_json_report(REQUIRED_REPORTS["new_source_blueprint"])
    return {
        "phase1_ready": phase1.get("decision", {}).get("phase1_ready"),
        "local_new_source_stack_closed": phase1.get("decision", {}).get("local_new_source_stack_closed"),
        "existing_artifact_promotable": priority.get("decision", {}).get("any_existing_artifact_promotable"),
        "clean_backbone_closed": clean_closure.get("decision", {}).get("close_clean_backbone_lane_now"),
        "clean_backbone_next_lane": clean_closure.get("decision", {}).get("next_lane"),
        "transfer_v2_public_058_candidate": transfer_v2.get("decision", {}).get(
            "public_058_plausible_candidate_exists"
        ),
        "combined_external_best_proxy": combined_external.get("decision", {}).get("best_proxy"),
        "combined_external_public_058": combined_external.get("decision", {}).get("public_058_plausible"),
        "new_source_blueprint_top_family": blueprint.get("decision", {}).get("top_new_source_family"),
    }


def phase2_stage_contract() -> list[dict[str, Any]]:
    return [
        {
            "stage": "remote_latest_closure_parity",
            "compute": "Kaggle CPU",
            "purpose": "Regenerate transfer-v2, external-prior, and combined weak-source closure reports in Kaggle runtime.",
            "allowed_outputs": [
                "remote_post_id341_transfer_gate_v2_joint_risk_20260618.json/.md",
                "remote_external_etri_prior_smoke_post_id341_20260618.json/.md",
                "remote_combined_sleep_evening_external_prior_gate_post_id341_20260618.json/.md",
            ],
            "stop_if": [
                "any report creates or requests a submission artifact",
                "any report unexpectedly finds a Public 0.58 candidate; stop for manual review instead of auto-promoting",
            ],
            "creates_submission": False,
        },
        {
            "stage": "raw_multimodal_sequence_source_smoke",
            "compute": "Kaggle CPU first; GPU/TPU disabled",
            "purpose": (
                "Build compact sequence/source summaries from raw event streams rather than recombining existing OOF "
                "prediction artifacts."
            ),
            "candidate_sources": [
                {
                    "name": "nightly_multimodal_bins_v1",
                    "raw_inputs": ["mScreenStatus", "mActivity", "mACStatus", "mLight", "wLight", "wHr", "wPedo"],
                    "representation": "12h-36h relative-night bins with screen/activity/charge/light/hr/pedo transitions",
                    "target_focus": ["Q1", "S1", "S2", "S3", "S4"],
                },
                {
                    "name": "environment_social_routine_bins_v1",
                    "raw_inputs": ["mWifi", "mBle", "mUsageStats", "mGps", "mAmbience"],
                    "representation": "hourly novelty/routine entropy and app/env context summaries, no BSSID direct memorization",
                    "target_focus": ["Q2", "Q3", "S4"],
                },
                {
                    "name": "subject_residual_sequence_drift_v1",
                    "raw_inputs": ["processed numeric features", "calendar covariates"],
                    "representation": "fold-internal per-subject residual trend and volatility summaries fitted without valid rows",
                    "target_focus": TARGETS,
                },
            ],
            "model_family_allowlist": [
                "LogisticRegression with strong regularization",
                "HistGradientBoostingClassifier shallow/regularized",
                "ExtraTreesClassifier shallow/leaf-regularized",
            ],
            "selection_rule": (
                "For every outer frozen split, fit preprocessing, feature selection, and any target/source selection only "
                "inside the training fold. Outer valid rows are score-only. Never select using public LB or full OOF."
            ),
            "allowed_outputs": [
                "reports/remote_phase2_raw_source_smoke_metrics_20260618.json/.md",
                "reports/remote_phase2_raw_source_candidate_geometry_20260618.json/.md",
                "kaggle_outputs/phase2_raw_source_manifest.json",
            ],
            "forbidden_outputs": [
                "submissions/*.csv",
                "artifacts/base_predictions/base_oof_id_*.csv",
                "artifacts/base_predictions/base_test_id_*.csv",
                "notebooks/history/submission_*.ipynb",
                "submission_log.csv row changes",
            ],
            "creates_submission": False,
        },
        {
            "stage": "transfer_and_joint_risk_gate",
            "compute": "Kaggle CPU or local after pulled outputs",
            "purpose": "Evaluate any temporary source predictions against ID178/287/329/341 and joint-risk metrics before artifact review.",
            "promotion_floor": {
                "must_beat_all_primary_splits_vs_id341": True,
                "must_have_non_positive_failed_projection": True,
                "must_have_proxy_below_id178_public": 0.6014113009,
                "public_058_review_cutoff": 0.590,
                "must_pass_joint_warning_lens": True,
            },
            "manual_review_required_even_if_passes": True,
            "creates_submission": False,
        },
    ]


def gate_invariants() -> dict[str, Any]:
    return {
        "no_dacon_submission": True,
        "no_kaggle_push_without_explicit_user_approval": True,
        "no_submission_csv_before_artifact_contract": True,
        "unique_filename_required_for_any_future_artifact": True,
        "full_artifact_contract_required_before_submission_file_promotion": [
            "submission CSV",
            "notebook/history",
            "metadata JSON",
            "decision MD",
            "base_oof",
            "base_test",
            "submission_log row",
        ],
        "public_lb_not_used_for_selection": True,
        "outer_valid_rows_score_only": True,
    }


def build_report() -> dict[str, Any]:
    statuses = report_status()
    missing = [row["path"] for row in statuses.values() if not row["exists"]]
    closure = current_closure_read()
    raw_meta = raw_file_meta()
    train_meta = parquet_meta(PROCESSED_TRAIN, include_target_prevalence=True)
    test_meta = parquet_meta(PROCESSED_TEST, include_target_prevalence=False)
    stages = phase2_stage_contract()
    phase2_ready = bool(
        not missing
        and closure.get("phase1_ready") is True
        and closure.get("local_new_source_stack_closed") is True
        and closure.get("existing_artifact_promotable") is False
        and closure.get("clean_backbone_closed") is True
        and closure.get("transfer_v2_public_058_candidate") is False
        and closure.get("combined_external_public_058") is False
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_phase2_raw_source_execution_spec",
        "purpose": (
            "Freeze the next approved Kaggle revision after phase-1 closure: stronger raw/sequence source smoke, "
            "not local micro-blending or contaminated upstream artifact replay."
        ),
        "required_report_status": statuses,
        "missing_required_reports": missing,
        "current_closure_read": closure,
        "data_readiness": {
            "processed_train": train_meta,
            "processed_test": test_meta,
            "raw_file_count": len(raw_meta),
            "raw_total_rows": int(sum(row.get("rows", 0) for row in raw_meta)),
            "raw_total_size_mb": round(sum(row.get("size_mb", 0.0) for row in raw_meta), 3),
            "raw_files": raw_meta,
        },
        "optional_local_smoke_evidence": optional_local_smoke_evidence(),
        "split_manifest_summary": split_manifest_summary(),
        "stage_contract": stages,
        "gate_invariants": gate_invariants(),
        "decision": {
            "phase2_spec_ready": phase2_ready,
            "recommended_next_revision": "remote_phase2_raw_multimodal_sequence_source_smoke",
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "run_heavy_rebuild_now": False,
            "next_action": (
                "After explicit approval and pulled green phase-1 runner reports, extend the Kaggle runner only with the "
                "stage-contract raw/sequence source smoke. Do not create ID342 unless the pulled temporary source passes "
                "transfer and joint-risk gates plus a separate artifact-contract review."
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
    data = report["data_readiness"]
    lines = [
        "# Post-ID341 phase-2 raw/sequence source execution spec",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It trains no model, creates no predictions or submission artifacts, and makes no Kaggle/DACON external action.",
        "",
        "## Closure read",
        "",
        "```json",
        json.dumps(report["current_closure_read"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Data readiness",
        "",
        f"- Processed train/test: `{data['processed_train']['shape']}` / `{data['processed_test']['shape']}`",
        f"- Raw files: `{data['raw_file_count']}`",
        f"- Raw total rows: `{data['raw_total_rows']}`",
        f"- Raw total size MB: `{data['raw_total_size_mb']}`",
        "",
        "## Stage contract",
        "",
    ]
    if report.get("optional_local_smoke_evidence"):
        lines.extend(
            [
                "## Optional local smoke evidence",
                "",
                "```json",
                json.dumps(report["optional_local_smoke_evidence"], indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    for stage in report["stage_contract"]:
        lines.extend(
            [
                f"### `{stage['stage']}`",
                "",
                stage["purpose"],
                "",
                f"- Compute: {stage['compute']}",
                f"- Creates submission: `{stage['creates_submission']}`",
            ]
        )
        if "candidate_sources" in stage:
            lines.extend(["", "| source | raw inputs | target focus | representation |", "|---|---|---|---|"])
            for source in stage["candidate_sources"]:
                lines.append(
                    f"| `{source['name']}` | `{', '.join(source['raw_inputs'])}` | "
                    f"`{', '.join(source['target_focus'])}` | {source['representation']} |"
                )
        if "allowed_outputs" in stage:
            lines.extend(["", "Allowed outputs:"])
            lines.extend(f"- `{item}`" for item in stage["allowed_outputs"])
        if "forbidden_outputs" in stage:
            lines.extend(["", "Forbidden outputs:"])
            lines.extend(f"- `{item}`" for item in stage["forbidden_outputs"])
        if "stop_if" in stage:
            lines.extend(["", "Stop if:"])
            lines.extend(f"- {item}" for item in stage["stop_if"])
        lines.append("")
    lines.extend(
        [
            "## Gate invariants",
            "",
            "```json",
            json.dumps(report["gate_invariants"], indent=2, ensure_ascii=False),
            "```",
            "",
            "## Decision",
            "",
            f"- Phase-2 spec ready: `{report['decision']['phase2_spec_ready']}`",
            f"- Recommended next revision: `{report['decision']['recommended_next_revision']}`",
            f"- Create submission artifact: `{report['decision']['create_submission_artifact']}`",
            f"- Submit now: `{report['decision']['submit_now']}`",
            f"- Push Kaggle now: `{report['decision']['push_kaggle_now']}`",
            f"- Run Kaggle now: `{report['decision']['run_kaggle_now']}`",
            f"- Run heavy rebuild now: `{report['decision']['run_heavy_rebuild_now']}`",
            f"- Next action: {report['decision']['next_action']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_phase2_raw_source_execution_spec_20260618.json",
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
