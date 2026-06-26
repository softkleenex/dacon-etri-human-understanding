#!/usr/bin/env python3
"""Phase-1 readiness audit for the post-ID341 upstream rebuild.

This is a report-only audit intended to run locally and inside the CPU-only
Kaggle upstream rebuild runner before any heavier source reconstruction is
added. It checks frozen splits, anchor/public evidence, recent new-source gates,
and the availability of upstream/source artifacts.

It trains no model, creates no submission CSV/base prediction artifact/notebook
history artifact/submission-log row, and makes no DACON submission.
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
BASE_DIR = ROOT / "artifacts" / "base_predictions"
SUBMISSIONS_DIR = ROOT / "submissions"
LOG_PATH = SUBMISSIONS_DIR / "submission_log.csv"
KST = timezone(timedelta(hours=9))

UPSTREAM_MODEL_LAYER = [221, 226, 227, 228, 237]
A1_REPLAY_LAYER = [238, 239, 241]
B1_REPLAY_LAYER = [254, 255, 256, 257, 258, 266]
PUBLIC_ANCHOR_LAYER = [178, 287, 329, 341]
RECENT_NEW_SOURCE_REPORTS = [
    "post_id341_public_transfer_audit_20260618.json",
    "post_id341_transfer_gate_v2_joint_risk_20260618.json",
    "post_id341_new_source_blueprint_20260618.json",
    "sleep_boundary_prepromotion_gate_post_id341_20260618.json",
    "evening_load_recovery_smoke_post_id341_20260618.json",
    "combined_new_source_gate_post_id341_20260618.json",
    "combined_sequence_sleep_evening_gate_post_id341_20260618.json",
    "routine_stability_place_entropy_smoke_post_id341_20260618.json",
    "place_social_novelty_smoke_fixedlist_post_id341_20260618.json",
    "sensor_adherence_dropout_smoke_post_id341_20260618.json",
    "external_etri_prior_smoke_post_id341_20260618.json",
    "combined_sleep_evening_external_prior_gate_post_id341_20260618.json",
    "post_id341_phase2_raw_source_execution_spec_20260618.json",
]
REQUIRED_REPORTS = [
    "post_id341_upstream_rebuild_blueprint_20260618.json",
    "id340_validation_split_manifest_20260618.json",
    "public_joint_distribution_risk_post_id341_20260618.json",
    "joint_risk_candidate_rerank_post_id341_20260618.json",
    "post_id341_family_priority_20260618.json",
    "post_id341_a1_qfree_lane_closure_20260618.json",
    "b1_pair_blend_source_dependency_audit_20260618.json",
    *RECENT_NEW_SOURCE_REPORTS,
]


def existing_report(path_name: str) -> dict[str, Any]:
    path = REPORTS_DIR / path_name
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def load_json_report(path_name: str) -> dict[str, Any] | None:
    path = REPORTS_DIR / path_name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find_base_files(candidate_id: int) -> dict[str, Any]:
    oof = sorted(BASE_DIR.glob(f"base_oof_id_{candidate_id}_*.csv"))
    test = sorted(BASE_DIR.glob(f"base_test_id_{candidate_id}_*.csv"))
    return {
        "id": candidate_id,
        "base_oof_exists": bool(oof),
        "base_test_exists": bool(test),
        "base_oof_file": str(oof[0].relative_to(ROOT)) if oof else None,
        "base_test_file": str(test[0].relative_to(ROOT)) if test else None,
    }


def log_record(log: pd.DataFrame, candidate_id: int) -> dict[str, Any]:
    rows = log[log["id_num"].eq(candidate_id)]
    if rows.empty:
        return {"id": candidate_id, "log_row_exists": False}
    row = rows.iloc[-1].to_dict()
    return {
        "id": candidate_id,
        "log_row_exists": True,
        "submission_file": row.get("submission_file"),
        "local_cv": None if pd.isna(row.get("local_cv_score_num")) else float(row.get("local_cv_score_num")),
        "public_lb": None if pd.isna(row.get("public_lb_score_num")) else float(row.get("public_lb_score_num")),
        "status_clean": row.get("status_clean"),
        "model_description": row.get("model_description"),
        "artifact_exists": bool(row.get("artifact_exists", False)),
    }


def source_record(log: pd.DataFrame, candidate_id: int) -> dict[str, Any]:
    record = {**find_base_files(candidate_id), **log_record(log, candidate_id)}
    record["ready_for_read_only_replay"] = bool(
        record["base_oof_exists"] and record["base_test_exists"] and record["log_row_exists"]
    )
    return record


def split_manifest_summary() -> dict[str, Any]:
    path = REPORTS_DIR / "id340_validation_split_manifest_20260618.json"
    if not path.exists():
        return {"exists": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    split_counts = Counter(row["split_name"] for row in data.get("folds", []))
    valid_n_by_split: dict[str, list[int]] = {}
    for row in data.get("folds", []):
        valid_n_by_split.setdefault(row["split_name"], []).append(int(row["valid_n"]))
    return {
        "exists": True,
        "n_train_rows": data.get("n_train_rows"),
        "n_test_rows": data.get("n_test_rows"),
        "n_folds_total": len(data.get("folds", [])),
        "split_counts": dict(split_counts),
        "valid_n_by_split": {
            split: {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}
            for split, values in valid_n_by_split.items()
        },
    }


def recent_source_read() -> dict[str, Any]:
    out: dict[str, Any] = {}
    sleep = load_json_report("sleep_boundary_prepromotion_gate_post_id341_20260618.json")
    combined = load_json_report("combined_new_source_gate_post_id341_20260618.json")
    combined_sequence = load_json_report("combined_sequence_sleep_evening_gate_post_id341_20260618.json")
    routine = load_json_report("routine_stability_place_entropy_smoke_post_id341_20260618.json")
    evening = load_json_report("evening_load_recovery_smoke_post_id341_20260618.json")
    place_social = load_json_report("place_social_novelty_smoke_fixedlist_post_id341_20260618.json")
    sensor_adherence = load_json_report("sensor_adherence_dropout_smoke_post_id341_20260618.json")
    external_prior = load_json_report("external_etri_prior_smoke_post_id341_20260618.json")
    combined_sleep_evening_external = load_json_report(
        "combined_sleep_evening_external_prior_gate_post_id341_20260618.json"
    )
    phase2_spec = load_json_report("post_id341_phase2_raw_source_execution_spec_20260618.json")
    transfer_gate_v2 = load_json_report("post_id341_transfer_gate_v2_joint_risk_20260618.json")
    if sleep:
        out["sleep_boundary"] = {
            "reviewable_feature_lead": sleep.get("decision", {}).get("reviewable_low_movement_feature_lead"),
            "promote_submission_artifact": sleep.get("decision", {}).get("promote_submission_artifact"),
            "proxy": sleep.get("test_geometry", {}).get("proxy_public_risk"),
            "proxy_minus_public_id178": sleep.get("public_context", {}).get("proxy_minus_public_id178"),
        }
    if evening:
        out["evening_load_recovery"] = {
            "selected_targets": evening.get("decision", {}).get("selected_targets"),
            "best_proxy": evening.get("decision", {}).get("best_proxy"),
            "reviewable": evening.get("decision", {}).get("reviewable_low_movement_candidate_exists"),
        }
    if combined:
        out["combined_new_source"] = {
            "best_proxy": combined.get("decision", {}).get("best_proxy"),
            "proxy_below_id178_candidate_exists": combined.get("decision", {}).get(
                "proxy_below_id178_candidate_exists"
            ),
            "public_058_plausible": combined.get("decision", {}).get("public_058_plausible"),
        }
    if combined_sequence:
        out["combined_sequence_sleep_evening"] = {
            "best_proxy": combined_sequence.get("decision", {}).get("best_proxy"),
            "best_proxy_sequence_scale": combined_sequence.get("decision", {}).get("best_proxy_sequence_scale"),
            "proxy_below_id178_candidate_exists": combined_sequence.get("decision", {}).get(
                "proxy_below_id178_candidate_exists"
            ),
            "public_058_plausible": combined_sequence.get("decision", {}).get("public_058_plausible"),
        }
    if routine:
        out["routine_stability"] = {
            "selected_targets": routine.get("decision", {}).get("selected_targets"),
            "reviewable": routine.get("decision", {}).get("reviewable_low_movement_candidate_exists"),
        }
    if place_social:
        out["place_social_novelty"] = {
            "selected_targets": place_social.get("decision", {}).get("selected_targets"),
            "reviewable": place_social.get("decision", {}).get("reviewable_low_movement_candidate_exists"),
            "best_proxy": place_social.get("decision", {}).get("best_proxy"),
            "train_missing_mean": place_social.get("feature_coverage", {}).get("train_missing_mean"),
        }
    if sensor_adherence:
        out["sensor_adherence_dropout"] = {
            "selected_targets": sensor_adherence.get("decision", {}).get("selected_targets"),
            "reviewable": sensor_adherence.get("decision", {}).get("reviewable_low_movement_candidate_exists"),
            "best_proxy": sensor_adherence.get("decision", {}).get("best_proxy"),
            "train_missing_mean": sensor_adherence.get("feature_coverage", {}).get("train_missing_mean"),
        }
    if external_prior:
        out["external_etri_prior"] = {
            "targets_available": external_prior.get("coverage", {}).get("external_targets_available"),
            "best_proxy": external_prior.get("decision", {}).get("best_proxy"),
            "reviewable": external_prior.get("decision", {}).get(
                "external_prior_reviewable_low_movement_candidate_exists"
            ),
            "public_058_plausible_candidate_exists": external_prior.get("decision", {}).get(
                "external_prior_promotable_public_058_candidate_exists"
            ),
        }
    if combined_sleep_evening_external:
        out["combined_sleep_evening_external_prior"] = {
            "best_proxy": combined_sleep_evening_external.get("decision", {}).get("best_proxy"),
            "best_proxy_sleep_scale": combined_sleep_evening_external.get("decision", {}).get("best_proxy_sleep_scale"),
            "best_proxy_evening_scale": combined_sleep_evening_external.get("decision", {}).get(
                "best_proxy_evening_scale"
            ),
            "best_proxy_external_group": combined_sleep_evening_external.get("decision", {}).get(
                "best_proxy_external_group"
            ),
            "best_proxy_external_scale": combined_sleep_evening_external.get("decision", {}).get(
                "best_proxy_external_scale"
            ),
            "reviewable": combined_sleep_evening_external.get("decision", {}).get(
                "reviewable_low_movement_candidate_exists"
            ),
            "public_058_plausible_candidate_exists": combined_sleep_evening_external.get("decision", {}).get(
                "public_058_plausible"
            ),
        }
    if phase2_spec:
        out["phase2_raw_source_execution_spec"] = {
            "phase2_spec_ready": phase2_spec.get("decision", {}).get("phase2_spec_ready"),
            "recommended_next_revision": phase2_spec.get("decision", {}).get("recommended_next_revision"),
            "create_submission_artifact": phase2_spec.get("decision", {}).get("create_submission_artifact"),
            "run_heavy_rebuild_now": phase2_spec.get("decision", {}).get("run_heavy_rebuild_now"),
        }
    if transfer_gate_v2:
        out["transfer_gate_v2_joint_risk"] = {
            "combined_rows": transfer_gate_v2.get("coverage", {}).get("combined_rows"),
            "unscored_combined_rows": transfer_gate_v2.get("coverage", {}).get("unscored_combined_rows"),
            "public_058_plausible_candidate_exists": transfer_gate_v2.get("decision", {}).get(
                "public_058_plausible_candidate_exists"
            ),
            "any_candidate_passes_combined_public_058_gate": transfer_gate_v2.get("decision", {}).get(
                "any_candidate_passes_combined_public_058_gate"
            ),
            "best_combined_pass_id": transfer_gate_v2.get("decision", {}).get("best_combined_pass_id"),
        }
    out["local_new_source_stack_read"] = {
        "proxy_below_id178_found": any(
            bool(value.get("proxy_below_id178_candidate_exists")) for value in out.values() if isinstance(value, dict)
        ),
        "public_058_plausible_found": any(
            bool(
                value.get("public_058_plausible")
                or value.get("public_058_plausible_candidate_exists")
                or value.get("external_prior_promotable_public_058_candidate_exists")
            )
            for value in out.values()
            if isinstance(value, dict)
        ),
        "force_submission_artifact": False,
    }
    return out


def next_phase_queue(source_records: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {
            "phase": "payload_and_split_smoke",
            "compute": "Kaggle CPU or local",
            "purpose": "Verify project payload, frozen split manifest, and source artifact availability.",
            "ready_now": True,
            "creates_submission": False,
        },
        {
            "phase": "upstream_model_layer_rebuild_design",
            "compute": "Kaggle CPU recommended before any long run",
            "purpose": "Define model-level approximations for fresh upstream sources using frozen folds and no public-LB feedback.",
            "ready_now": all(row["ready_for_read_only_replay"] for row in source_records["upstream_model_layer"]),
            "creates_submission": False,
        },
        {
            "phase": "a1_replay_from_clean_sources",
            "compute": "Kaggle CPU after source rebuild design passes",
            "purpose": "Replay A1 qfree selectors only from rebuilt/clean sources, not contaminated full-OOF artifacts.",
            "ready_now": all(row["ready_for_read_only_replay"] for row in source_records["a1_replay_layer"]),
            "creates_submission": False,
        },
        {
            "phase": "b1_dependency_rebuild_plan",
            "compute": "Kaggle CPU after A1 read",
            "purpose": "Plan pair/logit source dependencies below IDs 254/255/256/257/258/266 without creating a candidate CSV.",
            "ready_now": all(row["ready_for_read_only_replay"] for row in source_records["b1_replay_layer"]),
            "creates_submission": False,
        },
        {
            "phase": "transfer_gate_only",
            "compute": "local or Kaggle CPU",
            "purpose": "Compare any rebuilt source lead against ID178/287/329/341 plus the joint-risk transfer gate v2 before artifact review.",
            "ready_now": all(row["ready_for_read_only_replay"] for row in source_records["public_anchor_layer"]),
            "creates_submission": False,
        },
    ]


def build_report() -> dict[str, Any]:
    log = pd.read_csv(LOG_PATH)
    log["id_num"] = pd.to_numeric(log["id"], errors="coerce").astype("Int64")
    for col in ["local_cv_score_num", "public_lb_score_num"]:
        log[col] = pd.to_numeric(log[col], errors="coerce")
    source_records = {
        "upstream_model_layer": [source_record(log, candidate_id) for candidate_id in UPSTREAM_MODEL_LAYER],
        "a1_replay_layer": [source_record(log, candidate_id) for candidate_id in A1_REPLAY_LAYER],
        "b1_replay_layer": [source_record(log, candidate_id) for candidate_id in B1_REPLAY_LAYER],
        "public_anchor_layer": [source_record(log, candidate_id) for candidate_id in PUBLIC_ANCHOR_LAYER],
    }
    report_checks = [existing_report(name) for name in REQUIRED_REPORTS]
    missing_reports = [row["path"] for row in report_checks if not row["exists"]]
    queue = next_phase_queue(source_records)
    recent_read = recent_source_read()
    local_stack_closed = bool(
        recent_read
        and not recent_read.get("local_new_source_stack_read", {}).get("proxy_below_id178_found", False)
        and not recent_read.get("local_new_source_stack_read", {}).get("public_058_plausible_found", False)
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_upstream_rebuild_phase1_readiness",
        "purpose": "Check readiness for the CPU-only report-first upstream rebuild after local new-source lanes failed to beat ID178 Public.",
        "required_report_checks": report_checks,
        "missing_required_reports": missing_reports,
        "split_manifest_summary": split_manifest_summary(),
        "source_records": source_records,
        "recent_new_source_read": recent_read,
        "next_phase_queue": queue,
        "decision": {
            "phase1_ready": not missing_reports
            and all(row["ready_for_read_only_replay"] for row in source_records["public_anchor_layer"]),
            "local_new_source_stack_closed": local_stack_closed,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_heavy_rebuild_now": False,
            "next_action": (
                "Use this refreshed readiness report inside the Kaggle upstream runner. Local new-source stacks did not "
                "beat the ID178 public anchor, so do not create ID342 locally; the next useful approved action is a "
                "CPU-only report-first upstream rebuild runner before any artifact promotion."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Post-ID341 upstream rebuild phase-1 readiness",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It trains no model, creates no submission artifact, and makes no DACON submission.",
        "",
        "## Required report checks",
        "",
        "| path | exists | bytes |",
        "|---|---|---:|",
    ]
    for row in report["required_report_checks"]:
        lines.append(f"| `{row['path']}` | `{row['exists']}` | {row['bytes']} |")
    lines.extend(
        [
            "",
            "## Split manifest",
            "",
            f"- Exists: `{report['split_manifest_summary'].get('exists')}`",
            f"- Train/test rows: `{report['split_manifest_summary'].get('n_train_rows')}` / `{report['split_manifest_summary'].get('n_test_rows')}`",
            f"- Total folds: `{report['split_manifest_summary'].get('n_folds_total')}`",
            "",
            "## Source artifact readiness",
            "",
        ]
    )
    for layer, records in report["source_records"].items():
        lines.extend(
            [
                f"### `{layer}`",
                "",
                "| id | log | oof | test | status | local | public | ready |",
                "|---:|---|---|---|---|---:|---:|---|",
            ]
        )
        for row in records:
            local = "" if row.get("local_cv") is None else f"{row['local_cv']:.6f}"
            public = "" if row.get("public_lb") is None else f"{row['public_lb']:.6f}"
            lines.append(
                f"| {row['id']} | `{row.get('log_row_exists')}` | `{row.get('base_oof_exists')}` | "
                f"`{row.get('base_test_exists')}` | `{row.get('status_clean', '')}` | {local} | {public} | "
                f"`{row.get('ready_for_read_only_replay')}` |"
            )
        lines.append("")
    lines.extend(
        [
            "## Recent new-source read",
            "",
            "```json",
            json.dumps(report["recent_new_source_read"], indent=2, ensure_ascii=False),
            "```",
            "",
            "## Next phase queue",
            "",
            "| phase | ready now | compute | creates submission | purpose |",
            "|---|---|---|---|---|",
        ]
    )
    for row in report["next_phase_queue"]:
        lines.append(
            f"| `{row['phase']}` | `{row['ready_now']}` | {row['compute']} | `{row['creates_submission']}` | {row['purpose']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Phase-1 ready: `{report['decision']['phase1_ready']}`",
            f"- Local new-source stack closed: `{report['decision']['local_new_source_stack_closed']}`",
            f"- Create submission artifact: `{report['decision']['create_submission_artifact']}`",
            f"- Submit now: `{report['decision']['submit_now']}`",
            f"- Push Kaggle now: `{report['decision']['push_kaggle_now']}`",
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
        default=REPORTS_DIR / "post_id341_upstream_rebuild_phase1_readiness_20260618.json",
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
