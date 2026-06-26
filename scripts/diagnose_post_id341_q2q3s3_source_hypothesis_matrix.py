#!/usr/bin/env python3
"""Post-ID341 Q2/Q3/S3 source-hypothesis matrix.

Report-only diagnostic. It turns the post-ID341 target gap into a concrete,
target-specific source plan for the three locally missing targets Q2, Q3, and
S3. The report reads existing audits and tiny processed-schema metadata only; it
trains no model and creates no submission artifact.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DATA_DIR = ROOT / "data"
KST = timezone(timedelta(hours=9))

FOCUS_TARGETS = ["Q2", "Q3", "S3"]
ID178_PUBLIC = 0.6014113009
PUBLIC_058_REVIEW_CUTOFF = 0.590
APPROVAL_PHRASE = "Kaggle upstream rebuild no-submit 실행 승인"

MISSING_INVENTORY_PATH = REPORTS_DIR / "post_id341_missing_target_signal_inventory_20260618.json"
TARGET_GAP_PATH = REPORTS_DIR / "post_id341_target_gap_budget_20260618.json"
UPSTREAM_PRIORITY_PATH = REPORTS_DIR / "post_id341_upstream_source_priority_matrix_20260618.json"
DATA_COVERAGE_PATH = REPORTS_DIR / "post_id341_data_source_coverage_audit_20260618.json"
PHASE2_PLAN_PATH = REPORTS_DIR / "phase2_raw_multimodal_execution_plan_post_id341_20260618.json"
PUBLIC_GAP_PATH = REPORTS_DIR / "post_id341_public_local_gap_diagnosis_20260618.json"
PROCESSED_TRAIN_PATH = DATA_DIR / "processed" / "df_train_raw.parquet"
METRICS_DESCRIPTION_PATH = DATA_DIR / "raw" / "ch2026_metrics_description_ko.md"
PAPER_NOTE_PATH = ROOT / "docs" / "domain" / "etri_lifelog_2024_paper.md"

TARGET_BLUEPRINTS: dict[str, dict[str, Any]] = {
    "Q2": {
        "definition": "pre-sleep subjective physical fatigue; 1 means lower/better fatigue than the subject mean",
        "label_risk": "individual-mean survey binarization makes subject-baseline leakage and temporal drift especially dangerous",
        "paper_baseline_hint": "The reference paper reports mUsageStats+DW as a strong Q2 baseline family.",
        "first_hypothesis": "evening/late app-load and recovery burden before sleep",
        "raw_sources": ["mUsageStats", "wHr", "wPedo", "mActivity", "mScreenStatus", "mLight", "wLight"],
        "processed_feature_patterns": [
            "mUsageStats",
            "usage_",
            "mUsage_",
            "screen_presleep",
            "screen_eve",
            "act_presleep",
            "pedo_evening",
            "wHr_",
            "slp_hr_",
            "mlight_eve",
            "wlight_eve",
        ],
        "external_value": "ETRI 2018-2020 survey fatigue scales may help as a proxy task, but direct row concatenation is blocked.",
        "fail_fast_rule": "stop if Q2 does not win at least 4/5 primary splits with negative mean delta vs ID341",
    },
    "Q3": {
        "definition": "pre-sleep subjective stress; 1 means lower/better stress than the subject mean",
        "label_risk": "stress is survey-relative and can look good under full-OOF calibration while failing public transfer",
        "paper_baseline_hint": "The reference paper shows wPedo+mUsageStats+DW as a strong Q3 combination.",
        "first_hypothesis": "social/routine novelty plus evening app and activity load",
        "raw_sources": ["mUsageStats", "mWifi", "mBle", "mGps", "mAmbience", "wPedo", "mActivity", "wHr"],
        "processed_feature_patterns": [
            "mUsageStats",
            "usage_",
            "mUsage_",
            "mGps_",
            "gps_",
            "mWifi_",
            "wifi_",
            "mBle_",
            "ble_",
            "mAmbience_",
            "amb_",
            "pedo_",
            "act_",
        ],
        "external_value": "External survey stress/fatigue scales are useful only as priors or pretraining labels, not as direct DACON rows.",
        "fail_fast_rule": "stop if the source increases joint-distribution risk or cannot beat ID341 in at least 4/5 primary splits",
    },
    "S3": {
        "definition": "objective sleep onset latency guideline adherence; 1 means acceptable sleep-onset latency",
        "label_risk": "sensor-derived onset labels are likely driven by evening boundary timing, missingness, and sleep-window alignment",
        "paper_baseline_hint": "The reference paper reports strong S3 signal from wPedo+DW and mLight+DW families.",
        "first_hypothesis": "pre-sleep quiet/dark transition and physiological settling before sleep",
        "raw_sources": ["mScreenStatus", "mActivity", "mACStatus", "mLight", "wLight", "wHr", "wPedo"],
        "processed_feature_patterns": [
            "screen_presleep",
            "screen_night",
            "act_presleep",
            "act_night",
            "ac_presleep",
            "ac_night",
            "mlight_night",
            "mlight_dark",
            "wlight_night",
            "wlight_dark",
            "slp_hr_",
            "wHr_sleep",
            "slp_pedo",
            "slp_act",
            "slp_screen",
        ],
        "external_value": "External sleep logs can inform onset/efficiency priors, but the current local gate forbids direct concat.",
        "fail_fast_rule": "stop if S3 cannot produce a low-movement all-primary improvement; do not borrow Q1/S4 sleep quality movement",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_load_json(path: Path) -> dict[str, Any] | None:
    return load_json(path) if path.exists() else None


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def processed_columns() -> list[str]:
    if not PROCESSED_TRAIN_PATH.exists():
        return []
    try:
        import pandas as pd

        return list(pd.read_parquet(PROCESSED_TRAIN_PATH).columns)
    except Exception:
        return []


def matching_columns(columns: list[str], patterns: list[str], limit: int = 24) -> dict[str, Any]:
    matches = sorted({column for column in columns for pattern in patterns if pattern in column})
    return {
        "match_count": len(matches),
        "examples": matches[:limit],
    }


def upstream_row(upstream: dict[str, Any], target: str, artifact_id: int) -> dict[str, Any] | None:
    for record in upstream.get("source_records", []):
        if record.get("id") != artifact_id:
            continue
        full_oof = record.get("full_oof", {})
        return {
            "id": artifact_id,
            "local_cv": record.get("local_cv"),
            "action": record.get("action"),
            "status_clean": record.get("status_clean"),
            "target_delta_vs_178": full_oof.get("target_delta_vs_178", {}).get(target),
            "target_delta_vs_341": full_oof.get("target_delta_vs_341", {}).get(target),
            "fold_wins_vs_341": record.get("frozen_split", {}).get("fold_wins_vs_341"),
            "fold_count": record.get("frozen_split", {}).get("fold_count"),
            "max_delta_vs_341_test": record.get("test_geometry", {}).get("max_abs_delta_vs_341"),
        }
    return None


def local_inventory_row(missing_inventory: dict[str, Any], target: str) -> dict[str, Any]:
    target_inventory = missing_inventory.get("target_inventory", {})
    row = target_inventory.get(target, {})
    local = row.get("local", {})
    return {
        "local_status": local.get("local_status"),
        "local_selected_row_count": local.get("local_selected_row_count"),
        "best_local_row": local.get("best_local_row"),
        "recommended_action": row.get("recommended_action"),
    }


def target_budget_summary(target_gap: dict[str, Any]) -> dict[str, Any]:
    strict = target_gap.get("strict_budget_projection", {})
    watch = target_gap.get("watch_budget_projection", {})
    constants = target_gap.get("constants", {})
    return {
        "strict_selected_targets": strict.get("selected_targets", []),
        "watch_selected_targets": watch.get("selected_targets", []),
        "strict_projected_proxy_from_id341": strict.get("projected_proxy_from_id341"),
        "watch_projected_proxy_from_id341": watch.get("projected_proxy_from_id341"),
        "additional_sum_delta_needed_to_beat_id178": strict.get("additional_sum_delta_needed_to_beat_id178"),
        "additional_sum_delta_needed_for_public_058": strict.get("additional_sum_delta_needed_for_public_058"),
        "required_sum_target_delta_to_beat_id178": constants.get("required_sum_target_delta_to_beat_id178"),
        "required_sum_target_delta_for_public_058": constants.get("required_sum_target_delta_for_public_058"),
    }


def data_source_summary(data_coverage: dict[str, Any] | None) -> dict[str, Any]:
    if not data_coverage:
        return {"available": False}
    raw_rows = {
        row.get("modality"): row.get("rows")
        for row in data_coverage.get("raw_modality_rows", [])
        if row.get("modality")
    }
    external_rows = {
        row.get("file"): row.get("rows") for row in data_coverage.get("external_etri_files", []) if row.get("file")
    }
    decision = data_coverage.get("decision", {})
    return {
        "available": True,
        "data_source_coverage_ready": decision.get("data_source_coverage_ready"),
        "all_major_sources_present": decision.get("all_major_sources_present"),
        "external_direct_concat_allowed": decision.get("external_direct_concat_allowed"),
        "raw_rows_by_modality": raw_rows,
        "external_rows_by_file": external_rows,
    }


def phase2_summary(phase2_plan: dict[str, Any] | None) -> dict[str, Any]:
    if not phase2_plan:
        return {"available": False}
    decision = phase2_plan.get("decision", {})
    blocks = phase2_plan.get("execution_blocks", [])
    return {
        "available": True,
        "phase2_raw_multimodal_plan_ready": decision.get("phase2_raw_multimodal_plan_ready"),
        "recommended_first_block": decision.get("recommended_first_block"),
        "block_order": [block.get("name") for block in blocks if block.get("name")],
        "create_submission_artifact": decision.get("create_submission_artifact"),
        "submit_now": decision.get("submit_now"),
        "push_kaggle_now": decision.get("push_kaggle_now"),
        "run_kaggle_now": decision.get("run_kaggle_now"),
    }


def public_gap_summary(public_gap: dict[str, Any] | None) -> dict[str, Any]:
    if not public_gap:
        return {"available": False}
    refs = public_gap.get("stress_references", {})
    decision = public_gap.get("decision", {})
    return {
        "available": True,
        "recent_gap_floor": refs.get("recent_gap_floor"),
        "recent_gap_stress": refs.get("recent_gap_stress"),
        "local_cv_needed_for_public_058_recent_floor": refs.get("local_cv_needed_for_public_058_recent_floor"),
        "local_cv_needed_for_public_058_recent_stress": refs.get("local_cv_needed_for_public_058_recent_stress"),
        "any_watch_candidate_passes_public_058_recent_floor_proxy": decision.get(
            "any_watch_candidate_passes_public_058_recent_floor_proxy"
        ),
        "create_submission_artifact": decision.get("create_submission_artifact"),
        "submit_now": decision.get("submit_now"),
    }


def target_matrix_row(
    target: str,
    columns: list[str],
    missing_inventory: dict[str, Any],
    upstream: dict[str, Any],
    data_summary: dict[str, Any],
) -> dict[str, Any]:
    blueprint = TARGET_BLUEPRINTS[target]
    id221 = upstream_row(upstream, target, 221)
    id226 = upstream_row(upstream, target, 226)
    local = local_inventory_row(missing_inventory, target)
    coverage = matching_columns(columns, blueprint["processed_feature_patterns"])
    raw_rows = data_summary.get("raw_rows_by_modality", {}) if data_summary.get("available") else {}
    source_row_counts = {source: raw_rows.get(source) for source in blueprint["raw_sources"]}
    id226_delta = safe_float((id226 or {}).get("target_delta_vs_178"))
    return {
        "target": target,
        "definition": blueprint["definition"],
        "label_risk": blueprint["label_risk"],
        "paper_baseline_hint": blueprint["paper_baseline_hint"],
        "first_hypothesis": blueprint["first_hypothesis"],
        "raw_sources": blueprint["raw_sources"],
        "source_row_counts": source_row_counts,
        "processed_feature_coverage": coverage,
        "external_value": blueprint["external_value"],
        "fail_fast_rule": blueprint["fail_fast_rule"],
        "local_inventory": local,
        "clean_root_id221": id221,
        "clean_backbone_id226": id226,
        "id226_has_negative_target_delta_vs_178": bool(id226_delta is not None and id226_delta < 0),
        "id226_all_folds_win_vs_341": bool(
            (id226 or {}).get("fold_wins_vs_341") is not None
            and (id226 or {}).get("fold_wins_vs_341") == (id226 or {}).get("fold_count")
        ),
        "recommended_route": "clean ID221 root -> clean ID226 target-wise backbone -> output-intake gate -> separate artifact review",
    }


def build_report() -> dict[str, Any]:
    missing_inventory = load_json(MISSING_INVENTORY_PATH)
    target_gap = load_json(TARGET_GAP_PATH)
    upstream = load_json(UPSTREAM_PRIORITY_PATH)
    data_coverage = maybe_load_json(DATA_COVERAGE_PATH)
    phase2_plan = maybe_load_json(PHASE2_PLAN_PATH)
    public_gap = maybe_load_json(PUBLIC_GAP_PATH)
    columns = processed_columns()
    data_summary = data_source_summary(data_coverage)

    matrix = {
        target: target_matrix_row(target, columns, missing_inventory, upstream, data_summary)
        for target in FOCUS_TARGETS
    }
    all_focus_missing = all(row["local_inventory"].get("local_status") == "missing" for row in matrix.values())
    all_focus_have_id226_signal = all(row["id226_has_negative_target_delta_vs_178"] for row in matrix.values())
    all_focus_id226_all_folds = all(row["id226_all_folds_win_vs_341"] for row in matrix.values())
    any_feature_coverage_empty = any(row["processed_feature_coverage"]["match_count"] == 0 for row in matrix.values())
    budget = target_budget_summary(target_gap)
    data = data_summary
    phase2 = phase2_summary(phase2_plan)
    gap = public_gap_summary(public_gap)

    decision = {
        "q2q3s3_source_hypothesis_matrix_ready": True,
        "all_focus_targets_missing_locally": all_focus_missing,
        "all_focus_targets_have_historical_id226_signal": all_focus_have_id226_signal,
        "all_focus_targets_id226_win_all_folds_vs_341": all_focus_id226_all_folds,
        "processed_feature_coverage_sufficient_for_design": not any_feature_coverage_empty,
        "external_direct_concat_allowed": data.get("external_direct_concat_allowed"),
        "local_q2q3s3_artifact_promotable": False,
        "create_submission_artifact": False,
        "submit_now": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "requires_explicit_kaggle_approval": True,
        "approval_phrase": APPROVAL_PHRASE,
        "recommended_next_action": (
            "Keep local ID342 closed. Use this matrix as the score-relevant design compass for the explicit-approved "
            "no-submit Kaggle upstream rebuild; only pulled reports that beat ID178 and pass Public-0.58 review gates "
            "may open a separate artifact-contract review."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_q2q3s3_source_hypothesis_matrix",
        "purpose": (
            "Convert the post-ID341 Q2/Q3/S3 gap into target-specific source hypotheses and fail-fast rules before "
            "any new model artifact, Kaggle run, or DACON submission is allowed."
        ),
        "inputs": {
            "missing_inventory": str(MISSING_INVENTORY_PATH.relative_to(ROOT)),
            "target_gap": str(TARGET_GAP_PATH.relative_to(ROOT)),
            "upstream_priority": str(UPSTREAM_PRIORITY_PATH.relative_to(ROOT)),
            "data_coverage": str(DATA_COVERAGE_PATH.relative_to(ROOT)),
            "phase2_plan": str(PHASE2_PLAN_PATH.relative_to(ROOT)),
            "public_gap": str(PUBLIC_GAP_PATH.relative_to(ROOT)),
            "processed_train": str(PROCESSED_TRAIN_PATH.relative_to(ROOT)),
            "metrics_description": str(METRICS_DESCRIPTION_PATH.relative_to(ROOT)),
            "paper_note": str(PAPER_NOTE_PATH.relative_to(ROOT)),
        },
        "constants": {
            "focus_targets": FOCUS_TARGETS,
            "id178_public": ID178_PUBLIC,
            "public_058_review_cutoff": PUBLIC_058_REVIEW_CUTOFF,
            "approval_phrase": APPROVAL_PHRASE,
        },
        "processed_schema": {
            "column_count": len(columns),
            "targets_present": [target for target in ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"] if target in columns],
        },
        "target_gap_budget": budget,
        "public_gap_stop_rule": gap,
        "data_source_summary": data,
        "phase2_plan_summary": phase2,
        "target_source_matrix": matrix,
        "decision": decision,
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def fmt(value: Any, digits: int = 6) -> str:
    out = safe_float(value)
    return "" if out is None else f"{out:.{digits}f}"


def matrix_markdown_row(target: str, row: dict[str, Any]) -> str:
    local = row["local_inventory"]
    id226 = row.get("clean_backbone_id226") or {}
    coverage = row["processed_feature_coverage"]
    return (
        f"| {target} | `{local.get('local_status')}` | "
        f"{fmt(id226.get('target_delta_vs_178'))} | "
        f"{id226.get('fold_wins_vs_341', '')}/{id226.get('fold_count', '')} | "
        f"{coverage['match_count']} | {row['first_hypothesis']} | {row['fail_fast_rule']} |"
    )


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    budget = report["target_gap_budget"]
    gap = report["public_gap_stop_rule"]
    lines = [
        "# Post-ID341 Q2/Q3/S3 source-hypothesis matrix",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Matrix ready: `{decision['q2q3s3_source_hypothesis_matrix_ready']}`",
        f"- All focus targets missing locally: `{decision['all_focus_targets_missing_locally']}`",
        f"- All focus targets have historical ID226 signal: `{decision['all_focus_targets_have_historical_id226_signal']}`",
        f"- Local Q2/Q3/S3 artifact promotable: `{decision['local_q2q3s3_artifact_promotable']}`",
        f"- External direct concat allowed: `{decision['external_direct_concat_allowed']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- Requires explicit Kaggle approval: `{decision['requires_explicit_kaggle_approval']}`",
        f"- Approval phrase: `{decision['approval_phrase']}`",
        "",
        decision["recommended_next_action"],
        "",
        "## Budget and public-gap stop line",
        "",
        f"- Strict selected targets today: `{', '.join(budget['strict_selected_targets'])}`",
        f"- Additional sum delta needed to beat ID178: `{fmt(budget['additional_sum_delta_needed_to_beat_id178'])}`",
        f"- Additional sum delta needed for Public-0.58 review: `{fmt(budget['additional_sum_delta_needed_for_public_058'])}`",
        f"- Local CV needed for Public 0.58 under recent floor: `{fmt(gap.get('local_cv_needed_for_public_058_recent_floor'))}`",
        f"- Local CV needed for Public 0.58 under recent stress: `{fmt(gap.get('local_cv_needed_for_public_058_recent_stress'))}`",
        "",
        "## Target-source matrix",
        "",
        "| target | local status | ID226 Δ178 | ID226 fold wins | feature hits | first hypothesis | fail-fast rule |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for target in report["constants"]["focus_targets"]:
        lines.append(matrix_markdown_row(target, report["target_source_matrix"][target]))
    lines.extend(["", "## Target details", ""])
    for target, row in report["target_source_matrix"].items():
        examples = ", ".join(f"`{column}`" for column in row["processed_feature_coverage"]["examples"][:10])
        raw_sources = ", ".join(f"`{source}`" for source in row["raw_sources"])
        lines.extend(
            [
                f"### {target}",
                "",
                f"- Definition: {row['definition']}",
                f"- Label risk: {row['label_risk']}",
                f"- Paper hint: {row['paper_baseline_hint']}",
                f"- Raw sources: {raw_sources}",
                f"- Processed feature examples: {examples}",
                f"- External-data use: {row['external_value']}",
                f"- Route: {row['recommended_route']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "Q2/Q3/S3 are the right next battleground, but not because there is a safe local artifact waiting. They are missing locally, while historical ID226 carries target-wise signal across all primary folds. The clean move is therefore a no-submit upstream rebuild, not another same-family local blend.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_q2q3s3_source_hypothesis_matrix_20260618.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
