#!/usr/bin/env python3
"""Report-only self-supervised subject-state embedding design after ID341.

This diagnostic freezes a heavy-compute idea without executing it. It inspects
only lightweight metadata and row contracts, then records why a label-free
multimodal day-state embedding could be materially different from the closed
local lanes. It does not read full raw streams into memory, train embeddings,
write feature matrices, create predictions or submission artifacts, edit the
submission log, run Kaggle, or submit to DACON.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
TRAIN_PATH = ROOT / "data" / "raw" / "ch2026_metrics_train.csv"
SAMPLE_PATH = ROOT / "data" / "raw" / "ch2026_submission_sample.csv"
TRAIN_PROCESSED = ROOT / "data" / "processed" / "df_train_raw.parquet"
TEST_PROCESSED = ROOT / "data" / "processed" / "df_test_raw.parquet"
KST = timezone(timedelta(hours=9))

SOURCE_FAMILY = "self_supervised_subject_state_embedding_v1"
TARGET_FOCUS = ["Q2", "Q3", "S3"]
APPROVAL_PHRASE = "Kaggle upstream rebuild no-submit 실행 승인"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_self_supervised_subject_state_embedding_design_20260619.json"

RAW_MODALITIES = {
    "mACStatus": "ch2025_mACStatus.parquet",
    "mActivity": "ch2025_mActivity.parquet",
    "mAmbience": "ch2025_mAmbience.parquet",
    "mBle": "ch2025_mBle.parquet",
    "mGps": "ch2025_mGps.parquet",
    "mLight": "ch2025_mLight.parquet",
    "mScreenStatus": "ch2025_mScreenStatus.parquet",
    "mUsageStats": "ch2025_mUsageStats.parquet",
    "mWifi": "ch2025_mWifi.parquet",
    "wHr": "ch2025_wHr.parquet",
    "wLight": "ch2025_wLight.parquet",
    "wPedo": "ch2025_wPedo.parquet",
}

CONTEXT_REPORTS = {
    "material_new_source_queue": "post_id341_material_new_source_queue_20260619.json",
    "no_approval_frontier": "post_id341_no_approval_frontier_20260619.json",
    "data_source_coverage": "post_id341_data_source_coverage_audit_20260618.json",
    "public_local_gap": "post_id341_public_local_gap_diagnosis_20260618.json",
    "local_source_closure": "post_id341_local_source_closure_synthesis_20260618.json",
    "operator_handoff": "post_id341_operator_handoff_contract_20260619.json",
}

FORBIDDEN_ACTIONS = {
    "read_full_raw_streams_now": False,
    "fit_embedding_now": False,
    "write_feature_matrix_now": False,
    "train_model_now": False,
    "write_oof_or_test_predictions_now": False,
    "create_submission_artifact": False,
    "modify_submission_log": False,
    "push_kaggle_now": False,
    "run_kaggle_now": False,
    "submit_now": False,
}


def rel(path: Path) -> str:
    path = path if path.is_absolute() else ROOT / path
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json_report(filename: str) -> dict[str, Any]:
    path = REPORTS_DIR / filename
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "decision": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "path": rel(path),
            "exists": True,
            "parse_ok": False,
            "decision": {},
            "error": str(exc),
        }
    payload["path"] = rel(path)
    payload["exists"] = True
    payload["parse_ok"] = True
    return payload


def decision(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("decision", {})
    return value if isinstance(value, dict) else {}


def parquet_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": rel(path),
            "exists": False,
            "num_rows": 0,
            "num_columns": 0,
            "columns": [],
            "has_subject_id": False,
            "has_timestamp": False,
            "size_mb": 0.0,
        }
    parquet = pq.ParquetFile(path)
    columns = list(parquet.schema.names)
    return {
        "path": rel(path),
        "exists": True,
        "num_rows": int(parquet.metadata.num_rows),
        "num_columns": int(parquet.metadata.num_columns),
        "columns": columns,
        "has_subject_id": "subject_id" in columns,
        "has_timestamp": "timestamp" in columns,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
    }


def raw_modality_manifest() -> dict[str, Any]:
    modalities = {name: parquet_metadata(RAW_DIR / filename) for name, filename in RAW_MODALITIES.items()}
    total_rows = sum(item["num_rows"] for item in modalities.values())
    total_size_mb = round(sum(item["size_mb"] for item in modalities.values()), 3)
    all_present = all(item["exists"] for item in modalities.values())
    all_stream_keyed = all(item["has_subject_id"] and item["has_timestamp"] for item in modalities.values())
    return {
        "raw_modality_count": len(modalities),
        "expected_raw_modality_count": len(RAW_MODALITIES),
        "all_expected_modalities_present": all_present,
        "all_streams_have_subject_and_timestamp": all_stream_keyed,
        "total_raw_rows": int(total_rows),
        "total_raw_size_mb": total_size_mb,
        "modalities": modalities,
    }


def processed_contract() -> dict[str, Any]:
    train_csv = pd.read_csv(TRAIN_PATH, usecols=["subject_id", "sleep_date", "lifelog_date"])
    test_csv = pd.read_csv(SAMPLE_PATH, usecols=["subject_id", "sleep_date", "lifelog_date"])
    train_meta = parquet_metadata(TRAIN_PROCESSED)
    test_meta = parquet_metadata(TEST_PROCESSED)
    target_columns = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
    key_columns = ["subject_id", "sleep_date", "lifelog_date"]
    return {
        "official_train_rows": int(len(train_csv)),
        "official_test_rows": int(len(test_csv)),
        "official_total_rows": int(len(train_csv) + len(test_csv)),
        "processed_train_rows": train_meta["num_rows"],
        "processed_test_rows": test_meta["num_rows"],
        "processed_train_columns": train_meta["num_columns"],
        "processed_test_columns": test_meta["num_columns"],
        "key_columns": key_columns,
        "target_columns": target_columns,
        "train_key_duplicates": int(train_csv.duplicated(key_columns).sum()),
        "test_key_duplicates": int(test_csv.duplicated(key_columns).sum()),
        "row_contract_exact": bool(
            len(train_csv) == 450
            and len(test_csv) == 250
            and train_meta["num_rows"] == 450
            and test_meta["num_rows"] == 250
        ),
    }


def context_summary(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    queue_dec = decision(reports["material_new_source_queue"])
    frontier_dec = decision(reports["no_approval_frontier"])
    coverage_dec = decision(reports["data_source_coverage"])
    public_gap_dec = decision(reports["public_local_gap"])
    closure_dec = decision(reports["local_source_closure"])
    handoff_dec = decision(reports["operator_handoff"])
    queue_items = reports["material_new_source_queue"].get("source_queue", [])
    self_item = next((item for item in queue_items if item.get("name") == SOURCE_FAMILY), {})
    return {
        "all_context_reports_present": all(
            report.get("exists") is True and report.get("parse_ok") is True for report in reports.values()
        ),
        "queue_ready": queue_dec.get("material_new_source_queue_ready") is True,
        "queue_records_self_supervised_heavy_deferred": bool(
            self_item.get("name") == SOURCE_FAMILY
            and self_item.get("type") == "heavy_design_only"
            and self_item.get("local_no_approval_allowed") is False
            and self_item.get("train_model_now") is False
            and self_item.get("create_artifact_allowed") is False
        ),
        "all_local_no_approval_score_routes_closed": bool(
            frontier_dec.get("all_local_no_approval_score_routes_closed") is True
            or queue_dec.get("local_no_approval_score_route_exists") is False
        ),
        "local_source_stack_closed": closure_dec.get("local_source_stack_closed_for_submission_artifact") is True,
        "data_source_coverage_ready": coverage_dec.get("data_source_coverage_ready") is True,
        "public_058_candidate_exists": bool(public_gap_dec.get("public_058_plausible_candidate_exists", False)),
        "operator_handoff_ready": handoff_dec.get("operator_handoff_ready") is True,
        "only_score_relevant_open_route": queue_dec.get("only_score_relevant_open_route"),
        "approval_phrase": APPROVAL_PHRASE,
    }


def distinctness_map() -> list[dict[str, Any]]:
    return [
        {
            "closed_lane": "usage_semantic_burst_markov_v1",
            "overlap_risk": "medium",
            "distinction": "Embedding would use synchronized multimodal day-state tokens, not app-state Markov counts alone.",
            "do_not_repeat": "Do not turn this into another app category/session aggregate scan.",
        },
        {
            "closed_lane": "gps_path_topology_sequence_v1",
            "overlap_risk": "medium",
            "distinction": "GPS is only one channel in the state token; topology/path features are not the primary representation.",
            "do_not_repeat": "Do not replay route topology or dwell-segment ordering.",
        },
        {
            "closed_lane": "circadian_hr_light_phase_shape_v1",
            "overlap_risk": "medium",
            "distinction": "Phase-shape used hand-built HR/light curves; embedding would learn cross-modal co-occurrence states.",
            "do_not_repeat": "Do not rerun circadian curve summaries with tiny model changes.",
        },
        {
            "closed_lane": "proximity_ambience_cooccurrence_v1",
            "overlap_risk": "medium",
            "distinction": "Co-occurrence was fixed WiFi/BLE/ambience windows; embedding would tokenize all sensors with masked reconstruction.",
            "do_not_repeat": "Do not produce another proximity/ambience-only matrix.",
        },
        {
            "closed_lane": "phone_wearable_discordance_v1",
            "overlap_risk": "medium",
            "distinction": "Discordance used fixed phone-vs-wearable rules; embedding would learn broader latent day states.",
            "do_not_repeat": "Do not build another deterministic discordance feature matrix.",
        },
        {
            "closed_lane": "local_oof_combo_or_second_stage_calibration",
            "overlap_risk": "critical",
            "distinction": "This design is source-level and label-free before downstream folds, not a recombination of existing OOF predictions.",
            "do_not_repeat": "No full-OOF calibration, public-anchor blend, or ID287/ID329-style cascade.",
        },
    ]


def design_stages() -> list[dict[str, Any]]:
    return [
        {
            "stage": "metadata_only_current_step",
            "allowed_now": True,
            "output": "this report only",
            "guard": "No full raw read, no embedding fit, no feature matrix, no model.",
        },
        {
            "stage": "approved_label_free_tokenization",
            "allowed_now": False,
            "output": "bounded event-token schema report only",
            "guard": "Requires a new explicit heavy-compute contract; fixed vocabularies only, no target-guided tokens.",
        },
        {
            "stage": "approved_self_supervised_pretraining",
            "allowed_now": False,
            "output": "temporary embeddings under ignored build/artifact area only",
            "guard": "Masked modality/time-bin reconstruction only; no target labels, public scores, or submission files.",
        },
        {
            "stage": "fold_internal_downstream_smoke",
            "allowed_now": False,
            "output": "report-only subject-hole fold losses",
            "guard": "Outer validation rows must not influence embedding selection, target calibration, or blending.",
        },
        {
            "stage": "artifact_contract_review",
            "allowed_now": False,
            "output": "none unless Public-0.58 gates pass later",
            "guard": "A separate decision note and exact user approval are required before any submission artifact.",
        },
    ]


def build_report() -> dict[str, Any]:
    reports = {name: read_json_report(filename) for name, filename in CONTEXT_REPORTS.items()}
    raw_manifest = raw_modality_manifest()
    row_contract = processed_contract()
    context = context_summary(reports)
    design_ready = bool(
        context["all_context_reports_present"]
        and context["queue_ready"]
        and context["queue_records_self_supervised_heavy_deferred"]
        and context["all_local_no_approval_score_routes_closed"]
        and context["local_source_stack_closed"]
        and context["data_source_coverage_ready"]
        and raw_manifest["all_expected_modalities_present"]
        and raw_manifest["all_streams_have_subject_and_timestamp"]
        and raw_manifest["raw_modality_count"] == raw_manifest["expected_raw_modality_count"]
        and raw_manifest["total_raw_rows"] >= 6_000_000
        and row_contract["row_contract_exact"]
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC%z").replace("+0900", "+09:00"),
        "diagnostic": "post_id341_self_supervised_subject_state_embedding_design",
        "report_only": True,
        "source_family": SOURCE_FAMILY,
        "target_focus": TARGET_FOCUS,
        "purpose": (
            "Freeze a possible label-free multimodal subject-state embedding route as design-only, while blocking "
            "heavy local execution and preserving the explicit-approved upstream rebuild as the only score route."
        ),
        "context_status": context,
        "raw_modality_manifest": raw_manifest,
        "row_contract": row_contract,
        "distinctness_map": distinctness_map(),
        "design_stages": design_stages(),
        "safety_contract": {
            **FORBIDDEN_ACTIONS,
            "requires_new_heavy_compute_contract_before_execution": True,
            "requires_new_artifact_contract_before_any_submission_file": True,
            "current_approval_phrase_only_applies_to_upstream_rebuild": True,
        },
        "decision": {
            "design_ready": design_ready,
            "source_family": SOURCE_FAMILY,
            "target_focus": TARGET_FOCUS,
            "materially_new_but_heavy": True,
            "local_no_approval_allowed": False,
            "requires_explicit_heavy_compute_approval": True,
            "current_score_relevant_route": context["only_score_relevant_open_route"],
            "approval_phrase": APPROVAL_PHRASE,
            "public_058_evidence": False,
            "score_route_opened": False,
            "feature_matrix_allowed_now": False,
            "train_model_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_local_allowed_step": "none_without_new_explicit_heavy_compute_contract",
            "next_action": (
                "Keep this as a design-only route. If pursued later, first write a separate heavy-compute contract; "
                "otherwise the only score-relevant action remains the explicit-approved no-submit upstream rebuild."
            ),
        },
        "no_feature_matrix_created": True,
        "no_model_trained": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
        "context_reports": {
            name: {"path": report.get("path"), "exists": report.get("exists"), "parse_ok": report.get("parse_ok")}
            for name, report in reports.items()
        },
    }


def build_markdown(report: dict[str, Any]) -> str:
    dec = report["decision"]
    raw = report["raw_modality_manifest"]
    row = report["row_contract"]
    lines = [
        "# Post-ID341 self-supervised subject-state embedding design",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        (
            "This is report-only. It reads metadata, not full raw streams, and creates no embedding, feature "
            "matrix, prediction file, submission artifact, Kaggle run, or DACON submission."
        ),
        "",
        "## Decision",
        "",
        f"- Design ready: `{dec['design_ready']}`",
        f"- Source family: `{dec['source_family']}`",
        f"- Target focus: `{','.join(dec['target_focus'])}`",
        f"- Materially new but heavy: `{dec['materially_new_but_heavy']}`",
        f"- Local no-approval allowed: `{dec['local_no_approval_allowed']}`",
        f"- Requires explicit heavy-compute approval: `{dec['requires_explicit_heavy_compute_approval']}`",
        f"- Current score-relevant route: `{dec['current_score_relevant_route']}`",
        f"- Public `0.58` evidence: `{dec['public_058_evidence']}`",
        f"- Score route opened: `{dec['score_route_opened']}`",
        f"- Feature matrix allowed now: `{dec['feature_matrix_allowed_now']}`",
        f"- Train model now: `{dec['train_model_now']}`",
        f"- Create submission artifact: `{dec['create_submission_artifact']}`",
        f"- Push Kaggle now: `{dec['push_kaggle_now']}`",
        f"- Run Kaggle now: `{dec['run_kaggle_now']}`",
        f"- Submit now: `{dec['submit_now']}`",
        f"- Next local allowed step: `{dec['next_local_allowed_step']}`",
        f"- Next action: {dec['next_action']}",
        "",
        "## Metadata feasibility",
        "",
        f"- Raw modalities present: `{raw['raw_modality_count']}/{raw['expected_raw_modality_count']}`",
        f"- Raw rows available: `{raw['total_raw_rows']}`",
        f"- Raw size MB: `{raw['total_raw_size_mb']}`",
        f"- Streams have subject/timestamp keys: `{raw['all_streams_have_subject_and_timestamp']}`",
        f"- Official train/test rows: `{row['official_train_rows']}` / `{row['official_test_rows']}`",
        f"- Processed train/test rows: `{row['processed_train_rows']}` / `{row['processed_test_rows']}`",
        "",
        "## Design stages",
        "",
        "| stage | allowed now | output | guard |",
        "|---|---|---|---|",
    ]
    for stage in report["design_stages"]:
        lines.append(f"| `{stage['stage']}` | `{stage['allowed_now']}` | {stage['output']} | {stage['guard']} |")
    lines.extend(
        [
            "",
            "## Closed-lane distinctness",
            "",
            "| closed lane | overlap risk | distinction | do not repeat |",
            "|---|---|---|---|",
        ]
    )
    for item in report["distinctness_map"]:
        lines.append(
            f"| `{item['closed_lane']}` | `{item['overlap_risk']}` | {item['distinction']} | {item['do_not_repeat']} |"
        )
    lines.extend(
        [
            "",
            "## Safety contract",
            "",
            "```json",
            json.dumps(report["safety_contract"], indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report()
    write_report(report, args.output)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.md')}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
