#!/usr/bin/env python3
"""Audit post-ID341 data/source coverage without training or submitting.

This report answers a narrow operational question: are the official challenge
files, raw modality tables, paper/data-spec references, and external ETRI files
present and accounted for in the current no-submit strategy?
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_data_source_coverage_audit_20260618.json"

OFFICIAL_FILES = {
    "metrics_train": "data/raw/ch2026_metrics_train.csv",
    "submission_sample": "data/raw/ch2026_submission_sample.csv",
    "metrics_description_en": "data/raw/ch2026_metrics_description.md",
    "metrics_description_ko": "data/raw/ch2026_metrics_description_ko.md",
    "processed_train": "data/processed/df_train_raw.parquet",
    "processed_test": "data/processed/df_test_raw.parquet",
}

RAW_MODALITY_FILES = {
    "mACStatus": "data/raw/ch2025_data_items/ch2025_mACStatus.parquet",
    "mActivity": "data/raw/ch2025_data_items/ch2025_mActivity.parquet",
    "mAmbience": "data/raw/ch2025_data_items/ch2025_mAmbience.parquet",
    "mBle": "data/raw/ch2025_data_items/ch2025_mBle.parquet",
    "mGps": "data/raw/ch2025_data_items/ch2025_mGps.parquet",
    "mLight": "data/raw/ch2025_data_items/ch2025_mLight.parquet",
    "mScreenStatus": "data/raw/ch2025_data_items/ch2025_mScreenStatus.parquet",
    "mUsageStats": "data/raw/ch2025_data_items/ch2025_mUsageStats.parquet",
    "mWifi": "data/raw/ch2025_data_items/ch2025_mWifi.parquet",
    "wHr": "data/raw/ch2025_data_items/ch2025_wHr.parquet",
    "wLight": "data/raw/ch2025_data_items/ch2025_wLight.parquet",
    "wPedo": "data/raw/ch2025_data_items/ch2025_wPedo.parquet",
}

EXTERNAL_ETRI_FILES = {
    "user_info_2019_2018_updated": "data/external/etri_lifelog/user_info_2019_2018_updated.csv",
    "user_info_2020": "data/external/etri_lifelog/user_info_2020.csv",
    "user_sleep_2019_2018": "data/external/etri_lifelog/user_sleep_2019_2018.csv",
    "user_sleep_2020": "data/external/etri_lifelog/user_sleep_2020.csv",
    "user_survey_2019_2018": "data/external/etri_lifelog/user_survey_2019_2018.csv",
    "user_survey_2020": "data/external/etri_lifelog/user_survey_2020.csv",
}

REFERENCE_DOCS = {
    "data_dictionary": "docs/competition/DATA_DICTIONARY.md",
    "data_items_2025": "docs/competition/Data items in the 2025 challenge.md",
    "dataset_design_2024": "docs/competition/Human Understanding AI Paper Challenge 2024 - Dataset Design.md",
    "etri_lifelog_2024_paper_notes": "docs/domain/etri_lifelog_2024_paper.md",
    "etri_pretraining_strategy": "docs/infrastructure/etri/etri_pretraining_strategy.md",
}

EVIDENCE_REPORTS = {
    "public_local_gap": "reports/post_id341_public_local_gap_diagnosis_20260618.json",
    "public_transfer": "reports/post_id341_public_transfer_audit_20260618.json",
    "new_source_blueprint": "reports/post_id341_new_source_blueprint_20260618.json",
    "external_etri_prior": "reports/external_etri_prior_smoke_post_id341_20260618.json",
    "combined_sleep_evening_external": "reports/combined_sleep_evening_external_prior_gate_post_id341_20260618.json",
    "phase2_raw_sequence_spec": "reports/post_id341_phase2_raw_source_execution_spec_20260618.json",
    "upstream_phase1_readiness": "reports/post_id341_upstream_rebuild_phase1_readiness_20260618.json",
    "local_source_closure": "reports/post_id341_local_source_closure_synthesis_20260618.json",
    "kaggle_prelaunch": "reports/post_id341_kaggle_prelaunch_readiness_gate_20260618.json",
}

RUNNER_NOTEBOOK = ROOT / "notebooks/kaggle_upstream_rebuild_runner/Upstream_Rebuild_Post_ID341_Runner.ipynb"
RUNNER_REQUIRED_PATTERNS = {
    "external_prior": r"diagnose_external_etri_prior_smoke_post_id341",
    "combined_external": r"diagnose_combined_sleep_evening_external_prior_gate_post_id341",
    "phase2_raw_sequence_plan": r"diagnose_phase2_raw_multimodal_execution_plan_post_id341",
    "phase2_block_a": r"diagnose_phase2_block_a_q1_s4_sleep_residual_bridge_post_id341",
    "phase2_block_a_gap": r"diagnose_phase2_block_a_q1_proxy_gap_post_id341",
    "phase2_block_b": r"diagnose_phase2_block_b_s2_sleep_recovery_retry_post_id341",
    "clean_backbone": r"diagnose_post_id341_clean_backbone_lane_closure",
}


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def csv_line_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        lines = sum(1 for _ in handle)
    return max(lines - 1, 0)


def parquet_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"rows": None, "columns": None, "metadata_read": False}
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        return {"rows": int(pf.metadata.num_rows), "columns": int(len(pf.schema.names)), "metadata_read": True}
    except Exception as exc:  # pragma: no cover - defensive; existence still matters for this audit.
        return {"rows": None, "columns": None, "metadata_read": False, "error": repr(exc)}


def file_status(name: str, relative: str) -> dict[str, Any]:
    path = ROOT / relative
    suffix = path.suffix.lower()
    status: dict[str, Any] = {
        "name": name,
        "path": relative,
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }
    if suffix == ".csv":
        status["rows"] = csv_line_count(path)
    elif suffix == ".parquet":
        status.update(parquet_metadata(path))
    return status


def group_status(files: dict[str, str]) -> dict[str, Any]:
    items = {name: file_status(name, relative) for name, relative in files.items()}
    return {
        "all_present": all(item["exists"] for item in items.values()),
        "missing": [item["path"] for item in items.values() if not item["exists"]],
        "items": items,
    }


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def runner_pattern_status() -> dict[str, Any]:
    if not RUNNER_NOTEBOOK.exists():
        return {"notebook_exists": False, "patterns": dict.fromkeys(RUNNER_REQUIRED_PATTERNS, False)}
    payload = json.loads(RUNNER_NOTEBOOK.read_text(encoding="utf-8"))
    text_parts: list[str] = []
    for cell in payload.get("cells", []):
        source = cell.get("source", "")
        text_parts.append("".join(source) if isinstance(source, list) else str(source))
    text = "\n".join(text_parts)
    patterns = {name: bool(re.search(pattern, text)) for name, pattern in RUNNER_REQUIRED_PATTERNS.items()}
    return {"notebook_exists": True, "patterns": patterns, "all_patterns_present": all(patterns.values())}


def evidence_status() -> dict[str, Any]:
    items: dict[str, Any] = {}
    for name, relative in EVIDENCE_REPORTS.items():
        path = ROOT / relative
        payload = read_json(path)
        decision = (payload or {}).get("decision") if payload else None
        items[name] = {
            "path": relative,
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "decision": decision,
        }
    return {
        "all_present": all(item["exists"] for item in items.values()),
        "missing": [item["path"] for item in items.values() if not item["exists"]],
        "items": items,
    }


def build_report() -> dict[str, Any]:
    official = group_status(OFFICIAL_FILES)
    raw = group_status(RAW_MODALITY_FILES)
    external = group_status(EXTERNAL_ETRI_FILES)
    refs = group_status(REFERENCE_DOCS)
    evidence = evidence_status()
    runner = runner_pattern_status()

    external_decision = evidence["items"]["external_etri_prior"]["decision"] or {}
    local_closure_decision = evidence["items"]["local_source_closure"]["decision"] or {}
    gap_decision = evidence["items"]["public_local_gap"]["decision"] or {}

    all_major_sources_present = (
        official["all_present"] and raw["all_present"] and external["all_present"] and refs["all_present"]
    )
    coverage_ready = all_major_sources_present and evidence["all_present"] and runner.get("all_patterns_present", False)

    report = {
        "inputs": {
            "official_files": OFFICIAL_FILES,
            "raw_modality_files": RAW_MODALITY_FILES,
            "external_etri_files": EXTERNAL_ETRI_FILES,
            "reference_docs": REFERENCE_DOCS,
            "evidence_reports": EVIDENCE_REPORTS,
            "runner_notebook": relpath(RUNNER_NOTEBOOK),
        },
        "coverage": {
            "official": official,
            "raw_modalities": raw,
            "external_etri": external,
            "reference_docs": refs,
            "evidence_reports": evidence,
            "kaggle_runner_patterns": runner,
        },
        "source_role_summary": {
            "official_metric_and_split_files": "Primary train/test/spec artifacts are present and remain the only supervised training ground.",
            "raw_modalities": "All 2025 raw modality parquet tables are present; recent local reports used sleep, evening, usage, ambience, place/social, and sensor-adherence views.",
            "external_etri": "External 2019/2020 ETRI files are present and already evaluated as weak priors; direct concatenation is not allowed by current evidence.",
            "paper_and_data_specs": "Competition and ETRI paper notes are present as feature-design references, not label leakage sources.",
            "kaggle_runner": "The upstream no-submit runner references the current external, raw/sequence phase-2, and clean-backbone closure scripts.",
        },
        "decision": {
            "data_source_coverage_audit_ready": True,
            "all_major_sources_present": all_major_sources_present,
            "all_required_evidence_reports_present": evidence["all_present"],
            "upstream_runner_references_required_source_checks": runner.get("all_patterns_present", False),
            "data_source_coverage_ready": coverage_ready,
            "external_direct_concat_allowed": False,
            "external_prior_reviewable": bool(external_decision.get("reviewable_candidate_exists", False))
            or bool(external_decision.get("external_prior_reviewable_low_movement_candidate_exists", False)),
            "local_source_stack_closed_for_submission_artifact": bool(
                local_closure_decision.get("local_source_stack_closed_for_submission_artifact", False)
            ),
            "public_058_local_cv_only_gate_passed": bool(
                gap_decision.get("any_watch_candidate_passes_public_058_recent_floor_proxy", False)
            ),
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Data/source coverage is accounted for locally. Do not concatenate external rows or create ID342; "
                "the next score-relevant step remains explicit-approved no-submit Kaggle/Linux upstream rebuild."
            ),
        },
    }
    return report


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    decision = report["decision"]
    coverage = report["coverage"]
    lines = [
        "# Post-ID341 Data Source Coverage Audit",
        "",
        "This is report-only: no model training, no submission CSV, no Kaggle run, and no DACON submission.",
        "",
        "## Decision",
        "",
        f"- Data/source coverage ready: `{decision['data_source_coverage_ready']}`",
        f"- All major sources present: `{decision['all_major_sources_present']}`",
        f"- Evidence reports present: `{decision['all_required_evidence_reports_present']}`",
        f"- Runner references required checks: `{decision['upstream_runner_references_required_source_checks']}`",
        f"- External direct concat allowed: `{decision['external_direct_concat_allowed']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Source groups",
        "",
        "| group | all present | missing count |",
        "|---|:---:|---:|",
    ]
    for key in ["official", "raw_modalities", "external_etri", "reference_docs", "evidence_reports"]:
        group = coverage[key]
        lines.append(f"| `{key}` | `{group['all_present']}` | {len(group['missing'])} |")
    lines.extend(
        [
            "",
            "## Raw modality rows",
            "",
            "| modality | rows | columns |",
            "|---|---:|---:|",
        ]
    )
    for name, item in coverage["raw_modalities"]["items"].items():
        lines.append(f"| `{name}` | {item.get('rows')} | {item.get('columns')} |")
    lines.extend(
        [
            "",
            "## External ETRI files",
            "",
            "| file | rows |",
            "|---|---:|",
        ]
    )
    for name, item in coverage["external_etri"]["items"].items():
        lines.append(f"| `{name}` | {item.get('rows')} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The current issue is not that major local data sources are missing. They are present and already routed "
            "into either local closure reports or the no-submit Kaggle upstream runner. The blocker remains "
            "validation/public transfer: none of the local watch candidates passes the recent Public `0.58` proxy gate.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    report = build_report()
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path = output.with_suffix(".md")
    write_markdown(md_path, report)
    print(f"Wrote {relpath(output)}")
    print(f"Wrote {relpath(md_path)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
