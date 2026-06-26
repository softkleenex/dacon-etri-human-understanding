#!/usr/bin/env python3
"""Prepare the 2026-06-23 DACON three-slot slate policy.

This validates selected generated-unsubmitted artifacts and writes an exact
path/memo policy report that can be passed to the guarded `make dacon-submit`
target. It does not submit anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS_DIR = ROOT / "submissions"
NOTEBOOK_DIR = ROOT / "notebooks" / "history"
REPORTS_DIR = ROOT / "reports"
BASE_DIR = ROOT / "artifacts" / "base_predictions"
LOG_PATH = SUBMISSIONS_DIR / "submission_log.csv"
OUTPUT_JSON = REPORTS_DIR / "dacon_three_slot_slate_20260623.json"
OUTPUT_MD = REPORTS_DIR / "dacon_three_slot_slate_20260623.md"
TRANSFER_GATE_PATH = REPORTS_DIR / "post_id342_transfer_gate_v2_joint_risk_20260623.json"
EXPECTED_COLUMNS = ["subject_id", "sleep_date", "lifelog_date", "Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
TARGETS = EXPECTED_COLUMNS[3:]
KST = timezone(timedelta(hours=9))


@dataclass(frozen=True)
class SlateItem:
    slot: int
    submission_id: int
    file: str
    memo: str
    role: str
    rationale: str
    domain_read: str


SLATE_ITEMS = [
    SlateItem(
        slot=1,
        submission_id=338,
        file="submission_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.csv",
        memo="20260623 slate 1/3 id338 id179 s1 randomconv safety probe",
        role="conservative_anchor_variant",
        rationale="Tests whether the ID179 tempered anchor plus the same low-movement S1 source improves on ID337.",
        domain_read=(
            "S1 is a sleep-quality/sleep-state target where the random-conv temporal sequence source was the "
            "only recent remotely reproduced signal. This slot keeps all non-S1 targets anchored."
        ),
    ),
    SlateItem(
        slot=2,
        submission_id=342,
        file="submission_id_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.csv",
        memo="20260623 slate 2/3 id342 id178 id221 featuresim shrink050 independent probe",
        role="independent_feature_similarity_probe",
        rationale=(
            "Uses the unconsumed ID221 strict-feature/feature-similarity family, shrunk 50% toward ID178 to keep "
            "max movement below 0.10."
        ),
        domain_read=(
            "Feature-similarity is a non-sequence behavioral/source-family probe: it tests whether subject/day "
            "covariate similarity transfers where recent sequence micro-probes compressed near public 0.60."
        ),
    ),
    SlateItem(
        slot=3,
        submission_id=339,
        file="submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv",
        memo="20260623 slate 3/3 id339 q123 seqsum id337 s1 information probe",
        role="informative_sequence_summary_probe",
        rationale=(
            "More informative but higher-movement Q1/Q2/Q3 sequence-summary probe with ID337 S1; useful if the "
            "first two slots do not move public below the 0.60 band."
        ),
        domain_read=(
            "Q targets are subjective condition/routine responses. The sequence-summary source summarizes daily "
            "sensor trajectories, so this slot probes whether broader temporal shape helps those subjective labels."
        ),
    ),
]


def now_kst() -> str:
    return datetime.now(tz=KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def transfer_row_by_id(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in [
        "joint_safe_but_transfer_blocked",
        "top_transfer_proxy_rows",
        "local_cv_frontier_combined",
        "recent_control_rows",
    ]:
        rows.extend(report.get(key) or [])
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        candidate_id = row.get("id")
        if candidate_id is not None and int(candidate_id) not in out:
            out[int(candidate_id)] = row
    return out


def file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_glob_one(pattern: str) -> str | None:
    matches = sorted(BASE_DIR.glob(pattern))
    if len(matches) != 1:
        return None
    return str(matches[0].relative_to(ROOT))


def validate_item(item: SlateItem, log: pd.DataFrame, transfer_rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    submission_path = SUBMISSIONS_DIR / item.file
    notebook_path = NOTEBOOK_DIR / item.file.replace("submission_id_", "submission_").replace(".csv", ".ipynb")
    metadata_matches = sorted(REPORTS_DIR.glob(f"experiment_{item.submission_id:03d}_*.json"))
    decision_matches = sorted(REPORTS_DIR.glob(f"decision_{item.submission_id:03d}_*.md"))
    base_oof = artifact_glob_one(f"base_oof_id_{item.submission_id:03d}_*.csv")
    base_test = artifact_glob_one(f"base_test_id_{item.submission_id:03d}_*.csv")

    if not submission_path.exists():
        errors.append("submission_csv_missing")
        csv_shape = None
        csv_na_total = None
        target_min = None
        target_max = None
        csv_columns = None
        sha256 = None
    else:
        csv = pd.read_csv(submission_path)
        csv_shape = [int(csv.shape[0]), int(csv.shape[1])]
        csv_columns = list(csv.columns)
        csv_na_total = int(csv.isna().sum().sum())
        if csv_shape != [250, 10]:
            errors.append("csv_shape_not_250x10")
        if csv_columns != EXPECTED_COLUMNS:
            errors.append("csv_columns_mismatch")
        if csv_na_total:
            errors.append("csv_has_missing_values")
        target_values = csv[TARGETS].to_numpy(dtype=float) if csv_columns == EXPECTED_COLUMNS else None
        target_min = float(target_values.min()) if target_values is not None else None
        target_max = float(target_values.max()) if target_values is not None else None
        sha256 = file_sha256(submission_path)

    log_rows = log[log["submission_file"].astype(str).eq(item.file)]
    if len(log_rows) != 1:
        errors.append("submission_log_match_count_not_one")
        log_row: dict[str, Any] = {}
    else:
        log_row = log_rows.iloc[0].to_dict()
        if str(log_row.get("id")) != str(item.submission_id):
            errors.append("submission_log_id_mismatch")
        if str(log_row.get("is_submitted_bool")).lower() != "false":
            errors.append("submission_log_not_unsubmitted")
        if str(log_row.get("artifact_exists")).lower() != "true":
            errors.append("submission_log_artifact_exists_not_true")
        if str(log_row.get("public_lb_score_num", "")).strip() not in {"", "nan"}:
            errors.append("submission_log_public_score_already_present")

    if not notebook_path.exists():
        errors.append("notebook_missing")
    if len(metadata_matches) != 1:
        errors.append("metadata_match_count_not_one")
    if len(decision_matches) != 1:
        errors.append("decision_match_count_not_one")
    if base_oof is None:
        errors.append("base_oof_match_count_not_one")
    if base_test is None:
        errors.append("base_test_match_count_not_one")

    transfer = transfer_rows.get(item.submission_id, {})
    return {
        "slot": item.slot,
        "id": item.submission_id,
        "file": item.file,
        "path": str(submission_path.relative_to(ROOT)),
        "memo": item.memo,
        "role": item.role,
        "rationale": item.rationale,
        "domain_read": item.domain_read,
        "errors": errors,
        "ready_for_exact_approval": not errors,
        "csv_shape": csv_shape,
        "csv_columns": csv_columns,
        "csv_na_total": csv_na_total,
        "target_min": target_min,
        "target_max": target_max,
        "sha256": sha256,
        "local_cv": float(log_row["local_cv_score_num"]) if log_row else None,
        "status_clean": log_row.get("status_clean") if log_row else None,
        "notebook": str(notebook_path.relative_to(ROOT)),
        "metadata": str(metadata_matches[0].relative_to(ROOT)) if len(metadata_matches) == 1 else None,
        "decision_note": str(decision_matches[0].relative_to(ROOT)) if len(decision_matches) == 1 else None,
        "base_oof": base_oof,
        "base_test": base_test,
        "transfer_joint": {
            key: transfer.get(key)
            for key in [
                "family",
                "proxy_public_risk",
                "transfer_risk_tier",
                "joint_risk_tier",
                "joint_warning_count",
                "mean_delta_vs_178",
                "max_delta_vs_178",
                "projection_failed",
                "combined_tier",
                "block_reasons",
            ]
        },
    }


def build_report() -> dict[str, Any]:
    log = pd.read_csv(LOG_PATH, dtype=str).fillna("")
    transfer = load_json(TRANSFER_GATE_PATH)
    transfer_rows = transfer_row_by_id(transfer)
    items = [validate_item(item, log, transfer_rows) for item in SLATE_ITEMS]
    ready = all(item["ready_for_exact_approval"] for item in items)
    return {
        "created_at_kst": now_kst(),
        "diagnostic": "dacon_three_slot_slate_20260623",
        "purpose": (
            "Exact-path/memo policy for the user-approved 3-slot DACON slate. This report validates artifacts "
            "and can be used as DACON_SUBMIT_POLICY_REPORT, but it does not submit anything."
        ),
        "public_context": {
            "id178_public": 0.6014113009,
            "id337_public": 0.6006206286219425,
            "id340_public": 0.601379175253277,
            "id329_public": 0.6017908577,
            "id287_public": 0.6698787318,
            "recent_public_band_floor": 0.6006206286219425,
            "target_public": 0.58,
            "combined_public_058_gate_pass_exists": transfer.get("decision", {}).get(
                "public_058_plausible_candidate_exists"
            ),
        },
        "decision": {
            "submission_spend_policy_ready": ready,
            "score_improvement_submission_recommended": ready,
            "public_sanity_probe_recommended_by_default": ready,
            "public_sanity_probe_if_user_insists_id": SLATE_ITEMS[0].submission_id if ready else None,
            "public_sanity_probe_if_user_insists_file": SLATE_ITEMS[0].file if ready else None,
            "allowed_dacon_submit_now": ready,
            "allowed_kaggle_network_now": False,
            "allowed_new_submission_artifact_now": False,
            "public_058_plausible_candidate_exists": False,
            "requires_exact_dacon_csv_approval": True,
            "max_submission_slots": 3,
            "primary_next_action": (
                "Submit the three files one by one only after exact SUB_PATH/SUB_MEMO approval. "
                "Stop immediately on day_max_count, API failure, or unexpected public deterioration."
            ),
        },
        "slate_items": items,
        "all_items_ready": ready,
        "no_dacon_submission_made": True,
        "no_kaggle_push_or_run_made": True,
        "no_submission_artifact_created_by_this_report": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# DACON three-slot slate - 2026-06-23",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is an exact-path/memo policy report. It validates selected artifacts and writes no submission.",
        "",
        "## Decision",
        "",
    ]
    dec = report["decision"]
    lines.extend(
        [
            f"- Slate ready: `{report['all_items_ready']}`",
            f"- Submit slots allowed by this policy: `{dec['max_submission_slots']}`",
            f"- Public 0.58 plausible candidate exists: `{dec['public_058_plausible_candidate_exists']}`",
            f"- Requires exact DACON CSV approval: `{dec['requires_exact_dacon_csv_approval']}`",
            f"- Kaggle network now: `{dec['allowed_kaggle_network_now']}`",
            f"- Next action: {dec['primary_next_action']}",
            "",
            "## Public Context",
            "",
            f"- Recent public floor: `{report['public_context']['recent_public_band_floor']}`",
            f"- ID178 public: `{report['public_context']['id178_public']}`",
            f"- ID337 public: `{report['public_context']['id337_public']}`",
            f"- ID340 public: `{report['public_context']['id340_public']}`",
            f"- ID329 public: `{report['public_context']['id329_public']}`",
            f"- ID287 public collapse: `{report['public_context']['id287_public']}`",
            "",
            "## Exact Slate",
            "",
            "| slot | id | local | proxy | joint | max d178 | role | exact path | exact memo |",
            "|---:|---:|---:|---:|---|---:|---|---|---|",
        ]
    )
    for item in report["slate_items"]:
        tj = item["transfer_joint"]
        lines.append(
            f"| {item['slot']} | {item['id']} | {item['local_cv']:.6f} | "
            f"{tj.get('proxy_public_risk'):.6f} | `{tj.get('joint_risk_tier')}` | "
            f"{tj.get('max_delta_vs_178'):.6f} | `{item['role']}` | `{item['path']}` | `{item['memo']}` |"
        )
    lines.extend(["", "## Domain Rationale", ""])
    for item in report["slate_items"]:
        lines.extend(
            [
                f"### Slot {item['slot']} - ID{item['id']}",
                "",
                f"- Rationale: {item['rationale']}",
                f"- Domain read: {item['domain_read']}",
                f"- Transfer/joint blocks: `{', '.join(item['transfer_joint'].get('block_reasons') or [])}`",
                f"- Ready for exact approval: `{item['ready_for_exact_approval']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Artifact Checks",
            "",
            "| slot | id | csv | notebook | metadata | base oof | base test | errors |",
            "|---:|---:|---|---|---|---|---|---|",
        ]
    )
    for item in report["slate_items"]:
        lines.append(
            f"| {item['slot']} | {item['id']} | `{item['path']}` | `{item['notebook']}` | "
            f"`{item['metadata']}` | `{item['base_oof']}` | `{item['base_test']}` | "
            f"`{','.join(item['errors']) if item['errors'] else 'none'}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    report = build_report()
    OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_MD.relative_to(ROOT)}")
    print(json.dumps(report["decision"] | {"all_items_ready": report["all_items_ready"]}, indent=2))
    return 0 if report["all_items_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
