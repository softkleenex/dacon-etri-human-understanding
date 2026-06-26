#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""In-memory feature-generation dry run for phone/wearable discordance v1.

This report-only gate assembles the frozen 85-column phone/wearable
discordance schema in memory, verifies exact 450 train + 250 test row
alignment, checks finite bounded values, then stops. It does not persist a
feature matrix, train a model, create predictions or submission artifacts, edit
the submission log, run Kaggle, or submit to DACON.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id341_phone_wearable_discordance_design import (  # noqa: E402
    FEATURE_GROUPS,
    SOURCE_FAMILY,
    TARGET_FOCUS,
    WINDOWS,
    build_hourly_frame,
    load_rows,
    rel,
)
from scripts.diagnose_post_id341_phone_wearable_discordance_feature_spec import (  # noqa: E402
    DEFAULT_OUTPUT as FEATURE_SPEC_OUTPUT,
)
from scripts.diagnose_post_id341_phone_wearable_discordance_feature_spec import (
    FEATURE_COLUMN_BOUND,
    FEATURE_PREFIX,
    KEY_COLUMNS,
    decision,
    feature_names,
    load_required_reports,
)
from scripts.diagnose_post_id341_phone_wearable_discordance_feature_spec import (
    NEXT_ALLOWED_STEP as SPEC_ALLOWED_STEP,
)

REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))

DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_phone_wearable_discordance_feature_generation_dryrun_20260619.json"
FEATURE_SPEC_REPORT = FEATURE_SPEC_OUTPUT
QUEUE_REPORT = REPORTS_DIR / "post_id341_material_new_source_queue_20260619.json"
CURRENT_QUEUE_STEP = "report_only_phone_wearable_discordance_feature_generation_dryrun_no_matrix"
NEXT_ALLOWED_STEP = "report_only_phone_wearable_discordance_feature_matrix_smoke_contract"
POST_CONTRACT_ALLOWED_STEP = "write_phone_wearable_discordance_feature_matrix_under_contract_no_model_no_submission"
POST_VALIDATION_ALLOWED_STEP = (
    "validate_phone_wearable_discordance_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval"
)
LANE_CLOSED_ALLOWED_STEP = "stop_or_wait_for_explicit_kaggle_upstream_rebuild_no_submit_approval"
ALLOWED_QUEUE_STEP_TYPES = {
    "report_only_phone_wearable_discordance_feature_spec_no_matrix",
    CURRENT_QUEUE_STEP,
    NEXT_ALLOWED_STEP,
    POST_CONTRACT_ALLOWED_STEP,
    POST_VALIDATION_ALLOWED_STEP,
    LANE_CLOSED_ALLOWED_STEP,
}
FORBIDDEN_DECISION_KEYS = [
    "feature_matrix_allowed_now",
    "train_model_now",
    "create_submission_artifact",
    "submit_now",
    "push_kaggle_now",
    "run_kaggle_now",
]


def load_json_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "parse_ok": False, "decision": {}, "error": None}
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


def no_forbidden_actions(report: dict[str, Any]) -> bool:
    dec = decision(report)
    return all(dec.get(key) is not True for key in FORBIDDEN_DECISION_KEYS)


def queue_contract_ready(queue: dict[str, Any]) -> bool:
    queue_decision = decision(queue)
    items = queue.get("source_queue", []) if isinstance(queue.get("source_queue"), list) else []
    item = next((entry for entry in items if entry.get("name") == SOURCE_FAMILY), {})
    queue_item_ready = (
        (item.get("status") == "feature_spec_ready_dryrun_only" and item.get("next_allowed_step") == CURRENT_QUEUE_STEP)
        or (
            item.get("status") == "feature_generation_dryrun_passed_contract_only"
            and item.get("next_allowed_step") == NEXT_ALLOWED_STEP
        )
        or (
            item.get("status") == "feature_matrix_smoke_contract_ready_write_only_under_contract"
            and item.get("next_allowed_step") == POST_CONTRACT_ALLOWED_STEP
        )
        or (
            item.get("status")
            in {"feature_matrix_materialized_under_contract_no_model", "validated_stop_completed_no_model"}
            and item.get("feature_matrix_materialized_under_contract") is True
        )
    )
    queue_item_future_matrix_only = bool(
        item.get("status") == "feature_matrix_smoke_contract_ready_write_only_under_contract"
        and item.get("create_artifact_allowed") is True
        and item.get("feature_matrix_write_allowed_next_under_contract") is True
        and item.get("submission_artifact_allowed") is False
    )
    queue_item_no_matrix_yet_or_future_only = bool(
        (
            item.get("create_artifact_allowed") is False
            and item.get("feature_matrix_write_allowed_next_under_contract") is False
        )
        or queue_item_future_matrix_only
    )
    queue_item_local_boundary_ok = bool(
        (queue_decision.get("next_no_approval_step") == SOURCE_FAMILY and item.get("local_no_approval_allowed") is True)
        or (
            queue_decision.get("next_no_approval_step") == "none_local_no_approval_lanes_closed"
            and item.get("status") == "validated_stop_completed_no_model"
            and item.get("local_no_approval_allowed") is False
            and item.get("next_allowed_step") == "closed_stop_or_wait_for_explicit_kaggle_approval"
        )
    )
    return bool(
        queue.get("parse_ok") is True
        and queue_decision.get("material_new_source_queue_ready") is True
        and queue_decision.get("next_no_approval_step") in {SOURCE_FAMILY, "none_local_no_approval_lanes_closed"}
        and queue_decision.get("next_no_approval_step_type") in ALLOWED_QUEUE_STEP_TYPES
        and item.get("name") == SOURCE_FAMILY
        and queue_item_ready
        and queue_item_local_boundary_ok
        and item.get("train_model_now") is False
        and queue_item_no_matrix_yet_or_future_only
        and item.get("submission_artifact_allowed") is False
        and no_forbidden_actions(queue)
    )


def annotated_hourly_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = load_rows()
    hourly = build_hourly_frame(rows)
    hourly = hourly.merge(rows[["row_id", "sleep_date"]], on="row_id", how="left", validate="many_to_one")
    hourly["screen_observed"] = hourly["screen_records"].fillna(0).gt(0)
    hourly["charge_observed"] = hourly["charge_records"].fillna(0).gt(0)
    hourly["activity_observed"] = hourly["activity_records"].fillna(0).gt(0)
    hourly["pedo_observed"] = hourly["pedo_records"].fillna(0).gt(0)
    hourly["hr_observed"] = hourly["hr_records"].fillna(0).gt(0)
    hourly["wlight_observed"] = hourly["wlight_records"].fillna(0).gt(0)
    hourly["mlight_observed"] = hourly["mlight_records"].fillna(0).gt(0)

    hourly["screen_off"] = hourly["screen_on_rate"].fillna(0.0).le(0.05) & hourly["screen_observed"]
    hourly["charging_state"] = hourly["charging_rate"].fillna(0.0).ge(0.5) & hourly["charge_observed"]
    hourly["phone_still_state"] = hourly["phone_still_rate"].fillna(0.0).ge(0.5) & hourly["activity_observed"]
    hourly["phone_active_state"] = hourly["phone_active_rate"].fillna(0.0).ge(0.5) & hourly["activity_observed"]
    hourly["step_active_state"] = hourly["step_active_rate"].fillna(0.0).ge(0.05) & hourly["pedo_observed"]
    hourly["wearable_light_present"] = hourly["wlight_observed"] & hourly["wlight_mean"].fillna(0.0).ge(0.0)
    hourly["wearable_active_state"] = (
        hourly["step_active_state"] | hourly["hr_observed"] | hourly["wearable_light_present"]
    )
    hourly["phone_rest_state"] = (hourly["screen_off"] | hourly["charging_state"] | hourly["phone_still_state"]) & (
        hourly["screen_observed"] | hourly["charge_observed"] | hourly["activity_observed"]
    )
    hourly["screen_off_step_active"] = hourly["screen_off"] & hourly["step_active_state"]
    hourly["charging_step_active"] = hourly["charging_state"] & hourly["step_active_state"]
    hourly["phone_rest_wearable_active"] = hourly["phone_rest_state"] & hourly["wearable_active_state"]
    hourly["phone_dark_wearable_light"] = hourly["screen_off"] & hourly["wearable_light_present"]
    hourly["discordance_any"] = (
        hourly["screen_off_step_active"]
        | hourly["charging_step_active"]
        | hourly["phone_rest_wearable_active"]
        | hourly["phone_dark_wearable_light"]
    )
    return rows, hourly


def first_after_phone_rest(part: pd.DataFrame) -> tuple[float, float]:
    rest_hours = sorted(part.loc[part["phone_rest_state"], "hour"].astype(int).tolist())
    wearable_hours = sorted(part.loc[part["wearable_active_state"], "hour"].astype(int).tolist())
    if not rest_hours or not wearable_hours:
        return 0.0, 24.0
    for rest_hour in rest_hours:
        after = [hour for hour in wearable_hours if hour >= rest_hour]
        if after:
            first = after[0]
            return float(first + 1), float(first - rest_hour)
    return 0.0, 24.0


def max_contiguous_true(hours: list[int]) -> int:
    if not hours:
        return 0
    ordered = sorted(set(hours))
    best = 1
    current = 1
    for prev, cur in zip(ordered, ordered[1:], strict=False):
        if cur == prev + 1:
            current += 1
        else:
            best = max(best, current)
            current = 1
    return max(best, current)


def aggregate_window(hourly: pd.DataFrame, window: str) -> pd.DataFrame:
    hours = set(WINDOWS[window]["hours"])
    part = hourly[hourly["hour"].isin(hours)].copy()
    prefix = f"{FEATURE_PREFIX}{window}"
    records: list[dict[str, Any]] = []
    for (row_id, subject_id, sleep_date, lifelog_date, split), group in part.groupby(
        ["row_id", "subject_id", "sleep_date", "lifelog_date", "split"], sort=False
    ):
        hour_count = max(len(hours), 1)
        first_wearable_hour, wearable_gap = first_after_phone_rest(group)
        phone_active_hours = group.loc[group["phone_active_state"], "hour"].astype(int)
        record = {
            "row_id": row_id,
            "subject_id": subject_id,
            "sleep_date": sleep_date,
            "lifelog_date": lifelog_date,
            "split": split,
            f"{prefix}_phone_rest_state_screen_off_rate": float(group["screen_off"].sum() / hour_count),
            f"{prefix}_phone_rest_state_charging_rate": float(group["charging_state"].sum() / hour_count),
            f"{prefix}_phone_rest_state_phone_still_rate": float(group["phone_still_state"].sum() / hour_count),
            f"{prefix}_phone_rest_state_screen_off_charging_bins": float(
                (group["screen_off"] & group["charging_state"]).sum()
            ),
            f"{prefix}_wearable_activation_state_step_active_rate": float(
                group["step_active_state"].sum() / hour_count
            ),
            f"{prefix}_wearable_activation_state_step_sum_log1p": float(
                np.log1p(group["step_sum"].fillna(0.0).clip(lower=0.0).sum())
            ),
            f"{prefix}_wearable_activation_state_hr_observed_rate": float(group["hr_observed"].sum() / hour_count),
            f"{prefix}_wearable_activation_state_wearable_light_log1p": float(
                np.log1p(group["wlight_mean"].fillna(0.0).clip(lower=0.0).sum())
            ),
            f"{prefix}_discordance_state_screen_off_step_active_bins": float(group["screen_off_step_active"].sum()),
            f"{prefix}_discordance_state_charging_step_active_bins": float(group["charging_step_active"].sum()),
            f"{prefix}_discordance_state_phone_rest_wearable_active_bins": float(
                group["phone_rest_wearable_active"].sum()
            ),
            f"{prefix}_discordance_state_phone_dark_wearable_light_bins": float(
                group["phone_dark_wearable_light"].sum()
            ),
            f"{prefix}_transition_hysteresis_last_phone_active_hour": float(
                int(phone_active_hours.max()) + 1 if len(phone_active_hours) else 0
            ),
            f"{prefix}_transition_hysteresis_first_wearable_active_hour_after_phone_rest": first_wearable_hour,
            f"{prefix}_transition_hysteresis_wearable_after_phone_rest_gap_hours": wearable_gap,
        }
        records.append(record)
    return pd.DataFrame.from_records(records)


def aggregate_global(hourly: pd.DataFrame) -> pd.DataFrame:
    source_cols = [
        "screen_observed",
        "charge_observed",
        "activity_observed",
        "pedo_observed",
        "hr_observed",
        "wlight_observed",
        "mlight_observed",
    ]
    records: list[dict[str, Any]] = []
    for (row_id, subject_id, sleep_date, lifelog_date, split), group in hourly.groupby(
        ["row_id", "subject_id", "sleep_date", "lifelog_date", "split"], sort=False
    ):
        covered_count = sum(bool(group[col].any()) for col in source_cols)
        discordance_hours = group.loc[group["discordance_any"], "hour"].astype(int).tolist()
        phone_rest_total = int(group["phone_rest_state"].sum())
        discordance_total = int(group["discordance_any"].sum())
        late_mask = group["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5])
        late_discordance = int((group["discordance_any"] & late_mask).sum())
        records.append(
            {
                "row_id": row_id,
                "subject_id": subject_id,
                "sleep_date": sleep_date,
                "lifelog_date": lifelog_date,
                "split": split,
                f"{FEATURE_PREFIX}global_day_source_coverage_index": float(covered_count / len(source_cols)),
                f"{FEATURE_PREFIX}global_phone_rest_total_hour_bins": float(phone_rest_total),
                f"{FEATURE_PREFIX}global_wearable_active_total_hour_bins": float(group["wearable_active_state"].sum()),
                f"{FEATURE_PREFIX}global_discordance_total_hour_bins": float(discordance_total),
                f"{FEATURE_PREFIX}global_discordance_to_phone_rest_ratio": float(
                    discordance_total / max(phone_rest_total, 1)
                ),
                f"{FEATURE_PREFIX}global_late_discordance_share": float(late_discordance / max(discordance_total, 1)),
                f"{FEATURE_PREFIX}global_first_discordance_hour": float(
                    min(discordance_hours) + 1 if discordance_hours else 0
                ),
                f"{FEATURE_PREFIX}global_last_discordance_hour": float(
                    max(discordance_hours) + 1 if discordance_hours else 0
                ),
                f"{FEATURE_PREFIX}global_max_contiguous_discordance_hours": float(
                    max_contiguous_true(discordance_hours)
                ),
                f"{FEATURE_PREFIX}global_source_missingness_guard_count": float(len(source_cols) - covered_count),
            }
        )
    return pd.DataFrame.from_records(records)


def assemble_feature_frame() -> pd.DataFrame:
    rows, hourly = annotated_hourly_frame()
    frame = rows[["row_id", *KEY_COLUMNS, "split"]].copy()
    for window in WINDOWS:
        frame = frame.merge(
            aggregate_window(hourly, window),
            on=["row_id", *KEY_COLUMNS, "split"],
            how="left",
            validate="one_to_one",
        )
    frame = frame.merge(
        aggregate_global(hourly),
        on=["row_id", *KEY_COLUMNS, "split"],
        how="left",
        validate="one_to_one",
    )
    names = feature_names()
    frame[names] = frame[names].fillna(0.0).astype(float)
    return frame[[*KEY_COLUMNS, "split", *names]]


def feature_summary(frame: pd.DataFrame) -> dict[str, Any]:
    names = feature_names()
    values = frame[names].to_numpy(dtype=float)
    return {
        "row_count": int(len(frame)),
        "train_rows": int(frame["split"].eq("train").sum()),
        "test_rows": int(frame["split"].eq("test").sum()),
        "column_count_including_keys": int(frame.shape[1]),
        "feature_column_count": int(len(names)),
        "expected_feature_column_count": 85,
        "feature_columns_match_spec": list(frame.columns[-len(names) :]) == names,
        "duplicate_key_rows": int(frame.duplicated(KEY_COLUMNS + ["split"]).sum()),
        "finite_all_values": bool(np.isfinite(values).all()),
        "nan_count": int(np.isnan(values).sum()),
        "feature_min": float(np.nanmin(values)) if values.size else None,
        "feature_max": float(np.nanmax(values)) if values.size else None,
        "zero_variance_feature_count": int((frame[names].nunique(dropna=False) <= 1).sum()),
        "nonzero_feature_count": int((frame[names].sum(axis=0).abs() > 0).sum()),
    }


def build_report() -> dict[str, Any]:
    _design, spec_from_helper = load_required_reports()
    spec = load_json_report(FEATURE_SPEC_REPORT)
    queue = load_json_report(QUEUE_REPORT)
    spec_decision = decision(spec)
    frame = assemble_feature_frame()
    summary = feature_summary(frame)
    names = feature_names()
    spec_contract_ready = bool(
        spec.get("exists")
        and spec.get("parse_ok")
        and spec_decision.get("feature_spec_ready") is True
        and spec_decision.get("source_family") == SOURCE_FAMILY
        and spec_decision.get("target_focus") == TARGET_FOCUS
        and spec_decision.get("feature_count") == len(names)
        and spec_decision.get("next_local_allowed_step") == SPEC_ALLOWED_STEP
        and spec_from_helper.get("parse_ok") is True
        and no_forbidden_actions(spec)
    )
    frame_contract_ready = bool(
        summary["row_count"] == 700
        and summary["train_rows"] == 450
        and summary["test_rows"] == 250
        and summary["feature_column_count"] == 85
        and summary["feature_column_count"] <= FEATURE_COLUMN_BOUND
        and summary["feature_columns_match_spec"]
        and summary["duplicate_key_rows"] == 0
        and summary["finite_all_values"]
        and summary["nan_count"] == 0
        and summary["feature_min"] is not None
        and summary["feature_min"] >= 0.0
        and summary["feature_max"] is not None
        and summary["feature_max"] <= 24.0
        and summary["nonzero_feature_count"] >= 70
        and summary["zero_variance_feature_count"] <= 10
    )
    queue_ready = queue_contract_ready(queue)
    dryrun_passed = bool(spec_contract_ready and frame_contract_ready and queue_ready)
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_phone_wearable_discordance_feature_generation_dryrun",
        "report_only": True,
        "source_family": SOURCE_FAMILY,
        "target_focus": TARGET_FOCUS,
        "input_reports": {
            "feature_spec": {
                "path": rel(FEATURE_SPEC_REPORT),
                "exists": spec.get("exists"),
                "parse_ok": spec.get("parse_ok"),
            },
            "material_new_source_queue": {
                "path": rel(QUEUE_REPORT),
                "exists": queue.get("exists"),
                "parse_ok": queue.get("parse_ok"),
            },
        },
        "feature_frame_summary": summary,
        "checks": {
            "spec_contract_ready": spec_contract_ready,
            "queue_contract_ready": queue_ready,
            "frame_contract_ready": frame_contract_ready,
            "feature_count_within_bound": summary["feature_column_count"] <= FEATURE_COLUMN_BOUND,
            "all_features_prefixed": all(name.startswith(FEATURE_PREFIX) for name in names),
            "no_feature_matrix_written": True,
            "no_model_boundary": True,
        },
        "feature_groups": {
            name: {
                "metric_count": len(metrics),
                "windowed_feature_count": len(metrics) * len(WINDOWS),
            }
            for name, metrics in FEATURE_GROUPS.items()
        },
        "decision": {
            "feature_generation_dryrun_passed": dryrun_passed,
            "source_family": SOURCE_FAMILY,
            "target_focus": TARGET_FOCUS,
            "selected_targets": [],
            "feature_count": len(names),
            "public_058_evidence": False,
            "score_route_opened": False,
            "feature_matrix_allowed_now": False,
            "train_model_now": False,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_local_allowed_step": NEXT_ALLOWED_STEP,
            "next_action": (
                "Dry run passed. A future no-approval step may only write a report-only feature-matrix smoke "
                "contract; do not write a feature matrix, train a model, create ID342, run Kaggle, or submit."
                if dryrun_passed
                else "Fix feature-spec, queue, or frame-contract prerequisites before any next step."
            ),
        },
        "no_feature_matrix_created": True,
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
    dec = report["decision"]
    summary = report["feature_frame_summary"]
    lines = [
        "# Post-ID341 phone/wearable discordance feature-generation dry run",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only and in-memory. It creates no feature matrix, model, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.",
        "",
        "## Decision",
        "",
        f"- Dry run passed: `{dec['feature_generation_dryrun_passed']}`",
        f"- Source family: `{dec['source_family']}`",
        f"- Target focus: `{','.join(dec['target_focus'])}`",
        f"- Feature count: `{dec['feature_count']}`",
        f"- Public `0.58` evidence: `{dec['public_058_evidence']}`",
        f"- Score route opened: `{dec['score_route_opened']}`",
        f"- Feature matrix allowed now: `{dec['feature_matrix_allowed_now']}`",
        f"- Train model now: `{dec['train_model_now']}`",
        f"- Create submission artifact: `{dec['create_submission_artifact']}`",
        f"- Submit now: `{dec['submit_now']}`",
        f"- Push Kaggle now: `{dec['push_kaggle_now']}`",
        f"- Run Kaggle now: `{dec['run_kaggle_now']}`",
        f"- Next local allowed step: `{dec['next_local_allowed_step']}`",
        f"- Next action: {dec['next_action']}",
        "",
        "## Feature frame summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Checks", ""])
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Feature groups", ""])
    for group, row in report["feature_groups"].items():
        lines.append(
            f"- `{group}`: metrics `{row['metric_count']}`, windowed features `{row['windowed_feature_count']}`"
        )
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
    return 0 if report["decision"]["feature_generation_dryrun_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
