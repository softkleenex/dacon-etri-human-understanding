#!/usr/bin/env python3
"""Prioritize post-ID341 upstream source rebuild work.

This diagnostic is report-only. It inspects existing OOF/test base prediction
artifacts, frozen validation splits, known public-failure directions, and
dependency roles to decide which upstream sources should be rebuilt first.

It trains no model, creates no submission CSV/base prediction artifact/notebook
history artifact/submission-log row, and makes no DACON submission.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
BASE_DIR = ROOT / "artifacts" / "base_predictions"
LOG_PATH = ROOT / "submissions" / "submission_log.csv"
TRAIN_LABEL_PATH = ROOT / "data" / "raw" / "ch2026_metrics_train.csv"
SPLIT_MANIFEST_PATH = REPORTS_DIR / "id340_validation_split_manifest_20260618.json"
KST = timezone(timedelta(hours=9))

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SOURCE_LAYERS = {
    "upstream_model_layer": [221, 226, 227, 228, 237],
    "a1_replay_layer": [238, 239, 241],
    "b1_replay_layer": [254, 255, 256, 257, 258, 266],
    "public_anchor_layer": [178, 287, 329, 341],
}
ALL_SOURCE_IDS = [candidate_id for ids in SOURCE_LAYERS.values() for candidate_id in ids]
PUBLIC_FAILURE_IDS = [287, 329]
REFERENCE_ID = 178
CONTROL_ID = 341

DEPENDENCY_READS = {
    178: "public_anchor_reference",
    221: "root public-feature-similarity overlay; prerequisite for ID226/ID238 lineage",
    226: "primary target-wise convex backbone; prerequisite for ID227/228/237 and downstream A1/B1",
    227: "second-stage full-OOF calibration derivative of ID226; replay only after clean ID226",
    228: "second-stage full-OOF calibration derivative of ID226; replay only after clean ID226",
    237: "temperature-grid/S4-q160 derivative of ID226; replay only after clean ID226",
    238: "A1 pair/logit source depending on 221/226/227/228; not clean as a root",
    239: "A1 recalibration on ID238; not clean as a root",
    241: "A1 q160/S4 isotonic recalibration on ID238; not clean as a root",
    254: "B1 bridge artifact with high non-nested dependencies; not clean as a root",
    255: "B1 bridge artifact with q240/high non-nested dependencies; not clean as a root",
    256: "B1 bridge artifact with q240/high non-nested dependencies; not clean as a root",
    257: "B1 bridge artifact with q240/high non-nested dependencies; not clean as a root",
    258: "B1 bridge artifact with high non-nested dependencies; not clean as a root",
    266: "B1 pair blend on contaminated B1 sources; not clean as a root",
    287: "known public-collapse calibration artifact",
    329: "known public-gap aggregate combo artifact",
    341: "current generated-unsubmitted ID178-neighborhood control",
}


def find_base_file(kind: str, candidate_id: int) -> Path:
    matches = sorted(BASE_DIR.glob(f"base_{kind}_id_{candidate_id}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"Missing base_{kind} file for ID{candidate_id}")
    return matches[0]


def load_labels() -> pd.DataFrame:
    labels = pd.read_csv(TRAIN_LABEL_PATH)
    labels = labels.rename(columns={"lifelog_date": "date"}).reset_index(names="row_idx")
    required = ["row_idx", "subject_id", "date", *TARGETS]
    missing = [column for column in required if column not in labels.columns]
    if missing:
        raise ValueError(f"Label file missing columns: {missing}")
    labels["date"] = labels["date"].astype(str)
    return labels[required]


def load_log() -> pd.DataFrame:
    log = pd.read_csv(LOG_PATH)
    log["id_num"] = pd.to_numeric(log["id"], errors="coerce").astype("Int64")
    for column in ["local_cv_score_num", "public_lb_score_num"]:
        log[column] = pd.to_numeric(log[column], errors="coerce")
    return log


def log_record(log: pd.DataFrame, candidate_id: int) -> dict[str, Any]:
    rows = log[log["id_num"].eq(candidate_id)]
    if rows.empty:
        return {"id": candidate_id, "log_row_exists": False}
    row = rows.iloc[-1]
    return {
        "id": candidate_id,
        "log_row_exists": True,
        "submission_file": row.get("submission_file", ""),
        "model_description": row.get("model_description", ""),
        "local_cv": None if pd.isna(row.get("local_cv_score_num")) else float(row.get("local_cv_score_num")),
        "public_lb": None if pd.isna(row.get("public_lb_score_num")) else float(row.get("public_lb_score_num")),
        "status_clean": row.get("status_clean", ""),
        "artifact_exists": bool(row.get("artifact_exists", False)),
    }


def load_oof(candidate_id: int, labels: pd.DataFrame) -> np.ndarray:
    path = find_base_file("oof", candidate_id)
    pred = pd.read_csv(path)
    missing = [column for column in ["subject_id", "date", *TARGETS] if column not in pred.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    pred["date"] = pred["date"].astype(str)
    merged = labels[["row_idx", "subject_id", "date"]].merge(
        pred[["subject_id", "date", *TARGETS]],
        on=["subject_id", "date"],
        how="left",
        validate="one_to_one",
    )
    if merged[TARGETS].isna().any().any():
        raise ValueError(f"{path} does not align to train labels.")
    merged = merged.sort_values("row_idx")
    return merged[TARGETS].astype(float).to_numpy()


def load_test(candidate_id: int) -> np.ndarray:
    path = find_base_file("test", candidate_id)
    pred = pd.read_csv(path)
    missing = [target for target in TARGETS if target not in pred.columns]
    if missing:
        raise ValueError(f"{path} missing target columns: {missing}")
    return pred[TARGETS].astype(float).to_numpy()


def binary_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pred = np.clip(y_pred.astype(float), 1e-6, 1 - 1e-6)
    true = y_true.astype(float)
    return float(-(true * np.log(pred) + (1 - true) * np.log(1 - pred)).mean())


def target_losses(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {target: binary_log_loss(y_true[:, idx], y_pred[:, idx]) for idx, target in enumerate(TARGETS)}


def average_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(list(target_losses(y_true, y_pred).values())))


def projection(vec: np.ndarray, direction: np.ndarray) -> float:
    denom = float(np.dot(direction, direction))
    return float(np.dot(vec, direction) / denom) if denom > 0 else 0.0


def layer_for(candidate_id: int) -> str:
    for layer, ids in SOURCE_LAYERS.items():
        if candidate_id in ids:
            return layer
    return "unknown"


def split_rows() -> list[dict[str, Any]]:
    data = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    return data.get("folds", [])


def fold_loss_rows(y_true: np.ndarray, oof_by_id: dict[int, np.ndarray]) -> dict[int, list[dict[str, Any]]]:
    rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for fold in split_rows():
        valid_idx = np.array(fold["valid_idx"], dtype=int)
        for candidate_id, pred in oof_by_id.items():
            rows[candidate_id].append(
                {
                    "split_name": fold["split_name"],
                    "fold_id": int(fold["fold_id"]),
                    "loss": average_loss(y_true[valid_idx], pred[valid_idx]),
                    "valid_n": int(len(valid_idx)),
                }
            )
    return rows


def summarize_split_rows(
    candidate_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ref_by_key = {(row["split_name"], row["fold_id"]): row["loss"] for row in reference_rows}
    ctl_by_key = {(row["split_name"], row["fold_id"]): row["loss"] for row in control_rows}
    split_summary: dict[str, dict[str, Any]] = {}
    fold_delta_ref = []
    fold_delta_ctl = []
    wins_ref = 0
    wins_ctl = 0
    for row in candidate_rows:
        key = (row["split_name"], row["fold_id"])
        delta_ref = float(row["loss"] - ref_by_key[key])
        delta_ctl = float(row["loss"] - ctl_by_key[key])
        fold_delta_ref.append(delta_ref)
        fold_delta_ctl.append(delta_ctl)
        wins_ref += int(delta_ref < 0)
        wins_ctl += int(delta_ctl < 0)
        slot = split_summary.setdefault(
            row["split_name"],
            {"losses": [], "delta_vs_178": [], "delta_vs_341": []},
        )
        slot["losses"].append(float(row["loss"]))
        slot["delta_vs_178"].append(delta_ref)
        slot["delta_vs_341"].append(delta_ctl)
    split_summary_out = {
        split: {
            "mean_loss": float(np.mean(values["losses"])),
            "mean_delta_vs_178": float(np.mean(values["delta_vs_178"])),
            "mean_delta_vs_341": float(np.mean(values["delta_vs_341"])),
            "max_delta_vs_341": float(np.max(values["delta_vs_341"])),
        }
        for split, values in split_summary.items()
    }
    return {
        "fold_wins_vs_178": int(wins_ref),
        "fold_wins_vs_341": int(wins_ctl),
        "fold_count": int(len(candidate_rows)),
        "mean_fold_delta_vs_178": float(np.mean(fold_delta_ref)),
        "mean_fold_delta_vs_341": float(np.mean(fold_delta_ctl)),
        "worst_fold_delta_vs_341": float(np.max(fold_delta_ctl)),
        "split_summary": split_summary_out,
    }


def priority_read(candidate_id: int, record: dict[str, Any]) -> dict[str, Any]:
    layer = layer_for(candidate_id)
    if candidate_id == 221:
        queue = 1
        action = "rebuild_clean_root_feature_similarity"
        reason = "ID221 is weaker than later calibration artifacts but is an upstream prerequisite for ID226/ID238."
    elif candidate_id == 226:
        queue = 2
        action = "rebuild_primary_convex_backbone"
        reason = "ID226 carries the main reusable upstream signal and sits below ID227/228/237 and downstream A1/B1 artifacts."
    elif candidate_id in {227, 228, 237}:
        queue = 3
        action = "replay_only_after_clean_226"
        reason = "This is an ID226 calibration derivative; rebuilding it before a clean ID226 would preserve the same full-OOF failure mode."
    elif candidate_id in {238, 239, 241}:
        queue = 4
        action = "hold_until_clean_upstream_layer_exists"
        reason = "A1 artifacts have signal but depend on the upstream calibration chain; use them as replay targets, not roots."
    elif candidate_id in {254, 255, 256, 257, 258, 266}:
        queue = 5
        action = "dependency_audit_only_for_now"
        reason = (
            "B1 artifacts trace to high/non-nested dependencies and should wait until A1/upstream sources are rebuilt."
        )
    else:
        queue = 9
        action = "reference_only"
        reason = "Anchor/failure/control row."

    high_failure_projection = record["test_geometry"]["projection_failed_id287"] > 0.05
    high_control_distance = record["test_geometry"]["max_delta_vs_341"] > 0.16
    has_large_local_gain = record["full_oof"]["gain_vs_178"] > 0.03
    artifact_promotable = False
    risk_flags = []
    if high_failure_projection:
        risk_flags.append("failed_direction_exposure")
    if high_control_distance:
        risk_flags.append("large_test_geometry_shift")
    if layer in {"a1_replay_layer", "b1_replay_layer"} or candidate_id in {227, 228, 237, 287, 329}:
        risk_flags.append("non_nested_or_public_gap_lineage")
    if has_large_local_gain and risk_flags:
        risk_flags.append("local_gain_untrusted_until_rebuilt")

    return {
        "queue_rank": queue,
        "recommended_action": action,
        "reason": reason,
        "risk_flags": risk_flags,
        "artifact_promotable": artifact_promotable,
    }


def build_report() -> dict[str, Any]:
    labels = load_labels()
    y_true = labels[TARGETS].astype(float).to_numpy()
    log = load_log()
    oof_by_id = {candidate_id: load_oof(candidate_id, labels) for candidate_id in ALL_SOURCE_IDS}
    test_by_id = {candidate_id: load_test(candidate_id) for candidate_id in ALL_SOURCE_IDS}
    fold_rows = fold_loss_rows(y_true, oof_by_id)
    reference_oof = oof_by_id[REFERENCE_ID]
    control_oof = oof_by_id[CONTROL_ID]
    reference_test = test_by_id[REFERENCE_ID]
    control_test = test_by_id[CONTROL_ID]
    failed_direction = (test_by_id[287] - reference_test).reshape(-1)
    gap_direction = (test_by_id[329] - reference_test).reshape(-1)

    records = []
    reference_loss = average_loss(y_true, reference_oof)
    control_loss = average_loss(y_true, control_oof)
    for candidate_id in ALL_SOURCE_IDS:
        pred = oof_by_id[candidate_id]
        test = test_by_id[candidate_id]
        losses = target_losses(y_true, pred)
        full_loss = float(np.mean(list(losses.values())))
        target_delta_vs_178 = {
            target: float(losses[target] - target_losses(y_true, reference_oof)[target]) for target in TARGETS
        }
        target_delta_vs_341 = {
            target: float(losses[target] - target_losses(y_true, control_oof)[target]) for target in TARGETS
        }
        geom_vec = (test - reference_test).reshape(-1)
        record = {
            "id": candidate_id,
            "layer": layer_for(candidate_id),
            **log_record(log, candidate_id),
            "dependency_read": DEPENDENCY_READS.get(candidate_id, ""),
            "base_files": {
                "oof": str(find_base_file("oof", candidate_id).relative_to(ROOT)),
                "test": str(find_base_file("test", candidate_id).relative_to(ROOT)),
            },
            "full_oof": {
                "loss": full_loss,
                "gain_vs_178": float(reference_loss - full_loss),
                "gain_vs_341": float(control_loss - full_loss),
                "target_loss": losses,
                "target_delta_vs_178": target_delta_vs_178,
                "target_delta_vs_341": target_delta_vs_341,
            },
            "frozen_split": summarize_split_rows(
                fold_rows[candidate_id],
                fold_rows[REFERENCE_ID],
                fold_rows[CONTROL_ID],
            ),
            "test_geometry": {
                "mean_delta_vs_178": float(np.mean(np.abs(test - reference_test))),
                "max_delta_vs_178": float(np.max(np.abs(test - reference_test))),
                "mean_delta_vs_341": float(np.mean(np.abs(test - control_test))),
                "max_delta_vs_341": float(np.max(np.abs(test - control_test))),
                "projection_failed_id287": projection(geom_vec, failed_direction),
                "projection_gap_id329": projection(geom_vec, gap_direction),
            },
        }
        record["priority_read"] = priority_read(candidate_id, record)
        records.append(record)

    upstream_records = [row for row in records if row["layer"] == "upstream_model_layer"]
    first_rebuild_queue = sorted(
        [
            {
                "id": row["id"],
                "queue_rank": row["priority_read"]["queue_rank"],
                "recommended_action": row["priority_read"]["recommended_action"],
                "gain_vs_178": row["full_oof"]["gain_vs_178"],
                "fold_wins_vs_341": row["frozen_split"]["fold_wins_vs_341"],
                "projection_failed_id287": row["test_geometry"]["projection_failed_id287"],
                "reason": row["priority_read"]["reason"],
            }
            for row in upstream_records
        ],
        key=lambda row: (row["queue_rank"], -row["gain_vs_178"]),
    )
    all_artifact_promotable = any(row["priority_read"]["artifact_promotable"] for row in records)
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_upstream_source_priority_matrix",
        "purpose": (
            "Prioritize the upstream rebuild after ID341 by comparing existing source artifacts on full OOF loss, "
            "frozen split stability, dependency role, and known public-failure test geometry."
        ),
        "reference_ids": {
            "public_anchor": REFERENCE_ID,
            "current_control": CONTROL_ID,
            "public_failure_ids": PUBLIC_FAILURE_IDS,
        },
        "source_records": records,
        "first_rebuild_queue": first_rebuild_queue,
        "decision": {
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_heavy_rebuild_now": False,
            "any_existing_artifact_promotable": all_artifact_promotable,
            "first_rebuild_focus": ["ID221 clean feature-similarity root", "ID226 clean target-wise convex backbone"],
            "hold_until_clean_backbone": ["ID227", "ID228", "ID237", "ID238", "ID239", "ID241"],
            "next_action": (
                "Use this matrix to extend the Kaggle upstream runner with report-first ID221/ID226 clean rebuild design. "
                "Do not create ID342 or replay A1/B1 artifacts until a clean upstream backbone exists and passes transfer gates."
            ),
        },
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_submission_log_row_created": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Post-ID341 upstream source priority matrix",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It trains no model, creates no submission artifact, and makes no DACON submission.",
        "",
        "## Rebuild queue",
        "",
        "| rank | id | action | gain vs ID178 | wins vs ID341 | proj failed ID287 | reason |",
        "|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in report["first_rebuild_queue"]:
        lines.append(
            f"| {row['queue_rank']} | {row['id']} | `{row['recommended_action']}` | "
            f"{row['gain_vs_178']:+.6f} | {row['fold_wins_vs_341']} | "
            f"{row['projection_failed_id287']:+.4f} | {row['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Source matrix",
            "",
            "| id | layer | status | local log | full OOF | gain vs 178 | gain vs 341 | fold wins vs 341 | max Δ341 test | proj failed | action |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["source_records"]:
        local = "" if row.get("local_cv") is None else f"{row['local_cv']:.6f}"
        lines.append(
            f"| {row['id']} | `{row['layer']}` | `{row.get('status_clean', '')}` | {local} | "
            f"{row['full_oof']['loss']:.6f} | {row['full_oof']['gain_vs_178']:+.6f} | "
            f"{row['full_oof']['gain_vs_341']:+.6f} | {row['frozen_split']['fold_wins_vs_341']}/{row['frozen_split']['fold_count']} | "
            f"{row['test_geometry']['max_delta_vs_341']:.6f} | {row['test_geometry']['projection_failed_id287']:+.4f} | "
            f"`{row['priority_read']['recommended_action']}` |"
        )
    lines.extend(
        [
            "",
            "## Upstream target deltas versus ID178",
            "",
            "| id | Q1 | Q2 | Q3 | S1 | S2 | S3 | S4 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["source_records"]:
        if row["layer"] != "upstream_model_layer":
            continue
        deltas = row["full_oof"]["target_delta_vs_178"]
        lines.append(f"| {row['id']} | " + " | ".join(f"{deltas[target]:+.6f}" for target in TARGETS) + " |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Any existing artifact promotable: `{report['decision']['any_existing_artifact_promotable']}`",
            f"- First rebuild focus: `{', '.join(report['decision']['first_rebuild_focus'])}`",
            f"- Hold until clean backbone: `{', '.join(report['decision']['hold_until_clean_backbone'])}`",
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
        default=REPORTS_DIR / "post_id341_upstream_source_priority_matrix_20260618.json",
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
