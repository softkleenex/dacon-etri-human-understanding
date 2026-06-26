#!/usr/bin/env python3
"""Stable-lens triage for existing post-ID342 OOF candidates.

This report-only diagnostic applies the public-alignment lens policy to the
existing base OOF catalog. It asks whether any already-materialized candidate
beats ID342 without moving in the known failed-frontier directions represented
by ID287 and ID329.

It creates no submission CSV, base prediction artifact, notebook-history
artifact, submission-log row, Kaggle push/run, or DACON submission.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id342_public_alignment_validation_lens import (  # noqa: E402
    CURRENT_ANCHOR_ID,
    FAILED_FRONTIER_IDS,
    PUBLIC_055,
    RECENT_SLATE_IDS,
    SCORED_IDS,
    TARGETS,
    full_target_losses,
    load_oof_values,
    load_public_scores,
    load_train_sorted,
    mean_loss,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw"
BASE_PREDICTION_DIR = ROOT / "artifacts" / "base_predictions"
SUBMISSION_LOG_PATH = ROOT / "submissions" / "submission_log.csv"
VALIDATION_LENS_REPORT_PATH = REPORTS_DIR / "post_id342_public_alignment_validation_lens_20260624.json"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_stable_lens_existing_candidate_triage_20260624.json"

ID342_PUBLIC = 0.5960832519
OOF_ID_PATTERN = re.compile(r"base_oof_id_(\d+)_.*\.csv$")
EPS = 1e-12

MIN_MATERIAL_LOCAL_GAIN = -0.0010
MAX_FRONTIER_LOW_LOSS = 0.5200
MAX_OOF_MEAN_MOVEMENT = 0.0600
MAX_OOF_MAX_MOVEMENT = 0.3000
MAX_TEST_MEAN_MOVEMENT = 0.0350
MAX_TEST_MAX_MOVEMENT = 0.2500
MAX_FAILED_DIRECTION_COSINE = 0.25
MAX_FAILED_DIRECTION_PROJECTION = 0.08
MAX_TARGET_MEAN_L1_INCREASE = 0.0020


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def candidate_oof_catalog() -> dict[int, Path]:
    catalog: dict[int, Path] = {}
    for path in BASE_PREDICTION_DIR.glob("base_oof_id_*_*.csv"):
        match = OOF_ID_PATTERN.match(path.name)
        if match:
            candidate_id = int(match.group(1))
            catalog.setdefault(candidate_id, path)
    return dict(sorted(catalog.items()))


def load_oof_values_from_path(train: pd.DataFrame, candidate_id: int, path: Path) -> np.ndarray:
    pred = pd.read_csv(path)
    pred["date"] = pd.to_datetime(pred["date"]).dt.date
    key = train[["subject_id", "date"]].copy()
    merged = key.merge(pred, on=["subject_id", "date"], how="left", validate="one_to_one")
    if merged[TARGETS].isna().any().any():
        missing = merged[TARGETS].isna().any(axis=1).sum()
        raise ValueError(f"OOF ID{candidate_id} does not align with sorted train rows; missing rows={missing}")
    return np.clip(merged[TARGETS].astype(float).to_numpy(), 1e-6, 1.0 - 1e-6)


def find_test(candidate_id: int) -> Path | None:
    paths = sorted(BASE_PREDICTION_DIR.glob(f"base_test_id_{candidate_id}_*.csv"))
    return paths[0] if paths else None


def load_test_keys_sorted() -> pd.DataFrame:
    test = pd.read_csv(RAW_DIR / "ch2026_submission_sample.csv")
    test["date"] = pd.to_datetime(test["lifelog_date"]).dt.date
    return test.sort_values(["subject_id", "date"]).reset_index(drop=True)


def load_test_values(test: pd.DataFrame, candidate_id: int) -> tuple[np.ndarray | None, Path | None]:
    path = find_test(candidate_id)
    if path is None:
        return None, None
    pred = pd.read_csv(path)
    pred["date"] = pd.to_datetime(pred["date"]).dt.date
    key = test[["subject_id", "date"]].copy()
    merged = key.merge(pred, on=["subject_id", "date"], how="left", validate="one_to_one")
    if merged[TARGETS].isna().any().any():
        return None, path
    return merged[TARGETS].astype(float).to_numpy(), path


def load_submission_log_status() -> dict[int, dict[str, Any]]:
    if not SUBMISSION_LOG_PATH.exists():
        return {}
    log = pd.read_csv(SUBMISSION_LOG_PATH)
    if "id" not in log.columns:
        return {}
    by_id: dict[int, dict[str, Any]] = {}
    for _, row in log.dropna(subset=["id"]).iterrows():
        candidate_id = int(row["id"])
        by_id[candidate_id] = {
            "submission_file": row.get("submission_file"),
            "model_description": row.get("model_description"),
            "notes": row.get("notes"),
            "notebook_file": row.get("notebook_file"),
            "status_clean": row.get("status_clean"),
            "artifact_role": row.get("artifact_role"),
            "is_submitted_bool": bool(row.get("is_submitted_bool")) if not pd.isna(row.get("is_submitted_bool")) else None,
            "public_lb_score_num": (
                float(row["public_lb_score_num"]) if not pd.isna(row.get("public_lb_score_num")) else None
            ),
            "local_cv_score_num": (
                float(row["local_cv_score_num"]) if not pd.isna(row.get("local_cv_score_num")) else None
            ),
        }
    return by_id


def safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.ravel()
    right_flat = right.ravel()
    denom = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    if denom <= EPS:
        return 0.0
    return float(np.dot(left_flat, right_flat) / denom)


def projection_ratio(delta: np.ndarray, direction: np.ndarray) -> float:
    direction_flat = direction.ravel()
    denom = float(np.dot(direction_flat, direction_flat))
    if denom <= EPS:
        return 0.0
    return float(np.dot(delta.ravel(), direction_flat) / denom)


def movement_stats(values: np.ndarray, anchor_values: np.ndarray) -> dict[str, Any]:
    delta = values - anchor_values
    return {
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "max_abs_delta": float(np.max(np.abs(delta))),
        "signed_mean_delta": float(np.mean(delta)),
        "per_target_mean_abs_delta": {
            target: float(np.mean(np.abs(delta[:, idx])))
            for idx, target in enumerate(TARGETS)
        },
        "per_target_signed_mean_delta": {
            target: float(np.mean(delta[:, idx]))
            for idx, target in enumerate(TARGETS)
        },
    }


def target_mean_l1_to_train(values: np.ndarray, train: pd.DataFrame) -> float:
    train_means = train[TARGETS].astype(float).mean().to_numpy()
    pred_means = values.mean(axis=0)
    return float(np.mean(np.abs(pred_means - train_means)))


def failed_direction_stats(
    candidate_values: np.ndarray,
    anchor_values: np.ndarray,
    failed_values: dict[int, np.ndarray],
) -> dict[str, Any]:
    candidate_delta = candidate_values - anchor_values
    rows: dict[str, dict[str, float]] = {}
    for failed_id, values in failed_values.items():
        failed_delta = values - anchor_values
        rows[str(failed_id)] = {
            "cosine": safe_cosine(candidate_delta, failed_delta),
            "projection_ratio": projection_ratio(candidate_delta, failed_delta),
            "mean_abs_distance_to_failed": float(np.mean(np.abs(candidate_values - values))),
            "anchor_mean_abs_distance_to_failed": float(np.mean(np.abs(anchor_values - values))),
        }
    return {
        "by_failed_id": rows,
        "max_positive_cosine": float(max(max(row["cosine"], 0.0) for row in rows.values())),
        "max_positive_projection_ratio": float(
            max(max(row["projection_ratio"], 0.0) for row in rows.values())
        ),
        "closer_to_any_failed_than_id342": bool(
            any(row["mean_abs_distance_to_failed"] < row["anchor_mean_abs_distance_to_failed"] for row in rows.values())
        ),
    }


def gate_checks(
    candidate_id: int,
    full_loss: float,
    delta_vs_id342: float,
    public_score: float | None,
    oof_move: dict[str, Any],
    test_move: dict[str, Any] | None,
    failed_stats: dict[str, Any],
    target_mean_l1_delta: float,
    lineage_ok: bool,
) -> dict[str, bool]:
    return {
        "not_already_public_scored": public_score is None,
        "not_known_failed_frontier": candidate_id not in FAILED_FRONTIER_IDS,
        "lineage_not_public_guided_or_unnested": lineage_ok,
        "material_local_gain_vs_id342": delta_vs_id342 <= MIN_MATERIAL_LOCAL_GAIN,
        "not_frontier_low_loss": full_loss >= MAX_FRONTIER_LOW_LOSS,
        "oof_movement_within_anchor_band": (
            oof_move["mean_abs_delta"] <= MAX_OOF_MEAN_MOVEMENT
            and oof_move["max_abs_delta"] <= MAX_OOF_MAX_MOVEMENT
        ),
        "test_artifact_present": test_move is not None,
        "test_movement_within_anchor_band": (
            test_move is not None
            and test_move["mean_abs_delta"] <= MAX_TEST_MEAN_MOVEMENT
            and test_move["max_abs_delta"] <= MAX_TEST_MAX_MOVEMENT
        ),
        "failed_direction_cosine_low": failed_stats["max_positive_cosine"] <= MAX_FAILED_DIRECTION_COSINE,
        "failed_direction_projection_low": (
            failed_stats["max_positive_projection_ratio"] <= MAX_FAILED_DIRECTION_PROJECTION
        ),
        "not_closer_to_failed_frontier_than_id342": not failed_stats["closer_to_any_failed_than_id342"],
        "target_mean_l1_not_worse_than_id342": target_mean_l1_delta <= MAX_TARGET_MEAN_L1_INCREASE,
    }


def risk_reasons(checks: dict[str, bool], full_loss: float, public_score: float | None) -> list[str]:
    reasons = [name for name, ok in checks.items() if not ok]
    if public_score is not None:
        reasons.append("already_has_public_feedback")
    if full_loss < MAX_FRONTIER_LOW_LOSS:
        reasons.append("frontier_low_loss_band")
    return sorted(set(reasons))


def lineage_is_strict_enough(status: dict[str, Any]) -> bool:
    text = " ".join(
        str(status.get(key) or "")
        for key in ["submission_file", "model_description", "notes", "notebook_file"]
    ).lower()
    blocked_markers = [
        "public-structure",
        "public structure",
        "not strict",
        "not fold-nested",
        "not strict/nested",
        "oracle",
        "extreme probe",
    ]
    return not any(marker in text for marker in blocked_markers)


def candidate_row(
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidate_id: int,
    oof_path: Path,
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray | None,
    failed_oof: dict[int, np.ndarray],
    id342_loss: float,
    id342_target_mean_l1: float,
    public_scores: dict[int, dict[str, Any]],
    log_status: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    values = load_oof_values_from_path(train, candidate_id, oof_path)
    target_losses = full_target_losses(train, values)
    full_loss = mean_loss(target_losses)
    public_score = (
        float(public_scores[candidate_id]["public_score"])
        if candidate_id in public_scores
        else log_status.get(candidate_id, {}).get("public_lb_score_num")
    )
    oof_move = movement_stats(values, anchor_oof)
    failed_stats = failed_direction_stats(values, anchor_oof, failed_oof)
    test_values, test_path = load_test_values(test, candidate_id)
    test_move = (
        movement_stats(test_values, anchor_test)
        if test_values is not None and anchor_test is not None
        else None
    )
    target_l1 = target_mean_l1_to_train(values, train)
    delta_vs_id342 = float(full_loss - id342_loss)
    status = log_status.get(candidate_id, {})
    checks = gate_checks(
        candidate_id=candidate_id,
        full_loss=full_loss,
        delta_vs_id342=delta_vs_id342,
        public_score=public_score,
        oof_move=oof_move,
        test_move=test_move,
        failed_stats=failed_stats,
        target_mean_l1_delta=float(target_l1 - id342_target_mean_l1),
        lineage_ok=lineage_is_strict_enough(status),
    )
    stable_lens_review_pass = bool(all(checks.values()))
    public_proxy = ID342_PUBLIC + delta_vs_id342
    return {
        "id": candidate_id,
        "oof_path": rel(oof_path),
        "test_path": rel(test_path) if test_path is not None else None,
        "full_oof_loss": full_loss,
        "local_delta_vs_id342": delta_vs_id342,
        "target_losses": target_losses,
        "public_score": public_score,
        "public_proxy_from_id342_delta": public_proxy,
        "gap_to_public_055_proxy": float(public_proxy - PUBLIC_055),
        "submission_log_status": status,
        "oof_movement_vs_id342": oof_move,
        "test_movement_vs_id342": test_move,
        "failed_frontier_direction": failed_stats,
        "target_mean_l1_to_train": target_l1,
        "target_mean_l1_delta_vs_id342": float(target_l1 - id342_target_mean_l1),
        "checks": checks,
        "risk_reasons": risk_reasons(checks, full_loss, public_score),
        "stable_lens_review_pass": stable_lens_review_pass,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    train = load_train_sorted()
    test = load_test_keys_sorted()
    public_scores = load_public_scores()
    log_status = load_submission_log_status()
    anchor_oof, _anchor_oof_path = load_oof_values(train, CURRENT_ANCHOR_ID)
    anchor_test, _anchor_test_path = load_test_values(test, CURRENT_ANCHOR_ID)
    failed_oof = {
        failed_id: load_oof_values(train, failed_id)[0]
        for failed_id in FAILED_FRONTIER_IDS
    }
    id342_target_losses = full_target_losses(train, anchor_oof)
    id342_loss = mean_loss(id342_target_losses)
    id342_target_l1 = target_mean_l1_to_train(anchor_oof, train)
    catalog = candidate_oof_catalog()
    rows = [
        candidate_row(
            train=train,
            test=test,
            candidate_id=candidate_id,
            oof_path=oof_path,
            anchor_oof=anchor_oof,
            anchor_test=anchor_test,
            failed_oof=failed_oof,
            id342_loss=id342_loss,
            id342_target_mean_l1=id342_target_l1,
            public_scores=public_scores,
            log_status=log_status,
        )
        for candidate_id, oof_path in catalog.items()
    ]
    rows.sort(key=lambda row: row["full_oof_loss"])
    passing_rows = [row for row in rows if row["stable_lens_review_pass"]]
    lineage_near_miss_rows = [
        row
        for row in rows
        if not row["checks"]["lineage_not_public_guided_or_unnested"]
        and all(
            ok
            for check_name, ok in row["checks"].items()
            if check_name != "lineage_not_public_guided_or_unnested"
        )
    ]
    unscored_rows = [row for row in rows if row["public_score"] is None]
    unscored_rows.sort(key=lambda row: (row["stable_lens_review_pass"] is False, row["full_oof_loss"]))
    public_scored_reference = [row for row in rows if row["id"] in SCORED_IDS]
    public_scored_reference.sort(key=lambda row: row["public_score"] if row["public_score"] is not None else 999.0)
    review_rows = [
        row
        for row in rows
        if row["id"] not in SCORED_IDS and (row["local_delta_vs_id342"] < 0.0 or row["id"] in RECENT_SLATE_IDS)
    ][:60]
    decision = {
        "scanned_oof_candidates": int(len(rows)),
        "unscored_candidates": int(len([row for row in rows if row["public_score"] is None])),
        "stable_lens_review_pass_count": int(len(passing_rows)),
        "best_unscored_candidate_id": int(unscored_rows[0]["id"]) if unscored_rows else None,
        "best_unscored_full_oof": unscored_rows[0]["full_oof_loss"] if unscored_rows else None,
        "best_unscored_delta_vs_id342": unscored_rows[0]["local_delta_vs_id342"] if unscored_rows else None,
        "best_passing_candidate_id": int(passing_rows[0]["id"]) if passing_rows else None,
        "lineage_near_miss_count": int(len(lineage_near_miss_rows)),
        "lineage_near_miss_ids": [int(row["id"]) for row in lineage_near_miss_rows[:10]],
        "create_submission_artifact": False,
        "submit_now": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "reason": (
            "At least one existing unscored row passed the mechanical stable-lens review, but this triage still "
            "creates no artifact; require a separate decision note and exact approval."
            if passing_rows
            else "No existing unscored candidate passed the stable-lens review. Strong local rows remain dominated by "
            "failed-frontier, public-scored, missing-test, or movement-risk checks."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_stable_lens_existing_candidate_triage",
        "purpose": "Apply the ID342 public-alignment lens policy to existing OOF candidates.",
        "validation_lens_report": rel(VALIDATION_LENS_REPORT_PATH),
        "thresholds": {
            "min_material_local_gain": MIN_MATERIAL_LOCAL_GAIN,
            "max_frontier_low_loss": MAX_FRONTIER_LOW_LOSS,
            "max_oof_mean_movement": MAX_OOF_MEAN_MOVEMENT,
            "max_oof_max_movement": MAX_OOF_MAX_MOVEMENT,
            "max_test_mean_movement": MAX_TEST_MEAN_MOVEMENT,
            "max_test_max_movement": MAX_TEST_MAX_MOVEMENT,
            "max_failed_direction_cosine": MAX_FAILED_DIRECTION_COSINE,
            "max_failed_direction_projection": MAX_FAILED_DIRECTION_PROJECTION,
            "max_target_mean_l1_increase": MAX_TARGET_MEAN_L1_INCREASE,
        },
        "anchor": {
            "id": CURRENT_ANCHOR_ID,
            "full_oof_loss": id342_loss,
            "public_score": ID342_PUBLIC,
            "gap_to_public_055": float(ID342_PUBLIC - PUBLIC_055),
            "target_losses": id342_target_losses,
            "target_mean_l1_to_train": id342_target_l1,
        },
        "failed_frontier_ids": FAILED_FRONTIER_IDS,
        "public_scored_reference_rows": public_scored_reference,
        "passing_rows": passing_rows,
        "lineage_near_miss_rows": lineage_near_miss_rows,
        "review_rows": review_rows,
        "top_local_rows": rows[:40],
        "decision": decision,
        "no_prediction_file_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def fmt_float(value: float | None, digits: int = 6) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# Post-ID342 stable-lens existing-candidate triage",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission artifacts, base predictions, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- Scanned OOF candidates: `{decision['scanned_oof_candidates']}`",
        f"- Unscored candidates: `{decision['unscored_candidates']}`",
        f"- Stable-lens review pass count: `{decision['stable_lens_review_pass_count']}`",
        f"- Lineage near-miss count: `{decision['lineage_near_miss_count']}`",
        f"- Lineage near-miss IDs: `{decision['lineage_near_miss_ids']}`",
        f"- Best unscored candidate: `{decision['best_unscored_candidate_id']}`",
        f"- Best unscored full OOF / delta vs ID342: `{fmt_float(decision['best_unscored_full_oof'])}` / `{fmt_float(decision['best_unscored_delta_vs_id342'], 9)}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- Reason: {decision['reason']}",
        "",
        "## Public-Scored References",
        "",
        "| id | full OOF | public | delta 342 | failed cos | test mean delta | pass | risks |",
        "|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in report["public_scored_reference_rows"]:
        test_move = row["test_movement_vs_id342"]
        lines.append(
            "| {id} | {loss:.6f} | {public} | {delta:+.6f} | {cos:.3f} | {test_mean} | `{passed}` | `{risks}` |".format(
                id=row["id"],
                loss=row["full_oof_loss"],
                public=fmt_float(row["public_score"], 9),
                delta=row["local_delta_vs_id342"],
                cos=row["failed_frontier_direction"]["max_positive_cosine"],
                test_mean=fmt_float(test_move["mean_abs_delta"] if test_move else None),
                passed=row["stable_lens_review_pass"],
                risks=",".join(row["risk_reasons"][:6]),
            )
        )
    lines.extend(
        [
            "",
            "## Passing Rows",
            "",
            "| rank | id | full OOF | delta 342 | proxy | gap 0.55 | failed cos | test mean delta | risks |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for rank, row in enumerate(report["passing_rows"], start=1):
        test_move = row["test_movement_vs_id342"]
        lines.append(
            "| {rank} | {id} | {loss:.6f} | {delta:+.6f} | {proxy:.6f} | {gap:+.6f} | {cos:.3f} | {test_mean} | `{risks}` |".format(
                rank=rank,
                id=row["id"],
                loss=row["full_oof_loss"],
                delta=row["local_delta_vs_id342"],
                proxy=row["public_proxy_from_id342_delta"],
                gap=row["gap_to_public_055_proxy"],
                cos=row["failed_frontier_direction"]["max_positive_cosine"],
                test_mean=fmt_float(test_move["mean_abs_delta"] if test_move else None),
                risks=",".join(row["risk_reasons"][:6]),
            )
        )
    lines.extend(
        [
            "",
            "## Lineage Near Misses",
            "",
            "| rank | id | full OOF | delta 342 | proxy | failed cos | test mean delta | blocking note |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for rank, row in enumerate(report["lineage_near_miss_rows"][:10], start=1):
        test_move = row["test_movement_vs_id342"]
        status = row["submission_log_status"]
        note = status.get("notes") or status.get("model_description") or ""
        lines.append(
            "| {rank} | {id} | {loss:.6f} | {delta:+.6f} | {proxy:.6f} | {cos:.3f} | {test_mean} | {note} |".format(
                rank=rank,
                id=row["id"],
                loss=row["full_oof_loss"],
                delta=row["local_delta_vs_id342"],
                proxy=row["public_proxy_from_id342_delta"],
                cos=row["failed_frontier_direction"]["max_positive_cosine"],
                test_mean=fmt_float(test_move["mean_abs_delta"] if test_move else None),
                note=str(note).replace("|", "/")[:220],
            )
        )
    lines.extend(
        [
            "",
            "## Top Local Rows",
            "",
            "| rank | id | full OOF | delta 342 | public | failed cos | failed proj | test mean delta | pass | risks |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
        ]
    )
    for rank, row in enumerate(report["top_local_rows"][:25], start=1):
        test_move = row["test_movement_vs_id342"]
        lines.append(
            "| {rank} | {id} | {loss:.6f} | {delta:+.6f} | {public} | {cos:.3f} | {proj:.3f} | {test_mean} | `{passed}` | `{risks}` |".format(
                rank=rank,
                id=row["id"],
                loss=row["full_oof_loss"],
                delta=row["local_delta_vs_id342"],
                public=fmt_float(row["public_score"], 9),
                cos=row["failed_frontier_direction"]["max_positive_cosine"],
                proj=row["failed_frontier_direction"]["max_positive_projection_ratio"],
                test_mean=fmt_float(test_move["mean_abs_delta"] if test_move else None),
                passed=row["stable_lens_review_pass"],
                risks=",".join(row["risk_reasons"][:6]),
            )
        )
    lines.extend(
        [
            "",
            "## Review Rows",
            "",
            "| rank | id | full OOF | delta 342 | proxy | gap 0.55 | failed cos | test mean delta | pass | risks |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
        ]
    )
    for rank, row in enumerate(report["review_rows"][:25], start=1):
        test_move = row["test_movement_vs_id342"]
        lines.append(
            "| {rank} | {id} | {loss:.6f} | {delta:+.6f} | {proxy:.6f} | {gap:+.6f} | {cos:.3f} | {test_mean} | `{passed}` | `{risks}` |".format(
                rank=rank,
                id=row["id"],
                loss=row["full_oof_loss"],
                delta=row["local_delta_vs_id342"],
                proxy=row["public_proxy_from_id342_delta"],
                gap=row["gap_to_public_055_proxy"],
                cos=row["failed_frontier_direction"]["max_positive_cosine"],
                test_mean=fmt_float(test_move["mean_abs_delta"] if test_move else None),
                passed=row["stable_lens_review_pass"],
                risks=",".join(row["risk_reasons"][:6]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path = args.output.with_suffix(".md")
    md_path.write_text(build_markdown(report), encoding="utf-8")
    print(f"Wrote {rel(args.output)} and {rel(md_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
