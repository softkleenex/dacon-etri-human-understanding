#!/usr/bin/env python3
"""Post-ID341 Q3/S3 anchor failure profile.

This report-only diagnostic asks why Q3/S3 local OOF improvements are no longer
enough to justify submission artifacts. It compares existing OOF anchors and
known public outcomes for ID178, ID287, and ID329, while also profiling the
unsubmitted ID221/ID226 upstream-style candidates.

It trains no model and creates no new predictions, submission CSVs, base
prediction artifacts, notebook/history artifacts, submission-log rows, Kaggle
pushes/runs, or DACON submissions.
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
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

REPORTS_DIR = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw"
BASE_PREDICTION_DIR = ROOT / "artifacts" / "base_predictions"
SUBMISSION_LOG = ROOT / "submissions" / "submission_log.csv"
SPLIT_MANIFEST_PATH = REPORTS_DIR / "id340_validation_split_manifest_20260618.json"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_q3s3_anchor_failure_profile_20260619.json"

KST = timezone(timedelta(hours=9))
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
TARGET_FOCUS = ["Q3", "S3"]
CONTEXT_TARGETS = ["Q2", "Q3", "S3"]
ANCHOR_ID = 178
CANDIDATE_IDS = [178, 221, 226, 287, 329, 341]
KNOWN_PUBLIC_IDS = [178, 287, 329]
PRIMARY_GATE_SPLITS = [
    "subject_hole_5",
    "actual_test_band_holdout",
    "test_boundary_neighbor2_holdout",
    "test_boundary_neighbor5_holdout",
    "loso_10",
]
EPS = 1e-6
PUBLIC_TRAP_LOCAL_GAIN_CEILING = -0.005
PUBLIC_WORSE_THAN_ANCHOR_FLOOR = 0.0


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_train_sorted() -> pd.DataFrame:
    train = pd.read_csv(RAW_DIR / "ch2026_metrics_train.csv")
    train["date"] = pd.to_datetime(train["lifelog_date"]).dt.date
    return train.sort_values(["subject_id", "date"]).reset_index(drop=True)


def load_split_manifest(path: Path = SPLIT_MANIFEST_PATH) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for row in manifest["folds"]:
        splits.setdefault(row["split_name"], []).append(
            (np.asarray(row["train_idx"], dtype=int), np.asarray(row["valid_idx"], dtype=int))
        )
    return splits


def find_oof(candidate_id: int) -> Path:
    paths = sorted(BASE_PREDICTION_DIR.glob(f"base_oof_id_{candidate_id}_*.csv"))
    if not paths:
        raise FileNotFoundError(f"Missing base OOF ID{candidate_id}")
    return paths[0]


def load_oof_values(train: pd.DataFrame, candidate_id: int) -> pd.DataFrame:
    path = find_oof(candidate_id)
    pred = pd.read_csv(path)
    pred["date"] = pd.to_datetime(pred["date"]).dt.date
    merged = train[["subject_id", "date"]].merge(pred, on=["subject_id", "date"], how="left", validate="one_to_one")
    if merged[TARGETS].isna().any().any():
        raise ValueError(f"OOF ID{candidate_id} does not align with train rows.")
    return merged


def clip_pred(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(float), EPS, 1 - EPS)


def target_logloss(y: np.ndarray, pred: np.ndarray) -> float:
    return float(log_loss(y.astype(int), clip_pred(pred), labels=[0, 1]))


def split_loss(
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    target: str,
    pred_frame: pd.DataFrame,
) -> float:
    y = train[target].astype(int).to_numpy()
    pred = pred_frame[target].astype(float).to_numpy()
    total = 0
    weighted = 0.0
    for _fit_idx, valid_idx in folds:
        loss = target_logloss(y[valid_idx], pred[valid_idx])
        weighted += loss * len(valid_idx)
        total += len(valid_idx)
    return float(weighted / total)


def load_public_scores() -> dict[int, dict[str, Any]]:
    log = pd.read_csv(SUBMISSION_LOG)
    out: dict[int, dict[str, Any]] = {}
    for candidate_id in CANDIDATE_IDS:
        rows = log[log["id"].eq(candidate_id)]
        if rows.empty:
            continue
        row = rows.iloc[0].to_dict()
        public = row.get("public_lb_score_num")
        out[candidate_id] = {
            "submission_file": str(row.get("submission_file")),
            "local_cv": None if pd.isna(row.get("local_cv_score_num")) else float(row.get("local_cv_score_num")),
            "public": None if pd.isna(public) else float(public),
            "is_submitted": bool(row.get("is_submitted_bool")),
            "status_clean": str(row.get("status_clean")),
        }
    return out


def movement_summary(candidate: pd.DataFrame, anchor: pd.DataFrame, target: str) -> dict[str, Any]:
    diff = (candidate[target].astype(float) - anchor[target].astype(float)).to_numpy()
    abs_diff = np.abs(diff)
    return {
        "target": target,
        "mean_abs_move_vs_id178": float(abs_diff.mean()),
        "max_abs_move_vs_id178": float(abs_diff.max()),
        "mean_signed_move_vs_id178": float(diff.mean()),
        "share_abs_move_gt_0_02": float(np.mean(abs_diff > 0.02)),
        "share_abs_move_gt_0_05": float(np.mean(abs_diff > 0.05)),
        "share_abs_move_gt_0_10": float(np.mean(abs_diff > 0.10)),
    }


def target_profile_rows(
    train: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    predictions: dict[int, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    anchor = predictions[ANCHOR_ID]
    for candidate_id, pred_frame in predictions.items():
        for target in CONTEXT_TARGETS:
            y = train[target].astype(int).to_numpy()
            full_loss = target_logloss(y, pred_frame[target].astype(float).to_numpy())
            anchor_full_loss = target_logloss(y, anchor[target].astype(float).to_numpy())
            split_losses = {
                split_name: split_loss(train, folds, target, pred_frame)
                for split_name, folds in splits.items()
                if split_name in PRIMARY_GATE_SPLITS
            }
            anchor_split_losses = {
                split_name: split_loss(train, folds, target, anchor)
                for split_name, folds in splits.items()
                if split_name in PRIMARY_GATE_SPLITS
            }
            split_deltas = {
                split_name: float(split_losses[split_name] - anchor_split_losses[split_name])
                for split_name in split_losses
            }
            movement = movement_summary(pred_frame, anchor, target)
            rows.append(
                {
                    "candidate_id": int(candidate_id),
                    "target": target,
                    "full_loss": full_loss,
                    "id178_full_loss": anchor_full_loss,
                    "delta_vs_id178_full": float(full_loss - anchor_full_loss),
                    "primary_wins_vs_id178": int(sum(delta < 0 for delta in split_deltas.values())),
                    "primary_mean_delta_vs_id178": float(np.mean(list(split_deltas.values()))),
                    "primary_max_delta_vs_id178": float(np.max(list(split_deltas.values()))),
                    "split_deltas_vs_id178": split_deltas,
                    "prediction_mean": float(pred_frame[target].astype(float).mean()),
                    "label_mean": float(y.mean()),
                    **{k: v for k, v in movement.items() if k != "target"},
                }
            )
    return rows


def persistence_slices(train: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    ordered = train.sort_values(["subject_id", "date"]).copy()
    for target in TARGET_FOCUS:
        lag = ordered.groupby("subject_id")[target].shift(1)
        labels = []
        for current, prev in zip(ordered[target].astype(int), lag, strict=False):
            if pd.isna(prev):
                labels.append("first_subject_row")
            elif int(prev) == int(current):
                labels.append("persistent")
            elif int(prev) == 0 and int(current) == 1:
                labels.append("onset_0_to_1")
            else:
                labels.append("offset_1_to_0")
        frame = pd.DataFrame({"slice": labels, "y": ordered[target].astype(int).to_numpy()})
        rows = []
        for name, group in frame.groupby("slice", sort=True):
            rows.append({"slice": str(name), "rows": int(len(group)), "positive_rate": float(group["y"].mean())})
        out[target] = rows
    return out


def slice_loss_rows(train: pd.DataFrame, predictions: dict[int, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered = train.sort_values(["subject_id", "date"]).reset_index(drop=True)
    slice_labels: dict[str, list[str]] = {}
    for target in TARGET_FOCUS:
        lag = ordered.groupby("subject_id")[target].shift(1)
        labels = []
        for current, prev in zip(ordered[target].astype(int), lag, strict=False):
            if pd.isna(prev):
                labels.append("first_subject_row")
            elif int(prev) == int(current):
                labels.append("persistent")
            elif int(prev) == 0 and int(current) == 1:
                labels.append("onset_0_to_1")
            else:
                labels.append("offset_1_to_0")
        slice_labels[target] = labels

    anchor = predictions[ANCHOR_ID]
    for target in TARGET_FOCUS:
        y = ordered[target].astype(int).to_numpy()
        labels = np.asarray(slice_labels[target], dtype=object)
        for candidate_id, pred in predictions.items():
            pred_values = pred[target].astype(float).to_numpy()
            anchor_values = anchor[target].astype(float).to_numpy()
            for slice_name in sorted(set(labels)):
                idx = labels == slice_name
                if int(idx.sum()) == 0:
                    continue
                # log_loss needs both labels; if a slice is single-class, report BCE directly.
                y_slice = y[idx]
                p_slice = clip_pred(pred_values[idx])
                a_slice = clip_pred(anchor_values[idx])
                loss = float(-np.mean(y_slice * np.log(p_slice) + (1 - y_slice) * np.log(1 - p_slice)))
                anchor_loss = float(-np.mean(y_slice * np.log(a_slice) + (1 - y_slice) * np.log(1 - a_slice)))
                rows.append(
                    {
                        "target": target,
                        "candidate_id": int(candidate_id),
                        "slice": str(slice_name),
                        "rows": int(idx.sum()),
                        "loss": loss,
                        "delta_vs_id178": float(loss - anchor_loss),
                        "mean_abs_move_vs_id178": float(np.abs(p_slice - a_slice).mean()),
                    }
                )
    return rows


def known_public_traps(
    profile_rows: list[dict[str, Any]], public_scores: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    id178_public = public_scores.get(ANCHOR_ID, {}).get("public")
    if id178_public is None:
        return []
    by_candidate: dict[int, dict[str, Any]] = {}
    for row in profile_rows:
        if row["target"] not in TARGET_FOCUS:
            continue
        candidate_id = int(row["candidate_id"])
        bucket = by_candidate.setdefault(candidate_id, {"q3s3_deltas": [], "q3s3_moves": [], "q3s3_wins": []})
        bucket["q3s3_deltas"].append(float(row["delta_vs_id178_full"]))
        bucket["q3s3_moves"].append(float(row["mean_abs_move_vs_id178"]))
        bucket["q3s3_wins"].append(int(row["primary_wins_vs_id178"]))

    traps = []
    for candidate_id in KNOWN_PUBLIC_IDS:
        if candidate_id == ANCHOR_ID or candidate_id not in by_candidate:
            continue
        public = public_scores.get(candidate_id, {}).get("public")
        if public is None:
            continue
        q3s3_mean_delta = float(np.mean(by_candidate[candidate_id]["q3s3_deltas"]))
        public_delta = float(public - id178_public)
        is_trap = q3s3_mean_delta <= PUBLIC_TRAP_LOCAL_GAIN_CEILING and public_delta > PUBLIC_WORSE_THAN_ANCHOR_FLOOR
        traps.append(
            {
                "candidate_id": int(candidate_id),
                "public": public,
                "public_delta_vs_id178": public_delta,
                "q3s3_mean_oof_delta_vs_id178": q3s3_mean_delta,
                "q3s3_mean_abs_move_vs_id178": float(np.mean(by_candidate[candidate_id]["q3s3_moves"])),
                "q3s3_min_primary_wins_vs_id178": int(min(by_candidate[candidate_id]["q3s3_wins"])),
                "is_known_q3s3_oof_public_trap": bool(is_trap),
            }
        )
    return traps


def build_report() -> dict[str, Any]:
    train = load_train_sorted()
    splits = load_split_manifest()
    predictions = {candidate_id: load_oof_values(train, candidate_id) for candidate_id in CANDIDATE_IDS}
    public_scores = load_public_scores()
    profile_rows = target_profile_rows(train, splits, predictions)
    slice_rows = slice_loss_rows(train, predictions)
    traps = known_public_traps(profile_rows, public_scores)
    public_trap_ids = [row["candidate_id"] for row in traps if row["is_known_q3s3_oof_public_trap"]]
    q3s3_rows = [row for row in profile_rows if row["target"] in TARGET_FOCUS]
    upstream_rows = [row for row in q3s3_rows if row["candidate_id"] in {221, 226}]
    unsubmitted_upstream_has_q3s3_oof_signal = any(row["delta_vs_id178_full"] < -0.005 for row in upstream_rows)

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_q3s3_anchor_failure_profile",
        "purpose": (
            "Profile existing OOF anchors and known public outcomes to show why Q3/S3 local OOF gains alone "
            "cannot justify another submission artifact."
        ),
        "target_focus": TARGET_FOCUS,
        "context_targets": CONTEXT_TARGETS,
        "candidate_ids": CANDIDATE_IDS,
        "known_public_ids": KNOWN_PUBLIC_IDS,
        "inputs": {
            "split_manifest": rel(SPLIT_MANIFEST_PATH),
            "submission_log": rel(SUBMISSION_LOG),
            "oof_files": {str(candidate_id): rel(find_oof(candidate_id)) for candidate_id in CANDIDATE_IDS},
        },
        "public_scores": public_scores,
        "target_profile_rows": profile_rows,
        "persistence_slices": persistence_slices(train),
        "slice_loss_rows": slice_rows,
        "known_public_traps": traps,
        "decision": {
            "anchor_failure_profile_ready": True,
            "q3s3_known_oof_public_trap_ids": public_trap_ids,
            "q3s3_oof_gain_alone_is_submission_evidence": False,
            "unsubmitted_upstream_has_q3s3_oof_signal": bool(unsubmitted_upstream_has_q3s3_oof_signal),
            "unsubmitted_upstream_requires_rebuild_not_direct_submit": True,
            "public_058_evidence": False,
            "manual_probe_if_spending_submission": None,
            "manual_probe_family_status": "consumed_after_id341_id340_id337_public_micro_probe_slate",
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Treat Q3/S3 OOF gains as rebuild-prior evidence only. Direct local artifact promotion remains blocked; "
                "the score-relevant route is still the explicit-approved clean ID221→ID226 no-submit upstream rebuild."
            ),
        },
        "no_model_trained": True,
        "no_new_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    rows = report["target_profile_rows"]
    top_rows = [
        row for row in rows if row["target"] in TARGET_FOCUS and row["candidate_id"] in [178, 221, 226, 287, 329, 341]
    ]
    lines = [
        "# Post-ID341 Q3/S3 anchor failure profile",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It trains no model and creates no new predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.",
        "",
        "## Known public traps",
        "",
        "| candidate | public Δ178 | mean Q3/S3 OOF Δ178 | mean Q3/S3 move | trap |",
        "|---:|---:|---:|---:|---|",
    ]
    if not report["known_public_traps"]:
        lines.append("| _(none)_ |  |  |  |  |")
    for row in report["known_public_traps"]:
        lines.append(
            f"| {row['candidate_id']} | {row['public_delta_vs_id178']:+.6f} | "
            f"{row['q3s3_mean_oof_delta_vs_id178']:+.6f} | "
            f"{row['q3s3_mean_abs_move_vs_id178']:.6f} | `{row['is_known_q3s3_oof_public_trap']}` |"
        )
    lines.extend(
        [
            "",
            "## Q3/S3 anchor profile",
            "",
            "| candidate | target | OOF Δ178 | primary wins | mean move | max move | pred mean |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top_rows:
        lines.append(
            f"| {row['candidate_id']} | `{row['target']}` | {row['delta_vs_id178_full']:+.6f} | "
            f"{row['primary_wins_vs_id178']} | {row['mean_abs_move_vs_id178']:.6f} | "
            f"{row['max_abs_move_vs_id178']:.6f} | {row['prediction_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Persistence slices",
            "",
        ]
    )
    for target, slice_rows in report["persistence_slices"].items():
        lines.extend([f"### `{target}`", "", "| slice | rows | positive rate |", "|---|---:|---:|"])
        for row in slice_rows:
            lines.append(f"| `{row['slice']}` | {row['rows']} | {row['positive_rate']:.6f} |")
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            f"- Anchor failure profile ready: `{decision['anchor_failure_profile_ready']}`",
            f"- Q3/S3 known OOF-public trap IDs: `{decision['q3s3_known_oof_public_trap_ids']}`",
            f"- Q3/S3 OOF gain alone is submission evidence: `{decision['q3s3_oof_gain_alone_is_submission_evidence']}`",
            f"- Unsubmitted upstream has Q3/S3 OOF signal: `{decision['unsubmitted_upstream_has_q3s3_oof_signal']}`",
            f"- Unsubmitted upstream requires rebuild, not direct submit: `{decision['unsubmitted_upstream_requires_rebuild_not_direct_submit']}`",
            f"- Public `0.58` evidence: `{decision['public_058_evidence']}`",
            f"- Manual probe if spending a DACON submission: `{decision['manual_probe_if_spending_submission']}`",
            f"- Manual probe family status: `{decision['manual_probe_family_status']}`",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
            f"- Submit now: `{decision['submit_now']}`",
            f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
            f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
            f"- Next action: {decision['next_action']}",
            "",
        ]
    )
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
