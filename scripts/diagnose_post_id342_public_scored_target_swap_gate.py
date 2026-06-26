#!/usr/bin/env python3
"""ID342-aware public-scored Q3/S3 target-swap gate.

This report-only diagnostic checks whether Q3/S3 predictions from public-scored
anchors can safely improve ID342 under the stable-lens policy. It is intentionally
conservative: donor rows from known failed-frontier IDs are reported as traps,
not as promotable evidence.

It creates no prediction CSV, base prediction artifact, notebook-history
artifact, submission-log row, Kaggle push/run, or DACON submission.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9))
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"
BASE_PREDICTION_DIR = ROOT / "artifacts" / "base_predictions"
SPLIT_MANIFEST_PATH = REPORTS_DIR / "id340_validation_split_manifest_20260618.json"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_public_scored_target_swap_gate_20260624.json"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
FOCUS_TARGETS = ["Q3", "S3"]
ANCHOR_ID = 342
STABLE_DONOR_IDS = [178, 337, 338, 339, 340]
FAILED_FRONTIER_IDS = [287, 329]
DONOR_IDS = [*STABLE_DONOR_IDS, *FAILED_FRONTIER_IDS]
PRIMARY_SPLITS = [
    "subject_hole_5",
    "actual_test_band_holdout",
    "test_boundary_neighbor2_holdout",
    "test_boundary_neighbor5_holdout",
    "loso_10",
]
WEIGHTS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]

EPS = 1e-6
ID342_PUBLIC = 0.5960832519
PUBLIC_055 = 0.55
TARGET_WIN_FLOOR = 4
TARGET_MEAN_DELTA_CEILING = -0.00005
TARGET_MAX_DELTA_CEILING = 0.00075
MAX_FAILED_DIRECTION_CORR = 0.80

PUBLIC_SCORES = {
    178: 0.6014113009,
    287: 0.6698787318,
    329: 0.6017908577,
    337: 0.6006206286219425,
    338: 0.5992145177,
    339: 0.5985187787,
    340: 0.601379175253277,
    342: ID342_PUBLIC,
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_train_sorted() -> pd.DataFrame:
    train = pd.read_csv(RAW_DIR / "ch2026_metrics_train.csv")
    train["date"] = pd.to_datetime(train["lifelog_date"]).dt.date
    return train.sort_values(["subject_id", "date"]).reset_index(drop=True)


def load_split_manifest(path: Path = SPLIT_MANIFEST_PATH) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for row in manifest["folds"]:
        splits.setdefault(row["split_name"], []).append(
            (
                np.asarray(row["train_idx"], dtype=int),
                np.asarray(row["valid_idx"], dtype=int),
            )
        )
    return splits


def find_oof(candidate_id: int) -> Path:
    paths = sorted(BASE_PREDICTION_DIR.glob(f"base_oof_id_{candidate_id}_*.csv"))
    if not paths:
        raise FileNotFoundError(f"Missing base OOF ID{candidate_id}")
    return paths[0]


def load_reference_values(train: pd.DataFrame, candidate_id: int) -> np.ndarray:
    pred = pd.read_csv(find_oof(candidate_id))
    pred["date"] = pd.to_datetime(pred["date"]).dt.date
    merged = train[["subject_id", "date"]].merge(pred, on=["subject_id", "date"], how="left")
    if merged[TARGETS].isna().any().any():
        raise ValueError(f"Reference ID{candidate_id} does not align with sorted train rows.")
    return merged[TARGETS].astype(float).to_numpy()


def fold_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(log_loss(y_true.astype(int), np.clip(y_pred, EPS, 1 - EPS), labels=[0, 1]))


def weighted_split_loss(y: np.ndarray, pred: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> float:
    weighted = 0.0
    total = 0
    for _fit_idx, valid_idx in folds:
        weighted += fold_loss(y[valid_idx], pred[valid_idx]) * len(valid_idx)
        total += len(valid_idx)
    return float(weighted / total)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return None
    aa = a[mask]
    bb = b[mask]
    if float(np.std(aa)) <= 0.0 or float(np.std(bb)) <= 0.0:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def direction_metrics(
    blended: np.ndarray,
    anchor_col: np.ndarray,
    failed_cols: dict[int, np.ndarray],
) -> dict[str, Any]:
    candidate_delta = blended - anchor_col
    corrs: dict[str, float | None] = {}
    agreements: dict[str, float | None] = {}
    corr_values: list[float] = []
    for candidate_id, failed_col in failed_cols.items():
        failed_delta = failed_col - anchor_col
        corr = safe_corr(candidate_delta, failed_delta)
        corrs[f"id{candidate_id}"] = corr
        if corr is not None:
            corr_values.append(corr)
        active = (np.abs(candidate_delta) > 1e-9) & (np.abs(failed_delta) > 1e-9)
        agreements[f"id{candidate_id}"] = (
            float((np.sign(candidate_delta[active]) == np.sign(failed_delta[active])).mean()) if active.any() else None
        )
    return {
        "mean_abs_movement_vs_id342": float(np.mean(np.abs(candidate_delta))),
        "max_abs_movement_vs_id342": float(np.max(np.abs(candidate_delta))),
        "failed_direction_corr": corrs,
        "failed_direction_sign_agreement": agreements,
        "max_failed_direction_corr": float(max(corr_values)) if corr_values else None,
    }


def target_scores(train: pd.DataFrame, values: np.ndarray) -> dict[str, float]:
    return {
        target: fold_loss(train[target].astype(int).to_numpy(), values[:, target_idx])
        for target_idx, target in enumerate(TARGETS)
    }


def scan_rows(
    train: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    refs: dict[int, np.ndarray],
) -> list[dict[str, Any]]:
    anchor = refs[ANCHOR_ID]
    rows: list[dict[str, Any]] = []
    for donor_id in DONOR_IDS:
        donor = refs[donor_id]
        donor_type = "failed_frontier" if donor_id in FAILED_FRONTIER_IDS else "stable_public_scored"
        for target in FOCUS_TARGETS:
            target_idx = TARGETS.index(target)
            y = train[target].astype(int).to_numpy()
            failed_cols = {candidate_id: refs[candidate_id][:, target_idx] for candidate_id in FAILED_FRONTIER_IDS}
            for weight in WEIGHTS:
                blended = np.clip(
                    (1.0 - weight) * anchor[:, target_idx] + weight * donor[:, target_idx],
                    EPS,
                    1.0 - EPS,
                )
                split_deltas: dict[str, float] = {}
                split_losses: dict[str, float] = {}
                split_anchor_losses: dict[str, float] = {}
                wins = 0
                for split_name in PRIMARY_SPLITS:
                    folds = splits[split_name]
                    anchor_loss = weighted_split_loss(y, anchor[:, target_idx], folds)
                    candidate_loss = weighted_split_loss(y, blended, folds)
                    delta = candidate_loss - anchor_loss
                    split_deltas[split_name] = delta
                    split_losses[split_name] = candidate_loss
                    split_anchor_losses[split_name] = anchor_loss
                    wins += int(delta < 0)
                direction = direction_metrics(blended, anchor[:, target_idx], failed_cols)
                max_failed_corr = direction["max_failed_direction_corr"]
                mean_delta = float(np.mean(list(split_deltas.values())))
                max_delta = float(np.max(list(split_deltas.values())))
                stable_lens_eligible = bool(
                    donor_type == "stable_public_scored"
                    and wins >= TARGET_WIN_FLOOR
                    and mean_delta <= TARGET_MEAN_DELTA_CEILING
                    and max_delta <= TARGET_MAX_DELTA_CEILING
                    and (max_failed_corr is None or max_failed_corr <= MAX_FAILED_DIRECTION_CORR)
                )
                failed_frontier_trap = bool(donor_type == "failed_frontier" and wins >= TARGET_WIN_FLOOR and mean_delta < 0)
                rows.append(
                    {
                        "config": f"ID342_{target}_donor{donor_id}_w{weight:g}",
                        "donor_id": donor_id,
                        "donor_type": donor_type,
                        "donor_public_score": PUBLIC_SCORES.get(donor_id),
                        "target": target,
                        "weight": float(weight),
                        "wins": int(wins),
                        "mean_delta": mean_delta,
                        "max_delta": max_delta,
                        "min_delta": float(np.min(list(split_deltas.values()))),
                        "estimated_all_target_delta": mean_delta / len(TARGETS),
                        "projected_public_proxy": ID342_PUBLIC + mean_delta / len(TARGETS),
                        "split_deltas": split_deltas,
                        "split_losses": split_losses,
                        "split_anchor_losses": split_anchor_losses,
                        "direction_metrics": direction,
                        "stable_lens_eligible": stable_lens_eligible,
                        "failed_frontier_trap": failed_frontier_trap,
                    }
                )
    return sorted(rows, key=lambda row: (not row["stable_lens_eligible"], row["mean_delta"], row["max_delta"]))


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    train = load_train_sorted()
    splits = load_split_manifest()
    missing_splits = [split for split in PRIMARY_SPLITS if split not in splits]
    if missing_splits:
        raise ValueError(f"Missing required split names: {missing_splits}")
    refs = {candidate_id: load_reference_values(train, candidate_id) for candidate_id in [ANCHOR_ID, *DONOR_IDS]}
    rows = scan_rows(train, splits, refs)
    stable_rows = [row for row in rows if row["donor_type"] == "stable_public_scored"]
    failed_rows = [row for row in rows if row["donor_type"] == "failed_frontier"]
    stable_pass_rows = [row for row in stable_rows if row["stable_lens_eligible"]]
    failed_trap_rows = [row for row in failed_rows if row["failed_frontier_trap"]]

    selected_by_target: dict[str, dict[str, Any]] = {}
    for target in FOCUS_TARGETS:
        target_rows = [row for row in stable_pass_rows if row["target"] == target]
        if target_rows:
            selected_by_target[target] = sorted(target_rows, key=lambda row: (row["mean_delta"], row["max_delta"]))[0]

    target_score_table = {
        str(candidate_id): {
            "public_score": PUBLIC_SCORES.get(candidate_id),
            "target_scores": target_scores(train, refs[candidate_id]),
        }
        for candidate_id in [178, 287, 329, 337, 338, 339, 340, 342]
    }

    estimated_all_target_delta = float(sum(row["mean_delta"] for row in selected_by_target.values()) / len(TARGETS))
    projected_public_proxy = ID342_PUBLIC + estimated_all_target_delta
    decision = {
        "stable_public_donor_pass_count": int(len(stable_pass_rows)),
        "failed_frontier_trap_count": int(len(failed_trap_rows)),
        "failed_frontier_local_trap_present": bool(failed_trap_rows),
        "selected_stable_targets": sorted(selected_by_target),
        "q3_selected_from_stable_public_donor": "Q3" in selected_by_target,
        "s3_selected_from_stable_public_donor": "S3" in selected_by_target,
        "estimated_all_target_delta_vs_id342": estimated_all_target_delta,
        "projected_public_proxy_from_id342": projected_public_proxy,
        "gap_to_public_055_after_proxy": projected_public_proxy - PUBLIC_055,
        "public_scored_target_swap_has_actionable_signal": bool(selected_by_target),
        "public_055_evidence": False,
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "submit_now": False,
        "next_action": (
            "Do not use ID287/ID329 target swaps despite large local gains. Stable public-scored donors do not "
            "pass the Q3/S3 gate, so this lane is anatomy evidence only."
        ),
    }

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_public_scored_target_swap_gate",
        "purpose": "Check whether public-scored anchor Q3/S3 predictions can safely improve ID342.",
        "anchor": {"id": ANCHOR_ID, "public_score": ID342_PUBLIC},
        "validation": {
            "primary_splits": PRIMARY_SPLITS,
            "weights": WEIGHTS,
            "stable_donor_ids": STABLE_DONOR_IDS,
            "failed_frontier_ids": FAILED_FRONTIER_IDS,
            "target_win_floor": TARGET_WIN_FLOOR,
            "target_mean_delta_ceiling": TARGET_MEAN_DELTA_CEILING,
            "target_max_delta_ceiling": TARGET_MAX_DELTA_CEILING,
            "max_failed_direction_corr": MAX_FAILED_DIRECTION_CORR,
        },
        "target_score_table": target_score_table,
        "top_stable_rows": sorted(stable_rows, key=lambda row: (row["mean_delta"], row["max_delta"]))[:20],
        "top_failed_frontier_rows": sorted(failed_rows, key=lambda row: (row["mean_delta"], row["max_delta"]))[:20],
        "stable_pass_rows": stable_pass_rows[:20],
        "failed_frontier_trap_rows": sorted(failed_trap_rows, key=lambda row: (row["mean_delta"], row["max_delta"]))[:20],
        "decision": decision,
        "no_predictions_created": True,
        "no_feature_matrix_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_action": True,
        "no_dacon_submission": True,
    }


def format_row(row: dict[str, Any]) -> str:
    return (
        f"| `{row['config']}` | `{row['donor_type']}` | {row['wins']} | "
        f"{row['mean_delta']:.9f} | {row['max_delta']:.9f} | "
        f"{row['projected_public_proxy']:.9f} |"
    )


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision = report["decision"]
    lines = [
        "# Post-ID342 public-scored target-swap gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- Stable public-donor pass count: `{decision['stable_public_donor_pass_count']}`",
        f"- Failed-frontier trap count: `{decision['failed_frontier_trap_count']}`",
        f"- Selected stable targets: `{decision['selected_stable_targets']}`",
        f"- Projected Public proxy: `{decision['projected_public_proxy_from_id342']:.9f}`",
        f"- Gap to Public 0.55: `{decision['gap_to_public_055_after_proxy']:.9f}`",
        f"- Kaggle now: `{decision['run_kaggle_now']}`",
        f"- DACON submit now: `{decision['submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Top Stable Donor Rows",
        "",
        "| config | donor type | wins | mean delta | max delta | projected public |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines.extend(format_row(row) for row in report["top_stable_rows"][:10])
    lines.extend(
        [
            "",
            "## Top Failed-Frontier Rows",
            "",
            "These rows are deliberately not promotable because ID287 and ID329 are confirmed public failures.",
            "",
            "| config | donor type | wins | mean delta | max delta | projected public |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    lines.extend(format_row(row) for row in report["top_failed_frontier_rows"][:10])
    lines.extend([""])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.output)
    decision = report["decision"]
    print(f"Wrote {rel(args.output)}")
    print(f"Wrote {rel(args.output.with_suffix('.md'))}")
    print(
        "Stable pass count: "
        f"{decision['stable_public_donor_pass_count']} | failed frontier traps: "
        f"{decision['failed_frontier_trap_count']} | selected={decision['selected_stable_targets']}"
    )


if __name__ == "__main__":
    main()
