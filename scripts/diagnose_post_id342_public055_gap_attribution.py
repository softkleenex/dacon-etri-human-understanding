#!/usr/bin/env python3
"""Post-ID342 Public-0.55 gap attribution over existing OOF shelf.

This report-only diagnostic asks whether any existing OOF candidate shelf item
is plausibly close to the Public 0.55 target after the ID342 public result. It
uses scored rows to estimate the stable public-minus-local gap, computes the
local threshold required for Public 0.55, and scans existing base OOF files for
local score, movement, and frontier-trap risk.

It creates no predictions, feature matrices, base prediction artifacts,
submission CSVs, submission-log rows, Kaggle actions, or DACON submissions.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw"
BASE_PREDICTION_DIR = ROOT / "artifacts" / "base_predictions"
SUBMISSION_LOG = ROOT / "submissions" / "submission_log.csv"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_public055_gap_attribution_20260624.json"
KST = timezone(timedelta(hours=9))

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
ID342_PUBLIC = 0.5960832519
PUBLIC_055 = 0.55
ANCHOR_ID = 342
FAILED_FRONTIER_IDS = {287, 329}
KNOWN_STABLE_PUBLIC_IDS = {178, 337, 338, 339, 340, 342}
EPS = 1e-6


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_train_sorted() -> pd.DataFrame:
    train = pd.read_csv(RAW_DIR / "ch2026_metrics_train.csv")
    train["date"] = pd.to_datetime(train["lifelog_date"]).dt.date
    return train.sort_values(["subject_id", "date"]).reset_index(drop=True)


def candidate_id_from_path(path: Path) -> int:
    match = re.search(r"base_oof_id_(\d+)_", path.name)
    if not match:
        raise ValueError(f"Cannot parse candidate id from {path.name}")
    return int(match.group(1))


def load_oof(path: Path, train: pd.DataFrame) -> np.ndarray:
    pred = pd.read_csv(path)
    pred["date"] = pd.to_datetime(pred["date"]).dt.date
    key = train[["subject_id", "date"]].copy()
    merged = key.merge(pred, on=["subject_id", "date"], how="left")
    if merged[TARGETS].isna().any().any():
        raise ValueError(f"{path.name} does not align with sorted train rows")
    return np.clip(merged[TARGETS].astype(float).to_numpy(), EPS, 1.0 - EPS)


def mean_log_loss(y_true: np.ndarray, pred: np.ndarray) -> tuple[float, dict[str, float]]:
    losses = {}
    for idx, target in enumerate(TARGETS):
        losses[target] = float(log_loss(y_true[:, idx].astype(int), pred[:, idx], labels=[0, 1]))
    return float(np.mean(list(losses.values()))), losses


def public_scores() -> dict[int, float]:
    log = pd.read_csv(SUBMISSION_LOG)
    scores: dict[int, float] = {}
    for _, row in log.iterrows():
        score = row.get("public_lb_score_num")
        if pd.notna(score):
            scores[int(row["id"])] = float(score)
    return scores


def stable_gap_summary(rows: list[dict[str, Any]], scores: dict[int, float]) -> dict[str, Any]:
    scored = [row for row in rows if row["id"] in scores]
    for row in scored:
        row["public_score"] = scores[row["id"]]
        row["public_minus_local"] = float(scores[row["id"]] - row["local_loss"])
    stable = [row for row in scored if row["id"] in KNOWN_STABLE_PUBLIC_IDS]
    gaps = [row["public_minus_local"] for row in stable]
    median_gap = float(np.median(gaps))
    mean_gap = float(np.mean(gaps))
    required_local = PUBLIC_055 - median_gap
    return {
        "scored_ids": [row["id"] for row in scored],
        "stable_public_ids": [row["id"] for row in stable],
        "stable_public_gap_median": median_gap,
        "stable_public_gap_mean": mean_gap,
        "stable_public_gap_min": float(np.min(gaps)),
        "stable_public_gap_max": float(np.max(gaps)),
        "required_full_local_for_public055_at_median_gap": required_local,
        "id342_local_to_required_margin": float(next(row["local_loss"] for row in rows if row["id"] == ANCHOR_ID) - required_local),
    }


def risk_label(row: dict[str, Any], stable_threshold: float) -> str:
    candidate_id = row["id"]
    name = row["filename"].lower()
    if candidate_id in FAILED_FRONTIER_IDS:
        return "scored_failed_frontier_trap"
    if "frontier" in name or "zerowide" in name or "scorepressure" in name:
        return "frontier_family_unscored"
    if (
        ("direct_lite" in name or "delta_pair" in name or "riskpen" in name or "post271" in name)
        and "blend" in name
    ):
        return "legacy_public_guided_pair_blend_family"
    if row["max_abs_delta_vs_id342"] > 0.35 or row["mean_abs_delta_vs_id342"] > 0.08:
        return "large_distribution_move"
    if row["local_loss"] <= stable_threshold:
        return "local_below_public055_threshold_but_needs_public_gap_proof"
    return "not_close_to_public055_threshold"


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    train = load_train_sorted()
    y_true = train[TARGETS].astype(int).to_numpy()
    anchor_path = sorted(BASE_PREDICTION_DIR.glob(f"base_oof_id_{ANCHOR_ID}_*.csv"))[0]
    anchor_pred = load_oof(anchor_path, train)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in sorted(BASE_PREDICTION_DIR.glob("base_oof_id_*.csv")):
        try:
            candidate_id = candidate_id_from_path(path)
            pred = load_oof(path, train)
            local_loss, target_losses = mean_log_loss(y_true, pred)
            delta = pred - anchor_pred
            rows.append(
                {
                    "id": candidate_id,
                    "filename": path.name,
                    "path": rel(path),
                    "local_loss": local_loss,
                    "target_losses": target_losses,
                    "mean_abs_delta_vs_id342": float(np.mean(np.abs(delta))),
                    "max_abs_delta_vs_id342": float(np.max(np.abs(delta))),
                    "target_mean_abs_delta_vs_id342": {
                        target: float(np.mean(np.abs(delta[:, idx]))) for idx, target in enumerate(TARGETS)
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": rel(path), "error": f"{type(exc).__name__}: {exc}"})
    scores = public_scores()
    gap = stable_gap_summary(rows, scores)
    required_local = gap["required_full_local_for_public055_at_median_gap"]
    for row in rows:
        row["projected_public_at_stable_median_gap"] = float(row["local_loss"] + gap["stable_public_gap_median"])
        row["gap_to_public055_at_stable_median_gap"] = float(
            row["projected_public_at_stable_median_gap"] - PUBLIC_055
        )
        row["public055_possible_at_stable_median_gap"] = bool(row["local_loss"] <= required_local)
        row["risk_label"] = risk_label(row, required_local)
        if row["id"] in scores:
            row["public_score"] = scores[row["id"]]
            row["public_minus_local"] = float(scores[row["id"]] - row["local_loss"])
    local_below_threshold = [row for row in rows if row["public055_possible_at_stable_median_gap"]]
    credible_below_threshold = [
        row
        for row in local_below_threshold
        if row["risk_label"] == "local_below_public055_threshold_but_needs_public_gap_proof"
    ]
    top_projected = sorted(rows, key=lambda row: row["projected_public_at_stable_median_gap"])[:30]
    top_low_movement = sorted(
        [row for row in rows if row["max_abs_delta_vs_id342"] <= 0.12],
        key=lambda row: row["projected_public_at_stable_median_gap"],
    )[:30]
    single_target_required_delta = float((ID342_PUBLIC - PUBLIC_055) * len(TARGETS))
    decision = {
        "candidate_oof_files_scanned": len(rows),
        "scan_errors": len(errors),
        "stable_public_gap_median": gap["stable_public_gap_median"],
        "required_full_local_for_public055_at_median_gap": required_local,
        "id342_local_to_required_margin": gap["id342_local_to_required_margin"],
        "single_target_loss_improvement_needed_if_only_one_target_moves": single_target_required_delta,
        "existing_rows_below_required_local": len(local_below_threshold),
        "credible_rows_below_required_local": len(credible_below_threshold),
        "best_projected_public_at_stable_gap": top_projected[0]["projected_public_at_stable_median_gap"],
        "best_projected_candidate_id": top_projected[0]["id"],
        "best_low_movement_projected_public": top_low_movement[0]["projected_public_at_stable_median_gap"]
        if top_low_movement
        else None,
        "best_low_movement_candidate_id": top_low_movement[0]["id"] if top_low_movement else None,
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "dacon_submit_now": False,
        "next_action": (
            "Do not submit existing low-local frontier rows. Public 0.55 requires either a validated local score "
            "near the frontier without frontier-family movement risk, or a new mechanism that compresses the "
            "public-minus-local gap; continue with validation redesign/public-gap attribution."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_public055_gap_attribution",
        "purpose": "Quantify Public 0.55 requirements after ID342 and triage existing OOF shelf proximity.",
        "stable_gap_summary": gap,
        "scan_errors": errors[:20],
        "decision": decision,
        "top_projected_rows": top_projected,
        "top_low_movement_rows": top_low_movement,
        "local_below_required_rows": sorted(local_below_threshold, key=lambda row: row["local_loss"])[:50],
        "credible_below_required_rows": sorted(credible_below_threshold, key=lambda row: row["local_loss"])[:50],
        "no_predictions_created": True,
        "no_feature_matrix_created": True,
        "no_base_prediction_artifact_created": True,
        "no_submission_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_action": True,
        "no_dacon_submission": True,
    }


def row_line(row: dict[str, Any]) -> str:
    return (
        f"| `{row['id']}` | `{row['local_loss']:.6f}` | "
        f"`{row['projected_public_at_stable_median_gap']:.6f}` | "
        f"`{row['gap_to_public055_at_stable_median_gap']:+.6f}` | "
        f"`{row['mean_abs_delta_vs_id342']:.6f}` | `{row['max_abs_delta_vs_id342']:.6f}` | "
        f"`{row['risk_label']}` | `{row['filename']}` |"
    )


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision = report["decision"]
    lines = [
        "# Post-ID342 Public-0.55 gap attribution",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- OOF files scanned: `{decision['candidate_oof_files_scanned']}`",
        f"- Stable public gap median: `{decision['stable_public_gap_median']:.9f}`",
        f"- Required full local for Public 0.55: `{decision['required_full_local_for_public055_at_median_gap']:.9f}`",
        f"- ID342 local margin above required: `{decision['id342_local_to_required_margin']:+.9f}`",
        f"- Existing rows below required local: `{decision['existing_rows_below_required_local']}`",
        f"- Credible rows below required local: `{decision['credible_rows_below_required_local']}`",
        f"- Best projected candidate: `ID{decision['best_projected_candidate_id']}` / `{decision['best_projected_public_at_stable_gap']:.9f}`",
        f"- Best low-movement projected candidate: `ID{decision['best_low_movement_candidate_id']}` / `{decision['best_low_movement_projected_public']:.9f}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Top Projected Rows",
        "",
        "| id | local | projected public | gap 0.55 | mean move | max move | risk | file |",
        "|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    lines.extend(row_line(row) for row in report["top_projected_rows"][:20])
    lines.extend(
        [
            "",
            "## Low-Movement Rows",
            "",
            "| id | local | projected public | gap 0.55 | mean move | max move | risk | file |",
            "|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    lines.extend(row_line(row) for row in report["top_low_movement_rows"][:20])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, output)
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(f"Credible rows below required local: {report['decision']['credible_rows_below_required_local']}")
    print(f"DACON submit now: {report['decision']['dacon_submit_now']}")


if __name__ == "__main__":
    main()
