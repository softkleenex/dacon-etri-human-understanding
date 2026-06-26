#!/usr/bin/env python3
"""Post-ID342 public-gap risk lens over existing OOF predictions.

This report-only diagnostic studies why low local OOF rows do not transfer to
Public LB. It extracts prediction-shape features from all base OOF files,
compares scored candidates against public outcomes, and derives a conservative
public-gap risk lens for the remaining shelf.

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
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_public_gap_risk_lens_20260624.json"
KST = timezone(timedelta(hours=9))

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
ANCHOR_ID = 342
PUBLIC_055 = 0.55
EPS = 1e-6
STABLE_PUBLIC_IDS = {178, 337, 338, 339, 340, 342}
FAILED_PUBLIC_IDS = {287, 329}
LEGACY_PUBLIC_GUIDED_TOKENS = (
    "frontier",
    "zerowide",
    "scorepressure",
    "direct_lite",
    "delta_pair",
    "riskpen",
    "post271",
)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def candidate_id_from_path(path: Path) -> int:
    match = re.search(r"base_oof_id_(\d+)_", path.name)
    if not match:
        raise ValueError(f"Cannot parse candidate id from {path.name}")
    return int(match.group(1))


def load_train_sorted() -> pd.DataFrame:
    train = pd.read_csv(RAW_DIR / "ch2026_metrics_train.csv")
    train["date"] = pd.to_datetime(train["lifelog_date"]).dt.date
    return train.sort_values(["subject_id", "date"]).reset_index(drop=True)


def load_oof(path: Path, train: pd.DataFrame) -> np.ndarray:
    pred = pd.read_csv(path)
    pred["date"] = pd.to_datetime(pred["date"]).dt.date
    merged = train[["subject_id", "date"]].merge(pred, on=["subject_id", "date"], how="left")
    if merged[TARGETS].isna().any().any():
        raise ValueError(f"{path.name} does not align with sorted train rows.")
    return np.clip(merged[TARGETS].astype(float).to_numpy(), EPS, 1.0 - EPS)


def public_scores() -> dict[int, float]:
    log = pd.read_csv(SUBMISSION_LOG)
    scores: dict[int, float] = {}
    for _, row in log.iterrows():
        score = row.get("public_lb_score_num")
        if pd.notna(score):
            scores[int(row["id"])] = float(score)
    return scores


def mean_loss(y_true: np.ndarray, pred: np.ndarray) -> tuple[float, dict[str, float]]:
    losses = {}
    for idx, target in enumerate(TARGETS):
        losses[target] = float(log_loss(y_true[:, idx].astype(int), pred[:, idx], labels=[0, 1]))
    return float(np.mean(list(losses.values()))), losses


def entropy(pred: np.ndarray) -> float:
    clipped = np.clip(pred, EPS, 1.0 - EPS)
    return float(np.mean(-(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))))


def legacy_family(filename: str) -> bool:
    lower = filename.lower()
    return any(token in lower for token in LEGACY_PUBLIC_GUIDED_TOKENS)


def row_features(
    path: Path,
    pred: np.ndarray,
    anchor: np.ndarray,
    train_target_rate: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, Any]:
    candidate_id = candidate_id_from_path(path)
    local, target_losses = mean_loss(y_true, pred)
    delta = pred - anchor
    target_mean_abs_delta = {
        target: float(np.mean(np.abs(delta[:, idx]))) for idx, target in enumerate(TARGETS)
    }
    target_max_abs_delta = {
        target: float(np.max(np.abs(delta[:, idx]))) for idx, target in enumerate(TARGETS)
    }
    pred_mean = pred.mean(axis=0)
    mean_rate_shift = np.abs(pred_mean - train_target_rate)
    return {
        "id": candidate_id,
        "filename": path.name,
        "path": rel(path),
        "local_loss": local,
        "target_losses": target_losses,
        "mean_abs_delta_vs_id342": float(np.mean(np.abs(delta))),
        "max_abs_delta_vs_id342": float(np.max(np.abs(delta))),
        "target_mean_abs_delta_vs_id342": target_mean_abs_delta,
        "target_max_abs_delta_vs_id342": target_max_abs_delta,
        "max_target_mean_abs_delta_vs_id342": float(max(target_mean_abs_delta.values())),
        "max_target_abs_delta_vs_id342": float(max(target_max_abs_delta.values())),
        "prob_entropy": entropy(pred),
        "mean_prediction": {target: float(pred_mean[idx]) for idx, target in enumerate(TARGETS)},
        "max_abs_mean_rate_shift": float(np.max(mean_rate_shift)),
        "mean_abs_mean_rate_shift": float(np.mean(mean_rate_shift)),
        "extreme_probability_rate": float(np.mean((pred <= 0.02) | (pred >= 0.98))),
        "legacy_public_guided_family": legacy_family(path.name),
    }


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    x = pd.Series(xs).rank(method="average").to_numpy()
    y = pd.Series(ys).rank(method="average").to_numpy()
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    train = load_train_sorted()
    y_true = train[TARGETS].astype(int).to_numpy()
    train_target_rate = y_true.mean(axis=0)
    anchor_path = sorted(BASE_PREDICTION_DIR.glob(f"base_oof_id_{ANCHOR_ID}_*.csv"))[0]
    anchor = load_oof(anchor_path, train)
    scores = public_scores()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in sorted(BASE_PREDICTION_DIR.glob("base_oof_id_*.csv")):
        try:
            row = row_features(path, load_oof(path, train), anchor, train_target_rate, y_true)
            if row["id"] in scores:
                row["public_score"] = scores[row["id"]]
                row["public_minus_local"] = float(scores[row["id"]] - row["local_loss"])
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": rel(path), "error": f"{type(exc).__name__}: {exc}"})
    scored = [row for row in rows if "public_score" in row]
    stable = [row for row in scored if row["id"] in STABLE_PUBLIC_IDS]
    failed = [row for row in scored if row["id"] in FAILED_PUBLIC_IDS]
    stable_gap_median = float(np.median([row["public_minus_local"] for row in stable]))
    required_local = PUBLIC_055 - stable_gap_median
    feature_names = [
        "local_loss",
        "mean_abs_delta_vs_id342",
        "max_abs_delta_vs_id342",
        "max_target_mean_abs_delta_vs_id342",
        "max_target_abs_delta_vs_id342",
        "prob_entropy",
        "max_abs_mean_rate_shift",
        "mean_abs_mean_rate_shift",
        "extreme_probability_rate",
    ]
    correlations = []
    for name in feature_names:
        xs = [float(row[name]) for row in scored]
        y_gap = [float(row["public_minus_local"]) for row in scored]
        y_public = [float(row["public_score"]) for row in scored]
        correlations.append(
            {
                "feature": name,
                "pearson_vs_public_minus_local": pearson(xs, y_gap),
                "spearman_vs_public_minus_local": spearman(xs, y_gap),
                "pearson_vs_public_score": pearson(xs, y_public),
                "spearman_vs_public_score": spearman(xs, y_public),
            }
        )
    stable_envelope = {
        "mean_abs_delta_vs_id342_p95": quantile([row["mean_abs_delta_vs_id342"] for row in stable], 0.95),
        "max_abs_delta_vs_id342_p95": quantile([row["max_abs_delta_vs_id342"] for row in stable], 0.95),
        "max_target_mean_abs_delta_vs_id342_p95": quantile(
            [row["max_target_mean_abs_delta_vs_id342"] for row in stable], 0.95
        ),
        "max_abs_mean_rate_shift_p95": quantile([row["max_abs_mean_rate_shift"] for row in stable], 0.95),
        "extreme_probability_rate_p95": quantile([row["extreme_probability_rate"] for row in stable], 0.95),
    }

    def lens_label(row: dict[str, Any]) -> str:
        if row["id"] in FAILED_PUBLIC_IDS:
            return "scored_failed_public_trap"
        if row["legacy_public_guided_family"]:
            return "legacy_public_guided_family"
        if row["mean_abs_delta_vs_id342"] > stable_envelope["mean_abs_delta_vs_id342_p95"]:
            return "outside_stable_mean_move_envelope"
        if row["max_abs_delta_vs_id342"] > stable_envelope["max_abs_delta_vs_id342_p95"]:
            return "outside_stable_max_move_envelope"
        if row["max_target_mean_abs_delta_vs_id342"] > stable_envelope["max_target_mean_abs_delta_vs_id342_p95"]:
            return "outside_stable_target_move_envelope"
        if row["max_abs_mean_rate_shift"] > stable_envelope["max_abs_mean_rate_shift_p95"]:
            return "outside_stable_rate_shift_envelope"
        if row["extreme_probability_rate"] > stable_envelope["extreme_probability_rate_p95"]:
            return "outside_stable_extreme_probability_envelope"
        return "stable_public_shape_envelope"

    for row in rows:
        row["public_gap_risk_label"] = lens_label(row)
        row["projected_public_at_stable_gap"] = float(row["local_loss"] + stable_gap_median)
        row["gap_to_public055_at_stable_gap"] = float(row["projected_public_at_stable_gap"] - PUBLIC_055)
        row["public055_local_threshold_pass"] = bool(row["local_loss"] <= required_local)
        row["public055_shape_lens_pass"] = bool(
            row["public055_local_threshold_pass"] and row["public_gap_risk_label"] == "stable_public_shape_envelope"
        )

    shape_pass = [row for row in rows if row["public_gap_risk_label"] == "stable_public_shape_envelope"]
    public055_shape_pass = [row for row in rows if row["public055_shape_lens_pass"]]
    best_shape = sorted(shape_pass, key=lambda row: row["projected_public_at_stable_gap"])[:30]
    failed_public_summary = [
        {
            "id": row["id"],
            "local_loss": row["local_loss"],
            "public_score": row["public_score"],
            "public_minus_local": row["public_minus_local"],
            "mean_abs_delta_vs_id342": row["mean_abs_delta_vs_id342"],
            "max_abs_delta_vs_id342": row["max_abs_delta_vs_id342"],
            "public_gap_risk_label": row["public_gap_risk_label"],
        }
        for row in failed
    ]
    decision = {
        "oof_files_scanned": len(rows),
        "scan_errors": len(errors),
        "scored_rows": len(scored),
        "stable_gap_median": stable_gap_median,
        "required_local_for_public055_at_stable_gap": required_local,
        "stable_shape_envelope_rows": len(shape_pass),
        "public055_shape_lens_pass_count": len(public055_shape_pass),
        "best_shape_envelope_candidate_id": best_shape[0]["id"] if best_shape else None,
        "best_shape_envelope_projected_public": best_shape[0]["projected_public_at_stable_gap"] if best_shape else None,
        "best_shape_envelope_gap_to_055": best_shape[0]["gap_to_public055_at_stable_gap"] if best_shape else None,
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "dacon_submit_now": False,
        "next_action": (
            "The stable-shape lens leaves no Public-0.55 shelf candidate. The next experiment should search for "
            "new signal inside the stable movement envelope or redesign validation around public-gap compression, "
            "not amplify local frontier blends."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_public_gap_risk_lens",
        "purpose": "Derive a conservative public-gap risk lens from scored OOF prediction shapes.",
        "stable_public_ids": sorted(STABLE_PUBLIC_IDS),
        "failed_public_ids": sorted(FAILED_PUBLIC_IDS),
        "stable_shape_envelope": stable_envelope,
        "correlations": correlations,
        "failed_public_summary": failed_public_summary,
        "decision": decision,
        "top_shape_envelope_rows": best_shape,
        "public055_shape_lens_pass_rows": sorted(public055_shape_pass, key=lambda row: row["local_loss"]),
        "scan_errors": errors[:20],
        "no_predictions_created": True,
        "no_feature_matrix_created": True,
        "no_base_prediction_artifact_created": True,
        "no_submission_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_action": True,
        "no_dacon_submission": True,
    }


def format_row(row: dict[str, Any]) -> str:
    return (
        f"| `{row['id']}` | `{row['local_loss']:.6f}` | "
        f"`{row['projected_public_at_stable_gap']:.6f}` | "
        f"`{row['gap_to_public055_at_stable_gap']:+.6f}` | "
        f"`{row['mean_abs_delta_vs_id342']:.6f}` | `{row['max_abs_delta_vs_id342']:.6f}` | "
        f"`{row['public_gap_risk_label']}` | `{row['filename']}` |"
    )


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision = report["decision"]
    lines = [
        "# Post-ID342 public-gap risk lens",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- OOF files scanned: `{decision['oof_files_scanned']}`",
        f"- Scored rows: `{decision['scored_rows']}`",
        f"- Stable gap median: `{decision['stable_gap_median']:.9f}`",
        f"- Required local for Public 0.55: `{decision['required_local_for_public055_at_stable_gap']:.9f}`",
        f"- Stable shape-envelope rows: `{decision['stable_shape_envelope_rows']}`",
        f"- Public-0.55 shape-lens pass count: `{decision['public055_shape_lens_pass_count']}`",
        f"- Best shape-envelope candidate: `ID{decision['best_shape_envelope_candidate_id']}` / `{decision['best_shape_envelope_projected_public']:.9f}`",
        f"- Best shape-envelope gap to 0.55: `{decision['best_shape_envelope_gap_to_055']:+.9f}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Stable Shape Envelope",
        "",
        "```json",
        json.dumps(report["stable_shape_envelope"], indent=2, sort_keys=True),
        "```",
        "",
        "## Top Shape-Envelope Rows",
        "",
        "| id | local | projected public | gap 0.55 | mean move | max move | lens | file |",
        "|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    lines.extend(format_row(row) for row in report["top_shape_envelope_rows"][:20])
    lines.extend(
        [
            "",
            "## Public Gap Correlations",
            "",
            "| feature | pearson gap | spearman gap | pearson public | spearman public |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["correlations"]:
        lines.append(
            "| `{feature}` | `{pg}` | `{sg}` | `{pp}` | `{sp}` |".format(
                feature=row["feature"],
                pg=None if row["pearson_vs_public_minus_local"] is None else f"{row['pearson_vs_public_minus_local']:.3f}",
                sg=None
                if row["spearman_vs_public_minus_local"] is None
                else f"{row['spearman_vs_public_minus_local']:.3f}",
                pp=None if row["pearson_vs_public_score"] is None else f"{row['pearson_vs_public_score']:.3f}",
                sp=None if row["spearman_vs_public_score"] is None else f"{row['spearman_vs_public_score']:.3f}",
            )
        )
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
    print(f"Public-0.55 shape-lens pass count: {report['decision']['public055_shape_lens_pass_count']}")
    print(f"DACON submit now: {report['decision']['dacon_submit_now']}")


if __name__ == "__main__":
    main()
