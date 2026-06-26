#!/usr/bin/env python3
"""Post-ID342 stable-envelope micro-blend scan.

This report-only diagnostic tests whether any existing OOF direction can improve
ID342 while remaining inside the public-stable prediction-shape envelope. It
convexly blends ID342 toward each existing OOF source over a small alpha grid,
then rejects rows that exceed the movement/rate/extreme-probability envelope
learned by `post_id342_public_gap_risk_lens`.

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
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_stable_envelope_microblend_scan_20260625.json"
RISK_LENS_REPORT = REPORTS_DIR / "post_id342_public_gap_risk_lens_20260624.json"
KST = timezone(timedelta(hours=9))

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
ANCHOR_ID = 342
PUBLIC_055 = 0.55
EPS = 1e-6
ALPHA_GRID = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12]
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
        raise ValueError(f"{path.name} does not align with sorted train rows")
    return np.clip(merged[TARGETS].astype(float).to_numpy(), EPS, 1.0 - EPS)


def mean_loss(y_true: np.ndarray, pred: np.ndarray) -> tuple[float, dict[str, float]]:
    losses = {}
    for idx, target in enumerate(TARGETS):
        losses[target] = float(log_loss(y_true[:, idx].astype(int), pred[:, idx], labels=[0, 1]))
    return float(np.mean(list(losses.values()))), losses


def envelope_from_report(path: Path = RISK_LENS_REPORT) -> tuple[dict[str, float], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["stable_shape_envelope"], float(payload["decision"]["stable_gap_median"])


def legacy_public_guided(filename: str) -> bool:
    lower = filename.lower()
    return any(token in lower for token in LEGACY_PUBLIC_GUIDED_TOKENS)


def shape_metrics(pred: np.ndarray, anchor: np.ndarray, train_target_rate: np.ndarray) -> dict[str, Any]:
    delta = pred - anchor
    target_mean_abs_delta = {target: float(np.mean(np.abs(delta[:, idx]))) for idx, target in enumerate(TARGETS)}
    pred_mean = pred.mean(axis=0)
    mean_rate_shift = np.abs(pred_mean - train_target_rate)
    return {
        "mean_abs_delta_vs_id342": float(np.mean(np.abs(delta))),
        "max_abs_delta_vs_id342": float(np.max(np.abs(delta))),
        "target_mean_abs_delta_vs_id342": target_mean_abs_delta,
        "max_target_mean_abs_delta_vs_id342": float(max(target_mean_abs_delta.values())),
        "max_abs_mean_rate_shift": float(np.max(mean_rate_shift)),
        "extreme_probability_rate": float(np.mean((pred <= 0.02) | (pred >= 0.98))),
    }


def envelope_pass(metrics: dict[str, Any], envelope: dict[str, float]) -> bool:
    return bool(
        metrics["mean_abs_delta_vs_id342"] <= envelope["mean_abs_delta_vs_id342_p95"]
        and metrics["max_abs_delta_vs_id342"] <= envelope["max_abs_delta_vs_id342_p95"]
        and metrics["max_target_mean_abs_delta_vs_id342"] <= envelope["max_target_mean_abs_delta_vs_id342_p95"]
        and metrics["max_abs_mean_rate_shift"] <= envelope["max_abs_mean_rate_shift_p95"]
        and metrics["extreme_probability_rate"] <= envelope["extreme_probability_rate_p95"]
    )


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    train = load_train_sorted()
    y_true = train[TARGETS].astype(int).to_numpy()
    train_target_rate = y_true.mean(axis=0)
    envelope, stable_gap_median = envelope_from_report()
    required_local = PUBLIC_055 - stable_gap_median
    anchor_path = sorted(BASE_PREDICTION_DIR.glob(f"base_oof_id_{ANCHOR_ID}_*.csv"))[0]
    anchor_pred = load_oof(anchor_path, train)
    anchor_loss, anchor_target_losses = mean_loss(y_true, anchor_pred)
    rows: list[dict[str, Any]] = []
    source_best_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for source_path in sorted(BASE_PREDICTION_DIR.glob("base_oof_id_*.csv")):
        try:
            source_id = candidate_id_from_path(source_path)
            if source_id == ANCHOR_ID:
                continue
            source_pred = load_oof(source_path, train)
            source_rows = []
            for alpha in ALPHA_GRID:
                pred = np.clip((1.0 - alpha) * anchor_pred + alpha * source_pred, EPS, 1.0 - EPS)
                local_loss, target_losses = mean_loss(y_true, pred)
                metrics = shape_metrics(pred, anchor_pred, train_target_rate)
                row = {
                    "source_id": source_id,
                    "source_file": source_path.name,
                    "source_path": rel(source_path),
                    "alpha": alpha,
                    "local_loss": local_loss,
                    "target_losses": target_losses,
                    "delta_local_vs_id342": float(local_loss - anchor_loss),
                    "projected_public_at_stable_gap": float(local_loss + stable_gap_median),
                    "gap_to_public055_at_stable_gap": float(local_loss + stable_gap_median - PUBLIC_055),
                    "public055_local_threshold_pass": bool(local_loss <= required_local),
                    "legacy_public_guided_source": legacy_public_guided(source_path.name),
                    **metrics,
                }
                row["stable_envelope_pass"] = envelope_pass(row, envelope)
                row["gate_pass"] = bool(
                    row["stable_envelope_pass"]
                    and row["delta_local_vs_id342"] < -0.00005
                    and not row["legacy_public_guided_source"]
                )
                rows.append(row)
                source_rows.append(row)
            source_best_rows.append(
                min(
                    source_rows,
                    key=lambda item: (
                        not item["stable_envelope_pass"],
                        item["projected_public_at_stable_gap"],
                        item["alpha"],
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": rel(source_path), "error": f"{type(exc).__name__}: {exc}"})
    envelope_rows = [row for row in rows if row["stable_envelope_pass"]]
    gate_rows = [row for row in rows if row["gate_pass"]]
    public055_rows = [row for row in rows if row["stable_envelope_pass"] and row["public055_local_threshold_pass"]]
    best_envelope = sorted(envelope_rows, key=lambda row: row["projected_public_at_stable_gap"])[:50]
    best_gate = sorted(gate_rows, key=lambda row: row["projected_public_at_stable_gap"])[:50]
    best_by_source = sorted(source_best_rows, key=lambda row: row["projected_public_at_stable_gap"])[:50]
    best_nonlegacy_envelope = [row for row in best_envelope if not row["legacy_public_guided_source"]][:30]
    decision = {
        "sources_scanned": len(source_best_rows),
        "blend_rows_scanned": len(rows),
        "scan_errors": len(errors),
        "anchor_local_loss": anchor_loss,
        "anchor_projected_public_at_stable_gap": float(anchor_loss + stable_gap_median),
        "stable_gap_median": stable_gap_median,
        "required_local_for_public055": required_local,
        "stable_envelope_rows": len(envelope_rows),
        "public055_envelope_rows": len(public055_rows),
        "gate_pass_count": len(gate_rows),
        "best_envelope_source_id": best_envelope[0]["source_id"] if best_envelope else None,
        "best_envelope_alpha": best_envelope[0]["alpha"] if best_envelope else None,
        "best_envelope_projected_public": best_envelope[0]["projected_public_at_stable_gap"] if best_envelope else None,
        "best_envelope_delta_local_vs_id342": best_envelope[0]["delta_local_vs_id342"] if best_envelope else None,
        "best_gate_source_id": best_gate[0]["source_id"] if best_gate else None,
        "best_gate_projected_public": best_gate[0]["projected_public_at_stable_gap"] if best_gate else None,
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "dacon_submit_now": False,
        "next_action": (
            "No submission artifact should be created unless a row both improves ID342 inside the stable envelope "
            "and clears legacy-source risk. If no gate row exists, continue with validation/source redesign."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_stable_envelope_microblend_scan",
        "purpose": "Scan tiny ID342-convex blends while enforcing the public-stable prediction-shape envelope.",
        "alpha_grid": ALPHA_GRID,
        "stable_shape_envelope": envelope,
        "anchor": {
            "id": ANCHOR_ID,
            "path": rel(anchor_path),
            "local_loss": anchor_loss,
            "target_losses": anchor_target_losses,
        },
        "decision": decision,
        "top_envelope_rows": best_envelope,
        "top_nonlegacy_envelope_rows": best_nonlegacy_envelope,
        "top_gate_rows": best_gate,
        "top_source_best_rows": best_by_source,
        "scan_errors": errors[:20],
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
        f"| `{row['source_id']}` | `{row['alpha']:.4g}` | `{row['local_loss']:.6f}` | "
        f"`{row['delta_local_vs_id342']:+.6f}` | `{row['projected_public_at_stable_gap']:.6f}` | "
        f"`{row['gap_to_public055_at_stable_gap']:+.6f}` | `{row['mean_abs_delta_vs_id342']:.6f}` | "
        f"`{row['max_abs_delta_vs_id342']:.6f}` | `{row['legacy_public_guided_source']}` | `{row['source_file']}` |"
    )


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision = report["decision"]
    lines = [
        "# Post-ID342 stable-envelope micro-blend scan",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- Sources scanned: `{decision['sources_scanned']}`",
        f"- Blend rows scanned: `{decision['blend_rows_scanned']}`",
        f"- Anchor projected public: `{decision['anchor_projected_public_at_stable_gap']:.9f}`",
        f"- Stable-envelope rows: `{decision['stable_envelope_rows']}`",
        f"- Public-0.55 envelope rows: `{decision['public055_envelope_rows']}`",
        f"- Gate pass count: `{decision['gate_pass_count']}`",
        f"- Best envelope source/alpha: `ID{decision['best_envelope_source_id']}` / `{decision['best_envelope_alpha']}`",
        f"- Best envelope projected public: `{decision['best_envelope_projected_public']:.9f}`",
        f"- Best envelope local delta vs ID342: `{decision['best_envelope_delta_local_vs_id342']:+.9f}`",
        f"- Best gate source: `{decision['best_gate_source_id']}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Top Envelope Rows",
        "",
        "| source | alpha | local | delta local | projected public | gap 0.55 | mean move | max move | legacy | file |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    lines.extend(row_line(row) for row in report["top_envelope_rows"][:25])
    lines.extend(
        [
            "",
            "## Top Nonlegacy Envelope Rows",
            "",
            "| source | alpha | local | delta local | projected public | gap 0.55 | mean move | max move | legacy | file |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    lines.extend(row_line(row) for row in report["top_nonlegacy_envelope_rows"][:25])
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
    print(f"Gate pass count: {report['decision']['gate_pass_count']}")
    print(f"Best envelope projected public: {report['decision']['best_envelope_projected_public']}")
    print(f"DACON submit now: {report['decision']['dacon_submit_now']}")


if __name__ == "__main__":
    main()
