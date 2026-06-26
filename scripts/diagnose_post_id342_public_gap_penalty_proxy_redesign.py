#!/usr/bin/env python3
"""Post-ID342 public-gap penalty proxy redesign.

This report-only diagnostic turns the public-gap risk lens into a concrete
validation-redesign screen. It fits a tiny nonnegative penalty on prediction
shape excess over the stable ID342 neighborhood, checks leave-one-out behavior
on the scored rows, and applies the proxy to the full OOF shelf.

It creates no prediction CSV, feature matrix artifact, base prediction artifact,
submission CSV, submission-log row, Kaggle action, or DACON submission.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id342_public_gap_risk_lens import (  # noqa: E402
    ANCHOR_ID,
    BASE_PREDICTION_DIR,
    FAILED_PUBLIC_IDS,
    PUBLIC_055,
    STABLE_PUBLIC_IDS,
    TARGETS,
    load_oof,
    load_train_sorted,
    public_scores,
    rel,
    row_features,
)

KST = timezone(timedelta(hours=9))
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_public_gap_penalty_proxy_redesign_20260625.json"
RISK_LENS_REPORT = REPORTS_DIR / "post_id342_public_gap_risk_lens_20260624.json"

PENALTY_FEATURES = [
    ("mean_abs_delta_vs_id342", "mean_abs_delta_vs_id342_p95"),
    ("max_abs_delta_vs_id342", "max_abs_delta_vs_id342_p95"),
    ("max_target_mean_abs_delta_vs_id342", "max_target_mean_abs_delta_vs_id342_p95"),
    ("max_abs_mean_rate_shift", "max_abs_mean_rate_shift_p95"),
    ("extreme_probability_rate", "extreme_probability_rate_p95"),
]


def load_stable_shape_envelope() -> dict[str, float]:
    payload = json.loads(RISK_LENS_REPORT.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in payload["stable_shape_envelope"].items()}


def collect_oof_rows() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
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
    return rows, errors


def stable_gap_median(rows: list[dict[str, Any]]) -> float:
    stable = [
        float(row["public_minus_local"])
        for row in rows
        if row.get("id") in STABLE_PUBLIC_IDS and "public_minus_local" in row
    ]
    if stable:
        return float(np.median(stable))
    scored = [float(row["public_minus_local"]) for row in rows if "public_minus_local" in row]
    return float(np.median(scored))


def excess_vector(row: dict[str, Any], envelope: dict[str, float]) -> np.ndarray:
    values = []
    for feature, envelope_key in PENALTY_FEATURES:
        threshold = max(float(envelope[envelope_key]), 1e-12)
        excess = max(0.0, float(row[feature]) - threshold) / threshold
        values.append(excess)
    return np.asarray(values, dtype=float)


def fit_nonnegative_penalty(rows: list[dict[str, Any]], envelope: dict[str, float], stable_gap: float) -> np.ndarray:
    scored = [row for row in rows if "public_score" in row]
    if not scored:
        return np.zeros(len(PENALTY_FEATURES), dtype=float)
    x = np.vstack([excess_vector(row, envelope) for row in scored])
    y = np.asarray(
        [max(0.0, float(row["public_score"]) - (float(row["local_loss"]) + stable_gap)) for row in scored],
        dtype=float,
    )
    if not np.isfinite(x).all() or float(np.abs(x).sum()) <= 0.0:
        return np.zeros(len(PENALTY_FEATURES), dtype=float)
    ridge = 1e-6 * np.eye(x.shape[1])
    weights = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    return np.clip(weights, 0.0, None)


def proxy_public(row: dict[str, Any], envelope: dict[str, float], stable_gap: float, weights: np.ndarray) -> float:
    penalty = float(excess_vector(row, envelope) @ weights)
    return float(row["local_loss"] + stable_gap + penalty)


def lens_label(row: dict[str, Any], envelope: dict[str, float]) -> str:
    if row["id"] in FAILED_PUBLIC_IDS:
        return "scored_failed_public_trap"
    if row["legacy_public_guided_family"]:
        return "legacy_public_guided_family"
    checks = [
        ("mean_abs_delta_vs_id342", "mean_abs_delta_vs_id342_p95", "outside_stable_mean_move_envelope"),
        ("max_abs_delta_vs_id342", "max_abs_delta_vs_id342_p95", "outside_stable_max_move_envelope"),
        (
            "max_target_mean_abs_delta_vs_id342",
            "max_target_mean_abs_delta_vs_id342_p95",
            "outside_stable_target_move_envelope",
        ),
        ("max_abs_mean_rate_shift", "max_abs_mean_rate_shift_p95", "outside_stable_rate_shift_envelope"),
        ("extreme_probability_rate", "extreme_probability_rate_p95", "outside_stable_extreme_probability_envelope"),
    ]
    for feature, envelope_key, label in checks:
        if float(row[feature]) > float(envelope[envelope_key]):
            return label
    return "stable_public_shape_envelope"


def pairwise_agreement(rows: list[dict[str, Any]], prediction_key: str) -> float | None:
    if len(rows) < 2:
        return None
    total = 0
    agree = 0
    for left_idx, left in enumerate(rows):
        for right in rows[left_idx + 1 :]:
            public_sign = np.sign(float(left["public_score"]) - float(right["public_score"]))
            pred_sign = np.sign(float(left[prediction_key]) - float(right[prediction_key]))
            if public_sign == 0 or pred_sign == 0:
                continue
            total += 1
            agree += int(public_sign == pred_sign)
    return float(agree / total) if total else None


def loo_predictions(scored_rows: list[dict[str, Any]], envelope: dict[str, float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for held_out in scored_rows:
        train_rows = [row for row in scored_rows if row["id"] != held_out["id"]]
        gap = stable_gap_median(train_rows)
        weights = fit_nonnegative_penalty(train_rows, envelope, gap)
        stable_projection = float(held_out["local_loss"] + gap)
        penalty_projection = proxy_public(held_out, envelope, gap, weights)
        out.append(
            {
                "id": int(held_out["id"]),
                "public_score": float(held_out["public_score"]),
                "local_loss": float(held_out["local_loss"]),
                "stable_gap_projection": stable_projection,
                "penalty_proxy_projection": penalty_projection,
                "stable_gap_abs_error": abs(stable_projection - float(held_out["public_score"])),
                "penalty_proxy_abs_error": abs(penalty_projection - float(held_out["public_score"])),
                "trained_stable_gap": gap,
                "trained_penalty_weights": {
                    feature: float(weight)
                    for (feature, _envelope_key), weight in zip(PENALTY_FEATURES, weights, strict=True)
                },
            }
        )
    return out


def error_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    if not rows:
        return {"mae": None, "rmse": None}
    errors = np.asarray([float(row[key]) for row in rows], dtype=float)
    return {"mae": float(errors.mean()), "rmse": float(np.sqrt(np.mean(errors**2)))}


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    rows, errors = collect_oof_rows()
    envelope = load_stable_shape_envelope()
    gap = stable_gap_median(rows)
    weights = fit_nonnegative_penalty(rows, envelope, gap)
    scored_rows = [row for row in rows if "public_score" in row]
    for row in rows:
        row["stable_gap_projection"] = float(row["local_loss"] + gap)
        row["penalty_proxy_projection"] = proxy_public(row, envelope, gap, weights)
        row["penalty_proxy_gap_to_public055"] = float(row["penalty_proxy_projection"] - PUBLIC_055)
        row["stable_shape_label"] = lens_label(row, envelope)
        row["public055_penalty_proxy_pass"] = bool(row["penalty_proxy_projection"] <= PUBLIC_055)
        row["credible_public055_penalty_proxy_pass"] = bool(
            row["public055_penalty_proxy_pass"]
            and row["stable_shape_label"] == "stable_public_shape_envelope"
            and not row["legacy_public_guided_family"]
            and row["id"] not in FAILED_PUBLIC_IDS
        )
    for row in scored_rows:
        row["stable_gap_abs_error"] = abs(float(row["stable_gap_projection"]) - float(row["public_score"]))
        row["penalty_proxy_abs_error"] = abs(float(row["penalty_proxy_projection"]) - float(row["public_score"]))

    loo_rows = loo_predictions(scored_rows, envelope)
    credible_pass_rows = [row for row in rows if row["credible_public055_penalty_proxy_pass"]]
    penalty_pass_rows = [row for row in rows if row["public055_penalty_proxy_pass"]]
    top_penalty_rows = sorted(rows, key=lambda row: row["penalty_proxy_projection"])[:30]
    top_credible_rows = sorted(
        [row for row in rows if row["stable_shape_label"] == "stable_public_shape_envelope"],
        key=lambda row: row["penalty_proxy_projection"],
    )[:30]
    decision = {
        "oof_files_scanned": int(len(rows)),
        "scan_errors": int(len(errors)),
        "scored_rows": int(len(scored_rows)),
        "stable_gap_median": gap,
        "global_penalty_weights": {
            feature: float(weight) for (feature, _envelope_key), weight in zip(PENALTY_FEATURES, weights, strict=True)
        },
        "stable_gap_loo_error": error_summary(loo_rows, "stable_gap_abs_error"),
        "penalty_proxy_loo_error": error_summary(loo_rows, "penalty_proxy_abs_error"),
        "stable_gap_scored_error": error_summary(scored_rows, "stable_gap_abs_error"),
        "penalty_proxy_scored_error": error_summary(scored_rows, "penalty_proxy_abs_error"),
        "stable_gap_pairwise_agreement": pairwise_agreement(scored_rows, "stable_gap_projection"),
        "penalty_proxy_pairwise_agreement": pairwise_agreement(scored_rows, "penalty_proxy_projection"),
        "public055_penalty_proxy_pass_count": int(len(penalty_pass_rows)),
        "credible_public055_penalty_proxy_pass_count": int(len(credible_pass_rows)),
        "best_penalty_proxy_candidate": {
            key: top_penalty_rows[0][key]
            for key in [
                "id",
                "filename",
                "local_loss",
                "penalty_proxy_projection",
                "penalty_proxy_gap_to_public055",
                "stable_shape_label",
                "legacy_public_guided_family",
            ]
        }
        if top_penalty_rows
        else None,
        "best_credible_shape_candidate": {
            key: top_credible_rows[0][key]
            for key in [
                "id",
                "filename",
                "local_loss",
                "penalty_proxy_projection",
                "penalty_proxy_gap_to_public055",
                "stable_shape_label",
                "legacy_public_guided_family",
            ]
        }
        if top_credible_rows
        else None,
        "validation_redesign_gate_pass": bool(credible_pass_rows),
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "dacon_submit_now": False,
        "next_action": (
            "Use the penalty proxy as a stricter review lens, not as a candidate generator. If no credible "
            "Public-0.55 row appears, continue with validation redesign or a materially new source; do not "
            "submit the local-frontier rows that only pass before shape penalties."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_public_gap_penalty_proxy_redesign",
        "purpose": (
            "Fit a tiny shape-excess penalty over the public-gap risk lens, evaluate leave-one-out behavior on "
            "scored rows, and apply the proxy to the OOF shelf before any new candidate work."
        ),
        "stable_shape_envelope": envelope,
        "penalty_features": [
            {"feature": feature, "envelope_key": envelope_key} for feature, envelope_key in PENALTY_FEATURES
        ],
        "scored_loo_rows": loo_rows,
        "top_penalty_proxy_rows": [trim_row(row) for row in top_penalty_rows],
        "top_credible_shape_rows": [trim_row(row) for row in top_credible_rows],
        "public055_penalty_proxy_pass_rows": [trim_row(row) for row in penalty_pass_rows[:30]],
        "credible_public055_penalty_proxy_pass_rows": [trim_row(row) for row in credible_pass_rows[:30]],
        "scan_errors": errors,
        "decision": decision,
        "no_predictions_created": True,
        "no_feature_matrix_created": True,
        "no_base_prediction_artifact_created": True,
        "no_submission_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_action": True,
        "no_dacon_submission": True,
    }


def trim_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "id",
        "filename",
        "local_loss",
        "public_score",
        "public_minus_local",
        "stable_gap_projection",
        "penalty_proxy_projection",
        "penalty_proxy_gap_to_public055",
        "stable_shape_label",
        "public055_penalty_proxy_pass",
        "credible_public055_penalty_proxy_pass",
        "legacy_public_guided_family",
        "mean_abs_delta_vs_id342",
        "max_abs_delta_vs_id342",
        "max_target_mean_abs_delta_vs_id342",
        "max_abs_mean_rate_shift",
        "extreme_probability_rate",
    ]
    return {key: row.get(key) for key in keys if key in row}


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision = report["decision"]
    lines = [
        "# Post-ID342 public-gap penalty proxy redesign",
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
        f"- Stable-gap LOO MAE: `{decision['stable_gap_loo_error']['mae']}`",
        f"- Penalty-proxy LOO MAE: `{decision['penalty_proxy_loo_error']['mae']}`",
        f"- Stable-gap pairwise agreement: `{decision['stable_gap_pairwise_agreement']}`",
        f"- Penalty-proxy pairwise agreement: `{decision['penalty_proxy_pairwise_agreement']}`",
        f"- Public-0.55 penalty proxy pass count: `{decision['public055_penalty_proxy_pass_count']}`",
        f"- Credible Public-0.55 penalty proxy pass count: `{decision['credible_public055_penalty_proxy_pass_count']}`",
        f"- Validation redesign gate pass: `{decision['validation_redesign_gate_pass']}`",
        f"- Kaggle now: `{decision['run_kaggle_now']}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Penalty Weights",
        "",
        "| feature | weight |",
        "|---|---:|",
    ]
    for feature, weight in decision["global_penalty_weights"].items():
        lines.append(f"| `{feature}` | `{weight:.9f}` |")
    lines.extend(
        ["", "## Top Proxy Rows", "", "| id | proxy | gap to 0.55 | label | file |", "|---:|---:|---:|---|---|"]
    )
    for row in report["top_penalty_proxy_rows"][:12]:
        lines.append(
            f"| `{row['id']}` | `{row['penalty_proxy_projection']:.9f}` | "
            f"`{row['penalty_proxy_gap_to_public055']:.9f}` | `{row['stable_shape_label']}` | "
            f"`{row['filename']}` |"
        )
    lines.extend(["", "## Top Credible Shape Rows", "", "| id | proxy | gap to 0.55 | file |", "|---:|---:|---:|---|"])
    for row in report["top_credible_shape_rows"][:12]:
        lines.append(
            f"| `{row['id']}` | `{row['penalty_proxy_projection']:.9f}` | "
            f"`{row['penalty_proxy_gap_to_public055']:.9f}` | `{row['filename']}` |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


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
    decision = report["decision"]
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(
        "Credible penalty proxy passes: "
        f"{decision['credible_public055_penalty_proxy_pass_count']} | "
        f"validation_redesign_gate_pass={decision['validation_redesign_gate_pass']}"
    )


if __name__ == "__main__":
    main()
