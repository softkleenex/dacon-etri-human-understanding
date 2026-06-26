#!/usr/bin/env python3
"""Diagnose public-score risk from target joint-distribution geometry.

This is a report-only validation diagnostic. It uses submission CSVs with real
public scores and compares each test prediction matrix against the train-label
joint structure: target means, target correlations, prediction sharpness, and
same-subject temporal volatility. It creates no submission CSV, no base
prediction artifact, no notebook/history artifact, no submission-log row, and no
DACON submission.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
SUBMISSIONS_DIR = ROOT / "submissions"
LOG_PATH = SUBMISSIONS_DIR / "submission_log.csv"
TRAIN_PATH = ROOT / "data" / "processed" / "df_train_raw.parquet"
TEST_PATH = ROOT / "data" / "processed" / "df_test_raw.parquet"
KST = timezone(timedelta(hours=9))

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
RECENT_ANCHOR_IDS = [178, 287, 329, 341]
PUBLIC_SCORE_MAX_PLAUSIBLE = 2.0
EPS = 1e-6


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def load_submission(path: Path, reference_key: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted({"subject_id", "sleep_date", "lifelog_date", *TARGETS} - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    key = df[["subject_id", "sleep_date", "lifelog_date"]].copy()
    key["sleep_date"] = pd.to_datetime(key["sleep_date"]).dt.date
    key["lifelog_date"] = pd.to_datetime(key["lifelog_date"]).dt.date
    ref = reference_key.copy()
    ref["sleep_date"] = pd.to_datetime(ref["sleep_date"]).dt.date
    ref["lifelog_date"] = pd.to_datetime(ref["lifelog_date"]).dt.date
    if not key.equals(ref):
        merged = ref.merge(df, on=["subject_id", "sleep_date", "lifelog_date"], how="left", validate="one_to_one")
        if merged[TARGETS].isna().any().any():
            raise ValueError(f"{path} does not align with reference test rows")
        return merged[["subject_id", "sleep_date", "lifelog_date", *TARGETS]]
    return df[["subject_id", "sleep_date", "lifelog_date", *TARGETS]]


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(matrix, k=1)
    return matrix[idx]


def correlation_matrix(values: pd.DataFrame) -> np.ndarray:
    corr = values.corr().to_numpy(dtype=float)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def binary_temporal_volatility(frame: pd.DataFrame, value_cols: list[str]) -> float:
    rows = frame[["subject_id", "sleep_date", *value_cols]].copy()
    rows["sleep_date"] = pd.to_datetime(rows["sleep_date"])
    diffs = []
    for _subject, group in rows.sort_values(["subject_id", "sleep_date"]).groupby("subject_id"):
        if len(group) <= 1:
            continue
        arr = group[value_cols].to_numpy(dtype=float)
        diffs.append(np.abs(np.diff(arr, axis=0)).mean())
    return float(np.mean(diffs)) if diffs else float("nan")


def prediction_entropy(arr: np.ndarray) -> float:
    clipped = np.clip(arr, EPS, 1 - EPS)
    entropy = -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))
    return float(np.mean(entropy))


def expected_pair11_matrix(probs: np.ndarray) -> np.ndarray:
    n_targets = probs.shape[1]
    out = np.zeros((n_targets, n_targets), dtype=float)
    for i in range(n_targets):
        for j in range(n_targets):
            out[i, j] = float(np.mean(probs[:, i] * probs[:, j]))
    return out


def train_reference_stats(train: pd.DataFrame) -> dict[str, Any]:
    label = train[["subject_id", "sleep_date", *TARGETS]].copy()
    label["sleep_date"] = pd.to_datetime(label["sleep_date"])
    values = label[TARGETS].astype(float)
    corr = correlation_matrix(values)
    means = values.mean().to_numpy(dtype=float)
    pair11 = expected_pair11_matrix(values.to_numpy(dtype=float))
    return {
        "target_means": dict(zip(TARGETS, means, strict=True)),
        "target_std": dict(zip(TARGETS, values.std(ddof=0).to_numpy(dtype=float), strict=True)),
        "corr_matrix": corr,
        "corr_upper": upper_triangle_values(corr),
        "pair11_matrix": pair11,
        "pair11_upper": upper_triangle_values(pair11),
        "temporal_volatility": binary_temporal_volatility(label, TARGETS),
    }


def candidate_metrics(
    row: pd.Series,
    submission: pd.DataFrame,
    train_stats: dict[str, Any],
    anchor_predictions: dict[int, np.ndarray],
) -> dict[str, Any]:
    probs = np.clip(submission[TARGETS].astype(float).to_numpy(), EPS, 1 - EPS)
    means = probs.mean(axis=0)
    corr = correlation_matrix(pd.DataFrame(probs, columns=TARGETS))
    pair11 = expected_pair11_matrix(probs)
    train_means = np.asarray(list(train_stats["target_means"].values()), dtype=float)
    train_corr_upper = np.asarray(train_stats["corr_upper"], dtype=float)
    train_pair_upper = np.asarray(train_stats["pair11_upper"], dtype=float)
    temporal_vol = binary_temporal_volatility(submission[["subject_id", "sleep_date", *TARGETS]], TARGETS)
    metrics = {
        "id": int(row["id_num"]),
        "submission_file": row["submission_file"],
        "local_cv": safe_float(row.get("local_cv_score_num")),
        "public_lb": safe_float(row.get("public_lb_score_num")),
        "status_clean": row.get("status_clean"),
        "target_mean_l1_to_train": float(np.mean(np.abs(means - train_means))),
        "target_mean_max_abs_to_train": float(np.max(np.abs(means - train_means))),
        "corr_l2_to_train": float(np.sqrt(np.mean((upper_triangle_values(corr) - train_corr_upper) ** 2))),
        "corr_l1_to_train": float(np.mean(np.abs(upper_triangle_values(corr) - train_corr_upper))),
        "pair11_l1_to_train": float(np.mean(np.abs(upper_triangle_values(pair11) - train_pair_upper))),
        "temporal_volatility": temporal_vol,
        "temporal_volatility_ratio_to_train": float(temporal_vol / train_stats["temporal_volatility"]),
        "prediction_entropy": prediction_entropy(probs),
        "prediction_sharpness": float(np.mean(np.abs(probs - 0.5))),
        "extreme_rate_01_99": float(np.mean((probs <= 0.01) | (probs >= 0.99))),
        "extreme_rate_05_95": float(np.mean((probs <= 0.05) | (probs >= 0.95))),
        "target_means": dict(zip(TARGETS, means, strict=True)),
        "test_corr_upper": dict(
            zip(
                [f"{a}_{b}" for i, a in enumerate(TARGETS) for b in TARGETS[i + 1 :]],
                upper_triangle_values(corr),
                strict=True,
            )
        ),
    }
    for anchor_id, anchor_probs in anchor_predictions.items():
        if anchor_id == metrics["id"]:
            continue
        metrics[f"mean_abs_delta_vs_id{anchor_id}"] = float(np.mean(np.abs(probs - anchor_probs)))
        metrics[f"max_abs_delta_vs_id{anchor_id}"] = float(np.max(np.abs(probs - anchor_probs)))
    if metrics["local_cv"] is not None and metrics["public_lb"] is not None:
        metrics["public_minus_local"] = float(metrics["public_lb"] - metrics["local_cv"])
    return metrics


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return None
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return None
    xr = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=float)
    return pearson_corr(xr, yr)


def metric_correlations(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "local_cv",
        "target_mean_l1_to_train",
        "target_mean_max_abs_to_train",
        "corr_l2_to_train",
        "corr_l1_to_train",
        "pair11_l1_to_train",
        "temporal_volatility_ratio_to_train",
        "prediction_entropy",
        "prediction_sharpness",
        "extreme_rate_01_99",
        "extreme_rate_05_95",
        "mean_abs_delta_vs_id178",
        "max_abs_delta_vs_id178",
    ]
    public = np.asarray([row.get("public_lb", np.nan) for row in candidate_rows], dtype=float)
    gap = np.asarray([row.get("public_minus_local", np.nan) for row in candidate_rows], dtype=float)
    out = []
    for metric in metrics:
        values = np.asarray([row.get(metric, np.nan) for row in candidate_rows], dtype=float)
        out.append(
            {
                "metric": metric,
                "n": int(np.isfinite(values).sum()),
                "pearson_vs_public": pearson_corr(values, public),
                "spearman_vs_public": spearman_corr(values, public),
                "pearson_vs_public_minus_local": pearson_corr(values, gap),
                "spearman_vs_public_minus_local": spearman_corr(values, gap),
            }
        )
    return sorted(
        out,
        key=lambda row: max(
            abs(row["pearson_vs_public"] or 0.0),
            abs(row["spearman_vs_public"] or 0.0),
            abs(row["pearson_vs_public_minus_local"] or 0.0),
            abs(row["spearman_vs_public_minus_local"] or 0.0),
        ),
        reverse=True,
    )


def build_report() -> dict[str, Any]:
    log = pd.read_csv(LOG_PATH)
    log["id_num"] = pd.to_numeric(log["id"], errors="coerce")
    for col in ["local_cv_score_num", "public_lb_score_num"]:
        log[col] = pd.to_numeric(log[col], errors="coerce")
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)
    reference_key = test[["subject_id", "sleep_date", "lifelog_date"]].copy()
    train_stats = train_reference_stats(train)
    scored = log[
        log["id_num"].notna()
        & log["public_lb_score_num"].notna()
        & log["public_lb_score_num"].gt(0)
        & log["public_lb_score_num"].lt(PUBLIC_SCORE_MAX_PLAUSIBLE)
        & log["submission_file"].notna()
    ].copy()
    candidate_rows = []
    skipped = []
    raw_predictions: dict[int, np.ndarray] = {}
    loaded_submissions: dict[int, pd.DataFrame] = {}
    for _idx, row in scored.iterrows():
        candidate_id = int(row["id_num"])
        path = SUBMISSIONS_DIR / str(row["submission_file"])
        if not path.exists():
            skipped.append({"id": candidate_id, "submission_file": row["submission_file"], "reason": "missing_file"})
            continue
        try:
            submission = load_submission(path, reference_key)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"id": candidate_id, "submission_file": row["submission_file"], "reason": str(exc)})
            continue
        loaded_submissions[candidate_id] = submission
        raw_predictions[candidate_id] = submission[TARGETS].astype(float).to_numpy()
    anchor_predictions = {
        anchor_id: raw_predictions[anchor_id] for anchor_id in RECENT_ANCHOR_IDS if anchor_id in raw_predictions
    }
    for _idx, row in scored.iterrows():
        candidate_id = int(row["id_num"])
        if candidate_id not in loaded_submissions:
            continue
        candidate_rows.append(candidate_metrics(row, loaded_submissions[candidate_id], train_stats, anchor_predictions))
    correlations = metric_correlations(candidate_rows)
    top_by_public = sorted(candidate_rows, key=lambda row: row["public_lb"])[:10]
    recent_anchor_rows = [row for row in candidate_rows if row["id"] in RECENT_ANCHOR_IDS]
    strong_signals = [
        row
        for row in correlations
        if max(
            abs(row["pearson_vs_public"] or 0.0),
            abs(row["spearman_vs_public"] or 0.0),
            abs(row["pearson_vs_public_minus_local"] or 0.0),
            abs(row["spearman_vs_public_minus_local"] or 0.0),
        )
        >= 0.45
        and row["n"] >= 10
    ]
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "public_joint_distribution_risk_post_id341",
        "purpose": "Test whether target joint-distribution geometry helps explain public-score risk after ID178/287/329.",
        "input_counts": {
            "submission_log_rows": int(len(log)),
            "plausible_public_rows": int(len(scored)),
            "loaded_submission_rows": int(len(candidate_rows)),
            "skipped_rows": int(len(skipped)),
        },
        "train_reference": {
            "target_means": train_stats["target_means"],
            "target_std": train_stats["target_std"],
            "temporal_volatility": train_stats["temporal_volatility"],
        },
        "metric_correlations": correlations,
        "strong_signal_metrics": strong_signals,
        "recent_anchor_rows": recent_anchor_rows,
        "top_public_rows": top_by_public,
        "skipped": skipped,
        "decision": {
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "joint_distribution_gate_useful": bool(strong_signals),
            "strong_signal_metric_names": [row["metric"] for row in strong_signals],
            "next_action": (
                "Use strong joint-distribution risk metrics as report-level candidate-selection warnings only; "
                "do not create a submission artifact from this diagnostic. If no robust signal appears, keep the "
                "Kaggle upstream rebuild as the main next action."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_dacon_submission_made": True,
    }


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(val):
        return ""
    return f"{val:.{digits}f}"


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Public joint-distribution risk — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, or DACON submission.",
        "",
        "## Input coverage",
        "",
        f"- Plausible public rows loaded: `{report['input_counts']['loaded_submission_rows']}` / `{report['input_counts']['plausible_public_rows']}`",
        f"- Skipped rows: `{report['input_counts']['skipped_rows']}`",
        f"- Train temporal volatility: `{report['train_reference']['temporal_volatility']:.6f}`",
        "",
        "## Metric correlations",
        "",
        "| metric | n | Pearson public | Spearman public | Pearson gap | Spearman gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["metric_correlations"]:
        lines.append(
            f"| `{row['metric']}` | {row['n']} | {fmt(row['pearson_vs_public'])} | "
            f"{fmt(row['spearman_vs_public'])} | {fmt(row['pearson_vs_public_minus_local'])} | "
            f"{fmt(row['spearman_vs_public_minus_local'])} |"
        )
    lines.extend(
        [
            "",
            "## Recent public anchors",
            "",
            "| id | local | public | mean L1 | corr L2 | pair11 L1 | vol ratio | entropy | sharp | Δ178 mean |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(report["recent_anchor_rows"], key=lambda item: item["id"]):
        lines.append(
            f"| {row['id']} | {fmt(row.get('local_cv'))} | {fmt(row.get('public_lb'))} | "
            f"{fmt(row.get('target_mean_l1_to_train'))} | {fmt(row.get('corr_l2_to_train'))} | "
            f"{fmt(row.get('pair11_l1_to_train'))} | {fmt(row.get('temporal_volatility_ratio_to_train'))} | "
            f"{fmt(row.get('prediction_entropy'))} | {fmt(row.get('prediction_sharpness'))} | "
            f"{fmt(row.get('mean_abs_delta_vs_id178'))} |"
        )
    lines.extend(
        [
            "",
            "## Best public rows by actual score",
            "",
            "| id | public | local | mean L1 | corr L2 | pair11 L1 | vol ratio | sharp | file |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["top_public_rows"]:
        lines.append(
            f"| {row['id']} | {fmt(row.get('public_lb'))} | {fmt(row.get('local_cv'))} | "
            f"{fmt(row.get('target_mean_l1_to_train'))} | {fmt(row.get('corr_l2_to_train'))} | "
            f"{fmt(row.get('pair11_l1_to_train'))} | {fmt(row.get('temporal_volatility_ratio_to_train'))} | "
            f"{fmt(row.get('prediction_sharpness'))} | `{row['submission_file']}` |"
        )
    lines.extend(["", "## Decision", ""])
    if report["strong_signal_metrics"]:
        lines.append(
            "- Strong warning metrics: " + ", ".join(f"`{row['metric']}`" for row in report["strong_signal_metrics"])
        )
    else:
        lines.append("- Strong warning metrics: _(none)_")
    lines.extend(
        [
            f"- Joint-distribution gate useful: `{report['decision']['joint_distribution_gate_useful']}`",
            f"- Create submission artifact: `{report['decision']['create_submission_artifact']}`",
            f"- Submit now: `{report['decision']['submit_now']}`",
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
        default=REPORTS_DIR / "public_joint_distribution_risk_post_id341_20260618.json",
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
