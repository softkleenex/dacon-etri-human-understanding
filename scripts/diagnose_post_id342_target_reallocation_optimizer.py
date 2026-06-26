#!/usr/bin/env python3
"""Post-ID342 target reallocation optimizer.

This report-only diagnostic asks a narrower question after the stable-envelope
multi-source ceiling: can the same movement budget be reallocated target by
target, rather than by whole-source blends, to get closer to Public 0.55?

It scans strict nonlegacy OOF sources, builds target-specific ID342-convex
options, and runs a beam search over one option per target under the same
public-stable prediction-shape envelope.

It creates no predictions, feature matrices, base prediction artifacts,
submission CSVs, submission-log rows, Kaggle actions, or DACON submissions.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
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
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_target_reallocation_optimizer_20260625.json"
RISK_LENS_REPORT = REPORTS_DIR / "post_id342_public_gap_risk_lens_20260624.json"
KST = timezone(timedelta(hours=9))

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
ANCHOR_ID = 342
PUBLIC_055 = 0.55
EPS = 1e-6
SOURCE_ALPHA_GRID = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14]
CONFIRMED_FAILED_PUBLIC_IDS = {287, 329}
STRICT_EXCLUSION_TOKENS = (
    "frontier",
    "zerowide",
    "scorepressure",
    "direct_lite",
    "delta_pair",
    "riskpen",
    "post271",
    "public_probe",
    "outer_supported",
    "extreme_public",
    "targetwise_delta",
    "convex_bridge",
    "second_stage",
    "oracle",
)
TARGET_DESCRIPTIONS = {
    "Q1": "subjective sleep quality vs personal average",
    "Q2": "subjective physical fatigue before sleep vs personal average",
    "Q3": "subjective stress before sleep vs personal average",
    "S1": "objective total sleep time guideline",
    "S2": "objective sleep efficiency guideline",
    "S3": "objective sleep onset latency guideline",
    "S4": "objective WASO / wakefulness after sleep onset guideline",
}


@dataclass(frozen=True)
class TargetOption:
    target_idx: int
    target: str
    source_id: int | None
    source_file: str
    alpha: float
    target_loss: float
    delta_target_loss_vs_anchor: float
    delta_col: np.ndarray
    mean_abs_delta: float
    max_abs_delta: float


@dataclass(frozen=True)
class BeamState:
    pred: np.ndarray
    choices: tuple[TargetOption, ...]
    local_loss: float
    projected_public: float
    metrics: dict[str, Any]


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


def excluded_source(path: Path) -> bool:
    source_id = candidate_id_from_path(path)
    lower = path.name.lower()
    return source_id in CONFIRMED_FAILED_PUBLIC_IDS or any(token in lower for token in STRICT_EXCLUSION_TOKENS)


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


def target_losses(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        target: float(log_loss(y_true[:, idx].astype(int), pred[:, idx], labels=[0, 1]))
        for idx, target in enumerate(TARGETS)
    }


def mean_loss(y_true: np.ndarray, pred: np.ndarray) -> tuple[float, dict[str, float]]:
    losses = target_losses(y_true, pred)
    return float(np.mean(list(losses.values()))), losses


def envelope_from_report(path: Path = RISK_LENS_REPORT) -> tuple[dict[str, float], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["stable_shape_envelope"], float(payload["decision"]["stable_gap_median"])


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


def make_anchor_option(target_idx: int, anchor_losses: dict[str, float], row_count: int) -> TargetOption:
    target = TARGETS[target_idx]
    return TargetOption(
        target_idx=target_idx,
        target=target,
        source_id=None,
        source_file="id342_anchor",
        alpha=0.0,
        target_loss=anchor_losses[target],
        delta_target_loss_vs_anchor=0.0,
        delta_col=np.zeros(row_count, dtype=float),
        mean_abs_delta=0.0,
        max_abs_delta=0.0,
    )


def scan_target_options(
    train: pd.DataFrame,
    y_true: np.ndarray,
    anchor: np.ndarray,
    anchor_losses: dict[str, float],
    per_target_limit: int,
) -> tuple[dict[str, list[TargetOption]], list[dict[str, Any]], list[dict[str, str]]]:
    options = {target: [make_anchor_option(idx, anchor_losses, len(anchor))] for idx, target in enumerate(TARGETS)}
    excluded_rows: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    for path in sorted(BASE_PREDICTION_DIR.glob("base_oof_id_*.csv")):
        try:
            source_id = candidate_id_from_path(path)
            if source_id == ANCHOR_ID:
                continue
            if excluded_source(path):
                excluded_rows.append({"source_id": str(source_id), "source_file": path.name})
                continue
            source_pred = load_oof(path, train)
            for target_idx, target in enumerate(TARGETS):
                y_col = y_true[:, target_idx].astype(int)
                for alpha in SOURCE_ALPHA_GRID:
                    pred_col = np.clip(
                        (1.0 - alpha) * anchor[:, target_idx] + alpha * source_pred[:, target_idx],
                        EPS,
                        1.0 - EPS,
                    )
                    loss = float(log_loss(y_col, pred_col, labels=[0, 1]))
                    delta_loss = float(loss - anchor_losses[target])
                    if delta_loss >= -0.00001:
                        continue
                    delta_col = pred_col - anchor[:, target_idx]
                    options[target].append(
                        TargetOption(
                            target_idx=target_idx,
                            target=target,
                            source_id=source_id,
                            source_file=path.name,
                            alpha=alpha,
                            target_loss=loss,
                            delta_target_loss_vs_anchor=delta_loss,
                            delta_col=delta_col,
                            mean_abs_delta=float(np.mean(np.abs(delta_col))),
                            max_abs_delta=float(np.max(np.abs(delta_col))),
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": rel(path), "error": f"{type(exc).__name__}: {exc}"})

    for target in TARGETS:
        anchor_option = options[target][0]
        improving = sorted(options[target][1:], key=lambda item: item.target_loss)[:per_target_limit]
        options[target] = [anchor_option, *improving]
    option_rows = []
    for target in TARGETS:
        for option in options[target][1:]:
            option_rows.append(option_summary(option))
    return options, option_rows, errors


def evaluate_state(
    pred: np.ndarray,
    choices: tuple[TargetOption, ...],
    y_true: np.ndarray,
    anchor: np.ndarray,
    train_target_rate: np.ndarray,
    envelope: dict[str, float],
    stable_gap_median: float,
) -> BeamState | None:
    metrics = shape_metrics(pred, anchor, train_target_rate)
    if not envelope_pass(metrics, envelope):
        return None
    local_loss, losses = mean_loss(y_true, pred)
    metrics["target_losses"] = losses
    return BeamState(
        pred=pred,
        choices=choices,
        local_loss=local_loss,
        projected_public=float(local_loss + stable_gap_median),
        metrics=metrics,
    )


def run_beam_search(
    options: dict[str, list[TargetOption]],
    anchor: np.ndarray,
    y_true: np.ndarray,
    train_target_rate: np.ndarray,
    envelope: dict[str, float],
    stable_gap_median: float,
    beam_width: int,
) -> tuple[list[BeamState], list[dict[str, Any]]]:
    start = evaluate_state(anchor.copy(), (), y_true, anchor, train_target_rate, envelope, stable_gap_median)
    if start is None:
        raise RuntimeError("ID342 anchor unexpectedly fails the stable envelope")
    beam = [start]
    seen: list[BeamState] = [start]
    summaries = []
    for target_idx, target in enumerate(TARGETS):
        expanded = []
        for state in beam:
            base_pred = state.pred
            for option in options[target]:
                next_pred = base_pred.copy()
                next_pred[:, target_idx] = np.clip(anchor[:, target_idx] + option.delta_col, EPS, 1.0 - EPS)
                next_state = evaluate_state(
                    next_pred,
                    (*state.choices, option),
                    y_true,
                    anchor,
                    train_target_rate,
                    envelope,
                    stable_gap_median,
                )
                if next_state is not None:
                    expanded.append(next_state)
        beam = sorted(expanded, key=lambda item: item.local_loss)[:beam_width]
        seen.extend(beam)
        summaries.append(
            {
                "target": target,
                "options_considered": len(options[target]),
                "expanded_envelope_pass_states": len(expanded),
                "beam_best_local_loss": beam[0].local_loss if beam else None,
                "beam_best_projected_public": beam[0].projected_public if beam else None,
            }
        )
        if not beam:
            break
    return sorted(seen, key=lambda item: item.local_loss), summaries


def option_summary(option: TargetOption) -> dict[str, Any]:
    return {
        "target": option.target,
        "target_description": TARGET_DESCRIPTIONS[option.target],
        "source_id": option.source_id,
        "source_file": option.source_file,
        "alpha": option.alpha,
        "target_loss": option.target_loss,
        "delta_target_loss_vs_id342": option.delta_target_loss_vs_anchor,
        "mean_abs_delta_vs_id342": option.mean_abs_delta,
        "max_abs_delta_vs_id342": option.max_abs_delta,
    }


def summarize_state(
    state: BeamState,
    anchor_loss: float,
    anchor_target_losses: dict[str, float],
    required_local: float,
    stable_gap_median: float,
) -> dict[str, Any]:
    target_rows = []
    for option in state.choices:
        target_loss = state.metrics["target_losses"][option.target]
        target_rows.append(
            {
                **option_summary(option),
                "selected_target_loss": target_loss,
                "selected_delta_target_loss_vs_id342": float(target_loss - anchor_target_losses[option.target]),
            }
        )
    return {
        "local_loss": state.local_loss,
        "delta_local_vs_id342": float(state.local_loss - anchor_loss),
        "projected_public_at_stable_gap": state.projected_public,
        "gap_to_public055_at_stable_gap": float(state.projected_public - PUBLIC_055),
        "public055_local_threshold_pass": bool(state.local_loss <= required_local),
        "required_local_for_public055": required_local,
        "stable_gap_median": stable_gap_median,
        "active_target_count": sum(1 for row in target_rows if row["source_id"] is not None),
        "target_choices": target_rows,
        **state.metrics,
    }


def target_gap_rows(
    anchor_target_losses: dict[str, float],
    best_row: dict[str, Any],
    required_local: float,
) -> list[dict[str, Any]]:
    current_losses = best_row["target_losses"]
    remaining_total_loss_reduction = 7.0 * (best_row["local_loss"] - required_local)
    rows = []
    for target in TARGETS:
        rows.append(
            {
                "target": target,
                "target_description": TARGET_DESCRIPTIONS[target],
                "anchor_loss": anchor_target_losses[target],
                "best_loss": current_losses[target],
                "delta_loss_vs_id342": float(current_losses[target] - anchor_target_losses[target]),
                "remaining_if_single_target_only": remaining_total_loss_reduction,
                "single_target_required_loss": float(current_losses[target] - remaining_total_loss_reduction),
                "target_mean_abs_delta_vs_id342": best_row["target_mean_abs_delta_vs_id342"][target],
            }
        )
    return sorted(rows, key=lambda item: item["best_loss"], reverse=True)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    train = load_train_sorted()
    y_true = train[TARGETS].astype(int).to_numpy()
    train_target_rate = y_true.mean(axis=0)
    envelope, stable_gap_median = envelope_from_report(args.risk_lens_report)
    required_local = PUBLIC_055 - stable_gap_median
    anchor_path = sorted(BASE_PREDICTION_DIR.glob(f"base_oof_id_{ANCHOR_ID}_*.csv"))[0]
    anchor_pred = load_oof(anchor_path, train)
    anchor_loss, anchor_target_losses = mean_loss(y_true, anchor_pred)
    options, option_rows, errors = scan_target_options(
        train, y_true, anchor_pred, anchor_target_losses, args.per_target_option_limit
    )
    states, beam_summaries = run_beam_search(
        options,
        anchor_pred,
        y_true,
        train_target_rate,
        envelope,
        stable_gap_median,
        args.beam_width,
    )
    top_rows = [
        summarize_state(state, anchor_loss, anchor_target_losses, required_local, stable_gap_median)
        for state in states[: args.report_row_count]
    ]
    best = top_rows[0]
    gap_rows = target_gap_rows(anchor_target_losses, best, required_local)
    decision = {
        "strict_failed_public_ids_excluded": sorted(CONFIRMED_FAILED_PUBLIC_IDS),
        "strict_exclusion_tokens": STRICT_EXCLUSION_TOKENS,
        "target_options_scanned": len(option_rows),
        "beam_states_retained": len(states),
        "anchor_local_loss": anchor_loss,
        "anchor_projected_public_at_stable_gap": float(anchor_loss + stable_gap_median),
        "stable_gap_median": stable_gap_median,
        "required_local_for_public055": required_local,
        "best_local_loss": best["local_loss"],
        "best_delta_local_vs_id342": best["delta_local_vs_id342"],
        "best_projected_public_at_stable_gap": best["projected_public_at_stable_gap"],
        "best_gap_to_public055_at_stable_gap": best["gap_to_public055_at_stable_gap"],
        "best_active_target_count": best["active_target_count"],
        "public055_envelope_rows": sum(1 for row in top_rows if row["public055_local_threshold_pass"]),
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "dacon_submit_now": False,
        "artifact_review_required": bool(best["public055_local_threshold_pass"]),
        "next_action": (
            "If no Public-0.55 row appears, treat target reallocation as closed for strict existing sources and "
            "prioritize genuinely new S4/S3/Q1 signal or validation-gap redesign."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_target_reallocation_optimizer",
        "purpose": "Strict report-only target-wise source reallocation under the public-stable shape envelope.",
        "parameters": {
            "source_alpha_grid": SOURCE_ALPHA_GRID,
            "per_target_option_limit": args.per_target_option_limit,
            "beam_width": args.beam_width,
        },
        "stable_shape_envelope": envelope,
        "anchor": {
            "id": ANCHOR_ID,
            "path": rel(anchor_path),
            "local_loss": anchor_loss,
            "target_losses": anchor_target_losses,
        },
        "decision": decision,
        "beam_summaries": beam_summaries,
        "top_target_options": sorted(option_rows, key=lambda row: row["delta_target_loss_vs_id342"])[:100],
        "top_rows": top_rows,
        "target_gap_rows": gap_rows,
        "scan_errors": errors[:20],
        "no_predictions_created": True,
        "no_feature_matrix_created": True,
        "no_base_prediction_artifact_created": True,
        "no_submission_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_action": True,
        "no_dacon_submission": True,
    }


def choice_summary(row: dict[str, Any]) -> str:
    parts = []
    for choice in row["target_choices"]:
        if choice["source_id"] is None:
            continue
        parts.append(f"{choice['target']}:ID{choice['source_id']}@{choice['alpha']:.3f}")
    return ", ".join(parts) if parts else "anchor_only"


def row_line(row: dict[str, Any]) -> str:
    return (
        f"| `{row['local_loss']:.6f}` | `{row['delta_local_vs_id342']:+.6f}` | "
        f"`{row['projected_public_at_stable_gap']:.6f}` | `{row['gap_to_public055_at_stable_gap']:+.6f}` | "
        f"`{row['active_target_count']}` | `{row['mean_abs_delta_vs_id342']:.6f}` | "
        f"`{row['max_abs_delta_vs_id342']:.6f}` | {choice_summary(row)} |"
    )


def gap_line(row: dict[str, Any]) -> str:
    return (
        f"| `{row['target']}` | {row['target_description']} | `{row['best_loss']:.6f}` | "
        f"`{row['delta_loss_vs_id342']:+.6f}` | `{row['target_mean_abs_delta_vs_id342']:.6f}` | "
        f"`{row['single_target_required_loss']:.6f}` |"
    )


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision = report["decision"]
    lines = [
        "# Post-ID342 target reallocation optimizer",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- Target options scanned: `{decision['target_options_scanned']}`",
        f"- Beam states retained: `{decision['beam_states_retained']}`",
        f"- Anchor projected public: `{decision['anchor_projected_public_at_stable_gap']:.9f}`",
        f"- Best projected public: `{decision['best_projected_public_at_stable_gap']:.9f}`",
        f"- Best local delta vs ID342: `{decision['best_delta_local_vs_id342']:+.9f}`",
        f"- Best gap to Public 0.55: `{decision['best_gap_to_public055_at_stable_gap']:+.9f}`",
        f"- Best active target count: `{decision['best_active_target_count']}`",
        f"- Public-0.55 envelope rows in report: `{decision['public055_envelope_rows']}`",
        f"- Artifact review required: `{decision['artifact_review_required']}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Top Rows",
        "",
        "| local | delta local | projected public | gap 0.55 | active targets | mean move | max move | choices |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    lines.extend(row_line(row) for row in report["top_rows"][:25])
    lines.extend(
        [
            "",
            "## Target Gap Rows",
            "",
            "| target | domain | best loss | delta vs ID342 | target mean move | loss needed if alone |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    lines.extend(gap_line(row) for row in report["target_gap_rows"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--risk-lens-report", type=Path, default=RISK_LENS_REPORT)
    parser.add_argument("--per-target-option-limit", type=int, default=60)
    parser.add_argument("--beam-width", type=int, default=400)
    parser.add_argument("--report-row-count", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    args.risk_lens_report = (
        args.risk_lens_report if args.risk_lens_report.is_absolute() else ROOT / args.risk_lens_report
    )
    report = build_report(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, output)
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(f"Best projected public: {report['decision']['best_projected_public_at_stable_gap']}")
    print(f"Public-0.55 envelope rows: {report['decision']['public055_envelope_rows']}")
    print(f"DACON submit now: {report['decision']['dacon_submit_now']}")


if __name__ == "__main__":
    main()
