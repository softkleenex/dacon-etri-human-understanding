#!/usr/bin/env python3
"""Post-ID342 stable-envelope multi-source optimizer.

This report-only diagnostic tests whether multiple nonlegacy ID342-neighborhood
OOF directions can combine inside the public-stable prediction-shape envelope.
It first scans every nonlegacy base OOF source for an envelope-safe single-source
micro-blend, then runs a bounded beam search over tiny convex weights around
ID342.

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
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_stable_envelope_multisource_optimizer_20260625.json"
RISK_LENS_REPORT = REPORTS_DIR / "post_id342_public_gap_risk_lens_20260624.json"
MICROBLEND_REPORT = REPORTS_DIR / "post_id342_stable_envelope_microblend_scan_20260625.json"
KST = timezone(timedelta(hours=9))

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
ANCHOR_ID = 342
PUBLIC_055 = 0.55
EPS = 1e-6
DEFAULT_SOURCE_LIMIT = 18
DEFAULT_BEAM_WIDTH = 50
DEFAULT_STEP_COUNT = 36
DEFAULT_WEIGHT_STEP = 0.005
DEFAULT_MAX_TOTAL_WEIGHT = 0.18
SOURCE_ALPHA_GRID = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14]
LEGACY_PUBLIC_GUIDED_TOKENS = (
    "frontier",
    "zerowide",
    "scorepressure",
    "direct_lite",
    "delta_pair",
    "riskpen",
    "post271",
)


@dataclass(frozen=True)
class Source:
    source_id: int
    path: Path
    single_best_projected_public: float
    single_best_alpha: float
    single_best_local_loss: float
    single_best_mean_move: float
    single_best_max_move: float


@dataclass(frozen=True)
class BeamState:
    weights: tuple[int, ...]
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


def scan_gate_sources(
    train: pd.DataFrame,
    y_true: np.ndarray,
    anchor: np.ndarray,
    anchor_loss: float,
    train_target_rate: np.ndarray,
    envelope: dict[str, float],
    stable_gap_median: float,
    limit: int,
) -> tuple[list[Source], list[np.ndarray], list[dict[str, Any]]]:
    source_rows: list[dict[str, Any]] = []
    source_predictions_by_id: dict[int, np.ndarray] = {}
    for source_path in sorted(BASE_PREDICTION_DIR.glob("base_oof_id_*.csv")):
        source_id = candidate_id_from_path(source_path)
        if source_id == ANCHOR_ID or legacy_public_guided(source_path.name):
            continue
        source_pred = load_oof(source_path, train)
        source_predictions_by_id[source_id] = source_pred
        best_row: dict[str, Any] | None = None
        for alpha in SOURCE_ALPHA_GRID:
            pred = np.clip((1.0 - alpha) * anchor + alpha * source_pred, EPS, 1.0 - EPS)
            metrics = shape_metrics(pred, anchor, train_target_rate)
            if not envelope_pass(metrics, envelope):
                continue
            local_loss, _target_losses = mean_loss(y_true, pred)
            row = {
                "source_id": source_id,
                "source_file": source_path.name,
                "source_path": rel(source_path),
                "family": source_family_label(source_path.name),
                "single_best_alpha": alpha,
                "single_best_local_loss": local_loss,
                "single_best_delta_local_vs_id342": float(local_loss - anchor_loss),
                "single_best_projected_public": float(local_loss + stable_gap_median),
                "single_best_mean_move": metrics["mean_abs_delta_vs_id342"],
                "single_best_max_move": metrics["max_abs_delta_vs_id342"],
            }
            if best_row is None or row["single_best_projected_public"] < best_row["single_best_projected_public"]:
                best_row = row
        if best_row and best_row["single_best_delta_local_vs_id342"] < -0.00005:
            source_rows.append(best_row)

    ordered = sorted(source_rows, key=lambda item: item["single_best_projected_public"])
    sources: list[Source] = []
    source_predictions: list[np.ndarray] = []
    for row in ordered[:limit]:
        source_path = BASE_PREDICTION_DIR / row["source_file"]
        sources.append(
            Source(
                source_id=int(row["source_id"]),
                path=source_path,
                single_best_projected_public=float(row["single_best_projected_public"]),
                single_best_alpha=float(row["single_best_alpha"]),
                single_best_local_loss=float(row["single_best_local_loss"]),
                single_best_mean_move=float(row["single_best_mean_move"]),
                single_best_max_move=float(row["single_best_max_move"]),
            )
        )
        source_predictions.append(source_predictions_by_id[int(row["source_id"])])
    return sources, source_predictions, ordered


def blend_from_weights(
    anchor: np.ndarray, source_deltas: np.ndarray, weights: tuple[int, ...], step: float
) -> np.ndarray:
    weight_array = np.asarray(weights, dtype=float) * step
    return np.clip(anchor + np.tensordot(weight_array, source_deltas, axes=(0, 0)), EPS, 1.0 - EPS)


def evaluate_state(
    weights: tuple[int, ...],
    step: float,
    anchor: np.ndarray,
    source_deltas: np.ndarray,
    y_true: np.ndarray,
    train_target_rate: np.ndarray,
    envelope: dict[str, float],
    stable_gap_median: float,
) -> BeamState | None:
    pred = blend_from_weights(anchor, source_deltas, weights, step)
    metrics = shape_metrics(pred, anchor, train_target_rate)
    if not envelope_pass(metrics, envelope):
        return None
    local_loss, target_losses = mean_loss(y_true, pred)
    metrics["target_losses"] = target_losses
    return BeamState(
        weights=weights,
        local_loss=local_loss,
        projected_public=float(local_loss + stable_gap_median),
        metrics=metrics,
    )


def source_family_label(filename: str) -> str:
    lower = filename.lower()
    if "post285" in lower:
        return "post285_fixedtemp"
    if "post282" in lower:
        return "post282_tempguard"
    if "post278" in lower:
        return "post278_gapaware"
    if "post274" in lower:
        return "post274_refined"
    return "other"


def nonzero_weights(state: BeamState, sources: list[Source], step: float) -> list[dict[str, Any]]:
    rows = []
    for idx, ticks in enumerate(state.weights):
        if ticks <= 0:
            continue
        source = sources[idx]
        rows.append(
            {
                "source_id": source.source_id,
                "source_file": source.path.name,
                "source_path": rel(source.path),
                "family": source_family_label(source.path.name),
                "weight": float(ticks * step),
                "single_best_alpha": source.single_best_alpha,
                "single_best_projected_public": source.single_best_projected_public,
            }
        )
    return rows


def summarize_state(
    state: BeamState,
    sources: list[Source],
    step: float,
    anchor_loss: float,
    required_local: float,
) -> dict[str, Any]:
    weights = nonzero_weights(state, sources, step)
    family_weights: dict[str, float] = {}
    for row in weights:
        family_weights[row["family"]] = family_weights.get(row["family"], 0.0) + row["weight"]
    return {
        "local_loss": state.local_loss,
        "delta_local_vs_id342": float(state.local_loss - anchor_loss),
        "projected_public_at_stable_gap": state.projected_public,
        "gap_to_public055_at_stable_gap": float(state.projected_public - PUBLIC_055),
        "public055_local_threshold_pass": bool(state.local_loss <= required_local),
        "total_source_weight": float(sum(row["weight"] for row in weights)),
        "active_source_count": len(weights),
        "source_weights": weights,
        "family_weights": family_weights,
        **state.metrics,
    }


def run_beam_search(
    anchor: np.ndarray,
    source_predictions: list[np.ndarray],
    y_true: np.ndarray,
    train_target_rate: np.ndarray,
    envelope: dict[str, float],
    stable_gap_median: float,
    args: argparse.Namespace,
) -> tuple[list[BeamState], list[dict[str, Any]]]:
    source_deltas = np.stack([pred - anchor for pred in source_predictions], axis=0)
    source_count = len(source_predictions)
    max_ticks = int(round(args.max_total_weight / args.weight_step))
    zero = tuple(0 for _ in range(source_count))
    start = evaluate_state(
        zero,
        args.weight_step,
        anchor,
        source_deltas,
        y_true,
        train_target_rate,
        envelope,
        stable_gap_median,
    )
    if start is None:
        raise RuntimeError("ID342 anchor unexpectedly fails the stable envelope")

    beam = [start]
    best_seen: dict[tuple[int, ...], BeamState] = {zero: start}
    iteration_summaries = []
    for iteration in range(1, args.step_count + 1):
        expanded: dict[tuple[int, ...], BeamState] = {}
        for state in beam:
            if sum(state.weights) >= max_ticks:
                continue
            for source_idx in range(source_count):
                next_weights = list(state.weights)
                next_weights[source_idx] += 1
                next_tuple = tuple(next_weights)
                if sum(next_tuple) > max_ticks or next_tuple in best_seen or next_tuple in expanded:
                    continue
                next_state = evaluate_state(
                    next_tuple,
                    args.weight_step,
                    anchor,
                    source_deltas,
                    y_true,
                    train_target_rate,
                    envelope,
                    stable_gap_median,
                )
                if next_state is not None:
                    expanded[next_tuple] = next_state
        if not expanded:
            iteration_summaries.append(
                {
                    "iteration": iteration,
                    "expanded_envelope_pass_states": 0,
                    "beam_best_local_loss": beam[0].local_loss,
                    "beam_best_projected_public": beam[0].projected_public,
                    "stopped": True,
                }
            )
            break
        for key, value in expanded.items():
            best_seen[key] = value
        beam = sorted(expanded.values(), key=lambda item: item.local_loss)[: args.beam_width]
        iteration_summaries.append(
            {
                "iteration": iteration,
                "expanded_envelope_pass_states": len(expanded),
                "beam_best_local_loss": beam[0].local_loss,
                "beam_best_projected_public": beam[0].projected_public,
                "beam_best_total_weight": float(sum(beam[0].weights) * args.weight_step),
                "stopped": False,
            }
        )
    return sorted(best_seen.values(), key=lambda item: item.local_loss), iteration_summaries


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    train = load_train_sorted()
    y_true = train[TARGETS].astype(int).to_numpy()
    train_target_rate = y_true.mean(axis=0)
    envelope, stable_gap_median = envelope_from_report(args.risk_lens_report)
    required_local = PUBLIC_055 - stable_gap_median
    anchor_path = sorted(BASE_PREDICTION_DIR.glob(f"base_oof_id_{ANCHOR_ID}_*.csv"))[0]
    anchor_pred = load_oof(anchor_path, train)
    anchor_loss, anchor_target_losses = mean_loss(y_true, anchor_pred)
    sources, source_predictions, scanned_source_rows = scan_gate_sources(
        train,
        y_true,
        anchor_pred,
        anchor_loss,
        train_target_rate,
        envelope,
        stable_gap_median,
        args.source_limit,
    )
    states, iteration_summaries = run_beam_search(
        anchor_pred,
        source_predictions,
        y_true,
        train_target_rate,
        envelope,
        stable_gap_median,
        args,
    )
    top_rows = [
        summarize_state(state, sources, args.weight_step, anchor_loss, required_local)
        for state in states[: args.report_row_count]
    ]
    best = top_rows[0]
    multi_source_rows = [row for row in top_rows if row["active_source_count"] >= 2]
    best_multi = multi_source_rows[0] if multi_source_rows else None
    source_rows = [
        {
            "source_id": source.source_id,
            "source_file": source.path.name,
            "source_path": rel(source.path),
            "family": source_family_label(source.path.name),
            "single_best_alpha": source.single_best_alpha,
            "single_best_local_loss": source.single_best_local_loss,
            "single_best_projected_public": source.single_best_projected_public,
            "single_best_mean_move": source.single_best_mean_move,
            "single_best_max_move": source.single_best_max_move,
        }
        for source in sources
    ]
    family_count = len({row["family"] for row in source_rows})
    decision = {
        "candidate_source_count": len(sources),
        "scanned_nonlegacy_gate_source_count": len(scanned_source_rows),
        "candidate_source_family_count": family_count,
        "beam_states_retained": len(states),
        "anchor_local_loss": anchor_loss,
        "anchor_projected_public_at_stable_gap": float(anchor_loss + stable_gap_median),
        "stable_gap_median": stable_gap_median,
        "required_local_for_public055": required_local,
        "best_local_loss": best["local_loss"],
        "best_delta_local_vs_id342": best["delta_local_vs_id342"],
        "best_projected_public_at_stable_gap": best["projected_public_at_stable_gap"],
        "best_gap_to_public055_at_stable_gap": best["gap_to_public055_at_stable_gap"],
        "best_total_source_weight": best["total_source_weight"],
        "best_active_source_count": best["active_source_count"],
        "best_multi_source_projected_public": None
        if best_multi is None
        else best_multi["projected_public_at_stable_gap"],
        "public055_envelope_rows": sum(1 for row in top_rows if row["public055_local_threshold_pass"]),
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "dacon_submit_now": False,
        "artifact_review_required": bool(best["public055_local_threshold_pass"]),
        "next_action": (
            "If this report still misses Public 0.55 under the stable-gap projection, keep DACON closed and "
            "treat the result as validation-compression evidence. If it passes 0.55, perform a separate "
            "artifact-contract review before creating any CSV."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_stable_envelope_multisource_optimizer",
        "purpose": (
            "Report-only beam search over nonlegacy single-source gate directions around ID342 under the "
            "public-stable shape envelope."
        ),
        "parameters": {
            "source_limit": args.source_limit,
            "beam_width": args.beam_width,
            "step_count": args.step_count,
            "weight_step": args.weight_step,
            "max_total_weight": args.max_total_weight,
            "source_alpha_grid": SOURCE_ALPHA_GRID,
        },
        "input_reports": {
            "risk_lens_report": rel(args.risk_lens_report),
            "microblend_report": rel(args.microblend_report),
        },
        "stable_shape_envelope": envelope,
        "anchor": {
            "id": ANCHOR_ID,
            "path": rel(anchor_path),
            "local_loss": anchor_loss,
            "target_losses": anchor_target_losses,
        },
        "candidate_sources": source_rows,
        "top_scanned_nonlegacy_gate_sources": scanned_source_rows[: args.report_row_count],
        "decision": decision,
        "iteration_summaries": iteration_summaries,
        "top_rows": top_rows,
        "no_predictions_created": True,
        "no_feature_matrix_created": True,
        "no_base_prediction_artifact_created": True,
        "no_submission_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_action": True,
        "no_dacon_submission": True,
    }


def weight_summary(row: dict[str, Any]) -> str:
    parts = [f"ID{weight['source_id']}:{weight['weight']:.3f}" for weight in row["source_weights"][:8]]
    return ", ".join(parts) if parts else "anchor_only"


def row_line(row: dict[str, Any]) -> str:
    return (
        f"| `{row['local_loss']:.6f}` | `{row['delta_local_vs_id342']:+.6f}` | "
        f"`{row['projected_public_at_stable_gap']:.6f}` | "
        f"`{row['gap_to_public055_at_stable_gap']:+.6f}` | `{row['total_source_weight']:.3f}` | "
        f"`{row['active_source_count']}` | `{row['mean_abs_delta_vs_id342']:.6f}` | "
        f"`{row['max_abs_delta_vs_id342']:.6f}` | {weight_summary(row)} |"
    )


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision = report["decision"]
    lines = [
        "# Post-ID342 stable-envelope multi-source optimizer",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Decision",
        "",
        f"- Candidate sources: `{decision['candidate_source_count']}`",
        f"- Scanned nonlegacy gate sources: `{decision['scanned_nonlegacy_gate_source_count']}`",
        f"- Candidate source families: `{decision['candidate_source_family_count']}`",
        f"- Beam states retained: `{decision['beam_states_retained']}`",
        f"- Anchor projected public: `{decision['anchor_projected_public_at_stable_gap']:.9f}`",
        f"- Best projected public: `{decision['best_projected_public_at_stable_gap']:.9f}`",
        f"- Best local delta vs ID342: `{decision['best_delta_local_vs_id342']:+.9f}`",
        f"- Best gap to Public 0.55: `{decision['best_gap_to_public055_at_stable_gap']:+.9f}`",
        f"- Best total source weight: `{decision['best_total_source_weight']:.3f}`",
        f"- Best active source count: `{decision['best_active_source_count']}`",
        f"- Public-0.55 envelope rows in report: `{decision['public055_envelope_rows']}`",
        f"- Artifact review required: `{decision['artifact_review_required']}`",
        f"- DACON submit now: `{decision['dacon_submit_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Top Rows",
        "",
        "| local | delta local | projected public | gap 0.55 | total weight | active sources | mean move | max move | weights |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    lines.extend(row_line(row) for row in report["top_rows"][:25])
    lines.extend(
        [
            "",
            "## Candidate Sources",
            "",
            "| source | family | single alpha | single projected public | file |",
            "|---:|---|---:|---:|---|",
        ]
    )
    for row in report["candidate_sources"]:
        lines.append(
            f"| `ID{row['source_id']}` | `{row['family']}` | `{row['single_best_alpha']:.3f}` | "
            f"`{row['single_best_projected_public']:.6f}` | `{row['source_file']}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--risk-lens-report", type=Path, default=RISK_LENS_REPORT)
    parser.add_argument("--microblend-report", type=Path, default=MICROBLEND_REPORT)
    parser.add_argument("--source-limit", type=int, default=DEFAULT_SOURCE_LIMIT)
    parser.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH)
    parser.add_argument("--step-count", type=int, default=DEFAULT_STEP_COUNT)
    parser.add_argument("--weight-step", type=float, default=DEFAULT_WEIGHT_STEP)
    parser.add_argument("--max-total-weight", type=float, default=DEFAULT_MAX_TOTAL_WEIGHT)
    parser.add_argument("--report-row-count", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    args.risk_lens_report = (
        args.risk_lens_report if args.risk_lens_report.is_absolute() else ROOT / args.risk_lens_report
    )
    args.microblend_report = (
        args.microblend_report if args.microblend_report.is_absolute() else ROOT / args.microblend_report
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
