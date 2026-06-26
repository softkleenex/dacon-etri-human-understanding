#!/usr/bin/env python3
"""Outer-fold diagnostic for public-structure calibration selection.

This script does not create a submission artifact. It asks whether calibration
choices for a base public-structure artifact survive when the target fold is not
used to choose the calibration method.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_public_tempered_extreme_calibration_probe_experiment import load_artifact_pair  # noqa: E402
from scripts.run_public_temporal_anchor_probe_experiment import (  # noqa: E402
    add_date_dt,
    make_interleaved_subject_folds,
)
from scripts.run_strict_raw_feature_experiment import REPORTS_DIR, TARGETS, build_feature_frames  # noqa: E402

EPS = 1e-6


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    method: str
    temperature: float | None = None
    shift: float | None = None
    c_value: float | None = None
    q: int | None = None


def logit(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)
    return np.log(arr / (1.0 - arr))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def apply_temp_shift(values: np.ndarray | pd.Series, temperature: float, shift: float) -> np.ndarray:
    return np.clip(sigmoid(logit(values) / temperature + shift), EPS, 1.0 - EPS)


def beta2_features(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)
    return np.column_stack([np.log(arr), np.log(1.0 - arr)])


def platt_features(values: np.ndarray | pd.Series) -> np.ndarray:
    return logit(values).reshape(-1, 1)


def fit_quantile_calibrator(values: np.ndarray, y: np.ndarray, q: int) -> dict[str, Any]:
    values = np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, q + 1)))
    if len(edges) <= 2:
        return {"edges": edges.tolist(), "bin_values": [float(np.mean(y))], "global_mean": float(np.mean(y))}
    bins = np.digitize(values, edges[1:-1], right=True)
    bin_values = []
    for bin_idx in range(len(edges) - 1):
        mask = bins == bin_idx
        if mask.any():
            bin_values.append(float((y[mask].sum() + 0.5) / (mask.sum() + 1.0)))
        else:
            bin_values.append(float(np.mean(y)))
    return {"edges": edges.tolist(), "bin_values": bin_values, "global_mean": float(np.mean(y))}


def apply_quantile_calibrator(values: np.ndarray, calibrator: dict[str, Any]) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)
    edges = np.asarray(calibrator["edges"], dtype=float)
    bin_values = np.asarray(calibrator["bin_values"], dtype=float)
    if len(edges) <= 2 or len(bin_values) == 1:
        return np.full(len(values), float(calibrator["global_mean"]), dtype=float)
    bins = np.digitize(values, edges[1:-1], right=True)
    bins = np.clip(bins, 0, len(bin_values) - 1)
    return np.clip(bin_values[bins], EPS, 1.0 - EPS)


def fit_predict_candidate(
    spec: CandidateSpec,
    train_values: np.ndarray,
    y_train: np.ndarray,
    query_values: np.ndarray,
) -> np.ndarray:
    train_values = np.clip(np.asarray(train_values, dtype=float), EPS, 1.0 - EPS)
    query_values = np.clip(np.asarray(query_values, dtype=float), EPS, 1.0 - EPS)
    if spec.method == "identity":
        return query_values
    if spec.method == "temp_shift":
        if spec.temperature is None or spec.shift is None:
            raise ValueError(f"Missing temp_shift parameters for {spec.name}")
        return apply_temp_shift(query_values, spec.temperature, spec.shift)
    if len(np.unique(y_train)) < 2:
        return np.full(len(query_values), float(np.mean(y_train)))
    if spec.method == "beta2":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=float(spec.c_value), solver="liblinear", random_state=43),
        )
        model.fit(beta2_features(train_values), y_train)
        return np.clip(model.predict_proba(beta2_features(query_values))[:, 1], EPS, 1.0 - EPS)
    if spec.method == "platt":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=float(spec.c_value), solver="liblinear", random_state=43),
        )
        model.fit(platt_features(train_values), y_train)
        return np.clip(model.predict_proba(platt_features(query_values))[:, 1], EPS, 1.0 - EPS)
    if spec.method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1.0 - EPS)
        model.fit(train_values, y_train)
        return np.clip(model.predict(query_values), EPS, 1.0 - EPS)
    if spec.method == "quantile":
        if spec.q is None:
            raise ValueError(f"Missing quantile q for {spec.name}")
        calibrator = fit_quantile_calibrator(train_values, y_train, spec.q)
        return apply_quantile_calibrator(query_values, calibrator)
    raise ValueError(f"Unknown candidate method: {spec.method}")


def make_candidates(include_extreme: bool) -> list[CandidateSpec]:
    candidates = [CandidateSpec(name="identity", method="identity")]
    for temperature in [0.65, 0.75, 0.85, 0.95, 1.05, 1.15]:
        for shift in [-0.25, -0.15, -0.08, 0.0, 0.08, 0.15, 0.25]:
            candidates.append(
                CandidateSpec(
                    name=f"temp_t{temperature:g}_s{shift:g}",
                    method="temp_shift",
                    temperature=temperature,
                    shift=shift,
                )
            )
    for c_value in [0.25, 0.5, 1.0, 2.0, 4.0]:
        candidates.append(CandidateSpec(name=f"beta2_C{c_value:g}", method="beta2", c_value=c_value))
        candidates.append(CandidateSpec(name=f"platt_C{c_value:g}", method="platt", c_value=c_value))
    if include_extreme:
        candidates.append(CandidateSpec(name="isotonic", method="isotonic"))
        for q in [5, 10, 20, 40]:
            candidates.append(CandidateSpec(name=f"quantile{q}", method="quantile", q=q))
    return candidates


def candidate_inner_oof_loss(
    spec: CandidateSpec,
    values: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[float, np.ndarray]:
    preds = np.zeros(len(train_idx), dtype=float)
    for out_pos, inner_fold in enumerate(sorted(np.unique(fold_ids[train_idx]))):
        _ = out_pos  # keeps loop intent explicit for future per-fold diagnostics
        fit_idx = train_idx[fold_ids[train_idx] != inner_fold]
        valid_idx = train_idx[fold_ids[train_idx] == inner_fold]
        if len(valid_idx) == 0:
            continue
        if len(fit_idx) == 0:
            preds[np.isin(train_idx, valid_idx)] = np.mean(y[train_idx])
            continue
        pred = fit_predict_candidate(spec, values[fit_idx], y[fit_idx], values[valid_idx])
        preds[np.isin(train_idx, valid_idx)] = pred
    return float(log_loss(y[train_idx], preds, labels=[0, 1])), preds


def select_candidate(
    candidates: list[CandidateSpec],
    values: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    train_idx: np.ndarray,
    guard_margin: float,
) -> tuple[CandidateSpec, list[dict[str, Any]], float, float]:
    scored: list[tuple[float, CandidateSpec]] = []
    for spec in candidates:
        loss, _preds = candidate_inner_oof_loss(spec, values, y, fold_ids, train_idx)
        scored.append((loss, spec))
    scored.sort(key=lambda item: (item[0], item[1].name))
    identity_loss = next(loss for loss, spec in scored if spec.name == "identity")
    best_loss, best_spec = scored[0]
    if best_loss >= identity_loss - guard_margin:
        chosen = next(spec for loss, spec in scored if spec.name == "identity")
    else:
        chosen = best_spec
    top = [{"loss": float(loss), "candidate": asdict(spec)} for loss, spec in scored[:10]]
    return chosen, top, float(best_loss), float(identity_loss)


def evaluate_outer_selection(
    candidates: list[CandidateSpec],
    values: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    guard_margin: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    outer_oof = np.zeros(len(y), dtype=float)
    fold_details: list[dict[str, Any]] = []
    all_idx = np.arange(len(y))
    for outer_fold in sorted(np.unique(fold_ids)):
        train_idx = all_idx[fold_ids != outer_fold]
        valid_idx = all_idx[fold_ids == outer_fold]
        chosen, top, best_loss, identity_loss = select_candidate(
            candidates=candidates,
            values=values,
            y=y,
            fold_ids=fold_ids,
            train_idx=train_idx,
            guard_margin=guard_margin,
        )
        pred = fit_predict_candidate(chosen, values[train_idx], y[train_idx], values[valid_idx])
        outer_oof[valid_idx] = pred
        fold_details.append(
            {
                "outer_fold": int(outer_fold),
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
                "chosen": asdict(chosen),
                "inner_best_loss": best_loss,
                "inner_identity_loss": identity_loss,
                "valid_loss": float(log_loss(y[valid_idx], pred, labels=[0, 1])),
                "top_inner_candidates": top,
            }
        )
    return np.clip(outer_oof, EPS, 1.0 - EPS), fold_details


def run_diagnostic(base_id: int, fixed_id: int | None, output_path: Path) -> dict[str, Any]:
    train, test, _metrics_sorted, _sample_original = build_feature_frames()
    base_oof, _base_test, base_source = load_artifact_pair(train, test, base_id)
    fixed_oof = None
    fixed_source = None
    if fixed_id is not None:
        fixed_oof, _fixed_test, fixed_source = load_artifact_pair(train, test, fixed_id)
    fold_ids = make_interleaved_subject_folds(add_date_dt(train))
    pools = {
        "narrow": make_candidates(include_extreme=False),
        "wide": make_candidates(include_extreme=True),
    }
    guard_margins = [0.0, 0.001, 0.003]
    report: dict[str, Any] = {
        "diagnostic": "public_outer_calibration_selection",
        "base_id": base_id,
        "base_source": base_source,
        "fixed_id": fixed_id,
        "fixed_source": fixed_source,
        "validation_lens": "outer_interleaved_within_subject_blocked_selection",
        "guard_margins": guard_margins,
        "pool_sizes": {name: len(cands) for name, cands in pools.items()},
        "targets": {},
    }
    base_scores: dict[str, float] = {}
    fixed_scores: dict[str, float] = {}
    pool_means: dict[str, dict[str, float]] = {pool_name: {} for pool_name in pools}
    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_scores[target] = float(log_loss(y, values, labels=[0, 1]))
        target_report: dict[str, Any] = {
            "base_loss": base_scores[target],
            "pools": {},
        }
        if fixed_oof is not None:
            fixed_values = np.clip(fixed_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
            fixed_scores[target] = float(log_loss(y, fixed_values, labels=[0, 1]))
            target_report["fixed_loss"] = fixed_scores[target]
        for pool_name, candidates in pools.items():
            pool_report: dict[str, Any] = {}
            for margin in guard_margins:
                outer_oof, fold_details = evaluate_outer_selection(candidates, values, y, fold_ids, margin)
                loss = float(log_loss(y, outer_oof, labels=[0, 1]))
                pool_report[f"guard_{margin:g}"] = {
                    "loss": loss,
                    "improvement_vs_base": base_scores[target] - loss,
                    "improvement_vs_fixed": (fixed_scores[target] - loss) if fixed_oof is not None else None,
                    "chosen_counts": pd.Series([item["chosen"]["name"] for item in fold_details])
                    .value_counts()
                    .to_dict(),
                    "fold_details": fold_details,
                }
                pool_means[pool_name][f"guard_{margin:g}_{target}"] = loss
            full_train_choice, full_top, best_loss, identity_loss = select_candidate(
                candidates=candidates,
                values=values,
                y=y,
                fold_ids=fold_ids,
                train_idx=np.arange(len(y)),
                guard_margin=0.0,
            )
            pool_report["full_train_selection_for_possible_test"] = {
                "chosen": asdict(full_train_choice),
                "inner_best_loss": best_loss,
                "inner_identity_loss": identity_loss,
                "top_inner_candidates": full_top,
            }
            target_report["pools"][pool_name] = pool_report
        report["targets"][target] = target_report
    report["base_mean"] = float(np.mean(list(base_scores.values())))
    if fixed_scores:
        report["fixed_mean"] = float(np.mean(list(fixed_scores.values())))
    for pool_name in pools:
        report[f"{pool_name}_means"] = {
            f"guard_{margin:g}": float(
                np.mean(
                    [report["targets"][target]["pools"][pool_name][f"guard_{margin:g}"]["loss"] for target in TARGETS]
                )
            )
            for margin in guard_margins
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", type=int, default=168)
    parser.add_argument("--fixed-id", type=int, default=169)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "id168_outer_calibration_selection_diagnostic.json",
    )
    args = parser.parse_args()
    report = run_diagnostic(args.base_id, args.fixed_id, args.output)
    summary = {
        "base_mean": report["base_mean"],
        "fixed_mean": report.get("fixed_mean"),
        "narrow_means": report["narrow_means"],
        "wide_means": report["wide_means"],
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
