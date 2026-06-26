#!/usr/bin/env python3
"""Replay the A1 q240-free source stack inside outer folds.

This is a report-only diagnostic. It does not create a submission artifact.

Scope:
- Rebuild the ID238-like pair/logit source choice inside each outer fold.
- Refit the ID239/ID241-style q240-free calibrators on the outer-training rows.
- Replay the A1 cap-0.30 bridge from ID235 to those rebuilt q240-free sources.

Important limitation:
The upstream artifacts below ID238 (IDs 221/226/227/228/237) are still
pre-existing OOF products. This is therefore stricter than the prior A1 selector
replay, but still not a true model-level nested rebuild.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_id179_vs_id335_sequence_probe_review import (  # noqa: E402
    EPS,
    TARGETS,
    align,
    load_artifact,
    load_train_test,
    local_scores,
    robust_score,
    slice_summary,
    target_loss,
)
from scripts.diagnose_strict_demographic_feature_audit import make_loso_folds  # noqa: E402
from scripts.run_public_confirmed_anchor_bridge_probe_experiment import blend_values  # noqa: E402
from scripts.run_public_extreme_calibration_probe_experiment import (  # noqa: E402
    apply_quantile_calibrator,
    fit_quantile_calibrator,
)
from scripts.run_public_pair_logit_blend_probe_experiment import blend_values as blend_pair_values  # noqa: E402
from src.utils.validation import make_subject_hole_folds  # noqa: E402

REPORTS_DIR = ROOT / "reports"
PAIR_SCAN_PATH = REPORTS_DIR / "id221_237_narrow_pair_prob_logit_oracle_scan_20260616.json"
A1_REPLAY_PATH = REPORTS_DIR / "fold_internal_delta_capped_bridge_rebuild_post_id338_20260618.json"
OUTPUT_PATH = REPORTS_DIR / "a1_q240_free_nested_source_replay_post_id338_20260618.json"
KST = timezone(timedelta(hours=9))
PAIR_SOURCE_IDS = (221, 226, 227, 228, 237)
BASE_ID = 235
SOURCE_NAMES = ("id238_replay", "id239_iso_replay", "id241_q160_s4iso_replay")


@dataclass(frozen=True)
class PairChoice:
    kind: str
    id_a: int
    id_b: int
    weight_b: float
    train_loss: float
    train_pred: np.ndarray
    valid_pred: np.ndarray


@dataclass(frozen=True)
class BridgeChoice:
    source_name: str
    method: str
    alpha: float
    train_loss: float
    objective: float
    valid_pred: np.ndarray


def load_candidates(train: pd.DataFrame, candidate_ids: tuple[int, ...]) -> dict[int, pd.DataFrame]:
    return {candidate_id: align(load_artifact("oof", candidate_id), train) for candidate_id in candidate_ids}


def candidate_arrays(candidates: dict[int, pd.DataFrame], target: str) -> dict[int, np.ndarray]:
    return {
        candidate_id: np.clip(frame[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        for candidate_id, frame in candidates.items()
    }


def pair_weight_grid() -> tuple[float, ...]:
    scan = json.loads(PAIR_SCAN_PATH.read_text(encoding="utf-8"))
    weights = {round(float(value), 3) for value in np.linspace(0.0, 1.0, 101)}
    for target_info in scan["targets"].values():
        for row in target_info.get("top20", []):
            weights.add(round(float(row.get("w_b", 0.0)), 3))
        best = target_info.get("best", {})
        weights.add(round(float(best.get("w_b", 0.0)), 3))
    return tuple(sorted(weights))


def a1_alpha_grid() -> tuple[float, ...]:
    report = json.loads(A1_REPLAY_PATH.read_text(encoding="utf-8"))
    alphas: set[float] = set()
    for scenario in report["scenarios"]:
        if scenario["name"] == "no_q240_extremes":
            alphas.update(round(float(value), 3) for value in scenario["alphas"])
    if not alphas:
        alphas.update(round(float(value), 3) for value in np.linspace(0.0, 0.6, 25))
    return tuple(sorted(alphas))


def select_pair_source(
    *,
    y_train: np.ndarray,
    arrays: dict[int, np.ndarray],
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    weights: tuple[float, ...],
) -> PairChoice:
    pool: list[PairChoice] = []
    for candidate_id in PAIR_SOURCE_IDS:
        train_pred = arrays[candidate_id][train_idx]
        valid_pred = arrays[candidate_id][valid_idx]
        pool.append(
            PairChoice(
                kind="single",
                id_a=candidate_id,
                id_b=candidate_id,
                weight_b=0.0,
                train_loss=target_loss(y_train, train_pred),
                train_pred=train_pred,
                valid_pred=valid_pred,
            )
        )
    for id_a, id_b in combinations(PAIR_SOURCE_IDS, 2):
        a_train = arrays[id_a][train_idx]
        b_train = arrays[id_b][train_idx]
        a_valid = arrays[id_a][valid_idx]
        b_valid = arrays[id_b][valid_idx]
        for kind in ("prob_pair", "logit_pair"):
            for weight in weights:
                train_pred = blend_pair_values(a_train, b_train, weight, kind)
                pool.append(
                    PairChoice(
                        kind=kind,
                        id_a=id_a,
                        id_b=id_b,
                        weight_b=float(weight),
                        train_loss=target_loss(y_train, train_pred),
                        train_pred=train_pred,
                        valid_pred=blend_pair_values(a_valid, b_valid, weight, kind),
                    )
                )
    return min(
        pool,
        key=lambda row: (
            row.train_loss,
            row.kind != "single",
            abs(row.weight_b - 0.5),
            row.id_a,
            row.id_b,
        ),
    )


def fit_isotonic(train_values: np.ndarray, y_train: np.ndarray, values: np.ndarray) -> np.ndarray:
    model = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1.0 - EPS)
    model.fit(np.clip(train_values, EPS, 1.0 - EPS), y_train)
    return np.clip(model.predict(np.clip(values, EPS, 1.0 - EPS)), EPS, 1.0 - EPS)


def fit_quantile(train_values: np.ndarray, y_train: np.ndarray, values: np.ndarray, q: int) -> np.ndarray:
    calibrator = fit_quantile_calibrator(np.clip(train_values, EPS, 1.0 - EPS), y_train, q=q)
    return apply_quantile_calibrator(np.clip(values, EPS, 1.0 - EPS), calibrator)


def build_q240_free_sources(
    *,
    target: str,
    y_train: np.ndarray,
    pair_choice: PairChoice,
) -> dict[str, dict[str, np.ndarray]]:
    raw_train = pair_choice.train_pred
    raw_valid = pair_choice.valid_pred
    sources = {
        "id238_replay": {
            "train": raw_train,
            "valid": raw_valid,
        },
        "id239_iso_replay": {
            "train": raw_train,
            "valid": raw_valid,
        },
        "id241_q160_s4iso_replay": {
            "train": raw_train,
            "valid": raw_valid,
        },
    }
    if target != "S3":
        iso_train = fit_isotonic(raw_train, y_train, raw_train)
        iso_valid = fit_isotonic(raw_train, y_train, raw_valid)
        sources["id239_iso_replay"] = {"train": iso_train, "valid": iso_valid}
    if target in {"Q1", "Q2", "Q3", "S1", "S2"}:
        q_train = fit_quantile(raw_train, y_train, raw_train, q=160)
        q_valid = fit_quantile(raw_train, y_train, raw_valid, q=160)
        sources["id241_q160_s4iso_replay"] = {"train": q_train, "valid": q_valid}
    elif target == "S4":
        iso_train = fit_isotonic(raw_train, y_train, raw_train)
        iso_valid = fit_isotonic(raw_train, y_train, raw_valid)
        sources["id241_q160_s4iso_replay"] = {"train": iso_train, "valid": iso_valid}
    return sources


def select_bridge(
    *,
    y_train: np.ndarray,
    base_train: np.ndarray,
    base_valid: np.ndarray,
    source_variants: dict[str, dict[str, np.ndarray]],
    alphas: tuple[float, ...],
    cap: float,
    delta_penalty: float,
) -> BridgeChoice:
    pool = [
        BridgeChoice(
            source_name="base_235",
            method="base",
            alpha=0.0,
            train_loss=target_loss(y_train, base_train),
            objective=target_loss(y_train, base_train),
            valid_pred=base_valid,
        )
    ]
    for source_name, values in source_variants.items():
        for method in ("prob", "logit"):
            for alpha in alphas:
                if alpha <= 0.0:
                    continue
                train_pred = blend_values(base_train, values["train"], alpha=alpha, method=method)
                delta = train_pred - base_train
                if float(np.max(np.abs(delta))) > cap + 1e-12:
                    continue
                loss = target_loss(y_train, train_pred)
                mean_delta = float(np.mean(np.abs(delta)))
                objective = float(loss + delta_penalty * mean_delta)
                pool.append(
                    BridgeChoice(
                        source_name=source_name,
                        method=method,
                        alpha=float(alpha),
                        train_loss=loss,
                        objective=objective,
                        valid_pred=blend_values(base_valid, values["valid"], alpha=alpha, method=method),
                    )
                )
    return min(
        pool,
        key=lambda row: (
            row.objective,
            row.train_loss,
            row.source_name,
            row.method,
            row.alpha,
        ),
    )


def run_split(
    *,
    split_name: str,
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    candidates: dict[int, pd.DataFrame],
    pair_weights: tuple[float, ...],
    alphas: tuple[float, ...],
    cap: float,
    delta_penalty: float,
) -> dict[str, Any]:
    selected = train[["subject_id", "date"]].copy()
    for target in TARGETS:
        selected[target] = np.nan
    pair_choices: dict[str, list[dict[str, Any]]] = {target: [] for target in TARGETS}
    bridge_choices: dict[str, list[dict[str, Any]]] = {target: [] for target in TARGETS}
    target_arrays = {target: candidate_arrays(candidates, target) for target in TARGETS}
    base_arrays = candidate_arrays({BASE_ID: candidates[BASE_ID]}, "Q1")
    del base_arrays  # explicit reminder that base arrays are target-specific below

    for fold_idx, (raw_train_idx, raw_valid_idx) in enumerate(folds):
        train_idx = np.asarray(raw_train_idx, dtype=int)
        valid_idx = np.asarray(raw_valid_idx, dtype=int)
        for target in TARGETS:
            y_train = train[target].iloc[train_idx].astype(int).to_numpy()
            arrays = target_arrays[target]
            pair_choice = select_pair_source(
                y_train=y_train,
                arrays=arrays,
                train_idx=train_idx,
                valid_idx=valid_idx,
                weights=pair_weights,
            )
            sources = build_q240_free_sources(target=target, y_train=y_train, pair_choice=pair_choice)
            base_values = arrays[BASE_ID]
            bridge_choice = select_bridge(
                y_train=y_train,
                base_train=base_values[train_idx],
                base_valid=base_values[valid_idx],
                source_variants=sources,
                alphas=alphas,
                cap=cap,
                delta_penalty=delta_penalty,
            )
            selected.loc[valid_idx, target] = bridge_choice.valid_pred
            pair_choices[target].append(
                {
                    "fold": int(fold_idx),
                    "kind": pair_choice.kind,
                    "id_a": int(pair_choice.id_a),
                    "id_b": int(pair_choice.id_b),
                    "weight_b": float(pair_choice.weight_b),
                    "train_loss": float(pair_choice.train_loss),
                    "valid_loss": target_loss(
                        train[target].iloc[valid_idx].astype(int).to_numpy(),
                        pair_choice.valid_pred,
                    ),
                }
            )
            bridge_choices[target].append(
                {
                    "fold": int(fold_idx),
                    "source_name": bridge_choice.source_name,
                    "method": bridge_choice.method,
                    "alpha": float(bridge_choice.alpha),
                    "train_loss": float(bridge_choice.train_loss),
                    "valid_loss": target_loss(
                        train[target].iloc[valid_idx].astype(int).to_numpy(),
                        bridge_choice.valid_pred,
                    ),
                    "objective": float(bridge_choice.objective),
                }
            )
    if selected[TARGETS].isna().any().any():
        raise ValueError(f"{split_name} failed to fill all OOF predictions.")
    overall, target_scores = local_scores(train, selected)
    slices = slice_summary(train, selected)
    bridge_changed = sum(
        1
        for choices in bridge_choices.values()
        for choice in choices
        if choice["source_name"] != "base_235" and choice["alpha"] > 0
    )
    bridge_total = sum(len(choices) for choices in bridge_choices.values())
    return {
        "split": split_name,
        "overall": overall,
        "target_scores": target_scores,
        "robust_score": robust_score(overall, slices),
        "slice_summary": {
            "worst_subject": slices["worst_subject"],
            "worst_subject_loss": slices["worst_subject_loss"],
            "worst_date_bin": slices["worst_date_bin"],
            "worst_date_loss": slices["worst_date_loss"],
            "worst_target": slices["worst_target"],
            "worst_target_loss": slices["worst_target_loss"],
            "late_minus_early": slices["late_minus_early"],
        },
        "bridge_changed_choice_rate": float(bridge_changed / bridge_total) if bridge_total else 0.0,
        "pair_choice_counts": {
            target: dict(
                Counter(
                    f"{choice['kind']}:{choice['id_a']:03d}:{choice['id_b']:03d}:w{choice['weight_b']:.3f}"
                    for choice in choices
                )
            )
            for target, choices in pair_choices.items()
        },
        "bridge_choice_counts": {
            target: dict(
                Counter(f"{choice['source_name']}:{choice['method']}:a{choice['alpha']:.3f}" for choice in choices)
            )
            for target, choices in bridge_choices.items()
        },
        "pair_choices": pair_choices,
        "bridge_choices": bridge_choices,
    }


def base_summary(train: pd.DataFrame, candidates: dict[int, pd.DataFrame]) -> dict[str, Any]:
    base = candidates[BASE_ID]
    overall, target_scores = local_scores(train, base)
    slices = slice_summary(train, base)
    return {
        "id": BASE_ID,
        "overall": overall,
        "target_scores": target_scores,
        "robust_score": robust_score(overall, slices),
        "slice_summary": {
            "worst_subject": slices["worst_subject"],
            "worst_subject_loss": slices["worst_subject_loss"],
            "worst_target": slices["worst_target"],
            "worst_target_loss": slices["worst_target_loss"],
        },
    }


def reference_qfree() -> dict[str, dict[str, float]]:
    report = json.loads(A1_REPLAY_PATH.read_text(encoding="utf-8"))
    out: dict[str, dict[str, float]] = {}
    for split_name, rows in report["splits"].items():
        matches = [row for row in rows if row["scenario"]["name"] == "no_q240_extremes"]
        best = min(matches, key=lambda row: (float(row["overall"]), float(row["robust_score"])))
        out[split_name] = {
            "overall": float(best["overall"]),
            "robust_score": float(best["robust_score"]),
            "cap": float(best["scenario"]["cap"]),
        }
    return out


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# A1 q240-free nested source replay — post ID338",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. No CSV, notebook, base artifact, submission-log row, or DACON submission was created.",
        "",
        "## Results",
        "",
        "| split | replay overall | replay robust | selector-only qfree | delta vs selector-only | worst target | worst subject | changed |",
        "|---|---:|---:|---:|---:|---|---|---:|",
    ]
    references = report["selector_only_q240_free_reference"]
    for row in report["splits"]:
        ref = references[row["split"]]
        slices = row["slice_summary"]
        lines.append(
            f"| `{row['split']}` | {row['overall']:.6f} | {row['robust_score']:.6f} | "
            f"{ref['overall']:.6f} | {row['overall'] - ref['overall']:+.6f} | "
            f"{slices['worst_target']} `{slices['worst_target_loss']:.6f}` | "
            f"{slices['worst_subject']} `{slices['worst_subject_loss']:.6f}` | "
            f"{row['bridge_changed_choice_rate']:.3f} |"
        )
    base = report["base_235_summary"]
    lines.extend(
        [
            "",
            "## Base reference",
            "",
            f"ID235 base OOF is `{base['overall']:.6f}` / robust `{base['robust_score']:.6f}`.",
            "",
            "## Predominant choices",
            "",
            "### Pair source choices",
            "",
        ]
    )
    for row in report["splits"]:
        lines.append(f"#### `{row['split']}`")
        for target, counts in row["pair_choice_counts"].items():
            top = Counter(counts).most_common(2)
            top_text = ", ".join(f"`{key}` x{value}" for key, value in top)
            lines.append(f"- `{target}`: {top_text}")
        lines.extend(["", "Bridge choices:"])
        for target, counts in row["bridge_choice_counts"].items():
            top = Counter(counts).most_common(2)
            top_text = ", ".join(f"`{key}` x{value}" for key, value in top)
            lines.append(f"- `{target}`: {top_text}")
        lines.append("")
    decision = report["decision"]
    lines.extend(
        [
            "## Decision",
            "",
            decision["reason"],
            "",
            f"- Source replay survives: `{decision['source_replay_survives']}`",
            f"- Public 0.58 plausible: `{decision['public_058_plausible']}`",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(output_json: Path) -> dict[str, Any]:
    train, _test = load_train_test()
    candidate_ids = tuple(sorted(set(PAIR_SOURCE_IDS) | {BASE_ID}))
    candidates = load_candidates(train, candidate_ids)
    pair_weights = pair_weight_grid()
    alphas = a1_alpha_grid()
    split_results = [
        run_split(
            split_name="subject_hole_source_replay",
            train=train,
            folds=make_subject_hole_folds(train),
            candidates=candidates,
            pair_weights=pair_weights,
            alphas=alphas,
            cap=0.30,
            delta_penalty=0.02,
        ),
        run_split(
            split_name="loso_source_replay",
            train=train,
            folds=make_loso_folds(train),
            candidates=candidates,
            pair_weights=pair_weights,
            alphas=alphas,
            cap=0.30,
            delta_penalty=0.02,
        ),
    ]
    base = base_summary(train, candidates)
    references = reference_qfree()
    # Map the selector-only names to this diagnostic's split names for display.
    selector_reference = {
        "subject_hole_source_replay": references["subject_hole_fold_internal"],
        "loso_source_replay": references["loso_fold_internal"],
    }
    survives = all(row["overall"] < base["overall"] - 0.02 for row in split_results)
    reason = (
        "The q240-free A1 stack still improves over ID235 after refitting the ID238 pair source and ID239/241 "
        "calibration inside outer folds, but it remains far weaker than the selector-only qfree reference. "
        "This justifies a deeper source rebuild, not a submission."
        if survives
        else "The q240-free A1 stack does not preserve enough signal after outer-fold source replay. Deprioritize "
        "this lane before any Kaggle-heavy model-level rebuild."
    )
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "a1_q240_free_nested_source_replay_post_id338",
        "input_reports": {
            "pair_scan": str(PAIR_SCAN_PATH.relative_to(ROOT)),
            "a1_selector_replay": str(A1_REPLAY_PATH.relative_to(ROOT)),
        },
        "limitations": [
            "Upstream artifacts 221/226/227/228/237 are still pre-existing OOF products.",
            "Source selection and calibration are learned on each outer-training fold; no DACON-facing test predictions are produced.",
            "This is a screening diagnostic before any Kaggle-heavy model-level source rebuild.",
        ],
        "base_235_summary": base,
        "selector_only_q240_free_reference": selector_reference,
        "pair_weight_grid_size": len(pair_weights),
        "a1_alpha_grid_size": len(alphas),
        "splits": split_results,
        "decision": {
            "source_replay_survives": bool(survives),
            "public_058_plausible": False,
            "create_submission_artifact": False,
            "next_action": (
                "If compute is available, rebuild upstream sources 221/226/227/228/237 model-level on Kaggle; "
                "otherwise use this replay as the strongest local-only screen."
            ),
            "reason": reason,
        },
        "no_submission_created": True,
        "no_dacon_submission_made": True,
    }
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_json.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    report = run(args.output_json)
    print(f"Wrote {args.output_json.relative_to(ROOT)}")
    print(f"Wrote {args.output_json.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
