#!/usr/bin/env python3
"""Fast full-OOF scan for stability-selected LGBM/XGB overlays.

This report-only diagnostic tests the most transferable idea from the external
Public 0.5917 notebook: shallow LGBM/XGB side models on fold-stability-selected
features, blended lightly into existing anchors. It intentionally does not
promote artifacts because the target-wise candidate choice is full-OOF.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_public_stability_selected_raw_overlay_experiment import (  # noqa: E402
    feature_matrix,
    select_stable_columns,
)
from scripts.run_public_tempered_extreme_calibration_probe_experiment import load_artifact_pair  # noqa: E402
from scripts.run_public_temporal_anchor_probe_experiment import make_interleaved_subject_folds  # noqa: E402
from scripts.run_strict_raw_feature_experiment import (  # noqa: E402
    REPORTS_DIR,
    TARGETS,
    build_feature_frames,
    numeric_feature_columns,
)

warnings.filterwarnings("ignore")

EPS = 1e-6
TOP_K_VALUES = [16, 32]
BLEND_WEIGHTS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
DEFAULT_BASE_IDS = [221, 226, 231]


@dataclass(frozen=True)
class BoosterSpec:
    name: str
    kind: str
    max_depth: int
    n_estimators: int
    learning_rate: float


BOOSTER_SPECS: tuple[BoosterSpec, ...] = (
    BoosterSpec("lgbm_d2_n80_lr03", "lgbm", max_depth=2, n_estimators=80, learning_rate=0.03),
    BoosterSpec("xgb_d2_n80_lr03", "xgb", max_depth=2, n_estimators=80, learning_rate=0.03),
)


def make_model(spec: BoosterSpec, seed: int):
    if spec.kind == "lgbm":
        return LGBMClassifier(
            n_estimators=spec.n_estimators,
            max_depth=spec.max_depth,
            num_leaves=7,
            learning_rate=spec.learning_rate,
            min_child_samples=24,
            reg_alpha=4.0,
            reg_lambda=8.0,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            verbose=-1,
            n_jobs=2,
        )
    if spec.kind == "xgb":
        return XGBClassifier(
            n_estimators=spec.n_estimators,
            max_depth=spec.max_depth,
            learning_rate=spec.learning_rate,
            min_child_weight=8.0,
            reg_alpha=4.0,
            reg_lambda=8.0,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=2,
            verbosity=0,
        )
    raise ValueError(f"Unknown booster kind: {spec.kind}")


def positive_proba(model: Any, features: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(features)
    classes = np.asarray(model.classes_)
    if 1 not in classes:
        return np.full(features.shape[0], 1.0 if classes[0] == 1 else 0.0)
    return proba[:, int(np.where(classes == 1)[0][0])]


def fit_stable_booster_oof(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    columns: list[str],
    top_k: int,
    spec: BoosterSpec,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    oof = np.zeros(len(y), dtype=float)
    test_pred = np.zeros(x_test.shape[0], dtype=float)
    details = []
    n_folds = 0
    for fold in sorted(np.unique(fold_ids)):
        train_idx = np.where(fold_ids != fold)[0]
        valid_idx = np.where(fold_ids == fold)[0]
        selected_idx, top_features = select_stable_columns(x_train, x_test, y, train_idx, columns, top_k)
        if len(np.unique(y[train_idx])) < 2:
            model = DummyClassifier(strategy="prior")
        else:
            model = clone(make_model(spec, seed=2600 + int(fold)))
        model.fit(x_train[train_idx][:, selected_idx], y[train_idx])
        oof[valid_idx] = positive_proba(model, x_train[valid_idx][:, selected_idx])
        test_pred += positive_proba(model, x_test[:, selected_idx])
        details.append({"fold": int(fold), "top_features": top_features})
        n_folds += 1
    return np.clip(oof, EPS, 1.0 - EPS), np.clip(test_pred / n_folds, EPS, 1.0 - EPS), details


def candidate_key(top_k: int, model_name: str, weight: float) -> str:
    if weight == 0.0:
        return "base"
    return f"stable_top{top_k}__{model_name}__w{weight:g}"


def run_diagnostic(base_ids: list[int], output_path: Path) -> dict[str, Any]:
    train, test, _metrics_sorted, _sample_original = build_feature_frames()
    fold_ids = make_interleaved_subject_folds(train.assign(date_dt=pd.to_datetime(train["date"])))
    columns = numeric_feature_columns(train, test)
    x_train = feature_matrix(train, columns)
    x_test = feature_matrix(test, columns)

    side_predictions: dict[str, dict[str, dict[str, Any]]] = {}
    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        side_predictions[target] = {}
        for top_k in TOP_K_VALUES:
            for spec in BOOSTER_SPECS:
                key = f"stable_top{top_k}__{spec.name}"
                oof, test_pred, details = fit_stable_booster_oof(
                    x_train=x_train,
                    x_test=x_test,
                    y=y,
                    fold_ids=fold_ids,
                    columns=columns,
                    top_k=top_k,
                    spec=spec,
                )
                side_predictions[target][key] = {
                    "oof": oof,
                    "test": test_pred,
                    "details": details,
                    "top_k": top_k,
                    "model": spec.name,
                }

    results = []
    for base_id in base_ids:
        base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
        base_scores = {
            target: float(log_loss(train[target].astype(int), base_oof[target].astype(float), labels=[0, 1]))
            for target in TARGETS
        }
        target_results: dict[str, Any] = {}
        best_losses = []
        for target in TARGETS:
            y = train[target].astype(int).to_numpy()
            base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
            base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
            candidates: dict[str, dict[str, Any]] = {
                "base": {"loss": base_scores[target], "kind": "base", "test_mean": float(np.mean(base_test_values))}
            }
            for side_key, payload in side_predictions[target].items():
                raw_oof = payload["oof"]
                raw_test = payload["test"]
                raw_loss = float(log_loss(y, raw_oof, labels=[0, 1]))
                candidates[f"raw__{side_key}"] = {
                    "kind": "raw",
                    "loss": raw_loss,
                    "test_mean": float(np.mean(raw_test)),
                    "top_k": payload["top_k"],
                    "model": payload["model"],
                }
                for weight in BLEND_WEIGHTS:
                    name = candidate_key(int(payload["top_k"]), str(payload["model"]), weight)
                    if name == "base":
                        continue
                    blended = np.clip((1.0 - weight) * base_values + weight * raw_oof, EPS, 1.0 - EPS)
                    blended_test = np.clip((1.0 - weight) * base_test_values + weight * raw_test, EPS, 1.0 - EPS)
                    candidates[name] = {
                        "kind": "blend",
                        "loss": float(log_loss(y, blended, labels=[0, 1])),
                        "test_mean": float(np.mean(blended_test)),
                        "test_max_abs_delta": float(np.max(np.abs(blended_test - base_test_values))),
                        "top_k": payload["top_k"],
                        "model": payload["model"],
                        "weight": weight,
                    }
            selectable = {name: value for name, value in candidates.items() if value["kind"] in {"base", "blend"}}
            best_name, best_info = min(selectable.items(), key=lambda item: (item[1]["loss"], item[0]))
            best_losses.append(float(best_info["loss"]))
            target_results[target] = {
                "base_loss": base_scores[target],
                "best": best_name,
                "best_loss": float(best_info["loss"]),
                "improvement_vs_base": float(base_scores[target] - best_info["loss"]),
                "top10_selectable": sorted(selectable.items(), key=lambda item: (item[1]["loss"], item[0]))[:10],
                "top10_all": sorted(candidates.items(), key=lambda item: (item[1]["loss"], item[0]))[:10],
            }
        results.append(
            {
                "base_id": base_id,
                "base_source": base_source,
                "base_mean": float(np.mean(list(base_scores.values()))),
                "targetwise_best_mean": float(np.mean(best_losses)),
                "mean_improvement_vs_base": float(np.mean(list(base_scores.values())) - np.mean(best_losses)),
                "base_scores": base_scores,
                "targets": target_results,
            }
        )
    report = {
        "diagnostic": "public_booster_stability_overlay_fast_full_oof_scan",
        "validation_lens": "public_structure_interleaved_within_subject_blocked_cv",
        "caution": "Full-OOF target-wise candidate selection; use only as a triage scan before nested diagnostics.",
        "base_ids": base_ids,
        "n_input_features": len(columns),
        "top_k_values": TOP_K_VALUES,
        "blend_weights": BLEND_WEIGHTS,
        "booster_specs": [asdict(spec) for spec in BOOSTER_SPECS],
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Public booster-stability overlay fast scan",
        "",
        "- Report-only; no submission artifact was generated.",
        f"- Base IDs: `{report['base_ids']}`",
        f"- Input numeric features: `{report['n_input_features']}`",
        "",
        "## Summary",
        "",
    ]
    for result in report["results"]:
        lines.append(
            f"- ID{result['base_id']:03d}: base `{result['base_mean']:.6f}` -> full-OOF targetwise "
            f"`{result['targetwise_best_mean']:.6f}` "
            f"(improvement `{result['mean_improvement_vs_base']:+.6f}`)."
        )
        for target in TARGETS:
            info = result["targets"][target]
            lines.append(
                f"  - {target}: `{info['base_loss']:.6f}` -> `{info['best_loss']:.6f}` via `{info['best']}` "
                f"(gain `{info['improvement_vs_base']:+.6f}`)."
            )
    lines.extend(["", "Full-OOF gains must be nested before promotion.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ids", default=",".join(str(value) for value in DEFAULT_BASE_IDS))
    parser.add_argument("--report-name", default="public_booster_stability_overlay_fast_scan_20260616")
    args = parser.parse_args()
    base_ids = [int(value.strip()) for value in args.base_ids.split(",") if value.strip()]
    json_path = REPORTS_DIR / f"{args.report_name}.json"
    md_path = REPORTS_DIR / f"{args.report_name}.md"
    report = run_diagnostic(base_ids, json_path)
    write_markdown(report, md_path)
    print(json.dumps({"json": str(json_path.relative_to(ROOT)), "md": str(md_path.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
