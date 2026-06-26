#!/usr/bin/env python3
"""Diagnose/promote window-pair sequence overlays.

This is a light, anchor-preserving representation inspired by the external
Public 0.5917 notebook's window-pair and screen-stability discussion. It
summarizes aligned 44-bin lifelog-day tensors over afternoon/evening/prebed/
sleep/wake windows and adds transition features between biologically plausible
window pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_public_random_conv_sequence import (  # noqa: E402
    EPS,
    RAW_DIR,
    ModelSpec,
    align_table,
    build_daily_sequence_table,
    fit_oof_model,
    table_to_tensor,
)
from scripts.run_public_tempered_extreme_calibration_probe_experiment import load_artifact_pair  # noqa: E402
from scripts.run_public_temporal_anchor_probe_experiment import make_interleaved_subject_folds  # noqa: E402
from scripts.run_strict_raw_feature_experiment import (  # noqa: E402
    BASE_PREDICTION_DIR,
    NOTEBOOK_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
    TARGETS,
    append_submission_log,
    build_feature_frames,
    map_predictions_to_original_sample,
    validate_submission,
)
from src.utils.validation import clip_probabilities  # noqa: E402

DEFAULT_BASE_ID = 215
CHANNEL_KEYWORDS = (
    "screen",
    "usage",
    "charging",
    "activity",
    "active",
    "still",
    "phone_light",
    "watch_light",
    "hr",
    "step",
    "distance",
    "pedo",
)
WINDOWS: dict[str, tuple[int, int]] = {
    "afternoon_12_18": (0, 12),
    "evening_18_21": (12, 18),
    "prebed_21_24": (18, 24),
    "late_00_03": (24, 30),
    "deep_03_06": (30, 36),
    "wake_06_09": (36, 42),
    "postwake_09_10": (42, 44),
    "sleep_21_09": (18, 42),
    "night_00_06": (24, 36),
    "all_12_10": (0, 44),
}
WINDOW_PAIRS: tuple[tuple[str, str], ...] = (
    ("afternoon_12_18", "evening_18_21"),
    ("evening_18_21", "prebed_21_24"),
    ("prebed_21_24", "sleep_21_09"),
    ("late_00_03", "deep_03_06"),
    ("deep_03_06", "wake_06_09"),
    ("sleep_21_09", "wake_06_09"),
    ("prebed_21_24", "wake_06_09"),
    ("night_00_06", "wake_06_09"),
)
MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("lr_c0_03", "lr", c_value=0.03),
    ModelSpec("lr_c0_1", "lr", c_value=0.1),
    ModelSpec("lr_c0_3", "lr", c_value=0.3),
    ModelSpec("hgb_l7", "hgb", max_leaf_nodes=7),
    ModelSpec("hgb_l15", "hgb", max_leaf_nodes=15),
    ModelSpec("et_d4", "et", max_depth=4),
)
BLEND_WEIGHTS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relpath(path: Path) -> Path:
    return path.resolve().relative_to(ROOT)


def selected_channel_indices(channel_names: list[str]) -> list[int]:
    selected: list[int] = []
    for idx, name in enumerate(channel_names):
        lowered = name.lower()
        if any(keyword in lowered for keyword in CHANNEL_KEYWORDS):
            selected.append(idx)
    return selected


def add_block(
    blocks: list[np.ndarray],
    names: list[str],
    block: np.ndarray,
    channel_names: list[str],
    suffix: str,
) -> None:
    values = np.asarray(block, dtype=np.float32)
    blocks.append(values)
    names.extend(f"{channel}__{suffix}" for channel in channel_names)


def safe_window(values: np.ndarray, window_name: str) -> np.ndarray:
    start, end = WINDOWS[window_name]
    return values[:, :, start:end]


def window_pair_features(tensor: np.ndarray, channel_names: list[str]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    selected = selected_channel_indices(channel_names)
    if not selected:
        raise ValueError("No channels selected for window-pair features.")
    selected_names = [channel_names[idx] for idx in selected]
    values = np.asarray(tensor[:, selected, :], dtype=np.float32)
    n_rows = values.shape[0]
    blocks: list[np.ndarray] = []
    names: list[str] = []
    window_stats: dict[str, dict[str, np.ndarray]] = {}

    for window_name in WINDOWS:
        window_values = safe_window(values, window_name)
        first_half = window_values[:, :, : max(1, window_values.shape[2] // 2)]
        second_half = window_values[:, :, window_values.shape[2] // 2 :]
        if second_half.shape[2] == 0:
            second_half = first_half
        mean = window_values.mean(axis=2)
        std = window_values.std(axis=2)
        min_value = window_values.min(axis=2)
        max_value = window_values.max(axis=2)
        sum_value = window_values.sum(axis=2)
        trend = second_half.mean(axis=2) - first_half.mean(axis=2)
        if window_values.shape[2] > 1:
            roughness = np.abs(np.diff(window_values, axis=2)).mean(axis=2)
        else:
            roughness = np.zeros_like(mean)
        window_stats[window_name] = {
            "mean": mean,
            "std": std,
            "min": min_value,
            "max": max_value,
            "sum": sum_value,
            "trend": trend,
            "roughness": roughness,
        }
        add_block(blocks, names, mean, selected_names, f"{window_name}__mean")
        add_block(blocks, names, std, selected_names, f"{window_name}__std")
        add_block(blocks, names, min_value, selected_names, f"{window_name}__min")
        add_block(blocks, names, max_value, selected_names, f"{window_name}__max")
        add_block(blocks, names, sum_value, selected_names, f"{window_name}__sum")
        add_block(blocks, names, trend, selected_names, f"{window_name}__trend")
        add_block(blocks, names, roughness, selected_names, f"{window_name}__roughness")

    for left_name, right_name in WINDOW_PAIRS:
        left = window_stats[left_name]
        right = window_stats[right_name]
        pair_name = f"{left_name}_to_{right_name}"
        mean_delta = right["mean"] - left["mean"]
        sum_delta = right["sum"] - left["sum"]
        std_delta = right["std"] - left["std"]
        rough_delta = right["roughness"] - left["roughness"]
        normalized_shift = mean_delta / (np.abs(left["mean"]) + np.abs(right["mean"]) + 1e-3)
        add_block(blocks, names, mean_delta, selected_names, f"{pair_name}__mean_delta")
        add_block(blocks, names, sum_delta, selected_names, f"{pair_name}__sum_delta")
        add_block(blocks, names, std_delta, selected_names, f"{pair_name}__std_delta")
        add_block(blocks, names, rough_delta, selected_names, f"{pair_name}__rough_delta")
        add_block(blocks, names, normalized_shift, selected_names, f"{pair_name}__normalized_shift")

    features = np.concatenate([block.reshape(n_rows, -1) for block in blocks], axis=1)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if features.shape[1] != len(names):
        raise ValueError(f"Feature/name mismatch: {features.shape[1]} vs {len(names)}")
    info = {
        "n_selected_channels": len(selected),
        "selected_channel_names": selected_names,
        "windows": WINDOWS,
        "window_pairs": WINDOW_PAIRS,
        "feature_name_sample": names[:60],
        "representation": (
            "Window-level mean/std/min/max/sum/trend/roughness and pairwise transition deltas "
            "over screen, usage, activity, light, HR, pedometer, and presence channels."
        ),
    }
    return features, names, info


def build_window_pair_data() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str], dict[str, Any]
]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    seq_table = build_daily_sequence_table(RAW_DIR)
    train_seq, test_seq, feature_cols = align_table(train, test, seq_table)
    train_tensor, test_tensor, channel_names = table_to_tensor(train_seq, test_seq, feature_cols)
    x_train, feature_names, info = window_pair_features(train_tensor, channel_names)
    x_test, _feature_names, _ = window_pair_features(test_tensor, channel_names)
    info.update(
        {
            "n_sequence_rows": int(len(seq_table)),
            "n_total_channels": int(len(channel_names)),
            "n_bins": int(train_tensor.shape[2]),
            "n_window_pair_features": int(x_train.shape[1]),
            "n_feature_names": int(len(feature_names)),
        }
    )
    return train, test, sample_original, x_train, x_test, channel_names, info


def compute_model_predictions(
    train: pd.DataFrame,
    x_train: np.ndarray,
    x_test: np.ndarray,
    fold_ids: np.ndarray,
    needed_models: set[str] | None = None,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], list[dict[str, Any]]]:
    spec_lookup = {spec.name: spec for spec in MODEL_SPECS}
    predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    specs_used = []
    for model_name, spec in spec_lookup.items():
        if needed_models is not None and model_name not in needed_models:
            continue
        specs_used.append(asdict(spec))
        predictions[model_name] = {}
        for target in TARGETS:
            y = train[target].astype(int).to_numpy()
            oof_pred, test_pred = fit_oof_model(x_train, x_test, y, fold_ids, spec)
            predictions[model_name][target] = {"oof": oof_pred, "test": test_pred}
    return predictions, specs_used


def candidate_key(model_name: str, weight: float) -> str:
    return f"blend__window_pair__{model_name}__w{weight:g}"


def parse_candidate_key(key: str) -> dict[str, Any]:
    if key == "base":
        return {"kind": "base"}
    if key.startswith("raw__"):
        return {"kind": "raw", "feature_set": "window_pair", "model": key.split("__", 1)[1]}
    kind, feature_set, model_name, weight_raw = key.split("__", 3)
    if kind != "blend" or feature_set != "window_pair" or not weight_raw.startswith("w"):
        raise ValueError(f"Unexpected candidate key: {key}")
    return {"kind": "blend", "feature_set": feature_set, "model": model_name, "weight": float(weight_raw[1:])}


def run_diagnostic(base_id: int, output_path: Path) -> dict[str, Any]:
    train, test, _sample_original, x_train, x_test, _channel_names, feature_info = build_window_pair_data()
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    fold_ids = make_interleaved_subject_folds(train.assign(date_dt=pd.to_datetime(train["date"])))
    predictions, specs_used = compute_model_predictions(train, x_train, x_test, fold_ids)
    base_scores = {
        target: float(log_loss(train[target].astype(int), base_oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    report: dict[str, Any] = {
        "diagnostic": "public_window_pair_sequence_overlay",
        "base_id": base_id,
        "base_source": base_source,
        "validation_lens": "public_structure_interleaved_within_subject_blocked_cv",
        "feature_info": feature_info,
        "model_specs": specs_used,
        "blend_weights": BLEND_WEIGHTS,
        "base_scores": base_scores,
        "base_mean": float(np.mean(list(base_scores.values()))),
        "targets": {},
    }
    target_best_losses = []
    for target in TARGETS:
        y = train[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        candidates: dict[str, dict[str, Any]] = {
            "base": {"kind": "base", "loss": base_scores[target], "test_mean": float(np.mean(base_test_values))}
        }
        for model_name, by_target in predictions.items():
            raw_oof = by_target[target]["oof"]
            raw_test = by_target[target]["test"]
            candidates[f"raw__{model_name}"] = {
                "kind": "raw",
                "feature_set": "window_pair",
                "model": model_name,
                "loss": float(log_loss(y, raw_oof, labels=[0, 1])),
                "test_mean": float(np.mean(raw_test)),
            }
            for weight in BLEND_WEIGHTS:
                pred = np.clip((1.0 - weight) * base_values + weight * raw_oof, EPS, 1.0 - EPS)
                blended_test = np.clip((1.0 - weight) * base_test_values + weight * raw_test, EPS, 1.0 - EPS)
                candidates[candidate_key(model_name, weight)] = {
                    "kind": "blend",
                    "feature_set": "window_pair",
                    "model": model_name,
                    "weight": weight,
                    "loss": float(log_loss(y, pred, labels=[0, 1])),
                    "test_mean": float(np.mean(blended_test)),
                    "test_max_abs_delta": float(np.max(np.abs(blended_test - base_test_values))),
                }
        selectable = {name: info for name, info in candidates.items() if name == "base" or info.get("kind") == "blend"}
        best_name, best_info = min(selectable.items(), key=lambda item: (item[1]["loss"], item[0]))
        target_best_losses.append(float(best_info["loss"]))
        report["targets"][target] = {
            "base_loss": base_scores[target],
            "best": best_name,
            "best_config": parse_candidate_key(best_name),
            "best_loss": float(best_info["loss"]),
            "improvement_vs_base": float(base_scores[target] - best_info["loss"]),
            "top10_selectable": sorted(selectable.items(), key=lambda item: (item[1]["loss"], item[0]))[:10],
            "top10_all": sorted(candidates.items(), key=lambda item: (item[1]["loss"], item[0]))[:10],
        }
    report["targetwise_best_mean"] = float(np.mean(target_best_losses))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def predict_from_report(
    base_id: int, report_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, Any], pd.DataFrame]:
    report_path = report_path.resolve()
    report = json.loads(report_path.read_text())
    if int(report["base_id"]) != base_id:
        raise ValueError(f"Report base_id={report['base_id']} does not match requested base_id={base_id}.")
    train, test, sample_original, x_train, x_test, _channel_names, feature_info = build_window_pair_data()
    base_oof, base_test, base_source = load_artifact_pair(train, test, base_id)
    fold_ids = make_interleaved_subject_folds(train.assign(date_dt=pd.to_datetime(train["date"])))
    selected_configs = {target: report["targets"][target]["best_config"] for target in TARGETS}
    needed_models = {config["model"] for config in selected_configs.values() if config.get("kind") in {"blend", "raw"}}
    predictions, specs_used = compute_model_predictions(train, x_train, x_test, fold_ids, needed_models)
    oof = base_oof.copy()
    test_pred = base_test.copy()
    target_details: dict[str, Any] = {}
    for target in TARGETS:
        config = selected_configs[target]
        y = train[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        if config["kind"] == "base":
            selected_oof = base_values
            selected_test = base_test_values
        else:
            side = predictions[config["model"]][target]
            raw_oof = side["oof"]
            raw_test = side["test"]
            if config["kind"] == "raw":
                selected_oof = raw_oof
                selected_test = raw_test
            else:
                weight = float(config["weight"])
                selected_oof = np.clip((1.0 - weight) * base_values + weight * raw_oof, EPS, 1.0 - EPS)
                selected_test = np.clip((1.0 - weight) * base_test_values + weight * raw_test, EPS, 1.0 - EPS)
        oof[target] = selected_oof
        test_pred[target] = selected_test
        target_details[target] = {
            "config": config,
            "base_loss": float(log_loss(y, base_values, labels=[0, 1])),
            "selected_loss": float(log_loss(y, selected_oof, labels=[0, 1])),
            "test_prediction_mean_before": float(np.mean(base_test_values)),
            "test_prediction_mean_after": float(np.mean(selected_test)),
            "test_prediction_max_abs_delta": float(np.max(np.abs(selected_test - base_test_values))),
        }
    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "diagnostic_report": str(relpath(report_path)),
        "selected_configs": selected_configs,
        "modified_targets": [target for target, config in selected_configs.items() if config["kind"] != "base"],
        "unchanged_targets": [target for target, config in selected_configs.items() if config["kind"] == "base"],
        "feature_info": feature_info,
        "model_specs_used": specs_used,
        "target_details": target_details,
        "selection_contract": (
            f"ID{base_id:03d} predictions are lightly blended target-wise with deterministic window-pair "
            "sequence models selected from the listed local public-structure diagnostic."
        ),
        "caution": (
            "Public-structure/non-strict window-pair sequence overlay. It is inspired by the external notebook's "
            "screen/window-pair stability idea but remains local OOF evidence only."
        ),
    }
    return oof, test_pred, target_scores, selector_info, sample_original


def write_repro_notebook(
    notebook_path: Path,
    submission_path: Path,
    metadata_path: Path,
    experiment_id: int,
    slug: str,
    local_cv: float,
    base_id: int,
    diagnostic_report: Path,
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: window-pair sequence overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{relpath(metadata_path)}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Diagnostic report: `{relpath(diagnostic_report)}`\n"
            f"- Public-structure CV: `{local_cv:.6f}`\n\n"
            "This artifact uses deterministic sleep-window transition features over the aligned raw tensor."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_window_pair_sequence_overlay_experiment.py "
            f"--promote --experiment-id {experiment_id} --slug {slug} --base-id {base_id} "
            f"--diagnostic-report {relpath(diagnostic_report)}"
        ),
        nbf.v4.new_markdown_cell("### Integrity note\n\nReview metadata before any DACON submission."),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def promote_from_report(experiment_id: int, slug: str, base_id: int, diagnostic_report: Path) -> dict[str, Any]:
    diagnostic_report = diagnostic_report.resolve()
    if experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")
    oof, test_pred, target_scores, selector_info, sample_original = predict_from_report(base_id, diagnostic_report)
    local_cv = float(np.mean(list(target_scores.values())))
    status = "unsubmitted"
    filename = f"submission_id_{experiment_id:03d}_{slug}_cv{local_cv:.6f}_{status}.csv"
    notebook_name = f"submission_{experiment_id:03d}_{slug}_cv{local_cv:.6f}_{status}.ipynb"
    metadata_name = f"experiment_{experiment_id:03d}_{slug}_cv{local_cv:.6f}_{status}.json"
    base_oof_name = f"base_oof_id_{experiment_id:03d}_{slug}_cv{local_cv:.6f}.csv"
    base_test_name = f"base_test_id_{experiment_id:03d}_{slug}_cv{local_cv:.6f}.csv"
    submission_path = SUBMISSIONS_DIR / filename
    notebook_path = NOTEBOOK_DIR / notebook_name
    metadata_path = REPORTS_DIR / metadata_name
    base_oof_path = BASE_PREDICTION_DIR / base_oof_name
    base_test_path = BASE_PREDICTION_DIR / base_test_name
    for path in [submission_path, notebook_path, metadata_path, base_oof_path, base_test_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {relpath(path)}")
    out = clip_probabilities(map_predictions_to_original_sample(sample_original, test_pred), TARGETS)
    if not validate_submission(out, sample_original):
        raise ValueError("Generated submission failed sample-format validation.")
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    BASE_PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(submission_path, index=False)
    oof.to_csv(base_oof_path, index=False)
    test_pred.to_csv(base_test_path, index=False)
    metadata = {
        "experiment_id": experiment_id,
        "slug": slug,
        "method": f"Target-wise window-pair sequence overlay on ID{base_id:03d}",
        "submission_file": submission_path.name,
        "notebook_file": str(relpath(notebook_path)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_interleaved_within_subject_blocked_cv_window_pair_sequence",
        "strict_subject_hole_comparable": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(relpath(base_oof_path)),
        "base_test_file": str(relpath(base_test_path)),
        "validation": "Target-wise window-pair sequence overlay under public-structure interleaved CV.",
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(
        notebook_path, submission_path, metadata_path, experiment_id, slug, local_cv, base_id, diagnostic_report
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=(
            f"Public-structure window-pair sequence overlay on ID{base_id:03d}; target-wise selected from "
            f"{relpath(diagnostic_report)}. Inspired by external Public 0.5917 notebook screen/window-pair "
            "stability idea; not strict Subject-Hole CV; no DACON submission was made."
        ),
        model_description=f"window-pair sequence overlay on ID{base_id:03d}",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "id215_window_pair_sequence_diagnostic.json",
    )
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--experiment-id", type=int)
    parser.add_argument("--slug", default="id215_window_pair_sequence_overlay")
    parser.add_argument("--diagnostic-report", type=Path)
    args = parser.parse_args()
    if args.promote:
        if args.experiment_id is None or args.diagnostic_report is None:
            raise ValueError("--promote requires --experiment-id and --diagnostic-report.")
        metadata = promote_from_report(args.experiment_id, args.slug, args.base_id, args.diagnostic_report)
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return 0

    report = run_diagnostic(args.base_id, args.output)
    summary = {
        "base_id": report["base_id"],
        "base_mean": report["base_mean"],
        "targetwise_best_mean": report["targetwise_best_mean"],
        "target_best": {target: info["best"] for target, info in report["targets"].items()},
        "target_best_losses": {target: info["best_loss"] for target, info in report["targets"].items()},
        "feature_info": {
            key: value for key, value in report["feature_info"].items() if key != "feature_name_sample"
        },
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
