#!/usr/bin/env python3
"""Create a public-structure S3/S4 raw hourly-sequence overlay on top of ID123."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from functools import reduce
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

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
from scripts.run_strict_shutdown_feature_set_selector_experiment import align_candidate  # noqa: E402
from src.utils.validation import clip_probabilities  # noqa: E402

warnings.filterwarnings("ignore")

EPS = 1e-6
DEFAULT_BASE_ID = 123
OVERLAY_CONFIG = {
    "S3": {
        "model": "hgb_l7_l2",
        "blend_weight": 0.07,
    },
    "S4": {
        "model": "et_d6_l12",
        "blend_weight": 0.20,
    },
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_base_artifact(train: pd.DataFrame, test: pd.DataFrame, base_id: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    oof_paths = sorted(BASE_PREDICTION_DIR.glob(f"base_oof_id_{base_id:03d}_*.csv"))
    test_paths = sorted(BASE_PREDICTION_DIR.glob(f"base_test_id_{base_id:03d}_*.csv"))
    if len(oof_paths) != 1 or len(test_paths) != 1:
        raise FileNotFoundError(f"Expected exactly one ID{base_id:03d} base OOF/test artifact.")
    return (
        align_candidate(pd.read_csv(oof_paths[0]), train),
        align_candidate(pd.read_csv(test_paths[0]), test),
        f"{oof_paths[0].name}; {test_paths[0].name}",
    )


def anchor_lifelog_window(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    hour = out["timestamp"].dt.hour
    out["date"] = out["timestamp"].dt.normalize()
    out.loc[hour < 12, "date"] = out.loc[hour < 12, "date"] - pd.Timedelta(days=1)
    out["rel_hour"] = hour.astype(float) + (hour < 12).astype(float) * 24.0 + out["timestamp"].dt.minute / 60.0
    out["hour_bin"] = np.floor(out["rel_hour"]).astype(int)
    return out[(out["rel_hour"] >= 12.0) & (out["rel_hour"] < 34.0)].copy()


def mean_array(value: object) -> float:
    try:
        arr = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return np.nan
    arr = arr[np.isfinite(arr) & (arr > 0)]
    return float(arr.mean()) if arr.size else np.nan


def hourly_numeric_features(
    raw_dir: Path,
    filename: str,
    value_col: str,
    prefix: str,
    aggs: tuple[str, ...],
) -> pd.DataFrame:
    df = anchor_lifelog_window(pd.read_parquet(raw_dir / filename))
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    grouped = df.groupby(["subject_id", "date", "hour_bin"])[value_col].agg(list(aggs)).reset_index()
    frames = []
    for agg_name in aggs:
        pivot = grouped.pivot_table(index=["subject_id", "date"], columns="hour_bin", values=agg_name, aggfunc="first")
        pivot.columns = [f"seq_{prefix}_h{int(col)}_{agg_name}" for col in pivot.columns]
        frames.append(pivot.reset_index())
    return reduce(lambda left, right: pd.merge(left, right, on=["subject_id", "date"], how="outer"), frames)


def hourly_activity_features(raw_dir: Path) -> pd.DataFrame:
    df = anchor_lifelog_window(pd.read_parquet(raw_dir / "ch2025_mActivity.parquet"))
    df["still"] = (df["m_activity"] == 0).astype(float)
    df["active"] = df["m_activity"].isin([3, 7, 8]).astype(float)
    frames = []
    for col in ["still", "active", "m_activity"]:
        grouped = df.groupby(["subject_id", "date", "hour_bin"])[col].mean().reset_index()
        pivot = grouped.pivot_table(index=["subject_id", "date"], columns="hour_bin", values=col, aggfunc="first")
        pivot.columns = [f"seq_act_{col}_h{int(hour)}_mean" for hour in pivot.columns]
        frames.append(pivot.reset_index())
    return reduce(lambda left, right: pd.merge(left, right, on=["subject_id", "date"], how="outer"), frames)


def hourly_hr_features(raw_dir: Path) -> pd.DataFrame:
    df = anchor_lifelog_window(pd.read_parquet(raw_dir / "ch2025_wHr.parquet"))
    df["hr"] = df["heart_rate"].map(mean_array)
    grouped = df.groupby(["subject_id", "date", "hour_bin"])["hr"].agg(["mean", "std", "max", "min"]).reset_index()
    frames = []
    for agg_name in ["mean", "std", "max", "min"]:
        pivot = grouped.pivot_table(index=["subject_id", "date"], columns="hour_bin", values=agg_name, aggfunc="first")
        pivot.columns = [f"seq_hr_h{int(hour)}_{agg_name}" for hour in pivot.columns]
        frames.append(pivot.reset_index())
    return reduce(lambda left, right: pd.merge(left, right, on=["subject_id", "date"], how="outer"), frames)


def hourly_pedo_features(raw_dir: Path) -> pd.DataFrame:
    df = anchor_lifelog_window(pd.read_parquet(raw_dir / "ch2025_wPedo.parquet"))
    specs = {
        "step": "sum",
        "distance": "sum",
        "speed": "mean",
        "step_frequency": "mean",
        "burned_calories": "sum",
    }
    frames = []
    for col, agg_func in specs.items():
        grouped = df.groupby(["subject_id", "date", "hour_bin"])[col].agg(agg_func).reset_index()
        pivot = grouped.pivot_table(index=["subject_id", "date"], columns="hour_bin", values=col, aggfunc="first")
        pivot.columns = [f"seq_pedo_{col}_h{int(hour)}_{agg_func}" for hour in pivot.columns]
        frames.append(pivot.reset_index())
    return reduce(lambda left, right: pd.merge(left, right, on=["subject_id", "date"], how="outer"), frames)


def build_raw_hourly_sequence_features(raw_dir: Path) -> pd.DataFrame:
    frames = [
        hourly_numeric_features(raw_dir, "ch2025_mScreenStatus.parquet", "m_screen_use", "screen", ("mean", "sum")),
        hourly_numeric_features(raw_dir, "ch2025_mACStatus.parquet", "m_charging", "charge", ("mean", "sum")),
        hourly_activity_features(raw_dir),
        hourly_numeric_features(raw_dir, "ch2025_mLight.parquet", "m_light", "mlight", ("mean", "max")),
        hourly_numeric_features(raw_dir, "ch2025_wLight.parquet", "w_light", "wlight", ("mean", "max")),
        hourly_hr_features(raw_dir),
        hourly_pedo_features(raw_dir),
    ]
    out = reduce(lambda left, right: pd.merge(left, right, on=["subject_id", "date"], how="outer"), frames)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out.sort_values(["subject_id", "date"]).reset_index(drop=True)


def align_sequence_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    seq: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_key = train[["subject_id", "date"]].copy()
    test_key = test[["subject_id", "date"]].copy()
    train_key["date"] = pd.to_datetime(train_key["date"]).dt.date
    test_key["date"] = pd.to_datetime(test_key["date"]).dt.date
    train_seq = train_key.merge(seq, on=["subject_id", "date"], how="left")
    test_seq = test_key.merge(seq, on=["subject_id", "date"], how="left")
    for prefix in ["seq_screen", "seq_charge", "seq_act", "seq_mlight", "seq_wlight", "seq_hr", "seq_pedo"]:
        cols = [col for col in seq.columns if col.startswith(prefix)]
        train_seq[f"{prefix}_present_frac"] = train_seq[cols].notna().mean(axis=1) if cols else 0.0
        test_seq[f"{prefix}_present_frac"] = test_seq[cols].notna().mean(axis=1) if cols else 0.0
    feature_cols = [col for col in train_seq.columns if col not in ["subject_id", "date"]]
    return train_seq, test_seq, feature_cols


def make_sequence_model(model_name: str):
    if model_name == "hgb_l7_l2":
        return HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.03,
            max_leaf_nodes=7,
            l2_regularization=2.0,
            random_state=45,
        )
    if model_name == "et_d6_l12":
        return ExtraTreesClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=12,
            max_features=0.5,
            random_state=43,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown sequence model: {model_name}")


def fit_public_sequence_candidate(
    train: pd.DataFrame,
    test_seq: pd.DataFrame,
    train_seq: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    fold_ids = make_interleaved_subject_folds(train.assign(date_dt=pd.to_datetime(train["date"])))
    y = train[target].astype(int).to_numpy()
    oof = np.zeros(len(train), dtype=float)
    test_pred = np.zeros(len(test_seq), dtype=float)
    n_folds = 0
    for fold in sorted(np.unique(fold_ids)):
        train_idx = np.where(fold_ids != fold)[0]
        valid_idx = np.where(fold_ids == fold)[0]
        if len(np.unique(y[train_idx])) < 2:
            model = DummyClassifier(strategy="prior")
        else:
            model = make_pipeline(SimpleImputer(strategy="median"), clone(make_sequence_model(model_name)))
        model.fit(train_seq.iloc[train_idx][feature_cols], y[train_idx])
        oof[valid_idx] = model.predict_proba(train_seq.iloc[valid_idx][feature_cols])[:, 1]
        test_pred += model.predict_proba(test_seq[feature_cols])[:, 1]
        n_folds += 1
    return np.clip(oof, EPS, 1.0 - EPS), np.clip(test_pred / n_folds, EPS, 1.0 - EPS)


def predict_sequence_s34_overlay(
    base_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    base_oof, base_test, base_source = load_base_artifact(train, test, base_id)
    seq = build_raw_hourly_sequence_features(ROOT / "data" / "raw" / "ch2025_data_items")
    train_seq, test_seq, feature_cols = align_sequence_features(train, test, seq)

    oof = base_oof.copy()
    test_pred = base_test.copy()
    target_details: dict[str, dict] = {}
    for target, config in OVERLAY_CONFIG.items():
        model_name = str(config["model"])
        weight = float(config["blend_weight"])
        y = train[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_test_values = np.clip(base_test[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        raw_oof, raw_test = fit_public_sequence_candidate(
            train=train,
            test_seq=test_seq,
            train_seq=train_seq,
            feature_cols=feature_cols,
            target=target,
            model_name=model_name,
        )
        selected_oof = np.clip((1.0 - weight) * base_values + weight * raw_oof, EPS, 1.0 - EPS)
        selected_test = np.clip((1.0 - weight) * base_test_values + weight * raw_test, EPS, 1.0 - EPS)
        oof[target] = selected_oof
        test_pred[target] = selected_test
        target_details[target] = {
            "model": model_name,
            "blend_weight": weight,
            "base_loss": float(log_loss(y, base_values, labels=[0, 1])),
            "raw_sequence_loss": float(log_loss(y, raw_oof, labels=[0, 1])),
            "overlay_loss": float(log_loss(y, selected_oof, labels=[0, 1])),
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
        "modified_targets": sorted(OVERLAY_CONFIG),
        "unchanged_targets": [target for target in TARGETS if target not in OVERLAY_CONFIG],
        "n_sequence_rows": int(len(seq)),
        "n_sequence_features": int(len(feature_cols)),
        "sequence_window": "lifelog date 12:00 through next-day 09:59, flattened by integer hour bins",
        "sensor_families": ["screen", "charging", "activity", "phone_light", "watch_light", "heart_rate", "pedometer"],
        "target_details": target_details,
        "selection_contract": (
            f"ID{base_id:03d} predictions are preserved for Q1/Q2/Q3/S1/S2. S3/S4 receive fixed blends of the "
            "base prediction and raw hourly-sequence model OOF predictions under public-structure interleaved "
            "within-subject folds. Blend weights were selected from a small diagnostic grid, so this remains a "
            "public-structure probe rather than a strict local improvement."
        ),
        "external_idea_note": (
            "Inspired by Pixleep-style sensor-to-image/sequence modeling, but implemented as lightweight hourly "
            "flattened sequence features to stay memory-safe locally."
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
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: public S3/S4 raw hourly-sequence overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Public-structure probe CV: `{local_cv:.6f}`\n\n"
            "This artifact changes only S3/S4 using lightweight Pixleep-inspired hourly sequence features. "
            "It is not strict Subject-Hole CV."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_public_sequence_s34_overlay_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\n"
            f"The exact generated file is `{submission_path.name}`. "
            "The artifact preserves all non-S3/S4 target columns from the base ID."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="s34_raw_hourly_sequence_overlay_public_probe")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = predict_sequence_s34_overlay(args.base_id)
    local_cv = float(np.mean(list(target_scores.values())))

    status = "unsubmitted"
    filename = f"submission_id_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}_{status}.csv"
    notebook_name = f"submission_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}_{status}.ipynb"
    metadata_name = f"experiment_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}_{status}.json"
    base_oof_name = f"base_oof_id_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}.csv"
    base_test_name = f"base_test_id_{args.experiment_id:03d}_{args.slug}_cv{local_cv:.6f}.csv"

    submission_path = SUBMISSIONS_DIR / filename
    notebook_path = NOTEBOOK_DIR / notebook_name
    metadata_path = REPORTS_DIR / metadata_name
    base_oof_path = BASE_PREDICTION_DIR / base_oof_name
    base_test_path = BASE_PREDICTION_DIR / base_test_name
    for path in [submission_path, notebook_path, metadata_path, base_oof_path, base_test_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path.relative_to(ROOT)}")

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
        "experiment_id": args.experiment_id,
        "slug": args.slug,
        "method": "Public-structure S3/S4 raw hourly-sequence overlay",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "local_cv_name": "public_structure_interleaved_within_subject_blocked_cv_plus_non_nested_blend_grid",
        "strict_subject_hole_comparable": False,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": (
            "S3/S4 raw hourly-sequence OOF predictions are generated with interleaved within-subject folds and "
            "then blended with the base ID using fixed diagnostic-grid weights. This is not strict Subject-Hole CV."
        ),
        "overlay_config": OVERLAY_CONFIG,
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(
        notebook_path, submission_path, metadata_path, args.experiment_id, args.slug, local_cv, args.base_id
    )

    notes = (
        f"Public-structure probe on ID{args.base_id:03d}: S3/S4 only raw hourly-sequence overlay inspired by "
        "sensor-to-image/sequence methods; Q1/Q2/Q3/S1/S2 unchanged. Not strict Subject-Hole CV and no DACON "
        "submission was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"Public-structure S3/S4 raw hourly-sequence overlay on ID{args.base_id:03d}",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
