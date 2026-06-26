#!/usr/bin/env python3
"""Promote a Q3-only GPS routine overlay on top of a strict base artifact."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_strict_raw_feature_experiment import (  # noqa: E402
    BASE_PREDICTION_DIR,
    NOTEBOOK_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
    TARGETS,
    append_submission_log,
    build_feature_frames,
    map_predictions_to_original_sample,
    sha256_file,
    validate_submission,
)
from scripts.run_strict_shutdown_feature_set_selector_experiment import align_candidate  # noqa: E402
from src.utils.validation import clip_probabilities, make_subject_hole_folds  # noqa: E402

warnings.filterwarnings("ignore")

EPS = 1e-6
TARGET = "Q3"
DEFAULT_BASE_ID = 118
RAW_CONTEXT_DIR = ROOT / "data" / "raw" / "ch2025_data_items"
WINDOWS = {
    "aft12_18": (12.0, 18.0),
    "eve18_22": (18.0, 22.0),
    "late22_24": (22.0, 24.0),
    "all12_24": (12.0, 24.0),
}
BLEND_WEIGHTS = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25]


def summarize_gps_points(points: object) -> dict[str, float] | None:
    lats: list[float] = []
    lons: list[float] = []
    speeds: list[float] = []
    altitudes: list[float] = []
    if not isinstance(points, (list, tuple, np.ndarray)):
        return None
    for point in points:
        if not isinstance(point, dict):
            continue
        try:
            lat = float(point.get("latitude", point.get("lat", np.nan)))
            lon = float(point.get("longitude", point.get("lon", np.nan)))
            speed = float(point.get("speed", np.nan))
            altitude = float(point.get("altitude", np.nan))
        except (TypeError, ValueError):
            continue
        if np.isfinite(lat) and np.isfinite(lon) and not (lat == 0.0 and lon == 0.0):
            lats.append(lat)
            lons.append(lon)
        if np.isfinite(speed):
            speeds.append(speed)
        if np.isfinite(altitude):
            altitudes.append(altitude)
    if not lats:
        return None

    lat_arr = np.asarray(lats, dtype=float)
    lon_arr = np.asarray(lons, dtype=float)
    speed_arr = np.asarray(speeds, dtype=float)
    altitude_arr = np.asarray(altitudes, dtype=float)
    path_delta = 0.0
    if len(lat_arr) > 1:
        path_delta = float(np.sqrt(np.diff(lat_arr) ** 2 + np.diff(lon_arr) ** 2).sum())
    return {
        "lat_mean": float(lat_arr.mean()),
        "lon_mean": float(lon_arr.mean()),
        "lat_std": float(lat_arr.std()),
        "lon_std": float(lon_arr.std()),
        "speed_mean": float(speed_arr.mean()) if len(speed_arr) else np.nan,
        "speed_max": float(speed_arr.max()) if len(speed_arr) else np.nan,
        "moving_ratio": float((speed_arr > 0.5).mean()) if len(speed_arr) else np.nan,
        "alt_mean": float(altitude_arr.mean()) if len(altitude_arr) else np.nan,
        "n_points": float(len(lat_arr)),
        "path_delta": path_delta,
    }


def empty_window_features(row: dict[str, float | object], window_name: str) -> None:
    for feature_name in [
        "records",
        "points",
        "speed_mean",
        "speed_max",
        "moving_ratio",
        "path_delta",
        "home_dist_mean",
        "home_dist_max",
        "away_ratio",
        "cell4_unique",
        "cell3_unique",
        "lat_span",
        "lon_span",
        "alt_span",
    ]:
        row[f"gpsr_{window_name}_{feature_name}"] = np.nan


def add_window_features(row: dict[str, float | object], group: pd.DataFrame, window_name: str) -> None:
    row[f"gpsr_{window_name}_records"] = float(len(group))
    row[f"gpsr_{window_name}_points"] = float(group["n_points"].sum())
    row[f"gpsr_{window_name}_speed_mean"] = float(group["speed_mean"].mean())
    row[f"gpsr_{window_name}_speed_max"] = float(group["speed_max"].max())
    row[f"gpsr_{window_name}_moving_ratio"] = float(group["moving_ratio"].mean())
    row[f"gpsr_{window_name}_path_delta"] = float(group["path_delta"].sum())
    row[f"gpsr_{window_name}_home_dist_mean"] = float(group["home_dist"].mean())
    row[f"gpsr_{window_name}_home_dist_max"] = float(group["home_dist"].max())
    row[f"gpsr_{window_name}_away_ratio"] = float((group["home_dist"] > 0.001).mean())
    row[f"gpsr_{window_name}_cell4_unique"] = float(group["cell4"].nunique())
    row[f"gpsr_{window_name}_cell3_unique"] = float(group["cell3"].nunique())
    row[f"gpsr_{window_name}_lat_span"] = float(group["lat_mean"].max() - group["lat_mean"].min())
    row[f"gpsr_{window_name}_lon_span"] = float(group["lon_mean"].max() - group["lon_mean"].min())
    row[f"gpsr_{window_name}_alt_span"] = float(group["alt_mean"].max() - group["alt_mean"].min())


def build_gps_routine_features(raw_dir: Path = RAW_CONTEXT_DIR) -> tuple[pd.DataFrame, dict]:
    path = raw_dir / "ch2025_mGps.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing GPS parquet: {path.relative_to(ROOT)}")

    df = pd.read_parquet(path)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour_float"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0

    rows: list[dict[str, float | object]] = []
    for record in df.itertuples(index=False):
        summary = summarize_gps_points(record.m_gps)
        if summary is None:
            continue
        summary.update(
            {
                "subject_id": record.subject_id,
                "date": record.date,
                "hour_float": float(record.hour_float),
            }
        )
        rows.append(summary)
    flat = pd.DataFrame(rows)
    if flat.empty:
        raise ValueError("No usable GPS routine rows were built.")

    home: dict[str, tuple[float, float]] = {}
    late = flat[(flat["hour_float"] >= 20.0) & (flat["hour_float"] < 24.0)]
    for subject_id, group in late.groupby("subject_id", sort=False):
        home[str(subject_id)] = (float(group["lat_mean"].median()), float(group["lon_mean"].median()))

    daily_rows: list[dict[str, float | object]] = []
    for (subject_id, date), group in flat.groupby(["subject_id", "date"], sort=False):
        home_lat, home_lon = home.get(str(subject_id), (float(group["lat_mean"].median()), float(group["lon_mean"].median())))
        group = group.copy()
        group["home_dist"] = np.sqrt((group["lat_mean"] - home_lat) ** 2 + (group["lon_mean"] - home_lon) ** 2)
        group["cell4"] = group["lat_mean"].round(4).astype(str) + "_" + group["lon_mean"].round(4).astype(str)
        group["cell3"] = group["lat_mean"].round(3).astype(str) + "_" + group["lon_mean"].round(3).astype(str)

        row: dict[str, float | object] = {"subject_id": subject_id, "date": date}
        for window_name, (lo, hi) in WINDOWS.items():
            subset = group[(group["hour_float"] >= lo) & (group["hour_float"] < hi)]
            if subset.empty:
                empty_window_features(row, window_name)
            else:
                add_window_features(row, subset, window_name)

        day_group = group[(group["hour_float"] >= 12.0) & (group["hour_float"] < 24.0)]
        moving = day_group[day_group["moving_ratio"].fillna(0.0) > 0.2]
        away = day_group[day_group["home_dist"] > 0.001]
        homeish = day_group[day_group["home_dist"] <= 0.001]
        row["gpsr_last_move_hour"] = float(moving["hour_float"].max()) if len(moving) else -1.0
        row["gpsr_first_homeish_hour"] = float(homeish["hour_float"].min()) if len(homeish) else 25.0
        row["gpsr_last_away_hour"] = float(away["hour_float"].max()) if len(away) else -1.0
        daily_rows.append(row)

    features = pd.DataFrame(daily_rows).sort_values(["subject_id", "date"]).reset_index(drop=True)
    info = {
        "raw_context_file": str(path.relative_to(ROOT)),
        "raw_rows": int(len(df)),
        "usable_gps_rows": int(len(flat)),
        "daily_feature_rows": int(len(features)),
        "home_definition": "subject median GPS lat/lon over 20:00-24:00 covariate rows; label-free and not stored",
        "privacy_note": "Raw coordinates are used only transiently; artifact metadata stores no coordinates.",
    }
    return features, info


def attach_gps_routine_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    features, info = build_gps_routine_features()
    features = features.copy()
    features["date"] = pd.to_datetime(features["date"]).dt.date
    if features.duplicated(["subject_id", "date"]).any():
        duplicated = features.loc[features.duplicated(["subject_id", "date"], keep=False), ["subject_id", "date"]]
        raise ValueError(f"GPS routine features are not unique by subject/date:\n{duplicated.head()}")

    train_key = train[["subject_id", "date", *TARGETS]].copy()
    test_key = test[["subject_id", "date"]].copy()
    train_key["date"] = pd.to_datetime(train_key["date"]).dt.date
    test_key["date"] = pd.to_datetime(test_key["date"]).dt.date
    train_gps = train_key.merge(features, on=["subject_id", "date"], how="left", validate="one_to_one")
    test_gps = test_key.merge(features, on=["subject_id", "date"], how="left", validate="one_to_one")
    feature_cols = [
        col
        for col in train_gps.columns
        if col.startswith("gpsr_") and pd.api.types.is_numeric_dtype(train_gps[col]) and col in test_gps.columns
    ]
    if not feature_cols:
        raise ValueError("No numeric GPS routine feature columns were built.")
    info.update(
        {
            "n_feature_columns": len(feature_cols),
            "missing_all_features": {
                "train": int(train_gps[feature_cols].isna().all(axis=1).sum()),
                "test": int(test_gps[feature_cols].isna().all(axis=1).sum()),
            },
        }
    )
    return train_gps, test_gps, feature_cols, info


def model_pool() -> dict[str, object]:
    return {
        "lgbm_l7": LGBMClassifier(
            n_estimators=160,
            learning_rate=0.025,
            num_leaves=7,
            max_depth=3,
            min_child_samples=18,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=25.0,
            random_state=42,
            verbosity=-1,
        ),
        "hgb_l7": HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.03,
            max_leaf_nodes=7,
            l2_regularization=2.0,
            random_state=43,
        ),
        "et_d5": ExtraTreesClassifier(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=12,
            max_features=0.7,
            random_state=44,
            n_jobs=-1,
        ),
        "rf_d5": RandomForestClassifier(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=45,
            n_jobs=-1,
        ),
    }


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


def fit_predict_one_model(
    model_proto: object,
    train_gps: pd.DataFrame,
    test_gps: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    folds = make_subject_hole_folds(train_gps, n_folds=5)
    y = train_gps[TARGET].astype(int).reset_index(drop=True)
    oof = np.zeros(len(train_gps), dtype=float)
    has_prediction = np.zeros(len(train_gps), dtype=bool)
    test_pred = np.zeros(len(test_gps), dtype=float)

    for train_idx, valid_idx in folds:
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(train_gps.iloc[train_idx][feature_cols])
        x_valid = imputer.transform(train_gps.iloc[valid_idx][feature_cols])
        x_test = imputer.transform(test_gps[feature_cols])
        if y.iloc[train_idx].nunique() < 2:
            model = DummyClassifier(strategy="prior")
        else:
            model = clone(model_proto)
        model.fit(x_train, y.iloc[train_idx])
        oof[valid_idx] = model.predict_proba(x_valid)[:, 1]
        test_pred += model.predict_proba(x_test)[:, 1]
        has_prediction[valid_idx] = True

    if not has_prediction.all():
        raise ValueError("Folds did not produce predictions for every Q3 row.")
    return np.clip(oof, EPS, 1.0 - EPS), np.clip(test_pred / len(folds), EPS, 1.0 - EPS)


def predict_q3_gps_routine_overlay(
    base_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train, test, _metrics_sorted, sample_original = build_feature_frames()
    base_oof, base_test, base_source = load_base_artifact(train, test, base_id)
    train_gps, test_gps, feature_cols, feature_info = attach_gps_routine_features(train, test)

    y = train_gps[TARGET].astype(int).to_numpy()
    base_values = np.clip(base_oof[TARGET].astype(float).to_numpy(), EPS, 1.0 - EPS)
    base_test_values = np.clip(base_test[TARGET].astype(float).to_numpy(), EPS, 1.0 - EPS)
    candidate_oof = {f"id{base_id:03d}_base": base_values}
    candidate_test = {f"id{base_id:03d}_base": base_test_values}
    candidate_details: dict[str, dict] = {f"id{base_id:03d}_base": {"source": "base"}}

    for model_name, model_proto in model_pool().items():
        raw_oof, raw_test = fit_predict_one_model(model_proto, train_gps, test_gps, feature_cols)
        raw_name = f"gps_routine__{model_name}"
        candidate_oof[raw_name] = raw_oof
        candidate_test[raw_name] = raw_test
        candidate_details[raw_name] = {
            "feature_family": "gps_routine_numeric_summary",
            "model": model_name,
            "blend": False,
        }
        for weight in BLEND_WEIGHTS:
            blend_name = f"blend_base{1.0 - weight:.2f}_{raw_name}_{weight:.2f}"
            candidate_oof[blend_name] = np.clip((1.0 - weight) * base_values + weight * raw_oof, EPS, 1.0 - EPS)
            candidate_test[blend_name] = np.clip(
                (1.0 - weight) * base_test_values + weight * raw_test,
                EPS,
                1.0 - EPS,
            )
            candidate_details[blend_name] = {
                "blend_candidate": raw_name,
                "base_weight": 1.0 - weight,
                "candidate_weight": weight,
            }

    candidate_scores = {name: float(log_loss(y, values, labels=[0, 1])) for name, values in candidate_oof.items()}
    folds = make_subject_hole_folds(train_gps, n_folds=5)
    selected_q3_oof = np.zeros(len(train_gps), dtype=float)
    nested_choices: list[dict] = []
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        losses = {name: log_loss(y[train_idx], values[train_idx], labels=[0, 1]) for name, values in candidate_oof.items()}
        best_name, best_loss = min(losses.items(), key=lambda item: item[1])
        selected_q3_oof[valid_idx] = candidate_oof[best_name][valid_idx]
        nested_choices.append(
            {
                "fold": fold_idx,
                "candidate": best_name,
                "train_fold_log_loss": float(best_loss),
                "candidate_losses": {name: float(value) for name, value in sorted(losses.items())},
            }
        )

    full_best_name, full_best_loss = min(candidate_scores.items(), key=lambda item: item[1])
    oof = base_oof.copy()
    test_pred = base_test.copy()
    oof[TARGET] = np.clip(selected_q3_oof, EPS, 1.0 - EPS)
    test_pred[TARGET] = np.clip(candidate_test[full_best_name], EPS, 1.0 - EPS)

    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "feature_info": feature_info,
        "feature_family": "gps_routine_numeric_summary",
        "feature_columns": feature_cols,
        "candidate_pool_size": len(candidate_oof),
        "candidate_scores": candidate_scores,
        "candidate_details": candidate_details,
        "nested_choices": {TARGET: nested_choices},
        "full_train_test_choices": {
            TARGET: {
                "candidate": full_best_name,
                "full_oof_log_loss": float(full_best_loss),
                "candidate_losses": {name: float(value) for name, value in sorted(candidate_scores.items())},
            }
        },
        "selection_contract": (
            f"ID{base_id:03d} predictions are preserved for all targets except Q3. Q3 validation rows receive "
            "the GPS-routine candidate selected using only each outer fold's training rows. Final test Q3 uses "
            "the candidate with the best full-training OOF log loss."
        ),
        "privacy_note": "Raw GPS coordinates are used only transiently to build numeric routine features; metadata stores no coordinates.",
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
            f"# Submission {experiment_id:03d}: Q3 GPS routine overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Local Subject-Hole CV: `{local_cv:.6f}`\n\n"
            "This notebook is a reproduction pointer for an immutable artifact. "
            f"The generating script keeps ID{base_id:03d} predictions for every non-Q3 target and replaces Q3 only."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_strict_q3_gps_routine_overlay_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\n"
            f"The exact generated file is `{submission_path.name}`. "
            "Raw GPS coordinates are not stored in the metadata."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="q3_gps_routine_overlay")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict-era experiments must use id >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = predict_q3_gps_routine_overlay(args.base_id)
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
        "method": "Strict Q3-only GPS routine overlay",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": "Subject-Hole CV with Q3-only nested candidate selection over GPS routine numeric features.",
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_repro_notebook(notebook_path, submission_path, metadata_path, args.experiment_id, args.slug, local_cv, args.base_id)

    notes = (
        f"Strict Q3-only GPS routine overlay on ID{args.base_id:03d}. Uses privacy-safe GPS routine numeric "
        "summaries with fold-nested Q3 candidate selection; all non-Q3 targets are preserved. No DACON submission "
        "was made."
    )
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=f"Strict Q3 GPS routine overlay on ID{args.base_id:03d}",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
