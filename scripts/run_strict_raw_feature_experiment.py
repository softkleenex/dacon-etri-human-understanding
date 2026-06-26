#!/usr/bin/env python3
"""Run a strict raw-feature ExtraTrees experiment.

This promotes a fixed, non-Optuna raw feature model into the strict artifact
contract. The default configuration intentionally avoids future target
interpolation; only fold-wise same-subject target priors are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.train_ultimate_pipeline_clean_v32 import (  # noqa: E402
    add_clean_base_features,
    add_clean_rolling_features,
)
from src.utils.validation import clip_probabilities, make_subject_hole_folds, validate_submission  # noqa: E402

SUBMISSIONS_DIR = ROOT / "submissions"
BASE_PREDICTION_DIR = ROOT / "artifacts" / "base_predictions"
NOTEBOOK_DIR = ROOT / "notebooks" / "history"
REPORTS_DIR = ROOT / "reports"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


def extract_date(df: pd.DataFrame) -> pd.Series:
    for col in ["date", "lifelog_date", "sleep_date"]:
        if col in df.columns:
            return pd.to_datetime(df[col]).dt.date
    raise ValueError("No usable date/lifelog_date/sleep_date column found.")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_known_covariate_features(df: pd.DataFrame, id_col: str = "subject_id") -> pd.DataFrame:
    """Add label-free date/position features using known train+test covariates."""

    out = df.copy()
    out["date_dt"] = pd.to_datetime(out["date"])
    out["date_ord"] = out["date_dt"].map(lambda value: float(value.toordinal()))
    out["dayofyear2"] = out["date_dt"].dt.dayofyear.astype(float)
    out["weekofyear"] = out["date_dt"].dt.isocalendar().week.astype(float)
    out["subject_code"] = out[id_col].str.extract(r"(\d+)").astype(float)[0]
    out["subject_date_rank_all"] = out.groupby(id_col)["date_dt"].rank(method="dense").astype(float)
    out["subject_n_dates_all"] = out.groupby(id_col)["date_dt"].transform("nunique").astype(float)
    out["subject_date_frac_all"] = out["subject_date_rank_all"] / (out["subject_n_dates_all"] + 1.0)
    subject_min = out.groupby(id_col)["date_dt"].transform("min")
    subject_max = out.groupby(id_col)["date_dt"].transform("max")
    out["days_since_subject_start_all"] = (out["date_dt"] - subject_min).dt.days.astype(float)
    out["days_to_subject_end_all"] = (subject_max - out["date_dt"]).dt.days.astype(float)
    return out


def build_feature_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_raw_path = ROOT / "data" / "processed" / "df_train_raw.parquet"
    test_raw_path = ROOT / "data" / "processed" / "df_test_raw.parquet"
    metrics_path = ROOT / "data" / "raw" / "ch2026_metrics_train.csv"
    sample_path = ROOT / "data" / "raw" / "ch2026_submission_sample.csv"

    if not train_raw_path.exists() or not test_raw_path.exists():
        raise FileNotFoundError("Missing processed raw parquet caches. Rebuild them before this experiment.")

    train_raw = pd.read_parquet(train_raw_path)
    test_raw = pd.read_parquet(test_raw_path)
    metrics = pd.read_csv(metrics_path)
    sample = pd.read_csv(sample_path)

    train_raw["is_test"] = 0
    test_raw["is_test"] = 1
    all_raw = pd.concat([train_raw, test_raw], ignore_index=True)
    all_features = add_clean_base_features(all_raw)
    all_features = add_clean_rolling_features(all_features, id_col="subject_id", date_col="date")
    all_features = add_known_covariate_features(all_features)

    train = (
        all_features[all_features["is_test"] == 0]
        .drop(columns=["is_test"])
        .sort_values(["subject_id", "date"])
        .reset_index(drop=True)
    )
    test = (
        all_features[all_features["is_test"] == 1]
        .drop(columns=["is_test"])
        .sort_values(["subject_id", "date"])
        .reset_index(drop=True)
    )

    metrics = metrics.copy()
    metrics["date"] = extract_date(metrics)
    metrics_sorted = metrics.sort_values(["subject_id", "date"]).reset_index(drop=True)

    sample_sorted = sample.copy()
    sample_sorted["date"] = extract_date(sample_sorted)
    sample_sorted = sample_sorted.sort_values(["subject_id", "date"]).reset_index(drop=True)

    train_key = train[["subject_id", "date"]].copy()
    train_key["date"] = pd.to_datetime(train_key["date"]).dt.date
    test_key = test[["subject_id", "date"]].copy()
    test_key["date"] = pd.to_datetime(test_key["date"]).dt.date

    if not train_key.reset_index(drop=True).equals(metrics_sorted[["subject_id", "date"]]):
        raise ValueError("Processed train frame does not align with sorted metrics.")
    if not test_key.reset_index(drop=True).equals(sample_sorted[["subject_id", "date"]]):
        raise ValueError("Processed test frame does not align with sorted sample submission.")

    return train, test, metrics_sorted, sample


def numeric_feature_columns(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    excluded = set(TARGETS + ["subject_id", "sleep_date", "lifelog_date", "date", "date_dt"])
    cols: list[str] = []
    for col in train.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(train[col]) and (train[col].notna().any() or test[col].notna().any()):
            cols.append(col)
    return cols


def make_model(seed: int = 42) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=10,
        max_features=0.5,
        random_state=seed,
        n_jobs=-1,
    )


def fixed_raw_model_pool() -> dict[str, object]:
    return {
        "et_d6_l5": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "et_d8_l10": make_model(),
        "rf_d6_l5": RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "hgb_l2": HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.03,
            l2_regularization=2.0,
            max_leaf_nodes=7,
            random_state=42,
        ),
    }


def add_past_target_features(
    df: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    target: str,
    id_col: str = "subject_id",
) -> pd.DataFrame:
    """Create fold-wise target features that only look backward in time."""

    tmp = df[[id_col, "date"]].copy()
    tmp["original_index"] = np.arange(len(df))
    tmp["date_dt"] = pd.to_datetime(tmp["date"])
    tmp["y_train"] = np.nan
    tmp.loc[train_idx, "y_train"] = np.asarray(y)[train_idx]
    tmp = tmp.sort_values([id_col, "date_dt"]).reset_index(drop=True)
    tmp["known_date"] = tmp["date_dt"].where(tmp["y_train"].notna())

    tmp["prev_source"] = tmp.groupby(id_col)["y_train"].shift(1)
    tmp["prev_date_source"] = tmp.groupby(id_col)["known_date"].shift(1)
    tmp[f"{target}_past_prev"] = tmp.groupby(id_col)["prev_source"].ffill()
    tmp[f"{target}_past_prev_date"] = tmp.groupby(id_col)["prev_date_source"].ffill()

    tmp["known_count"] = tmp.groupby(id_col)["y_train"].transform(lambda series: series.notna().cumsum())
    tmp["known_sum"] = tmp.groupby(id_col)["y_train"].transform(lambda series: series.fillna(0).cumsum())
    tmp["past_count_raw"] = tmp["known_count"] - tmp["y_train"].notna().astype(int)
    tmp["past_sum_raw"] = tmp["known_sum"] - tmp["y_train"].fillna(0)
    tmp[f"{target}_past_exp_mean_raw"] = tmp["past_sum_raw"] / tmp["past_count_raw"].replace(0, np.nan)
    tmp[f"{target}_past_exp_mean"] = tmp.groupby(id_col)[f"{target}_past_exp_mean_raw"].ffill()

    y_series = pd.Series(y).reset_index(drop=True)
    subject_means = y_series.iloc[train_idx].groupby(df.iloc[train_idx][id_col]).mean()
    global_mean = float(y_series.iloc[train_idx].mean())
    tmp[f"{target}_subject_prior"] = tmp[id_col].map(subject_means).fillna(global_mean)
    tmp[f"{target}_past_prev"] = tmp[f"{target}_past_prev"].fillna(tmp[f"{target}_subject_prior"])
    tmp[f"{target}_past_exp_mean"] = tmp[f"{target}_past_exp_mean"].fillna(tmp[f"{target}_subject_prior"])

    delta_prev = (tmp["date_dt"] - pd.to_datetime(tmp[f"{target}_past_prev_date"])).dt.days
    tmp[f"{target}_past_log_delta_prev"] = np.log1p(delta_prev.fillna(999.0))
    tmp = tmp.sort_values("original_index")
    return tmp[
        [
            f"{target}_subject_prior",
            f"{target}_past_prev",
            f"{target}_past_exp_mean",
            f"{target}_past_log_delta_prev",
        ]
    ].reset_index(drop=True)


def augment_with_target_variant(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    target: str,
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    id_col = "subject_id"
    train_aug = train.reset_index(drop=True).copy()
    test_aug = test.reset_index(drop=True).copy()
    if variant == "base":
        return train_aug, test_aug, []

    if variant == "subject_prior":
        subject_means = y.iloc[train_idx].groupby(train.iloc[train_idx][id_col]).mean()
        global_mean = float(y.iloc[train_idx].mean())
        prior_col = f"{target}_subject_prior"
        train_aug[prior_col] = train_aug[id_col].map(subject_means).fillna(global_mean)
        test_aug[prior_col] = test_aug[id_col].map(subject_means).fillna(global_mean)
        return train_aug, test_aug, [prior_col]

    if variant == "past":
        combined = pd.concat([train_aug, test_aug], ignore_index=True, sort=False)
        combined_y = pd.concat([y.reset_index(drop=True), pd.Series(np.nan, index=range(len(test_aug)))])
        past_features = add_past_target_features(combined, combined_y, train_idx, target)
        train_past = past_features.iloc[: len(train_aug)].reset_index(drop=True)
        test_past = past_features.iloc[len(train_aug) :].reset_index(drop=True)
        train_aug = pd.concat([train_aug, train_past], axis=1)
        test_aug = pd.concat([test_aug, test_past], axis=1)
        return train_aug, test_aug, list(past_features.columns)

    raise ValueError(f"Unknown raw feature variant: {variant}")


def fit_predict_fold_model(
    model_proto: object,
    train_aug: pd.DataFrame,
    test_aug: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    active_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train_aug.iloc[train_idx][active_cols])
    x_valid = imputer.transform(train_aug.iloc[valid_idx][active_cols])
    x_test = imputer.transform(test_aug[active_cols])

    if y.iloc[train_idx].nunique() < 2:
        model = DummyClassifier(strategy="prior")
    else:
        model = clone(model_proto)
    model.fit(x_train, y.iloc[train_idx])
    return model.predict_proba(x_valid)[:, 1], model.predict_proba(x_test)[:, 1]


def predict_raw_candidate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    variant: str,
    model_proto: object,
) -> tuple[np.ndarray, np.ndarray]:
    folds = make_subject_hole_folds(train, n_folds=5)
    oof = np.zeros((len(train), len(TARGETS)), dtype=float)
    test_pred = np.zeros((len(test), len(TARGETS)), dtype=float)

    for target_idx, target in enumerate(TARGETS):
        y = train[target].astype(int).reset_index(drop=True)
        fold_count = 0
        for train_idx, valid_idx in folds:
            train_idx = np.asarray(train_idx)
            valid_idx = np.asarray(valid_idx)
            train_aug, test_aug, extra_cols = augment_with_target_variant(
                train=train,
                test=test,
                y=y,
                train_idx=train_idx,
                target=target,
                variant=variant,
            )
            valid_pred, test_fold_pred = fit_predict_fold_model(
                model_proto=model_proto,
                train_aug=train_aug,
                test_aug=test_aug,
                y=y,
                train_idx=train_idx,
                valid_idx=valid_idx,
                active_cols=feature_cols + extra_cols,
            )
            oof[valid_idx, target_idx] = valid_pred
            test_pred[:, target_idx] += test_fold_pred
            fold_count += 1
        test_pred[:, target_idx] /= fold_count

    return np.clip(oof, 1e-6, 1.0 - 1e-6), np.clip(test_pred, 1e-6, 1.0 - 1e-6)


def predict_raw_et_subject_prior(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    id_col = "subject_id"
    folds = make_subject_hole_folds(train, n_folds=5)
    oof = pd.DataFrame({"subject_id": train[id_col].values, "date": pd.to_datetime(train["date"]).dt.date})
    test_pred = pd.DataFrame({"subject_id": test[id_col].values, "date": pd.to_datetime(test["date"]).dt.date})
    target_scores: dict[str, float] = {}

    for target in TARGETS:
        y = train[target].astype(int).reset_index(drop=True)
        oof_values = np.zeros(len(train), dtype=float)
        has_prediction = np.zeros(len(train), dtype=bool)
        test_values = np.zeros(len(test), dtype=float)
        fold_count = 0

        for train_idx, valid_idx in folds:
            train_idx = np.asarray(train_idx)
            valid_idx = np.asarray(valid_idx)
            train_fold = train.copy()
            test_fold = test.copy()
            subject_means = y.iloc[train_idx].groupby(train.iloc[train_idx][id_col]).mean()
            global_mean = float(y.iloc[train_idx].mean())
            prior_col = f"{target}_subject_prior"
            train_fold[prior_col] = train_fold[id_col].map(subject_means).fillna(global_mean)
            test_fold[prior_col] = test_fold[id_col].map(subject_means).fillna(global_mean)
            active_cols = feature_cols + [prior_col]

            imputer = SimpleImputer(strategy="median")
            x_train = imputer.fit_transform(train_fold.iloc[train_idx][active_cols])
            x_valid = imputer.transform(train_fold.iloc[valid_idx][active_cols])
            x_test = imputer.transform(test_fold[active_cols])

            if y.iloc[train_idx].nunique() < 2:
                model = DummyClassifier(strategy="prior")
            else:
                model = make_model()
            model.fit(x_train, y.iloc[train_idx])
            oof_values[valid_idx] = model.predict_proba(x_valid)[:, 1]
            test_values += model.predict_proba(x_test)[:, 1]
            has_prediction[valid_idx] = True
            fold_count += 1

        if not has_prediction.all():
            raise ValueError(f"Folds did not produce predictions for every row for target {target}.")

        oof[target] = np.clip(oof_values, 1e-6, 1.0 - 1e-6)
        test_pred[target] = np.clip(test_values / fold_count, 1e-6, 1.0 - 1e-6)
        target_scores[target] = float(log_loss(y, oof[target], labels=[0, 1]))

    return oof, test_pred, target_scores


def frame_from_prediction_array(frame: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"subject_id": frame["subject_id"].values, "date": pd.to_datetime(frame["date"]).dt.date})
    for target_idx, target in enumerate(TARGETS):
        out[target] = np.clip(values[:, target_idx], 1e-6, 1.0 - 1e-6)
    return out


def predict_nested_raw_selector(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict]:
    candidate_oof: dict[str, np.ndarray] = {}
    candidate_test: dict[str, np.ndarray] = {}

    for variant in ["base", "subject_prior", "past"]:
        for model_name, model_proto in fixed_raw_model_pool().items():
            candidate_name = f"{variant}_{model_name}"
            oof_values, test_values = predict_raw_candidate(
                train=train,
                test=test,
                feature_cols=feature_cols,
                variant=variant,
                model_proto=model_proto,
            )
            candidate_oof[candidate_name] = oof_values
            candidate_test[candidate_name] = test_values

    average_pairs = [
        ("subject_prior_et_d8_l10", "past_rf_d6_l5"),
        ("base_et_d8_l10", "past_rf_d6_l5"),
        ("subject_prior_et_d8_l10", "past_et_d6_l5"),
    ]
    for left, right in average_pairs:
        name = f"avg_{left}__{right}"
        candidate_oof[name] = 0.5 * candidate_oof[left] + 0.5 * candidate_oof[right]
        candidate_test[name] = 0.5 * candidate_test[left] + 0.5 * candidate_test[right]

    folds = make_subject_hole_folds(train, n_folds=5)
    selected_oof = np.zeros((len(train), len(TARGETS)), dtype=float)
    nested_choices: dict[str, list[dict]] = {target: [] for target in TARGETS}

    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        for target_idx, target in enumerate(TARGETS):
            y = train[target].astype(int).values
            losses = {
                name: log_loss(y[train_idx], values[train_idx, target_idx], labels=[0, 1])
                for name, values in candidate_oof.items()
            }
            best_name, best_loss = min(losses.items(), key=lambda item: item[1])
            selected_oof[valid_idx, target_idx] = candidate_oof[best_name][valid_idx, target_idx]
            nested_choices[target].append(
                {"fold": fold_idx, "candidate": best_name, "train_fold_log_loss": float(best_loss)}
            )

    target_scores = {
        target: float(log_loss(train[target].astype(int), selected_oof[:, target_idx], labels=[0, 1]))
        for target_idx, target in enumerate(TARGETS)
    }

    selected_test = np.zeros((len(test), len(TARGETS)), dtype=float)
    full_train_choices: dict[str, dict] = {}
    for target_idx, target in enumerate(TARGETS):
        y = train[target].astype(int).values
        losses = {name: log_loss(y, values[:, target_idx], labels=[0, 1]) for name, values in candidate_oof.items()}
        best_name, best_loss = min(losses.items(), key=lambda item: item[1])
        selected_test[:, target_idx] = candidate_test[best_name][:, target_idx]
        full_train_choices[target] = {"candidate": best_name, "full_oof_log_loss": float(best_loss)}

    selector_info = {
        "candidate_names": sorted(candidate_oof),
        "nested_choices": nested_choices,
        "full_train_test_choices": full_train_choices,
        "selection_contract": (
            "OOF score is nested: every outer validation hole receives predictions from the candidate selected "
            "using only that outer fold's training rows. Final test predictions use the candidate with the best "
            "full-training OOF score per target."
        ),
    }
    return (
        frame_from_prediction_array(train, selected_oof),
        frame_from_prediction_array(test, selected_test),
        target_scores,
        selector_info,
    )


def map_predictions_to_original_sample(sample_original: pd.DataFrame, test_pred: pd.DataFrame) -> pd.DataFrame:
    id_col = "subject_id"
    pred = test_pred.copy()
    pred["date"] = pd.to_datetime(pred["date"]).dt.date
    pred_maps = {target: pred.set_index([id_col, "date"])[target].to_dict() for target in TARGETS}

    out = sample_original.copy()
    original_dates = extract_date(out)
    keys = list(zip(out[id_col], original_dates, strict=True))
    for target in TARGETS:
        out[target] = [pred_maps[target][key] for key in keys]
    return clip_probabilities(out, TARGETS)


def write_repro_notebook(
    notebook_path: Path,
    submission_path: Path,
    metadata_path: Path,
    experiment_id: int,
    slug: str,
    method: str,
    local_cv: float,
) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id:03d}: strict raw-feature ExtraTrees\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Method: `{method}`\n"
            f"- Local Subject-Hole CV: `{local_cv:.6f}`\n\n"
            "This notebook is a reproduction pointer for an immutable artifact. "
            "The generating script uses fixed ExtraTrees hyperparameters, fold-wise same-subject priors, "
            "and no future-target interpolation or same-OOF hyperparameter search."
        ),
        nbf.v4.new_code_cell(
            f"!python scripts/run_strict_raw_feature_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --method {method}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\n"
            f"The exact generated file is `{submission_path.name}`. "
            "Do not submit it before reviewing the metadata, audit reports, and validation gap."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def append_submission_log(
    log_path: Path,
    experiment_id: int,
    submission_path: Path,
    notebook_path: Path,
    local_cv: float,
    notes: str,
    model_description: str,
) -> None:
    log_df = pd.read_csv(log_path, dtype=str).fillna("")
    if (log_df["id"].astype(str) == str(experiment_id)).any():
        raise ValueError(f"submission_log.csv already has id={experiment_id}")

    row = dict.fromkeys(log_df.columns, "")
    row.update(
        {
            "id": str(experiment_id),
            "date": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S"),
            "submission_file": submission_path.name,
            "model_description": model_description,
            "local_cv_score": f"{local_cv:.6f}",
            "public_lb_score": "",
            "is_submitted": "False",
            "notes": notes,
            "is_submitted_bool": "False",
            "local_cv_score_num": f"{local_cv:.6f}",
            "public_lb_score_num": "",
            "cv_lb_gap": "",
            "artifact_exists": "True",
            "status_clean": "generated_unsubmitted",
            "risk_reason": "Generated locally but not submitted/scored.",
            "safe_for_default_ensemble": "False",
            "artifact_role": "submission",
            "duplicate_file_count": "1",
            "notebook_file": str(notebook_path.relative_to(ROOT)),
        }
    )
    pd.concat([log_df, pd.DataFrame([row])], ignore_index=True).to_csv(log_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--slug", default="raw_et_subject_prior")
    parser.add_argument(
        "--method",
        choices=["subject-prior-et", "raw-nested-selector"],
        default="subject-prior-et",
    )
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict experiments must use id >= 94.")

    train, test, _metrics_sorted, sample_original = build_feature_frames()
    feature_cols = numeric_feature_columns(train, test)
    selector_info: dict = {}
    if args.method == "raw-nested-selector":
        oof, test_pred, target_scores, selector_info = predict_nested_raw_selector(train, test, feature_cols)
    else:
        oof, test_pred, target_scores = predict_raw_et_subject_prior(train, test, feature_cols)
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

    out = map_predictions_to_original_sample(sample_original, test_pred)
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
        "method": args.method,
        "submission_file": submission_path.name,
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "target_scores": target_scores,
        "targets": TARGETS,
        "n_features": len(feature_cols),
        "feature_contract": (
            "Known train+test covariates are used for label-free date/rank features; "
            "target-derived subject priors are fit inside each Subject-Hole fold only; "
            "no future target interpolation or same-OOF hyperparameter search."
        ),
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": "Subject-Hole CV with fixed hyperparameters selected before artifact generation.",
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    write_repro_notebook(
        notebook_path=notebook_path,
        submission_path=submission_path,
        metadata_path=metadata_path,
        experiment_id=args.experiment_id,
        slug=args.slug,
        method=args.method,
        local_cv=local_cv,
    )

    if args.method == "raw-nested-selector":
        notes = (
            "Strict raw-feature nested selector over fixed base/subject-prior/past-only raw candidates. "
            "Nested OOF candidate selection uses only each outer fold's training rows; no future target "
            "interpolation, Optuna, same-OOF clipping, or DACON submission."
        )
        model_description = "Strict raw-feature nested selector without future target interpolation"
    else:
        notes = (
            "Strict raw-feature ExtraTrees artifact using fixed et_d8_l10 parameters and fold-wise subject priors. "
            "No future target interpolation, Optuna, same-OOF clipping, or DACON submission. "
            "Created as a cleaner new-signal base after ID97 trimmed stacker CV 0.617675."
        )
        model_description = "Strict raw-feature ExtraTrees with fold-wise subject priors"
    append_submission_log(
        log_path=SUBMISSIONS_DIR / "submission_log.csv",
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        model_description=model_description,
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
