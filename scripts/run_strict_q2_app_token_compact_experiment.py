#!/usr/bin/env python3
"""Promote a compact Q2 top-app-token overlay on top of ID114."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_strict_q2_booster_compact_experiment import build_q2_booster_frames  # noqa: E402
from scripts.run_strict_raw_feature_experiment import (  # noqa: E402
    BASE_PREDICTION_DIR,
    NOTEBOOK_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
    TARGETS,
    map_predictions_to_original_sample,
    validate_submission,
)
from scripts.run_strict_shutdown_feature_set_selector_experiment import align_candidate  # noqa: E402
from src.utils.validation import clip_probabilities, make_subject_hole_folds  # noqa: E402

warnings.filterwarnings("ignore")

TARGET = "Q2"
DEFAULT_BASE_ID = 114
TOP_N_APPS = 80
BLEND_WEIGHTS = [0.15, 0.20, 0.25, 0.35, 0.50]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_app_name(value: object) -> str:
    text = str(value or "").strip().lower().replace("\xa0", "")
    text = re.sub(r"\s+", " ", text)
    return text if text else "unknown"


def safe_app_token(value: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z가-힣]+", "_", value).strip("_")[:30]
    return token or "app"


def load_base_artifact(
    train: pd.DataFrame,
    test: pd.DataFrame,
    base_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    oof_paths = sorted(BASE_PREDICTION_DIR.glob(f"base_oof_id_{base_id:03d}_*.csv"))
    test_paths = sorted(BASE_PREDICTION_DIR.glob(f"base_test_id_{base_id:03d}_*.csv"))
    if len(oof_paths) != 1 or len(test_paths) != 1:
        raise FileNotFoundError(f"Expected exactly one ID{base_id:03d} base OOF/test artifact.")
    base_oof = align_candidate(pd.read_csv(oof_paths[0]), train)
    base_test = align_candidate(pd.read_csv(test_paths[0]), test)
    return base_oof, base_test, f"{oof_paths[0].name}; {test_paths[0].name}"


def build_top_app_features(raw_dir: str | Path = "data/raw/ch2025_data_items", top_n: int = TOP_N_APPS) -> tuple[pd.DataFrame, list[str]]:
    usage = pd.read_parquet(Path(raw_dir) / "ch2025_mUsageStats.parquet")
    usage["timestamp"] = pd.to_datetime(usage["timestamp"])
    usage["date"] = usage["timestamp"].dt.date
    usage["hour_float"] = usage["timestamp"].dt.hour + usage["timestamp"].dt.minute / 60.0

    app_durations: dict[str, float] = defaultdict(float)
    for apps in usage["m_usage_stats"]:
        if not isinstance(apps, (list, np.ndarray)):
            continue
        for app in apps:
            if isinstance(app, dict):
                app_durations[clean_app_name(app.get("app_name", ""))] += float(app.get("total_time", 0) or 0)
    top_apps = [name for name, _ in sorted(app_durations.items(), key=lambda item: -item[1])[:top_n]]
    app_to_idx = {name: idx for idx, name in enumerate(top_apps)}
    windows = {"all12_24": (12.0, 24.0), "eve18_24": (18.0, 24.0), "late22_24": (22.0, 24.0)}

    rows: list[dict] = []
    for (subject_id, date), group in usage.groupby(["subject_id", "date"], sort=False):
        row: dict[str, float | str | object] = {"subject_id": subject_id, "date": date}
        for window_name, (lo, hi) in windows.items():
            subset = group[(group["hour_float"] >= lo) & (group["hour_float"] < hi)]
            top_totals = np.zeros(top_n, dtype=float)
            other_total = 0.0
            total = 0.0
            n_events = 0.0
            for apps in subset["m_usage_stats"]:
                if not isinstance(apps, (list, np.ndarray)):
                    continue
                for app in apps:
                    if not isinstance(app, dict):
                        continue
                    minutes = float(app.get("total_time", 0) or 0) / 60000.0
                    if minutes <= 0:
                        continue
                    total += minutes
                    n_events += 1.0
                    idx = app_to_idx.get(clean_app_name(app.get("app_name", "")))
                    if idx is None:
                        other_total += minutes
                    else:
                        top_totals[idx] += minutes

            denom = total + 1e-6
            shares = top_totals[top_totals > 0] / denom
            row[f"app_{window_name}_total_min"] = total
            row[f"app_{window_name}_other_min"] = other_total
            row[f"app_{window_name}_n_events"] = n_events
            row[f"app_{window_name}_entropy_top"] = float(-(shares * np.log(shares + 1e-9)).sum()) if shares.size else 0.0
            row[f"app_{window_name}_top_share"] = float(top_totals.max() / denom)
            for app_name, idx in app_to_idx.items():
                token = safe_app_token(app_name)
                row[f"app_{window_name}_{idx:02d}_{token}_min"] = np.log1p(top_totals[idx])
                row[f"app_{window_name}_{idx:02d}_{token}_share"] = top_totals[idx] / denom
        rows.append(row)
    return pd.DataFrame(rows), top_apps


def build_q2_app_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train, test, sample_original = build_q2_booster_frames()
    app_daily, top_apps = build_top_app_features()
    train_app = train.merge(app_daily, on=["subject_id", "date"], how="left")
    test_app = test.merge(app_daily, on=["subject_id", "date"], how="left")
    return train_app, test_app, sample_original, top_apps


def app_feature_columns(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    return [col for col in train.columns if col.startswith("app_") and col in test.columns]


def compact_app_model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=180,
        learning_rate=0.03,
        num_leaves=7,
        max_depth=3,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=10.0,
        random_state=42,
        verbosity=-1,
    )


def fit_predict_app_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    folds = make_subject_hole_folds(train, n_folds=5)
    y = train[TARGET].astype(int).reset_index(drop=True)
    oof = np.zeros(len(train), dtype=float)
    has_prediction = np.zeros(len(train), dtype=bool)
    test_pred = np.zeros(len(test), dtype=float)

    for train_idx, valid_idx in folds:
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        if y.iloc[train_idx].nunique() < 2:
            model = DummyClassifier(strategy="prior")
        else:
            model = clone(compact_app_model())
        model.fit(train.iloc[train_idx][feature_cols], y.iloc[train_idx])
        oof[valid_idx] = model.predict_proba(train.iloc[valid_idx][feature_cols])[:, 1]
        test_pred += model.predict_proba(test[feature_cols])[:, 1]
        has_prediction[valid_idx] = True

    if not has_prediction.all():
        raise ValueError("Folds did not produce predictions for every Q2 row.")
    return np.clip(oof, 1e-6, 1.0 - 1e-6), np.clip(test_pred / len(folds), 1e-6, 1.0 - 1e-6)


def predict_q2_app_token_compact_overlay(
    base_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict, pd.DataFrame]:
    train, test, sample_original, top_apps = build_q2_app_frames()
    base_oof, base_test, base_source = load_base_artifact(train, test, base_id)
    feature_cols = app_feature_columns(train, test)
    if not feature_cols:
        raise ValueError("No app token feature columns were generated.")

    y = train[TARGET].astype(int).values
    base_candidate_name = f"id{base_id:03d}_base"
    raw_candidate_name = "app_only__lgbm_l7_lr03_n180"
    raw_oof, raw_test = fit_predict_app_model(train=train, test=test, feature_cols=feature_cols)

    candidate_oof = {
        base_candidate_name: base_oof[TARGET].astype(float).to_numpy(),
        raw_candidate_name: raw_oof,
    }
    candidate_test = {
        base_candidate_name: base_test[TARGET].astype(float).to_numpy(),
        raw_candidate_name: raw_test,
    }
    candidate_details = {
        base_candidate_name: {"source": "base"},
        raw_candidate_name: {
            "feature_set": "app_only",
            "model": "lgbm_l7_lr03_n180",
            "n_features": len(feature_cols),
            "blend": False,
        },
    }
    for weight in BLEND_WEIGHTS:
        blend_name = f"blend_base{1.0 - weight:.2f}_{raw_candidate_name}_{weight:.2f}"
        candidate_oof[blend_name] = np.clip(
            (1.0 - weight) * candidate_oof[base_candidate_name] + weight * raw_oof,
            1e-6,
            1.0 - 1e-6,
        )
        candidate_test[blend_name] = np.clip(
            (1.0 - weight) * candidate_test[base_candidate_name] + weight * raw_test,
            1e-6,
            1.0 - 1e-6,
        )
        candidate_details[blend_name] = {
            "blend_candidate": raw_candidate_name,
            "base_weight": 1.0 - weight,
            "candidate_weight": weight,
        }

    candidate_scores = {
        name: float(log_loss(y, values, labels=[0, 1])) for name, values in sorted(candidate_oof.items())
    }
    folds = make_subject_hole_folds(train, n_folds=5)
    selected_q2_oof = np.zeros(len(train), dtype=float)
    nested_choices: list[dict] = []
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        losses = {
            name: log_loss(y[train_idx], values[train_idx], labels=[0, 1]) for name, values in candidate_oof.items()
        }
        best_name, best_loss = min(losses.items(), key=lambda item: item[1])
        selected_q2_oof[valid_idx] = candidate_oof[best_name][valid_idx]
        nested_choices.append(
            {
                "fold": fold_idx,
                "candidate": best_name,
                "train_fold_log_loss": float(best_loss),
                "candidate_losses": {name: float(value) for name, value in sorted(losses.items())},
            }
        )

    full_losses = {name: log_loss(y, values, labels=[0, 1]) for name, values in candidate_oof.items()}
    full_best_name, full_best_loss = min(full_losses.items(), key=lambda item: item[1])

    oof = base_oof.copy()
    test_pred = base_test.copy()
    oof[TARGET] = np.clip(selected_q2_oof, 1e-6, 1.0 - 1e-6)
    test_pred[TARGET] = np.clip(candidate_test[full_best_name], 1e-6, 1.0 - 1e-6)

    target_scores = {
        target: float(log_loss(train[target].astype(int), oof[target].astype(float), labels=[0, 1]))
        for target in TARGETS
    }
    selector_info = {
        "base_source": base_source,
        "feature_family": "top_app_token_usage",
        "top_n_apps": TOP_N_APPS,
        "top_apps": top_apps,
        "n_features": len(feature_cols),
        "candidate_pool_size": len(candidate_oof),
        "candidate_pool_origin": (
            "Compact Q2 top-app-token pool retained after reports/strict_q23_app_token_diagnostic_id114.json "
            "and confirmed in reports/strict_q2_app_token_compact_diagnostic_id114.json. The broad app-token "
            "pool overfit slightly under nested selection, but the one-family compact LGBM blend improved nested Q2."
        ),
        "q2_candidate_scores": candidate_scores,
        "candidate_details": candidate_details,
        "nested_choices": {TARGET: nested_choices},
        "full_train_test_choices": {
            TARGET: {
                "candidate": full_best_name,
                "full_oof_log_loss": float(full_best_loss),
                "candidate_losses": {name: float(value) for name, value in sorted(full_losses.items())},
            }
        },
        "selection_contract": (
            f"ID{base_id:03d} predictions are preserved for all targets except Q2. Q2 validation rows receive "
            "the compact app-token candidate selected using only each outer fold's training rows. Final test Q2 "
            "uses the candidate with the best full-training OOF log loss."
        ),
        "contamination_note": (
            "Top-app vocabulary is selected from unlabeled train/test app-usage rows and is therefore label-free "
            "transductive. No target labels are used to choose the vocabulary or construct app-token features."
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
            f"# Submission {experiment_id:03d}: Q2 app-token compact overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Method slug: `{slug}`\n"
            f"- Base artifact id: `{base_id:03d}`\n"
            f"- Local Subject-Hole CV: `{local_cv:.6f}`\n\n"
            "This notebook is a reproduction pointer for an immutable artifact. "
            f"The generating script keeps ID{base_id:03d} predictions for every non-Q2 target and replaces Q2 only."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/run_strict_q2_app_token_compact_experiment.py "
            f"--experiment-id {experiment_id} --slug {slug} --base-id {base_id}"
        ),
        nbf.v4.new_markdown_cell(
            "### Integrity note\n\n"
            f"The exact generated file is `{submission_path.name}`. "
            "Do not submit it before reviewing the metadata, audit reports, and validation gap."
        ),
    ]
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")


def append_submission_log(
    experiment_id: int,
    submission_path: Path,
    notebook_path: Path,
    local_cv: float,
    notes: str,
    base_id: int,
) -> None:
    log_path = SUBMISSIONS_DIR / "submission_log.csv"
    log_df = pd.read_csv(log_path, dtype=str).fillna("")
    if (log_df["id"].astype(str) == str(experiment_id)).any():
        raise ValueError(f"submission_log.csv already has id={experiment_id}")

    row = dict.fromkeys(log_df.columns, "")
    row.update(
        {
            "id": str(experiment_id),
            "date": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S"),
            "submission_file": submission_path.name,
            "model_description": f"Strict Q2 app-token compact overlay on ID{base_id:03d}",
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
    parser.add_argument("--slug", default="q2_app_token_compact_overlay")
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    args = parser.parse_args()

    if args.experiment_id < 94:
        raise ValueError("Strict experiments must use id >= 94.")

    oof, test_pred, target_scores, selector_info, sample_original = predict_q2_app_token_compact_overlay(
        base_id=args.base_id,
    )
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
        "method": "Q2 compact top-app-token overlay",
        "submission_file": submission_path.name,
        "notebook_file": str(notebook_path.relative_to(ROOT)),
        "submission_sha256": sha256_file(submission_path),
        "local_cv": local_cv,
        "target_scores": target_scores,
        "targets": TARGETS,
        "base_oof_file": str(base_oof_path.relative_to(ROOT)),
        "base_test_file": str(base_test_path.relative_to(ROOT)),
        "validation": "Subject-Hole CV with a Q2-only compact LGBM blend over top app-token usage features.",
        "selector_info": selector_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    write_repro_notebook(
        notebook_path=notebook_path,
        submission_path=submission_path,
        metadata_path=metadata_path,
        experiment_id=args.experiment_id,
        slug=args.slug,
        local_cv=local_cv,
        base_id=args.base_id,
    )

    notes = (
        f"Strict Q2-only compact app-token overlay on ID{args.base_id:03d}. Uses top-app duration/share "
        "features from raw mUsageStats; broad app-token search overfit slightly, but the compact one-family "
        "LGBM blend improved nested Q2 and changes test predictions. No DACON submission was made."
    )
    append_submission_log(
        experiment_id=args.experiment_id,
        submission_path=submission_path,
        notebook_path=notebook_path,
        local_cv=local_cv,
        notes=notes,
        base_id=args.base_id,
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
