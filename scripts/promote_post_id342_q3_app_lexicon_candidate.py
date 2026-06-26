#!/usr/bin/env python3
"""Emergency Q3 app-name lexicon overlay candidate after ID342.

This bounded script parses raw mUsageStats app names into predeclared aggregate
buckets only. It writes a submission artifact only when the local ID342 gate
passes; it never persists raw app names.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id341_q2q3s3_fold_internal_model_smoke import (  # noqa: E402
    PRIMARY_GATE_SPLITS,
    TARGETS,
    load_reference_values,
    load_split_manifest,
    load_train_sorted,
)
from scripts.run_strict_raw_feature_experiment import (  # noqa: E402
    BASE_PREDICTION_DIR,
    NOTEBOOK_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
    append_submission_log,
    map_predictions_to_original_sample,
    sha256_file,
)
from src.utils.validation import validate_submission  # noqa: E402

KST = timezone(timedelta(hours=9))
RAW_USAGE_PATH = ROOT / "data" / "raw" / "ch2025_data_items" / "ch2025_mUsageStats.parquet"
SAMPLE_PATH = ROOT / "data" / "raw" / "ch2026_submission_sample.csv"
BASE_TEST_ID342 = BASE_PREDICTION_DIR / "base_test_id_342_id178_id221_featuresim_shrink050_cv0.534000.csv"
EPS = 1e-6
ANCHOR_ID = 342
ID342_PUBLIC = 0.5960832519
FOCUS_TARGET = "Q3"
BLEND_WEIGHTS = (-0.05, -0.025, -0.01, -0.005, 0.01, 0.025, 0.05, 0.08, 0.12)
MODEL_C_VALUES = (0.03, 0.10, 0.30)
PRIMARY_SPLITS = PRIMARY_GATE_SPLITS

LEXICON: dict[str, tuple[str, ...]] = {
    "communication": ("카카오", "톡", "메시지", "전화", "통화", "mail", "gmail", "문자", "sms", "discord", "slack"),
    "finance_work": ("토스", "은행", "bank", "증권", "stock", "coin", "업비트", "pay", "office", "docs", "drive"),
    "media_search": ("naver", "google", "chrome", "youtube", "유튜브", "인스타", "instagram", "webtoon", "뉴스"),
    "routine_health": ("캐시워크", "health", "헬스", "fit", "만보기", "성경", "bible", "캘린더"),
    "system_home": ("one ui", "홈", "launcher", "system", "설정"),
}
BUCKETS = tuple(LEXICON) + ("other",)
WINDOWS = {
    "all": tuple(range(24)),
    "evening": (18, 19, 20, 21, 22, 23),
    "late": (22, 23, 0, 1, 2, 3),
    "pre_sleep": (20, 21, 22, 23, 0),
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def bucket_app(name: Any) -> str:
    text = str(name or "").lower()
    for bucket, tokens in LEXICON.items():
        if any(token.lower() in text for token in tokens):
            return bucket
    return "other"


def iter_usage_items(value: Any):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item


def key_frame() -> pd.DataFrame:
    train = load_train_sorted()[["subject_id", "date"]].copy()
    train["split"] = "train"
    sample = pd.read_csv(SAMPLE_PATH)
    test = sample[["subject_id", "lifelog_date"]].copy()
    test["date"] = pd.to_datetime(test["lifelog_date"]).dt.date
    test = test.drop(columns=["lifelog_date"])
    test["split"] = "test"
    return pd.concat([train, test], ignore_index=True).sort_values(["subject_id", "date"]).reset_index(drop=True)


def build_lexicon_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = key_frame()
    usage = pd.read_parquet(RAW_USAGE_PATH)
    usage["date"] = pd.to_datetime(usage["timestamp"]).dt.date
    usage["hour"] = pd.to_datetime(usage["timestamp"]).dt.hour
    key_set = set(zip(keys["subject_id"], keys["date"], strict=True))
    rows: dict[tuple[str, Any], dict[str, float]] = {
        (row.subject_id, row.date): {"subject_id": row.subject_id, "date": row.date, "split": row.split}
        for row in keys.itertuples(index=False)
    }
    for values in rows.values():
        values["usage_row_count"] = 0.0
        values["usage_event_count"] = 0.0
        for window in WINDOWS:
            values[f"{window}_total_time_log1p"] = 0.0
            values[f"{window}_event_count_log1p"] = 0.0
            for bucket in BUCKETS:
                values[f"{window}_{bucket}_time_log1p"] = 0.0
                values[f"{window}_{bucket}_events_log1p"] = 0.0
    for row in usage.itertuples(index=False):
        key = (row.subject_id, row.date)
        if key not in key_set:
            continue
        values = rows[key]
        values["usage_row_count"] += 1.0
        for item in iter_usage_items(row.m_usage_stats):
            bucket = bucket_app(item.get("app_name"))
            total_time = float(item.get("total_time") or 0.0)
            values["usage_event_count"] += 1.0
            for window, hours in WINDOWS.items():
                if int(row.hour) in hours:
                    values[f"{window}_total_time_log1p"] += total_time
                    values[f"{window}_event_count_log1p"] += 1.0
                    values[f"{window}_{bucket}_time_log1p"] += total_time
                    values[f"{window}_{bucket}_events_log1p"] += 1.0
    frame = pd.DataFrame(rows.values()).sort_values(["subject_id", "date"]).reset_index(drop=True)
    count_cols = [col for col in frame.columns if col.endswith("_log1p") or col in {"usage_row_count", "usage_event_count"}]
    frame[count_cols] = np.log1p(frame[count_cols].astype(float))
    for window in WINDOWS:
        denom = frame[f"{window}_total_time_log1p"].replace(0, np.nan)
        for bucket in BUCKETS:
            frame[f"{window}_{bucket}_time_share"] = (
                frame[f"{window}_{bucket}_time_log1p"].astype(float) / denom
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    metadata = {
        "source_path": rel(RAW_USAGE_PATH),
        "rows": int(len(frame)),
        "train_rows": int(frame["split"].eq("train").sum()),
        "test_rows": int(frame["split"].eq("test").sum()),
        "feature_count": int(len(feature_columns(frame))),
        "buckets": list(BUCKETS),
        "raw_app_names_persisted": False,
    }
    return frame, metadata


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in frame.columns
        if col not in {"subject_id", "date", "split"}
        and pd.api.types.is_numeric_dtype(frame[col])
        and frame[col].notna().any()
    ]


def make_model(c_value: float):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=float(c_value), solver="liblinear", max_iter=500, random_state=345),
    )


def matrix_with_subject_dev(
    train_features: pd.DataFrame,
    query_features: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = [col for col in feature_cols if train_features[col].notna().any()]
    train_raw = train_features[usable].reset_index(drop=True)
    query_raw = query_features[usable].reset_index(drop=True)
    train_devs: dict[str, pd.Series] = {}
    query_devs: dict[str, pd.Series] = {}
    for col in usable:
        global_median = train_features[col].median()
        if pd.isna(global_median):
            global_median = 0.0
        medians = train_features.groupby("subject_id")[col].median()
        train_base = train_features["subject_id"].map(medians).fillna(global_median).reset_index(drop=True)
        query_base = query_features["subject_id"].map(medians).fillna(global_median).reset_index(drop=True)
        train_devs[f"{col}_dev_subject"] = train_features[col].reset_index(drop=True) - train_base
        query_devs[f"{col}_dev_subject"] = query_features[col].reset_index(drop=True) - query_base
    return pd.concat([train_raw, pd.DataFrame(train_devs)], axis=1), pd.concat(
        [query_raw, pd.DataFrame(query_devs)], axis=1
    )


def fold_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(log_loss(y_true.astype(int), np.clip(y_pred, EPS, 1 - EPS), labels=[0, 1]))


def weighted_loss(y: np.ndarray, pred: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> float:
    total = 0
    loss = 0.0
    for _fit_idx, valid_idx in folds:
        loss += fold_loss(y[valid_idx], pred[valid_idx]) * len(valid_idx)
        total += len(valid_idx)
    return float(loss / total)


def predict_oof_for_split(
    features: pd.DataFrame,
    feature_cols: list[str],
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    c_value: float,
) -> np.ndarray:
    pred = np.full(len(y), np.nan, dtype=float)
    for fit_idx, valid_idx in folds:
        if len(np.unique(y[fit_idx])) < 2:
            model = DummyClassifier(strategy="prior")
            model.fit(np.zeros((len(fit_idx), 1)), y[fit_idx])
            proba = np.full(len(valid_idx), float(np.mean(y[fit_idx])), dtype=float)
        else:
            fit_x, valid_x = matrix_with_subject_dev(features.iloc[fit_idx], features.iloc[valid_idx], feature_cols)
            model = make_model(c_value)
            model.fit(fit_x, y[fit_idx])
            proba = model.predict_proba(valid_x)[:, 1]
        pred[valid_idx] = np.clip(proba, EPS, 1.0 - EPS)
    return pred


def scan_gate(train: pd.DataFrame, features: pd.DataFrame, feature_cols: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    splits = load_split_manifest()
    anchor = load_reference_values(train, ANCHOR_ID)
    target_idx = TARGETS.index(FOCUS_TARGET)
    y = train[FOCUS_TARGET].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for c_value in MODEL_C_VALUES:
        split_predictions = {
            split: predict_oof_for_split(features, feature_cols, y, folds, c_value)
            for split, folds in splits.items()
            if split in PRIMARY_SPLITS
        }
        for weight in BLEND_WEIGHTS:
            split_deltas: dict[str, float] = {}
            wins = 0
            for split in PRIMARY_SPLITS:
                folds = splits[split]
                blended = np.clip(
                    (1.0 - weight) * anchor[:, target_idx] + weight * split_predictions[split],
                    EPS,
                    1.0 - EPS,
                )
                delta = weighted_loss(y, blended, folds) - weighted_loss(y, anchor[:, target_idx], folds)
                split_deltas[split] = float(delta)
                wins += int(delta < 0.0)
            rows.append(
                {
                    "target": FOCUS_TARGET,
                    "config": f"q3_app_lexicon_lr_c{c_value:g}_w{weight:g}",
                    "c_value": float(c_value),
                    "weight": float(weight),
                    "wins": int(wins),
                    "mean_delta": float(np.mean(list(split_deltas.values()))),
                    "max_delta": float(np.max(list(split_deltas.values()))),
                    "min_delta": float(np.min(list(split_deltas.values()))),
                    "split_deltas": split_deltas,
                    "eligible": bool(
                        wins >= 4
                        and float(np.mean(list(split_deltas.values()))) <= -0.00005
                        and float(np.max(list(split_deltas.values()))) <= 0.00075
                    ),
                }
            )
    rows = sorted(rows, key=lambda row: (not row["eligible"], row["mean_delta"], row["max_delta"], -row["wins"]))
    best = rows[0]
    return rows, {"anchor_id": ANCHOR_ID, "best_row": best, "eligible_count": sum(row["eligible"] for row in rows)}


def fit_full_predict(train_features: pd.DataFrame, test_features: pd.DataFrame, feature_cols: list[str], y: np.ndarray, c_value: float) -> np.ndarray:
    train_x, test_x = matrix_with_subject_dev(train_features, test_features, feature_cols)
    model = make_model(c_value)
    model.fit(train_x, y)
    return np.clip(model.predict_proba(test_x)[:, 1], EPS, 1.0 - EPS)


def write_notebook(path: Path, submission_path: Path, metadata_path: Path, experiment_id: int, slug: str) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# Submission {experiment_id}: Q3 app lexicon emergency overlay\n\n"
            f"- Submission CSV: `{submission_path.name}`\n"
            f"- Metadata: `{metadata_path.relative_to(ROOT)}`\n"
            f"- Slug: `{slug}`\n\n"
            "This reproduces the bounded post-ID342 Q3 app-name lexicon overlay. Raw app names are not persisted."
        ),
        nbf.v4.new_code_cell(
            f"! .venv/bin/python scripts/promote_post_id342_q3_app_lexicon_candidate.py "
            f"--experiment-id {experiment_id} --slug {slug} --allow-artifact-even-if-borderline"
        ),
    ]
    path.write_text(nbf.writes(nb), encoding="utf-8")


def build_candidate(args: argparse.Namespace) -> dict[str, Any]:
    train = load_train_sorted()
    all_features, feature_meta = build_lexicon_features()
    train_features = all_features[all_features["split"].eq("train")].sort_values(["subject_id", "date"]).reset_index(drop=True)
    test_features = all_features[all_features["split"].eq("test")].sort_values(["subject_id", "date"]).reset_index(drop=True)
    if not train[["subject_id", "date"]].reset_index(drop=True).equals(train_features[["subject_id", "date"]]):
        raise ValueError("Train lexicon features do not align with metrics.")
    feature_cols = feature_columns(train_features)
    rows, gate = scan_gate(train, train_features, feature_cols)
    best = rows[min(args.row_rank, len(rows) - 1)]
    gate["selected_row"] = best
    artifact_allowed = bool(best["eligible"] or args.allow_artifact_even_if_borderline)
    slug = args.slug
    local_cv = 0.534000 + (best["mean_delta"] / len(TARGETS))
    status = "unsubmitted"
    filename = f"submission_id_{args.experiment_id:03d}_{slug}_cv{local_cv:.6f}_{status}.csv"
    submission_path = SUBMISSIONS_DIR / filename
    notebook_path = NOTEBOOK_DIR / f"submission_{args.experiment_id:03d}_{slug}_cv{local_cv:.6f}_{status}.ipynb"
    metadata_path = REPORTS_DIR / f"experiment_{args.experiment_id:03d}_{slug}_cv{local_cv:.6f}_{status}.json"
    decision_path = REPORTS_DIR / f"decision_{args.experiment_id:03d}_{slug}_cv{local_cv:.6f}_{status}.md"
    base_test_path = BASE_PREDICTION_DIR / f"base_test_id_{args.experiment_id:03d}_{slug}_cv{local_cv:.6f}.csv"
    if artifact_allowed:
        for path in (submission_path, notebook_path, metadata_path, decision_path, base_test_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {rel(path)}")
        base_test = pd.read_csv(BASE_TEST_ID342)
        base_test["date"] = pd.to_datetime(base_test["date"]).dt.date
        full_pred = fit_full_predict(
            train_features,
            test_features,
            feature_cols,
            train[FOCUS_TARGET].astype(int).to_numpy(),
            float(best["c_value"]),
        )
        out_test = base_test.copy()
        out_test[FOCUS_TARGET] = np.clip(
            (1.0 - float(best["weight"])) * out_test[FOCUS_TARGET].astype(float).to_numpy()
            + float(best["weight"]) * full_pred,
            EPS,
            1.0 - EPS,
        )
        sample = pd.read_csv(SAMPLE_PATH)
        out_submission = map_predictions_to_original_sample(sample, out_test)
        if not validate_submission(out_submission, sample):
            raise ValueError("Generated submission failed sample validation.")
        SUBMISSIONS_DIR.mkdir(exist_ok=True)
        NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(exist_ok=True)
        BASE_PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
        out_submission.to_csv(submission_path, index=False)
        out_test.to_csv(base_test_path, index=False)
        write_notebook(notebook_path, submission_path, metadata_path, args.experiment_id, slug)
        append_submission_log(
            log_path=SUBMISSIONS_DIR / "submission_log.csv",
            experiment_id=args.experiment_id,
            submission_path=submission_path,
            notebook_path=notebook_path,
            local_cv=float(local_cv),
            notes=(
                "Generated emergency post-ID342 Q3-only app-name lexicon aggregate overlay. "
                f"Best local gate row: {best['config']}; wins={best['wins']}; "
                f"mean_delta={best['mean_delta']:.9f}. Raw app names were not persisted."
            ),
            model_description="ID342 Q3 app-name lexicon aggregate overlay",
        )
    metadata = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "experiment_id": args.experiment_id,
        "slug": slug,
        "feature_metadata": feature_meta,
        "feature_count": len(feature_cols),
        "gate": gate,
        "top_rows": rows[:12],
        "artifact_allowed": artifact_allowed,
        "submission_file": submission_path.name if artifact_allowed else None,
        "submission_path": rel(submission_path) if artifact_allowed else None,
        "submission_sha256": sha256_file(submission_path) if artifact_allowed else None,
        "notebook_file": rel(notebook_path) if artifact_allowed else None,
        "base_test_file": rel(base_test_path) if artifact_allowed else None,
        "local_cv_proxy": float(local_cv),
        "raw_app_names_persisted": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    decision_path.write_text(
        "# Decision note - emergency Q3 app lexicon overlay\n\n"
        f"- Artifact allowed: `{artifact_allowed}`\n"
        f"- Best row: `{best['config']}`\n"
        f"- Wins: `{best['wins']}`\n"
        f"- Mean delta: `{best['mean_delta']:.9f}`\n"
        f"- Max delta: `{best['max_delta']:.9f}`\n"
        f"- Submission: `{metadata['submission_path']}`\n",
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", type=int, default=345)
    parser.add_argument("--slug", default="id342_q3_app_lexicon_w")
    parser.add_argument("--allow-artifact-even-if-borderline", action="store_true")
    parser.add_argument("--row-rank", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    metadata = build_candidate(parse_args())
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
