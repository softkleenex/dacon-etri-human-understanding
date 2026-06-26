#!/usr/bin/env python3
"""Post-ID342 solar/seasonal/holiday prior gate.

This report-only diagnostic tests a materially external, date-derived source:
Korea-centered photoperiod/seasonal phase and Korean holiday proximity. It asks
whether these priors open a stable Q3/S3/S4 overlay against ID342 without
reusing the existing OOF shelf, processed sleep columns, or raw sensor streams.

It creates no prediction CSV, feature matrix artifact, base prediction
artifact, submission CSV, submission-log row, Kaggle action, or DACON
submission.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_post_id341_q2q3s3_fold_internal_model_smoke import (  # noqa: E402,I001
    PRIMARY_GATE_SPLITS,
    TARGETS,
    load_reference_values,
    load_split_manifest,
    load_train_sorted,
)

KST = timezone(timedelta(hours=9))
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id342_solar_seasonal_holiday_prior_gate_20260625.json"

ANCHOR_ID = 342
ID342_PUBLIC = 0.5960832519
PUBLIC_055 = 0.55
EPS = 1e-6
FOCUS_TARGETS = ["Q3", "S3", "S4"]
PRIMARY_SPLITS = PRIMARY_GATE_SPLITS
BLEND_WEIGHTS = [0.0025, 0.005, 0.01, 0.02, 0.05]
TARGET_WIN_FLOOR = 4
TARGET_MEAN_DELTA_CEILING = -0.00005
TARGET_MAX_DELTA_CEILING = 0.00075

SEOUL_LAT_DEG = 37.5665
SEOUL_LON_DEG = 126.9780
KOREA_STANDARD_MERIDIAN_DEG = 135.0

KOREA_PUBLIC_HOLIDAYS_2024 = {
    "2024-08-15": "liberation_day",
    "2024-09-16": "chuseok_day_minus1",
    "2024-09-17": "chuseok",
    "2024-09-18": "chuseok_day_plus1",
    "2024-10-03": "national_foundation_day",
    "2024-10-09": "hangeul_day",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    c_value: float | None = None
    max_leaf_nodes: int | None = None


MODEL_SPECS = (
    ModelSpec("lr_c0_02", "lr", c_value=0.02),
    ModelSpec("lr_c0_08", "lr", c_value=0.08),
    ModelSpec("hgb_l5", "hgb", max_leaf_nodes=5),
)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def fold_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(log_loss(y_true.astype(int), np.clip(y_pred, EPS, 1.0 - EPS), labels=[0, 1]))


def weighted_loss(y: np.ndarray, pred: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> float:
    weighted = 0.0
    total = 0
    for _fit_idx, valid_idx in folds:
        weighted += fold_loss(y[valid_idx], pred[valid_idx]) * int(len(valid_idx))
        total += int(len(valid_idx))
    if total == 0:
        raise ValueError("Validation folds have no rows.")
    return float(weighted / total)


def make_model(spec: ModelSpec):
    if spec.kind == "lr":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=float(spec.c_value), solver="liblinear", max_iter=500, random_state=71),
        )
    if spec.kind == "hgb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_leaf_nodes=int(spec.max_leaf_nodes),
                learning_rate=0.03,
                max_iter=80,
                l2_regularization=0.20,
                random_state=71,
            ),
        )
    raise ValueError(f"Unknown model kind: {spec.kind}")


def load_test_sorted() -> pd.DataFrame:
    test = pd.read_csv(RAW_DIR / "ch2026_submission_sample.csv")
    test["date"] = pd.to_datetime(test["lifelog_date"]).dt.date
    return test.sort_values(["subject_id", "date"]).reset_index(drop=True)


def solar_features_for_dates(dates: pd.Series) -> pd.DataFrame:
    date_ts = pd.to_datetime(dates)
    day_of_year = date_ts.dt.dayofyear.astype(float)
    gamma = 2.0 * np.pi * (day_of_year - 1.0) / 365.0
    declination = np.deg2rad(23.44) * np.sin(2.0 * np.pi * (day_of_year + 284.0) / 365.0)
    lat_rad = math.radians(SEOUL_LAT_DEG)
    hour_angle = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(declination), -1.0, 1.0))
    daylight_hours = 24.0 * hour_angle / np.pi
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    solar_noon_minutes = 12.0 * 60.0 - 4.0 * (SEOUL_LON_DEG - KOREA_STANDARD_MERIDIAN_DEG) - equation_of_time
    sunrise_minutes = solar_noon_minutes - daylight_hours * 30.0
    sunset_minutes = solar_noon_minutes + daylight_hours * 30.0
    seasonal_phase = 2.0 * np.pi * day_of_year / 365.0
    return pd.DataFrame(
        {
            "solar_daylight_hours": daylight_hours,
            "solar_night_hours": 24.0 - daylight_hours,
            "solar_declination_sin": np.sin(declination),
            "solar_declination_cos": np.cos(declination),
            "solar_daylight_change_1d": pd.Series(daylight_hours).diff().fillna(0.0).to_numpy(),
            "solar_sunrise_minutes": sunrise_minutes,
            "solar_sunset_minutes": sunset_minutes,
            "solar_noon_minutes": solar_noon_minutes,
            "season_sin": np.sin(seasonal_phase),
            "season_cos": np.cos(seasonal_phase),
            "late_summer_to_autumn_index": np.clip((day_of_year - 213.0) / 100.0, 0.0, 1.0),
        }
    )


def holiday_features_for_dates(dates: pd.Series) -> pd.DataFrame:
    date_ts = pd.to_datetime(dates)
    holiday_dates = pd.to_datetime(sorted(KOREA_PUBLIC_HOLIDAYS_2024))
    out = pd.DataFrame(index=date_ts.index)
    out["weekday"] = date_ts.dt.dayofweek.astype(float)
    out["is_weekend"] = date_ts.dt.dayofweek.isin([5, 6]).astype(float)
    out["weekday_sin"] = np.sin(2.0 * np.pi * out["weekday"] / 7.0)
    out["weekday_cos"] = np.cos(2.0 * np.pi * out["weekday"] / 7.0)
    out["is_korean_public_holiday"] = date_ts.isin(holiday_dates).astype(float)
    if len(holiday_dates):
        day_delta = np.asarray(
            [
                np.min(np.abs((holiday_dates - date_value).days))
                for date_value in date_ts
            ],
            dtype=float,
        )
        signed_delta = np.asarray(
            [
                (holiday_dates[np.argmin(np.abs((holiday_dates - date_value).days))] - date_value).days
                for date_value in date_ts
            ],
            dtype=float,
        )
    else:
        day_delta = np.zeros(len(date_ts), dtype=float)
        signed_delta = np.zeros(len(date_ts), dtype=float)
    out["days_to_nearest_korean_holiday_abs"] = np.minimum(day_delta, 14.0)
    out["days_to_nearest_korean_holiday_signed"] = np.clip(signed_delta, -14.0, 14.0)
    out["near_korean_holiday_3d"] = (day_delta <= 3).astype(float)
    out["near_korean_holiday_7d"] = (day_delta <= 7).astype(float)
    out["post_korean_holiday_3d"] = ((signed_delta < 0) & (np.abs(signed_delta) <= 3)).astype(float)
    out["pre_korean_holiday_3d"] = ((signed_delta > 0) & (signed_delta <= 3)).astype(float)
    return out


def build_feature_frame(keys: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], dict[str, list[str]]]:
    out = keys[["subject_id", "date"]].copy().reset_index(drop=True)
    date_ts = pd.to_datetime(out["date"])
    solar = solar_features_for_dates(date_ts)
    holiday = holiday_features_for_dates(date_ts)
    out = pd.concat([out, solar, holiday], axis=1)
    out["solar_daylight_x_weekend"] = out["solar_daylight_hours"] * out["is_weekend"]
    out["solar_daylight_x_holiday_near7"] = out["solar_daylight_hours"] * out["near_korean_holiday_7d"]
    out["solar_night_x_holiday_near7"] = out["solar_night_hours"] * out["near_korean_holiday_7d"]
    out["autumn_x_weekend"] = out["late_summer_to_autumn_index"] * out["is_weekend"]
    out["autumn_x_holiday_near7"] = out["late_summer_to_autumn_index"] * out["near_korean_holiday_7d"]
    out = out.sort_values(["subject_id", "date"]).reset_index(drop=True)
    out["subject_solar_daylight_delta"] = out.groupby("subject_id")["solar_daylight_hours"].diff().fillna(0.0)
    out["subject_holiday_gap_delta"] = (
        out.groupby("subject_id")["days_to_nearest_korean_holiday_signed"].diff().fillna(0.0)
    )
    families = {
        "solar_photoperiod": [
            col for col in out.columns if col.startswith("solar_") or col in {"season_sin", "season_cos"}
        ],
        "korean_holiday_calendar": [
            col
            for col in out.columns
            if col
            in {
                "weekday",
                "weekday_sin",
                "weekday_cos",
                "is_weekend",
                "is_korean_public_holiday",
                "days_to_nearest_korean_holiday_abs",
                "days_to_nearest_korean_holiday_signed",
                "near_korean_holiday_3d",
                "near_korean_holiday_7d",
                "post_korean_holiday_3d",
                "pre_korean_holiday_3d",
            }
        ],
        "solar_holiday_interaction": [
            "solar_daylight_x_weekend",
            "solar_daylight_x_holiday_near7",
            "solar_night_x_holiday_near7",
            "autumn_x_weekend",
            "autumn_x_holiday_near7",
            "subject_solar_daylight_delta",
            "subject_holiday_gap_delta",
        ],
    }
    families["solar_holiday_all"] = sorted({col for cols in families.values() for col in cols})
    metadata = {
        "feature_source": (
            "Date-derived Seoul/Korea photoperiod approximation plus fixed 2024 Korean public-holiday proximity. "
            "No raw sensor payloads, processed sleep columns, labels, public scores, or candidate predictions are "
            "used to create features."
        ),
        "seoul_latitude": SEOUL_LAT_DEG,
        "seoul_longitude": SEOUL_LON_DEG,
        "holiday_dates": KOREA_PUBLIC_HOLIDAYS_2024,
        "rows": int(len(out)),
        "families": {
            name: {
                "feature_count": int(len(cols)),
                "feature_sample": cols[:12],
                "mean_missing_rate": float(out[cols].isna().mean().mean()) if cols else None,
            }
            for name, cols in families.items()
        },
    }
    return out, metadata, families


def predict_split(
    features: pd.DataFrame,
    feature_cols: list[str],
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    spec: ModelSpec,
) -> tuple[np.ndarray, int]:
    pred = np.full(len(y), np.nan, dtype=float)
    covered = 0
    for fit_idx, valid_idx in folds:
        if len(np.unique(y[fit_idx])) < 2:
            model = DummyClassifier(strategy="prior")
            model.fit(np.zeros((len(fit_idx), 1)), y[fit_idx])
            proba = np.full(len(valid_idx), float(np.mean(y[fit_idx])), dtype=float)
        else:
            model = make_model(spec)
            model.fit(features.iloc[fit_idx][feature_cols], y[fit_idx])
            proba = model.predict_proba(features.iloc[valid_idx][feature_cols])[:, 1]
        pred[valid_idx] = np.clip(proba, EPS, 1.0 - EPS)
        covered += int(len(valid_idx))
    return pred, covered


def scan_family(
    train: pd.DataFrame,
    features: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    anchor_values: np.ndarray,
    family_name: str,
    feature_cols: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in FOCUS_TARGETS:
        target_idx = TARGETS.index(target)
        y = train[target].astype(int).to_numpy()
        for spec in MODEL_SPECS:
            split_predictions: dict[str, np.ndarray] = {}
            split_coverage: dict[str, int] = {}
            for split_name in PRIMARY_SPLITS:
                pred, covered = predict_split(features, feature_cols, y, splits[split_name], spec)
                split_predictions[split_name] = pred
                split_coverage[split_name] = covered
            for weight in BLEND_WEIGHTS:
                split_deltas: dict[str, float] = {}
                split_losses: dict[str, float] = {}
                split_anchor_losses: dict[str, float] = {}
                wins = 0
                for split_name in PRIMARY_SPLITS:
                    folds = splits[split_name]
                    blended = np.clip(
                        (1.0 - weight) * anchor_values[:, target_idx] + weight * split_predictions[split_name],
                        EPS,
                        1.0 - EPS,
                    )
                    anchor_loss = weighted_loss(y, anchor_values[:, target_idx], folds)
                    candidate_loss = weighted_loss(y, blended, folds)
                    delta = candidate_loss - anchor_loss
                    split_anchor_losses[split_name] = anchor_loss
                    split_losses[split_name] = candidate_loss
                    split_deltas[split_name] = delta
                    wins += int(delta < 0)
                mean_delta = float(np.mean(list(split_deltas.values())))
                max_delta = float(np.max(list(split_deltas.values())))
                rows.append(
                    {
                        "family_name": family_name,
                        "target": target,
                        "config": f"{family_name}_{target}_{spec.name}_w{weight:g}",
                        "feature_count": int(len(feature_cols)),
                        "model_spec": asdict(spec),
                        "weight": float(weight),
                        "valid_rows_by_split": split_coverage,
                        "wins": int(wins),
                        "mean_delta": mean_delta,
                        "max_delta": max_delta,
                        "min_delta": float(np.min(list(split_deltas.values()))),
                        "split_deltas": split_deltas,
                        "split_losses": split_losses,
                        "split_anchor_losses": split_anchor_losses,
                        "eligible": bool(
                            wins >= TARGET_WIN_FLOOR
                            and mean_delta <= TARGET_MEAN_DELTA_CEILING
                            and max_delta <= TARGET_MAX_DELTA_CEILING
                        ),
                    }
                )
    return rows


def build_report(_args: argparse.Namespace) -> dict[str, Any]:
    train = load_train_sorted()
    test = load_test_sorted()
    splits = load_split_manifest()
    anchor_values = load_reference_values(train, ANCHOR_ID)
    train_features, train_metadata, families = build_feature_frame(train[["subject_id", "date"]])
    test_features, test_metadata, _test_families = build_feature_frame(test[["subject_id", "date"]])
    rows: list[dict[str, Any]] = []
    for family_name, feature_cols in families.items():
        rows.extend(scan_family(train, train_features, splits, anchor_values, family_name, feature_cols))
    rows = sorted(rows, key=lambda row: (not row["eligible"], row["mean_delta"], row["max_delta"], -row["wins"]))
    eligible_rows = [row for row in rows if row["eligible"]]
    selected_targets = sorted({row["target"] for row in eligible_rows})
    estimated_all_target_delta = (
        float(sum(row["mean_delta"] for row in eligible_rows[: len(FOCUS_TARGETS)]) / len(TARGETS))
        if eligible_rows
        else 0.0
    )
    projected_public_proxy = float(ID342_PUBLIC + estimated_all_target_delta)
    decision = {
        "eligible_row_count": int(len(eligible_rows)),
        "selected_targets": selected_targets,
        "q3_selected": "Q3" in selected_targets,
        "s3_selected": "S3" in selected_targets,
        "s4_selected": "S4" in selected_targets,
        "best_row": rows[0] if rows else None,
        "estimated_all_target_delta_vs_id342": estimated_all_target_delta,
        "projected_public_proxy_from_id342": projected_public_proxy,
        "gap_to_public_055_after_proxy": projected_public_proxy - PUBLIC_055,
        "solar_seasonal_holiday_prior_gate_pass": bool(eligible_rows),
        "public_055_evidence": False,
        "create_submission_artifact": False,
        "modify_submission_log": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "submit_now": False,
        "next_action": (
            "Treat solar/seasonal/holiday priors as report-only domain evidence. Do not create artifacts, Kaggle "
            "runs, or DACON submissions unless a future independent gate produces a Public-0.55 plausible row."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id342_solar_seasonal_holiday_prior_gate",
        "purpose": "Gate external date-derived solar/seasonal/Korean-holiday priors against ID342.",
        "anchor": {"id": ANCHOR_ID, "public_score": ID342_PUBLIC},
        "validation": {
            "primary_splits": PRIMARY_SPLITS,
            "model_specs": [asdict(spec) for spec in MODEL_SPECS],
            "blend_weights": BLEND_WEIGHTS,
            "target_win_floor": TARGET_WIN_FLOOR,
            "target_mean_delta_ceiling": TARGET_MEAN_DELTA_CEILING,
            "target_max_delta_ceiling": TARGET_MAX_DELTA_CEILING,
            "focus_targets": FOCUS_TARGETS,
        },
        "train_feature_metadata": train_metadata,
        "test_feature_metadata": test_metadata,
        "top_rows": rows[:36],
        "eligible_rows": eligible_rows[:36],
        "decision": decision,
        "no_predictions_created": True,
        "no_feature_matrix_created": True,
        "no_base_prediction_artifact_created": True,
        "no_submission_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_action": True,
        "no_dacon_submission": True,
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    md_path = output.with_suffix(".md")
    decision = report["decision"]
    lines = [
        "# Post-ID342 solar/seasonal/holiday prior gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.",
        "",
        "## Feature Families",
        "",
        "| family | features | mean missing rate | sample |",
        "|---|---:|---:|---|",
    ]
    for family_name, metadata in report["train_feature_metadata"]["families"].items():
        sample = ", ".join(f"`{col}`" for col in metadata["feature_sample"][:6])
        lines.append(
            f"| `{family_name}` | {metadata['feature_count']} | {metadata['mean_missing_rate']:.6f} | {sample} |"
        )
    lines.extend(
        [
            "",
            "## Top Rows",
            "",
            "| family | target | config | wins | mean delta | max delta | eligible |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in report["top_rows"][:16]:
        lines.append(
            f"| `{row['family_name']}` | `{row['target']}` | `{row['config']}` | {row['wins']} | "
            f"{row['mean_delta']:.9f} | {row['max_delta']:.9f} | `{row['eligible']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Eligible rows: `{decision['eligible_row_count']}`",
            f"- Selected targets: `{decision['selected_targets']}`",
            f"- Projected Public proxy: `{decision['projected_public_proxy_from_id342']:.9f}`",
            f"- Gap to Public 0.55: `{decision['gap_to_public_055_after_proxy']:.9f}`",
            f"- Gate pass: `{decision['solar_seasonal_holiday_prior_gate_pass']}`",
            f"- Kaggle now: `{decision['run_kaggle_now']}`",
            f"- DACON submit now: `{decision['submit_now']}`",
            f"- Next action: {decision['next_action']}",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, output)
    decision = report["decision"]
    print(f"Wrote {rel(output)}")
    print(f"Wrote {rel(output.with_suffix('.md'))}")
    print(f"Eligible rows: {decision['eligible_row_count']} | selected={decision['selected_targets']}")


if __name__ == "__main__":
    main()
