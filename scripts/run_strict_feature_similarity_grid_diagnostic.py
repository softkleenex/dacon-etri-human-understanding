#!/usr/bin/env python3
"""Grid diagnostic for fixed same-subject feature-similarity overlays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_public_temporal_anchor_probe_experiment import add_date_dt, load_base_artifact  # noqa: E402
from scripts.run_strict_feature_similarity_anchor_overlay_experiment import (  # noqa: E402
    blocked_feature_similarity_candidate_oof,
)
from scripts.run_strict_fixed_feature_similarity_overlay_experiment import feature_family_tokens  # noqa: E402
from scripts.run_strict_raw_feature_experiment import (  # noqa: E402
    REPORTS_DIR,
    TARGETS,
    build_feature_frames,
    numeric_feature_columns,
)
from src.utils.validation import make_subject_hole_folds  # noqa: E402

EPS = 1e-6
DEFAULT_TARGETS = ["Q2", "Q3", "S2", "S3", "S4"]
DEFAULT_BASE_ID = 146


def parse_csv_arg(value: str, *, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def build_feature_families(train: pd.DataFrame, test: pd.DataFrame, env_family: str) -> dict[str, list[str]]:
    numeric_cols = numeric_feature_columns(train, test)
    families = {"all": numeric_cols}
    for family, tokens in feature_family_tokens(env_family).items():
        cols = [col for col in numeric_cols if any(token in col.lower() for token in tokens)]
        if not cols:
            raise ValueError(f"No columns found for family {family!r}.")
        families[family] = cols
    return families


def scale_families(
    train: pd.DataFrame,
    test: pd.DataFrame,
    families: dict[str, list[str]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    scaled: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for family, cols in families.items():
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        combined = pd.concat([train[cols], test[cols]], ignore_index=True)
        values = scaler.fit_transform(imputer.fit_transform(combined))
        scaled[family] = (values[: len(train)], values[len(train) :])
    return scaled


def candidate_name(config: dict[str, float | int | str]) -> str:
    return (
        f"{config['family']}_k{int(config['k'])}"
        f"_dw{float(config['date_weight']):g}_w{float(config['blend_weight']):g}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--families", default="all,date_time,screen_app,move_gps,light_hr,env")
    parser.add_argument("--k-values", default="1,2,3,5,7,10,14")
    parser.add_argument("--date-weights", default="0,0.03,0.05,0.1")
    parser.add_argument("--blend-weights", default="0.05,0.1,0.15,0.2,0.3,0.4")
    parser.add_argument("--env-family", default="env24")
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    targets = parse_csv_arg(args.targets)
    unknown_targets = sorted(set(targets) - set(TARGETS))
    if unknown_targets:
        raise ValueError(f"Unknown targets: {unknown_targets}")
    family_names = parse_csv_arg(args.families)
    k_values = parse_csv_arg(args.k_values, cast=int)
    date_weights = parse_csv_arg(args.date_weights, cast=float)
    blend_weights = parse_csv_arg(args.blend_weights, cast=float)

    train_raw, test_raw, _metrics_sorted, _sample_original = build_feature_frames()
    train = add_date_dt(train_raw)
    test = add_date_dt(test_raw)
    families = build_feature_families(train_raw, test_raw, args.env_family)
    unknown_families = sorted(set(family_names) - set(families))
    if unknown_families:
        raise ValueError(f"Unknown families: {unknown_families}")
    selected_families = {family: families[family] for family in family_names}
    scaled = scale_families(train_raw, test_raw, selected_families)
    folds = [(np.asarray(train_idx), np.asarray(valid_idx)) for train_idx, valid_idx in make_subject_hole_folds(train)]
    base_oof, _base_test, base_source = load_base_artifact(train_raw, test_raw, args.base_id)

    report: dict[str, object] = {
        "diagnostic": "strict_feature_similarity_grid",
        "base_id": args.base_id,
        "base_source": base_source,
        "targets": {},
        "families": {family: len(cols) for family, cols in selected_families.items()},
        "grid": {
            "k_values": k_values,
            "date_weights": date_weights,
            "blend_weights": blend_weights,
            "env_family": args.env_family,
        },
        "caution": (
            "This diagnostic ranks fixed full-OOF candidates. Validation-hole anchor computation respects "
            "Subject-Hole labels, but candidate selection is not nested."
        ),
    }

    for target in targets:
        y = train_raw[target].astype(int).to_numpy()
        base_values = np.clip(base_oof[target].astype(float).to_numpy(), EPS, 1.0 - EPS)
        base_loss = float(log_loss(y, base_values, labels=[0, 1]))
        candidates: list[dict] = []
        for family in family_names:
            for k in k_values:
                for date_weight in date_weights:
                    for blend_weight in blend_weights:
                        config = {
                            "family": family,
                            "k": k,
                            "date_weight": date_weight,
                            "blend_weight": blend_weight,
                        }
                        oof, fold_details = blocked_feature_similarity_candidate_oof(
                            train=train,
                            test=test,
                            scaled=scaled,
                            y=y,
                            base_values=base_values,
                            folds=folds,
                            config=config,
                        )
                        loss = float(log_loss(y, oof, labels=[0, 1]))
                        candidates.append(
                            {
                                "name": candidate_name(config),
                                "config": config,
                                "loss": loss,
                                "delta_vs_base": loss - base_loss,
                                "fold_losses": [detail["valid_loss"] for detail in fold_details],
                            }
                        )
        candidates = sorted(candidates, key=lambda item: item["loss"])
        report["targets"][target] = {
            "base_loss": base_loss,
            "best": candidates[0],
            "top": candidates[: args.top_n],
            "n_candidates": len(candidates),
        }
        print(
            f"{target}: base={base_loss:.6f} best={candidates[0]['loss']:.6f} "
            f"{candidates[0]['name']} delta={candidates[0]['delta_vs_base']:.6f}"
        )

    output_path = Path(args.output) if args.output else REPORTS_DIR / f"strict_feature_similarity_grid_id{args.base_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        display_path = output_path.resolve().relative_to(ROOT)
    except ValueError:
        display_path = output_path
    print(f"Wrote {display_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
