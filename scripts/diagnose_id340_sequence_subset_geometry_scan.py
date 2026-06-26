#!/usr/bin/env python3
"""Target-subset geometry scan for the ID340 sequence-summary lead.

The all-target exact design passed primary split probes but failed the full
geometry gate because its proxy was worse than ID337. This report scans smaller
target subsets and weights using the same sequence-summary source family.

It is report-only: no CSV, base prediction artifact, notebook/history artifact,
submission-log row, or DACON submission is created.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_id179_vs_id335_sequence_probe_review import geometry  # noqa: E402
from scripts.diagnose_id340_exact_sequence_full_geometry_gate import (  # noqa: E402
    MODEL_SPEC,
    PUBLIC_RERANK_PATH,
    conservative_public_proxy,
    frame_like,
    load_artifact_pair,
    max_delta_by_target,
    mean_delta_by_target,
    risk_tier,
    train_full_sequence_source,
)
from scripts.diagnose_id340_sequence_summary_split_probe import load_split_manifest  # noqa: E402
from scripts.diagnose_public_random_conv_sequence import (  # noqa: E402
    RAW_DIR,
    align_table,
    build_daily_sequence_table,
    make_model,
    table_to_tensor,
)
from scripts.diagnose_public_sequence_summary_embedding import sequence_summary_features  # noqa: E402
from scripts.run_strict_raw_feature_experiment import TARGETS, build_feature_frames  # noqa: E402

KST = timezone(timedelta(hours=9))
SPLIT_MANIFEST_PATH = ROOT / "reports" / "id340_validation_split_manifest_20260618.json"
PRIMARY_GATE_SPLITS = [
    "subject_hole_5",
    "actual_test_band_holdout",
    "test_boundary_neighbor2_holdout",
    "test_boundary_neighbor5_holdout",
    "loso_10",
]
WEIGHTS = [0.01, 0.02, 0.03, 0.05, 0.07]
TARGET_SETS: dict[str, list[str]] = {
    "q123": ["Q1", "Q2", "Q3"],
    "q123_s1": ["Q1", "Q2", "Q3", "S1"],
    "q123_s1_s4": ["Q1", "Q2", "Q3", "S1", "S4"],
    "all": TARGETS,
    "q2_only": ["Q2"],
    "q23": ["Q2", "Q3"],
    "s1_only": ["S1"],
}


def rerank_proxy_by_id(candidate_id: int) -> float | None:
    if not PUBLIC_RERANK_PATH.exists():
        return None
    report = json.loads(PUBLIC_RERANK_PATH.read_text(encoding="utf-8"))
    for row in report.get("top_all_by_proxy", []):
        if int(row["id"]) == candidate_id:
            return float(row["conservative_public_risk_proxy"])
    return None


def split_source_predictions(
    x_train: np.ndarray,
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.full((len(train), len(TARGETS)), np.nan, dtype=float)
    covered = np.zeros((len(train), len(TARGETS)), dtype=bool)
    for target_idx, target in enumerate(TARGETS):
        y = train[target].astype(int).to_numpy()
        for train_idx, valid_idx in folds:
            if len(np.unique(y[train_idx])) < 2:
                model = DummyClassifier(strategy="prior")
            else:
                model = clone(make_model(MODEL_SPEC))
            model.fit(x_train[train_idx], y[train_idx])
            values[valid_idx, target_idx] = model.predict_proba(x_train[valid_idx])[:, 1]
            covered[valid_idx, target_idx] = True
    return np.clip(values, 1e-6, 1 - 1e-6), covered


def evaluate_split_variant(
    *,
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    id178_values: np.ndarray,
    source_values: np.ndarray,
    covered: np.ndarray,
    targets: list[str],
    weight: float,
) -> dict[str, Any]:
    target_set = set(targets)
    candidate = id178_values.copy()
    for target_idx, target in enumerate(TARGETS):
        if target in target_set:
            candidate[:, target_idx] = np.where(
                covered[:, target_idx],
                np.clip(
                    (1.0 - weight) * id178_values[:, target_idx] + weight * source_values[:, target_idx], 1e-6, 1 - 1e-6
                ),
                id178_values[:, target_idx],
            )
    target_losses: dict[str, float] = {}
    for target_idx, target in enumerate(TARGETS):
        y = train[target].astype(int).to_numpy()
        total = 0
        weighted = 0.0
        for _train_idx, valid_idx in folds:
            pred = candidate[valid_idx, target_idx]
            loss = float(log_loss(y[valid_idx], pred, labels=[0, 1]))
            weighted += loss * len(valid_idx)
            total += len(valid_idx)
        target_losses[target] = float(weighted / total)
    return {
        "mean_loss": float(np.mean(list(target_losses.values()))),
        "target_losses": target_losses,
    }


def build_variant_values(
    base_values: np.ndarray, source_values: np.ndarray, targets: list[str], weight: float
) -> np.ndarray:
    target_set = set(targets)
    out = base_values.copy()
    for target_idx, target in enumerate(TARGETS):
        if target in target_set:
            out[:, target_idx] = np.clip(
                (1.0 - weight) * base_values[:, target_idx] + weight * source_values[:, target_idx],
                1e-6,
                1 - 1e-6,
            )
    return out


def build_report() -> dict[str, Any]:
    train, test, _metrics_sorted, _sample = build_feature_frames()
    splits = load_split_manifest(SPLIT_MANIFEST_PATH)
    references = {}
    for candidate_id in [178, 287, 329, 335, 337, 339]:
        references[candidate_id] = load_artifact_pair(train, test, candidate_id)
    base_oof = references[178][0]
    base_test = references[178][1]
    id178_oof_values = base_oof[TARGETS].astype(float).to_numpy()
    id178_test_values = base_test[TARGETS].astype(float).to_numpy()
    anchors = {candidate_id: references[candidate_id][1] for candidate_id in [178, 287, 329, 335]}

    seq_table = build_daily_sequence_table(RAW_DIR)
    train_seq, test_seq, feature_cols = align_table(train, test, seq_table)
    train_tensor, test_tensor, channel_names = table_to_tensor(train_seq, test_seq, feature_cols)
    x_train, feature_names = sequence_summary_features(train_tensor, channel_names)
    x_test, _ = sequence_summary_features(test_tensor, channel_names)
    _source_train_full, source_test_full, source_details = train_full_sequence_source(x_train, x_test, train)

    split_sources = {
        split_name: {
            "folds": folds,
            "source": split_source_predictions(x_train, train, folds),
        }
        for split_name, folds in splits.items()
    }
    id337_proxy = rerank_proxy_by_id(337)
    id339_proxy = rerank_proxy_by_id(339)

    variants: list[dict[str, Any]] = []
    for target_set_name, targets in TARGET_SETS.items():
        for weight in WEIGHTS:
            split_rows = []
            primary_pass = True
            subject_hole_loss = None
            for split_name, payload in split_sources.items():
                source_values, covered = payload["source"]
                split_eval = evaluate_split_variant(
                    train=train,
                    folds=payload["folds"],
                    id178_values=id178_oof_values,
                    source_values=source_values,
                    covered=covered,
                    targets=targets,
                    weight=weight,
                )
                id178_eval = evaluate_split_variant(
                    train=train,
                    folds=payload["folds"],
                    id178_values=id178_oof_values,
                    source_values=id178_oof_values,
                    covered=np.ones_like(covered, dtype=bool),
                    targets=[],
                    weight=0.0,
                )
                delta = float(split_eval["mean_loss"] - id178_eval["mean_loss"])
                if split_name == "subject_hole_5":
                    subject_hole_loss = float(split_eval["mean_loss"])
                if split_name in PRIMARY_GATE_SPLITS and delta >= 0:
                    primary_pass = False
                split_rows.append(
                    {
                        "split": split_name,
                        "mean_loss": split_eval["mean_loss"],
                        "id178_loss": id178_eval["mean_loss"],
                        "delta_vs_id178": delta,
                        "is_primary": split_name in PRIMARY_GATE_SPLITS,
                    }
                )
            candidate_test_values = build_variant_values(id178_test_values, source_test_full, targets, weight)
            candidate_test = frame_like(test, candidate_test_values)
            geom = geometry(candidate_test, anchors)
            proxy = conservative_public_proxy(geom, subject_hole_loss)
            full_geometry_pass = (
                primary_pass
                and geom["max_delta_vs_178"] <= 0.05
                and geom["projection_failed"] <= 0.002
                and (id337_proxy is None or proxy < id337_proxy)
                and (id339_proxy is None or proxy < id339_proxy)
            )
            variants.append(
                {
                    "name": f"{target_set_name}_w{weight:g}",
                    "target_set": target_set_name,
                    "targets": targets,
                    "weight": weight,
                    "split_rows": split_rows,
                    "primary_gate_pass": primary_pass,
                    "subject_hole_loss": subject_hole_loss,
                    "geometry": geom,
                    "risk_tier": risk_tier(geom),
                    "conservative_public_risk_proxy": proxy,
                    "proxy_delta_vs_id337": None if id337_proxy is None else float(proxy - id337_proxy),
                    "proxy_delta_vs_id339": None if id339_proxy is None else float(proxy - id339_proxy),
                    "full_geometry_pass": bool(full_geometry_pass),
                    "public_058_plausible": bool(proxy <= 0.590),
                    "test_delta_vs_id178_by_target": {
                        "max": max_delta_by_target(candidate_test, base_test),
                        "mean": mean_delta_by_target(candidate_test, base_test),
                    },
                }
            )

    variants = sorted(
        variants,
        key=lambda row: (
            not row["full_geometry_pass"],
            not row["primary_gate_pass"],
            row["conservative_public_risk_proxy"],
            row["geometry"]["max_delta_vs_178"],
            row["name"],
        ),
    )
    best = variants[0]
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "id340_sequence_subset_geometry_scan",
        "purpose": "Find a safer target subset/weight for the ID340 sequence-summary source after the all-target exact design failed proxy-vs-ID337.",
        "source_model": asdict(MODEL_SPEC),
        "weights": WEIGHTS,
        "target_sets": TARGET_SETS,
        "sequence_feature_summary": {
            "n_sequence_rows": int(len(seq_table)),
            "n_total_channels": int(len(channel_names)),
            "n_summary_features": int(x_train.shape[1]),
            "feature_family_count": int(len(feature_names) // max(len(channel_names), 1)),
        },
        "source_prediction_details": source_details,
        "id337_proxy": id337_proxy,
        "id339_proxy": id339_proxy,
        "variants": variants,
        "best_variant": best,
        "decision": {
            "promote_submission_candidate_now": False,
            "create_id340_artifact_now": False,
            "submit_now": False,
            "has_variant_for_artifact_review": bool(best["full_geometry_pass"]),
            "best_variant": best["name"],
            "read": (
                "A variant can move to artifact-review only if it passes all primary split lenses and improves "
                "the conservative proxy versus ID337/ID339. This scan still creates no artifact."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ID340 sequence subset geometry scan",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It scans smaller target subsets/weights for the same sequence-summary source and creates no CSV, base artifact, notebook/history artifact, submission-log row, or DACON submission.",
        "",
        f"- ID337 proxy: `{report['id337_proxy']:.6f}`",
        f"- ID339 proxy: `{report['id339_proxy']:.6f}`",
        "",
        "## Top variants",
        "",
        "| rank | variant | targets | primary pass | full geometry pass | proxy | Δ proxy vs 337 | max Δ178 | failed proj | subject-hole |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(report["variants"][:20], start=1):
        lines.append(
            f"| {rank} | `{row['name']}` | `{row['targets']}` | `{row['primary_gate_pass']}` | "
            f"`{row['full_geometry_pass']}` | {row['conservative_public_risk_proxy']:.6f} | "
            f"{row['proxy_delta_vs_id337']:+.6f} | {row['geometry']['max_delta_vs_178']:.6f} | "
            f"{row['geometry']['projection_failed']:+.6f} | {row['subject_hole_loss']:.6f} |"
        )
    best = report["best_variant"]
    lines.extend(
        [
            "",
            "## Best variant split rows",
            "",
            f"Best variant: `{best['name']}`",
            "",
            "| split | primary | loss | ID178 | Δ vs ID178 | pass |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in best["split_rows"]:
        delta = row["delta_vs_id178"]
        lines.append(
            f"| `{row['split']}` | `{row['is_primary']}` | {row['mean_loss']:.6f} | "
            f"{row['id178_loss']:.6f} | {delta:+.6f} | `{delta < 0}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Has variant for artifact review: `{report['decision']['has_variant_for_artifact_review']}`",
            "- Promote submission candidate now: **no**",
            "- Create ID340 artifact now: **no**",
            "- Submit now: **no**",
            "",
            report["decision"]["read"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "id340_sequence_subset_geometry_scan_20260618.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
