#!/usr/bin/env python3
"""Build the pre-ID340 validation split manifest.

The manifest freezes the validation lenses that a future ID340 source rebuild
must survive. It creates no model, prediction, submission CSV, or submission-log
row.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_test_split_fold_alignment import (  # noqa: E402
    _actual_test_band_holdout,
    _build_timeline,
    _test_neighbor_holdout,
)
from src.utils.validation import make_subject_hole_folds, make_subject_timeseries_folds  # noqa: E402

KST = timezone(timedelta(hours=9))


def fold_record(
    name: str, fold_id: int, train_idx: np.ndarray, valid_idx: np.ndarray, train: pd.DataFrame
) -> dict[str, Any]:
    valid = train.loc[valid_idx].copy()
    return {
        "split_name": name,
        "fold_id": int(fold_id),
        "train_idx": [int(x) for x in train_idx],
        "valid_idx": [int(x) for x in valid_idx],
        "train_n": int(len(train_idx)),
        "valid_n": int(len(valid_idx)),
        "valid_subject_counts": {str(k): int(v) for k, v in valid["subject_id"].value_counts().sort_index().items()},
        "valid_min_lifelog_date": str(valid["lifelog_date"].min().date()),
        "valid_max_lifelog_date": str(valid["lifelog_date"].max().date()),
    }


def loso_folds(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = []
    all_idx = train.index.to_numpy(dtype=int)
    for subject_id in sorted(train["subject_id"].astype(str).unique()):
        valid_idx = train.index[train["subject_id"].astype(str).eq(subject_id)].to_numpy(dtype=int)
        train_idx = np.setdiff1d(all_idx, valid_idx, assume_unique=False)
        folds.append((train_idx, valid_idx))
    return folds


def build_manifest() -> dict[str, Any]:
    train = pd.read_csv(ROOT / "data/raw/ch2026_metrics_train.csv", parse_dates=["sleep_date", "lifelog_date"])
    test = pd.read_csv(ROOT / "data/raw/ch2026_submission_sample.csv", parse_dates=["sleep_date", "lifelog_date"])
    timeline = _build_timeline(train, test)
    split_fns = {
        "subject_hole_5": make_subject_hole_folds(train, n_folds=5),
        "subject_timeseries_5": make_subject_timeseries_folds(train, n_folds=5),
        "actual_test_band_holdout": _actual_test_band_holdout(train, timeline, bins=10),
        "test_boundary_neighbor2_holdout": _test_neighbor_holdout(train, timeline, neighbor_positions=2),
        "test_boundary_neighbor5_holdout": _test_neighbor_holdout(train, timeline, neighbor_positions=5),
        "loso_10": loso_folds(train),
    }

    records: list[dict[str, Any]] = []
    split_summary: dict[str, Any] = {}
    for split_name, folds in split_fns.items():
        split_records = [
            fold_record(split_name, fold_id, np.asarray(train_idx), np.asarray(valid_idx), train)
            for fold_id, (train_idx, valid_idx) in enumerate(folds)
        ]
        records.extend(split_records)
        split_summary[split_name] = {
            "n_folds": len(split_records),
            "valid_n_min": int(min(row["valid_n"] for row in split_records)),
            "valid_n_max": int(max(row["valid_n"] for row in split_records)),
            "valid_n_mean": float(np.mean([row["valid_n"] for row in split_records])),
        }

    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "manifest": "id340_validation_split_manifest",
        "purpose": (
            "Freeze validation lenses before any ID340 source rebuild. These splits are for model/source evaluation, "
            "not for optimizing a public leaderboard proxy."
        ),
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "split_summary": split_summary,
        "folds": records,
        "promotion_gate_for_future_id340": {
            "requires_true_nested_source_rebuild": True,
            "requires_subject_hole_pass": True,
            "requires_test_geometry_stress_pass": True,
            "requires_loso_sanity": True,
            "requires_public_risk_proxy_below": 0.590,
            "forbids_existing_full_oof_recombination_only": True,
            "requires_unique_full_artifact_contract_before_submission_discussion": True,
        },
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# ID340 validation split manifest",
        "",
        f"Created: `{manifest['created_at_kst']}`",
        "",
        "This freezes the validation lenses for any future ID340 source rebuild. No model was trained, no predictions were created, and no DACON submission was made.",
        "",
        "## Split summary",
        "",
        "| split | folds | valid min | valid mean | valid max | role |",
        "|---|---:|---:|---:|---:|---|",
    ]
    roles = {
        "subject_hole_5": "current strict local baseline lens",
        "subject_timeseries_5": "chronological same-subject stress",
        "actual_test_band_holdout": "known sample-row relative-position band stress",
        "test_boundary_neighbor2_holdout": "near hidden-test block boundary stress",
        "test_boundary_neighbor5_holdout": "wider hidden-test block boundary stress",
        "loso_10": "subject robustness sanity check",
    }
    for split_name, row in manifest["split_summary"].items():
        lines.append(
            f"| `{split_name}` | {row['n_folds']} | {row['valid_n_min']} | "
            f"{row['valid_n_mean']:.1f} | {row['valid_n_max']} | {roles.get(split_name, '')} |"
        )
    gate = manifest["promotion_gate_for_future_id340"]
    lines.extend(
        [
            "",
            "## Future ID340 gate",
            "",
            f"- True nested/source rebuild required: `{gate['requires_true_nested_source_rebuild']}`",
            f"- Existing full-OOF recombination alone forbidden: `{gate['forbids_existing_full_oof_recombination_only']}`",
            f"- Public-risk proxy ceiling before artifact promotion: `{gate['requires_public_risk_proxy_below']}`",
            "- Required lenses: subject-hole, test-geometry stress, and LOSO sanity.",
            "- A unique full artifact contract is required before any submission discussion.",
            "",
            "## Operational read",
            "",
            "The next useful work is not ID340-as-a-CSV. It is a source rebuild that is selected inside these splits and only then compared to ID178/ID329/ID287 geometry. This is the guardrail that prevents another ID287/ID329-style local-CV mirage.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "id340_validation_split_manifest_20260618.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    manifest = build_manifest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(manifest) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(manifest["promotion_gate_for_future_id340"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
