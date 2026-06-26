#!/usr/bin/env python3
"""Post-ID340 micro target scan for sequence-summary blends.

ID340 found a safer Q2-only sequence-summary move at weight 0.01, but the
previous scan did not test smaller weights or all single/pair target subsets.
This diagnostic keeps the source family fixed and asks whether a still smaller
movement can improve the conservative transfer proxy without broad OOF
recombination.

Report-only: no submission CSV, base prediction artifact, notebook/history
artifact, submission-log row, or DACON submission is created.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diagnose_id179_vs_id335_sequence_probe_review import geometry  # noqa: E402
from scripts.diagnose_id340_exact_sequence_full_geometry_gate import (  # noqa: E402
    MODEL_SPEC,
    conservative_public_proxy,
    frame_like,
    load_artifact_pair,
    max_delta_by_target,
    mean_delta_by_target,
    risk_tier,
    train_full_sequence_source,
)
from scripts.diagnose_id340_sequence_subset_geometry_scan import (  # noqa: E402
    PRIMARY_GATE_SPLITS,
    SPLIT_MANIFEST_PATH,
    build_variant_values,
    evaluate_split_variant,
    load_split_manifest,
    rerank_proxy_by_id,
    split_source_predictions,
)
from scripts.diagnose_public_random_conv_sequence import (  # noqa: E402
    RAW_DIR,
    align_table,
    build_daily_sequence_table,
    table_to_tensor,
)
from scripts.diagnose_public_sequence_summary_embedding import sequence_summary_features  # noqa: E402
from scripts.run_strict_raw_feature_experiment import TARGETS, build_feature_frames  # noqa: E402

KST = timezone(timedelta(hours=9))
DEFAULT_OUTPUT = ROOT / "reports" / "post_id340_micro_sequence_target_scan_20260618.json"
BASE_ID = 178
REFERENCE_IDS = [178, 287, 329, 335, 337, 339, 340]
MICRO_WEIGHTS = [0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02]
MAX_TARGET_SET_SIZE = 2
ID340_PROXY = 0.604062


def target_sets() -> dict[str, list[str]]:
    """Build all single and pair target sets with stable compact names."""
    out: dict[str, list[str]] = {}
    for size in range(1, MAX_TARGET_SET_SIZE + 1):
        for combo in itertools.combinations(TARGETS, size):
            name = "_".join(target.lower() for target in combo)
            out[name] = list(combo)
    return out


def split_eval_id178(
    train: Any, folds: list[tuple[np.ndarray, np.ndarray]], id178_values: np.ndarray
) -> dict[str, Any]:
    return evaluate_split_variant(
        train=train,
        folds=folds,
        id178_values=id178_values,
        source_values=id178_values,
        covered=np.ones_like(id178_values, dtype=bool),
        targets=[],
        weight=0.0,
    )


def build_report() -> dict[str, Any]:
    train, test, _metrics_sorted, _sample = build_feature_frames()
    splits = load_split_manifest(SPLIT_MANIFEST_PATH)
    references = {candidate_id: load_artifact_pair(train, test, candidate_id) for candidate_id in REFERENCE_IDS}

    base_oof = references[BASE_ID][0]
    base_test = references[BASE_ID][1]
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
            "id178_eval": split_eval_id178(train, folds, id178_oof_values),
            "source": split_source_predictions(x_train, train, folds),
        }
        for split_name, folds in splits.items()
    }

    id337_proxy = rerank_proxy_by_id(337)
    id339_proxy = rerank_proxy_by_id(339)
    id340_geometry = geometry(references[340][1], anchors)
    id340_rows = []
    for split_name, payload in split_sources.items():
        id340_values = references[340][0][TARGETS].astype(float).to_numpy()
        id340_eval = evaluate_split_variant(
            train=train,
            folds=payload["folds"],
            id178_values=id340_values,
            source_values=id340_values,
            covered=np.ones_like(id340_values, dtype=bool),
            targets=[],
            weight=0.0,
        )
        id178_eval = payload["id178_eval"]
        id340_rows.append(
            {
                "split": split_name,
                "mean_loss": id340_eval["mean_loss"],
                "id178_loss": id178_eval["mean_loss"],
                "delta_vs_id178": float(id340_eval["mean_loss"] - id178_eval["mean_loss"]),
                "is_primary": split_name in PRIMARY_GATE_SPLITS,
            }
        )

    variants: list[dict[str, Any]] = []
    for target_set_name, targets in target_sets().items():
        for weight in MICRO_WEIGHTS:
            split_rows = []
            primary_pass = True
            subject_hole_loss = None
            worst_primary_delta = -1.0
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
                id178_eval = payload["id178_eval"]
                delta = float(split_eval["mean_loss"] - id178_eval["mean_loss"])
                if split_name == "subject_hole_5":
                    subject_hole_loss = float(split_eval["mean_loss"])
                if split_name in PRIMARY_GATE_SPLITS:
                    worst_primary_delta = max(worst_primary_delta, delta)
                    if delta >= 0:
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
            improves_id340_proxy = bool(proxy < ID340_PROXY)
            variants.append(
                {
                    "name": f"{target_set_name}_w{weight:g}",
                    "target_set": target_set_name,
                    "targets": targets,
                    "weight": weight,
                    "split_rows": split_rows,
                    "primary_gate_pass": primary_pass,
                    "worst_primary_delta_vs_id178": worst_primary_delta,
                    "subject_hole_loss": subject_hole_loss,
                    "geometry": geom,
                    "risk_tier": risk_tier(geom),
                    "conservative_public_risk_proxy": proxy,
                    "proxy_delta_vs_id337": None if id337_proxy is None else float(proxy - id337_proxy),
                    "proxy_delta_vs_id339": None if id339_proxy is None else float(proxy - id339_proxy),
                    "proxy_delta_vs_id340_reference": float(proxy - ID340_PROXY),
                    "full_geometry_pass": bool(full_geometry_pass),
                    "improves_id340_proxy": improves_id340_proxy,
                    "artifact_review_pass": bool(full_geometry_pass and improves_id340_proxy),
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
            not row["artifact_review_pass"],
            not row["full_geometry_pass"],
            not row["primary_gate_pass"],
            row["conservative_public_risk_proxy"],
            row["geometry"]["max_delta_vs_178"],
            row["name"],
        ),
    )
    best = variants[0]
    reviewable = [row for row in variants if row["artifact_review_pass"]]
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id340_micro_sequence_target_scan",
        "purpose": (
            "Search smaller single/pair target sequence-summary blends after ID340, while retaining "
            "the frozen split gate and conservative public-transfer proxy."
        ),
        "base_id": BASE_ID,
        "source_model": asdict(MODEL_SPEC),
        "weights": MICRO_WEIGHTS,
        "max_target_set_size": MAX_TARGET_SET_SIZE,
        "target_sets": target_sets(),
        "sequence_feature_summary": {
            "n_sequence_rows": int(len(seq_table)),
            "n_total_channels": int(len(channel_names)),
            "n_summary_features": int(x_train.shape[1]),
            "feature_family_count": int(len(feature_names) // max(len(channel_names), 1)),
        },
        "source_prediction_details": source_details,
        "references": {
            "id337_proxy": id337_proxy,
            "id339_proxy": id339_proxy,
            "id340_reference_proxy_from_prior_report": ID340_PROXY,
            "id340_geometry": id340_geometry,
            "id340_split_rows": id340_rows,
        },
        "variants": variants,
        "best_variant": best,
        "reviewable_variants": reviewable[:20],
        "decision": {
            "promote_submission_candidate_now": False,
            "submit_now": False,
            "has_variant_for_artifact_review": bool(reviewable),
            "best_variant": best["name"],
            "best_artifact_review_pass": bool(best["artifact_review_pass"]),
            "best_public_058_plausible": bool(best["public_058_plausible"]),
            "read": (
                "Only promote a new artifact if it beats ID340's proxy, passes all primary frozen splits, "
                "and remains low-movement versus ID178. This diagnostic itself creates no submission artifact."
            ),
        },
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Post-ID340 micro sequence target scan",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It scans smaller single/pair target sequence-summary blends and creates no CSV, base artifact, notebook/history artifact, submission-log row, or DACON submission.",
        "",
        f"- Base: `ID{report['base_id']}`",
        f"- Source model: `{report['source_model']}`",
        f"- Weights: `{report['weights']}`",
        f"- ID337 proxy: `{report['references']['id337_proxy']:.6f}`",
        f"- ID339 proxy: `{report['references']['id339_proxy']:.6f}`",
        f"- Prior ID340 proxy: `{report['references']['id340_reference_proxy_from_prior_report']:.6f}`",
        f"- Reviewable variants: `{len(report['reviewable_variants'])}`",
        "",
        "## Top variants",
        "",
        "| rank | variant | targets | artifact review | primary pass | full geometry | proxy | Δ proxy vs ID340 | max Δ178 | failed proj | worst primary Δ | subject-hole |",
        "|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(report["variants"][:30], start=1):
        lines.append(
            f"| {rank} | `{row['name']}` | `{row['targets']}` | `{row['artifact_review_pass']}` | "
            f"`{row['primary_gate_pass']}` | `{row['full_geometry_pass']}` | "
            f"{row['conservative_public_risk_proxy']:.6f} | {row['proxy_delta_vs_id340_reference']:+.6f} | "
            f"{row['geometry']['max_delta_vs_178']:.6f} | {row['geometry']['projection_failed']:+.6f} | "
            f"{row['worst_primary_delta_vs_id178']:+.6f} | {row['subject_hole_loss']:.6f} |"
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
            f"- Best artifact-review pass: `{report['decision']['best_artifact_review_pass']}`",
            f"- Best Public 0.58 plausible: `{report['decision']['best_public_058_plausible']}`",
            "- Promote submission candidate now: **no**",
            "- Submit now: **no**",
            "",
            report["decision"]["read"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
