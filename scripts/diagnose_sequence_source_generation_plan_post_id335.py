#!/usr/bin/env python3
"""Build a no-submit source-generation plan from the sequence survivor selector.

The fold-internal sequence selector showed local target-wise signal, but the
public-risk caps forced selection back to ID178. This report translates the
local-only selector choices into concrete report-only source-generation jobs,
preferably for Kaggle when raw sequence work is heavy.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))
SELECTOR_REPORT = REPORTS_DIR / "fold_internal_sequence_survivor_selector_post_id335_20260617.json"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

ID_TO_SOURCE = {
    178: {
        "family": "anchor_random_conv_sequence",
        "role": "confirmed public anchor source",
        "generator": "run_public_multiseed_random_conv_sequence_overlay_experiment.py",
    },
    179: {
        "family": "tempered_random_conv_anchor_blend",
        "role": "low-movement manual-review anchor",
        "generator": "run_public_random_conv_tempered_anchor_blend_experiment.py",
    },
    190: {
        "family": "random_conv_sequence_refresh",
        "role": "S1-heavy random-conv sequence refresh",
        "generator": "diagnose_public_random_conv_sequence.py",
    },
    191: {
        "family": "tempered_remote_s1_anchor_blend",
        "role": "bridge after ID190",
        "generator": "run_public_random_conv_tempered_anchor_blend_experiment.py",
    },
    193: {
        "family": "sequence_summary_embedding",
        "role": "sequence summary embedding overlay",
        "generator": "diagnose_public_sequence_summary_embedding.py",
    },
    198: {
        "family": "smartphone_light_compact",
        "role": "compact smartphone/light side model",
        "generator": "run_public_smartphone_light_compact_overlay_experiment.py",
    },
    200: {
        "family": "sax_paa_sequence_pattern",
        "role": "SAX/PAA sequence-pattern overlay",
        "generator": "run_public_sax_paa_sequence_overlay_experiment.py",
    },
    203: {
        "family": "recurrence_gaf_sequence_image",
        "role": "recurrence/GAF sequence-image overlay",
        "generator": "run_public_recurrence_gaf_sequence_overlay_experiment.py",
    },
    206: {
        "family": "wavelet_spectral_sequence",
        "role": "wavelet/spectral sequence overlay",
        "generator": "run_public_wavelet_spectral_sequence_overlay_experiment.py",
    },
}


def experiment_metadata(candidate_id: int) -> dict[str, Any]:
    matches = sorted(REPORTS_DIR.glob(f"experiment_{candidate_id}_*.json"))
    if len(matches) != 1:
        return {}
    metadata = json.loads(matches[0].read_text(encoding="utf-8"))
    selector = metadata.get("selector_info", {})
    return {
        "experiment_json": str(matches[0].relative_to(ROOT)),
        "slug": metadata.get("slug"),
        "method": metadata.get("method"),
        "local_cv": metadata.get("local_cv"),
        "base_source": selector.get("base_source"),
        "diagnostic_report": selector.get("diagnostic_report"),
        "diagnostic_reports": selector.get("diagnostic_reports"),
        "selection_contract": selector.get("selection_contract"),
        "caution": selector.get("caution"),
    }


def local_only_rows(selector_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for split_name, split_rows in selector_report["splits"].items():
        for row in split_rows:
            if row["scenario"]["name"] == "sequence_local_only":
                rows[split_name] = row
    return rows


def target_choice_counts(rows: dict[str, dict[str, Any]]) -> dict[str, dict[int, int]]:
    out: dict[str, Counter[int]] = {target: Counter() for target in TARGETS}
    for row in rows.values():
        for target, counts in row["choice_counts"].items():
            for raw_candidate_id, count in counts.items():
                out[target][int(raw_candidate_id)] += int(count)
    return {target: dict(counter) for target, counter in out.items()}


def consensus_plan(counts: dict[str, dict[int, int]]) -> list[dict[str, Any]]:
    plan = []
    for target in TARGETS:
        target_counts = Counter(counts[target])
        total = sum(target_counts.values())
        ranked = target_counts.most_common()
        winner_id, winner_count = ranked[0]
        source = ID_TO_SOURCE.get(winner_id, {"family": "unknown", "role": "unknown", "generator": None})
        support = winner_count / total if total else 0.0
        if winner_id == 178:
            priority = "anchor_hold"
        elif support >= 0.70:
            priority = "high"
        elif support >= 0.50:
            priority = "medium"
        else:
            priority = "low"
        plan.append(
            {
                "target": target,
                "winner_id": winner_id,
                "winner_count": winner_count,
                "total_folds": total,
                "support": support,
                "ranked_counts": ranked,
                "family": source["family"],
                "role": source["role"],
                "generator": source["generator"],
                "priority": priority,
                "metadata": experiment_metadata(winner_id),
            }
        )
    return plan


def report_only_commands() -> list[dict[str, Any]]:
    """Concrete no-promote commands. Output names are unique and report-only."""
    return [
        {
            "lane": "S1_random_conv_refresh",
            "targets": ["S1"],
            "priority": "high",
            "commands": [
                (
                    ".venv/bin/python scripts/diagnose_public_random_conv_sequence.py "
                    f"--base-id 173 --n-kernels {n_kernels} --seed {seed} "
                    f"--output reports/id173_random_conv_sequence_post_id335_s1_nk{n_kernels}_seed{seed}_20260617.json"
                )
                for n_kernels, seed in [
                    (512, 45501),
                    (512, 45502),
                    (768, 45601),
                    (768, 45602),
                    (1024, 45701),
                    (1024, 45702),
                ]
            ],
            "note": "Run on Kaggle if raw sequence expansion is heavy; report-only, no promotion.",
        },
        {
            "lane": "Q2_sequence_summary_embedding",
            "targets": ["Q2", "Q3"],
            "priority": "high",
            "commands": [
                (
                    ".venv/bin/python scripts/diagnose_public_sequence_summary_embedding.py "
                    "--base-id 191 "
                    "--output reports/id191_sequence_summary_embedding_post_id335_20260617.json"
                )
            ],
            "note": "Q2 is the strongest consensus target for ID193-like source generation.",
        },
        {
            "lane": "S3_smartphone_light_compact",
            "targets": ["S3"],
            "priority": "high",
            "commands": [
                (
                    ".venv/bin/python scripts/run_public_smartphone_light_compact_overlay_experiment.py "
                    "--base-id 193 "
                    "--output reports/id193_smartphone_light_compact_post_id335_20260617.json"
                )
            ],
            "note": "S3 selects ID198 in 12/15 local-only folds; diagnostic-only unless nested source improves.",
        },
        {
            "lane": "Q1_sax_paa_sequence_pattern",
            "targets": ["Q1"],
            "priority": "high",
            "commands": [
                (
                    ".venv/bin/python scripts/run_public_sax_paa_sequence_overlay_experiment.py "
                    "--base-id 198 "
                    "--output reports/id198_sax_paa_sequence_post_id335_20260617.json"
                )
            ],
            "note": "Q1 consensus is ID200, but support is less clean in subject-hole than LOSO.",
        },
        {
            "lane": "Q3_S2_S4_wavelet_spectral",
            "targets": ["Q3", "S2", "S4"],
            "priority": "medium",
            "commands": [
                (
                    ".venv/bin/python scripts/run_public_wavelet_spectral_sequence_overlay_experiment.py "
                    "--base-id 203 "
                    "--output reports/id203_wavelet_spectral_sequence_post_id335_20260617.json"
                )
            ],
            "note": "Wavelet/spectral dominates Q3/S4 and appears in S2, but movement risk kept caps at ID178.",
        },
        {
            "lane": "S4_recurrence_gaf_alternative",
            "targets": ["S4"],
            "priority": "low",
            "commands": [
                (
                    ".venv/bin/python scripts/run_public_recurrence_gaf_sequence_overlay_experiment.py "
                    "--base-id 200 "
                    "--output reports/id200_recurrence_gaf_sequence_post_id335_20260617.json"
                )
            ],
            "note": "Only an S4 alternative behind ID206; run after wavelet if needed.",
        },
    ]


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Sequence source-generation plan — post ID335",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no CSV, notebook, base artifact, submission-log row, Kaggle push, or DACON submission.",
        "",
        "## Target consensus from local-only fold-internal selector",
        "",
        "| target | winner | support | family | priority | ranked counts |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in report["target_consensus"]:
        lines.append(
            f"| {row['target']} | {row['winner_id']} | {row['support']:.2%} | "
            f"`{row['family']}` | `{row['priority']}` | `{row['ranked_counts']}` |"
        )
    lines.extend(
        [
            "",
            "## Report-only command plan",
            "",
            "Do not use `--promote` in this phase. Pull/review reports before creating any matched artifact.",
            "",
        ]
    )
    for lane in report["report_only_commands"]:
        lines.extend(
            [
                f"### {lane['lane']}",
                "",
                f"- Targets: `{lane['targets']}`",
                f"- Priority: `{lane['priority']}`",
                f"- Note: {lane['note']}",
                "",
                "Commands:",
                "",
            ]
        )
        for command in lane["commands"]:
            lines.extend(["```bash", command, "```", ""])
    lines.extend(["## Decision", "", report["decision"]["reason"], ""])
    return "\n".join(lines)


def run(output_json: Path) -> dict[str, Any]:
    selector_report = json.loads(SELECTOR_REPORT.read_text(encoding="utf-8"))
    rows = local_only_rows(selector_report)
    counts = target_choice_counts(rows)
    consensus = consensus_plan(counts)
    commands = report_only_commands()
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "sequence_source_generation_plan_post_id335",
        "source_selector_report": str(SELECTOR_REPORT.relative_to(ROOT)),
        "local_only_selector_summary": {
            split_name: {
                "overall": row["overall"],
                "robust_score": row["robust_score"],
                "choice_counts": row["choice_counts"],
            }
            for split_name, row in rows.items()
        },
        "target_choice_counts": counts,
        "target_consensus": consensus,
        "report_only_commands": commands,
        "decision": {
            "create_submission_artifact": False,
            "push_kaggle_kernel": False,
            "dacon_submission_made": False,
            "public_058_plausible": False,
            "reason": (
                "The sequence branch has target-wise local signal, but risk-controlled selectors revert to ID178. "
                "The next useful work is report-only source generation for the consensus target families, preferably "
                "on Kaggle for heavier raw-sequence jobs, followed by strict review before any artifact promotion."
            ),
        },
        "no_submission_created": True,
        "no_dacon_submission_made": True,
        "no_kaggle_push_made": True,
    }
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_json.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORTS_DIR / "sequence_source_generation_plan_post_id335_20260617.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.output_json)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    for row in report["target_consensus"]:
        print(
            f"{row['target']}: winner={row['winner_id']} support={row['support']:.2%} "
            f"family={row['family']} priority={row['priority']}"
        )
    print(report["decision"]["reason"])


if __name__ == "__main__":
    main()
