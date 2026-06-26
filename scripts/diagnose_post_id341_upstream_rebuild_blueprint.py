#!/usr/bin/env python3
"""Blueprint the post-ID341 upstream model-level Kaggle rebuild.

Local artifact recombination lanes are now closed or blocked. This diagnostic
does not train models; it freezes the exact report-only Kaggle rebuild contract
needed before spending remote compute on A1/B1 upstream source reconstruction.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
OUTPUT_PATH = REPORTS_DIR / "post_id341_upstream_rebuild_blueprint_20260618.json"
KST = timezone(timedelta(hours=9))

UPSTREAM_SOURCE_IDS = [221, 226, 227, 228, 237]
A1_REPLAY_SOURCE_IDS = [238, 239, 241]
B1_SOURCE_IDS = [254, 255, 256, 257, 258, 266]
ANCHOR_IDS = [178, 287, 329, 335, 341]
REQUIRED_REPORTS = [
    REPORTS_DIR / "post_id341_family_priority_20260618.json",
    REPORTS_DIR / "post_id341_a1_qfree_lane_closure_20260618.json",
    REPORTS_DIR / "b1_pair_blend_source_dependency_audit_20260618.json",
    REPORTS_DIR / "id340_validation_split_manifest_20260618.json",
]


def experiment_metadata_path(experiment_id: int) -> Path | None:
    paths = sorted(REPORTS_DIR.glob(f"experiment_{experiment_id}_*.json"))
    if len(paths) == 1:
        return paths[0]
    return None


def compact_experiment_record(experiment_id: int) -> dict[str, Any]:
    path = experiment_metadata_path(experiment_id)
    if path is None:
        return {
            "id": experiment_id,
            "metadata_path": None,
            "exists": False,
            "method": None,
            "local_cv": None,
            "validation_lens": None,
            "selection_contract": None,
            "direct_upstream_ids": [],
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    selector_info = data.get("selector_info", {})
    upstream: set[int] = set()
    for key in ["artifact_ids", "direct_upstream_ids", "upstream_ids"]:
        for value in selector_info.get(key, []) or []:
            try:
                upstream.add(int(value))
            except (TypeError, ValueError):
                pass
    target_details = selector_info.get("target_details", {})
    if isinstance(target_details, dict):
        for detail in target_details.values():
            if not isinstance(detail, dict):
                continue
            for key in ["id_a", "id_b", "base_id", "extreme_id"]:
                if key in detail:
                    try:
                        upstream.add(int(detail[key]))
                    except (TypeError, ValueError):
                        pass
    return {
        "id": experiment_id,
        "metadata_path": str(path.relative_to(ROOT)),
        "exists": True,
        "method": data.get("method"),
        "local_cv": data.get("local_cv"),
        "validation_lens": selector_info.get("validation_lens"),
        "selection_contract": selector_info.get("selection_contract"),
        "direct_upstream_ids": sorted(upstream),
    }


def check_required_files() -> list[dict[str, Any]]:
    required_paths = [
        ROOT / "dataset-metadata.json",
        ROOT / "pyproject.toml",
        ROOT / "Makefile",
        *REQUIRED_REPORTS,
    ]
    return [
        {
            "path": str(path.relative_to(ROOT)),
            "exists": path.exists(),
        }
        for path in required_paths
    ]


def build_report() -> dict[str, Any]:
    missing_required = [row["path"] for row in check_required_files() if not row["exists"]]
    upstream_records = [compact_experiment_record(experiment_id) for experiment_id in UPSTREAM_SOURCE_IDS]
    a1_records = [compact_experiment_record(experiment_id) for experiment_id in A1_REPLAY_SOURCE_IDS]
    b1_records = [compact_experiment_record(experiment_id) for experiment_id in B1_SOURCE_IDS]
    anchor_records = [compact_experiment_record(experiment_id) for experiment_id in ANCHOR_IDS]
    phases = [
        {
            "phase": "environment_and_payload_smoke",
            "compute": "Kaggle CPU, no GPU/TPU",
            "purpose": "Unpack project payload, verify data/raw + artifacts/base_predictions + frozen reports exist.",
            "must_output": ["payload_manifest.json", "environment_versions.json"],
        },
        {
            "phase": "frozen_split_reconstruction",
            "compute": "CPU",
            "purpose": "Load id340_validation_split_manifest and verify fold row counts, subject coverage, and target prevalence.",
            "must_output": ["frozen_split_audit.json", "frozen_split_audit.md"],
        },
        {
            "phase": "upstream_source_model_rebuild",
            "compute": "CPU first; GPU disabled unless a later DL-specific parity note exists",
            "purpose": (
                "Rebuild model-level approximations below IDs 221/226/227/228/237 inside frozen folds. "
                "Do not select/finalize using outer validation rows."
            ),
            "must_output": ["source_rebuild_oof.parquet", "source_rebuild_metrics.json", "source_rebuild_metrics.md"],
        },
        {
            "phase": "a1_b1_replay_from_rebuilt_sources",
            "compute": "CPU",
            "purpose": "Replay A1 qfree and B1 pair/logit selectors using rebuilt OOF sources only.",
            "must_output": ["a1_b1_replay_metrics.json", "a1_b1_replay_metrics.md"],
        },
        {
            "phase": "transfer_gate_only_no_artifact",
            "compute": "CPU",
            "purpose": "Compare any rebuilt-source candidate against ID178/287/329/335/341 geometry and ID341 proxy.",
            "must_output": ["transfer_gate_post_rebuild.json", "transfer_gate_post_rebuild.md"],
        },
    ]
    hard_gates = [
        "No DACON submission from the Kaggle runner.",
        "No generated submission CSV or submission_log row from the Kaggle runner.",
        "No full-OOF calibration or target selector may use the same rows for fit/select/score.",
        "A rebuilt candidate must beat ID341 proxy 0.6038297835 before local artifact review.",
        "Any promoted artifact after pullback requires unique ID > 341 and the full artifact contract.",
    ]
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_upstream_rebuild_blueprint",
        "purpose": "Freeze the report-only Kaggle upstream rebuild contract after local OOF artifact lanes closed.",
        "required_file_checks": check_required_files(),
        "missing_required_files": missing_required,
        "source_records": {
            "upstream_model_layer": upstream_records,
            "a1_replay_layer": a1_records,
            "b1_blocked_layer": b1_records,
            "public_anchor_layer": anchor_records,
        },
        "phases": phases,
        "hard_gates": hard_gates,
        "kaggle_runner": {
            "planned_notebook_dir": "notebooks/kaggle_upstream_rebuild_runner",
            "planned_slug": "softkleenex/dacon-etri-upstream-rebuild-post-id341",
            "enable_gpu": False,
            "enable_tpu": False,
            "dataset_sources": [
                "softkleenex/dacon-etri-human-base",
                "softkleenex/dacon-etri-upstream-patch-post-id341",
            ],
            "push_now": False,
        },
        "decision": {
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "blueprint_ready": not missing_required,
            "next_action": (
                "Create a report-only Kaggle runner scaffold for this blueprint, validate pre-flight locally, "
                "then ask before any kernel push/run if a long rebuild will be launched."
            ),
        },
        "no_submission_created": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Post-ID341 upstream rebuild blueprint",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It trains no model, creates no submission artifact, and makes no DACON submission.",
        "",
        "## Read",
        "",
        report["purpose"],
        "",
        "## Required file checks",
        "",
        "| path | exists |",
        "|---|---|",
    ]
    for row in report["required_file_checks"]:
        lines.append(f"| `{row['path']}` | `{row['exists']}` |")
    lines.extend(
        [
            "",
            "## Rebuild phases",
            "",
            "| phase | compute | purpose | must output |",
            "|---|---|---|---|",
        ]
    )
    for phase in report["phases"]:
        lines.append(
            f"| `{phase['phase']}` | {phase['compute']} | {phase['purpose']} | `{', '.join(phase['must_output'])}` |"
        )
    lines.extend(
        [
            "",
            "## Source records",
            "",
            "| layer | ids |",
            "|---|---|",
        ]
    )
    for layer, records in report["source_records"].items():
        ids = ", ".join(str(row["id"]) for row in records)
        lines.append(f"| `{layer}` | `{ids}` |")
    lines.extend(["", "## Hard gates", ""])
    for gate in report["hard_gates"]:
        lines.append(f"- {gate}")
    decision = report["decision"]
    runner = report["kaggle_runner"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Blueprint ready: `{decision['blueprint_ready']}`",
            "- Create submission artifact: **no**",
            "- Submit now: **no**",
            "- Push Kaggle now: **no**",
            f"- Planned notebook dir: `{runner['planned_notebook_dir']}`",
            f"- Planned slug: `{runner['planned_slug']}`",
            "",
            decision["next_action"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    output = args.output_json if args.output_json.is_absolute() else ROOT / args.output_json
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {output.with_suffix('.md').relative_to(ROOT)}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
