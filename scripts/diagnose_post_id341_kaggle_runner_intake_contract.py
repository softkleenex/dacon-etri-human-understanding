#!/usr/bin/env python3
"""Validate the post-ID341 Kaggle runner ↔ output-intake report contract.

This report-only diagnostic compares three no-submit contracts:
1. remote report filenames created by the Kaggle upstream rebuild notebook,
2. remote report filenames staged by that notebook into Kaggle outputs, and
3. remote report filenames required by the local output-intake gate.

It never runs Kaggle, trains models, creates predictions, or creates/submits CSVs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import nbformat

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))
DEFAULT_NOTEBOOK = ROOT / "notebooks" / "kaggle_upstream_rebuild_runner" / "Upstream_Rebuild_Post_ID341_Runner.ipynb"
DEFAULT_METADATA = ROOT / "notebooks" / "kaggle_upstream_rebuild_runner" / "kernel-metadata.json"
INTAKE_SCRIPT = ROOT / "scripts" / "diagnose_post_id341_kaggle_output_intake_gate.py"
REMOTE_REPORT_RE = re.compile(r"remote_[A-Za-z0-9_]+_2026061[89]\.json")
PATH_ASSIGN_RE = re.compile(
    r"(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*project_dir\s*/\s*[\"']reports[\"']\s*/\s*[\"'](?P<name>remote_[A-Za-z0-9_]+_2026061[89]\.json)[\"']"
)
REPORT_ROOTS_RE = re.compile(r"report_roots\s*=\s*\[(?P<body>.*?)\]", re.DOTALL)
IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
FORBIDDEN_NOTEBOOK_ACTION_PATTERNS = {
    "csv_write_call": re.compile(r"\.to_csv\s*\("),
    "submission_or_base_output_path": re.compile(
        r"out_dir\s*/\s*[\"'][^\"']*(?:submission|base_oof|base_test)[^\"']*[\"']",
        re.IGNORECASE,
    ),
    "submission_log_write": re.compile(
        r"submission_log\.csv[^\n]*(?:write_text|write_bytes|open\s*\(|to_csv|copy2\s*\([^,\n]+,\s*out_dir)",
        re.IGNORECASE,
    ),
    "dacon_submit_api": re.compile(
        r"\b(?:dacon_submit|dacon_submit_api|submit_api)\b|\bdacon\s+submit\b|\.submit\s*\(",
        re.IGNORECASE,
    ),
    "kaggle_network_cli": re.compile(
        r"[\"'][^\"']*\bkaggle\s+(?:datasets|kernels)\s+(?:version|push|output|status|init|pull)[^\"']*[\"']",
        re.IGNORECASE,
    ),
    "forbidden_true_decision_flag": re.compile(
        r"[\"'](?:create_submission_artifact|submit_now|push_kaggle_now|run_kaggle_now)[\"']\s*:\s*True"
    ),
    "base_prediction_write": re.compile(
        r"\b(?:base_oof|base_test)\b[^\n]*(?:write_text|write_bytes|to_pickle|to_parquet|np\.save|joblib\.dump|pickle\.dump)",
        re.IGNORECASE,
    ),
}

REMOTE_Q3S3_NEGATIVE_MAP = "remote_post_id341_q3s3_negative_evidence_map_20260618.json"
REMOTE_Q3S3_FOLD_NESTED_SIBLING = "remote_post_id341_q3s3_fold_nested_sibling_bridge_smoke_20260619.json"
REMOTE_Q3S3_REMAINING_INVENTORY = "remote_post_id341_q3s3_remaining_route_inventory_20260618.json"


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_intake_contract() -> tuple[list[str], str]:
    spec = importlib.util.spec_from_file_location("post_id341_kaggle_output_intake_gate", INTAKE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load intake script: {INTAKE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.EXPECTED_REMOTE_REPORTS), str(module.EXPECTED_RUNNER_MANIFEST)


def notebook_sources(notebook_path: Path) -> list[str]:
    nb = nbformat.read(notebook_path, as_version=4)
    return [cell.get("source", "") for cell in nb.cells]


def notebook_code_sources(notebook_path: Path) -> list[tuple[int, str]]:
    nb = nbformat.read(notebook_path, as_version=4)
    return [(index, cell.get("source", "")) for index, cell in enumerate(nb.cells) if cell.get("cell_type") == "code"]


def extract_generated_remote_reports(sources: list[str]) -> list[str]:
    names: set[str] = set()
    for source in sources:
        names.update(REMOTE_REPORT_RE.findall(source))
    return sorted(names)


def extract_staged_remote_reports(sources: list[str]) -> tuple[list[str], dict[str, str], list[str]]:
    full_source = "\n".join(sources)
    assignments = {match.group("var"): match.group("name") for match in PATH_ASSIGN_RE.finditer(full_source)}
    roots_match = REPORT_ROOTS_RE.search(full_source)
    if roots_match is None:
        return [], assignments, []
    root_vars = [name for name in IDENT_RE.findall(roots_match.group("body")) if name not in {"report_roots"}]
    staged = sorted({assignments[var] for var in root_vars if var in assignments})
    unresolved = [var for var in root_vars if var not in assignments]
    return staged, assignments, unresolved


def runner_manifest_contract(sources: list[str], expected_manifest: str) -> dict[str, Any]:
    full_source = "\n".join(sources)
    safety_flags = ["no_submission_created", "no_dacon_submission_made", "local_new_source_stack_closed"]
    return {
        "expected_manifest": expected_manifest,
        "runner_writes_expected_manifest": expected_manifest in full_source
        and "manifest_path.write_text" in full_source,
        "runner_manifest_safety_flags_present": all(flag in full_source for flag in safety_flags),
        "runner_manifest_staged_files_key_present": "staged_files" in full_source,
    }


def notebook_report_only_safety_contract(code_sources: list[tuple[int, str]]) -> dict[str, Any]:
    """Scan code cells for actions that would violate the no-submit runner contract."""
    forbidden_matches: dict[str, list[dict[str, Any]]] = {}
    for name, pattern in FORBIDDEN_NOTEBOOK_ACTION_PATTERNS.items():
        matches: list[dict[str, Any]] = []
        for cell_index, source in code_sources:
            lines = source.splitlines()
            for match in pattern.finditer(source):
                line_number = source.count("\n", 0, match.start()) + 1
                line_text = lines[line_number - 1].strip() if 0 < line_number <= len(lines) else ""
                matches.append(
                    {
                        "cell_index": cell_index,
                        "line_number": line_number,
                        "line": line_text[:240],
                    }
                )
        forbidden_matches[name] = matches
    no_forbidden_actions = all(not matches for matches in forbidden_matches.values())
    full_code = "\n".join(source for _, source in code_sources)
    required_safety_phrases = {
        "manifest_no_submission_created_true": '"no_submission_created": True' in full_code,
        "manifest_no_dacon_submission_made_true": '"no_dacon_submission_made": True' in full_code,
        "manifest_next_action_forbids_csv_submit": "Do not create or submit a CSV from this runner." in full_code,
        "staging_uses_report_roots": "report_roots = [" in full_code and "for root_path in report_roots:" in full_code,
    }
    return {
        "forbidden_patterns": list(FORBIDDEN_NOTEBOOK_ACTION_PATTERNS),
        "forbidden_matches": forbidden_matches,
        "no_forbidden_actions": no_forbidden_actions,
        "required_safety_phrases": required_safety_phrases,
        "required_safety_phrases_present": all(required_safety_phrases.values()),
        "notebook_report_only_contract_pass": bool(no_forbidden_actions and all(required_safety_phrases.values())),
    }


def q3s3_negative_map_handoff_contract(sources: list[str]) -> dict[str, Any]:
    """Ensure the remote Q3/S3 inventory is based on the remote negative-evidence map."""
    full_source = "\n".join(sources)
    negative_index = full_source.find(REMOTE_Q3S3_NEGATIVE_MAP)
    sibling_index = full_source.find(REMOTE_Q3S3_FOLD_NESTED_SIBLING)
    inventory_index = full_source.find(REMOTE_Q3S3_REMAINING_INVENTORY)
    has_negative_generation = (
        "diagnose_post_id341_q3s3_negative_evidence_map.py --output {q3s3_negative_evidence_path}" in full_source
    )
    has_sibling_generation = (
        "diagnose_post_id341_q3s3_fold_nested_sibling_bridge_smoke.py " in full_source
        and "--negative-evidence-map {q3s3_negative_evidence_path}" in full_source
        and "--output {q3s3_fold_nested_sibling_path}" in full_source
    )
    has_inventory_handoff = "--negative-evidence-map {q3s3_negative_evidence_path}" in full_source
    has_inventory_sibling_handoff = "--fold-nested-sibling-bridge {q3s3_fold_nested_sibling_path}" in full_source
    has_negative_staging = "q3s3_negative_evidence_path" in full_source and re.search(
        r"report_roots\s*=\s*\[(?P<body>.*?)q3s3_negative_evidence_path(?P<tail>.*?)\]",
        full_source,
        flags=re.DOTALL,
    )
    has_sibling_staging = "q3s3_fold_nested_sibling_path" in full_source and re.search(
        r"report_roots\s*=\s*\[(?P<body>.*?)q3s3_fold_nested_sibling_path(?P<tail>.*?)\]",
        full_source,
        flags=re.DOTALL,
    )
    requires_eight_closed_lanes = '["closed_local_lane_count"] == 8' in full_source
    return {
        "remote_negative_map": REMOTE_Q3S3_NEGATIVE_MAP,
        "remote_fold_nested_sibling_bridge": REMOTE_Q3S3_FOLD_NESTED_SIBLING,
        "remote_remaining_inventory": REMOTE_Q3S3_REMAINING_INVENTORY,
        "negative_map_name_present": negative_index >= 0,
        "fold_nested_sibling_name_present": sibling_index >= 0,
        "remaining_inventory_name_present": inventory_index >= 0,
        "negative_map_declared_before_inventory": bool(
            negative_index >= 0 and inventory_index >= 0 and negative_index < inventory_index
        ),
        "fold_nested_sibling_declared_between_negative_and_inventory": bool(
            negative_index >= 0
            and sibling_index >= 0
            and inventory_index >= 0
            and negative_index < sibling_index < inventory_index
        ),
        "negative_map_generation_command_present": has_negative_generation,
        "fold_nested_sibling_generation_command_present": has_sibling_generation,
        "remaining_inventory_uses_negative_map_arg": has_inventory_handoff,
        "remaining_inventory_uses_fold_nested_sibling_arg": has_inventory_sibling_handoff,
        "negative_map_staged_in_report_roots": bool(has_negative_staging),
        "fold_nested_sibling_staged_in_report_roots": bool(has_sibling_staging),
        "negative_map_asserts_eight_closed_lanes": requires_eight_closed_lanes,
        "q3s3_negative_map_handoff_pass": bool(
            negative_index >= 0
            and sibling_index >= 0
            and inventory_index >= 0
            and negative_index < sibling_index < inventory_index
            and has_negative_generation
            and has_sibling_generation
            and has_inventory_handoff
            and has_inventory_sibling_handoff
            and has_negative_staging
            and has_sibling_staging
            and requires_eight_closed_lanes
        ),
    }


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        return {"exists": False}
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "exists": True,
        "id": payload.get("id"),
        "enable_gpu": payload.get("enable_gpu"),
        "enable_tpu": payload.get("enable_tpu"),
        "dataset_sources": payload.get("dataset_sources"),
    }


def build_report(notebook_path: Path, metadata_path: Path) -> dict[str, Any]:
    expected_reports, expected_manifest = load_intake_contract()
    expected = sorted(expected_reports)
    sources = notebook_sources(notebook_path)
    code_sources = notebook_code_sources(notebook_path)
    generated = extract_generated_remote_reports(sources)
    staged, assignments, unresolved = extract_staged_remote_reports(sources)
    manifest_contract = runner_manifest_contract(sources, expected_manifest)
    safety_contract = notebook_report_only_safety_contract(code_sources)
    q3s3_handoff_contract = q3s3_negative_map_handoff_contract(sources)
    missing_from_generation = sorted(set(expected) - set(generated))
    missing_from_staging = sorted(set(expected) - set(staged))
    unexpected_generated = sorted(set(generated) - set(expected))
    unexpected_staged = sorted(set(staged) - set(expected))
    contract_pass = not (
        missing_from_generation
        or missing_from_staging
        or unexpected_generated
        or unexpected_staged
        or unresolved
        or not all(
            [
                manifest_contract["runner_writes_expected_manifest"],
                manifest_contract["runner_manifest_safety_flags_present"],
                manifest_contract["runner_manifest_staged_files_key_present"],
                safety_contract["notebook_report_only_contract_pass"],
                q3s3_handoff_contract["q3s3_negative_map_handoff_pass"],
            ]
        )
    )
    staged_non_report = [name for name in staged if not (name.endswith(".json") and name.startswith("remote_"))]
    staged_submission_like = [
        name
        for name in staged
        if name.lower().endswith(".csv")
        or "submission" in name.lower()
        or "base_oof" in name.lower()
        or "base_test" in name.lower()
    ]
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_kaggle_runner_intake_contract",
        "purpose": (
            "Ensure the no-submit Kaggle runner stages exactly the remote reports required by the local "
            "output-intake gate before any approved Kaggle push/run."
        ),
        "notebook": safe_relative(notebook_path),
        "intake_script": safe_relative(INTAKE_SCRIPT),
        "metadata": load_metadata(metadata_path),
        "counts": {
            "intake_expected": len(expected),
            "runner_generated": len(generated),
            "runner_staged": len(staged),
            "expected_runner_manifest": 1,
            "path_assignments": len(assignments),
        },
        "intake_expected_reports": expected,
        "intake_expected_runner_manifest": expected_manifest,
        "runner_generated_reports": generated,
        "runner_staged_reports": staged,
        "runner_manifest_contract": manifest_contract,
        "notebook_report_only_safety_contract": safety_contract,
        "q3s3_negative_map_handoff_contract": q3s3_handoff_contract,
        "missing_from_generation": missing_from_generation,
        "missing_from_staging": missing_from_staging,
        "unexpected_generated_reports": unexpected_generated,
        "unexpected_staged_reports": unexpected_staged,
        "unresolved_report_root_variables": unresolved,
        "staged_non_report_files": staged_non_report,
        "staged_submission_like_files": staged_submission_like,
        "decision": {
            "runner_intake_contract_pass": contract_pass,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "artifact_contract_review_allowed": False,
            "next_action": (
                "Runner staging matches the intake gate; explicit user approval is still required for Kaggle push/run."
                if contract_pass
                else "Fix runner/intake remote report filename mismatches before any Kaggle push/run."
            ),
        },
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_history_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# Post-ID341 Kaggle runner ↔ intake contract",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This report is local and no-submit. It compares the runner notebook's remote report filenames with the local output-intake gate.",
        "",
        "## Decision",
        "",
        f"- Runner/intake contract pass: `{decision['runner_intake_contract_pass']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Counts",
        "",
    ]
    for key, value in report["counts"].items():
        lines.append(f"- {key}: `{value}`")
    manifest_contract = report["runner_manifest_contract"]
    safety_contract = report["notebook_report_only_safety_contract"]
    q3s3_contract = report["q3s3_negative_map_handoff_contract"]
    lines.extend(
        [
            "",
            "## Runner manifest contract",
            "",
            f"- Expected manifest: `{manifest_contract['expected_manifest']}`",
            f"- Runner writes expected manifest: `{manifest_contract['runner_writes_expected_manifest']}`",
            f"- Safety flags present: `{manifest_contract['runner_manifest_safety_flags_present']}`",
            f"- Staged-files key present: `{manifest_contract['runner_manifest_staged_files_key_present']}`",
        ]
    )
    lines.extend(
        [
            "",
            "## Notebook report-only safety contract",
            "",
            f"- No forbidden actions: `{safety_contract['no_forbidden_actions']}`",
            f"- Required safety phrases present: `{safety_contract['required_safety_phrases_present']}`",
            f"- Report-only contract pass: `{safety_contract['notebook_report_only_contract_pass']}`",
            f"- Forbidden matches: `{safety_contract['forbidden_matches']}`",
            f"- Required safety phrases: `{safety_contract['required_safety_phrases']}`",
        ]
    )
    lines.extend(
        [
            "",
            "## Q3/S3 negative-map handoff contract",
            "",
            f"- Remote negative map: `{q3s3_contract['remote_negative_map']}`",
            f"- Remote remaining inventory: `{q3s3_contract['remote_remaining_inventory']}`",
            f"- Negative map before inventory: `{q3s3_contract['negative_map_declared_before_inventory']}`",
            f"- Inventory uses negative-map arg: `{q3s3_contract['remaining_inventory_uses_negative_map_arg']}`",
            f"- Negative map staged in report roots: `{q3s3_contract['negative_map_staged_in_report_roots']}`",
            f"- Asserts 8 closed lanes: `{q3s3_contract['negative_map_asserts_eight_closed_lanes']}`",
            f"- Handoff pass: `{q3s3_contract['q3s3_negative_map_handoff_pass']}`",
        ]
    )
    lines.extend(["", "## Mismatches", ""])
    mismatch_keys = [
        "missing_from_generation",
        "missing_from_staging",
        "unexpected_generated_reports",
        "unexpected_staged_reports",
        "unresolved_report_root_variables",
        "staged_submission_like_files",
    ]
    for key in mismatch_keys:
        values = report[key]
        lines.append(f"- {key}: `{values}`")
    lines.extend(["", "## Expected remote reports", ""])
    for name in report["intake_expected_reports"]:
        lines.append(f"- `{name}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_kaggle_runner_intake_contract_20260618.json",
    )
    args = parser.parse_args()
    notebook = args.notebook if args.notebook.is_absolute() else ROOT / args.notebook
    metadata = args.metadata if args.metadata.is_absolute() else ROOT / args.metadata
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = build_report(notebook, metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    print(f"Wrote {safe_relative(output)}")
    print(f"Wrote {safe_relative(output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
