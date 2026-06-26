#!/usr/bin/env python3
"""Validate freshness of the post-ID341 final prelaunch gate.

This report-only diagnostic checks that the final prelaunch report is newer
than every file it records as an input, and that `make kaggle-upstream-prelaunch-local`
runs the final material-source dependency chain before the final prelaunch gate
and downstream safe-wait/triage reports after the final prelaunch freshness
contract.

It does not run Kaggle, train models, create predictions, create a submission
CSV, edit the submission log, or submit to DACON.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))

DEFAULT_PRELAUNCH = REPORTS_DIR / "post_id341_kaggle_prelaunch_readiness_gate_20260618.json"
DEFAULT_MAKEFILE = ROOT / "Makefile"
DEFAULT_OUTPUT = REPORTS_DIR / "post_id341_prelaunch_freshness_contract_20260619.json"

MATERIAL_COVERAGE_CMD = "scripts/diagnose_post_id341_material_source_remote_coverage_contract.py"
MATERIAL_QUEUE_CMD = "scripts/diagnose_post_id341_material_new_source_queue.py"
PRELAUNCH_CMD = "scripts/diagnose_post_id341_kaggle_prelaunch_readiness_gate.py"
FRESHNESS_CMD = "scripts/diagnose_post_id341_prelaunch_freshness_contract.py"
SELF_SUPERVISED_DESIGN_CMD = "scripts/diagnose_post_id341_self_supervised_subject_state_embedding_design.py"
SLEEP_PROXY_DESIGN_CMD = "scripts/diagnose_post_id341_sleep_proxy_threshold_residual_design.py"
SELF_SUPERVISED_HEAVY_CMD = "scripts/diagnose_post_id341_self_supervised_heavy_compute_contract.py"
SAFE_WAIT_CMD = "scripts/diagnose_post_id341_safe_wait_state_contract.py"
NEXT_ROUTE_TRIAGE_CMD = "scripts/diagnose_post_id341_next_material_route_triage.py"


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def report_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": safe_relative(path), "exists": False, "parse_ok": False, "decision": {}}
    try:
        payload = load_json(path)
    except Exception as exc:  # noqa: BLE001
        return {
            "path": safe_relative(path),
            "exists": True,
            "parse_ok": False,
            "mtime": path.stat().st_mtime,
            "decision": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "path": safe_relative(path),
        "exists": True,
        "parse_ok": True,
        "mtime": path.stat().st_mtime,
        "created_at_kst": payload.get("created_at_kst"),
        "decision": payload.get("decision", {}),
    }


def prelaunch_input_rows(prelaunch_path: Path) -> list[dict[str, Any]]:
    payload = load_json(prelaunch_path)
    prelaunch_mtime = prelaunch_path.stat().st_mtime
    rows: list[dict[str, Any]] = []
    for key, value in payload.get("inputs", {}).items():
        if not isinstance(value, dict) or not value.get("path"):
            continue
        input_path = resolve_path(str(value["path"]))
        exists = input_path.exists()
        mtime = input_path.stat().st_mtime if exists else None
        rows.append(
            {
                "input_key": key,
                "path": safe_relative(input_path),
                "exists": exists,
                "parse_ok_recorded_by_prelaunch": value.get("parse_ok"),
                "mtime": mtime,
                "newer_than_prelaunch": bool(mtime is not None and mtime > prelaunch_mtime + 1e-6),
            }
        )
    return rows


def make_target_body(makefile_path: Path, target: str) -> list[str]:
    text = makefile_path.read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(target)}:\n(?P<body>(?:\t.*\n)+)", text, flags=re.MULTILINE)
    if not match:
        return []
    return [line.strip() for line in match.group("body").splitlines() if line.strip()]


def makefile_order_contract(makefile_path: Path) -> dict[str, Any]:
    body = make_target_body(makefile_path, "kaggle-upstream-prelaunch-local")
    self_supervised_indices = [index for index, line in enumerate(body) if SELF_SUPERVISED_DESIGN_CMD in line]
    sleep_proxy_indices = [index for index, line in enumerate(body) if SLEEP_PROXY_DESIGN_CMD in line]
    material_queue_indices = [index for index, line in enumerate(body) if MATERIAL_QUEUE_CMD in line]
    material_indices = [index for index, line in enumerate(body) if MATERIAL_COVERAGE_CMD in line]
    prelaunch_indices = [index for index, line in enumerate(body) if PRELAUNCH_CMD in line]
    freshness_indices = [index for index, line in enumerate(body) if FRESHNESS_CMD in line]
    self_supervised_heavy_indices = [index for index, line in enumerate(body) if SELF_SUPERVISED_HEAVY_CMD in line]
    safe_wait_indices = [index for index, line in enumerate(body) if SAFE_WAIT_CMD in line]
    next_triage_indices = [index for index, line in enumerate(body) if NEXT_ROUTE_TRIAGE_CMD in line]
    final_material_queue_index = material_queue_indices[-1] if material_queue_indices else None
    order_ok = bool(
        len(self_supervised_indices) == 1
        and len(sleep_proxy_indices) == 1
        and len(material_queue_indices) >= 1
        and len(material_indices) == 1
        and len(prelaunch_indices) == 1
        and len(freshness_indices) == 1
        and len(self_supervised_heavy_indices) == 1
        and len(safe_wait_indices) == 1
        and len(next_triage_indices) == 1
        and self_supervised_indices[0]
        < sleep_proxy_indices[0]
        < final_material_queue_index
        < material_indices[0]
        < prelaunch_indices[0]
        < freshness_indices[0]
        < self_supervised_heavy_indices[0]
        < safe_wait_indices[0]
        < next_triage_indices[0]
    )
    return {
        "makefile": safe_relative(makefile_path),
        "target_found": bool(body),
        "self_supervised_design_command_count": len(self_supervised_indices),
        "sleep_proxy_design_command_count": len(sleep_proxy_indices),
        "material_queue_command_count": len(material_queue_indices),
        "material_coverage_command_count": len(material_indices),
        "prelaunch_command_count": len(prelaunch_indices),
        "freshness_command_count": len(freshness_indices),
        "self_supervised_heavy_command_count": len(self_supervised_heavy_indices),
        "safe_wait_command_count": len(safe_wait_indices),
        "next_route_triage_command_count": len(next_triage_indices),
        "self_supervised_design_index": self_supervised_indices[0] if self_supervised_indices else None,
        "sleep_proxy_design_index": sleep_proxy_indices[0] if sleep_proxy_indices else None,
        "final_material_queue_index": final_material_queue_index,
        "material_coverage_index": material_indices[0] if material_indices else None,
        "prelaunch_index": prelaunch_indices[0] if prelaunch_indices else None,
        "freshness_index": freshness_indices[0] if freshness_indices else None,
        "self_supervised_heavy_index": self_supervised_heavy_indices[0] if self_supervised_heavy_indices else None,
        "safe_wait_index": safe_wait_indices[0] if safe_wait_indices else None,
        "next_route_triage_index": next_triage_indices[0] if next_triage_indices else None,
        "order_ok": order_ok,
    }


def build_report(prelaunch_path: Path = DEFAULT_PRELAUNCH, makefile_path: Path = DEFAULT_MAKEFILE) -> dict[str, Any]:
    prelaunch = report_status(prelaunch_path)
    input_rows = prelaunch_input_rows(prelaunch_path) if prelaunch["parse_ok"] else []
    missing_inputs = [row for row in input_rows if not row["exists"]]
    stale_inputs = [row for row in input_rows if row["newer_than_prelaunch"]]
    unparsed_recorded_inputs = [row for row in input_rows if row["parse_ok_recorded_by_prelaunch"] is False]
    makefile_order = makefile_order_contract(makefile_path)
    prelaunch_decision = prelaunch.get("decision", {})
    pass_contract = bool(
        prelaunch.get("parse_ok") is True
        and prelaunch_decision.get("prelaunch_ready_for_explicit_approval") is True
        and prelaunch_decision.get("requires_explicit_user_approval") is True
        and prelaunch_decision.get("execute_kaggle_now") is False
        and prelaunch_decision.get("push_kaggle_now") is False
        and prelaunch_decision.get("run_kaggle_now") is False
        and prelaunch_decision.get("create_submission_artifact") is False
        and prelaunch_decision.get("submit_now") is False
        and input_rows
        and not missing_inputs
        and not stale_inputs
        and not unparsed_recorded_inputs
        and makefile_order["order_ok"]
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_prelaunch_freshness_contract",
        "purpose": "Ensure the final prelaunch ready report is not stale relative to its recorded input files.",
        "inputs": {
            "prelaunch_report": prelaunch,
            "makefile": safe_relative(makefile_path),
        },
        "prelaunch_input_rows": input_rows,
        "makefile_order_contract": makefile_order,
        "counts": {
            "prelaunch_recorded_input_paths": len(input_rows),
            "missing_inputs": len(missing_inputs),
            "stale_inputs_newer_than_prelaunch": len(stale_inputs),
            "unparsed_recorded_inputs": len(unparsed_recorded_inputs),
        },
        "stale_inputs": stale_inputs,
        "missing_inputs": missing_inputs,
        "unparsed_recorded_inputs": unparsed_recorded_inputs,
        "decision": {
            "prelaunch_freshness_contract_pass": pass_contract,
            "prelaunch_ready_for_explicit_approval": prelaunch_decision.get("prelaunch_ready_for_explicit_approval"),
            "requires_explicit_user_approval": True,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Prelaunch is fresh; explicit approval is still required for any Kaggle network run."
                if pass_contract
                else "Regenerate the stale prelaunch chain before consuming Kaggle approval."
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
    counts = report["counts"]
    lines = [
        "# Post-ID341 prelaunch freshness contract",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is local and no-network. It only checks prelaunch input freshness and Makefile ordering.",
        "",
        "## Decision",
        "",
        f"- Prelaunch freshness contract pass: `{decision['prelaunch_freshness_contract_pass']}`",
        f"- Prelaunch ready for explicit approval: `{decision['prelaunch_ready_for_explicit_approval']}`",
        f"- Requires explicit user approval: `{decision['requires_explicit_user_approval']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Counts",
        "",
    ]
    for key, value in counts.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Makefile order",
            "",
            f"- Order OK: `{report['makefile_order_contract']['order_ok']}`",
            f"- self-supervised design index: `{report['makefile_order_contract']['self_supervised_design_index']}`",
            f"- sleep-proxy design index: `{report['makefile_order_contract']['sleep_proxy_design_index']}`",
            f"- final material queue index: `{report['makefile_order_contract']['final_material_queue_index']}`",
            f"- material coverage index: `{report['makefile_order_contract']['material_coverage_index']}`",
            f"- prelaunch index: `{report['makefile_order_contract']['prelaunch_index']}`",
            f"- freshness index: `{report['makefile_order_contract']['freshness_index']}`",
            f"- self-supervised heavy-contract index: `{report['makefile_order_contract']['self_supervised_heavy_index']}`",
            f"- safe-wait index: `{report['makefile_order_contract']['safe_wait_index']}`",
            f"- next-route triage index: `{report['makefile_order_contract']['next_route_triage_index']}`",
            "",
        ]
    )
    if report["stale_inputs"] or report["missing_inputs"] or report["unparsed_recorded_inputs"]:
        lines.extend(
            [
                "## Findings",
                "",
                "```json",
                json.dumps(
                    {
                        "stale_inputs": report["stale_inputs"],
                        "missing_inputs": report["missing_inputs"],
                        "unparsed_recorded_inputs": report["unparsed_recorded_inputs"],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                "```",
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
    print(f"Wrote {safe_relative(output)}")
    print(f"Wrote {safe_relative(output.with_suffix('.md'))}")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
