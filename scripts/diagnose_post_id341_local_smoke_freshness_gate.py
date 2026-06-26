#!/usr/bin/env python3
"""Check whether the latest post-ID341 local runner smoke is still fresh.

This report-only diagnostic compares the input fingerprints recorded by the
local no-network runner smoke with the current runner/package inputs. It prevents
using a stale smoke pass after the notebook, packaging Makefile, runner metadata,
or runner-called scripts have changed.

It creates no predictions, submission artifacts, Kaggle runs, or DACON
submissions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))
SMOKE_REPORT = REPORTS_DIR / "post_id341_kaggle_runner_local_smoke_20260618.json"
SMOKE_SCRIPT = ROOT / "scripts" / "run_post_id341_kaggle_runner_local_smoke.py"
INTAKE_REPORT = REPORTS_DIR / "post_id341_kaggle_local_runner_smoke_intake_20260618.json"


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_smoke_module():
    spec = importlib.util.spec_from_file_location("post_id341_local_smoke", SMOKE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SMOKE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report_file_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": safe_relative(path), "exists": False, "parse_ok": False, "decision": None, "error": None}
    try:
        payload = load_json(path)
        return {
            "path": safe_relative(path),
            "exists": True,
            "parse_ok": True,
            "bytes": path.stat().st_size,
            "created_at_kst": payload.get("created_at_kst"),
            "decision": payload.get("decision", {}),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "path": safe_relative(path),
            "exists": True,
            "parse_ok": False,
            "bytes": path.stat().st_size,
            "decision": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def fingerprint_diff(
    recorded: dict[str, dict[str, Any]], current: dict[str, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    missing_now: list[dict[str, Any]] = []
    new_now: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    for rel, old in sorted(recorded.items()):
        new = current.get(rel)
        if new is None:
            missing_now.append({"path": rel, "recorded": old})
            continue
        if old.get("exists") != new.get("exists") or old.get("sha256") != new.get("sha256"):
            changed.append(
                {
                    "path": rel,
                    "recorded_sha256": old.get("sha256"),
                    "current_sha256": new.get("sha256"),
                    "recorded_exists": old.get("exists"),
                    "current_exists": new.get("exists"),
                }
            )
    for rel, new in sorted(current.items()):
        if rel not in recorded:
            new_now.append({"path": rel, "current": new})
    return {"missing_now": missing_now, "new_now": new_now, "changed": changed}


def build_report() -> dict[str, Any]:
    smoke_status = report_file_status(SMOKE_REPORT)
    intake_status = report_file_status(INTAKE_REPORT)
    smoke_payload = load_json(SMOKE_REPORT) if smoke_status["parse_ok"] else {}
    intake_payload = load_json(INTAKE_REPORT) if intake_status["parse_ok"] else {}
    recorded_fingerprints = smoke_payload.get("input_fingerprints", {})
    smoke_module = load_smoke_module()
    current_fingerprints = smoke_module.collect_input_fingerprints()
    diff = (
        fingerprint_diff(recorded_fingerprints, current_fingerprints)
        if recorded_fingerprints
        else {
            "missing_now": [],
            "new_now": [],
            "changed": [],
        }
    )
    intake_decision = intake_payload.get("decision", {})
    fingerprints_match = bool(
        recorded_fingerprints and not diff["missing_now"] and not diff["new_now"] and not diff["changed"]
    )
    smoke_decision = smoke_payload.get("decision", {})
    local_smoke_fresh = bool(
        smoke_decision.get("local_runner_smoke_passed") is True
        and smoke_decision.get("create_submission_artifact") is False
        and smoke_decision.get("submit_now") is False
        and smoke_decision.get("push_kaggle_now") is False
        and smoke_decision.get("run_kaggle_now") is False
        and intake_decision.get("all_expected_reports_present") is True
        and intake_decision.get("runner_manifest_ok") is True
        and intake_decision.get("unexpected_submission_like_files_present") is False
        and intake_decision.get("remote_forbidden_actions_present") is False
        and intake_decision.get("create_submission_artifact") is False
        and intake_decision.get("submit_now") is False
        and fingerprints_match
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_local_smoke_freshness_gate",
        "purpose": "Verify the latest local no-network runner smoke still matches current runner/package inputs.",
        "inputs": {
            "smoke_report": smoke_status,
            "smoke_intake_report": intake_status,
            "smoke_script": safe_relative(SMOKE_SCRIPT),
        },
        "recorded_input_fingerprint_count": len(recorded_fingerprints),
        "current_input_fingerprint_count": len(current_fingerprints),
        "fingerprint_diff": diff,
        "checks": {
            "smoke_report_passed": smoke_decision.get("local_runner_smoke_passed") is True,
            "smoke_intake_safe": bool(
                intake_decision.get("all_expected_reports_present") is True
                and intake_decision.get("runner_manifest_ok") is True
                and intake_decision.get("unexpected_submission_like_files_present") is False
                and intake_decision.get("remote_forbidden_actions_present") is False
            ),
            "fingerprints_recorded": bool(recorded_fingerprints),
            "fingerprints_match_current_inputs": fingerprints_match,
        },
        "decision": {
            "local_runner_smoke_fresh": local_smoke_fresh,
            "requires_local_runner_smoke_rerun": not local_smoke_fresh,
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "next_action": (
                "Local runner smoke is fresh; explicit approval is still required for any Kaggle network run."
                if local_smoke_fresh
                else "Rerun make kaggle-local-runner-smoke before consuming Kaggle approval."
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
    checks = report["checks"]
    diff = report["fingerprint_diff"]
    lines = [
        "# Post-ID341 local smoke freshness gate",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is local and no-network. It checks whether the latest local runner smoke still matches the current runner/package inputs.",
        "",
        "## Decision",
        "",
        f"- Local runner smoke fresh: `{decision['local_runner_smoke_fresh']}`",
        f"- Requires local runner smoke rerun: `{decision['requires_local_runner_smoke_rerun']}`",
        f"- Create submission artifact: `{decision['create_submission_artifact']}`",
        f"- Submit now: `{decision['submit_now']}`",
        f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
        f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
        f"- Next action: {decision['next_action']}",
        "",
        "## Checks",
        "",
    ]
    for key, value in checks.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Fingerprints",
            "",
            f"- Recorded count: `{report['recorded_input_fingerprint_count']}`",
            f"- Current count: `{report['current_input_fingerprint_count']}`",
            f"- Changed: `{len(diff['changed'])}`",
            f"- Missing now: `{len(diff['missing_now'])}`",
            f"- New now: `{len(diff['new_now'])}`",
            "",
        ]
    )
    if diff["changed"] or diff["missing_now"] or diff["new_now"]:
        lines.extend(["```json", json.dumps(diff, indent=2, ensure_ascii=False), "```", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "post_id341_local_smoke_freshness_gate_20260618.json",
    )
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
