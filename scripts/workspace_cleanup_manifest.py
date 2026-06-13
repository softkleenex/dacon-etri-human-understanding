#!/usr/bin/env python3
"""Summarize intended cleanup changes versus pre-existing workspace churn."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTENDED_PATHS = {
    ".gitignore",
    "AGENTS.md",
    "Makefile",
    "README.md",
    "scripts/audit_competition_workspace.py",
    "scripts/ensemble.py",
    "scripts/normalize_submission_log.py",
    "scripts/workspace_cleanup_manifest.py",
    "reports/workspace_audit.md",
    "reports/workspace_audit.json",
    "reports/submission_blacklist.json",
    "reports/submission_log_normalized.csv",
    "reports/submission_log_original_backup.csv",
    "submissions/submission_log.csv",
}
INTENDED_PREFIXES = ("lightning_logs/", "catboost_info/")


def git_status() -> list[str]:
    output = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    return output.splitlines()


def parse_path(line: str) -> str:
    # Handles normal status lines and rename lines conservatively.
    path = line[3:]
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.strip().strip('"')


def main() -> int:
    intended = []
    external = []
    generated_untracked = []
    for line in git_status():
        path = parse_path(line)
        if path in INTENDED_PATHS or path.startswith(INTENDED_PREFIXES):
            intended.append(line)
        elif path.startswith((".kaggle_sota/", ".obsidian/", "checkpoints/", "scratch/")):
            generated_untracked.append(line)
        else:
            external.append(line)

    report = ["# Workspace Cleanup Manifest\n"]
    report.append("## Intended cleanup changes\n")
    report.extend(f"- `{line}`\n" for line in intended[:400])
    report.append("\n## Generated/untracked workspace clutter\n")
    report.extend(f"- `{line}`\n" for line in generated_untracked[:200])
    report.append("\n## Pre-existing or review-required changes\n")
    report.extend(f"- `{line}`\n" for line in external[:300])
    report.append("\n## Suggested commit boundary\n")
    report.append(
        "Commit intended cleanup files separately from modeling changes. Do not mix leaderboard experiments with repository hygiene.\n"
    )
    out = ROOT / "reports" / "workspace_cleanup_manifest.md"
    out.write_text("\n".join(report))
    print(f"Wrote {out.relative_to(ROOT)}")
    print(f"intended={len(intended)} generated_untracked={len(generated_untracked)} review_required={len(external)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
