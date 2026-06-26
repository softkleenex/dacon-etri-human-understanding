#!/usr/bin/env python3
"""Summarize report-only Kaggle sequence-source outputs after post-ID335 runner pulls."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "build" / "kaggle_outputs" / "sequence_source_post_id335"
DEFAULT_REPORT_JSON = ROOT / "reports" / "kaggle_sequence_source_remote_summary_post_id335_20260617.json"
DEFAULT_REPORT_MD = ROOT / "reports" / "kaggle_sequence_source_remote_summary_post_id335_20260617.md"


@dataclass(frozen=True)
class DiagnosticRow:
    path: str
    base_id: int
    n_kernels: int
    seed: int
    base_mean: float
    targetwise_best_mean: float
    improvement: float
    best_by_target: dict[str, str]
    improvement_by_target: dict[str, float]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_sequence_diagnostic(data: dict[str, Any]) -> bool:
    return data.get("diagnostic") == "public_random_conv_sequence" and str(data.get("validation_lens", "")).startswith(
        "public_structure"
    )


def _diagnostic_row(path: Path, data: dict[str, Any]) -> DiagnosticRow:
    targets = data.get("targets", {})
    best_by_target = {target: str(info.get("best")) for target, info in sorted(targets.items())}
    improvement_by_target = {
        target: float(info.get("improvement_vs_base", 0.0)) for target, info in sorted(targets.items())
    }
    base_mean = float(data["base_mean"])
    targetwise_best_mean = float(data["targetwise_best_mean"])
    return DiagnosticRow(
        path=str(path),
        base_id=int(data["base_id"]),
        n_kernels=int(data["n_kernels"]),
        seed=int(data["seed"]),
        base_mean=base_mean,
        targetwise_best_mean=targetwise_best_mean,
        improvement=base_mean - targetwise_best_mean,
        best_by_target=best_by_target,
        improvement_by_target=improvement_by_target,
    )


def collect_diagnostics(input_dir: Path) -> tuple[list[DiagnosticRow], list[dict[str, Any]]]:
    rows: list[DiagnosticRow] = []
    manifests: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("**/*.json")):
        data = _load_json(path)
        if _is_sequence_diagnostic(data):
            rows.append(_diagnostic_row(path, data))
        elif "sweep_jobs" in data and "copied_artifacts" in data:
            manifests.append({"path": str(path), **data})
    rows.sort(key=lambda row: (row.targetwise_best_mean, -row.improvement, row.n_kernels, row.seed))
    return rows, manifests


def target_summary(rows: list[DiagnosticRow]) -> dict[str, Any]:
    by_target: dict[str, list[tuple[float, DiagnosticRow, str]]] = {}
    for row in rows:
        for target, improvement in row.improvement_by_target.items():
            by_target.setdefault(target, []).append((improvement, row, row.best_by_target[target]))
    summary: dict[str, Any] = {}
    for target, values in sorted(by_target.items()):
        values.sort(key=lambda item: (-item[0], item[1].targetwise_best_mean, item[1].seed))
        improvement, row, best = values[0]
        summary[target] = {
            "best_improvement_vs_base": improvement,
            "best_model": best,
            "source_file": row.path,
            "n_kernels": row.n_kernels,
            "seed": row.seed,
        }
    return summary


def write_reports(
    rows: list[DiagnosticRow],
    manifests: list[dict[str, Any]],
    output_json: Path,
    output_md: Path,
    input_dir: Path,
    report_name: str,
    title: str,
) -> dict[str, Any]:
    payload = {
        "report": report_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "n_diagnostics": len(rows),
        "n_manifests": len(manifests),
        "best_overall": rows[0].__dict__ if rows else None,
        "rows": [row.__dict__ for row in rows],
        "target_summary": target_summary(rows),
        "manifests": manifests,
        "promotion_policy": (
            "Report-only summary. This does not authorize a DACON submission or artifact promotion. "
            "Promotion requires a separate decision note and unique CSV/notebook/log/base-prediction contract."
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# {title}",
        "",
        f"Input dir: `{input_dir}`",
        f"Diagnostics found: `{len(rows)}`",
        f"Manifest files found: `{len(manifests)}`",
        "",
    ]
    if rows:
        best = rows[0]
        lines.extend(
            [
                "## Best overall diagnostic",
                "",
                f"- File: `{best.path}`",
                f"- Base ID: `{best.base_id}`",
                f"- Kernels/seed: `{best.n_kernels}` / `{best.seed}`",
                f"- Base mean: `{best.base_mean:.6f}`",
                f"- Targetwise best mean: `{best.targetwise_best_mean:.6f}`",
                f"- Improvement vs base: `{best.improvement:.6f}`",
                "",
                "## Ranked diagnostics",
                "",
                "| rank | kernels | seed | targetwise best | improvement | file |",
                "|---:|---:|---:|---:|---:|---|",
            ]
        )
        for rank, row in enumerate(rows, start=1):
            lines.append(
                f"| {rank} | {row.n_kernels} | {row.seed} | {row.targetwise_best_mean:.6f} | "
                f"{row.improvement:.6f} | `{Path(row.path).name}` |"
            )
        lines.extend(
            [
                "",
                "## Target-level best improvements",
                "",
                "| target | improvement | model | kernels | seed |",
                "|---|---:|---|---:|---:|",
            ]
        )
        for target, info in payload["target_summary"].items():
            lines.append(
                f"| {target} | {info['best_improvement_vs_base']:.6f} | `{info['best_model']}` | "
                f"{info['n_kernels']} | {info['seed']} |"
            )
    else:
        lines.extend(
            [
                "## No diagnostics found",
                "",
                "No `public_random_conv_sequence` diagnostic JSON files were found. Check the Kaggle kernel logs before retrying.",
            ]
        )
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "This is a report-only summary. It creates no DACON submission and does not promote any artifact. Any promotion must be a separate, uniquely named, fully matched CSV/notebook/log/base-prediction decision.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--report-name", default="kaggle_sequence_source_remote_summary_post_id335")
    parser.add_argument("--title", default="Kaggle sequence-source remote summary post-ID335")
    args = parser.parse_args()
    rows, manifests = collect_diagnostics(args.input_dir)
    payload = write_reports(
        rows,
        manifests,
        args.output_json,
        args.output_md,
        args.input_dir,
        report_name=args.report_name,
        title=args.title,
    )
    print(
        json.dumps(
            {"n_diagnostics": payload["n_diagnostics"], "best_overall": payload["best_overall"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
