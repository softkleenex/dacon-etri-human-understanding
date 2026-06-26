#!/usr/bin/env python3
"""Phase-3 ambience-context closure audit after ID341.

This report-only diagnostic does not train models. It reads the historical
semantic ambience-context diagnostics and decides whether the raw `mAmbience`
family should be reopened after the post-ID341 sleep/residual/usage closures.

It creates no submission CSV, no base prediction artifact, no notebook/history
artifact, no submission-log row, no Kaggle push/run, and no DACON submission.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))

AMBIENCE_REPORT_PATTERNS = [
    "strict_q2_ambience_context_diagnostic_id*.json",
    "strict_q3_ambience_context_diagnostic_id*.json",
    "strict_s4_ambience_context_diagnostic_id*.json",
    "strict_s4_ambience_context_confirm_diagnostic_id*.json",
]
POST_ID341_CONTEXT_REPORTS = {
    "phase2_block_a_q1_proxy_gap": "phase2_block_a_q1_proxy_gap_post_id341_20260618.json",
    "phase2_block_b_s2_sleep_recovery_retry": "phase2_block_b_s2_sleep_recovery_retry_post_id341_20260618.json",
    "phase3_usage_temporal_microstructure": "phase3_usage_temporal_microstructure_smoke_post_id341_20260618.json",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ambience_report_paths() -> list[Path]:
    paths: set[Path] = set()
    for pattern in AMBIENCE_REPORT_PATTERNS:
        paths.update(REPORTS_DIR.glob(pattern))
    return sorted(paths)


def summarize_ambience_report(path: Path) -> dict[str, Any]:
    report = load_json(path)
    target = str(report.get("target"))
    base_id = int(report.get("base_id"))
    target_scores = report.get("target_scores", {})
    selector = report.get("selector_info", {})
    full_choices = selector.get("full_train_test_choices", {}).get(target, {})
    candidate_losses = full_choices.get("candidate_losses", {})
    base_key = f"id{base_id}_base"
    base_loss = candidate_losses.get(base_key, full_choices.get("full_oof_log_loss"))
    best_full_candidate = None
    best_full_loss = None
    if candidate_losses:
        best_full_candidate, best_full_loss = min(candidate_losses.items(), key=lambda item: float(item[1]))
    final_nested_target_loss = target_scores.get(target)
    full_oof_delta_vs_base = (
        float(best_full_loss) - float(base_loss) if best_full_loss is not None and base_loss is not None else None
    )
    nested_delta_vs_base = (
        float(final_nested_target_loss) - float(base_loss)
        if final_nested_target_loss is not None and base_loss is not None
        else None
    )
    nested_choices = selector.get("nested_choices", {}).get(target, [])
    nested_base_win_count = 0
    nested_nonbase_choice_count = 0
    nested_fold_count = len(nested_choices) if isinstance(nested_choices, list) else 0
    if isinstance(nested_choices, list):
        for row in nested_choices:
            candidate_losses_fold = row.get("candidate_losses", {})
            if not candidate_losses_fold:
                continue
            best_fold_candidate, _best_fold_loss = min(candidate_losses_fold.items(), key=lambda item: float(item[1]))
            if best_fold_candidate == base_key:
                nested_base_win_count += 1
            else:
                nested_nonbase_choice_count += 1
    ambience_report = selector.get("ambience_report", {})
    return {
        "path": str(path.relative_to(ROOT)),
        "target": target,
        "base_id": base_id,
        "candidate_pool_size": selector.get("candidate_pool_size"),
        "base_target_loss": float(base_loss) if base_loss is not None else None,
        "best_full_oof_candidate": best_full_candidate,
        "best_full_oof_loss": float(best_full_loss) if best_full_loss is not None else None,
        "full_oof_delta_vs_base": full_oof_delta_vs_base,
        "final_nested_target_loss": float(final_nested_target_loss) if final_nested_target_loss is not None else None,
        "nested_delta_vs_base": nested_delta_vs_base,
        "nested_improves_base": bool(nested_delta_vs_base is not None and nested_delta_vs_base < 0),
        "nested_fold_count": nested_fold_count,
        "nested_base_win_count": nested_base_win_count,
        "nested_nonbase_choice_count": nested_nonbase_choice_count,
        "ambience_feature_shape": ambience_report.get("feature_shape"),
        "ambience_feature_columns": ambience_report.get("n_feature_columns"),
        "contamination_note": selector.get("contamination_note"),
    }


def context_status() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, name in POST_ID341_CONTEXT_REPORTS.items():
        path = REPORTS_DIR / name
        if not path.exists():
            out[key] = {"path": str(path.relative_to(ROOT)), "exists": False, "decision": None}
            continue
        report = load_json(path)
        out[key] = {
            "path": str(path.relative_to(ROOT)),
            "exists": True,
            "decision": {
                "selected_targets": report.get("decision", {}).get("selected_targets"),
                "final_block_a_gate_pass": report.get("decision", {}).get("final_block_a_gate_pass"),
                "final_block_b_gate_pass": report.get("decision", {}).get("final_block_b_gate_pass"),
                "final_usage_temporal_gate_pass": report.get("decision", {}).get("final_usage_temporal_gate_pass"),
                "public_058_reviewable": report.get("decision", {}).get("public_058_reviewable"),
                "projected_public_proxy_from_id341": report.get("decision", {}).get(
                    "projected_public_proxy_from_id341"
                ),
            },
        }
    return out


def build_report() -> dict[str, Any]:
    rows = [summarize_ambience_report(path) for path in ambience_report_paths()]
    rows = sorted(
        rows,
        key=lambda row: (
            row["target"],
            row["base_id"],
            row["path"],
        ),
    )
    full_oof_positive_rows = [
        row for row in rows if row["full_oof_delta_vs_base"] is not None and row["full_oof_delta_vs_base"] < 0
    ]
    nested_positive_rows = [row for row in rows if row["nested_improves_base"]]
    s4_full_oof_trap_rows = [
        row
        for row in rows
        if row["target"] == "S4"
        and row["full_oof_delta_vs_base"] is not None
        and row["full_oof_delta_vs_base"] < 0
        and not row["nested_improves_base"]
    ]
    best_full_row = (
        min(full_oof_positive_rows, key=lambda row: row["full_oof_delta_vs_base"]) if full_oof_positive_rows else None
    )
    worst_nested_collapse = (
        max(
            [row for row in rows if row["nested_delta_vs_base"] is not None],
            key=lambda row: row["nested_delta_vs_base"],
        )
        if rows
        else None
    )
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "phase3_ambience_context_closure_post_id341",
        "purpose": (
            "Read existing semantic ambience-context diagnostics and decide whether mAmbience should be reopened "
            "after post-ID341 raw-source closures."
        ),
        "post_id341_context": context_status(),
        "ambience_report_count": len(rows),
        "ambience_rows": rows,
        "aggregate": {
            "full_oof_positive_count": len(full_oof_positive_rows),
            "nested_positive_count": len(nested_positive_rows),
            "s4_full_oof_trap_count": len(s4_full_oof_trap_rows),
            "best_full_oof_row": best_full_row,
            "worst_nested_collapse_row": worst_nested_collapse,
        },
        "decision": {
            "create_submission_artifact": False,
            "submit_now": False,
            "push_kaggle_now": False,
            "run_kaggle_now": False,
            "ambience_reopen_now": False,
            "ambience_has_historical_full_oof_signal": bool(full_oof_positive_rows),
            "ambience_has_nested_positive_signal": bool(nested_positive_rows),
            "ambience_s4_full_oof_is_transfer_trap": bool(s4_full_oof_trap_rows),
            "public_058_reviewable": False,
            "next_action": (
                "Do not create ID342 from semantic ambience context. Reopen mAmbience only if a future upstream "
                "model-level rebuild can nest source generation before selector choice; do not replay broad/full-OOF "
                "S4 ambience overlays."
            ),
        },
        "no_model_trained": True,
        "no_predictions_created": True,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def build_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    aggregate = report["aggregate"]
    lines = [
        "# Phase-3 ambience context closure — post ID341",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It reads existing ambience diagnostics only; no model is trained and no predictions or submission artifacts are created.",
        "",
        "## Aggregate read",
        "",
        f"- Ambience reports read: `{report['ambience_report_count']}`",
        f"- Full-OOF positive rows: `{aggregate['full_oof_positive_count']}`",
        f"- Nested positive rows: `{aggregate['nested_positive_count']}`",
        f"- S4 full-OOF trap rows: `{aggregate['s4_full_oof_trap_count']}`",
        "",
    ]
    if aggregate["best_full_oof_row"]:
        row = aggregate["best_full_oof_row"]
        lines.extend(
            [
                "Best full-OOF ambience row:",
                "",
                f"- `{row['path']}` `{row['target']}` base ID `{row['base_id']}`",
                f"- Full-OOF Δ vs base: `{row['full_oof_delta_vs_base']:+.6f}`",
                f"- Nested Δ vs base: `{row['nested_delta_vs_base']:+.6f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Rows",
            "",
            "| report | target | base | full-OOF Δ | nested Δ | nested improves | best full candidate |",
            "|---|---|---:|---:|---:|---|---|",
        ]
    )
    for row in report["ambience_rows"]:
        full_delta = row["full_oof_delta_vs_base"]
        nested_delta = row["nested_delta_vs_base"]
        lines.append(
            f"| `{row['path']}` | `{row['target']}` | {row['base_id']} | "
            f"{full_delta:+.6f} | {nested_delta:+.6f} | `{row['nested_improves_base']}` | "
            f"`{row['best_full_oof_candidate']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Ambience reopen now: `{decision['ambience_reopen_now']}`",
            f"- Historical full-OOF signal exists: `{decision['ambience_has_historical_full_oof_signal']}`",
            f"- Nested positive signal exists: `{decision['ambience_has_nested_positive_signal']}`",
            f"- S4 full-OOF is transfer trap: `{decision['ambience_s4_full_oof_is_transfer_trap']}`",
            f"- Public 0.58 reviewable: `{decision['public_058_reviewable']}`",
            f"- Create submission artifact: `{decision['create_submission_artifact']}`",
            f"- Submit now: `{decision['submit_now']}`",
            f"- Push Kaggle now: `{decision['push_kaggle_now']}`",
            f"- Run Kaggle now: `{decision['run_kaggle_now']}`",
            f"- Next action: {decision['next_action']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "phase3_ambience_context_closure_post_id341_20260618.json",
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
