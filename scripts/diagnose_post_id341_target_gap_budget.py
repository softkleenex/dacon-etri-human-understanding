#!/usr/bin/env python3
"""Post-ID341 target/source gap budget.

Report-only diagnostic. This reads existing post-ID341 no-submit source reports
and estimates how much target-level validation signal is still missing before a
candidate could plausibly beat the ID178 public anchor or open a Public-0.58
artifact review. It creates no submission CSV, base prediction artifact,
notebook/history artifact, submission-log row, Kaggle run, or DACON submission.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
KST = timezone(timedelta(hours=9))

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
ID341_PROXY = 0.6038297835
ID178_PUBLIC = 0.6014113009
PUBLIC_058_REVIEW_CUTOFF = 0.590

SOURCE_REPORTS = {
    "block_a_sleep_residual": REPORTS_DIR / "phase2_block_a_q1_s4_sleep_residual_bridge_post_id341_20260618.json",
    "sleep_evening_combined_sources": REPORTS_DIR / "combined_new_source_gate_post_id341_20260618.json",
    "sleep_evening_external_sources": REPORTS_DIR
    / "combined_sleep_evening_external_prior_gate_post_id341_20260618.json",
    "evening_load_recovery": REPORTS_DIR / "evening_load_recovery_smoke_post_id341_20260618.json",
    "block_b_s2_retry": REPORTS_DIR / "phase2_block_b_s2_sleep_recovery_retry_post_id341_20260618.json",
    "usage_temporal_microstructure": REPORTS_DIR
    / "phase3_usage_temporal_microstructure_smoke_post_id341_20260618.json",
    "routine_stability_place_entropy": REPORTS_DIR / "routine_stability_place_entropy_smoke_post_id341_20260618.json",
    "place_social_novelty": REPORTS_DIR / "place_social_novelty_smoke_fixedlist_post_id341_20260618.json",
    "sensor_adherence_dropout": REPORTS_DIR / "sensor_adherence_dropout_smoke_post_id341_20260618.json",
    "clean_backbone_closure": REPORTS_DIR / "post_id341_clean_backbone_lane_closure_20260618.json",
    "local_source_closure": REPORTS_DIR / "post_id341_local_source_closure_synthesis_20260618.json",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def walk_selected_configs(value: Any, source_name: str, branch: str = "selected_configs") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if "target" in value and "mean_delta_vs_id341" in value:
            target = str(value.get("target"))
            if target in TARGETS:
                mean_delta = safe_float(value.get("mean_delta_vs_id341"))
                max_delta = safe_float(value.get("max_delta_vs_id341"))
                wins = value.get("primary_wins_vs_id341")
                try:
                    wins_int = int(wins)
                except (TypeError, ValueError):
                    wins_int = None
                rows.append(
                    {
                        "source": source_name,
                        "branch": branch,
                        "target": target,
                        "candidate": value.get("candidate"),
                        "model": value.get("model"),
                        "anchor": value.get("anchor"),
                        "weight": value.get("weight"),
                        "primary_wins_vs_id341": wins_int,
                        "mean_delta_vs_id341": mean_delta,
                        "max_delta_vs_id341": max_delta,
                        "min_delta_vs_id341": safe_float(value.get("min_delta_vs_id341")),
                        "split_deltas_vs_id341": value.get("split_deltas_vs_id341"),
                    }
                )
        for key, item in value.items():
            child_branch = f"{branch}.{key}" if branch else str(key)
            rows.extend(walk_selected_configs(item, source_name, child_branch))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            rows.extend(walk_selected_configs(item, source_name, f"{branch}[{idx}]"))
    return rows


def collect_source_rows(reports: dict[str, Path] = SOURCE_REPORTS) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    report_summaries: list[dict[str, Any]] = []
    for source_name, path in reports.items():
        if not path.exists():
            report_summaries.append({"source": source_name, "path": str(path.relative_to(ROOT)), "exists": False})
            continue
        report = load_json(path)
        decision = report.get("decision", {})
        selected = report.get("selected_configs", {})
        source_rows = walk_selected_configs(selected, source_name)
        rows.extend(source_rows)
        report_summaries.append(
            {
                "source": source_name,
                "path": str(path.relative_to(ROOT)),
                "exists": True,
                "diagnostic": report.get("diagnostic"),
                "selected_config_count": len(source_rows),
                "selected_targets": sorted({row["target"] for row in source_rows}),
                "decision": {
                    key: decision.get(key)
                    for key in [
                        "selected_target_count",
                        "selected_targets",
                        "reviewable_low_movement_candidate_exists",
                        "proxy_below_id178_candidate_exists",
                        "public_058_reviewable",
                        "public_058_plausible",
                        "best_proxy",
                        "projected_public_proxy_from_id341",
                        "close_clean_backbone_lane_now",
                        "local_source_stack_closed_for_submission_artifact",
                        "any_local_lane_promotable",
                    ]
                    if key in decision
                },
            }
        )
    return rows, report_summaries


def best_by_target(
    rows: list[dict[str, Any]], min_wins: int, require_nonpositive_max: bool
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        mean_delta = safe_float(row.get("mean_delta_vs_id341"))
        max_delta = safe_float(row.get("max_delta_vs_id341"))
        wins = row.get("primary_wins_vs_id341")
        if mean_delta is None or mean_delta >= 0:
            continue
        if wins is None or int(wins) < min_wins:
            continue
        if require_nonpositive_max and (max_delta is None or max_delta > 0):
            continue
        grouped[row["target"]].append(row)
    out: dict[str, dict[str, Any]] = {}
    for target, target_rows in grouped.items():
        out[target] = sorted(
            target_rows,
            key=lambda item: (
                safe_float(item.get("mean_delta_vs_id341")) or 999.0,
                safe_float(item.get("max_delta_vs_id341")) or 999.0,
                str(item.get("source")),
            ),
        )[0]
    return out


def projection_from_target_rows(selected_by_target: dict[str, dict[str, Any]]) -> dict[str, Any]:
    target_deltas = {
        target: safe_float(row.get("mean_delta_vs_id341")) or 0.0 for target, row in selected_by_target.items()
    }
    sum_delta = float(sum(target_deltas.values()))
    mean_delta = float(sum_delta / len(TARGETS))
    projected_proxy = float(ID341_PROXY + mean_delta)
    return {
        "selected_target_count": len(selected_by_target),
        "selected_targets": sorted(selected_by_target),
        "target_deltas": target_deltas,
        "sum_target_delta": sum_delta,
        "mean_delta_over_all_targets": mean_delta,
        "projected_proxy_from_id341": projected_proxy,
        "gap_vs_id178_public": float(projected_proxy - ID178_PUBLIC),
        "gap_vs_public_058_cutoff": float(projected_proxy - PUBLIC_058_REVIEW_CUTOFF),
        "additional_sum_delta_needed_to_beat_id178": float((ID178_PUBLIC - ID341_PROXY) * len(TARGETS) - sum_delta),
        "additional_sum_delta_needed_for_public_058": float(
            (PUBLIC_058_REVIEW_CUTOFF - ID341_PROXY) * len(TARGETS) - sum_delta
        ),
        "beats_id178_public_anchor_proxy": projected_proxy < ID178_PUBLIC,
        "public_058_reviewable_by_budget": projected_proxy <= PUBLIC_058_REVIEW_CUTOFF,
    }


def build_report() -> dict[str, Any]:
    rows, report_summaries = collect_source_rows()
    strict = best_by_target(rows, min_wins=5, require_nonpositive_max=True)
    watch = best_by_target(rows, min_wins=4, require_nonpositive_max=False)
    strict_projection = projection_from_target_rows(strict)
    watch_projection = projection_from_target_rows(watch)
    missing_strict_targets = [target for target in TARGETS if target not in strict]
    missing_watch_targets = [target for target in TARGETS if target not in watch]
    ranked_rows = sorted(
        rows,
        key=lambda row: (
            row["target"],
            -(row.get("primary_wins_vs_id341") or 0),
            safe_float(row.get("mean_delta_vs_id341")) or 999.0,
            safe_float(row.get("max_delta_vs_id341")) or 999.0,
        ),
    )
    decision = {
        "create_submission_artifact": False,
        "submit_now": False,
        "push_kaggle_now": False,
        "run_kaggle_now": False,
        "strict_budget_beats_id178": strict_projection["beats_id178_public_anchor_proxy"],
        "watch_budget_beats_id178": watch_projection["beats_id178_public_anchor_proxy"],
        "strict_budget_public_058_reviewable": strict_projection["public_058_reviewable_by_budget"],
        "watch_budget_public_058_reviewable": watch_projection["public_058_reviewable_by_budget"],
        "q1_s4_only_is_insufficient": bool(
            set(strict_projection["selected_targets"]).issubset({"Q1", "S4"})
            and not strict_projection["beats_id178_public_anchor_proxy"]
        ),
        "needs_new_independent_targets": missing_strict_targets,
        "recommended_next_focus": "Q2/Q3/S3 independent raw or upstream source signal before more Q1/S4 tuning",
        "next_action": (
            "Do not create ID342 from current local source leads. The strict all-primary budget remains above ID178; "
            "future work should seek independent Q2/Q3/S3 signal or run the explicit-approved no-submit Kaggle upstream rebuild."
        ),
    }
    return {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S UTC+09:00"),
        "diagnostic": "post_id341_target_gap_budget",
        "purpose": (
            "Quantify whether current post-ID341 local source leads contain enough target-level signal to beat "
            "ID178/Public-0.58 before any new artifact is created."
        ),
        "constants": {
            "targets": TARGETS,
            "id341_proxy": ID341_PROXY,
            "id178_public": ID178_PUBLIC,
            "public_058_review_cutoff": PUBLIC_058_REVIEW_CUTOFF,
            "required_mean_delta_to_beat_id178": float(ID178_PUBLIC - ID341_PROXY),
            "required_sum_target_delta_to_beat_id178": float((ID178_PUBLIC - ID341_PROXY) * len(TARGETS)),
            "required_mean_delta_for_public_058": float(PUBLIC_058_REVIEW_CUTOFF - ID341_PROXY),
            "required_sum_target_delta_for_public_058": float((PUBLIC_058_REVIEW_CUTOFF - ID341_PROXY) * len(TARGETS)),
        },
        "source_report_summaries": report_summaries,
        "signal_rows_considered": len(rows),
        "strict_all_primary_best_by_target": strict,
        "watch_four_of_five_best_by_target": watch,
        "strict_budget_projection": strict_projection,
        "watch_budget_projection": watch_projection,
        "missing_strict_targets": missing_strict_targets,
        "missing_watch_targets": missing_watch_targets,
        "ranked_signal_rows": ranked_rows[:80],
        "decision": decision,
        "no_submission_created": True,
        "no_base_prediction_artifact_created": True,
        "no_notebook_artifact_created": True,
        "no_submission_log_row_created": True,
        "no_kaggle_push_or_run_made": True,
        "no_dacon_submission_made": True,
    }


def fmt(value: Any, digits: int = 6) -> str:
    out = safe_float(value)
    return "" if out is None else f"{out:.{digits}f}"


def table_for_best(rows_by_target: dict[str, dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for target in TARGETS:
        row = rows_by_target.get(target)
        if row is None:
            lines.append(f"| {target} |  |  |  |  |  |  |  |")
            continue
        lines.append(
            f"| {target} | `{row.get('source')}` | `{row.get('candidate')}` | "
            f"{row.get('primary_wins_vs_id341')} | {fmt(row.get('mean_delta_vs_id341'))} | "
            f"{fmt(row.get('max_delta_vs_id341'))} | `{row.get('anchor')}` | {row.get('weight')} |"
        )
    return lines


def build_markdown(report: dict[str, Any]) -> str:
    strict = report["strict_budget_projection"]
    watch = report["watch_budget_projection"]
    decision = report["decision"]
    lines = [
        "# Post-ID341 target gap budget",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.",
        "",
        "## Gap math",
        "",
        f"- ID341 proxy: `{report['constants']['id341_proxy']:.6f}`",
        f"- ID178 public anchor: `{report['constants']['id178_public']:.6f}`",
        f"- Public-0.58 review cutoff used by gates: `{report['constants']['public_058_review_cutoff']:.6f}`",
        f"- Required sum target delta to beat ID178: `{report['constants']['required_sum_target_delta_to_beat_id178']:+.6f}`",
        f"- Required sum target delta for Public-0.58 review: `{report['constants']['required_sum_target_delta_for_public_058']:+.6f}`",
        "",
        "## Strict all-primary budget",
        "",
        f"- Selected targets: `{', '.join(strict['selected_targets'])}`",
        f"- Sum target delta: `{strict['sum_target_delta']:+.6f}`",
        f"- Projected proxy: `{strict['projected_proxy_from_id341']:.6f}`",
        f"- Gap vs ID178: `{strict['gap_vs_id178_public']:+.6f}`",
        f"- Additional sum delta needed to beat ID178: `{strict['additional_sum_delta_needed_to_beat_id178']:+.6f}`",
        "",
        "| target | source | candidate | wins | mean Δ341 | max Δ341 | anchor | weight |",
        "|---|---|---|---:|---:|---:|---|---:|",
        *table_for_best(report["strict_all_primary_best_by_target"]),
        "",
        "## Watch 4/5 budget",
        "",
        f"- Selected targets: `{', '.join(watch['selected_targets'])}`",
        f"- Sum target delta: `{watch['sum_target_delta']:+.6f}`",
        f"- Projected proxy: `{watch['projected_proxy_from_id341']:.6f}`",
        f"- Gap vs ID178: `{watch['gap_vs_id178_public']:+.6f}`",
        f"- Additional sum delta needed to beat ID178: `{watch['additional_sum_delta_needed_to_beat_id178']:+.6f}`",
        "",
        "| target | source | candidate | wins | mean Δ341 | max Δ341 | anchor | weight |",
        "|---|---|---|---:|---:|---:|---|---:|",
        *table_for_best(report["watch_four_of_five_best_by_target"]),
        "",
        "## Decision",
        "",
        f"- Strict budget beats ID178: `{decision['strict_budget_beats_id178']}`",
        f"- Watch budget beats ID178: `{decision['watch_budget_beats_id178']}`",
        f"- Strict budget Public-0.58 reviewable: `{decision['strict_budget_public_058_reviewable']}`",
        f"- Watch budget Public-0.58 reviewable: `{decision['watch_budget_public_058_reviewable']}`",
        f"- Missing strict targets: `{', '.join(report['missing_strict_targets'])}`",
        f"- Recommended next focus: `{decision['recommended_next_focus']}`",
        "",
        decision["next_action"],
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "post_id341_target_gap_budget_20260618.json")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
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
