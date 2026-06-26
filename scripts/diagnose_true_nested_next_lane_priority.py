#!/usr/bin/env python3
"""Prioritize the next true-nested rebuild lane from existing diagnostics.

This is a report-only consolidation. It reads prior diagnostics, ranks rebuild
lanes by signal, dependency risk, and rebuildability, and creates no prediction
or submission artifacts.
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


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def split_best_value(report: dict[str, Any], field: str) -> float:
    values = [float(row[field]) for row in report.get("best_rows", {}).values()]
    return sum(values) / len(values) if values else float("nan")


def find_pair_direct_row(pair_report: dict[str, Any], scenario_name: str) -> dict[str, Any]:
    rows = [row for split_rows in pair_report["splits"].values() for row in split_rows]
    matching = [row for row in rows if row["scenario"]["name"] == scenario_name]
    if not matching:
        raise KeyError(scenario_name)
    return min(matching, key=lambda row: (row["overall"], row["robust_score"]))


def bridge_row(bridge_report: dict[str, Any], candidate_id: int) -> dict[str, Any]:
    for row in bridge_report["results"]:
        if int(row["id"]) == candidate_id:
            return row
    raise KeyError(candidate_id)


def recent_s1_row(s1_report: dict[str, Any], candidate_id: int) -> dict[str, Any]:
    for row in s1_report["ranked_candidates"]:
        if int(row["id"]) == candidate_id:
            return row
    raise KeyError(candidate_id)


def priority_score(*, signal: float, risk: float, rebuild_cost: float, public_anchor_penalty: float) -> float:
    """Heuristic ranking score. Higher is better; not a public-score predictor."""
    return float(signal - risk - rebuild_cost - public_anchor_penalty)


def build_lanes() -> dict[str, Any]:
    blueprint = load_json(REPORTS_DIR / "true_nested_rebuild_blueprint_post_id335_20260617.json")
    pair = load_json(REPORTS_DIR / "fold_internal_pair_blend_rebuild_post_id335_20260617.json")
    second_stage = load_json(REPORTS_DIR / "fold_internal_second_stage_calibration_post_id335_20260617.json")
    id238_safe = load_json(REPORTS_DIR / "id238_conservative_calibration_post_id335_20260617.json")
    bridge = load_json(REPORTS_DIR / "post_id249_targetwise_delta_capped_bridge_robustness_20260616.json")
    s1 = load_json(REPORTS_DIR / "recent_s1_probe_stress_review_20260618.json")
    b1_dependency = load_json(REPORTS_DIR / "b1_pair_blend_source_dependency_audit_20260618.json")

    direct_pair = find_pair_direct_row(pair, "direct_no_252_cap022")
    full_pair = find_pair_direct_row(pair, "full_upstream_cap030")
    bridge_247 = bridge_row(bridge, 247)
    bridge_248 = bridge_row(bridge, 248)
    bridge_249 = bridge_row(bridge, 249)
    id337 = recent_s1_row(s1, 337)
    id338 = recent_s1_row(s1, 338)

    lanes = [
        {
            "lane_id": "B1_pair_logit_direct_no_252_model_level_rebuild",
            "rank_group": "primary",
            "evidence": {
                "report": "reports/fold_internal_pair_blend_rebuild_post_id335_20260617.json",
                "dependency_audit": "reports/b1_pair_blend_source_dependency_audit_20260618.json",
                "best_scenario": direct_pair["scenario"],
                "fold_internal_overall": direct_pair["overall"],
                "fold_internal_robust": direct_pair["robust_score"],
                "worst_target": direct_pair["slice_summary"]["worst_target"],
                "source_ids": direct_pair["scenario"]["source_ids"],
                "direct_no_252_is_not_dependency_clean": b1_dependency["decision"][
                    "direct_no_252_is_not_dependency_clean"
                ],
                "hidden_extreme_ids": b1_dependency["hidden_extreme_ids"],
            },
            "signal": 0.115,
            "dependency_risk": 0.065,
            "rebuild_cost": 0.045,
            "public_anchor_penalty": 0.024,
            "why": (
                "Strong fold-internal pair/logit signal survives at the selector layer, but the dependency audit shows "
                "the source artifacts still trace into high/extreme full-OOF bridge/calibration chains."
            ),
            "next_step": (
                "Do not reuse source IDs 254/255/256/257/258/266 as clean sources. First rebuild the bridge/source "
                "layers below them inside outer folds, then replay target-wise pair/logit selection report-only."
            ),
            "blocked_by": ["B1 source artifacts are not dependency-clean; see b1_pair_blend_source_dependency_audit."],
            "no_submit": True,
        },
        {
            "lane_id": "A1_delta_capped_bridge_midrisk_rebuild",
            "rank_group": "primary",
            "evidence": {
                "report": "reports/post_id249_targetwise_delta_capped_bridge_robustness_20260616.json",
                "safe_reference_id": 247,
                "safe_reference_local": bridge_247["overall"],
                "safe_reference_robust": bridge_247["robustness_score"],
                "midrisk_reference_id": 248,
                "midrisk_reference_local": bridge_248["overall"],
                "midrisk_reference_robust": bridge_248["robustness_score"],
                "aggressive_reference_id": 249,
                "aggressive_reference_local": bridge_249["overall"],
                "aggressive_reference_robust": bridge_249["robustness_score"],
            },
            "signal": 0.082,
            "dependency_risk": 0.045,
            "rebuild_cost": 0.026,
            "public_anchor_penalty": 0.026,
            "why": (
                "Delta-capped bridge has a smooth risk ladder: ID247/248 are safer than ID249. "
                "A nested bridge-choice rebuild can test whether the ladder survives without using full-OOF target choices."
            ),
            "next_step": (
                "Replay capped bridge selection with caps around 0.15/0.20/0.30 inside outer folds, rejecting ID249-style "
                "aggressive movement unless it improves nested robust score and anchor geometry."
            ),
            "blocked_by": ["Needs exact bridge family reconstruction from ID235/244-249 scripts."],
            "no_submit": True,
        },
        {
            "lane_id": "S1_controlled_probe_holdout",
            "rank_group": "probe_only",
            "evidence": {
                "report": "reports/recent_s1_probe_stress_review_20260618.json",
                "id337_local": id337["overall"],
                "id337_robust": id337["robust_score"],
                "id337_proxy": id337["conservative_public_risk_proxy"],
                "id338_local": id338["overall"],
                "id338_robust": id338["robust_score"],
                "id338_proxy": id338["conservative_public_risk_proxy"],
            },
            "signal": 0.015,
            "dependency_risk": 0.010,
            "rebuild_cost": 0.006,
            "public_anchor_penalty": 0.008,
            "why": (
                "ID337 is the cleanest low-movement controlled probe, but the signal is tiny. "
                "This lane can spend public budget only with explicit approval; it will not repair validation."
            ),
            "next_step": "Keep ID337 as the exact safety-first probe candidate; do not submit automatically.",
            "blocked_by": ["Requires explicit user approval for exact CSV before DACON submission."],
            "no_submit": True,
        },
        {
            "lane_id": "C0_lower_layer_id238_safe_calibration",
            "rank_group": "deprioritized",
            "evidence": {
                "report": "reports/id238_conservative_calibration_post_id335_20260617.json",
                "avg_best_calibrated": split_best_value(id238_safe, "overall"),
                "avg_best_robust": split_best_value(id238_safe, "robust_score"),
            },
            "signal": 0.010,
            "dependency_risk": 0.020,
            "rebuild_cost": 0.020,
            "public_anchor_penalty": 0.018,
            "why": (
                "The lower ID238 layer has only tiny conservative temperature-shift salvage signal. "
                "This is useful for explaining why the ID242 chain is fragile, but not a high-upside next run."
            ),
            "next_step": "Do not prioritize unless the pair/bridge lanes fail or a stricter calibration baseline is needed.",
            "blocked_by": [],
            "no_submit": True,
        },
        {
            "lane_id": "C2_upper_second_stage_calibration_id242_chain",
            "rank_group": "blocked",
            "evidence": {
                "report": "reports/fold_internal_second_stage_calibration_post_id335_20260617.json",
                "avg_best_fold_internal": split_best_value(second_stage, "overall"),
                "avg_best_robust": split_best_value(second_stage, "robust_score"),
                "full_pair_with_252_overall": full_pair["overall"],
                "full_pair_with_252_robust": full_pair["robust_score"],
                "blueprint_sources": blueprint["source_ids"],
            },
            "signal": 0.160,
            "dependency_risk": 0.145,
            "rebuild_cost": 0.045,
            "public_anchor_penalty": 0.060,
            "why": (
                "The ID242/252 chain has enormous local signal, but it depends on extreme non-nested calibration. "
                "Lower-layer diagnostics show ID242-like calibration on ID238 collapses under fold-internal refitting."
            ),
            "next_step": (
                "Keep blocked. Do not rebuild upper ID252 descendants until a lower-layer model-level source replaces "
                "the q240/isotonic full-OOF shortcut."
            ),
            "blocked_by": [
                "ID242 lower-layer calibration is not validated.",
                "ID287/ID329 public failures warn against local-frontier calibration.",
            ],
            "no_submit": True,
        },
    ]
    for lane in lanes:
        lane["priority_score"] = priority_score(
            signal=lane["signal"],
            risk=lane["dependency_risk"],
            rebuild_cost=lane["rebuild_cost"],
            public_anchor_penalty=lane["public_anchor_penalty"],
        )
    lanes = sorted(lanes, key=lambda lane: (-lane["priority_score"], lane["lane_id"]))
    return {
        "blueprint": blueprint,
        "lanes": lanes,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# True nested rebuild next-lane priority",
        "",
        f"Created: `{report['created_at_kst']}`",
        "",
        "This is report-only. It consolidates existing diagnostics and creates no CSV, notebook, base prediction, or DACON submission.",
        "",
        "## Ranked lanes",
        "",
        "| rank | lane | group | priority | signal | risk | cost | anchor penalty | read |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for rank, lane in enumerate(report["ranked_lanes"], start=1):
        lines.append(
            f"| {rank} | `{lane['lane_id']}` | `{lane['rank_group']}` | {lane['priority_score']:+.3f} | "
            f"{lane['signal']:.3f} | {lane['dependency_risk']:.3f} | {lane['rebuild_cost']:.3f} | "
            f"{lane['public_anchor_penalty']:.3f} | {lane['why']} |"
        )
    lines.extend(
        [
            "",
            "## Next execution recommendation",
            "",
            f"Primary lane: `{report['decision']['primary_lane']}`.",
            "",
            report["decision"]["reason"],
            "",
            "## Lane details",
            "",
        ]
    )
    for lane in report["ranked_lanes"]:
        lines.extend(
            [
                f"### {lane['lane_id']}",
                "",
                f"- Group: `{lane['rank_group']}`",
                f"- Next step: {lane['next_step']}",
                f"- Blocked by: `{lane['blocked_by']}`",
                f"- Evidence report: `{lane['evidence'].get('report')}`",
                "",
            ]
        )
    return "\n".join(lines)


def run(output_json: Path) -> dict[str, Any]:
    built = build_lanes()
    lanes = built["lanes"]
    primary = next(lane for lane in lanes if lane["rank_group"] == "primary")
    report = {
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "diagnostic": "true_nested_next_lane_priority",
        "source_blueprint": "reports/true_nested_rebuild_blueprint_post_id335_20260617.json",
        "ranked_lanes": lanes,
        "decision": {
            "primary_lane": primary["lane_id"],
            "create_submission_artifact": False,
            "dacon_submission_made": False,
            "public_058_plausible": False,
            "reason": (
                "The immediate execution lane is the mid-risk delta-capped bridge rebuild. The pair/logit selector "
                "signal remains the downstream map, but the B1 dependency audit shows its current sources still trace "
                "into high/extreme full-OOF bridge/calibration chains. Rebuild the bridge/source layer first, then "
                "replay pair/logit selection. Keep ID337 as a separate explicit-approval public probe candidate, not "
                "as validation repair."
            ),
        },
        "no_submission_created": True,
        "no_dacon_submission_made": True,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_json.with_suffix(".md").write_text(build_markdown(report) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORTS_DIR / "true_nested_next_lane_priority_post_id338_20260618.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.output_json)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_json.with_suffix('.md')}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
