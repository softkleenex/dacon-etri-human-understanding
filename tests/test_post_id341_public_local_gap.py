import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_public_local_gap.py"


def load_module():
    spec = importlib.util.spec_from_file_location("public_local_gap", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_public_local_gap_report_blocks_local_cv_only_promotion():
    module = load_module()
    report = module.build_report()

    assert report["decision"]["public_local_gap_diagnosis_ready"] is True
    assert report["decision"]["mac_m4_environment_primary_explanation"] is False
    assert report["decision"]["any_watch_candidate_passes_public_058_recent_floor_proxy"] is False
    assert report["decision"]["any_watch_candidate_passes_public_058_recent_stress_proxy"] is False
    assert report["decision"]["create_submission_artifact"] is False
    assert report["decision"]["submit_now"] is False
    assert report["decision"]["run_kaggle_now"] is False


def test_recent_gap_thresholds_include_new_scored_micro_probes():
    module = load_module()
    report = module.build_report()
    refs = report["stress_references"]

    assert refs["reference_gaps"]["id178_gap"] == 0.062773301
    assert refs["reference_gaps"]["id329_gap"] == 0.085072858
    assert refs["reference_gaps"]["id337_gap"] == 0.062554629
    assert refs["reference_gaps"]["id340_gap"] == 0.062953175
    assert refs["recent_gap_floor"] == refs["reference_gaps"]["id337_gap"]
    assert refs["recent_gap_stress"] == refs["reference_gaps"]["id329_gap"]
    assert refs["local_cv_needed_for_public_058_recent_floor"] == 0.517445
    assert refs["local_cv_needed_for_public_058_recent_stress"] == 0.494927


def test_watch_candidates_all_fail_public_058_recent_proxy():
    module = load_module()
    report = module.build_report()
    watch = report["watch_candidates"]

    assert {row["id"] for row in watch} >= {334, 335, 336, 338, 339}
    assert all(not row["passes_public_058_floor_proxy"] for row in watch)
    assert all(not row["passes_public_058_stress_proxy"] for row in watch)
    assert min(row["recent_floor_proxy"] for row in watch) > 0.58


def test_markdown_stop_rule_mentions_no_id342_artifact(tmp_path):
    module = load_module()
    report = module.build_report()
    output = tmp_path / "gap.md"

    module.write_markdown(output, json.loads(json.dumps(report)))
    text = output.read_text(encoding="utf-8")

    assert "Create submission artifact: `False`" in text
    assert "Public 0.58 local CV target under recent floor: `0.517445`" in text
    assert "does not authorize a DACON submission or a new local ID342 artifact" in text
