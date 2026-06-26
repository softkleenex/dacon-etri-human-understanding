import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_post_id341_kaggle_runner_intake_contract.py"
LOCAL_SMOKE_SCRIPT = ROOT / "scripts" / "run_post_id341_kaggle_runner_local_smoke.py"


def load_module():
    spec = importlib.util.spec_from_file_location("runner_intake_contract", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_runner_contract_passes_and_is_report_only():
    module = load_module()
    report = module.build_report(module.DEFAULT_NOTEBOOK, module.DEFAULT_METADATA)
    safety = report["notebook_report_only_safety_contract"]
    notebook_source = module.DEFAULT_NOTEBOOK.read_text(encoding="utf-8")

    assert report["decision"]["runner_intake_contract_pass"] is True
    assert (
        "remote_post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop_20260619.json"
        in report["intake_expected_reports"]
    )
    assert (
        "remote_post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop_20260619.json"
        in report["runner_generated_reports"]
    )
    assert (
        "remote_post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop_20260619.json"
        in report["runner_staged_reports"]
    )
    assert (
        "remote_post_id341_self_supervised_subject_state_embedding_design_20260619.json"
        in report["intake_expected_reports"]
    )
    assert (
        "remote_post_id341_self_supervised_subject_state_embedding_design_20260619.json"
        in report["runner_generated_reports"]
    )
    assert (
        "remote_post_id341_self_supervised_subject_state_embedding_design_20260619.json"
        in report["runner_staged_reports"]
    )
    assert safety["no_forbidden_actions"] is True
    assert safety["required_safety_phrases_present"] is True
    assert safety["notebook_report_only_contract_pass"] is True
    assert report["q3s3_negative_map_handoff_contract"]["q3s3_negative_map_handoff_pass"] is True
    assert report["q3s3_negative_map_handoff_contract"]["remaining_inventory_uses_negative_map_arg"] is True
    assert report["q3s3_negative_map_handoff_contract"]["negative_map_staged_in_report_roots"] is True
    assert "KAGGLE_INPUT_DIR" in notebook_source
    assert "Synthesized runtime metadata" in notebook_source
    assert "Kaggle consumes dataset-metadata.json as dataset metadata" in notebook_source
    local_smoke_source = LOCAL_SMOKE_SCRIPT.read_text(encoding="utf-8")
    assert "local_contract_staging_no_full_notebook_execution" in local_smoke_source
    assert "stage_contract_outputs" in local_smoke_source
    assert "It intentionally does not execute the full upstream notebook locally." in local_smoke_source
    assert report["decision"]["create_submission_artifact"] is False
    assert report["decision"]["submit_now"] is False
    assert report["decision"]["push_kaggle_now"] is False
    assert report["decision"]["run_kaggle_now"] is False


def test_report_only_safety_contract_blocks_submission_like_actions():
    module = load_module()
    safety = module.notebook_report_only_safety_contract(
        [
            (
                7,
                "\n".join(
                    [
                        'manifest = {"no_submission_created": True, "no_dacon_submission_made": True}',
                        'manifest["next_allowed_action"] = "Do not create or submit a CSV from this runner."',
                        "report_roots = []",
                        "for root_path in report_roots: pass",
                        'preds.to_csv(out_dir / "submission_bad.csv")',
                        '{"submit_now": True}',
                    ]
                ),
            )
        ]
    )

    assert safety["no_forbidden_actions"] is False
    assert safety["notebook_report_only_contract_pass"] is False
    assert safety["forbidden_matches"]["csv_write_call"]
    assert safety["forbidden_matches"]["submission_or_base_output_path"]
    assert safety["forbidden_matches"]["forbidden_true_decision_flag"]


def test_runner_contract_markdown_contains_safety_section():
    module = load_module()
    report = module.build_report(module.DEFAULT_NOTEBOOK, module.DEFAULT_METADATA)
    markdown = module.build_markdown(json.loads(json.dumps(report)))

    assert "Notebook report-only safety contract" in markdown
    assert "Q3/S3 negative-map handoff contract" in markdown
    assert "Handoff pass: `True`" in markdown
    assert "No forbidden actions: `True`" in markdown
    assert "Report-only contract pass: `True`" in markdown
