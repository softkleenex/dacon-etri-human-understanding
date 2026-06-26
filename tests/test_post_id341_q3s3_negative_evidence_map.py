import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_post_id341_q3s3_negative_evidence_map.py"


def load_module():
    spec = importlib.util.spec_from_file_location("q3s3_negative_evidence_map", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_q3s3_negative_evidence_map_closes_known_local_lanes():
    module = load_module()
    report = module.build_report()
    decision = report["decision"]

    assert decision["q3s3_negative_evidence_map_passed"] is True
    assert decision["target_focus"] == ["Q3", "S3"]
    assert decision["closed_local_lane_count"] == 8
    assert decision["closed_optional_scout_count"] == 1
    assert decision["all_local_q3s3_lanes_closed"] is True
    assert decision["q3_s3_still_uncovered"] is True
    assert decision["public_058_evidence"] is False
    assert decision["local_artifact_allowed"] is False
    assert decision["primary_next_score_relevant_route"] == "explicit_approved_clean_upstream_id221_to_id226"
    assert decision["create_submission_artifact"] is False
    assert decision["submit_now"] is False
    assert decision["push_kaggle_now"] is False
    assert decision["run_kaggle_now"] is False


def test_q3s3_negative_evidence_map_is_report_only_and_names_lanes():
    module = load_module()
    report = module.build_report()
    lanes = {row["lane"] for row in report["lane_rows"]}

    assert lanes == {
        "physio_light_onset",
        "semantic_routine",
        "calendar_sleep_prior",
        "subject_drift",
        "event_cadence",
        "place_social_novelty",
        "routine_stability_place_entropy",
        "sensor_adherence_dropout",
    }
    assert report["checks"]["all_local_q3s3_lanes_closed"] is True
    assert report["checks"]["optional_scout_reports_closed"] is True
    assert report["checks"]["no_public_058_evidence"] is True
    scout_lanes = {row["lane"] for row in report["scout_rows"]}
    assert scout_lanes == {"cross_modal_disagreement_scout"}
    assert report["no_predictions_created"] is True
    assert report["no_submission_created"] is True
    assert report["no_base_prediction_artifact_created"] is True
    assert report["no_submission_log_row_created"] is True
    assert report["no_kaggle_push_or_run_made"] is True
    assert report["no_dacon_submission_made"] is True
