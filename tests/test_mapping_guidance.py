"""Guidance uses image evidence; it cannot direct a real aircraft or certify space."""
import copy
import json

import pytest

from follow_neo.mapping_guidance import build_mapping_guidance


def healthy(**changes):
    status = dict(state="running", mode="live", pose_valid=True,
                  tracking_state="tracking", frame_age_seconds=.08,
                  stale_after_seconds=1.0, last_processing_seconds=.03,
                  last_processed_source_frame_id=0, num_matches=150,
                  num_inliers=100, landmarks=300, keyframes=8,
                  diagnostics_fresh=True,
                  diagnostics=dict(num_features=360, feature_grid=[40] * 9,
                                   feature_occupied_cells=9))
    status.update(changes)
    return status


def test_healthy_evidence_gives_conditional_manual_guidance_and_no_room_coverage():
    result = build_mapping_guidance(healthy())
    assert result["code"] == "steady"
    assert result["direction"] == "neutral"
    assert "checking free space" in result["action"]
    assert "completion" in result["reason"]
    assert "not aircraft axes" in result["reference"]
    assert "Free space is unverified" in result["reference"]
    assert result["metrics"]["num_features"] == 360
    assert result["metrics"]["inlier_ratio"] == pytest.approx(2/3)


@pytest.mark.parametrize("state,code", [("idle", "inactive"), ("stopping", "inactive"),
                                        ("stopped", "inactive"), ("error", "error")])
def test_inactive_or_error_overrides_old_pose_and_image_evidence(state, code):
    result = build_mapping_guidance(healthy(state=state, message="Failed processing"))
    assert result["code"] == code
    assert result["direction"] == "neutral"
    assert "translation" not in result["action"]
    assert "pan or tilt" not in result["action"]


def test_selected_calibration_idle_copy_does_not_ask_to_select_again():
    result = build_mapping_guidance(dict(state="idle", calibration_selected=True))
    assert result["action"].startswith("Start live mapping")


@pytest.mark.parametrize("view", [None, {"camera_position": [1, 2, 3], "points": [[1, 2, 3]]}])
def test_pose_or_saved_cloud_without_current_measurements_cannot_guide(view):
    result = build_mapping_guidance(dict(state="running", pose_valid=True), view)
    assert result["code"] == "awaiting_frames"
    assert result["metrics"] == {}
    assert "Feature count unavailable" in result["missing"]
    assert result["direction"] == "neutral"


def test_processed_frame_zero_is_valid_and_missing_feature_count_is_not_low_texture():
    status = healthy(diagnostics={})
    result = build_mapping_guidance(status)
    assert result["code"] == "diagnostics_unavailable"
    assert "num_features" not in result["metrics"]
    assert result["direction"] == "neutral"


def test_absent_id_is_acceptable_when_actual_diagnostics_are_available():
    status = healthy()
    del status["last_processed_source_frame_id"]
    assert build_mapping_guidance(status)["code"] == "steady"


def test_stale_overrides_low_features_and_lost_tracking():
    status = healthy(frame_age_seconds=2.1, tracking_state="lost", pose_valid=False,
                     diagnostics=dict(num_features=0))
    result = build_mapping_guidance(status)
    assert result["code"] == "stale_video"
    assert "cause is not established" in result["reason"]
    assert "translation" not in result["action"]
    assert result["direction"] == "neutral"


@pytest.mark.parametrize("age", [.1, 2.0])
def test_long_processing_is_reported_separately_from_video_loss(age):
    result = build_mapping_guidance(healthy(frame_age_seconds=age, last_processing_seconds=1.4))
    assert result["code"] == "processing_lag"
    assert "processing" in result["reason"] or "computation" in result["reason"]
    assert "translation" not in result["action"]


def test_freshness_limit_is_configurable():
    result = build_mapping_guidance(healthy(frame_age_seconds=.4, stale_after_seconds=.2))
    assert result["code"] == "stale_video"


def test_explicit_stale_diagnostics_blocks_direction_even_with_pose():
    result = build_mapping_guidance(healthy(diagnostics_fresh=False))
    assert result["code"] == "diagnostics_stale"
    assert result["direction"] == "neutral"


def test_low_feature_count_reports_measurement_and_possible_causes_not_diagnoses():
    status = healthy(diagnostics=dict(num_features=12, feature_grid=[0] * 9,
                                     dark_fraction=.8, sharpness_laplacian_variance=6.))
    result = build_mapping_guidance(status)
    assert result["code"] == "low_features"
    assert "12 features" in result["reason"]
    assert "80% dark pixels" in result["reason"]
    assert "may contribute" in result["reason"]
    assert "blur or plain surfaces" in result["reason"]
    assert "not a confirmed cause" in result["reason"]
    assert result["direction"] == "neutral"


def test_bright_fraction_only_exposure_diagnostic_does_not_imply_darkness():
    result = build_mapping_guidance(healthy(diagnostics=dict(num_features=20, bright_fraction=.9)))
    assert "90% bright pixels" in result["reason"]
    assert "dark pixels" not in result["reason"]


def test_many_features_in_two_cells_is_reported_as_poor_distribution():
    result = build_mapping_guidance(healthy(diagnostics=dict(num_features=360,
                    feature_grid=[180, 180, 0, 0, 0, 0, 0, 0, 0])))
    assert result["code"] == "low_features"
    assert "2/9 image cells" in result["reason"]


def test_lost_asks_to_recover_view_not_reverse_flight():
    result = build_mapping_guidance(healthy(tracking_state="lost", pose_valid=False, num_inliers=0))
    assert result["code"] == "tracking_lost"
    assert "camera view" in result["action"]
    assert "do not blindly reverse flight" in result["action"]
    assert result["direction"] == "neutral"


@pytest.mark.parametrize("inliers,matches", [(12, 80), (40, 200)])
def test_weak_tracking_considers_count_and_ratio(inliers, matches):
    result = build_mapping_guidance(healthy(num_inliers=inliers, num_matches=matches))
    assert result["code"] == "weak_tracking"
    assert "heuristic" in result["reason"]
    assert "Slow down" in result["action"]


@pytest.mark.parametrize("reason", ["need_second_view", "need_baseline"])
def test_initialization_with_good_features_needs_translation_even_with_zero_pose_inliers(reason):
    status = healthy(tracking_state="initializing", pose_valid=False, num_inliers=0,
                     diagnostics=dict(num_features=300, initialization_reason=reason))
    result = build_mapping_guidance(status)
    assert result["code"] == "initialization_baseline"
    assert "checking free space" in result["action"]
    assert "Rotation alone is insufficient" in result["action"]
    assert result["direction"] == "neutral"


def test_initialization_needing_matches_does_not_ask_for_more_baseline():
    status = healthy(tracking_state="initializing", pose_valid=False, num_inliers=0,
                     diagnostics=dict(num_features=300, initialization_reason="need_matches"))
    result = build_mapping_guidance(status)
    assert result["code"] == "initialization_matches"
    assert "previous view" in result["action"]


def test_initialization_without_feature_evidence_does_not_suggest_translation():
    result = build_mapping_guidance(healthy(tracking_state="initializing", pose_valid=False,
                                            num_inliers=0, diagnostics={}))
    assert result["code"] == "initialization_wait"
    assert "translation" not in result["action"]


def test_capacity_is_reported_instead_of_continued_survey():
    result = build_mapping_guidance(healthy(mapping_capacity_reason="Keyframe limit reached"))
    assert result["code"] == "map_capacity"
    assert "Save" in result["action"]
    assert result["reason"] == "Keyframe limit reached"


def test_invalid_pose_blocks_steady_guidance_with_healthy_counts():
    result = build_mapping_guidance(healthy(pose_valid=False))
    assert result["code"] == "pose_unavailable"
    assert result["direction"] == "neutral"


def test_unknown_frame_age_is_not_fresh_and_does_not_create_direction():
    result = build_mapping_guidance(healthy(frame_age_seconds=None))
    assert result["code"] == "diagnostics_unavailable"
    assert "frame_age_seconds" not in result["metrics"]


def test_unique_sparse_top_image_sector_produces_camera_relative_hint():
    status = healthy(diagnostics=dict(num_features=240,
                feature_grid=[[0, 0, 0], [20, 20, 20], [60, 60, 60]]))
    result = build_mapping_guidance(status)
    assert result["code"] == "inspect_image_sector"
    assert result["direction"] == "up"
    assert "pan or tilt" in result["action"]
    assert "not mean" in result["reason"]
    assert "unmapped or clear" in result["reason"]
    assert "fly" not in result["action"]


@pytest.mark.parametrize("grid", [[40]*9, [1, 39, 40, 39, 40, 40, 40, 40, 40],
                                   [38, 38, 38, 40, 40, 40, 40, 40, 40]])
def test_equal_tied_or_small_sector_differences_do_not_invent_direction(grid):
    result = build_mapping_guidance(healthy(diagnostics=dict(num_features=360, feature_grid=grid)))
    assert result["code"] == "steady"
    assert result["direction"] == "neutral"


@pytest.mark.parametrize("state", ["running", "stopped", "complete"])
def test_simulation_is_explicit_and_does_not_suggest_real_flight(state):
    result = build_mapping_guidance(dict(state=state, mode="simulation", pose_valid=True))
    assert result["code"] in ("simulation", "simulation_stopped")
    assert "simulation" in result["title"].lower()
    assert "real room" in result["action"]
    assert result["direction"] == "neutral"


@pytest.mark.parametrize("bad", [None, True, "many", float("nan"), float("inf"), -1])
def test_missing_or_invalid_metrics_are_not_treated_as_zero(bad):
    status = healthy(diagnostics=dict(num_features=bad, feature_grid=[bad]*9))
    result = build_mapping_guidance(status)
    assert result["code"] == "diagnostics_unavailable"
    assert "num_features" not in result["metrics"]
    json.dumps(result, allow_nan=False)


def test_input_is_not_mutated_and_output_has_stable_display_contract():
    status = healthy()
    view = dict(points=[dict(id=3, position=[4, 5, 6])])
    before = copy.deepcopy((status, view))
    result = build_mapping_guidance(status, view)
    assert (status, view) == before
    assert set(result) == {"code", "severity", "title", "action", "reason", "metrics",
                           "missing", "reference", "direction"}
    assert isinstance(result["missing"], list)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("num_features,reason", [(300, "need_baseline"), (10, "need_features")])
@pytest.mark.parametrize("age,fresh", [(None, True), (None, None), (.1, False)])
def test_initializing_or_low_features_cannot_move_without_known_fresh_evidence(num_features, reason, age, fresh):
    result = build_mapping_guidance(healthy(
        tracking_state="initializing", pose_valid=False, num_inliers=0,
        frame_age_seconds=age, diagnostics_fresh=fresh,
        diagnostics=dict(num_features=num_features, initialization_reason=reason)))
    assert result["code"] in ("diagnostics_unavailable", "diagnostics_stale")
    assert "translation" not in result["action"]
    assert "reframe" not in result["action"]
    assert "pan or tilt" not in result["action"]
    assert result["direction"] == "neutral"


def test_actual_engine_diagnostic_contract_reaches_direction_advisor():
    import numpy as np
    from follow_neo.room_mapping import mapping_import

    diagnose = mapping_import("diagnostics").image_feature_diagnostics
    image = np.full((90, 90, 3), 128, dtype=np.uint8)
    # Real diagnostic producer receives 20 unique features in each middle cell,
    # 60 in each bottom cell, and none in the top third. No adapter renames keys.
    points = []
    for row, count in ((1, 20), (2, 60)):
        for column in range(3):
            points.extend((column * 30 + 1 + n % 10, row * 30 + 1 + n // 10)
                          for n in range(count))
    diagnostics = diagnose(image, np.asarray(points, dtype=float))
    diagnostics["median_matched_displacement_px"] = 6.5
    result = build_mapping_guidance(healthy(diagnostics=diagnostics))
    assert result["code"] == "inspect_image_sector"
    assert result["direction"] == "up"
    assert result["metrics"]["num_features"] == 240
    assert result["metrics"]["feature_occupied_cells"] == 6
    assert result["metrics"]["median_matched_displacement_px"] == 6.5
