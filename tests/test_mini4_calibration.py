"""Geometric and provenance contracts for the explicitly estimated intrinsics."""
import json
import math

import numpy as np
import pytest

from follow_neo.mini4_calibration import (
    ESTIMATE_METHOD, MAX_METADATA_BYTES, MEASURED_METHOD,
    calibration_info, estimate_mini4_calibration,
)


def _save(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("size", [(1920, 1080), (1280, 720), (3840, 2160), (854, 480)])
def test_backprojected_diagonal_rays_recover_assumed_fov(size):
    width, height = size
    report = estimate_mini4_calibration(width, height)
    inverse = np.linalg.inv(report["K"])
    # Outer image boundaries, not the centers of the corner pixels.
    a = inverse @ [-0.5, -0.5, 1.0]
    b = inverse @ [width - 0.5, height - 0.5, 1.0]
    angle = math.degrees(math.acos(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))))
    assert angle == pytest.approx(75.0, abs=1e-10)
    assert report["K"][0][0] == report["K"][1][1]
    assert (report["width"], report["height"]) == size


def test_scaling_obeys_pixel_center_convention():
    low = np.asarray(estimate_mini4_calibration(1280, 720)["K"])
    high = np.asarray(estimate_mini4_calibration(2560, 1440)["K"])
    pixel_rescale = np.array([[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]])
    np.testing.assert_allclose(high, pixel_rescale @ low)


@pytest.mark.parametrize("size", [
    (True, 720), (1280, False), (np.bool_(True), 720), (1280.0, 720),
    ("1280", 720), (None, 720), (0, 0), (-1280, 720),
    (1280, float("nan")), (float("inf"), 720), (10**1000, 720),
    (32768, 18432), (1280, 960), (1080, 1920), (1920, 1082), (2, 1),
])
def test_invalid_dimensions_rejected_without_writing(size, tmp_path):
    target = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        estimate_mini4_calibration(*size, output_path=target)
    assert not target.exists()


def test_numpy_integer_and_one_pixel_rounding_allowed():
    report = estimate_mini4_calibration(np.int64(1920), np.int32(1081))
    assert type(report["width"]) is int
    assert type(report["height"]) is int


def test_estimate_never_claims_measured_distortion_or_accuracy(tmp_path):
    target = tmp_path / "estimate.json"
    report = estimate_mini4_calibration(1920, 1080, output_path=target)
    assert json.loads(target.read_text(encoding="utf-8")) == report
    assert report["calibration_kind"] == "estimated"
    assert report["validated"] is False
    assert "success" not in report and "rms_px" not in report and "views" not in report
    assert report["distortion"] == [0.0] * 5
    provenance = report["provenance"]
    assert provenance["method"] == ESTIMATE_METHOD
    assert provenance["fov_axis"] == "diagonal (assumed)"
    assert provenance["video_fov_deg"] == 75.0
    assert provenance["distortion_measured"] is False
    assert provenance["metric_map_scale_established"] is False
    assert provenance["units"]["map_scale"] == "arbitrary"
    assert len(provenance["source_urls"]) == 2
    assert any("1x" in assumption for assumption in provenance["assumptions"])
    assert any("letterboxing" in assumption for assumption in provenance["assumptions"])
    assert any("not a lens measurement" in assumption for assumption in provenance["assumptions"])
    assert not list(tmp_path.glob("*.tmp"))


def test_existing_calibration_is_never_overwritten(tmp_path):
    target = tmp_path / "camera.json"
    target.write_text("existing measured calibration", encoding="utf-8")
    with pytest.raises(FileExistsError):
        estimate_mini4_calibration(1920, 1080, output_path=target)
    assert target.read_text(encoding="utf-8") == "existing measured calibration"


def test_estimate_loads_into_engine_and_preserves_original_frames(tmp_path):
    from rbd_slam.calibration import load_calibration
    target = tmp_path / "camera.json"
    report = estimate_mini4_calibration(1280, 720, output_path=target)
    calibration = load_calibration(target)
    np.testing.assert_array_equal(calibration.K, report["K"])
    frame = np.arange(720 * 1280 * 3, dtype=np.uint8).reshape(720, 1280, 3)
    prepared = calibration.prepare_frame(frame)
    np.testing.assert_array_equal(prepared, frame)
    assert not np.shares_memory(prepared, frame)


def test_info_estimated_overrides_false_success_claim(tmp_path):
    report = estimate_mini4_calibration(1920, 1080)
    report.update(success=True, validated=True)
    info = calibration_info(_save(tmp_path / "camera.json", report))
    assert info["kind"] == "estimated"
    assert info["validated"] is False
    assert "Estimated" in info["label"]
    assert info["provenance"]["metric_map_scale_established"] is False


def test_info_recognizes_existing_measured_workflow(tmp_path):
    report = {"success": True, "provenance": {
        "method": MEASURED_METHOD,
        "image_source": "original unresized stream frames",
        "metric_map_scale_established": False,
    }}
    info = calibration_info(_save(tmp_path / "camera.json", report))
    assert info["kind"] == "measured"
    assert info["validated"] is True
    assert any("metric SLAM scale" in warning for warning in info["warnings"])


@pytest.mark.parametrize("data", [
    {}, {"validated": True}, {"success": True}, {"calibration_kind": "measured"},
    {"success": True, "provenance": {"method": "foreign solver"}},
    {"success": False, "provenance": {"method": MEASURED_METHOD}},
    {"success": 1, "provenance": {"method": MEASURED_METHOD,
                                  "image_source": "original unresized stream frames"}},
    {"success": True, "validated": False, "provenance": {"method": MEASURED_METHOD,
                                  "image_source": "original unresized stream frames"}},
    {"provenance": "not an object", "warnings": "not a list"},
])
def test_info_foreign_unlabelled_and_failed_are_unknown(data, tmp_path):
    info = calibration_info(_save(tmp_path / "camera.json", data))
    assert info["kind"] == "unknown"
    assert info["validated"] is False
    assert info["warnings"]


def test_info_bounds_relevant_metadata(tmp_path):
    data = {"provenance": {"method": "foreign", "accepted_views": ["path"] * 10000,
                            "assumptions": ["a" * 10000] * 100},
            "warnings": ["b" * 10000] * 100}
    info = calibration_info(_save(tmp_path / "camera.json", data))
    assert "accepted_views" not in info["provenance"]
    assert len(info["provenance"]["assumptions"]) <= 16
    assert len(info["provenance"]["assumptions"][0]) <= 1000
    assert len(info["warnings"]) <= 16


@pytest.mark.parametrize("text", ["[]", "not-json", " " * (MAX_METADATA_BYTES + 1)],
                         ids=["array", "malformed", "oversize"])
def test_info_rejects_unreadable_metadata(text, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        calibration_info(path)
