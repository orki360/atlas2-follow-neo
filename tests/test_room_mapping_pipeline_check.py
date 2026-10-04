"""Actual rendered-image controller integration; no tracker or geometry mocks."""
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from follow_neo.room_mapping_pipeline_check import _read_ply, run_check


@pytest.fixture
def short_output_root():
    # Avoid nesting the generated run name and atomic-export suffix beneath a
    # long pytest node name on Windows installations with legacy MAX_PATH.
    parent = Path(__file__).resolve().parents[1] / "outputs"
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pipe-test-", dir=parent) as folder:
        path = Path(folder).resolve()
        assert path.parent == parent.resolve()
        yield path


def test_real_controller_tracks_rendered_frames_and_exports_valid_map(short_output_root):
    report = run_check(short_output_root)
    assert report["success"], report
    assert all(report["checks"].values())
    assert report["source"] == "synthetic_rendered_images"
    assert report["controller_code_path"] == "live"
    assert not report["real_video_tested"]
    assert not report["real_time_performance_validated"]
    assert not report["autonomous_flight_tested"]
    assert report["socket_attempts"] == 0
    assert report["receiver_lifecycle_calls"] == []
    assert report["algorithm_tracking_frames"] >= 3
    assert report["ply_vertices"] > 20
    assert report["keyframes"] >= 3
    assert report["stale_after_seconds"] == 1.0
    assert len(report["frames"]) == 13
    assert report["frames"][10]["tracking_state"] == "lost"
    assert report["frames"][12]["tracking_state"] == "tracking"
    # Exercise the actual ORB -> session diagnostics -> controller -> advisor
    # contract, including the real blank image rather than invented counters.
    for frame in report['frames']:
        measured = frame['diagnostics']
        assert measured['num_features'] == sum(sum(row) for row in measured['feature_grid'])
        assert frame['guidance']['reference'].startswith('Directions refer to the camera image')
    assert report['frames'][10]['diagnostics']['num_features'] == 0
    assert report['frames'][10]['guidance']['code'] in ('low_features','processing_lag','stale_video')
    if not report["last_view_relocalized"]:
        assert report["limitations_observed"]
    # A slow successful estimate must never be counted as a fresh live pose.
    for frame in report["frames"]:
        if frame["stale_at_observation"] or not frame["algorithm_tracking"]:
            assert not frame["fresh_pose_at_observation"]
    assert report["keyframe_trajectory_truth_comparison"]["rmse"] < .05
    assert not report["stopped_controller_status"]["pose_valid"]
    assert not report["stopped_session_status"]["worker_alive"]
    saved = json.loads((Path(report["output"]) / "pipeline-check.json").read_text(encoding="utf-8"))
    assert saved == report
    provenance = json.loads((Path(report["map_output"]) / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["source"] == "synthetic_rendered_images"
    assert not provenance["real_video_tested"]


@pytest.mark.parametrize("body", [
    "0 1 2 2 0.1\n",  # Header promises two vertices.
    "0 1 2\n3 4 5\n",  # Quality properties omitted.
    "0 1 nan 2 0.1\n3 4 5 2 0.1\n",
])
def test_ply_verification_rejects_truncated_or_invalid_exports(tmp_path, body):
    path = tmp_path / "invalid.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 2\nend_header\n" + body, encoding="ascii")
    with pytest.raises(AssertionError):
        _read_ply(path)


def test_ply_verification_reads_coordinates_independently(tmp_path):
    path = tmp_path / "points.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 2\nend_header\n0.5 1 -2 2 0.1\n3 4 5 3 0.2\n", encoding="ascii")
    np.testing.assert_array_equal(_read_ply(path), [[.5, 1, -2], [3, 4, 5]])
