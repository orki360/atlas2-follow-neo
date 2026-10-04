"""Offline rendered pixels through the production live-mode mapping path.

No receiver, aircraft, GUI or socket is opened. A receiver-shaped holder uses
the application's real Frame and LatestValue types, while RoomMappingController
and MappingSession run unchanged. Frames are paced until processing completes;
this is not a live-rate benchmark or evidence of real-camera performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time
from unittest.mock import patch
import uuid

import numpy as np

from .room_mapping import RoomMappingController, mapping_import
from .mapping_guidance import build_mapping_guidance
from .video import Frame, LatestValue


class RenderedFrameHolder:
    """Only decoded-frame publication; receiver lifecycle calls are errors."""

    def __init__(self):
        self.frames = LatestValue()
        self.state = "STREAMING"
        self.error = None
        self.lifecycle_calls = []

    def publish(self, frame_id, image):
        self.frames.set(Frame(frame_id, time.monotonic(), image))

    def start(self):
        self.lifecycle_calls.append("start")
        raise AssertionError("Mapping attempted to start the synthetic frame holder")

    def stop(self):
        self.lifecycle_calls.append("stop")
        raise AssertionError("Mapping attempted to stop the synthetic frame holder")


def _wait_processed(controller, frame_id, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = controller.mapping.status()
        if state["error"]:
            raise RuntimeError(str(state["error"]))
        control, _, _ = controller.read()
        if control["state"] == "error":
            raise RuntimeError(control["message"])
        if state["last_processed_source_frame_id"] == frame_id:
            return state
        if not controller.busy:
            raise RuntimeError("Mapping controller stopped before processing the frame")
        time.sleep(.005)
    raise TimeoutError(f"Frame {frame_id} did not finish within {timeout} seconds")


def _frame_record(state, controller, kind, rendered_image_index=None):
    control, _, _ = controller.read()
    return {
        "kind": kind,
        "rendered_image_index": rendered_image_index,
        "source_frame_id": state["last_processed_source_frame_id"],
        "tracker_frame_id": state["tracker_frame_id"],
        "tracking_state": state["tracking_state"],
        "algorithm_tracking": state["tracking_state"] == "tracking",
        "relocalized": state["relocalized"],
        "processing_seconds": state["last_processing_seconds"],
        "frame_age_seconds_at_observation": state["frame_age_seconds"],
        "stale_at_observation": state["stale"],
        "fresh_pose_at_observation": state["pose_valid"],
        "controller_pose_valid_at_observation": control["pose_valid"],
        "landmarks": state["landmarks"],
        "keyframes": state["keyframes"],
        "matches": state["num_matches"],
        "inliers": state["num_inliers"],
        "message": state["message"],
        "diagnostics": state.get("diagnostics", {}),
        "guidance": build_mapping_guidance(control),
    }


def _read_ply(path):
    """Independent ASCII reader for the saved positions and declared count."""
    lines = Path(path).read_text(encoding="ascii").splitlines()
    if lines[:2] != ["ply", "format ascii 1.0"]:
        raise AssertionError("Export is not an ASCII PLY")
    end = lines.index("end_header")
    count_lines = [line for line in lines[:end] if line.startswith("element vertex ")]
    if len(count_lines) != 1:
        raise AssertionError("PLY must declare one vertex count")
    count = int(count_lines[0].split()[-1])
    rows = [line.split() for line in lines[end + 1:] if line.strip()]
    if len(rows) != count or any(len(row) != 5 for row in rows):
        raise AssertionError("PLY count or exported property count is inconsistent")
    points = np.array([[float(value) for value in row[:3]] for row in rows]).reshape((-1, 3))
    if not np.isfinite(points).all():
        raise AssertionError("PLY contains nonfinite coordinates")
    return points


def run_check(base, *, frame_timeout=60.0):
    """Return and save truthful synthetic results, including failures.

    Production defaults, especially the one-second stale threshold, are never
    overridden. Success concerns software integration on these rendered frames;
    slow successful tracking is recorded separately from usable fresh poses.
    """
    if not np.isfinite(frame_timeout) or frame_timeout <= 0:
        raise ValueError("frame_timeout must be positive and finite")
    base = Path(base).resolve()
    output = base / "outputs" / ("pipeline-" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "success": False,
        "scope": "Synthetic rendered BGR -> Frame/LatestValue -> production RoomMappingController -> MappingSession -> saved map",
        "source": "synthetic_rendered_images",
        "controller_code_path": "live",
        "real_video_tested": False,
        "live_video_tested": False,
        "real_camera_calibration_used": False,
        "real_time_performance_validated": False,
        "autonomous_flight_tested": False,
        "flight_commands_sent": 0,
        "production_configuration_unchanged": True,
        "pacing": "Next source frame is published only after the previous frame finishes processing",
        "scene": {"seed": 32, "rendered_frames": 10, "textured_patches": 100},
        "frames": [],
        "checks": {},
        "output": str(output),
    }
    controller = RoomMappingController(output)
    receiver = RenderedFrameHolder()
    network_attempts = []

    def forbidden(*args, **kwargs):
        network_attempts.append("socket operation")
        raise AssertionError("Offline mapping validation attempted a socket operation")

    try:
        # Preload production modules before stamping the first decoded frame.
        # No mocked geometry, tracker, calibration, or map implementation is used.
        render_scene = mapping_import("render_scene")
        map_io = mapping_import("map_io")
        map_export = mapping_import("map_export")
        evaluation = mapping_import("evaluation")
        mapping_import("mapping_session")
        mapping_import("calibration")
        truth, grayscale = render_scene.render_sequence(seed=32, n_frames=10, n_points=100)
        images = [np.repeat(image[:, :, None], 3, axis=2) for image in grayscale]
        calibration = output / "synthetic-calibration.json"
        calibration.write_text(json.dumps({
            "K": truth.K.tolist(), "width": truth.width, "height": truth.height,
            "distortion": [], "provenance": "synthetic renderer; not measured DJI calibration",
        }, indent=2), encoding="utf-8")

        # Blocking socket construction also prevents accidentally creating a
        # second receiver or issuing datagrams, rather than only blocking connect.
        with patch.object(socket, "socket", forbidden), patch.object(socket, "create_connection", forbidden):
            try:
                receiver.publish(0, images[0])
                controller.start_live(receiver, calibration)
                for frame_id, image in enumerate(images):
                    if frame_id:
                        receiver.publish(frame_id, image)
                    state = _wait_processed(controller, frame_id, frame_timeout)
                    report["frames"].append(_frame_record(state, controller, "rendered", frame_id))
                report["landmarks_before_loss"] = state["landmarks"]
                receiver.publish(len(images), np.zeros_like(images[-1]))
                lost = _wait_processed(controller, len(images), frame_timeout)
                report["frames"].append(_frame_record(lost, controller, "blank_loss"))
                receiver.publish(len(images) + 1, images[-1])
                last_view = _wait_processed(controller, len(images) + 1, frame_timeout)
                report["frames"].append(_frame_record(last_view, controller, "return_to_last_view", len(images) - 1))
                report["last_view_relocalized"] = bool(last_view["tracking_state"] == "tracking" and last_view["relocalized"])
                # A stored earlier view has stronger mapped overlap. Always
                # test the same return path, and retain any failed last-view
                # attempt in the report instead of hiding it with config changes.
                receiver.publish(len(images) + 2, images[6])
                recovered = _wait_processed(controller, len(images) + 2, frame_timeout)
                report["frames"].append(_frame_record(recovered, controller, "return_to_earlier_mapped_view", 6))
                report["earlier_mapped_view_tracking"] = recovered["tracking_state"] == "tracking"
                report["relocalization_observed_after_loss"] = bool(report["last_view_relocalized"] or (recovered["tracking_state"] == "tracking" and recovered["relocalized"]))
                report["limitations_observed"] = []
                if not report["last_view_relocalized"]:
                    report["limitations_observed"].append("After a blank frame, returning to the last rendered view did not relocalize with production defaults; the earlier mapped-view return is a separate recovery attempt.")
                report["stale_after_seconds"] = recovered["stale_after_seconds"]

                # Save while mapping is still running, then compare to the final
                # auto-export after stop. This exercises both user operations.
                controller.request_save()
                deadline = time.monotonic() + frame_timeout
                while time.monotonic() < deadline:
                    control, _, _ = controller.read()
                    if control["state"] == "error":
                        raise RuntimeError(control["message"])
                    if control.get("saved"):
                        break
                    time.sleep(.01)
                else:
                    raise TimeoutError("Requested map export did not finish")
                requested_map = map_io.load_map(controller.output / "map.json")
                requested_payload = json.loads((controller.output / "map.json").read_text(encoding="utf-8"))
                report["requested_export_landmarks"] = len(requested_map.landmarks)
            finally:
                controller.stop()
                if controller.thread:
                    controller.thread.join(frame_timeout)
                if controller.busy:
                    raise TimeoutError("Mapping controller is still stopping")

            control, view, _ = controller.read()
            stopped = controller.mapping.status()
            report["stopped_controller_status"] = control
            report["stopped_session_status"] = stopped
            report["map_output"] = str(controller.output)
            (controller.output / "provenance.json").write_text(json.dumps({
                "source": "synthetic_rendered_images", "controller_code_path": "live",
                "real_video_tested": False, "flight_commands_sent": 0,
                "validation_report": str(output / "pipeline-check.json"),
            }, indent=2), encoding="utf-8")
            if control["state"] == "error":
                raise RuntimeError(control["message"])
            archived = map_io.load_map(controller.output / "map.json")
            snapshot = map_export.map_snapshot(archived)
            saved_view = json.loads((controller.output / "view.json").read_text(encoding="utf-8"))
            ply = _read_ply(controller.output / "cloud.ply")
            points = np.asarray([point["position"] for point in snapshot["points"]]).reshape((-1, 3))
            report["landmarks"] = len(archived.landmarks)
            report["keyframes"] = len(archived.keyframes)
            report["export_quality"] = snapshot["quality"]
            report["ply_vertices"] = len(ply)
            initial = report["frames"][:len(images)]
            report["algorithm_tracking_frames"] = sum(frame["algorithm_tracking"] for frame in initial)
            report["fresh_pose_frames_at_observation"] = sum(frame["fresh_pose_at_observation"] for frame in initial)
            report["stale_tracking_frames_at_observation"] = sum(frame["algorithm_tracking"] and frame["stale_at_observation"] for frame in initial)
            report["max_processing_seconds"] = max(frame["processing_seconds"] for frame in report["frames"])

            # Keyframe poses remain available in the saved map even when a
            # computation exceeds the unchanged live freshness threshold.
            # Alignment uses external renderer truth, not projections generated
            # from the estimated map. Its fitted scale is diagnostic only.
            expected_centers, estimated_centers = [], []
            for keyframe in sorted(archived.keyframes.values(), key=lambda item: item.frame_id):
                truth_index = report["frames"][keyframe.frame_id]["rendered_image_index"]
                if truth_index is None:
                    raise AssertionError("A blank frame became a mapped keyframe")
                pose = truth.frames[truth_index].pose
                expected_centers.append(-pose[:3, :3].T @ pose[:3, 3])
                estimated_centers.append(-keyframe.pose[:3, :3].T @ keyframe.pose[:3, 3])
            alignment = evaluation.evaluate_trajectory(expected_centers, estimated_centers)
            report["keyframe_trajectory_truth_comparison"] = {
                "poses": len(expected_centers), "alignment": "Sim(3), diagnostic with synthetic ground truth only",
                "rmse": alignment.rmse, "maximum_error": alignment.maximum_error,
                "fitted_scale": alignment.scale,
            }
            checks = report["checks"]
            checks.update({
                "all_source_frames_processed_once": stopped["processed_frames"] == len(images) + 3 and stopped["submitted_frames"] == len(images) + 3 and stopped["replaced_pending_frames"] == 0,
                "initialized_from_images": initial[0]["tracking_state"] == "initializing" and report["keyframes"] >= 3,
                "multiple_tracking_results": report["algorithm_tracking_frames"] >= 3,
                "blank_frame_lost": lost["tracking_state"] == "lost" and not lost["pose_valid"],
                "recovered_by_relocalization": report["relocalization_observed_after_loss"] and report["earlier_mapped_view_tracking"],
                "map_retained_after_loss": len(archived.landmarks) >= report["landmarks_before_loss"] > 20,
                "requested_save_matches_final_map": requested_payload == json.loads((controller.output / "map.json").read_text(encoding="utf-8")),
                "filtered_cloud_nonempty": len(points) > 20,
                "ply_matches_loaded_map": ply.shape == points.shape and np.allclose(ply, points, rtol=1e-12, atol=1e-12),
                "saved_view_matches_loaded_map": saved_view["points"] == snapshot["points"] and saved_view["keyframes"] == snapshot["keyframes"],
                "keyframe_trajectory_agrees_with_synthetic_truth": alignment.rmse < .05 and alignment.maximum_error < .1,
                "stale_threshold_unchanged": stopped["stale_after_seconds"] == 1.0,
                "stopped_pose_invalid": not control["pose_valid"] and not stopped["pose_valid"] and view["camera_position"] is None and saved_view["camera_position"] is None,
                "workers_stopped": not controller.busy and not stopped["worker_alive"] and stopped["state"] == "stopped",
                "receiver_lifecycle_untouched": not receiver.lifecycle_calls,
                "no_socket_operations": not network_attempts,
                "no_flight_commands": control["flight_commands_sent"] == 0,
            })
            report["success"] = bool(all(checks.values()))
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["controller_worker_alive_on_failure"] = controller.busy
        if controller.mapping:
            report["session_status_on_failure"] = controller.mapping.status()
    finally:
        report["receiver_lifecycle_calls"] = receiver.lifecycle_calls
        report["socket_attempts"] = len(network_attempts)
        (output / "pipeline-check.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = run_check(args.base)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
