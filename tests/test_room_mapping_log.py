"""Mapping logs preserve evidence from the real controller, without hardware."""
import json
from pathlib import Path

import pytest

from test_room_mapping_controller import (
    ExistingReceiver, calibration_path, controller, no_network, wait_until,
)


def records(path):
    """Read complete JSONL records while a worker may still be appending."""
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines(keepends=True)
            if line.endswith("\n")]


def session_records(controller):
    return records(controller.output / "mapping.jsonl")


def events(controller, event):
    return [row for row in session_records(controller) if row["event"] == event]


def stop_and_join(controller):
    controller.stop()
    controller.thread.join(5)
    assert not controller.busy


def processed_blank_sample(controller, frame_id):
    return next((row for row in events(controller, "status_sample")
                 if row["data"]["status"].get("last_processed_source_frame_id") == frame_id
                 and row["data"]["guidance"]["code"] == "low_features"), None)


def test_live_log_records_measured_diagnostics_guidance_and_final_export(controller, calibration_path):
    receiver = ExistingReceiver()
    receiver.publish(41)
    controller.start_live(receiver, calibration_path)
    wait_until(lambda: processed_blank_sample(controller, 41) is not None)
    sample = processed_blank_sample(controller, 41)["data"]
    assert sample["status"]["diagnostics"]["num_features"] == 0
    assert sample["status"]["diagnostics"]["feature_grid"] == [[0, 0, 0]] * 3
    assert sample["guidance"]["metrics"]["num_features"] == 0
    assert sample["guidance"]["action"]
    assert sample["guidance"]["missing"]
    assert sample["camera_position"] is None
    assert sample["units"] == "arbitrary_monocular"
    assert not sample["status"]["pose_valid"]

    stop_and_join(controller)
    rows = session_records(controller)
    assert rows[0]["event"] == "session_start"
    assert rows[-1]["event"] == "session_end"
    assert [row["record_id"] for row in rows] == list(range(1, len(rows) + 1))
    for row in rows:
        assert row["schema_version"] == 1
        assert row["timestamp_utc"]
        assert row["timestamp_epoch"] > 0
        assert row["timestamp_monotonic"] > 0
        assert row["elapsed_seconds"] >= 0
    assert all(left["elapsed_seconds"] <= right["elapsed_seconds"]
               for left, right in zip(rows, rows[1:]))

    metadata = rows[0]["data"]
    assert metadata["mode"] == "live"
    calibration = metadata["calibration"]
    assert Path(calibration["path"]) == calibration_path
    assert calibration["K"] == [[120, 0, 80], [0, 120, 60], [0, 0, 1]]
    assert (calibration["width"], calibration["height"]) == (160, 120)
    assert calibration["distortion"] == []
    assert metadata["slam_config"]["retain_map"] is True

    ordered_events = [row["event"] for row in rows]
    assert ordered_events.count("stop_requested") == 1
    assert ordered_events.index("stop_requested") < ordered_events.index("map_saved")
    saved = events(controller, "map_saved")[-1]["data"]
    assert saved["landmarks"] == 0
    assert saved["keyframes"] == 0
    assert "map.json" in json.dumps(saved["files"])
    assert "cloud.ply" in json.dumps(saved["files"])
    assert "view.json" in json.dumps(saved["files"])
    assert rows[-1]["data"]["state"] == "stopped"
    assert rows[-1]["data"]["saved"] is True
    status, _, _ = controller.read()
    assert Path(status["log_path"]) == controller.output / "mapping.jsonl"
    assert status["log_records"] == len(rows)
    assert status["log_closed"] is True
    assert not status["log_error"]
    assert status["flight_commands_sent"] == 0
    assert receiver.state == "STREAMING"


def test_disconnect_logs_exception_and_closes_after_preserving_map(controller, calibration_path):
    receiver = ExistingReceiver()
    receiver.publish(7)
    controller.start_live(receiver, calibration_path)
    wait_until(lambda: controller.mapping.status()["processed_frames"] == 1)
    receiver.error = "fixture receiver disconnected"
    receiver.state = "ERROR"
    controller.thread.join(5)
    assert not controller.busy

    rows = session_records(controller)
    errors = events(controller, "error")
    assert errors
    assert any("fixture receiver disconnected" in row["data"]["message"] for row in errors)
    assert any("RuntimeError" in row["data"]["type"] for row in errors)
    assert any("fixture receiver disconnected" in row["data"]["traceback"] for row in errors)
    assert (controller.output / "map.json").is_file()
    assert (controller.output / "cloud.ply").is_file()
    assert rows[-1]["event"] == "session_end"
    assert rows[-1]["data"]["state"] == "error"
    assert rows[-1]["data"]["saved"] is True
    status, _, _ = controller.read()
    assert status["state"] == "error"
    assert status["log_closed"] is True
    assert not status["pose_valid"]


def test_restarting_creates_separate_complete_logs(controller, calibration_path):
    receiver = ExistingReceiver()
    receiver.publish(1)
    controller.start_live(receiver, calibration_path)
    wait_until(lambda: controller.mapping.status()["processed_frames"] == 1)
    stop_and_join(controller)
    first_path = Path(controller.read()[0]["log_path"])
    first_contents = first_path.read_bytes()
    receiver.publish(2)
    controller.start_live(receiver, calibration_path)
    wait_until(lambda: controller.mapping.status()["processed_frames"] == 1)
    stop_and_join(controller)
    second_path = Path(controller.read()[0]["log_path"])
    assert first_path != second_path
    assert first_path.read_bytes() == first_contents
    for path in (first_path, second_path):
        rows = records(path)
        assert rows[0]["event"] == "session_start"
        assert rows[0]["record_id"] == 1
        assert rows[-1]["event"] == "session_end"


def test_simulation_log_labels_synthetic_units_and_final_result(controller, monkeypatch):
    from follow_neo import room_survey_sim

    class Simulator:
        def step(self):
            return dict(source="simulation", units="meters_simulated", state="complete",
                        message="fixture complete", points=[{"id": 1, "position": [1, 2, 3]}],
                        keyframes=[], coverage=.8, camera_position=[0, 0, 1],
                        target_position=None, path=[])

    monkeypatch.setattr(room_survey_sim, "RoomSurveySimulation", Simulator)
    controller.start_simulation()
    controller.thread.join(5)
    assert not controller.busy
    rows = session_records(controller)
    assert rows[0]["data"]["mode"] == "simulation"
    samples = events(controller, "status_sample")
    assert samples
    assert all(row["data"]["units"] == "meters_simulated" for row in samples)
    assert all(row["data"]["status"]["mode"] == "simulation" for row in samples)
    assert all(row["data"]["guidance"]["code"].startswith("simulation") for row in samples)
    saved = events(controller, "map_saved")[-1]["data"]
    assert saved["landmarks"] == 1
    assert saved["keyframes"] == 0
    assert rows[-1]["event"] == "session_end"
    assert rows[-1]["data"]["state"] == "complete"
    assert rows[-1]["data"]["saved"] is True
    assert not (controller.output / "map.json").exists()
    assert controller.read()[0]["log_closed"] is True


def test_log_write_failure_is_visible_but_mapping_and_map_export_continue(controller, calibration_path):
    receiver = ExistingReceiver()
    receiver.publish(10)
    controller.start_live(receiver, calibration_path)
    wait_until(lambda: processed_blank_sample(controller, 10) is not None)

    class FailingWrite:
        def __init__(self, stream):
            self.stream = stream
            self.attempts = 0

        def write(self, text):
            self.attempts += 1
            raise OSError("fixture mapping log disk is full")

        def flush(self):
            self.stream.flush()

        def close(self):
            self.stream.close()

    # Replace the actual disk boundary, not the logger's error-handling method.
    with controller.log._write_lock:
        failed_stream = FailingWrite(controller.log._stream)
        controller.log._stream = failed_stream
    wait_until(lambda: controller.read()[0]["log_error"] is not None)
    receiver.publish(11)
    wait_until(lambda: controller.mapping.status()["processed_frames"] == 2)
    stop_and_join(controller)
    status, _, _ = controller.read()
    assert "fixture mapping log disk is full" in status["log_error"]
    assert status["state"] == "stopped"
    assert status["saved"] is True
    assert status["log_closed"] is True
    assert (controller.output / "map.json").is_file()
    assert (controller.output / "cloud.ply").is_file()
    assert (controller.output / "view.json").is_file()
    assert failed_stream.stream.closed
    assert failed_stream.attempts == 1  # Persistent failure must not become a retry loop.
    rows = session_records(controller)
    assert status["log_records"] == len(rows)
    assert all(row["event"] != "session_end" for row in rows)


def test_worker_start_failure_still_closes_its_log(controller, calibration_path, monkeypatch):
    from follow_neo import room_mapping

    def cannot_start(self):
        raise RuntimeError("fixture cannot create mapper worker")

    monkeypatch.setattr(room_mapping.threading.Thread, "start", cannot_start)
    receiver = ExistingReceiver()
    receiver.publish(1)
    with pytest.raises(RuntimeError, match="cannot create mapper worker"):
        controller.start_live(receiver, calibration_path)
    assert not controller.busy
    rows = session_records(controller)
    assert rows[0]["event"] == "session_start"
    assert rows[-1]["event"] == "session_end"
    assert rows[-1]["data"]["state"] == "error"
    assert events(controller, "error")[0]["data"]["message"] == "fixture cannot create mapper worker"
    assert controller.read()[0]["log_closed"] is True
    # The shared fixture only joins threads that have actually started.
    controller.thread = None


def test_failed_new_log_keeps_previous_saved_session_intact(controller, calibration_path, monkeypatch):
    from follow_neo import room_mapping

    receiver = ExistingReceiver()
    receiver.publish(1)
    controller.start_live(receiver, calibration_path)
    wait_until(lambda: controller.mapping.status()["processed_frames"] == 1)
    stop_and_join(controller)
    previous_mapping = controller.mapping
    previous_output = controller.output
    previous_log = controller.log
    previous_thread = controller.thread
    previous_status = dict(controller._status)
    previous_view = controller._view
    previous_version = controller._version
    previous_log_bytes = (previous_output / "mapping.jsonl").read_bytes()
    attempted_paths = []

    def cannot_open_log(path, metadata):
        attempted_paths.append(Path(path))
        raise OSError("fixture cannot open next mapping log")

    monkeypatch.setattr(room_mapping, "MappingLog", cannot_open_log)
    receiver.publish(2)
    with pytest.raises(OSError, match="cannot open next mapping log"):
        controller.start_live(receiver, calibration_path)

    assert attempted_paths and attempted_paths[0].parent != previous_output
    assert controller.mapping is previous_mapping
    assert controller.output == previous_output
    assert controller.log is previous_log
    assert controller.thread is previous_thread
    assert controller._status == previous_status
    assert controller._view is previous_view
    assert controller._version == previous_version
    assert controller.stop_event.is_set()
    assert not controller.busy
    assert (previous_output / "mapping.jsonl").read_bytes() == previous_log_bytes
    controller.request_save()
    status, _, _ = controller.read()
    assert status["saved"] is True
    assert status["output"] == str(previous_output)
    assert status["message"] == "Map already saved: " + str(previous_output)
    assert Path(status["log_path"]) == previous_output / "mapping.jsonl"
    assert status["log_closed"] is True
    assert (previous_output / "map.json").is_file()
    assert not (attempted_paths[0].parent / "map.json").exists()
    assert receiver.state == "STREAMING"
