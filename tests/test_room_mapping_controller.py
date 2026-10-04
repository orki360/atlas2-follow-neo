"""Controller integration tests use decoded arrays, never a network receiver."""
import json
import socket
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from follow_neo import room_mapping


class Frames:
    def __init__(self):
        self.condition = threading.Condition()
        self.value = None
        self.version = 0

    def set(self, value):
        with self.condition:
            self.value = value
            self.version += 1
            self.condition.notify_all()

    def get(self):
        with self.condition:
            return self.value

    def wait_next(self, version, timeout):
        with self.condition:
            self.condition.wait_for(lambda: self.version != version, timeout)
            return self.value, self.version


class ExistingReceiver:
    def __init__(self):
        self.frames = Frames()
        self.state = "STREAMING"
        self.error = None

    def start(self):
        pytest.fail("The mapper must not start the existing receiver")

    def stop(self):
        pytest.fail("The mapper must not stop the existing receiver")

    def publish(self, frame_id, *, age=0., shape=(120, 160, 3)):
        image = np.zeros(shape, dtype=np.uint8)
        frame = SimpleNamespace(image=image, frame_id=frame_id, decoded_at=time.monotonic() - age)
        self.frames.set(frame)
        return frame


def wait_until(predicate, timeout=4.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    pytest.fail("Timed out waiting for mapping worker")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Room mapping opened a network connection")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


@pytest.fixture
def calibration_path(tmp_path):
    path = tmp_path / "camera.json"
    path.write_text(json.dumps({"K": [[120, 0, 80], [0, 120, 60], [0, 0, 1]],
                                "width": 160, "height": 120, "distortion": []}))
    return path


@pytest.fixture
def controller(tmp_path):
    instance = room_mapping.RoomMappingController(tmp_path / "maps")
    yield instance
    instance.stop()
    if instance.thread:
        instance.thread.join(timeout=5)
        assert not instance.thread.is_alive(), "A controller thread leaked"
    if instance.log:
        instance.log.close()


def test_existing_receiver_reused_frames_counted_and_saved(controller, calibration_path):
    receiver = ExistingReceiver()
    first = receiver.publish(20)
    controller.start_live(receiver, calibration_path)
    wait_until(lambda: controller.mapping.status()["processed_frames"] == 1)
    receiver.publish(21)
    wait_until(lambda: controller.mapping.status()["processed_frames"] == 2)
    assert controller.mapping.status()["submitted_frames"] == 2
    assert receiver.frames.get().frame_id == 21
    assert first.image.sum() == 0
    controller.stop()
    controller.thread.join(4)
    assert not controller.busy
    status, view, _ = controller.read()
    assert status["flight_commands_sent"] == 0
    assert not status["pose_valid"]
    assert status["saved"]
    assert receiver.state == "STREAMING"
    assert (controller.output / "map.json").is_file()
    assert (controller.output / "cloud.ply").is_file()
    saved = json.loads((controller.output / "view.json").read_text())
    assert saved["source"] == "live"
    assert saved["units"] == "arbitrary_monocular"
    assert saved["camera_position"] is None


def test_republished_source_id_is_not_a_second_accepted_frame(controller, calibration_path):
    receiver = ExistingReceiver()
    receiver.publish(7)
    controller.start_live(receiver, calibration_path)
    wait_until(lambda: controller.mapping.status()["processed_frames"] == 1)
    receiver.publish(7)
    wait_until(lambda: controller.mapping.status()["rejected_order_frames"] == 1)
    assert controller.mapping.status()["submitted_frames"] == 1


@pytest.mark.parametrize("kind", ["missing", "malformed", "wrong_size"])
def test_bad_calibration_creates_no_session(controller, calibration_path, kind):
    receiver = ExistingReceiver()
    receiver.publish(1)
    if kind == "missing":
        calibration_path.unlink()
    elif kind == "malformed":
        calibration_path.write_text('{"K": []}')
    else:
        receiver.publish(2, shape=(60, 80, 3))
    with pytest.raises((ValueError, FileNotFoundError)):
        controller.start_live(receiver, calibration_path)
    assert not controller.busy
    assert controller.mapping is None
    assert not controller.output_root.exists()


@pytest.mark.parametrize("kind", ["absent", "empty", "stale", "disconnected"])
def test_missing_or_stale_video_rejected(controller, calibration_path, kind):
    receiver = None if kind == "absent" else ExistingReceiver()
    if kind == "stale":
        receiver.publish(1, age=2.)
    elif kind == "disconnected":
        receiver.publish(1)
        receiver.state = "DISCONNECTED"
    with pytest.raises(ValueError):
        controller.start_live(receiver, calibration_path)
    assert not controller.busy
    assert not controller.output_root.exists()


def test_stop_and_save_requests_do_not_wait_for_snapshot(controller, calibration_path, monkeypatch):
    session_class = room_mapping.mapping_import("mapping_session").MappingSession
    original_snapshot = session_class.snapshot
    entered, release = threading.Event(), threading.Event()

    def blocked_snapshot(self):
        entered.set()
        if not release.wait(4):
            raise TimeoutError("test did not release snapshot")
        return original_snapshot(self)

    monkeypatch.setattr(session_class, "snapshot", blocked_snapshot)
    receiver = ExistingReceiver()
    receiver.publish(1)
    controller.start_live(receiver, calibration_path)
    try:
        assert entered.wait(2)
        started = time.monotonic()
        controller.request_save()
        controller.stop()
        assert time.monotonic() - started < .2
        assert controller.busy
        with pytest.raises(RuntimeError, match="Stop the current"):
            controller.start_live(receiver, calibration_path)
    finally:
        release.set()
    controller.thread.join(4)
    assert not controller.busy


def test_disconnect_reports_error_without_stopping_existing_receiver(controller, calibration_path):
    receiver = ExistingReceiver()
    receiver.publish(1)
    controller.start_live(receiver, calibration_path)
    wait_until(lambda: controller.mapping.status()["processed_frames"] == 1)
    receiver.state = "ERROR"
    receiver.error = "fixture video disconnected"
    controller.thread.join(4)
    assert not controller.busy
    status, _, _ = controller.read()
    assert status["state"] == "error"
    assert "fixture video disconnected" in status["message"]
    assert not status["pose_valid"]


def test_stopping_never_revives_a_fresh_pose(controller):
    live = dict(pose_valid=True, tracking_state="tracking", landmarks=4, keyframes=2,
                frame_age_seconds=.01, replaced_pending_frames=0, mapping_capacity_reason="",
                camera_position=[1, 2, 3], error=None)
    controller.mapping = SimpleNamespace(status=lambda: live)
    controller._status.update(state="stopping", mode="live", pose_valid=False)
    controller.stop_event.set()
    status, view, _ = controller.read()
    assert not status["pose_valid"]
    assert view["camera_position"] is None


def test_simulation_completion_export_is_explicitly_synthetic(controller, monkeypatch):
    from follow_neo import room_survey_sim
    expected = dict(source="simulation", units="meters_simulated", state="complete",
                    message="fixture survey complete", points=[{"id": 1, "position": [1, 2, 3]}],
                    keyframes=[], coverage=.8, camera_position=[0, 0, 1], target_position=None, path=[])

    class Simulator:
        def step(self):
            return dict(expected)
    monkeypatch.setattr(room_survey_sim, "RoomSurveySimulation", Simulator)
    controller.start_simulation()
    controller.thread.join(2)
    assert not controller.busy
    status, view, _ = controller.read()
    assert status["state"] == "complete"
    assert status["mode"] == "simulation"
    assert status["flight_commands_sent"] == 0
    assert status["saved"]
    assert not (controller.output / "map.json").exists()
    assert json.loads((controller.output / "view.json").read_text())["source"] == "simulation"
    ply = (controller.output / "cloud.ply").read_text()
    assert "synthetic room survey" in ply
    assert "element vertex 1" in ply


def test_save_before_start_is_not_claimed(controller):
    with pytest.raises(ValueError, match="Start a mapping"):
        controller.request_save()
    assert not controller.read()[0].get("saved")


def test_live_saved_views_share_the_archived_map_revision(controller):
    """A newer map may be saved after the UI snapshot was constructed."""
    slam = room_mapping.mapping_import('slam')
    map_io = room_mapping.mapping_import('map_io')
    map_export = room_mapping.mapping_import('map_export')
    calibration = np.array([[1000., 0, 320], [0, 1000, 240], [0, 0, 1]])
    archived = slam.SLAMMap(calibration, next_keyframe_id=2, next_landmark_id=1)
    descriptor = np.zeros(32, dtype=np.uint8)
    for index, pixel in enumerate(([320., 240.], [120., 240.])):
        pose = np.eye(4)
        pose[0, 3] = -index
        archived.keyframes[index] = slam.Keyframe(
            index, index, float(index), pose, np.array([pixel]),
            descriptor[None, :].copy(), np.array([0]))
    archived.landmarks[0] = slam.Landmark(0, np.array([0., 0., 5.]), descriptor, {0: 0, 1: 0})
    controller._begin('live')
    controller.mapping = SimpleNamespace(save=lambda path: map_io.save_map(archived, path))
    old_view = dict(room_mapping.empty_view(), camera_position=[9., 8., 7.],
                    points=[{'id': 99, 'position': [100., 200., 300.]}],
                    keyframes=[{'id': 99, 'position': [8., 9., 10.]}], path=[[8., 9., 10.]])
    controller._save(old_view)
    saved_map = map_io.load_map(controller.output / 'map.json')
    expected = map_export.map_snapshot(saved_map)
    saved_view = json.loads((controller.output / 'view.json').read_text())
    assert len(expected['points']) == 1  # Independent triangulation fixture is exportable.
    assert saved_view['points'] == expected['points']
    assert saved_view['keyframes'] == expected['keyframes']
    assert saved_view['quality'] == expected['quality']
    assert saved_view['path'] == [[0., 0., 0.], [1., 0., 0.]]
    assert saved_view['camera_position'] is None
    assert saved_view['source'] == 'live'
    assert old_view['points'][0]['id'] == 99  # Saving cannot mutate the active view.
    ply = (controller.output / 'cloud.ply').read_text().split('end_header\n', 1)[1]
    assert [float(value) for value in ply.split()[:3]] == [0., 0., 5.]


def test_stop_state_survives_an_inflight_snapshot(controller, calibration_path, monkeypatch):
    session_class = room_mapping.mapping_import('mapping_session').MappingSession
    original_snapshot, original_stop = session_class.snapshot, session_class.stop
    snapshot_entered, snapshot_release = threading.Event(), threading.Event()
    stop_entered, stop_release = threading.Event(), threading.Event()

    def blocked_snapshot(self):
        snapshot_entered.set()
        if not snapshot_release.wait(4):
            raise TimeoutError('Snapshot gate was not released')
        return original_snapshot(self)

    def blocked_stop(self, *args, **kwargs):
        stop_entered.set()
        if not stop_release.wait(4):
            raise TimeoutError('Stop gate was not released')
        return original_stop(self, *args, **kwargs)

    monkeypatch.setattr(session_class, 'snapshot', blocked_snapshot)
    monkeypatch.setattr(session_class, 'stop', blocked_stop)
    receiver = ExistingReceiver()
    receiver.publish(1)
    controller.start_live(receiver, calibration_path)
    try:
        assert snapshot_entered.wait(2)
        controller.stop()
        snapshot_release.set()
        assert stop_entered.wait(2)
        status, _, _ = controller.read()
        assert status['state'] == 'stopping'
        assert not status['pose_valid']
        assert controller.busy
    finally:
        snapshot_release.set()
        stop_release.set()
        controller.thread.join(4)
