"""Estimated camera intrinsics stay explicit from selection to saved map.

Arrays are synthetic blank frames. These tests do not validate the actual DJI
stream's crop, lens distortion, SLAM accuracy, or any hardware command.
"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from follow_neo import room_mapping
from follow_neo.mini4_calibration import (
    calibration_info, estimate_mini4_calibration,
)
from test_room_mapping_controller import (
    ExistingReceiver, calibration_path, controller, no_network, wait_until,
)
from test_room_mapping_gui import window


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_log(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


@pytest.fixture
def estimated_path(tmp_path):
    path = tmp_path / 'mini4-estimated.json'
    estimate_mini4_calibration(160, 90, output_path=path)
    return path


@pytest.fixture
def short_controller(controller):
    # The staged project path is long; atomic calibration sidecars append a
    # random suffix and otherwise exceed legacy Windows MAX_PATH under tmp_path.
    output_root = Path(__file__).resolve().parents[1] / 'outputs'
    output_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='m4-', dir=output_root) as folder:
        controller.output_root = Path(folder) / 'maps'
        try:
            yield controller
        finally:
            controller.stop()
            if controller.thread:
                controller.thread.join(5)
                assert not controller.thread.is_alive()
            if controller.log:
                controller.log.close()


@pytest.fixture
def mapping_window(window, monkeypatch):
    root, app = window
    messages = {'error': [], 'info': [], 'warning': []}
    for kind, method in [('error', 'showerror'), ('info', 'showinfo'), ('warning', 'showwarning')]:
        monkeypatch.setattr('follow_neo.room_mapping_window.messagebox.' + method,
                            lambda title, message, _kind=kind, **kw: messages[_kind].append(message))
    app.open_room_mapping()
    mapping = app.room_mapping
    mapping.window.withdraw()
    return root, app, mapping, messages


def test_estimate_button_uses_existing_original_frame_and_persists_selection(mapping_window):
    root, app, mapping, messages = mapping_window
    receiver = ExistingReceiver()
    original = receiver.publish(19, shape=(270, 480, 3))
    mapping.receiver_provider = lambda: receiver
    assert mapping.panel.estimate_calibration_button.cget('text') == 'Estimate Mini 4 Pro'
    mapping.panel.estimate_calibration_button.invoke()
    assert not messages['error']
    path = Path(mapping.calibration_path)
    assert path.is_file()
    assert path.name == 'camera-mini4-pro-estimated.json'
    assert path.parent.parent == app.base / 'logs' / 'calibration'
    data = read_json(path)
    assert (data['width'], data['height']) == (480, 270)
    assert calibration_info(path)['kind'] == 'estimated'
    calibration = room_mapping.mapping_import('calibration').load_calibration(path)
    assert calibration.width == 480 and calibration.height == 270
    assert np.isfinite(calibration.K).all()
    assert receiver.frames.get() is original
    assert original.image.shape == (270, 480, 3) and not original.image.any()
    assert read_json(app.config_path)['room_calibration_path'] == str(path)
    assert app.room_calibration_path == str(path)
    assert 'ESTIMATED' in mapping.panel.calibration_path.get().upper()
    assert not mapping.controller.busy and mapping.controller.output is None
    assert app.session is None and app.manual is None and app.gimbal_keyboard is None


@pytest.mark.parametrize('kind', ['no_receiver', 'no_frame', 'stale', 'disconnected', 'aspect'])
def test_estimate_rejection_preserves_selected_calibration(mapping_window, calibration_path, kind):
    _, app, mapping, messages = mapping_window
    mapping._set_calibration(calibration_path)
    receiver = None if kind == 'no_receiver' else ExistingReceiver()
    if receiver is not None and kind != 'no_frame':
        receiver.publish(1, age=2.0 if kind == 'stale' else 0.,
                         shape=(120, 160, 3) if kind == 'aspect' else (90, 160, 3))
        if kind == 'disconnected':
            receiver.state = 'DISCONNECTED'
    mapping.receiver_provider = lambda: receiver
    mapping.estimate_calibration()
    assert messages['error']
    assert mapping.calibration_path == str(calibration_path)
    assert app.room_calibration_path == str(calibration_path)
    assert read_json(app.config_path)['room_calibration_path'] == str(calibration_path)
    assert not list((app.base / 'logs' / 'calibration').glob('*/camera-mini4-pro-estimated.json'))
    assert not mapping.controller.busy


@pytest.mark.parametrize('busy_source', ['mapping', 'measured_calibration'])
def test_busy_worker_prevents_estimate_replacement(mapping_window, calibration_path, busy_source):
    _, app, mapping, messages = mapping_window
    mapping._set_calibration(calibration_path)
    receiver = ExistingReceiver()
    receiver.publish(1, shape=(90, 160, 3))
    mapping.receiver_provider = lambda: receiver
    # No background work or control I/O is needed to exercise the busy gate.
    worker = SimpleNamespace(is_alive=lambda: True)
    if busy_source == 'mapping':
        mapping.controller.thread = worker
    else:
        mapping.calibration_thread = worker
    try:
        mapping.estimate_calibration()
        assert any(messages.values())
        assert mapping.calibration_path == str(calibration_path)
        assert not list((app.base / 'logs' / 'calibration').glob('*/camera-mini4-pro-estimated.json'))
    finally:
        if busy_source == 'mapping':
            mapping.controller.thread = None
        else:
            mapping.calibration_thread = None


def test_loading_existing_estimate_restores_notice_in_both_views(mapping_window, estimated_path, monkeypatch):
    _, app, mapping, messages = mapping_window
    monkeypatch.setattr('follow_neo.room_mapping_window.filedialog.askopenfilename',
                        lambda **kw: str(estimated_path))
    mapping.choose_calibration()
    app.refresh_mapping_advice()
    mapping.window.after_cancel(mapping._poll_id)
    mapping.poll()
    assert not messages['error']
    assert 'ESTIMATED' in mapping.panel.calibration_path.get().upper()
    for panel in (app.mapping_advice, mapping.panel.guidance_panel):
        assert 'estimated' in panel.calibration_text.get().lower()
    assert not mapping.controller.busy


@pytest.mark.parametrize('new_kind', ['measured', 'unknown'])
def test_calibration_notice_refreshes_even_when_advice_is_unchanged(window, new_kind):
    _, app = window
    app.open_room_mapping()
    app.room_mapping.window.withdraw()
    base = dict(state='idle', mode='live', calibration_selected=True)
    for panel in (app.mapping_advice, app.room_mapping.panel.guidance_panel):
        panel.set_status(dict(base, calibration_kind='estimated',
                              calibration_label='ESTIMATED Mini 4 Pro',
                              calibration_warnings=['No measured lens distortion.']))
        original_advice = panel.last_guidance.copy()
        assert 'estimated' in panel.calibration_text.get().lower()
        panel.set_status(dict(base, calibration_kind=new_kind,
                              calibration_label='Measured camera' if new_kind == 'measured' else 'Unknown camera',
                              calibration_warnings=[]))
        assert panel.last_guidance == original_advice
        assert 'estimated' not in panel.calibration_text.get().lower()


def test_estimated_provenance_survives_processing_logging_and_export(
        short_controller, estimated_path, calibration_path):
    controller = short_controller
    initial = read_json(estimated_path)
    receiver = ExistingReceiver()
    receiver.publish(7, shape=(90, 160, 3))
    controller.start_live(receiver, estimated_path)
    wait_until(lambda: controller.mapping.status()['processed_frames'] == 1)
    status, _, _ = controller.read()
    assert status['calibration_kind'] == 'estimated'
    assert status['calibration_warnings']
    assert 'estimated' in status['calibration_label'].lower()
    # The saved map must retain this run's camera, even if its source file changes.
    changed = dict(initial, K=[[500, 0, 80], [0, 500, 45], [0, 0, 1]])
    estimated_path.write_text(json.dumps(changed), encoding='utf-8')
    controller.stop()
    controller.thread.join(4)
    assert not controller.busy
    status, _, _ = controller.read()
    assert status['saved'] and status['flight_commands_sent'] == 0
    saved = read_json(controller.output / 'view.json')
    assert saved['units'] == 'arbitrary_monocular'
    sidecar = read_json(controller.output / 'calibration.json')
    start = next(row['data']['calibration'] for row in read_log(controller.output / 'mapping.jsonl')
                 if row['event'] == 'session_start')
    for metadata in (start, saved['calibration'], sidecar):
        assert metadata['calibration_kind'] == 'estimated'
        assert metadata['validated'] is False
        assert metadata['K'] == initial['K']
        assert (metadata['width'], metadata['height']) == (160, 90)
        assert metadata['provenance'] == initial['provenance']
    restored = room_mapping.mapping_import('calibration').load_calibration(controller.output / 'calibration.json')
    np.testing.assert_array_equal(restored.K, initial['K'])
    assert calibration_info(controller.output / 'calibration.json')['kind'] == 'estimated'
    samples = [row['data']['status'] for row in read_log(controller.output / 'mapping.jsonl')
               if row['event'] == 'status_sample']
    assert samples and all(row['calibration_kind'] == 'estimated' for row in samples)
    # Selecting the next run's camera must not relabel the completed map.
    old_log = (controller.output / 'mapping.jsonl').read_bytes()
    controller.set_calibration_preview(calibration_info(calibration_path))
    status, view, _ = controller.read()
    assert status['calibration_kind'] == 'estimated'
    assert view['calibration']['calibration_kind'] == 'estimated'
    assert (controller.output / 'mapping.jsonl').read_bytes() == old_log


def test_legacy_numeric_calibration_is_not_reported_as_measured(short_controller, calibration_path):
    controller = short_controller
    receiver = ExistingReceiver()
    receiver.publish(1)
    controller.start_live(receiver, calibration_path)
    wait_until(lambda: controller.mapping.status()['processed_frames'] == 1)
    status, _, _ = controller.read()
    assert status['calibration_kind'] == 'unknown'
    controller.stop()
    controller.thread.join(4)
    saved = read_json(controller.output / 'calibration.json')
    assert saved['validated'] is False
    assert saved['calibration_kind'] == 'unknown'


def test_measured_sidecar_preserves_classification(short_controller, calibration_path):
    # A serialization fixture with the application's measured-report markers;
    # it is not a fresh measurement or validation of a physical camera.
    report = read_json(calibration_path)
    report.update(success=True, provenance={
        'method': 'OpenCV calibrateCamera (5 distortion coefficients)',
        'image_source': 'original unresized stream frames',
        'metric_map_scale_established': False,
    })
    calibration_path.write_text(json.dumps(report), encoding='utf-8')
    assert calibration_info(calibration_path)['kind'] == 'measured'
    controller = short_controller
    receiver = ExistingReceiver()
    receiver.publish(1)
    controller.start_live(receiver, calibration_path)
    wait_until(lambda: controller.mapping.status()['processed_frames'] == 1)
    controller.stop()
    controller.thread.join(4)
    assert not controller.busy
    info = calibration_info(controller.output / 'calibration.json')
    assert info['kind'] == 'measured' and info['validated'] is True
    assert info['provenance'] == report['provenance']
    assert read_json(controller.output / 'view.json')['units'] == 'arbitrary_monocular'
