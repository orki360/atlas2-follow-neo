"""Explicit passive loop opt-in and logging; no control app or hardware."""
import json
import time
import tkinter as tk
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from follow_neo import room_mapping
from follow_neo.mapping_panel import MappingPanel
from follow_neo.room_mapping_window import RoomMappingWindow
from test_room_mapping_controller import (
    ExistingReceiver, calibration_path, controller, no_network, wait_until,
)


@pytest.mark.parametrize('enabled', [False, True])
def test_existing_frame_path_uses_requested_flag_and_logs_actual_engine_status(
        controller, calibration_path, enabled):
    receiver = ExistingReceiver()
    receiver.publish(1)
    if enabled:
        controller.start_live(receiver, calibration_path, loop_closure=True)
    else:
        # The original paced replay API must keep loop closure disabled.
        controller.start_live(receiver, calibration_path)
    wait_until(lambda: controller.mapping.status()['processed_frames'] == 1)
    status, _, _ = controller.read()
    loop = status['loop_closure']
    assert loop['enabled'] is enabled and loop['active'] is enabled
    assert loop['available'] and loop['accepted'] == loop['candidates'] == 0
    assert loop['last_event'] is None
    assert controller.mapping._config.loop_closure_enabled is enabled
    assert not controller.mapping._config.map_maintenance_enabled
    controller.stop()
    controller.thread.join(5)
    assert not controller.busy
    assert controller.read()[0]['state'] == 'stopped'
    assert controller.read()[0]['saved']
    records = [json.loads(line) for line in (controller.output / 'mapping.jsonl').read_text().splitlines()]
    assert records[0]['data']['slam_config']['loop_closure_enabled'] is enabled
    samples = [row['data']['status']['loop_closure'] for row in records
               if row['event'] == 'status_sample']
    assert samples and all(item['enabled'] is enabled for item in samples)
    assert records[-1]['data']['loop_closure']['accepted'] == 0
    assert receiver.state == 'STREAMING'


@pytest.mark.parametrize('unsupported', ['configuration', 'cached_status'])
def test_explicit_loop_request_rejects_old_engine_before_starting(
        controller, calibration_path, monkeypatch, unsupported):
    receiver = ExistingReceiver()
    receiver.publish(1)
    original_import = room_mapping.mapping_import

    @dataclass
    class LegacyConfig:
        retain_map: bool = True

    def engine_import(name):
        if name == 'slam' and unsupported == 'configuration':
            return SimpleNamespace(SLAMConfig=LegacyConfig)
        if name == 'mapping_session' and unsupported == 'cached_status':
            return SimpleNamespace(MappingSession=type('LegacySession', (), {}))
        return original_import(name)

    monkeypatch.setattr(room_mapping, 'mapping_import', engine_import)
    with pytest.raises(RuntimeError, match='Update the engine'):
        controller.start_live(receiver, calibration_path, loop_closure=True)
    assert controller.mapping is None and not controller.busy
    assert not controller.output_root.exists()


def test_new_loop_counts_trigger_sample_inside_normal_sampling_interval(controller, monkeypatch):
    # Deliberate synthetic decision data tests log transport, not loop geometry.
    controller._begin('live')
    base = dict(state='running', mode='live', pose_valid=False, landmarks=0, keyframes=0,
                loop_closure=dict(enabled=True, accepted=0, rejected=0, queries=0,
                                  candidates=0, last_event=None, last_accepted_event=None))
    monkeypatch.setattr(controller, 'read', lambda: (base, {}, 1))
    controller._log_sample()
    controller._last_log_at = time.monotonic() + 100
    accepted = dict(reason='accepted', current_id=10, candidate_id=1)
    base['loop_closure'].update(accepted=1, candidates=1, queries=1,
                                last_event=accepted, last_accepted_event=accepted)
    controller._log_sample()
    base['loop_closure'].update(rejected=1, candidates=2, queries=2,
                                last_event=dict(reason='geometry_rejected', current_id=11))
    controller._last_log_at = time.monotonic() + 100
    controller._log_sample()
    controller._finish_log()
    records = [json.loads(line) for line in (controller.output / 'mapping.jsonl').read_text().splitlines()]
    counts = [row['data']['status']['loop_closure']['candidates'] for row in records
              if row['event'] == 'status_sample']
    assert counts[:3] == [0, 1, 2]
    assert records[-1]['data']['loop_closure']['last_accepted_event'] == accepted


@pytest.fixture
def root():
    try:
        window = tk.Tk()
    except tk.TclError as error:
        pytest.skip('Hidden Tk unavailable: ' + str(error))
    window.withdraw()
    yield window
    window.destroy()


def test_checkbox_is_explicit_locked_during_run_and_displays_accepted_count(root):
    panel = MappingPanel(root, on_start_live=lambda: None, on_start_simulation=lambda: None,
                         on_stop=lambda: None, on_save=lambda: None,
                         on_calibrate=lambda: None, on_choose_calibration=lambda: None)
    assert not panel.loop_closure_enabled.get()
    panel.loop_closure_check.invoke()
    assert panel.loop_closure_enabled.get()
    panel.set_busy(True)
    assert 'disabled' in panel.loop_closure_check.state()
    panel.loop_closure_check.invoke()
    assert panel.loop_closure_enabled.get()
    panel.set_status(dict(state='running', mode='live',
                          loop_closure=dict(enabled=True, accepted=2)))
    assert 'Loops accepted (total): 2 (experimental)' in panel.status_text.get()
    panel.set_busy(False)
    assert 'disabled' not in panel.loop_closure_check.state()
    panel.set_status(dict(state='stopped', mode='live', loop_closure=dict(enabled=False)))
    assert 'Loop closure: off' in panel.status_text.get()
    panel.destroy()


def test_room_window_checkbox_reaches_real_passive_controller(root, tmp_path, calibration_path):
    receiver = ExistingReceiver()
    receiver.publish(1)
    window = RoomMappingWindow(root, tmp_path, lambda: receiver,
                               calibration_path=str(calibration_path))
    window.window.withdraw()
    try:
        window.panel.loop_closure_check.invoke()
        window.panel.start_live_button.invoke()
        wait_until(lambda: window.controller.mapping is not None
                   and window.controller.mapping.status()['processed_frames'] == 1)
        assert window.controller.mapping.status()['loop_closure']['enabled']
        assert window.controller.mapping._config.loop_closure_enabled
        window.poll()
        assert 'Loops accepted (total): 0' in window.panel.status_text.get()
    finally:
        window.controller.stop()
        if window.controller.thread:
            window.controller.thread.join(5)
            assert not window.controller.busy
        if window._poll_id:
            window.window.after_cancel(window._poll_id)
        window.window.destroy()
