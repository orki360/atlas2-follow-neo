"""Measured advice and keyboard routing in the actual hidden GUI; no hardware."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from follow_neo.gui import normalized_event_key
from test_room_mapping_gui import window


def event(app, key, code=None, widget=None):
    return SimpleNamespace(keysym=key, keycode=code, widget=widget or app.canvas)


class Heading:
    def __init__(self):
        self.requests = []

    def gimbal_available(self):
        return True

    def queue_gimbal_step(self, request):
        self.requests.append(request)
        return True


def enable_gimbal(app):
    heading = Heading()
    app.session = SimpleNamespace(heading=heading, record_control=lambda *args: None)
    app.key_state_reader = None
    app.gimbal_enabled.set(True)
    app.toggle_gimbal_keyboard()
    assert app.gimbal_keyboard.read()['enabled']
    assert not heading.requests  # Enabling is not a physical action.
    return heading


def test_guidance_visible_in_both_windows_without_a_control_connection(window, monkeypatch):
    root, app = window
    app.open_room_mapping()
    app.room_mapping.window.withdraw()
    status = dict(state='running',mode='live',tracking_state='lost',pose_valid=False,
                  frame_age_seconds=.1,diagnostics_fresh=True,num_matches=0,num_inliers=0,
                  last_processed_source_frame_id=8,diagnostics=dict(num_features=7,feature_occupied_cells=1))
    monkeypatch.setattr(app.room_mapping.controller,'read',lambda:(status.copy(),{},2))
    app.refresh_mapping_advice()
    app.room_mapping.window.after_cancel(app.room_mapping._poll_id)
    app.room_mapping.poll()
    assert 'texture' in app.mapping_advice.title_text.get()
    assert 'Features: 7' in app.mapping_advice.metrics_text.get()
    assert 'Room coverage: unknown' in app.mapping_advice.metrics_text.get()
    assert app.room_mapping.panel.guidance_panel.last_guidance['code'] == 'low_features'
    assert app.manual is None
    assert app.gimbal_keyboard is None
    app.focus_pilot_video()
    assert app.room_mapping.controller.stop_event.is_set() is False


def test_stale_guidance_replaces_previous_direction(window):
    _, app = window
    state = dict(state='running',mode='live',tracking_state='tracking',pose_valid=True,
                 frame_age_seconds=.1,diagnostics_fresh=True,num_matches=150,num_inliers=110,
                 diagnostics=dict(num_features=300,feature_occupied_cells=6,
                                  feature_grid=[[0,0,0],[20,20,20],[60,60,60]]))
    app.mapping_advice.set_status(state)
    assert app.mapping_advice.last_guidance['direction'] == 'up'
    state.update(frame_age_seconds=2.,pose_valid=False,diagnostics_fresh=False)
    app.mapping_advice.set_status(state)
    assert app.mapping_advice.last_guidance['direction'] == 'neutral'
    assert 'fresh' in app.mapping_advice.action_text.get()


def test_gimbal_keypresses_are_separate_from_flight_and_repeat_requires_release(window):
    _, app = window
    heading = enable_gimbal(app)
    app.manual = Mock(mode='MANUAL')
    app.key_press(event(app,'i'))
    app.key_press(event(app,'i'))
    assert len(heading.requests) == 1
    assert heading.requests[0].direction == 1
    app.manual.update.assert_not_called()
    heading.requests[0].finish('acknowledged','Fake pitch acknowledged')
    # Enabling flight keyboard clears flight keys; held I must still be latched.
    app.manual.thread.is_alive.return_value = True
    app.manual.stop_event.is_set.return_value = False
    app.key_press(event(app,'e'))
    app.key_press(event(app,'i'))
    assert len(heading.requests) == 1
    app.key_release(event(app,'i'))
    app.key_press(event(app,'i'))
    assert len(heading.requests) == 2
    app.release_manual()
    assert not app.gimbal_keyboard.read()['enabled']
    assert heading.requests[-1].phase == 'done'


def test_gimbal_respects_text_focus_and_hebrew_physical_keycodes(window):
    import tkinter as tk
    root, app = window
    heading = enable_gimbal(app)
    entry = tk.Entry(root)
    app.key_press(event(app,'i',widget=entry))
    assert not heading.requests
    assert normalized_event_key(event(app,'ן',73),'win32') == 'i'
    assert normalized_event_key(event(app,'ל',75),'win32') == 'k'
    app.key_press(event(app,'ל',75))
    assert heading.requests[0].direction == -1
    app.disable_gimbal_keyboard()


def test_focus_loss_disables_gimbal_and_cancels_unsent_step(window, monkeypatch):
    _, app = window
    heading = enable_gimbal(app)
    app.key_press(event(app,'i'))
    monkeypatch.setattr('follow_neo.gui.safe_focus_widget',lambda root:None)
    callbacks = []
    monkeypatch.setattr(app.root,'after_idle',lambda callback:callbacks.append(callback) or 'test-after')
    app.focus_out(SimpleNamespace())
    callbacks[-1]()
    assert not app.gimbal_keyboard.read()['enabled']
    assert heading.requests[0].phase == 'done'


def test_gimbal_cannot_be_enabled_during_automatic_control(window):
    _, app = window
    heading = enable_gimbal(app)
    app.disable_gimbal_keyboard()
    app.dance_selected = True
    app.gimbal_enabled.set(True)
    app.toggle_gimbal_keyboard()
    app.key_press(event(app,'k'))
    assert not app.gimbal_enabled.get()
    assert not heading.requests


def test_enabling_with_a_held_gimbal_key_requires_a_fresh_press(window):
    _, app = window
    heading = enable_gimbal(app)
    app.disable_gimbal_keyboard()
    app.key_state_reader = lambda keys: {'i'} & keys
    app.gimbal_enabled.set(True)
    app.toggle_gimbal_keyboard()
    app.key_press(event(app,'i'))
    assert not heading.requests
    app.key_state_reader = lambda keys: set()
    app.reconcile_keyboard()
    app.key_press(event(app,'i'))
    assert len(heading.requests) == 1
    app.disable_gimbal_keyboard()
