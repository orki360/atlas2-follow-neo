"""Actual hidden GUI integration; every network connection is forbidden."""
import socket
from pathlib import Path
import tempfile
import time
import tkinter as tk

import pytest

from follow_neo.gui import FollowLabWindow


@pytest.fixture
def window(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('GUI mapping test attempted a network connection')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    # A nested pytest test name plus the app's run directory exceeds legacy
    # Windows MAX_PATH in this long staging workspace. Keep the fixture short.
    output_root = Path(__file__).resolve().parents[1] / 'outputs'
    output_root.mkdir(exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix='gui-', dir=output_root)
    test_root = Path(temporary.name).resolve()
    assert test_root.parent == output_root.resolve()
    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip('Hidden Tk unavailable: ' + str(error))
    root.withdraw()
    app = FollowLabWindow(root, test_root)
    try:
        yield root, app
    finally:
        mapping = app.room_mapping
        if mapping:
            mapping.controller.stop()
            if mapping.controller.thread:
                mapping.controller.thread.join(10)
            if mapping._poll_id:
                mapping.window.after_cancel(mapping._poll_id)
        try:
            root.destroy()
        except tk.TclError:
            pass
        temporary.cleanup()


def test_mapping_button_owns_window_without_video_or_control(window, monkeypatch):
    root, app = window
    app.room_map_button.invoke()
    app.room_mapping.window.withdraw()
    root.update_idletasks()
    assert app.session is None and app.manual is None
    assert app.room_mapping.panel.start_live_button.cget('text') == 'Start live mapping'
    errors = []
    monkeypatch.setattr('follow_neo.room_mapping_window.messagebox.showerror',
                        lambda title, message, **kwargs: errors.append(message))
    app.room_mapping.panel.start_live_button.invoke()
    assert 'calibration' in errors[-1].lower()
    assert not app.room_mapping.controller.busy


def test_autonomous_simulation_populates_actual_gui_and_saves(window):
    root, app = window
    app.open_room_mapping()
    mapping = app.room_mapping
    mapping.window.withdraw()
    mapping.panel.start_simulation_button.invoke()
    deadline = time.monotonic() + 15
    while mapping.controller.busy and time.monotonic() < deadline:
        root.update()
        time.sleep(.02)
    assert not mapping.controller.busy
    root.update()
    status, view, _ = mapping.controller.read()
    assert status['state'] == 'complete', status
    assert len(view['points']) > 100
    assert view['source'] == 'simulation'
    assert len({round(k['position'][2], 2) for k in view['keyframes']}) > 1
    assert 'SIMULATION' in mapping.panel.mode_badge.cget('text')
    assert (mapping.controller.output / 'cloud.ply').is_file()
    assert app.session is None and app.manual is None


def test_close_joins_mapping_before_destroying_root(window):
    root, app = window
    app.open_room_mapping()
    mapping = app.room_mapping
    mapping.window.withdraw()
    mapping.start_simulation()
    app.close()
    deadline = time.monotonic() + 10
    while mapping.controller.busy and time.monotonic() < deadline:
        root.update()
        time.sleep(.02)
    assert not mapping.controller.busy
    # Drive pending close callback once the simulation has stopped.
    while app.room_mapping is not None and time.monotonic() < deadline:
        root.update()
        time.sleep(.02)
    assert app.room_mapping is None
