"""Reproducible hidden-GUI survey check; socket connections are forbidden."""
import json
from pathlib import Path
import time
import tkinter as tk
from unittest.mock import patch

from .gui import FollowLabWindow
from .mapping_panel import render_cloud_image


def run_check(base):
    base = Path(base)
    def forbidden(*args, **kwargs):
        raise AssertionError('Offline mapping check attempted a network connection')
    with patch('socket.create_connection', forbidden), patch('socket.socket.connect', forbidden):
        root = tk.Tk()
        root.withdraw()
        app = FollowLabWindow(root, base)
        mapping = None
        try:
            app.room_map_button.invoke()
            mapping = app.room_mapping
            mapping.window.withdraw()
            mapping.panel.start_simulation_button.invoke()
            deadline = time.monotonic() + 30
            while mapping.controller.busy and time.monotonic() < deadline:
                root.update()
                time.sleep(.02)
            status, view, _ = mapping.controller.read()
            if mapping.controller.busy:
                raise TimeoutError('Simulation did not finish within 30 seconds')
            assert status['state'] == 'complete', status
            assert view['source'] == 'simulation'
            assert app.session is None and app.manual is None
            assert view['flight_commands_sent'] == 0
            heights = [k['position'][2] for k in view['keyframes']]
            assert max(heights) - min(heights) > .1
            assert len(view['points']) > 100
            output = mapping.controller.output
            render_cloud_image(view, (1100, 700)).save(output / 'preview.png')
            report = dict(success=True, source='simulation', live_video_tested=False,
                          flight_commands_sent=0, steps=view['steps'], points=len(view['points']),
                          keyframes=len(view['keyframes']), simulated_coverage=view['coverage'],
                          altitude_span_simulated_m=max(heights)-min(heights),
                          output=str(output), gui_button_tested=True)
            (output / 'check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            return report
        finally:
            if mapping:
                mapping.controller.stop()
                if mapping.controller.thread:
                    mapping.controller.thread.join(30)
                if mapping._poll_id:
                    mapping.window.after_cancel(mapping._poll_id)
            root.destroy()
