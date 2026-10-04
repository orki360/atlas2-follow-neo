"""Display-only advice shared by the map window and the pilot's video view."""
import tkinter as tk
from tkinter import ttk

from .mapping_guidance import build_mapping_guidance


KEYBOARD_LEGEND = ('W/S: height | A/D: yaw | Arrows: forward/back/left/right | '
                   'I/K: camera pitch (enable separately) | Q/Esc: release')


class MappingGuidancePanel(ttk.LabelFrame):
    def __init__(self, parent, *, compact=False):
        super().__init__(parent, text='Mapping assistant | manual pilot', padding=(8, 5))
        self.compact = compact
        self.last_guidance = None
        self.title_text = tk.StringVar()
        self.action_text = tk.StringVar()
        self.reason_text = tk.StringVar()
        self.metrics_text = tk.StringVar()
        self.missing_text = tk.StringVar()
        self.reference_text = tk.StringVar()
        self.calibration_text = tk.StringVar()
        self.labels = []
        for variable, color, bold in (
                (self.title_text, '#6ad5cb', True),
                (self.action_text, '#f4c477', True),
                (self.reason_text, '#e2edf4', False),
                (self.metrics_text, '#90a8b9', False),
                (self.calibration_text, '#f4c477', False)):
            label = ttk.Label(self, textvariable=variable, foreground=color,
                              font=('Segoe UI', 10, 'bold' if bold else 'normal'),
                              wraplength=800, justify='left')
            label.pack(fill='x')
            self.labels.append(label)
        if not compact:
            for variable in (self.missing_text, self.reference_text):
                label = ttk.Label(self, textvariable=variable, foreground='#90a8b9',
                                  wraplength=900, justify='left')
                label.pack(fill='x', pady=(2, 0))
                self.labels.append(label)
        self.bind('<Configure>', self._resize)
        self.set_status({'state': 'idle', 'mode': 'live'})

    def _resize(self, event):
        if event.widget is self:
            for label in self.labels:
                label.configure(wraplength=max(160, event.width-22))

    def set_status(self, status, view=None):
        status = status if isinstance(status, dict) else {}
        kind = status.get('calibration_kind')
        if status.get('mode') == 'simulation':
            self.calibration_text.set('Calibration: synthetic simulation')
        elif kind == 'estimated':
            self.calibration_text.set('Calibration: ESTIMATED | 16:9, 1x, standard lens; accuracy unverified')
        elif kind == 'measured':
            self.calibration_text.set('Calibration: measured board fit; real mapping accuracy unverified')
        elif kind == 'unknown':
            self.calibration_text.set('Calibration: supplied file; measurement provenance unknown')
        else:
            self.calibration_text.set('')
        advice = build_mapping_guidance(status, view)
        if advice == self.last_guidance:
            return
        self.last_guidance = advice
        direction = advice.get('direction', 'neutral')
        self.title_text.set(advice['title'] + (' | '+direction.upper() if direction != 'neutral' else ''))
        self.action_text.set(advice['action'])
        self.reason_text.set(advice['reason'])
        metrics = advice.get('metrics') or {}
        values = []
        for label, value in (('Features', metrics.get('num_features')),
                             ('Matches', metrics.get('num_matches')),
                             ('Inliers', metrics.get('num_inliers'))):
            values.append(f'{label}: {value if value is not None else "--"}')
        occupied = metrics.get('feature_occupied_cells')
        values.append(f'Image cells: {occupied}/9' if occupied is not None else 'Image cells: --')
        self.metrics_text.set('Last processed frame | ' + ' | '.join(values) + ' | Room coverage: unknown')
        self.missing_text.set('Still needed: ' + '; '.join(advice.get('missing') or ['No new mapping issue detected']))
        self.reference_text.set(advice.get('reference', ''))
