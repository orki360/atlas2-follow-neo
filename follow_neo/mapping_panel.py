"""Read-only 3-D map panel; callbacks belong to the application controller.

This module has no connection to the aircraft, video receiver, or mapping worker.
All widget methods must be called on the Tk thread. Rendering helpers also work
without a window and make the same view available for inspection and tests.
"""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, ImageTk

from .mapping_guidance_panel import MappingGuidancePanel


BACKGROUND = '#0b141e'
TEXT = '#e2edf4'
MUTED = '#90a8b9'
CYAN = '#6ad5cb'
AMBER = '#f4c477'
DEFAULT_YAW = 35.0
DEFAULT_PITCH = 25.0
MAX_DISPLAY_POINTS = 3000
FLIGHT_REQUIREMENTS = (
    'Autonomous flight unavailable: calibration, metric scale and free-space '
    'validation required'
)


def _xyz(value):
    """Return a finite XYZ triplet, or None for invalid display data."""
    try:
        if isinstance(value, (str, bytes)) or len(value) != 3:
            return None
        result = tuple(float(v) for v in value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if all(math.isfinite(v) for v in result) else None


def _positions(records: Iterable, *, dictionaries=False, limit=None):
    positions = []
    for record in records:
        value = record.get('position') if dictionaries and isinstance(record, dict) else record
        point = _xyz(value)
        if point is not None:
            positions.append(point)
    if limit is not None and len(positions) > limit:
        # Evenly spaced indices keep the selection stable between redraws and
        # include both ends, unlike random sampling on each mouse movement.
        if limit == 1:
            positions = positions[:1]
        else:
            positions = [positions[i * (len(positions) - 1) // (limit - 1)]
                         for i in range(limit)]
    return positions


def prepare_cloud(snapshot, max_points=MAX_DISPLAY_POINTS):
    """Copy display data and bound points/paths without modifying the map."""
    if not isinstance(max_points, int) or isinstance(max_points, bool) or max_points < 1:
        raise ValueError('max_points must be a positive integer')
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    points = _positions(snapshot.get('points') or (), dictionaries=True)
    return {
        'points': _positions(points, limit=max_points),
        'total_points': len(points),
        'keyframes': _positions(snapshot.get('keyframes') or (), dictionaries=True, limit=1500),
        'path': _positions(snapshot.get('path') or (), limit=1000),
        'camera_position': _xyz(snapshot.get('camera_position')),
        'target_position': _xyz(snapshot.get('target_position')),
        'source': 'simulation' if snapshot.get('source') == 'simulation' else 'live',
        'units': str(snapshot.get('units') or 'arbitrary_monocular'),
    }


def _bounds(points):
    if not points:
        return (0.0, 0.0, 0.0), 1.0
    low = [min(p[i] for p in points) for i in range(3)]
    high = [max(p[i] for p in points) for i in range(3)]
    center = tuple(low[i] / 2 + high[i] / 2 for i in range(3))
    radius = max(max(abs(p[i] / 2 - center[i] / 2) * 2
                     for i in range(3)) for p in points)
    return center, max(radius, 1e-6)


def _rotate(point, yaw, pitch):
    yaw, pitch = math.radians(yaw), math.radians(pitch)
    x, y, z = point
    horizontal = math.cos(yaw) * x - math.sin(yaw) * y
    forward = math.sin(yaw) * x + math.cos(yaw) * y
    return (horizontal, math.sin(pitch) * forward - math.cos(pitch) * z,
            math.cos(pitch) * forward + math.sin(pitch) * z)


def project_points(points, size=(900, 500), yaw=DEFAULT_YAW,
                   pitch=DEFAULT_PITCH, zoom=1.0, *, center=None, radius=None):
    """Orthographically project finite XYZ to (screen_x, screen_y, depth).

    Screen Y grows downwards. Yaw rotates about Z and pitch rotates the view;
    no world axis is assumed to be gravity. Invalid input points are omitted.
    Pass the same center/radius when projecting separate parts of one scene.
    """
    valid = _positions(points)
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError('image size must be positive')
    if not all(math.isfinite(float(v)) for v in (yaw, pitch, zoom)) or zoom <= 0:
        raise ValueError('view angles and positive zoom must be finite')
    fitted_center, fitted_radius = _bounds(valid)
    center = fitted_center if center is None else _xyz(center)
    radius = fitted_radius if radius is None else float(radius)
    if center is None or not math.isfinite(radius) or radius <= 0:
        raise ValueError('view center and positive radius must be finite')
    scale = min(width, height) * .32 * zoom
    projected = []
    for point in valid:
        # Divide first to avoid overflow for widely separated finite values.
        normalized = tuple(point[i] / radius - center[i] / radius for i in range(3))
        x, y, depth = _rotate(normalized, yaw, pitch)
        if all(math.isfinite(v) for v in (x, y, depth)):
            projected.append((width / 2 + scale * x, height / 2 + scale * y, depth))
    return projected


def _font(size=13):
    for name in ('segoeui.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _units_label(cloud):
    if cloud['source'] == 'simulation':
        return 'Synthetic meters'
    if cloud['units'] in ('m', 'meters', 'metres', 'metric'):
        return 'Meters (reported by map source)'
    return 'Arbitrary monocular units - scale is not meters'


def render_cloud_image(snapshot, size=(900, 500), yaw=DEFAULT_YAW,
                       pitch=DEFAULT_PITCH, zoom=1.0,
                       max_points=MAX_DISPLAY_POINTS):
    """Render the rotatable 3-D map without Tk, network or hardware access."""
    width, height = (int(v) for v in size)
    if width <= 0 or height <= 0:
        raise ValueError('image size must be positive')
    cloud = prepare_cloud(snapshot, max_points)
    fit = cloud['points'] + cloud['keyframes'] + cloud['path']
    fit += [p for p in (cloud['camera_position'], cloud['target_position']) if p is not None]
    center, radius = _bounds(fit)
    project = lambda points: project_points(points, (width, height), yaw, pitch, zoom,
                                            center=center, radius=radius)
    image = Image.new('RGB', (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    font, small = _font(13), _font(11)

    # A faint local coordinate reference, not an occupancy/floor estimate.
    for i in range(-4, 5):
        value = i * radius / 4
        lines = (
            [(center[0] - radius, center[1] + value, center[2]),
             (center[0] + radius, center[1] + value, center[2])],
            [(center[0] + value, center[1] - radius, center[2]),
             (center[0] + value, center[1] + radius, center[2])],
        )
        for line in lines:
            screen = project(line)
            if len(screen) == 2:
                draw.line([p[:2] for p in screen], fill='#152634', width=1)

    points = project(cloud['points'])
    for x, y, depth in sorted(points, key=lambda p: p[2]):
        if -3 <= x <= width + 3 and -3 <= y <= height + 3:
            shade = max(100, min(230, int(170 + depth * 30)))
            draw.ellipse((x-1.7, y-1.7, x+1.7, y+1.7), fill=(66, shade, 199))

    trajectory = project(cloud['keyframes'])
    if len(trajectory) > 1:
        draw.line([p[:2] for p in trajectory], fill='#679cdf', width=2)
    for x, y, _ in trajectory:
        draw.ellipse((x-2, y-2, x+2, y+2), fill='#8ab5ef')

    planned = project(cloud['path'])
    if len(planned) > 1:
        draw.line([p[:2] for p in planned], fill=AMBER, width=2)
    for position, label, color, offset in (
            (cloud['target_position'], 'Next viewpoint', AMBER, -28),
            (cloud['camera_position'], 'Camera', '#ffffff', 12)):
        if position is not None:
            screen = project([position])
            if screen:
                x, y, _ = screen[0]
                draw.polygon([(x,y-7), (x+7,y), (x,y+7), (x-7,y)], outline=color, width=2)
                text_position = (x+10,y+offset)
                bounds = draw.textbbox(text_position,label,font=small)
                draw.rectangle((bounds[0]-3,bounds[1]-2,bounds[2]+3,bounds[3]+2),fill=BACKGROUND)
                draw.text(text_position, label, fill=color, font=small)

    source = 'SIMULATION - synthetic room' if cloud['source'] == 'simulation' else 'LIVE SPARSE MAP'
    draw.text((16,12), source, fill=AMBER if cloud['source'] == 'simulation' else CYAN, font=font)
    draw.text((16,33), _units_label(cloud), fill=MUTED, font=small)
    total = snapshot.get('_display_total_points', cloud['total_points']) if isinstance(snapshot,dict) else cloud['total_points']
    if not isinstance(total,int) or total < cloud['total_points']:
        total = cloud['total_points']
    draw.text((16,52), f"{len(cloud['points']):,} displayed / {total:,} finite points", fill=MUTED, font=small)
    if not cloud['points']:
        draw.text((width / 2, height / 2), 'Waiting for 3D map points',
                  fill=MUTED, font=font, anchor='mm')

    # Screen-corner triad always remains visible while rotating and zooming.
    origin = (52, height-52)
    for vector, label, color in (((1,0,0),'X','#f58b88'), ((0,1,0),'Y','#82d29b'), ((0,0,1),'Z','#7baded')):
        x, y, _ = _rotate(vector, yaw, pitch)
        endpoint = (origin[0]+32*x, origin[1]+32*y)
        draw.line((origin, endpoint), fill=color, width=2)
        draw.text((endpoint[0]+3,endpoint[1]-7), label, fill=color, font=small)
    draw.text((100,height-28), 'Drag to rotate  |  Wheel to zoom  |  Double-click to reset', fill=MUTED, font=small)
    return image


class MappingPanel(ttk.Frame):
    """Resizable map view with explicit live/simulated mode and no flight API."""
    def __init__(self, parent, *, on_start_live, on_start_simulation, on_stop,
                 on_save, on_calibrate, on_choose_calibration, on_estimate_calibration=None):
        super().__init__(parent, padding=10)
        self._snapshot = {}
        self._photo = None
        self._redraw_id = None
        self._drag_position = None
        self._destroyed = False
        self.yaw = DEFAULT_YAW
        self.pitch = DEFAULT_PITCH
        self.zoom = 1.0
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)

        buttons = ttk.Frame(self)
        buttons.grid(row=0,column=0,sticky='ew',pady=(0,8))
        self.start_live_button = ttk.Button(buttons,text='Start live mapping',command=on_start_live)
        self.start_simulation_button = ttk.Button(buttons,text='Autonomous simulation',command=on_start_simulation)
        self.stop_button = ttk.Button(buttons,text='Stop',command=on_stop)
        self.save_button = ttk.Button(buttons,text='Save map',command=on_save)
        for button in (self.start_live_button,self.start_simulation_button,self.stop_button,self.save_button):
            button.pack(side='left',padx=(0,8))
        ttk.Button(buttons,text='Reset view',command=self.reset_view).pack(side='right')

        calibration = ttk.Frame(self)
        calibration.grid(row=1,column=0,sticky='ew',pady=(0,8))
        calibration.columnconfigure(0,weight=1)
        self.calibration_path = tk.StringVar(value='Calibration: not selected')
        self.calibration_label = ttk.Label(calibration,textvariable=self.calibration_path,foreground=MUTED)
        self.calibration_label.grid(row=0,column=0,sticky='w')
        self.choose_calibration_button = ttk.Button(calibration,text='Choose calibration',command=on_choose_calibration)
        self.choose_calibration_button.grid(row=0,column=1,padx=6)
        self.calibrate_button = ttk.Button(calibration,text='Calibrate from images',command=on_calibrate)
        self.calibrate_button.grid(row=0,column=2)
        self.estimate_calibration_button = ttk.Button(calibration,text='Estimate Mini 4 Pro',
                                                      command=on_estimate_calibration)
        self.estimate_calibration_button.grid(row=0,column=3,padx=(6,0))

        self.mode_badge = tk.Label(self,text='LIVE SPARSE MAP  |  Arbitrary monocular units',
                                  bg='#164d49',fg=TEXT,font=('Segoe UI',11,'bold'),anchor='w',padx=10,pady=6)
        self.mode_badge.grid(row=2,column=0,sticky='ew')
        self.mode_description = tk.StringVar(value=FLIGHT_REQUIREMENTS)
        self.description_label = ttk.Label(self,textvariable=self.mode_description,foreground=AMBER,wraplength=860)
        self.description_label.grid(row=3,column=0,sticky='ew',pady=(5,8))

        self.canvas = tk.Canvas(self,width=900,height=340,bg=BACKGROUND,highlightthickness=0)
        self.canvas.grid(row=4,column=0,sticky='nsew')
        self.canvas.bind('<Configure>',self._configure)
        self.canvas.bind('<ButtonPress-1>',self._drag_start)
        self.canvas.bind('<B1-Motion>',self._drag)
        self.canvas.bind('<ButtonRelease-1>',self._drag_end)
        self.canvas.bind('<MouseWheel>',self._wheel)
        self.canvas.bind('<Button-4>',self._wheel)
        self.canvas.bind('<Button-5>',self._wheel)
        self.canvas.bind('<Double-Button-1>',lambda event:self.reset_view())

        self.status_text = tk.StringVar(value='IDLE | Waiting for map | Pose unavailable')
        self.status_label = ttk.Label(self,textvariable=self.status_text,foreground=TEXT,wraplength=860)
        self.status_label.grid(row=5,column=0,sticky='ew',pady=(8,2))
        self.message_text = tk.StringVar(value='Live mapping uses the existing video frames. Simulation selects its own 3D viewpoints.')
        self.message_label = ttk.Label(self,textvariable=self.message_text,foreground=MUTED,wraplength=860)
        self.message_label.grid(row=6,column=0,sticky='ew')
        self.log_text = tk.StringVar(value='Mapping log: created automatically when a run starts')
        self.log_label = ttk.Label(self,textvariable=self.log_text,foreground=MUTED,wraplength=860)
        self.log_label.grid(row=7,column=0,sticky='ew',pady=(3,0))
        self.guidance_panel = MappingGuidancePanel(self)
        self.guidance_panel.grid(row=8,column=0,sticky='ew',pady=(6,0))
        self.bind('<Destroy>',self._on_destroy,add='+')
        self._schedule_redraw()

    def set_calibration_path(self, path, info=None):
        kind = (info or {}).get('kind', 'unknown')
        tag = 'ESTIMATED; 16:9, 1x, standard lens' if kind == 'estimated' else kind.upper()
        self.calibration_path.set(f'Calibration [{tag}]: {path}' if path else 'Calibration: not selected')
        self.calibration_label.configure(foreground=AMBER if path and kind != 'measured' else MUTED)

    def set_busy(self, busy):
        """Disable starts/calibration while a run is active; Stop stays usable."""
        state = 'disabled' if busy else 'normal'
        for button in (self.start_live_button,self.start_simulation_button,
                       self.choose_calibration_button,self.calibrate_button,self.estimate_calibration_button):
            button.configure(state=state)

    def _set_mode(self, source):
        if source == 'simulation':
            self.mode_badge.configure(text='SIMULATION  |  Synthetic room / meters',bg='#725325')
            self.mode_description.set('Simulator automatically selects 3D viewpoints; the displayed room and motion are synthetic.')
        else:
            self.mode_badge.configure(text='LIVE SPARSE MAP  |  Arbitrary monocular units',bg='#164d49')
            self.mode_description.set(FLIGHT_REQUIREMENTS)

    def set_snapshot(self, snapshot):
        # Freeze just display fields, rather than retaining a worker's mutable map.
        cloud = prepare_cloud(snapshot)
        self._snapshot = {
            'points': [{'position':p} for p in cloud['points']],
            'keyframes': [{'position':p} for p in cloud['keyframes']],
            'path': cloud['path'], 'camera_position': cloud['camera_position'],
            'target_position': cloud['target_position'], 'source': cloud['source'],
            'units': cloud['units'],
            '_display_total_points': cloud['total_points'],
        }
        self._set_mode(cloud['source'])
        self._schedule_redraw()

    def set_status(self, status):
        status = status if isinstance(status,dict) else {}
        fields = [str(status.get('state','IDLE')),
                  f"Points: {status.get('landmarks',0)}",
                  f"Keyframes: {status.get('keyframes',0)}",
                  'Pose valid' if status.get('pose_valid') else 'Pose unavailable']
        if status.get('calibration_kind'):
            fields.append('Run calibration: ' + str(status['calibration_kind']).upper())
        coverage = status.get('coverage')
        if isinstance(coverage,(int,float)) and math.isfinite(coverage):
            fields.append(f'Simulated coverage: {coverage:.1%}')
        capacity_reason = status.get('mapping_capacity_reason')
        message = str(status.get('message',''))
        if capacity_reason:
            fields.append('Map capacity reached')
            message += (' | Map additions limited: ' + str(capacity_reason)
                        + '. Existing map retained; tracking status is shown separately.')
        self.status_text.set(' | '.join(fields))
        self.message_text.set(message)
        log_path, log_error = status.get('log_path'), status.get('log_error')
        if log_error:
            self.log_text.set('Mapping log ERROR: ' + str(log_error) + ' | ' + str(log_path or ''))
        elif log_path:
            label = 'Mapping log (closed): ' if status.get('log_closed') else 'Mapping log: '
            self.log_text.set(label + str(log_path))
        else:
            self.log_text.set('Mapping log: created automatically when a run starts')
        self.log_label.configure(foreground=AMBER if log_error else MUTED)
        mode = status.get('mode')
        if mode in ('live','simulation'):
            self._set_mode(mode)
        self.guidance_panel.set_status(status, self._snapshot)

    def reset_view(self):
        self.yaw,self.pitch,self.zoom = DEFAULT_YAW,DEFAULT_PITCH,1.0
        self._schedule_redraw()

    def _configure(self,event):
        width = max(200,event.width-20)
        for label in (self.description_label,self.status_label,self.message_label,self.log_label):
            label.configure(wraplength=width)
        # Elide long absolute calibration paths rather than stretching the window.
        self.calibration_label.configure(wraplength=max(140,width-530))
        self._schedule_redraw()

    def _drag_start(self,event):
        self._drag_position = (event.x,event.y)

    def _drag(self,event):
        if self._drag_position is not None:
            x,y = self._drag_position
            self.yaw = (self.yaw+(event.x-x)*.5)%360
            self.pitch = max(-85,min(85,self.pitch+(event.y-y)*.5))
            self._drag_position = (event.x,event.y)
            self._schedule_redraw()

    def _drag_end(self,event):
        self._drag_position = None

    def _wheel(self,event):
        number = getattr(event,'num',None)
        delta = getattr(event,'delta',0)
        direction = 1 if number == 4 or delta > 0 else -1 if number == 5 or delta < 0 else 0
        if direction:
            self.zoom = max(.15,min(8.0,self.zoom*(1.15**direction)))
            self._schedule_redraw()
        return 'break'

    def _schedule_redraw(self):
        if not self._destroyed and self._redraw_id is None:
            self._redraw_id = self.after_idle(self._redraw)

    def _redraw(self):
        self._redraw_id = None
        if self._destroyed:
            return
        size = (max(100,self.canvas.winfo_width()),max(100,self.canvas.winfo_height()))
        self._photo = ImageTk.PhotoImage(render_cloud_image(self._snapshot,size,self.yaw,self.pitch,self.zoom),master=self)
        self.canvas.delete('map')
        self.canvas.create_image(0,0,anchor='nw',image=self._photo,tags='map')

    def _on_destroy(self,event):
        if event.widget is self:
            self._destroyed = True
            if self._redraw_id is not None:
                self.after_cancel(self._redraw_id)
                self._redraw_id = None
            self._photo = None
