"""Tk ownership and background task coordination for the 3D room map."""
from datetime import datetime
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import uuid

from .mapping_panel import MappingPanel
from .room_mapping import RoomMappingController
from .mini4_calibration import calibration_info, estimate_mini4_calibration


class RoomMappingWindow:
    def __init__(self, root, base, receiver_provider, *, calibration_path='', on_calibration_changed=None,
                 on_focus_video=None):
        self.base = Path(base)
        self.receiver_provider = receiver_provider
        self.calibration_path = calibration_path
        self.on_calibration_changed = on_calibration_changed or (lambda path: None)
        self.controller = RoomMappingController(self.base / 'logs' / 'room_mapping')
        self.window = tk.Toplevel(root)
        self.window.title('3D Room Mapping | Mini 4 Pro')
        self.window.geometry('1120x780')
        self.window.minsize(850, 600)
        self.window.protocol('WM_DELETE_WINDOW', self.hide)
        self.panel = MappingPanel(self.window, on_start_live=self.start_live,
            on_start_simulation=self.start_simulation, on_stop=self.controller.stop,
            on_save=self.save, on_calibrate=self.calibration_dialog,
            on_choose_calibration=self.choose_calibration, on_estimate_calibration=self.estimate_calibration)
        self.panel.pack(fill='both', expand=True)
        if on_focus_video is not None:
            ttk.Button(self.window, text='Pilot from main video / keyboard',
                       command=on_focus_video).pack(fill='x', padx=10, pady=(0, 6))
        try:
            info = calibration_info(calibration_path) if calibration_path else None
        except (OSError, ValueError):
            info = {'kind': 'unknown', 'label': 'Unavailable calibration file'}
        self.panel.set_calibration_path(calibration_path, info)
        self.controller.set_calibration_preview(info)
        self.events = queue.Queue()
        self.calibration_thread = None
        self.closing = False
        self._close_callback = None
        self._last_view = None
        self._last_version = -1
        self._poll_id = self.window.after(100, self.poll)

    def show(self):
        self.window.deiconify()
        self.window.lift()

    def hide(self):
        self.controller.stop()
        self.window.withdraw()

    def start_live(self):
        try:
            if not self.calibration_path:
                raise ValueError('Choose a camera calibration, use Calibrate from images, or Estimate Mini 4 Pro.')
            self.controller.start_live(self.receiver_provider(), self.calibration_path,
                                       loop_closure=bool(self.panel.loop_closure_enabled.get()))
        except Exception as exc:
            messagebox.showerror('Live room mapping', str(exc), parent=self.window)

    def start_simulation(self):
        try:
            self.controller.start_simulation()
        except Exception as exc:
            messagebox.showerror('Autonomous simulation', str(exc), parent=self.window)

    def save(self):
        try:
            self.controller.request_save()
        except Exception as exc:
            messagebox.showerror('Save room map', str(exc), parent=self.window)

    def _set_calibration(self, path):
        info = calibration_info(path)
        self.calibration_path = str(path)
        self.panel.set_calibration_path(self.calibration_path, info)
        self.controller.set_calibration_preview(info)
        self.on_calibration_changed(self.calibration_path)

    def estimate_calibration(self):
        """Use only the dimensions of an already decoded frame; no hardware IO."""
        try:
            if self.controller.busy or (self.calibration_thread and self.calibration_thread.is_alive()):
                raise ValueError('Stop mapping or wait for calibration to finish before changing calibration.')
            receiver = self.receiver_provider()
            frame = receiver.frames.get() if receiver else None
            if frame is None or receiver.state != 'STREAMING' or not 0 <= time.monotonic() - frame.decoded_at <= 1:
                raise ValueError('Connect the existing video and wait for a fresh original frame first.')
            if frame.image.ndim != 3 or frame.image.shape[2] != 3:
                raise ValueError('Expected an original BGR video frame.')
            height, width = frame.image.shape[:2]
            folder = self.base / 'logs' / 'calibration' / (
                datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
            path = folder / 'camera-mini4-pro-estimated.json'
            estimate_mini4_calibration(width, height, output_path=path)
            self._set_calibration(path)
        except Exception as exc:
            messagebox.showerror('Mini 4 Pro estimated calibration', str(exc), parent=self.window)

    def choose_calibration(self):
        path = filedialog.askopenfilename(parent=self.window, title='Choose camera calibration',
                                         filetypes=[('Calibration JSON', '*.json')])
        if path:
            try:
                from .room_mapping import mapping_import
                mapping_import('calibration').load_calibration(path)
                self._set_calibration(path)
            except Exception as exc:
                messagebox.showerror('Calibration', str(exc), parent=self.window)

    def calibration_dialog(self):
        if self.calibration_thread and self.calibration_thread.is_alive():
            messagebox.showinfo('Calibration', 'Calibration is already running.', parent=self.window)
            return
        dialog = tk.Toplevel(self.window)
        dialog.title('Camera calibration from the existing video')
        dialog.geometry('720x390')
        content = ttk.Frame(dialog, padding=16)
        content.pack(fill='both', expand=True)
        ttk.Label(content, text='9 x 6 inner corners | 25 mm squares', font=('Segoe UI', 14, 'bold')).pack(anchor='w')
        ttk.Label(content, wraplength=670, text=(
            'Print the board at actual size and mount it flat. Keep the camera stationary and move/tilt '
            'the board to capture at least 10 distinct views across the original video. '
            'The board must stay fully visible. Changing stream resolution, crop or zoom requires a new calibration. '
            'This does not fly the aircraft or set the SLAM map scale to meters.')).pack(fill='x', pady=10)
        folder = self.base / 'logs' / 'calibration' / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
        folder.mkdir(parents=True, exist_ok=False)
        status = tk.StringVar(value='Capture folder: ' + str(folder))
        ttk.Label(content, textvariable=status, wraplength=660).pack(fill='x', pady=8)
        count = [0]

        def board():
            path = filedialog.asksaveasfilename(parent=dialog, title='Save printable chessboard',
                defaultextension='.svg', initialfile='calibration-board-25mm.svg', filetypes=[('SVG', '*.svg')])
            if path:
                try:
                    from .room_calibration import write_board_svg
                    write_board_svg(path)
                    status.set('Print at 100%, verify 25 mm squares: ' + path)
                except Exception as exc:
                    messagebox.showerror('Board', str(exc), parent=dialog)

        def capture():
            try:
                import cv2
                receiver = self.receiver_provider()
                frame = receiver.frames.get() if receiver else None
                if frame is None or receiver.state != 'STREAMING' or time.monotonic()-frame.decoded_at > 1:
                    raise ValueError('Connect the existing video and wait for a fresh frame first.')
                path = folder / f'board_{count[0]:03d}.png'
                if not cv2.imwrite(str(path), frame.image.copy()):
                    raise OSError('Could not save the original video image.')
                count[0] += 1
                status.set(f'{count[0]} images saved. Move/tilt the board before capturing another.\n{folder}')
            except Exception as exc:
                messagebox.showerror('Capture calibration image', str(exc), parent=dialog)

        def calculate(use_existing=False):
            source = filedialog.askdirectory(parent=dialog, title='Original chessboard image folder') if use_existing else str(folder)
            if not source:
                return
            if self.calibration_thread and self.calibration_thread.is_alive():
                return
            output = folder / 'camera.json'
            status.set('Calculating in the background...')
            def work():
                try:
                    from .room_calibration import calibrate_directory
                    report = calibrate_directory(source, output)
                    if not report.get('success') or not output.is_file():
                        raise ValueError('Calibration did not pass quality checks: ' +
                                         '; '.join(report.get('warnings', ['No calibration file was saved.'])))
                    self.events.put(('calibrated', output, report))
                except Exception as exc:
                    self.events.put(('calibration_error', str(exc), None))
            self.calibration_thread = threading.Thread(target=work, name='room-camera-calibration', daemon=False)
            self.calibration_thread.start()
            dialog.destroy()

        for text, action in [('Save printable board', board), ('Capture original video frame', capture),
                             ('Calibrate captured images', calculate),
                             ('Calibrate existing image folder', lambda: calculate(True))]:
            ttk.Button(content, text=text, command=action).pack(fill='x', pady=3)

    def video_disconnected(self):
        status, _, _ = self.controller.read()
        if status.get('mode') == 'live':
            self.controller.stop()

    def poll(self):
        self._poll_id = None
        while True:
            try:
                kind, value, report = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == 'calibrated':
                self._set_calibration(value)
                if not self.closing:
                    messagebox.showinfo('Calibration saved', str(value), parent=self.window)
            elif not self.closing:
                messagebox.showerror('Calibration failed', value, parent=self.window)
        status, view, version = self.controller.read()
        status['calibration_selected'] = bool(self.calibration_path)
        self.panel.set_status(status)
        self.panel.set_busy(self.controller.busy)
        # Camera validity can change without a new cloud snapshot.
        signature = (version, status.get('pose_valid'), str(view.get('camera_position')))
        if signature != self._last_view:
            self.panel.set_snapshot(view)
            self._last_view = signature
        if self.closing and not self.controller.busy and not (self.calibration_thread and self.calibration_thread.is_alive()):
            callback = self._close_callback
            self.window.destroy()
            if callback:
                callback()
            return
        self._poll_id = self.window.after(150, self.poll)

    def request_close(self, on_closed):
        self.closing = True
        self._close_callback = on_closed
        self.controller.stop()
