"""Manual input and a control-only MSDKRemote connection (TCP 9998)."""
import socket
import threading
import time
import math
from .dance import dance_command


MOVEMENT_KEYS = frozenset(('w', 's', 'a', 'd', 'up', 'down', 'left', 'right'))


def movement(keys):
    """Same axes and speeds as FPVdemo; opposite keys cancel."""
    return tuple(scale * (int(pos in keys) - int(neg in keys))
                 for pos, neg, scale in (('d', 'a', .15), ('w', 's', .015),
                                         ('right', 'left', .015), ('up', 'down', .015)))


class ManualControl:
    def __init__(self, host, connector=socket.create_connection, on_event=None):
        self.host = host
        self.connector = connector
        self.on_event = on_event or (lambda event, data: None)
        self.mode = 'MANUAL'
        self.decision = None
        self.dance_started = 0.
        self.dance_status = 'Dance off'
        self.speed = 1.
        self.motion_hold = False
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.keys = set()
        self.updated = 0.
        self.wanted = False
        self.enabled = False
        self.action = None
        self.status = 'Connecting control: 9998'
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def update(self, keys, decision=None):
        with self.lock:
            self.keys = set(keys)
            self.decision = decision
            self.updated = time.monotonic()

    def set_speed(self, value):
        value = float(value)
        if not math.isfinite(value): raise ValueError('Speed must be finite')
        with self.lock:
            self.speed = max(0., min(1., value))

    def start_dance(self):
        with self.lock:
            self.keys.clear()
            self.action = None
            self.decision = None
            self.dance_started = time.monotonic()
            self.mode = 'DANCE'
            self.dance_status = 'Waiting for a new target'
        self.on_event('control_mode', {'mode': 'DANCE'})
        return True

    def manual_mode(self):
        with self.lock:
            self.mode = 'MANUAL'
            self.keys.clear()
            self.decision = None
            self.dance_status = 'Dance off'
        self.on_event('control_mode', {'mode': 'MANUAL'})

    def enable(self):
        with self.lock:
            self.decision = None
            self.motion_hold = False
            self.wanted = True
        self.on_event('control_enable_requested', {'mode': self.mode})

    def release(self):
        with self.lock:
            self.keys.clear()
            self.wanted = False
            self.action = None
            self.decision = None
            self.dance_status = 'Control released: Enable (E) to resume'

    def request(self, action):
        if action not in ('takeoff', 'land'):
            raise ValueError(action)
        with self.lock:
            if self.enabled and self.wanted and self.action is None:
                self.keys.clear()
                self.motion_hold = True
                self.dance_status = 'Flight action: Enable (E) to resume movement'
                self.action = action
                return True
        return False

    def stop(self):
        self.release()
        self.stop_event.set()

    def _run(self):
        sock = None
        authority = False
        buffer = b''

        def command(text):
            nonlocal buffer
            sock.sendall((text + '\r\n').encode('ascii'))
            deadline = time.monotonic() + 2.
            while b'\n' not in buffer:
                sock.settimeout(max(.001, deadline - time.monotonic()))
                chunk = sock.recv(4096)
                if not chunk:
                    raise ConnectionError('Control server disconnected')
                buffer += chunk
                if len(buffer) > 8192 or time.monotonic() > deadline:
                    raise TimeoutError('Control reply timed out')
            reply, buffer = buffer.split(b'\n', 1)
            reply = reply.decode('utf-8', errors='replace').strip()
            if reply != 'success':
                raise RuntimeError(reply)

        try:
            sock = self.connector((self.host, 9998), timeout=2.)
            self.status = 'Control connected | E: enable keyboard'
            while not self.stop_event.is_set():
                with self.lock:
                    fresh = time.monotonic() - self.updated <= .35
                    if not fresh:
                        self.wanted = False
                        self.keys.clear()
                        self.action = None
                        self.dance_status = 'Control released: UI heartbeat lost'
                    wanted = self.wanted
                    values = movement(self.keys)
                    mode = self.mode
                    selected_mode = self.mode
                    decision = self.decision
                    if mode == 'DANCE' and not self.keys:
                        values, self.dance_status = dance_command(decision, time.monotonic(), self.dance_started)
                    elif mode == 'DANCE':
                        mode = 'MANUAL'
                        self.dance_status = 'Keyboard override: release keys to resume Dance'
                    if self.motion_hold:
                        values = (0., 0., 0., 0.)
                        self.dance_status = 'Flight action: Enable (E) to resume movement'
                    speed = self.speed
                    values = tuple(value * speed for value in values)
                    action, self.action = self.action, None
                if wanted and not authority:
                    authority = True  # Also release if the enable reply is lost.
                    command('enable')
                    self.enabled = True
                    self.on_event('control_enabled', {'host': self.host})
                    self.status = 'KEYBOARD ACTIVE | Q / Esc: release'
                    continue  # Recheck focus/stop before sending any movement.
                if authority and not wanted:
                    command('rc 0 0 0 0')
                    command('disable')
                    authority = self.enabled = False
                    self.on_event('control_released', {})
                    self.status = 'Control released | E: enable keyboard'
                if authority and wanted:
                    command('rc ' + ' '.join(f'{value:.4f}' for value in values))
                    self.on_event('flight_command', {'mode': mode, 'command': 'rc',
                        'selected_mode': selected_mode, 'speed_factor': speed,
                        'values': list(values), 'server_acknowledged': True,
                        'measurement_id': decision.get('measurement_id') if mode == 'DANCE' and decision else None})
                    with self.lock:
                        action_allowed = (self.wanted and not self.stop_event.is_set()
                                          and time.monotonic() - self.updated <= .35)
                    if action and action_allowed:
                        command(action)
                        self.on_event('flight_command', {'mode': 'MANUAL', 'command': action,
                            'server_acknowledged': True})
                        self.status = f'{action}: success | Q / Esc: release'
                self.stop_event.wait(.05)
            self.status = 'Control disconnected'
        except Exception as exc:
            self.status = f'CONTROL ERROR: {exc}. Disconnect and reconnect control.'
            self.on_event('control_error', {'error': str(exc)})
        finally:
            self.enabled = False
            self.release()
            if sock:
                if authority:
                    try:
                        sock.settimeout(.3)
                        sock.sendall(b'rc 0 0 0 0\r\ndisable\r\n')
                        self.status += ' | Release requested'
                    except OSError:
                        self.status += ' | Release unconfirmed; use remote controller.'
                sock.close()
