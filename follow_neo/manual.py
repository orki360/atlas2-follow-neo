"""Manual input and a control-only MSDKRemote connection (TCP 9998)."""
import socket
import threading
import time
import math
from .dance import dance_command, DanceSmoother
from .edge_search import yaw_override


MOVEMENT_KEYS = frozenset(('w', 's', 'a', 'd', 'up', 'down', 'left', 'right'))
AXES = ('yaw', 'vertical', 'roll', 'forward')
DEFAULT_AXIS_LIMITS = {
    'yaw': .15,
    'vertical': .015,
    'roll': .015,
    'forward': .015,
}


def movement(keys):
    """Full-scale axis directions; final per-axis limits are applied later."""
    return tuple(float(int(pos in keys) - int(neg in keys))
                 for pos, neg in (('d', 'a'), ('w', 's'),
                                  ('right', 'left'), ('up', 'down')))


def shape_dance_axis(value,limit):
    """Continuous intent at every cap; no unmeasured hardware dead-zone boost."""
    return max(-1.,min(1.,float(value)))


class ManualControl:
    def __init__(self, host, connector=socket.create_connection, on_event=None, decision_provider=None):
        self.host = host
        self.connector = connector
        self.on_event = on_event or (lambda event, data: None)
        self.decision_provider=decision_provider
        self.smoother=DanceSmoother()
        self.mode = 'MANUAL'
        self.decision = None
        self.dance_started = 0.
        self.dance_status = 'Dance off'
        self.axis_limits = dict(DEFAULT_AXIS_LIMITS)
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

    def set_axis_limits(self, limits):
        checked = {}
        for axis in AXES:
            value = float(limits[axis])
            if not math.isfinite(value): raise ValueError(f'{axis} limit must be finite')
            checked[axis] = max(0., min(1., value))
        with self.lock:
            self.axis_limits = checked

    def start_dance(self):
        with self.lock:
            self.keys.clear()
            self.action = None
            self.decision = None
            self.dance_started = time.monotonic()
            self.mode = 'DANCE'
            self.smoother.reset()
            self.dance_status = 'Waiting for a new target'
        self.on_event('control_mode', {'mode': 'DANCE'})
        return True

    def manual_mode(self):
        with self.lock:
            self.mode = 'MANUAL'
            self.smoother.reset()
            self.keys.clear()
            self.decision = None
            self.dance_status = 'Dance off'
        self.on_event('control_mode', {'mode': 'MANUAL'})

    def enable(self):
        with self.lock:
            self.decision = None
            self.motion_hold = False
            self.wanted = True
            self.smoother.reset()
        self.on_event('control_enable_requested', {'mode': self.mode})

    def release(self):
        with self.lock:
            self.keys.clear()
            self.wanted = False
            self.smoother.reset()
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
                # Fetch the latest thread-safe decision independently of costly
                # Tk drawing. The UI heartbeat and explicit authority remain.
                latest=self.decision_provider() if self.decision_provider else None
                with self.lock:
                    command_time=time.monotonic()
                    fresh = command_time - self.updated <= .35
                    if not fresh:
                        self.wanted = False
                        self.keys.clear()
                        self.action = None
                        self.dance_status = 'Control released: UI heartbeat lost'
                    wanted = self.wanted
                    values = movement(self.keys)
                    mode = self.mode
                    selected_mode = self.mode
                    decision = latest if self.decision_provider else self.decision
                    if mode == 'DANCE' and not self.keys:
                        values, self.dance_status = dance_command(decision, command_time, self.dance_started)
                    elif mode == 'DANCE':
                        mode = 'MANUAL'
                        self.dance_status = 'Keyboard override: release keys to resume Dance'
                    if self.motion_hold:
                        values = (0., 0., 0., 0.)
                        self.dance_status = 'Flight action: Enable (E) to resume movement'
                    policy_values = (tuple(float(decision.get('intent',{}).get(axis,0.))
                                            for axis in AXES)
                                     if selected_mode=='DANCE' and decision else None)
                    raw_values = values
                    limits = tuple(self.axis_limits[axis] for axis in AXES)
                    # This is the final transport gate shared by keyboard and
                    # Dance. Raw intents are normalized to [-1, 1], so each
                    # factor is also a hard maximum for the transmitted axis.
                    gate_reason=self.dance_status if mode=='DANCE' else 'manual'
                    edge_override=(mode=='DANCE' and not self.motion_hold
                                   and gate_reason.startswith('Edge search /')
                                   and yaw_override(decision,command_time,self.dance_started)!=0.)
                    phase=decision.get('spacing_phase') if decision else None
                    if mode=='DANCE':
                        if edge_override:
                            # Explicit user-requested exception: yaw only at
                            # 100%, for this timed edge search, not normal Dance.
                            self.smoother.reset();shaped_values=raw_values
                        elif phase=='BRAKE' and not self.motion_hold and gate_reason.startswith('Tracking /'):
                            self.smoother.reset(); shaped_values=raw_values
                        else:
                            active=(wanted and self.enabled and not self.motion_hold
                                    and gate_reason.startswith(('Tracking /','Kalman recovery /'))
                                    and phase in ('APPROACH','VISUAL_HOLD'))
                            shaped_values=self.smoother.update(raw_values,command_time,active)
                        if gate_reason.startswith('Kalman recovery /'):
                            # Do not let the smoothing history delay the coast
                            # caps/decay, or carry old roll past its deadline.
                            shaped_values=tuple(math.copysign(min(abs(shaped),abs(raw)),raw)
                                if shaped*raw>0 else 0. for shaped,raw in zip(shaped_values,raw_values))
                            self.smoother.values=shaped_values
                    else:
                        self.smoother.reset(); shaped_values=raw_values
                    effective_limits=(1.,0.,0.,0.) if edge_override else limits
                    values = tuple(value * limit
                                   for value, limit in zip(shaped_values, effective_limits))
                    # Log exactly what is put on the wire. Truncation never
                    # rounds a command above an arbitrary user-entered cap.
                    values=tuple(math.trunc(v*10000)/10000 for v in values)
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
                        'selected_mode': selected_mode,
                        'axis_limits': dict(zip(AXES, limits)),
                        'effective_axis_limits':dict(zip(AXES,effective_limits)),
                        'edge_yaw_override':edge_override,
                        'edge_search':decision.get('edge_search') if decision else None,
                        'raw_values': list(raw_values),
                        'command_time':command_time,'gate_reason':gate_reason,
                        'decision_time':decision.get('prediction_time') if decision else None,
                        'spacing_phase':phase,
                        'brief_detection_gap':decision.get('brief_detection_gap',False) if decision else False,
                        'prediction_recovery':decision.get('prediction_recovery',False) if decision else False,
                        'prediction_forward_allowed':decision.get('prediction_forward_allowed',False) if decision else False,
                        'prediction_quality':decision.get('prediction_quality') if decision else None,
                        'track_support':decision.get('track_support') if decision else None,
                        'shaped_values':list(shaped_values),
                        'policy_values':list(policy_values) if policy_values is not None else None,
                        'intent_basis':decision.get('intent_basis') if mode=='DANCE' and decision else None,
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
                        command('rc 0 0 0 0')
                        command('disable')
                        self.status += ' | Release acknowledged'
                    except (OSError,ConnectionError,TimeoutError,RuntimeError):
                        self.status += ' | Release unconfirmed; use remote controller.'
                sock.close()
