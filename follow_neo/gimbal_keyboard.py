"""Operator-triggered pitch steps on the existing telemetry query connection.

No socket is created here. MSDKRemote's CommandServer accepts one client at a
time, so HeadingReceiver is the only owner of TCP 9997. A request can be pending
or in flight; neither keyboard repeats nor reconnects may replay it.

DJI 5.14.0 GimbalAngleRotationMode bytecode verifies RELATIVE_ANGLE=0,
ABSOLUTE_ANGLE=1, UNKNOWN=65535. The example's 65535 is therefore not used.
Positive pitch is upward and duration is seconds:
https://developer.dji.com/api-reference-v5/android-api/Components/IKeyManager/Value_Gimbal_Struct_GimbalAngleRotation.html
"""
from __future__ import annotations

import json
import threading
import time


def pitch_step_command(direction: int) -> bytes:
    """Small relative pitch only; limits/unsupported actions remain SDK errors."""
    if type(direction) is not int or direction not in (-1, 1):
        raise ValueError('Gimbal direction must be +1 or -1')
    value = dict(mode=0, pitch=3.0 * direction, roll=0.0, yaw=0.0,
                 pitchIgnored=False, rollIgnored=True, yawIgnored=True,
                 duration=0.3, jointReferenceUsed=False, timeout=0)
    # All fields are required by SDK fromJson; timeout=0 is its constructor default.
    return ('action Gimbal RotateByAngle ' + json.dumps(value, separators=(',', ':'))
            + '\r\n').encode('ascii')


class GimbalRequest:
    """One bounded request. Cancellation never waits for network I/O."""
    def __init__(self, direction, callback, clock=time.monotonic):
        self.command = pitch_step_command(direction)
        self.direction = direction
        self.callback = callback
        self.created_at = clock()
        self.expires_at = self.created_at + 0.35
        self.lock = threading.Lock()
        self.phase = 'pending'
        self.cancelled = False

    def cancel(self):
        with self.lock:
            self.cancelled = True
            pending = self.phase == 'pending'
        if pending:
            self.finish('cancelled', 'Unsent gimbal step cancelled')

    def begin_send(self, now):
        with self.lock:
            if self.phase != 'pending' or self.cancelled:
                return False
            expired = now > self.expires_at
            if not expired:
                self.phase = 'in_flight'
                return True
        self.finish('cancelled', 'Gimbal step expired before send; press again')
        return False

    def finish(self, outcome, message):
        with self.lock:
            if self.phase == 'done':
                return
            self.phase = 'done'
        self.callback(self, outcome, message)


class GimbalKeyboard:
    """GUI facade: callbacks may run on the telemetry thread; use read() in Tk.

    Constructing, enabling, disabling and stopping never connect to hardware.
    ``request_step`` is the only operation that schedules a physical action.
    ``stop`` permanently closes this facade, but not the shared heading receiver.
    Disabling cancels pending steps. A step already being sent may still finish.
    """
    def __init__(self, receiver, on_event=None):
        self.receiver = receiver
        self.on_event = on_event or (lambda *args: None)
        self._lock = threading.RLock()
        self._enabled = False
        self._closed = False
        self._request = None
        self._status = 'Gimbal keyboard disabled'
        self._error = None
        self._unknown = False
        self._last_acknowledged = False
        self._requested_pitch = None

    def read(self):
        with self._lock:
            return dict(enabled=self._enabled, busy=self._request is not None,
                        status=self._status, error=self._error,
                        completion_unknown=self._unknown,
                        last_acknowledged=self._last_acknowledged,
                        requested_pitch_deg=self._requested_pitch)

    def set_enabled(self, enabled):
        if not enabled:
            self.disable()
            return True
        with self._lock:
            if self._closed or self._request is not None:
                return False
            if not self.receiver.gimbal_available():
                self._status = 'Gimbal unavailable: query disconnected or previous result unresolved'
                return False
            self._enabled = True
            self._unknown = False
            self._error = None
            self._status = 'Gimbal enabled: I up / K down, 3 deg per press'
            return True

    def request_step(self, direction):
        # Validate before touching state, including rejecting bool as an integer.
        pitch_step_command(direction)
        with self._lock:
            if self._closed or not self._enabled or self._request is not None:
                return False
            request = GimbalRequest(direction, self._completed)
            self._request = request
            self._last_acknowledged = False
            self._requested_pitch = 3.0 * direction
            self._status = 'Gimbal step pending; additional presses are dropped'
            if not self.receiver.queue_gimbal_step(request):
                self._request = None
                self._status = 'Gimbal step not queued: query unavailable or busy'
                return False
            return True

    def disable(self):
        with self._lock:
            self._enabled = False
            request = self._request
            self._status = ('Gimbal disabled; a step already being sent may still finish'
                            if request else 'Gimbal keyboard disabled')
        if request is not None:
            request.cancel()

    def stop(self):
        with self._lock:
            self._closed = True
        self.disable()

    def _completed(self, request, outcome, message):
        with self._lock:
            if self._request is not request:
                return
            self._request = None
            self._last_acknowledged = outcome == 'acknowledged'
            self._unknown = outcome == 'unknown'
            self._error = message if outcome in ('unknown', 'rejected', 'error') else None
            if outcome in ('unknown', 'error'):
                self._enabled = False
            self._status = message
        try:
            self.on_event('gimbal_step', dict(outcome=outcome, message=message,
                          pitch_step_deg=3.0 * request.direction,
                          measured_pitch_deg=None))
        except Exception:
            # Logging/UI callbacks must not interrupt the single socket owner.
            pass
