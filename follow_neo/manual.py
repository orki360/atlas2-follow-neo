"""Manual input and a control-only MSDKRemote connection (TCP 9998)."""
import socket
import threading
import time
import math
from .dance import dance_command, DanceSmoother
from .edge_search import yaw_override
from .control_transport import ControlChannel
from .recovery_search import scan_command
from .search_test import SearchTest


MOVEMENT_KEYS = frozenset(('w', 's', 'a', 'd', 'up', 'down', 'left', 'right'))
AXES = ('yaw', 'vertical', 'roll', 'forward')
DEFAULT_AXIS_LIMITS = {
    'yaw': .50,
    'vertical': .50,
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
        self.communication_paused = False
        self.pause_reason = ''
        self.resume_requested = False
        self.auto_resume_allowed = False
        self.hold_before_pause = False
        self.input_keys = set()
        self.blocked_keys = set()
        self.connected = False
        self.last_acknowledged=None
        self.search_test=None;self.heading_provider=None
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
            self.input_keys = set(keys)
            self.blocked_keys.intersection_update(self.input_keys)
            if self.communication_paused:self.blocked_keys.update(self.input_keys)
            self.keys = self.input_keys-self.blocked_keys
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
            self._stop_search_test('dance_selected')
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
            self._stop_search_test('manual_selected')
            self.mode = 'MANUAL'
            self.smoother.reset()
            self.keys.clear()
            self.decision = None
            self.dance_status = 'Dance off'
        self.on_event('control_mode', {'mode': 'MANUAL'})

    def enable(self):
        with self.lock:
            if self.communication_paused:self.keys.clear()
            self.action = None
            if self.mode=='DANCE' and not (self.wanted and self.enabled):self.dance_started=time.monotonic()
            self.decision = None
            self.resume_requested = self.communication_paused
            self.motion_hold = self.communication_paused
            self.wanted = True
            self.smoother.reset()
        self.on_event('control_enable_requested', {'mode': self.mode})

    def pause_control(self, reason, *, automatic_recovery=False):
        with self.lock:
            first = not self.communication_paused
            if first:
                self.hold_before_pause = self.motion_hold
                self.auto_resume_allowed = automatic_recovery
            elif not automatic_recovery:
                self.auto_resume_allowed = False
            self.communication_paused = True
            self.pause_reason = reason
            self.motion_hold = True
            self.blocked_keys.update(self.input_keys)
            self.keys.clear()
            self.action = None
            self.decision = None
            self.smoother.reset()
            self._stop_search_test('control_paused')
            self.dance_status = 'Movement paused: '+reason
            self.status = 'CONTROL CONNECTED | MOVEMENT PAUSED: '+reason
        if first:
            self.on_event('control_movement_paused', {'reason':reason, 'connection_kept':True})

    def release(self):
        with self.lock:
            self._stop_search_test('control_released')
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
            if self.enabled and self.wanted and not self.communication_paused and self.action is None:
                self._stop_search_test('flight_action')
                self.keys.clear()
                self.motion_hold = True
                self.dance_status = 'Flight action: Enable (E) to resume movement'
                self.action = action
                return True
        return False

    def stop(self):
        self.release()
        self.stop_event.set()

    def _stop_search_test(self,reason):
        if self.search_test:
            self.search_test.stop(time.monotonic(),reason)
            if reason in ('keyboard_override','dance_selected','flight_action'):
                self.search_test.finish(time.monotonic())
        if self.mode=='SEARCH_TEST':self.mode='MANUAL';self.keys.clear()

    def stop_search_test(self):
        with self.lock:self._stop_search_test('operator_stop')

    def start_search_test(self,settings,heading_provider,scenario):
        with self.lock:
            if not self.enabled or not self.wanted or self.motion_hold or self.stop_event.is_set():
                raise ValueError('Connect control and Enable (E) before a search test')
            if time.monotonic()-self.updated>.35:raise ValueError('Fresh GUI heartbeat required')
            if self.search_test and not self.search_test.finished:
                raise ValueError('Wait for the 2-second post-stop measurement to finish')
            heading=heading_provider();now=time.monotonic()
            self.search_test=SearchTest(settings,heading,now,scenario,self.on_event)
            self.heading_provider=heading_provider;self.mode='SEARCH_TEST'
            self.keys.clear();self.action=None;self.smoother.reset()
            self.dance_status='Search test / '+scenario
        return True

    def control_snapshot(self):
        with self.lock:
            active=bool(self.enabled and self.wanted and not self.communication_paused and not self.stop_event.is_set()
                        and time.monotonic()-self.updated<=.35)
            return dict(active=active,mode=self.mode,reason=self.dance_status,
                connected=self.connected,paused=self.communication_paused,pause_reason=self.pause_reason,
                acknowledged=self.last_acknowledged,motion_hold=self.motion_hold,
                keyboard_override=bool(self.keys or self.blocked_keys),time=time.monotonic(),
                search_test=dict(self.search_test.last_sample or {},finished=self.search_test.finished)
                    if self.search_test else None)

    def _run(self):
        sock = None
        authority = False
        enable_pending = False
        channel = None
        failure = None

        def guard(motion_keys=None):
            with self.lock:
                if self.stop_event.is_set():return 'control stopped'
                if not self.wanted:return 'control released'
                if self.communication_paused:return self.pause_reason or 'movement paused'
                if time.monotonic()-self.updated>.35:return 'UI heartbeat lost'
                if motion_keys is not None and set(motion_keys)!=self.keys:
                    return 'movement keys changed while acknowledgement was pending'
                if motion_keys is not None and (self.mode!=selected_mode or self.motion_hold):
                    return 'movement keys changed while acknowledgement was pending'
            return None

        try:
            sock = self.connector((self.host, 9998), timeout=2.)
            self.connected = True
            channel = ControlChannel(sock,guard,self.on_event)
            def pause_channel():
                recoverable = (not channel.rejected and not channel.ambiguous and
                    (channel.pause_reason.startswith('Control acknowledgement delayed beyond ')
                     or channel.pause_reason=='UI heartbeat lost'))
                self.pause_control(channel.pause_reason,automatic_recovery=recoverable)
            def command(*args, **kwargs):
                result = channel.command(*args, **kwargs)
                if channel.paused:
                    channel.pause(channel.pause_reason)
                    pause_channel()
                return result
            self.status = 'Control connected | E: enable keyboard'
            while not self.stop_event.is_set():
                with self.lock:
                    stale = self.wanted and time.monotonic()-self.updated>.35
                    externally_paused = self.communication_paused
                    pause_reason = self.pause_reason
                if stale and not channel.paused:
                    channel.pause('UI heartbeat lost')
                if externally_paused and not channel.paused:
                    channel.pause(pause_reason)
                if channel.paused:
                    pause_channel()
                    with self.lock:
                        if self.search_test and not self.search_test.finished:
                            heading=self.heading_provider()
                            self.search_test.update(heading,time.monotonic(),False)
                    with self.lock:
                        wanted = self.wanted
                    if not wanted and authority:
                        channel.release_control()  # Q / Esc, not an automatic timeout action.
                        authority = self.enabled = False
                        enable_pending = False
                    channel.poll_pause()
                    # Recheck user release/focus/heartbeat after polling: an ACK
                    # received during Q / Esc must never re-enable authority.
                    with self.lock:
                        requested, self.resume_requested = self.resume_requested, False
                        automatic = self.auto_resume_allowed and not channel.rejected
                        fresh = time.monotonic()-self.updated<=.35
                        resumed = ((requested or automatic) and self.wanted and fresh
                                   and not self.stop_event.is_set() and channel.resume())
                        if resumed:
                            self.communication_paused = False
                            self.motion_hold = self.hold_before_pause and not requested
                            self.auto_resume_allowed = False
                            self.pause_reason = ''
                            self.blocked_keys.update(self.input_keys)
                            self.keys.clear();self.action=None;self.decision=None
                            self.smoother.reset();self.dance_started=time.monotonic()
                    if resumed:
                        # Repeat the enable handshake before allowing new movement.
                        enable_pending = True
                        self.enabled = False
                        self.on_event('control_movement_resumed', {'explicit_enable':requested,'automatic':not requested})
                        self.status = 'Control connected | Enabling after acknowledged neutral'
                    else:
                        detail = ('Manual reconnect required: ambiguous replies' if channel.ambiguous else
                                  ('Neutral acknowledged; waiting for GUI recovery' if automatic and not fresh else
                                   'Neutral acknowledged; press E to resume') if channel.neutral_acknowledged and not channel.pending else
                                  'Waiting for command and neutral acknowledgements')
                        self.status = 'CONTROL CONNECTED | MOVEMENT PAUSED | '+channel.pause_reason+' | '+detail
                    self.stop_event.wait(.02)
                    continue
                # Fetch the latest thread-safe decision independently of costly
                # Tk drawing. The UI heartbeat and explicit authority remain.
                latest=self.decision_provider() if self.decision_provider else None
                with self.lock:
                    command_time=time.monotonic()
                    fresh = command_time - self.updated <= .35
                    wanted = self.wanted
                    command_keys = frozenset(self.keys)
                    values = movement(self.keys)
                    mode = self.mode
                    selected_mode = self.mode
                    decision = latest if self.decision_provider else self.decision
                    test_values=None
                    if self.search_test and not self.search_test.finished:
                        if self.keys:self._stop_search_test('keyboard_override')
                        heading=self.heading_provider();command_time=time.monotonic()
                        test_values=self.search_test.update(heading,command_time,
                            wanted and self.enabled and self.mode=='SEARCH_TEST' and not self.motion_hold)
                        mode=selected_mode=self.mode
                    if self.mode=='SEARCH_TEST':
                        mode=selected_mode='SEARCH_TEST';values=test_values or (0.,0.,0.,0.)
                        decision=self.search_test.decision
                        self.dance_status='Search test / '+(self.search_test.reason or 'scanning')
                        if self.search_test.finished:self.mode='MANUAL'
                    if mode == 'DANCE' and not self.keys:
                        values, self.dance_status = dance_command(decision, command_time, self.dance_started)
                    elif mode == 'DANCE':
                        mode = 'MANUAL'
                        self.dance_status = 'Keyboard override: release keys to resume Dance'
                    if mode == 'DANCE' and self.blocked_keys:
                        values = (0., 0., 0., 0.)
                        self.dance_status = 'Keyboard override: release held keys after communication pause'
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
                    gate_reason=self.dance_status if mode in ('DANCE','SEARCH_TEST') else 'manual'
                    edge_override=(mode=='DANCE' and not self.motion_hold
                                   and gate_reason.startswith('Search /')
                                   and yaw_override(decision,command_time,self.dance_started)!=0.)
                    scan_override=(mode=='DANCE' and not self.motion_hold
                        and gate_reason.startswith('Scan /') and decision
                        and decision.get('edge_search',{}).get('yaw_full_scale') is True
                        and scan_command(decision,command_time,self.dance_started)!=0.)
                    phase=decision.get('spacing_phase') if decision else None
                    if mode=='DANCE':
                        if edge_override:
                            # Explicit user-requested exception: yaw only at
                            # 100%, for this timed edge search, not normal Dance.
                            self.smoother.reset();shaped_values=raw_values
                        elif gate_reason.startswith('Scan /'):
                            self.smoother.reset();shaped_values=raw_values
                        elif phase=='BRAKE' and not self.motion_hold and gate_reason.startswith('Tracking /'):
                            self.smoother.reset(); shaped_values=raw_values
                        else:
                            active=(wanted and self.enabled and not self.motion_hold
                                    and gate_reason.startswith(('Tracking /','Kalman recovery /'))
                                    and phase in ('APPROACH','VISUAL_HOLD','HOLD_REACQUIRE','CONFIRM_STOP'))
                            shaped_values=self.smoother.update(raw_values,command_time,active,
                                damped_yaw=bool(decision and decision.get('centering_control')))
                        if gate_reason.startswith('Kalman recovery /'):
                            # Do not let the smoothing history delay the coast
                            # caps/decay, or carry old roll past its deadline.
                            shaped_values=tuple(math.copysign(min(abs(shaped),abs(raw)),raw)
                                if shaped*raw>0 else 0. for shaped,raw in zip(shaped_values,raw_values))
                            self.smoother.values=shaped_values
                    else:
                        self.smoother.reset(); shaped_values=raw_values
                    effective_limits=(1.,0.,0.,0.) if edge_override or scan_override or mode=='SEARCH_TEST' else limits
                    values = tuple(value * limit
                                   for value, limit in zip(shaped_values, effective_limits))
                    # Log exactly what is put on the wire. Truncation never
                    # rounds a command above an arbitrary user-entered cap.
                    values=tuple(math.trunc(v*10000)/10000 for v in values)
                    action, self.action = self.action, None
                if wanted and (not authority or enable_pending):
                    authority = True  # Also release if the enable reply is lost.
                    if not command('rc 0 0 0 0'):continue
                    if not command('enable'):continue
                    if not command('rc 0 0 0 0'):continue
                    self.enabled = True
                    enable_pending = False
                    self.on_event('control_enabled', {'host': self.host})
                    self.status = 'KEYBOARD ACTIVE | Q / Esc: release'
                    continue  # Recheck focus/stop before sending any movement.
                if authority and not wanted:
                    if not command('rc 0 0 0 0',interruptible=False):continue
                    if not command('disable',interruptible=False):continue
                    authority = self.enabled = False
                    enable_pending = False
                    self.on_event('control_released', {})
                    self.status = 'Control released | E: enable keyboard'
                if authority and wanted:
                    applied=command('rc ' + ' '.join(f'{value:.4f}' for value in values),
                            motion_keys=command_keys if any(values) else None)
                    if not applied:
                        if channel.neutral_acknowledged and not channel.ambiguous:
                            self.last_acknowledged=dict(time=time.monotonic(),mode='MANUAL',values=[0.,0.,0.,0.],
                                reason='Neutral acknowledged')
                        continue  # Re-read current keys; never replay the superseded command.
                    self.last_acknowledged=dict(time=time.monotonic(),mode=mode,values=list(values),reason=gate_reason)
                    self.on_event('flight_command', {'mode': mode, 'command': 'rc',
                        'selected_mode': selected_mode,
                        'axis_limits': dict(zip(AXES, limits)),
                        'effective_axis_limits':dict(zip(AXES,effective_limits)),
                        'edge_yaw_override':edge_override,
                        'search_yaw_override':bool(edge_override or scan_override or mode=='SEARCH_TEST'),
                        'search_test_run_id':self.search_test.run_id if mode=='SEARCH_TEST' else None,
                        'edge_search':decision.get('edge_search') if decision else None,
                        'raw_values': list(raw_values),
                        'command_time':command_time,'gate_reason':gate_reason,
                        'decision_time':decision.get('prediction_time') if decision else None,
                        'spacing_phase':phase,
                        'brief_detection_gap':decision.get('brief_detection_gap',False) if decision else False,
                        'prediction_recovery':decision.get('prediction_recovery',False) if decision else False,
                        'prediction_forward_allowed':decision.get('prediction_forward_allowed',False) if decision else False,
                        'prediction_quality':decision.get('prediction_quality') if decision else None,
                        'centering_control':decision.get('centering_control') if decision else None,
                        'track_support':decision.get('track_support') if decision else None,
                        'shaped_values':list(shaped_values),
                        'policy_values':list(policy_values) if policy_values is not None else None,
                        'intent_basis':decision.get('intent_basis') if mode=='DANCE' and decision else None,
                        'values': list(values), 'server_acknowledged': True,
                        'measurement_id': decision.get('measurement_id') if mode == 'DANCE' and decision else None})
                    with self.lock:
                        action_allowed = (self.wanted and not self.communication_paused and not self.stop_event.is_set()
                                          and time.monotonic() - self.updated <= .35)
                    if action and action_allowed:
                        if not command(action):continue
                        self.on_event('flight_command', {'mode': 'MANUAL', 'command': action,
                            'server_acknowledged': True})
                        self.status = f'{action}: success | Q / Esc: release'
                self.stop_event.wait(.05)
            self.status = 'Control disconnected'
        except Exception as exc:
            failure=str(exc)
            if sock is not None and channel is not None and not isinstance(exc,OSError) and not self.stop_event.is_set():
                # An application error is not permission to disconnect a live
                # transport. Quarantine it until the operator chooses to close.
                try:
                    channel.pause('Control processing error: '+str(exc),ambiguous=True)
                    self.pause_control(channel.pause_reason)
                    while not self.stop_event.is_set():
                        if not self.wanted and authority:
                            channel.release_control();authority=self.enabled=False
                        channel.poll_pause()
                        self.stop_event.wait(.02)
                    self.status='Control disconnected'
                except OSError as transport_error:
                    self.status=f'CONTROL ERROR: {transport_error}'
                    self.on_event('control_error',{'error':str(transport_error)})
            else:
                self.status = f'CONTROL ERROR: {exc}. Disconnect and reconnect control.'
                self.on_event('control_error', {'error': str(exc)})
        finally:
            self.connected = False
            self.enabled = False
            self.release()
            if self.search_test and not self.search_test.finished:
                self.search_test.finish(time.monotonic())
            if sock:
                if authority and channel is not None:
                    if failure or channel.pending or channel.paused:
                        channel.stop_unconfirmed(failure or 'pending acknowledgement')
                        self.status += ' | STOP UNCONFIRMED; use physical controller.'
                    else:
                        try:
                            if (channel.command('rc 0 0 0 0',interruptible=False)
                                    and channel.command('disable',interruptible=False)):
                                self.on_event('control_released',{'server_acknowledged':True})
                                self.status += ' | Release acknowledged'
                            else:
                                channel.stop_unconfirmed('operator disconnect with pending acknowledgements')
                                self.status += ' | STOP UNCONFIRMED; use physical controller.'
                        except (OSError,RuntimeError) as exc:
                            channel.stop_unconfirmed(str(exc))
                            self.status += ' | STOP UNCONFIRMED; use physical controller.'
                sock.close()
