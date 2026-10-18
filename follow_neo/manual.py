"""Manual input and a control-only MSDKRemote connection (TCP 9998)."""
import socket
import threading
import time
import math
from .dance import dance_command, DanceSmoother, SearchSmoother
from .edge_search import yaw_override
from .control_transport import ControlChannel, VISUAL_CHANGE, MotionReduction, UNEXPECTED_REPLY
from .nano_guard import apply_visual
from .motion_continuity import pending_motion_status
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
    def __init__(self, host, connector=socket.create_connection, on_event=None, decision_provider=None,
                 *,require_target_policy=False):
        self.host = host
        self.connector = connector
        self.on_event = on_event or (lambda event, data: None)
        self.decision_provider=decision_provider
        self.require_target_policy=require_target_policy
        self.smoother=DanceSmoother()
        self.search_smoother=SearchSmoother()
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
        self.pause_started = None
        self.preserve_tracking = False
        self.resume_measurement_after = None
        self.input_keys = set()
        self.blocked_keys = set()
        self.connected = False
        self.last_acknowledged=None
        self.transport_status={}
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
            self.resume_measurement_after = None
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
            changed = first or reason != self.pause_reason
            if first:
                self.hold_before_pause = self.motion_hold
                self.auto_resume_allowed = automatic_recovery
                self.pause_started = time.monotonic()
                self.preserve_tracking = bool(automatic_recovery and self.enabled and self.wanted
                    and self.mode=='DANCE' and not self.motion_hold and not self.input_keys
                    and (reason.startswith('Control acknowledgement delayed beyond ')
                         or reason==UNEXPECTED_REPLY))
            elif not automatic_recovery:
                self.auto_resume_allowed = False
                self.preserve_tracking = False
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
            # The polling loop supplies the detailed waiting/reconnect status.
            # Rewriting it on every poll made the GUI/log alternate forever.
            if changed:self.status = 'CONTROL CONNECTED | MOVEMENT PAUSED: '+reason
        if first:
            self.on_event('control_movement_paused', {'reason':reason, 'connection_kept':True,
                'preserve_tracking':self.preserve_tracking,'pause_started':self.pause_started})

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

    def start_search_test(self,settings,heading_provider,scenario,*,boost=False):
        with self.lock:
            if not self.enabled or not self.wanted or self.motion_hold or self.stop_event.is_set():
                raise ValueError('Connect control and Enable (E) before a search test')
            if time.monotonic()-self.updated>.35:raise ValueError('Fresh GUI heartbeat required')
            if self.search_test and not self.search_test.finished:
                raise ValueError('Wait for the 2-second post-stop measurement to finish')
            heading=heading_provider();now=time.monotonic()
            self.search_test=SearchTest(settings,heading,now,scenario,self.on_event,boost=boost)
            self.heading_provider=heading_provider;self.mode='SEARCH_TEST'
            self.keys.clear();self.action=None;self.smoother.reset()
            self.dance_status='Search test / '+scenario
        return True

    def control_snapshot(self):
        with self.lock:
            now=time.monotonic();transport=dict(self.transport_status)
            pending_at=transport.get('oldest_pending_sent_at')
            transport['oldest_pending_age_ms']=max(0.,now-pending_at)*1000 if pending_at is not None else None
            active=bool(self.enabled and self.wanted and not self.communication_paused and not self.stop_event.is_set()
                        and now-self.updated<=.35)
            return dict(active=active,mode=self.mode,reason=self.dance_status,
                continuity_pause=bool(self.communication_paused and self.preserve_tracking
                    and self.enabled and self.wanted and self.mode=='DANCE' and not self.input_keys
                    and self.pause_started is not None and 0<=now-self.pause_started<=1.
                    and now-self.updated<=.35 and not self.stop_event.is_set()),
                connected=self.connected,paused=self.communication_paused,pause_reason=self.pause_reason,
                acknowledged=self.last_acknowledged,motion_hold=self.motion_hold,
                transport=transport,keyboard_override=bool(self.keys or self.blocked_keys),time=now,
                search_test=dict(self.search_test.last_sample or {},finished=self.search_test.finished)
                    if self.search_test else None)

    def _run(self):
        sock = None
        authority = False
        enable_pending = False
        channel = None
        failure = None
        pending_log_key = None

        def guard(motion_keys=None):
            nonlocal pending_log_key
            with self.lock:
                if self.stop_event.is_set():return 'control stopped'
                if not self.wanted:return 'control released'
                if self.communication_paused:return self.pause_reason or 'movement paused'
                if time.monotonic()-self.updated>.35:return 'UI heartbeat lost'
                if motion_keys is not None and set(motion_keys)!=self.keys:
                    return 'movement keys changed while acknowledgement was pending'
                if motion_keys is not None and (self.mode!=selected_mode or self.motion_hold):
                    return 'movement keys changed while acknowledgement was pending'
                if motion_keys is not None and selected_mode=='DANCE' and not motion_keys:
                    latest=self.decision_provider() if self.decision_provider else self.decision
                    if decision and 'generation' in decision and (
                            not latest or latest.get('generation')!=decision['generation']):
                        self.smoother.reset();self.search_smoother.reset()
                        self.dance_status='Video hold / tracking source invalidated'
                        return VISUAL_CHANGE
                    status=pending_motion_status(decision,latest,time.monotonic(),self.dance_started,
                        values,effective_limits,require_target_policy=self.require_target_policy,
                        resume_after=self.resume_measurement_after)
                    reason=status['reason']
                    key=(reason,status['renewed'],status['from_authority'],status['to_authority'])
                    if (reason or status['renewed']) and key!=pending_log_key:
                        pending_log_key=key
                        self.on_event('control_pending_motion',dict(status,wire_values=list(values),
                            measurement_time=decision.get('measurement_time') if decision else None,
                            pending_ack_deadline_unchanged=True))
                    if reason:
                        if status.get('reduction_values'):
                            return MotionReduction(tuple(status['reduction_values']))
                        self.smoother.reset();self.search_smoother.reset()
                        self.dance_status='Nano hold / '+reason
                        return VISUAL_CHANGE
            return None

        try:
            sock = self.connector((self.host, 9998), timeout=2.)
            self.connected = True
            def transport_event(event,data):
                with self.lock:self.transport_status=channel.diagnostics()
                self.on_event(event,data)
            channel = ControlChannel(sock,guard,transport_event)
            def resync_allowed():
                with self.lock:
                    return bool(authority and self.enabled and self.wanted and self.mode=='DANCE'
                        and not enable_pending and not self.input_keys and self.action is None
                        and not self.stop_event.is_set() and time.monotonic()-self.updated<=.35
                        and ((not self.communication_paused and not self.motion_hold)
                             or (self.preserve_tracking and self.pause_reason==UNEXPECTED_REPLY
                                 and self.auto_resume_allowed and not self.hold_before_pause)))
            def pause_channel():
                recoverable = (not channel.rejected and not channel.ambiguous and
                    (channel.pause_reason.startswith('Control acknowledgement delayed beyond ')
                     or channel.pause_reason=='UI heartbeat lost'
                     or channel.resync_confirmed and channel.pause_reason==UNEXPECTED_REPLY))
                recoverable = recoverable or (channel.can_resync and resync_allowed())
                self.pause_control(channel.pause_reason,automatic_recovery=recoverable)
            def command(*args, **kwargs):
                result = channel.command(*args, **kwargs)
                if channel.paused:
                    channel.pause(channel.pause_reason)
                    pause_channel()
                return result
            self.status = 'Control connected | E: enable keyboard'
            while not self.stop_event.is_set():
                cycle_started=time.monotonic()
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
                    if channel.can_resync and resync_allowed():
                        self.status='CONTROL CONNECTED | Verifying neutral and reply synchronization'
                        if not channel.synchronize_neutral(resync_allowed):
                            self.pause_control(channel.pause_reason,automatic_recovery=False)
                        else:
                            self.last_acknowledged=dict(time=time.monotonic(),mode='MANUAL',values=[0.,0.,0.,0.],
                                reason='Neutral confirmed by ordered server marker')
                    else:channel.poll_pause()
                    # Recheck user release/focus/heartbeat after polling: an ACK
                    # received during Q / Esc must never re-enable authority.
                    with self.lock:
                        requested, self.resume_requested = self.resume_requested, False
                        automatic = self.auto_resume_allowed and not channel.rejected
                        fresh = time.monotonic()-self.updated<=.35
                        reconnect = (requested and channel.ambiguous and self.wanted and fresh
                                     and not self.stop_event.is_set())
                        reconnect_reason=self.pause_reason
                        # A timeout sends neutral, not disable. Keep an already
                        # acknowledged enable on this same, unambiguous stream.
                        # Explicit E, rejected commands and incomplete handshakes
                        # still require a new enable handshake.
                        retain_authority = (automatic and not requested and authority
                                            and self.enabled and not enable_pending
                                            and not channel.ambiguous and not channel.rejected)
                        resumed = ((requested or automatic) and self.wanted and fresh
                                   and not self.stop_event.is_set() and channel.resume())
                        if resumed:
                            resumed_at=time.monotonic()
                            preserve = bool(retain_authority and self.preserve_tracking
                                and self.mode=='DANCE' and not self.input_keys
                                and self.pause_started is not None and 0<=resumed_at-self.pause_started<=1.)
                            self.communication_paused = False
                            self.motion_hold = self.hold_before_pause and not requested
                            self.auto_resume_allowed = False
                            self.pause_reason = ''
                            self.blocked_keys.update(self.input_keys)
                            self.keys.clear();self.action=None;self.decision=None
                            self.smoother.reset();self.search_smoother.reset()
                            self.resume_measurement_after=resumed_at if preserve else None
                            if not preserve:self.dance_started=resumed_at
                            self.preserve_tracking=False
                    if reconnect:
                        # E explicitly requests a NEW stream. An unknown reply
                        # must never be discarded and reused as an enable ACK.
                        self.on_event('control_reconnect_requested', {'reason':channel.pause_reason,'explicit_enable':True})
                        self.status='Reconnecting control after Enable | Movement held'
                        channel.stop_unconfirmed('explicit Enable reconnect')
                        sock.close();sock=None
                        authority=self.enabled=self.connected=False
                        with self.lock:
                            self.blocked_keys.update(self.input_keys)
                            self.keys.clear();self.action=None;self.decision=None
                            self.last_acknowledged=None;self.transport_status={}
                            self.smoother.reset();self.search_smoother.reset()
                        if self.stop_event.is_set():break
                        sock=self.connector((self.host,9998),timeout=2.)
                        channel=ControlChannel(sock,guard,transport_event)
                        self.connected=True
                        with self.lock:
                            # Focus loss during the blocking connect still wins.
                            if self.pause_reason==reconnect_reason:
                                self.communication_paused=False;self.pause_reason=''
                                self.motion_hold=False
                            self.auto_resume_allowed=False;self.resume_requested=False
                            self.dance_started=time.monotonic()
                        enable_pending=True
                        self.on_event('control_reconnected', {'explicit_enable':True,'handshake_required':True})
                        self.status='Control reconnected | Waiting for new enable acknowledgements'
                        continue
                    if resumed:
                        enable_pending = not retain_authority
                        if enable_pending:self.enabled = False
                        self.on_event('control_movement_resumed', {
                            'explicit_enable':requested,'automatic':not requested,
                            'authority_retained':retain_authority,'handshake_required':enable_pending,
                            'preserve_tracking':preserve,'resume_measurement_after':self.resume_measurement_after})
                        self.status = ('KEYBOARD ACTIVE | Recovered after acknowledged neutral; release held keys'
                                       if retain_authority else
                                       'Control connected | Enabling after acknowledged neutral')
                    else:
                        detail = ('Ambiguous replies; Enable (E) reconnects control' if channel.ambiguous else
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
                        # The source may cross 250 ms after the provider read.
                        # Use the exact same clock for handoff, gate and logging.
                        if decision:
                            decision=dict(decision,recovery_axis_limits=dict(self.axis_limits))
                            decision=apply_visual(decision,decision.get('nano'),command_time)
                        values, self.dance_status = dance_command(decision, command_time, self.dance_started,
                            require_target_policy=self.require_target_policy,resume_after=self.resume_measurement_after)
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
                                    and gate_reason.startswith(('Tracking /','Kalman recovery /','Nano bridge /','Recovery vector /'))
                                    and phase in ('APPROACH','VISUAL_HOLD','HOLD_REACQUIRE','CONFIRM_STOP'))
                            shaped_values=self.smoother.update(raw_values,command_time,active,
                                damped_yaw=bool(decision and decision.get('centering_control')),
                                damped_lateral=bool(decision and (decision.get('centering_control') or {}).get('lateral_control')))
                        if gate_reason.startswith('Kalman recovery /'):
                            # Do not let the smoothing history delay the coast
                            # caps/decay, or carry old roll past its deadline.
                            shaped_values=tuple(math.copysign(min(abs(shaped),abs(raw)),raw)
                                if shaped*raw>0 else 0. for shaped,raw in zip(shaped_values,raw_values))
                            self.smoother.values=shaped_values
                        if gate_reason.startswith('Nano bridge /'):
                            # Smoothing cannot retain translation beyond the
                            # verified forward bridge's cap or source deadline.
                            shaped_values=tuple(math.copysign(min(abs(shaped),abs(raw)),raw)
                                if axis in (0,3) and shaped*raw>0 else 0.
                                for axis,(shaped,raw) in enumerate(zip(shaped_values,raw_values)))
                            self.smoother.values=shaped_values
                        if gate_reason.startswith('Recovery vector /'):
                            shaped_values=tuple(math.copysign(min(abs(shaped),abs(raw)),raw)
                                if shaped*raw>0 else 0. for shaped,raw in zip(shaped_values,raw_values))
                            self.smoother.values=shaped_values
                    else:
                        self.smoother.reset(); shaped_values=raw_values
                    search_active=bool(mode=='DANCE' and wanted and self.enabled and not self.motion_hold
                                       and (edge_override or scan_override))
                    search_values=self.search_smoother.update(raw_values,command_time,search_active)
                    if search_active:shaped_values=(search_values[0],0.,raw_values[2],raw_values[3])
                    effective_limits=((1.,0.,0.,0.) if mode=='SEARCH_TEST' else
                        (1.,0.,limits[2],limits[3]) if edge_override or scan_override else limits)
                    values = tuple(value * limit
                                   for value, limit in zip(shaped_values, effective_limits))
                    # SearchSmoother may retain a larger earlier demand. Apply
                    # SCAN's independent cap after shaping and full-scale override.
                    scan_cap=None
                    if mode=='DANCE' and gate_reason.startswith('Scan /'):
                        scan_cap=(decision.get('edge_search') or {}).get('scan_yaw_limit',1.)
                        scan_cap=scan_cap if isinstance(scan_cap,(int,float)) and math.isfinite(scan_cap) and .10<=scan_cap<=1. else 0.
                        values=(max(-scan_cap,min(scan_cap,values[0])),*values[1:])
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
                    pending_log_key = None
                    motion_reduced=False
                    def accept_reduction(reduced):
                        nonlocal values,motion_reduced
                        with self.lock:
                            values=reduced;motion_reduced=True
                            # Seed from the transmitted reduction, preserving
                            # ramp timing instead of restarting all axes at zero.
                            self.smoother.values=tuple(v/limit if limit else 0.
                                for v,limit in zip(values,effective_limits))
                    applied=command('rc ' + ' '.join(f'{value:.4f}' for value in values),
                            motion_keys=command_keys if any(values) else None,
                            on_reduction=accept_reduction if mode=='DANCE' else None)
                    if not applied:
                        if channel.neutral_acknowledged and not channel.ambiguous:
                            self.last_acknowledged=dict(time=time.monotonic(),mode='MANUAL',values=[0.,0.,0.,0.],
                                reason='Neutral acknowledged')
                        continue  # Re-read current keys; never replay the superseded command.
                    self.last_acknowledged=dict(time=time.monotonic(),mode=mode,values=list(values),reason=gate_reason,
                        ack_ms=channel.last_ack['ack_ms'] if channel.last_ack else None)
                    self.on_event('flight_command', {'mode': mode, 'command': 'rc',
                        'selected_mode': selected_mode,
                        'target_policy':decision.get('target_policy') if decision else None,
                        'selected_class_id':decision.get('selected_class_id') if decision else None,
                        'selected_class_name':decision.get('selected_class_name') if decision else None,
                        'target_identity':decision.get('target_identity') if decision else None,
                        'target_class_source':decision.get('target_class_source') if decision else None,
                        'class_support':decision.get('class_support') if decision else None,
                        'nano':decision.get('nano') if decision else None,
                        'nano_authority':decision.get('nano_authority') if decision else None,
                        'nano_soft_veto_overridden':decision.get('nano_soft_veto_overridden') if decision else None,
                        'resume_measurement_after':self.resume_measurement_after,
                        'nano_handoff':decision.get('nano_handoff') if decision else None,
                        'real_neo_evidence':decision.get('real_neo_evidence') if decision else None,
                        'recovery_anchor':decision.get('recovery_anchor') if decision else None,
                        'recovery_motion':decision.get('recovery_motion') if decision else None,
                        'motion_settings':decision.get('motion_settings') if decision else None,
                        'transport':channel.diagnostics(),
                        'measurement_id':decision.get('measurement_id') if decision else None,
                        'measurement_time':decision.get('measurement_time') if decision else None,
                        'detection_summary':decision.get('detection_summary') if decision else None,
                        'axis_limits': dict(zip(AXES, limits)),
                        'effective_axis_limits':dict(zip(AXES,effective_limits)),
                        'edge_yaw_override':edge_override,
                        'search_yaw_override':bool(edge_override or scan_override or mode=='SEARCH_TEST'),
                        'search_yaw_ramped':search_active,
                        'scan_yaw_cap':scan_cap,
                        'motion_reduced':motion_reduced,
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
                        'forward_control':decision.get('forward_control') if decision else None,
                        'track_support':decision.get('track_support') if decision else None,
                        'nano_forward_source_time':decision.get('nano_forward_source_time') if decision else None,
                        'verified_forward_anchor':decision.get('verified_forward_anchor') if decision else None,
                        'shaped_values':list(shaped_values),
                        'policy_values':list(policy_values) if policy_values is not None else None,
                        'intent_basis':decision.get('intent_basis') if mode=='DANCE' and decision else None,
                        'values': list(values), 'server_acknowledged': True,
                        'android_stick_units': [math.trunc(value*660.) for value in values],
                        'android_quantized_to_zero': [axis for axis,value in zip(AXES,values)
                            if value and math.trunc(value*660.)==0],
                        'measurement_id': decision.get('measurement_id') if mode == 'DANCE' and decision else None})
                    with self.lock:
                        action_allowed = (self.wanted and not self.communication_paused and not self.stop_event.is_set()
                                          and time.monotonic() - self.updated <= .35)
                    if action and action_allowed:
                        if not command(action):continue
                        self.on_event('flight_command', {'mode': 'MANUAL', 'command': action,
                            'server_acknowledged': True})
                        self.status = f'{action}: success | Q / Esc: release'
                # 20 Hz start-to-start; ACK latency is already part of the
                # cycle. Do not add another unconditional 50 ms after it.
                self.stop_event.wait(max(0.,.05-(time.monotonic()-cycle_started)))
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
