"""FIFO command acknowledgements with a latched pause, not an ACK disconnect.

MSDKRemote replies in order without command IDs. Late replies must remain paired
with their original commands; an ambiguous stream cannot confirm a neutral RC.
"""
import socket
import time
from collections import deque
import math
import secrets
from dataclasses import dataclass
from .socket_options import enable_nodelay

RC_ACK_TIMEOUT = .25
POLL_SECONDS = .02
KEY_CHANGE = 'movement keys changed while acknowledgement was pending'
VISUAL_CHANGE = 'visual authorization changed while acknowledgement was pending'
ZERO = 'rc 0 0 0 0'
UNEXPECTED_REPLY = 'Unexpected control reply; reply stream requires manual reconnect'
RESYNC_TIMEOUT = .50


@dataclass(frozen=True)
class MotionReduction:
    """Already authorized same-target reduction; transport rechecks the envelope."""
    values: tuple


def reduced_command(text,proposal):
    try:
        parts=text.split()
        if parts[0]!='rc' or len(parts)!=5:return None
        previous=tuple(map(float,parts[1:]));values=tuple(proposal.values)
        if (len(values)!=4 or not any(values) or values==previous
                or not all(math.isfinite(a) and math.isfinite(b) and abs(a)<=1.
                           and (b==0 or a*b>0 and abs(b)<=abs(a))
                           for a,b in zip(previous,values))):return None
        return 'rc '+' '.join(f'{math.trunc(v*10000)/10000:.4f}' for v in values)
    except (ValueError,TypeError,IndexError):return None


class ControlChannel:
    def __init__(self, sock, guard, emit):
        self.sock, self.guard, self.emit = sock, guard, emit
        self.tcp_nodelay = enable_nodelay(sock)
        self.buffer = b''
        self.sequence = 0
        self.outstanding = []
        self.paused = False
        self.pause_reason = ''
        self.ambiguous = False
        self.neutral_sent = False
        self.neutral_acknowledged = False
        self.rejected = False
        self.last_ack = None
        self.ack_count = self.late_rc_ack_count = 0
        self.rc_ack_samples = deque(maxlen=120)
        self.unexpected_success_only = False
        self.resync_attempted = self.resync_confirmed = False

    def diagnostics(self, now=None):
        """Local send-to-reply timings, independent of motion authorization."""
        now=time.monotonic() if now is None else now
        samples=sorted(self.rc_ack_samples)
        return dict(last_ack=dict(self.last_ack) if self.last_ack else None,
            tcp_nodelay=self.tcp_nodelay, acknowledgement_scope='android_handler_not_aircraft_motion',
            ack_count=self.ack_count,late_rc_ack_count=self.late_rc_ack_count,
            rc_ack_p95_ms=samples[math.ceil(.95*len(samples))-1] if samples else None,
            rc_ack_sample_count=len(samples),rc_ack_limit_ms=RC_ACK_TIMEOUT*1000,
            pending_count=len(self.outstanding),
            oldest_pending_sent_at=self.outstanding[0][2] if self.outstanding else None,
            oldest_pending_age_ms=max(0.,now-self.outstanding[0][2])*1000 if self.outstanding else None,
            paused=self.paused,ambiguous=self.ambiguous,
            neutral_acknowledged=self.neutral_acknowledged,
            resync_attempted=self.resync_attempted,resync_confirmed=self.resync_confirmed)

    @property
    def pending(self):
        return bool(self.outstanding)

    def _send(self, value, *, neutral=False):
        self.sequence += 1
        stamp = time.monotonic()
        self.outstanding.append((self.sequence, value, stamp, neutral))
        if neutral:
            self.neutral_sent = True
            self.neutral_acknowledged = False
        self.emit('control_command_attempt', dict(sequence=self.sequence, command=value, sent_at=stamp))
        self.sock.settimeout(.10)
        try:
            self.sock.sendall((value+'\r\n').encode('ascii'))
        except socket.timeout:
            # A partial send has unknown framing. Keep the connection, but do
            # not claim later bytes acknowledge any particular command.
            self.paused = self.ambiguous = True
            self.pause_reason = 'Control send timed out; reply stream requires manual reconnect'
            self.emit('control_paused', dict(reason=self.pause_reason, connection_kept=True,
                                            server_acknowledged=False, ambiguous=True))

    def pause(self, reason, *, ambiguous=False):
        first = not self.paused
        self.paused = True
        self.ambiguous = self.ambiguous or ambiguous
        if self.ambiguous:self.neutral_acknowledged=False
        if first or ambiguous:
            self.pause_reason = reason
            self.emit('control_paused', dict(reason=reason, connection_kept=True,
                                            server_acknowledged=False, ambiguous=self.ambiguous,
                                            transport=self.diagnostics()))
        if not self.neutral_sent:
            self._send(ZERO, neutral=True)

    def _unmatched(self, data):
        lines=[line.strip() for line in data.splitlines() if line.strip()]
        self.unexpected_success_only = bool(lines and len(data)<=8192 and data.endswith(b'\n')
            and all(line==b'success' for line in lines))
        self.emit('control_unmatched_reply', dict(raw_hex=data[:1024].hex(), byte_count=len(data),
                                                outstanding=len(self.outstanding),
                                                success_only=self.unexpected_success_only,
                                                last_ack=self.last_ack))
        self.buffer = b''
        self.pause(UNEXPECTED_REPLY, ambiguous=True)

    @property
    def can_resync(self):
        return (self.paused and self.ambiguous and not self.rejected and
                self.pause_reason==UNEXPECTED_REPLY and self.unexpected_success_only
                and not self.resync_attempted)

    def synchronize_neutral(self, allowed):
        """One bounded recovery on the inspected MSDKRemote line protocol.

        Its reader handles RC synchronously, then echoes an unknown command's
        first word without performing an action. A fresh random echo AFTER a
        neutral line proves that neutral was processed on this same connection.
        Generic success replies never confirm this barrier. No enable/reconnect
        occurs here; the caller must still check authority and fresh evidence.
        """
        if not self.can_resync or not allowed():return False
        self.resync_attempted=True
        marker='python_sync_'+secrets.token_hex(16)
        expected=('Unknown command: '+marker).encode('ascii')
        started=time.monotonic();deadline=started+RESYNC_TIMEOUT
        self.emit('control_resync_started',dict(marker=marker,deadline=deadline,
            pending_sequences=[p[0] for p in self.outstanding],motion_held=True))
        self._send(ZERO,neutral=True)
        if self.pause_reason!=UNEXPECTED_REPLY:return False  # partial send is not recoverable
        self._send(marker)
        if self.pause_reason!=UNEXPECTED_REPLY:return False
        buffer=b'';discarded=0;failure='marker_timeout'
        while time.monotonic()<deadline:
            if not allowed():failure='operator_or_heartbeat_changed';break
            remaining=deadline-time.monotonic()
            if remaining<=0:break
            self.sock.settimeout(min(POLL_SECONDS,remaining))
            try:chunk=self.sock.recv(4096)
            except socket.timeout:continue
            if not chunk:raise ConnectionError('Control server disconnected during synchronization')
            buffer+=chunk
            if len(buffer)>8192:failure='oversized_reply';break
            while b'\n' in buffer:
                raw,buffer=buffer.split(b'\n',1);reply=raw.strip()
                if not reply:continue
                if reply==expected:
                    if buffer.strip() or not allowed():
                        failure='data_after_marker_or_operator_changed';break
                    # Do not assign quarantined successes to old sequence IDs.
                    old=[p[0] for p in self.outstanding]
                    self.outstanding.clear();self.buffer=b''
                    self.ambiguous=False;self.neutral_acknowledged=True
                    self.resync_confirmed=True
                    self.emit('control_resync_completed',dict(marker=marker,
                        elapsed_ms=(time.monotonic()-started)*1000,
                        quarantined_successes=discarded,retired_sequences=old,
                        neutral_confirmed_by='ordered_server_marker',motion_held=True))
                    return True
                self.emit('control_resync_quarantined_reply',dict(raw_hex=raw[:1024].hex()))
                if reply!=b'success':failure='unexpected_or_rejected_reply';break
                discarded+=1
            else:continue
            break
        self.emit('control_resync_failed',dict(reason=failure,
            elapsed_ms=(time.monotonic()-started)*1000,quarantined_successes=discarded,
            manual_enable_required=True))
        return False

    def _consume(self):
        while b'\n' in self.buffer:
            raw, self.buffer = self.buffer.split(b'\n', 1)
            if not raw.strip():
                continue
            if self.ambiguous or not self.outstanding:
                self._unmatched(raw+b'\n'+self.buffer)
                return
            seq, value, sent, neutral = self.outstanding.pop(0)
            reply = raw.decode('utf-8', errors='replace').strip()
            received=time.monotonic();ack_ms=max(0.,received-sent)*1000
            rc=value.startswith('rc ')
            self.last_ack=dict(sequence=seq,command=value,reply=reply,sent_at=sent,
                received_at=received,ack_ms=ack_ms,neutral=neutral,
                timeout_ms=(RC_ACK_TIMEOUT if rc else 2.)*1000,
                late=ack_ms>(RC_ACK_TIMEOUT if rc else 2.)*1000)
            self.ack_count+=1
            if rc:
                self.rc_ack_samples.append(ack_ms);self.late_rc_ack_count+=int(self.last_ack['late'])
            self.emit('control_command_ack',dict(self.last_ack,pending_after_reply=len(self.outstanding)))
            if reply != 'success':
                self.rejected = True
                # Bytes already received cannot acknowledge the neutral which
                # pause() is about to send.
                if self.buffer.strip():
                    self._unmatched(self.buffer)
                else:
                    self.buffer = b''
                    self.pause('Control command rejected: '+reply)
                return
            if neutral:
                self.neutral_acknowledged = True
                self.emit('control_motion_neutralized', dict(server_acknowledged=True))
        if not self.outstanding and self.buffer.strip():
            self._unmatched(self.buffer)
        elif not self.buffer.strip():
            self.buffer = b''

    def _receive(self, timeout=POLL_SECONDS):
        self.sock.settimeout(timeout)
        try:
            chunk = self.sock.recv(4096)
        except socket.timeout:
            return
        if not chunk:
            raise ConnectionError('Control server disconnected')
        if self.ambiguous:
            self.emit('control_quarantined_reply', dict(raw_hex=chunk[:1024].hex(), byte_count=len(chunk)))
            return
        self.buffer += chunk
        if len(self.buffer) > 8192:
            self._unmatched(self.buffer)
        else:
            self._consume()

    def poll_pause(self):
        """Drain known late replies without sending new movement or reconnecting."""
        self._receive()

    def resume(self):
        if (not self.paused or self.ambiguous or self.pending or self.buffer.strip()
                or not self.neutral_acknowledged):
            return False
        self.paused = False
        self.pause_reason = ''
        self.neutral_sent = self.neutral_acknowledged = False
        self.rejected = False
        return True

    def command(self, text, *, interruptible=True, motion_keys=None, on_reduction=None):
        if self.paused:
            return False
        neutralized = False
        reductions=0
        reason = self.guard(motion_keys) if interruptible else None
        if isinstance(reason,MotionReduction):
            replacement=reduced_command(text,reason) if on_reduction else None
            if replacement:
                original=text;text=replacement;reductions=1;reason=None
            else:reason=VISUAL_CHANGE
        if reason in (KEY_CHANGE,VISUAL_CHANGE) and text.startswith('rc '):
            text = ZERO
            motion_keys = None
            neutralized = True
        elif reason:
            self.pause(reason)
            return False
        if self.pending or self.buffer.strip():
            self.pause('Unexpected outstanding control reply', ambiguous=True)
            return False
        self.neutral_sent = self.neutral_acknowledged = False
        timeout = RC_ACK_TIMEOUT if text.startswith('rc ') else 2.
        deadline = time.monotonic()+timeout
        self._send(text, neutral=neutralized)
        if reductions and not self.paused:
            on_reduction(tuple(map(float,text.split()[1:])))
            self.emit('control_motion_reduced',dict(previous_command=original,command=text,
                before_send=True,deadline=deadline,ack_deadline_extended=False))
        while self.outstanding and not self.paused:
            reason = self.guard(None if neutralized else motion_keys) if interruptible else None
            if isinstance(reason,MotionReduction):
                replacement=(reduced_command(text,reason)
                             if not neutralized and not reductions and on_reduction
                             and time.monotonic()<deadline else None)
                if replacement:
                    original=text;text=replacement;reductions+=1
                    self._send(text)
                    if not self.paused:on_reduction(tuple(map(float,text.split()[1:])))
                    self.emit('control_motion_reduced',dict(previous_command=original,command=text,
                        before_send=False,deadline=deadline,ack_deadline_extended=False))
                    reason=None
                else:reason=VISUAL_CHANGE
            if reason in (KEY_CHANGE,VISUAL_CHANGE) and text.startswith('rc ') and not neutralized:
                neutralized = True
                self._send(ZERO, neutral=True)
                self.emit('control_motion_neutralizing', dict(reason='visual_authorization_changed' if reason==VISUAL_CHANGE else 'movement_keys_changed', deadline=deadline))
            elif reason:
                self.pause(reason)
                return False
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                self.pause(f'Control acknowledgement delayed beyond {timeout:.2f}s')
                return False
            self._receive(min(POLL_SECONDS, remaining))
        return not self.paused and not neutralized

    def release_control(self):
        """Explicit operator release while replies are still pending."""
        self._send(ZERO, neutral=True)
        self._send('disable')
        self.emit('control_release_requested', dict(server_acknowledged=False, connection_kept=True))

    def stop_unconfirmed(self, reason):
        delivered = False
        try:
            self.sock.settimeout(.10)
            self.sock.sendall(b'rc 0 0 0 0\r\ndisable\r\n')
            delivered = True
        except OSError:
            pass
        self.emit('control_stop_unconfirmed', dict(reason=reason, bytes_sent=delivered,
                                                   server_acknowledged=False))
