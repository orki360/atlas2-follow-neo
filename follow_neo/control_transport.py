"""FIFO command acknowledgements with a latched pause, not an ACK disconnect.

MSDKRemote replies in order without command IDs. Late replies must remain paired
with their original commands; an ambiguous stream cannot confirm a neutral RC.
"""
import socket
import time

RC_ACK_TIMEOUT = .25
POLL_SECONDS = .02
KEY_CHANGE = 'movement keys changed while acknowledgement was pending'
ZERO = 'rc 0 0 0 0'


class ControlChannel:
    def __init__(self, sock, guard, emit):
        self.sock, self.guard, self.emit = sock, guard, emit
        self.buffer = b''
        self.sequence = 0
        self.outstanding = []
        self.paused = False
        self.pause_reason = ''
        self.ambiguous = False
        self.neutral_sent = False
        self.neutral_acknowledged = False
        self.rejected = False

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
        if first or ambiguous:
            self.pause_reason = reason
            self.emit('control_paused', dict(reason=reason, connection_kept=True,
                                            server_acknowledged=False, ambiguous=self.ambiguous))
        if not self.neutral_sent:
            self._send(ZERO, neutral=True)

    def _unmatched(self, data):
        self.emit('control_unmatched_reply', dict(raw_hex=data[:1024].hex(), byte_count=len(data),
                                                outstanding=len(self.outstanding)))
        self.buffer = b''
        self.pause('Unexpected control reply; reply stream requires manual reconnect', ambiguous=True)

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
            self.emit('control_command_ack', dict(sequence=seq, command=value, reply=reply,
                                                  ack_ms=(time.monotonic()-sent)*1000))
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

    def command(self, text, *, interruptible=True, motion_keys=None):
        if self.paused:
            return False
        neutralized = False
        reason = self.guard(motion_keys) if interruptible else None
        if reason == KEY_CHANGE and text.startswith('rc '):
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
        while self.outstanding and not self.paused:
            reason = self.guard(None if neutralized else motion_keys) if interruptible else None
            if reason == KEY_CHANGE and text.startswith('rc ') and not neutralized:
                neutralized = True
                self._send(ZERO, neutral=True)
                self.emit('control_motion_neutralizing', dict(reason='movement_keys_changed', deadline=deadline))
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
