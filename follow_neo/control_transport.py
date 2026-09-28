"""Interruptible command acknowledgements; no reconnect or command replay.

After an interrupted exchange, bare `success` replies cannot be correlated.
Send a best-effort neutral/disable pair and close that channel without claiming
remote confirmation. A receiver-side watchdog is still required for link loss.
"""
import socket
import time

RC_ACK_TIMEOUT = .25
POLL_SECONDS = .02


class ControlChannel:
    def __init__(self, sock, guard, emit):
        self.sock, self.guard, self.emit = sock, guard, emit
        self.buffer = b''
        self.pending = False
        self.sequence = 0

    def command(self, text, *, interruptible=True, motion_keys=None):
        timeout = RC_ACK_TIMEOUT if text.startswith('rc ') else 2.
        def check():
            if interruptible:
                reason = self.guard(motion_keys)
                if reason:
                    raise RuntimeError('Control interrupted: '+reason)
        check()
        self.sequence += 1
        seq = self.sequence
        started = time.monotonic()
        self.emit('control_command_attempt', {'sequence':seq, 'command':text, 'sent_at':started})
        self.pending = True
        self.sock.settimeout(.10)
        self.sock.sendall((text+'\r\n').encode('ascii'))
        deadline = started+timeout
        while b'\n' not in self.buffer:
            check()
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f'Control acknowledgement exceeded {timeout:.2f}s')
            self.sock.settimeout(min(POLL_SECONDS,remaining))
            try:
                chunk = self.sock.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                raise ConnectionError('Control server disconnected')
            self.buffer += chunk
            if len(self.buffer)>8192:
                raise RuntimeError('Control reply too long')
        reply,self.buffer=self.buffer.split(b'\n',1)
        self.pending=False
        reply=reply.decode('utf-8',errors='replace').strip()
        self.emit('control_command_ack', {'sequence':seq,'command':text,'reply':reply,
                  'ack_ms':(time.monotonic()-started)*1000})
        if reply!='success':raise RuntimeError(reply)

    def stop_unconfirmed(self, reason):
        delivered=False
        try:
            # Do not wait for a possibly missing zero acknowledgement before
            # sending disable; the interrupted reply stream is never reused.
            self.sock.settimeout(.10)
            self.sock.sendall(b'rc 0 0 0 0\r\ndisable\r\n')
            delivered=True
        except OSError:
            pass
        self.emit('control_stop_unconfirmed', {'reason':reason,
                  'bytes_sent':delivered,'server_acknowledged':False})
