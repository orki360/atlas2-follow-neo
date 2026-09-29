"""Bounded acknowledgements and FIFO neutralization of manual RC changes.

MSDKRemote handles RC synchronously and queues one reply per line in order.
Only RC may have a neutral replacement outstanding. Both replies must arrive
before the ORIGINAL deadline; a missing reply always closes the channel.
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
        changed='movement keys changed while acknowledgement was pending'
        neutralized=False
        reason=self.guard(motion_keys) if interruptible else None
        if reason==changed and text.startswith('rc '):
            text='rc 0 0 0 0';motion_keys=None;neutralized=True
        elif reason:raise RuntimeError('Control interrupted: '+reason)
        if self.pending or self.buffer:
            raise RuntimeError('Unexpected outstanding control reply')
        started = time.monotonic()
        outstanding=[]
        def send(value):
            self.sequence+=1;stamp=time.monotonic()
            outstanding.append((self.sequence,value,stamp))
            self.emit('control_command_attempt',{'sequence':self.sequence,'command':value,'sent_at':stamp})
            self.pending=True;self.sock.settimeout(.10)
            self.sock.sendall((value+'\r\n').encode('ascii'))
        send(text)
        deadline = started+timeout
        while outstanding:
            reason=self.guard(None if neutralized else motion_keys) if interruptible else None
            if reason==changed and text.startswith('rc ') and not neutralized:
                # Deliver zero immediately; an old success can acknowledge ONLY
                # its original RC. Do not resume until the separate zero ACK arrives.
                neutralized=True
                send('rc 0 0 0 0')
                self.emit('control_motion_neutralizing',{'reason':'movement_keys_changed','deadline':deadline})
            elif reason:raise RuntimeError('Control interrupted: '+reason)
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f'Control acknowledgement exceeded {timeout:.2f}s')
            if b'\n' in self.buffer:
                reply,self.buffer=self.buffer.split(b'\n',1)
                seq,value,sent=outstanding.pop(0)
                reply=reply.decode('utf-8',errors='replace').strip()
                self.emit('control_command_ack',{'sequence':seq,'command':value,'reply':reply,
                    'ack_ms':(time.monotonic()-sent)*1000})
                if reply!='success':raise RuntimeError(reply)
                continue
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
        self.pending=False
        if self.buffer:raise RuntimeError('Unexpected extra control reply')
        if neutralized:self.emit('control_motion_neutralized',{'server_acknowledged':True})
        return not neutralized

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
