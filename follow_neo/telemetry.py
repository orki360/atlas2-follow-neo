"""Heading queries and explicit gimbal pitch steps on Yam's TCP 9997 protocol.

Heading polling remains read-only. The optional GimbalKeyboard facade submits
one operator-triggered pitch step using this same single-client connection.
No aircraft enable, movement or configuration commands are sent here.
Times are conservative local request times, not aircraft timestamps.
"""
import math
import re
import socket
import threading
import time

HEADING_MAX_AGE=.25
NUMBER=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'


def parse_heading(line):
    parts=line.strip().split(' ',2)
    if len(parts)!=3 or parts[0]!='FlightController':return None
    key,value=parts[1:]
    if key=='CompassHeading':
        if not re.fullmatch(NUMBER,value.strip()):return None
        yaw=float(value)
    elif key=='AircraftAttitude':
        match=re.search(r'\byaw[\"\x27]?\s*[:=]\s*[\"\x27]?('+NUMBER+r')(?=[\s,}\"\x27]|$)',value,re.I)
        if not match:return None
        yaw=float(match.group(1))
    else:return None
    if not math.isfinite(yaw) or not -180<=yaw<=180:return None
    return yaw,key


def fresh_heading(sample,now):
    return (isinstance(sample,dict) and isinstance(sample.get('time'),(int,float))
            and isinstance(sample.get('yaw_deg'),(int,float))
            and math.isfinite(sample['time']) and math.isfinite(sample['yaw_deg'])
            and -180<=sample['yaw_deg']<=180 and 0<=now-sample['time']<=HEADING_MAX_AGE)


def heading_snapshot(receiver,clock=time.monotonic):
    """Read the sample BEFORE its comparison time, even during concurrent updates."""
    sample=receiver.get()
    return sample,clock()


class HeadingReceiver:
    def __init__(self,host,on_event=None,connector=socket.create_connection):
        self.host=host;self.connector=connector;self.on_event=on_event or (lambda *args:None)
        self.stop_event=threading.Event();self.lock=threading.Lock();self.sample=None
        self.sock=None;self.status='Heading: connecting TCP 9997'
        self._gimbal_lock=threading.Lock();self._gimbal_request=None
        self._gimbal_unresolved=False
        self.thread=threading.Thread(target=self._run,name='follow-heading',daemon=True)

    def get(self):
        with self.lock:return None if self.sample is None else dict(self.sample)

    def start(self):self.thread.start()

    def stop(self):
        self.stop_event.set()
        self._abandon_gimbal('Query receiver stopped')
        sock=self.sock
        if sock:
            try:sock.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            sock.close()
        if self.thread.ident:self.thread.join(timeout=2.)

    def gimbal_available(self):
        with self._gimbal_lock:
            return (self.sock is not None and not self.stop_event.is_set()
                    and not self._gimbal_unresolved and self._gimbal_request is None)

    def queue_gimbal_step(self,request):
        """Accept one short-lived step, never hold it for a future connection."""
        from .gimbal_keyboard import GimbalRequest
        if not isinstance(request,GimbalRequest):return False
        with self._gimbal_lock:
            if (self.sock is None or self.stop_event.is_set()
                    or self._gimbal_unresolved or self._gimbal_request is not None):
                return False
            self._gimbal_request=request
            return True

    def _abandon_gimbal(self,reason):
        with self._gimbal_lock:
            request=self._gimbal_request;self._gimbal_request=None
        if request is not None:
            with request.lock:in_flight=request.phase=='in_flight'
            if in_flight:
                with self._gimbal_lock:self._gimbal_unresolved=True
            request.finish('unknown' if in_flight else 'error',
                           reason+('; completion unknown, no retry' if in_flight
                                   else '; unsent step discarded'))

    def _response(self,sock,buffer,prefix,deadline):
        """Read CRLF frames, retaining fragments and ignoring unrelated keys."""
        while not self.stop_event.is_set():
            if b'\n' in buffer:
                part,buffer=buffer.split(b'\n',1)
                candidate=part.decode('utf-8',errors='replace').strip()
                if candidate.startswith(prefix) or candidate.startswith('Unknown'):
                    return candidate,buffer
                if candidate.startswith('Gimbal RotateByAngle '):
                    # A timed-out action has no request ID. Consume its late
                    # result before permitting a NEW explicit operator step.
                    with self._gimbal_lock:self._gimbal_unresolved=False
                continue
            now=time.monotonic()
            if now>=deadline:
                error=TimeoutError('Query response timed out')
                error.query_buffer=buffer
                raise error
            sock.settimeout(max(.001,deadline-now))
            try:chunk=sock.recv(4096)
            except TimeoutError as error:
                error.query_buffer=buffer
                raise
            if not chunk:raise ConnectionError('Query connection closed')
            buffer+=chunk
            if len(buffer)>16384:raise ValueError('Query response too long')
        raise ConnectionError('Query receiver stopped')

    def _run_gimbal_step(self,sock,buffer):
        with self._gimbal_lock:request=self._gimbal_request
        if request is None:return buffer
        committed=False
        try:
            # Complete every potentially blocking setup step before committing
            # the action. Disabling during setup still cancels without a send.
            sock.settimeout(1.)
            with self._gimbal_lock:self._gimbal_unresolved=True
            with self.lock:self.sample=None
            if self.stop_event.is_set():request.cancel()
            committed=request.begin_send(time.monotonic())
            if not committed:
                with self._gimbal_lock:self._gimbal_unresolved=False
                return buffer
            # Nothing that waits or acquires a lock belongs between this commit
            # and sendall. A committed step can finish after a concurrent disable;
            # holding the request lock during I/O would block the GUI instead.
            sock.sendall(request.command)
            line,buffer=self._response(sock,buffer,'Gimbal RotateByAngle ',time.monotonic()+1.)
            with self._gimbal_lock:self._gimbal_unresolved=False
            if line=='Gimbal RotateByAngle success':
                request.finish('acknowledged','Gimbal step acknowledged; measured angle unavailable')
            else:
                request.finish('rejected','Gimbal rejected: '+line[:180])
        except (OSError,ValueError,RuntimeError) as exc:
            buffer=getattr(exc,'query_buffer',buffer)
            if committed:
                request.finish('unknown','Gimbal completion unknown: '+str(exc)+'; no retry')
            else:
                with self._gimbal_lock:self._gimbal_unresolved=False
                request.finish('error','Gimbal setup failed; unsent step discarded: '+str(exc))
            # Keep the query owner alive when possible. A late response is
            # drained by heading polling; no action is replayed on reconnect.
            if not isinstance(exc,TimeoutError):raise
        finally:
            with self._gimbal_lock:
                if self._gimbal_request is request:self._gimbal_request=None
        return buffer

    def _run(self):
        last_error=None
        while not self.stop_event.is_set():
            try:
                with self.connector((self.host,9997),timeout=.8) as sock:
                    self.sock=sock;buffer=b'';key='AircraftAttitude';last_log=0.
                    while not self.stop_event.is_set():
                        buffer=self._run_gimbal_step(sock,buffer)
                        if self.stop_event.is_set():break
                        sent=time.monotonic();deadline=sent+.22
                        sock.sendall(f'get FlightController {key}\r\n'.encode('ascii'))
                        line,buffer=self._response(sock,buffer,f'FlightController {key} ',deadline)
                        parsed=parse_heading(line)
                        if parsed is None:
                            if key=='AircraftAttitude':
                                key='CompassHeading';continue
                            raise ValueError('Heading unavailable: '+line[:140])
                        received=time.monotonic();yaw,source=parsed
                        sample=dict(yaw_deg=yaw,time=sent,received_at=received,source=source,
                                    timestamp_origin='local_request_monotonic',query_ms=(received-sent)*1000.)
                        with self.lock:self.sample=sample
                        self.status=f'Heading: {yaw:.1f} deg / {source}'
                        last_error=None
                        if received-last_log>=.10:
                            self.on_event('heading',sample);last_log=received
                        self.stop_event.wait(max(0.,.05-(received-sent)))
            except (OSError,ValueError,RuntimeError) as exc:
                if self.stop_event.is_set():break
                self.status='Heading unavailable: '+str(exc)
                if self.status!=last_error:
                    self.on_event('heading_unavailable',{'reason':str(exc)});last_error=self.status
            finally:
                self.sock=None
                self._abandon_gimbal('Query connection ended')
                with self.lock:self.sample=None
            self.stop_event.wait(1.)
