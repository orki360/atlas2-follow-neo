"""Read-only heading queries on Yam's original TCP 9997 protocol.

No enable, set, action or aircraft configuration commands are sent.
Times are conservative local request times, not aircraft timestamps.
"""
import math
import re
import socket
import threading
import time
from contextlib import contextmanager
from .socket_options import enable_nodelay

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
    def __init__(self,host,on_event=None,connector=socket.create_connection,prefer_listen=True):
        self.host=host;self.connector=connector;self.on_event=on_event or (lambda *args:None)
        self.stop_event=threading.Event();self.lock=threading.Lock();self.sample=None
        self.sock=None;self.status='Heading: connecting TCP 9997'
        self.prefer_listen=prefer_listen
        self.subscription_key=None
        self.thread=threading.Thread(target=self._run,name='follow-heading',daemon=True)

    def get(self):
        with self.lock:return None if self.sample is None else dict(self.sample)

    def start(self):self.thread.start()

    def stop(self):
        self.stop_event.set()
        # The bounded receive wakes normally so UNLISTEN can be sent before
        # closing. Force-close only if the worker cannot finish promptly.
        if self.thread.ident:self.thread.join(timeout=.35)
        sock=self.sock
        if sock:
            try:sock.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            sock.close()
        if self.thread.ident:self.thread.join(timeout=2.)

    @contextmanager
    def _connection(self):
        sock=self.connector((self.host,9997),timeout=.8)
        self.sock=sock;self.subscription_key=None
        try:yield sock
        finally:
            if self.subscription_key:
                try:
                    sock.settimeout(.05)
                    sock.sendall(f'unlisten FlightController {self.subscription_key}\r\n'.encode('ascii'))
                except OSError:pass
            self.subscription_key=None
            sock.close()

    def _run(self):
        last_error=None;failures=0;sent=None
        while not self.stop_event.is_set():
            sent=None
            try:
                with self._connection() as sock:
                    self.sock=sock;buffer=b'';key='AircraftAttitude';last_log=0.
                    nodelay=enable_nodelay(sock)
                    listening=False;listen_attempted=False;last_sample=None;pending=None
                    # GET establishes a baseline. LISTEN only supplies readings;
                    # this connection never sends SET, ACTION or flight commands.
                    self.on_event('heading_transport',dict(tcp_nodelay=nodelay,prefer_listen=self.prefer_listen))
                    while not self.stop_event.is_set():
                        now=time.monotonic()
                        if pending is None and (last_sample is None or not listening or now-last_sample>=.10):
                            sent=now;pending=sent
                            sock.sendall(f'get FlightController {key}\r\n'.encode('ascii'))
                        deadline=(pending+.22) if pending is not None else now+.10
                        line=None
                        while line is None:
                            if b'\n' in buffer:
                                part,buffer=buffer.split(b'\n',1)
                                candidate=part.decode('utf-8',errors='replace').strip()
                                if candidate.startswith(f'FlightController {key} ') or candidate.startswith('Unknown'):
                                    line=candidate
                                    break
                                continue
                            if time.monotonic()>=deadline:break
                            sock.settimeout(max(.001,deadline-time.monotonic()))
                            try:chunk=sock.recv(4096)
                            except socket.timeout:break
                            if not chunk:raise ConnectionError('Heading connection closed')
                            buffer+=chunk
                            if len(buffer)>16384:raise ValueError('Heading response too long')
                        if line is None:
                            if pending is not None and time.monotonic()>=pending+.22:
                                raise TimeoutError('Heading response timed out')
                            continue
                        # A LISTEN rejection is not a bad heading or a reason to
                        # disconnect: the original GET path remains available.
                        if "LISTEN" in line.upper() or line.startswith('Unknown command'):
                            listening=False
                            self.on_event('heading_listen_fallback',dict(reason=line[:140]))
                            continue
                        if line.endswith(' success'):continue  # UNLISTEN acknowledgement
                        parsed=parse_heading(line)
                        if parsed is None:
                            if key=='AircraftAttitude':
                                key='CompassHeading';pending=None;last_sample=None;continue
                            raise ValueError('Heading unavailable: '+line[:140])
                        received=time.monotonic();yaw,source=parsed
                        # Push messages have no aircraft timestamp. Reserve a
                        # transport margin; do not label receipt as capture time.
                        stamp=min(pending,received-.10) if pending is not None else received-.10
                        delivery='listen_or_get' if listening and pending is not None else 'listen' if listening else 'get'
                        sample=dict(yaw_deg=yaw,time=stamp,received_at=received,source=source,
                                    timestamp_origin='local_request_or_margin_not_aircraft',delivery=delivery,
                                    query_ms=None if pending is None else (received-pending)*1000.)
                        pending=None;last_sample=received
                        with self.lock:self.sample=sample
                        self.status=f'Heading: {yaw:.1f} deg / {source}'
                        last_error=None;failures=0
                        if received-last_log>=.10:
                            self.on_event('heading',sample);last_log=received
                        if self.prefer_listen and not listen_attempted:
                            # Cancel prior registrations after reconnect before
                            # subscribing once. The original server retains them.
                            sock.sendall(f'unlisten FlightController {key}\r\nlisten FlightController {key}\r\n'.encode('ascii'))
                            listening=True;listen_attempted=True
                            self.subscription_key=key
                            self.on_event('heading_listen_requested',dict(key=key))
                        # The conservative sample timestamp is not the GET send
                        # time. Using it here would remove the polling cadence.
                        if not listening:
                            query_seconds=(sample['query_ms'] or 0.)/1000.
                            self.stop_event.wait(max(0.,.05-query_seconds))
            except (OSError,ValueError,RuntimeError) as exc:
                if self.stop_event.is_set():break
                failures+=1
                self.status='Heading unavailable: '+str(exc)
                if self.status!=last_error:
                    self.on_event('heading_unavailable',{'reason':str(exc),'consecutive_failures':failures,
                        'request_started_at':sent,'observed_at':time.monotonic(),
                        'retry_seconds':heading_retry_seconds(failures)});last_error=self.status
            finally:
                self.sock=None
                with self.lock:self.sample=None
            self.stop_event.wait(heading_retry_seconds(failures))


def heading_retry_seconds(failures):
    """A transient timeout need not impose a full extra second without heading."""
    return min(1.,.20*2**min(3,max(0,failures-1)))
