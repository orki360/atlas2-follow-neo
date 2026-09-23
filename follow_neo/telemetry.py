"""Read-only heading queries on Yam's original TCP 9997 protocol.

No enable, set, action or aircraft configuration commands are sent.
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


class HeadingReceiver:
    def __init__(self,host,on_event=None,connector=socket.create_connection):
        self.host=host;self.connector=connector;self.on_event=on_event or (lambda *args:None)
        self.stop_event=threading.Event();self.lock=threading.Lock();self.sample=None
        self.sock=None;self.status='Heading: connecting TCP 9997'
        self.thread=threading.Thread(target=self._run,name='follow-heading',daemon=True)

    def get(self):
        with self.lock:return None if self.sample is None else dict(self.sample)

    def start(self):self.thread.start()

    def stop(self):
        self.stop_event.set()
        sock=self.sock
        if sock:
            try:sock.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            sock.close()
        if self.thread.ident:self.thread.join(timeout=2.)

    def _run(self):
        last_error=None
        while not self.stop_event.is_set():
            try:
                with self.connector((self.host,9997),timeout=.8) as sock:
                    self.sock=sock;buffer=b'';key='AircraftAttitude';last_log=0.
                    while not self.stop_event.is_set():
                        sent=time.monotonic();deadline=sent+.22
                        sock.sendall(f'get FlightController {key}\r\n'.encode('ascii'))
                        line=None
                        while line is None:
                            if b'\n' in buffer:
                                part,buffer=buffer.split(b'\n',1)
                                candidate=part.decode('utf-8',errors='replace').strip()
                                if candidate.startswith(f'FlightController {key} ') or candidate.startswith('Unknown'):
                                    line=candidate
                                    break
                                continue
                            if time.monotonic()>=deadline:raise TimeoutError('Heading response timed out')
                            sock.settimeout(max(.001,deadline-time.monotonic()))
                            chunk=sock.recv(4096)
                            if not chunk:raise ConnectionError('Heading connection closed')
                            buffer+=chunk
                            if len(buffer)>16384:raise ValueError('Heading response too long')
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
                with self.lock:self.sample=None
            self.stop_event.wait(1.)
