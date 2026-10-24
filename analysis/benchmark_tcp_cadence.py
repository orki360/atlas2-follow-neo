"""Offline sender benchmark: socketpair only, never a phone/aircraft connection.

Measures the actual ManualControl writer with a simulated 5 ms server ACK.
The relative-clock baseline reproduces the previous start-to-start wait budget.
No tracker/inference/video load: these results do not guarantee field rates.
"""
import argparse
import json
from pathlib import Path
import socket
import statistics
import sys
import threading
import time
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from follow_neo.diagnostics import activity
from follow_neo.transport.manual import ManualControl


class PreviousCadence:
    def __init__(self):self.reset()
    def reset(self):self.started=None
    def wait(self,stop,hz):
        if self.started is not None:
            if stop.wait(max(0.,1/hz-(time.monotonic()-self.started))):return False
        self.started=time.monotonic()
        return not stop.is_set()


def measure(hz,seconds,ack_ms):
    client,server=socket.socketpair();arrivals=[];errors=[]
    def serve():
        try:
            with server,server.makefile('rb') as stream:
                for row in stream:
                    line=row.decode().strip()
                    if line.startswith('rc 0.'):
                        arrivals.append(time.perf_counter())
                        time.sleep(ack_ms/1000.)
                    server.sendall(b'success\r\n')
        except OSError:pass
    peer=threading.Thread(target=serve,daemon=True);peer.start()
    def connect(address,timeout):
        assert address==('127.0.0.1',9998)
        return client
    def event(name,data):
        if name in ('control_error','control_movement_paused'):errors.append((name,data))
    control=ManualControl('127.0.0.1',connect,on_event=event,send_hz=hz)
    control.update({'up'});control.enable();control.start()
    start=time.perf_counter()
    try:
        while time.perf_counter()-start<seconds+.5:
            control.update({'up'});time.sleep(.01)
    finally:
        control.stop();control.thread.join(4);peer.join(1)
    samples=[t for t in arrivals if t>=start+.5]
    gaps=[(b-a)*1000 for a,b in zip(samples,samples[1:])]
    if len(gaps)<3 or errors:raise RuntimeError(f'Incomplete run: {len(gaps)} gaps, {errors}')
    ordered=sorted(gaps)
    return dict(target_hz=hz,actual_hz=1000/statistics.mean(gaps),commands=len(samples),
        mean_ms=statistics.mean(gaps),median_ms=statistics.median(gaps),
        p95_ms=ordered[int(.95*(len(ordered)-1))],max_ms=max(gaps),min_ms=min(gaps))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=5.)
    parser.add_argument('--ack-ms',type=float,default=5.)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not 2<=args.seconds<=60 or not 0<=args.ack_ms<=100:parser.error('Invalid duration or ACK delay')
    old=activity.enabled;activity.set_enabled(False)
    try:
        with patch('follow_neo.transport.manual.SendCadence',PreviousCadence):
            baseline=measure(20,args.seconds,args.ack_ms)
        results=[measure(hz,args.seconds,args.ack_ms) for hz in (20,25,30)]
    finally:activity.set_enabled(old)
    data=dict(scope='Offline socketpair, synthetic ACK, no video/inference/aircraft',
        simulated_ack_ms=args.ack_ms,seconds_per_rate=args.seconds,
        baseline_relative_monotonic=baseline,absolute_perf_counter=results)
    text=json.dumps(data,indent=2)
    if args.output:args.output.write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':main()
