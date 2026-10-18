"""Bounded, receive-only timing data; never changes stream/control authority."""
from contextlib import contextmanager
import os
import threading
import time


class VideoDiagnostics:
    def __init__(self):
        self.lock=threading.Lock()
        self.phase='idle';self.phase_since=time.monotonic()
        self.since=self.phase_since;self.cpu_since=time.process_time()
        self.stages={};self.counters={};self.last_bytes=None
        self.active_stages={}
        self.last_keyframe=None;self.connection_started=None

    @contextmanager
    def measure(self,phase):
        started=time.monotonic();cpu=time.thread_time()
        worker=threading.get_ident()
        with self.lock:
            self.phase=phase;self.phase_since=started
            self.active_stages[worker]=(threading.current_thread().name,phase,started)
        try:yield
        finally:
            elapsed=max(0.,time.monotonic()-started)*1000
            cpu_ms=max(0.,time.thread_time()-cpu)*1000
            with self.lock:
                value=self.stages.setdefault(phase,dict(count=0,total_ms=0.,max_ms=0.,cpu_ms=0.))
                value['count']+=1;value['total_ms']+=elapsed
                value['max_ms']=max(value['max_ms'],elapsed);value['cpu_ms']+=cpu_ms
                self.active_stages.pop(worker,None)
                if self.active_stages:
                    _,self.phase,self.phase_since=max(self.active_stages.values(),key=lambda v:v[2])
                else:self.phase='idle';self.phase_since=time.monotonic()

    def count(self,name,value=1):
        with self.lock:self.counters[name]=self.counters.get(name,0)+value

    def connected(self,now):
        with self.lock:
            self.connection_started=now;self.last_bytes=None;self.last_keyframe=None

    def received(self,size,now):
        with self.lock:
            if self.last_bytes is not None:
                gap=max(0.,now-self.last_bytes)*1000
                self.counters['max_receive_gap_ms']=max(self.counters.get('max_receive_gap_ms',0.),gap)
            self.last_bytes=now
            self.counters['bytes']=self.counters.get('bytes',0)+size
            self.counters['receives']=self.counters.get('receives',0)+1

    def decoded(self,keyframe,now):
        with self.lock:
            self.counters['decoded']=self.counters.get('decoded',0)+1
            if keyframe is True:
                self.last_keyframe=now;self.counters['keyframes']=self.counters.get('keyframes',0)+1

    def snapshot(self,reset=False):
        now=time.monotonic();cpu=time.process_time()
        with self.lock:
            elapsed=max(0.,now-self.since)
            cores=max(0.,cpu-self.cpu_since)/elapsed if elapsed else 0.
            out=dict(window_started=self.since,window_seconds=elapsed,
                active_stages=[dict(worker=w,stage=p,age_ms=max(0.,now-t)*1000)
                               for w,p,t in self.active_stages.values()],
                stage=self.phase,stage_age_ms=max(0.,now-self.phase_since)*1000,
                stages={k:dict(v) for k,v in self.stages.items()},counters=dict(self.counters),
                connection_started=self.connection_started,last_bytes_at=self.last_bytes,
                last_keyframe_at=self.last_keyframe,
                process_cpu_cores=cores,process_cpu_machine_percent=cores/max(1,os.cpu_count() or 1)*100,
                cpu_scope='whole Python process; stage CPU is the executing worker thread only')
            if reset:
                self.since=now;self.cpu_since=cpu;self.stages={};self.counters={}
            return out
