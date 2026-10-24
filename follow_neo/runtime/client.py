"""GUI-side process client. It never opens a phone connection."""
import itertools
import multiprocessing as mp
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace

from ..ipc.mailbox import Mailbox,FrameSlot,InputState
from follow_neo.transport.manual import DEFAULT_AXIS_LIMITS


class Flag:
    def __init__(self,value=False):self.value=value
    def is_set(self):return self.value
    def set(self):self.value=True
    def clear(self):self.value=False


class Value:
    def __init__(self,value=None):self.value=value
    def get(self):return self.value


class Alive:
    def __init__(self,owner):self.owner=owner
    def is_alive(self):return self.owner.alive and self.owner.client.alive


class RuntimeClient:
    def __init__(self,worker_target=None):
        self.worker_target=worker_target
        self.ctx=mp.get_context('spawn');self.process=None;self.error=None
        self.inputs=InputState(self.ctx);self.snapshots=Mailbox(self.ctx)
        # Two full-HD BGR slots (12.4 MB total), not a frame queue.
        self.live=FrameSlot(self.ctx);self.analyzed=FrameSlot(self.ctx)
        self.sequence=itertools.count(1);self.pending={};self.pending_lock=threading.Lock()
        self.outgoing=queue.Queue(maxsize=32);self.last_snapshot=-1
        self.session=None;self.manual=None;self.closed=False;self.last_metrics=0.
        self.minimum_snapshot_time=0.

    @property
    def alive(self):return bool(self.process and self.process.is_alive() and not self.error)

    def ensure(self):
        if self.process is not None:
            if not self.alive:raise RuntimeError(self.error or 'Runtime stopped; restart the application')
            return
        from .worker import worker_main
        parent,child=self.ctx.Pipe()
        self.connection=parent
        self.process=self.ctx.Process(target=self.worker_target or worker_main,args=(child,self.inputs,self.snapshots,self.live,self.analyzed),
                                      name='follow-runtime',daemon=True)
        self.process.start();child.close()
        threading.Thread(target=self._send,name='runtime-requests',daemon=True).start()
        threading.Thread(target=self._receive,name='runtime-replies',daemon=True).start()

    def _send(self):
        try:
            while True:
                message=self.outgoing.get()
                if message is None:return
                self.connection.send(message)
        except (OSError,EOFError) as exc:self.error='Runtime communication lost: '+str(exc)

    def _receive(self):
        try:
            while True:
                response=self.connection.recv()
                with self.pending_lock:pending=self.pending.get(response['id'])
                if pending:
                    pending[1].update(response);pending[0].set()
        except (EOFError,OSError):
            if not self.closed:self.error='Runtime stopped; flight state is unconfirmed. Reconnect explicitly.'
            with self.pending_lock:
                for event,result in self.pending.values():
                    result.update(ok=False,error=self.error or 'Runtime closed');event.set()

    def call(self,op,*args,identity=0,timeout=5.,**kwargs):
        self.ensure()
        number=next(self.sequence);event=threading.Event();response={}
        with self.pending_lock:self.pending[number]=(event,response)
        try:
            request=dict(id=number,op=op,args=args,kwargs=kwargs,identity=identity,
                         time=time.monotonic(),fence=self.inputs.fence.value)
            try:self.outgoing.put_nowait(request)
            except queue.Full:
                self.inputs.release();raise RuntimeError('Runtime request queue full; control released')
            if not event.wait(timeout):
                self.inputs.release();raise RuntimeError('Runtime request timed out; control released')
            if not response.get('ok'):raise RuntimeError(response.get('error','Runtime error'))
            if self.manual and response.get('manual'):self.manual.apply(response['manual'])
            return response.get('result')
        finally:
            with self.pending_lock:self.pending.pop(number,None)

    def poll(self):
        if not self.process:return
        value,version=self.snapshots.get(self.last_snapshot)
        if value and value['time']<self.minimum_snapshot_time:
            self.last_snapshot=version;value=None
        if value:
            self.last_snapshot=version
            if self.session and value['session_id']==self.session.identity and value.get('session'):
                self.session.apply(value['session'],value['real_time'])
                self.session.preview_error=value.get('preview_error')
            if self.manual and value.get('manual'):self.manual.apply(value['manual'])
        if not self.alive:
            if self.manual:
                self.manual.alive=False;self.manual.enabled=False;self.manual.wanted=False
                self.manual.status=self.error or 'Runtime process stopped; control state unconfirmed'
            if self.session:self.session.model_error=self.error or 'Runtime process stopped'

    def metrics(self,data):
        if not self.alive:return
        # Diagnostic requests are expendable and have no callback/GUI wait.
        request=dict(id=next(self.sequence),op='ui_metrics',args=(data,),kwargs={},identity=0,
                     time=time.monotonic(),fence=self.inputs.fence.value)
        try:self.outgoing.put_nowait(request)
        except queue.Full:pass

    def close(self):
        if self.closed:return
        self.inputs.release()
        if self.alive:
            try:self.call('shutdown',timeout=2)
            except RuntimeError:pass
        self.closed=True
        if self.process:
            self.process.join(timeout=6)
            self.connection.close()
        try:self.outgoing.put_nowait(None)
        except queue.Full:pass


class RemoteSession:
    remote=True
    def __init__(self,client,*args,**kwargs):
        self.client=client;self.control_provider=None;self.stopping=False
        value=client.call('session_create',*args,**kwargs)
        self.identity=value['session_id'];self.settings=args[3];self.real_time=kwargs.get('real_time',False)
        self.result=Value();self.decision=Value();self.frames=Value();self.analysis=Value()
        self.reset_event=Flag();self.recorder=None
        self.receiver=SimpleNamespace(frames=self.frames,host=args[1])
        self.heading=SimpleNamespace(status='',get=lambda:None)
        self.log=SimpleNamespace(error=None)
        self.receiver.timing_snapshot=lambda now=None:self.timing
        self.last_live_token=self.last_analysis_token=None
        self.preview_error=None
        client.session=self;self.apply(value['session'],self.real_time)

    def apply(self,data,real_time):
        self.real_time=real_time;self.output=Path(data['output']);self.model_path=Path(data['model_path'])
        self.settings=data['settings'];self.reset_event.value=data['reset']
        for name in ('model_info','model_error','inference_count','inference_skips','generation'):
            setattr(self,name,data[name])
        self.log.error=data['log_error'];self.heading.status=data['heading_status']
        heading=data['heading'];self.heading.get=lambda:heading
        for key,value in data['receiver'].items():setattr(self.receiver,key,value)
        self.timing=data['receiver']['timing'];self.decision.value=data['decision'];self.result.value=data['result']
        rec=data['recorder']
        self.recorder=None if not rec else SimpleNamespace(status=lambda:rec['status'],thread=SimpleNamespace(is_alive=lambda:rec['alive']))
        if real_time:
            self.frames.value=None;self.analysis.value=None
            self.last_live_token=self.last_analysis_token=None
            return
        meta=data.get('frame')
        if meta and meta['token']!=self.last_live_token:
            frame=self.client.live.get(meta)
            if frame is not None:self.frames.value=frame;self.last_live_token=meta['token']
        analysis=data.get('analysis')
        if analysis and analysis['frame']['token']!=self.last_analysis_token:
            frame=self.client.analyzed.get(analysis['frame'])
            if frame is not None:
                self.analysis.value=dict(analysis,result=dict(analysis['result'],frame=frame))
                self.last_analysis_token=analysis['frame']['token']
        if not data.get('decision'):
            self.analysis.value=None

    def control_decision(self):return self.decision.get() # display only
    def with_control_status(self,decision):
        return dict(decision,control_status=self.client.manual.control_snapshot()) if decision and self.client.manual else decision
    def presentation_analysis(self):return None if self.real_time else self.analysis.get()
    def get_settings(self):return self.settings
    def invoke(self,method,*args,**kwargs):
        return self.client.call('session.'+method,*args,identity=self.identity,**kwargs)
    def start(self):return self.invoke('start')
    def stop(self):
        self.stopping=True
        result=self.invoke('stop',timeout=15.)
        if self.client.session is self:self.client.session=None
        return result
    def configure(self,settings):
        self.invoke('configure',settings);self.settings=settings
    def configure_search(self,*args):return self.invoke('configure_search',*args)
    def reset(self):return self.invoke('reset')
    def restart_control(self):return self.invoke('restart_control')
    def record_control(self,*args):
        if self.stopping:return False
        return self.invoke('record_control',*args)
    def start_recording(self,mode):return self.invoke('start_recording',mode)
    def stop_recording(self):return self.invoke('stop_recording')
    def set_real_time(self,enabled):
        self.client.call('real_time',enabled);self.real_time=enabled
        if enabled:
            self.frames.value=None;self.analysis.value=None
            self.last_live_token=self.last_analysis_token=None


class RemoteManual:
    def __init__(self,client,host,*,on_event=None,decision_provider=None,require_target_policy=False,send_hz=20):
        self.client=client;self.host=host;self.axis_limits=dict(DEFAULT_AXIS_LIMITS)
        self.alive=False;self.stop_event=Flag();self.thread=Alive(self);self.updated=0.
        self.identity=0;self.snapshot={};self.last_state_time=-1.;self.search_test=None
        value=client.call('manual_create',host,require_target_policy=require_target_policy,send_hz=send_hz)
        self.identity=value['manual']['identity'];client.manual=self;self.apply(value['manual'])

    def apply(self,state):
        if state['identity']!=self.identity or state['time']<self.last_state_time:return
        self.last_state_time=state['time'];self.snapshot=state
        for key in ('host','mode','wanted','enabled','connected','communication_paused','pause_reason',
                    'motion_hold','dance_started','dance_status','axis_limits','status','last_acknowledged','alive'):
            setattr(self,key,state[key])
        self.stop_event.value=state['stopped']
        self.send_hz=state['send_hz']
        self.search_test=SimpleNamespace(run_id=state['search_run_id']) if state.get('search_run_id') else None

    def invoke(self,method,*args,**kwargs):return self.client.call('manual.'+method,*args,identity=self.identity,**kwargs)
    def update(self,keys,decision=None):
        # NEVER send the GUI's cached decision to the flight runtime.
        self.updated=time.monotonic();self.client.inputs.pulse(keys,self.identity)
    def control_snapshot(self):return dict(self.snapshot)
    @property
    def detailed_logging(self):return getattr(self,'_detailed_logging',True)
    @detailed_logging.setter
    def detailed_logging(self,value):
        self.invoke('detailed_logging',value);self._detailed_logging=value
    def start(self):return self.invoke('start')
    def stop(self):
        self.client.inputs.release();self.stop_event.set()
        if self.client.alive:return self.invoke('stop')
    def release(self):
        self.client.inputs.release();self.wanted=False
        # The fence is the immediate path, even if dispatcher is busy.
        if self.client.alive:return self.invoke('release')
    def enable(self):return self.invoke('enable')
    def start_dance(self):return self.invoke('start_dance')
    def manual_mode(self):return self.invoke('manual_mode')
    def pause_control(self,*args,**kwargs):return self.invoke('pause_control',*args,**kwargs)
    def request(self,action):return self.invoke('request',action)
    def set_axis_limits(self,limits):return self.invoke('set_axis_limits',limits)
    def set_send_hz(self,value):
        self.send_hz=self.invoke('set_send_hz',value)
        return self.send_hz
    def start_search_test(self,settings,heading_provider,scenario,*,boost=False):
        return self.invoke('start_search_test',scenario,boost=boost)
    def stop_search_test(self):return self.invoke('stop_search_test')
