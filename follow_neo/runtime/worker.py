"""The sole owner of original TCP connections and existing tracking objects.

The GUI receives snapshots only. In particular, its display decision is never
used by ManualControl. Explicit UI pulses retain their ORIGINAL source time.
"""
import os
from pathlib import Path
import threading
import time

from ..diagnostics.timing import TimingWindow
from ..diagnostics import activity


def manual_state(manual,identity):
    if manual is None:return None
    state=manual.control_snapshot()
    state.update(identity=identity,alive=manual.thread.is_alive(),stopped=manual.stop_event.is_set())
    for key in ('host','mode','wanted','enabled','connected','communication_paused','pause_reason',
                'motion_hold','dance_started','dance_status','axis_limits','status','last_acknowledged'):
        state[key]=getattr(manual,key)
    if manual.search_test:state['search_run_id']=manual.search_test.run_id
    return state


class Runtime:
    def __init__(self,inputs,snapshots,live,analyzed):
        self.inputs=inputs;self.snapshots=snapshots;self.live=live;self.analyzed=analyzed
        self.session=None;self.manual=None;self.identity=0;self.session_id=0
        self.stop=threading.Event();self.timings=TimingWindow();self.ui_metrics={}
        self.real_time=False;self.preview_error=None;self.last_pulse=0.
        self.last_fence=inputs.fence.value;self.closed=False
        self.input_version=-1
        self.input_lock=threading.Lock()
        self.input_origin='gui';self.hard=False;self.hard_ready=False;self.console_pulse=None
        self.presentation_gate=threading.Event();self.presentation_gate.set()
        self.presentation_lock=threading.Lock()
        self.last_live_key=None;self.live_meta=None;self.last_analysis_key=None;self.analysis_meta=None

    def decision(self):
        s=self.session;m=self.manual
        if (s and m and m.host==s.receiver.host and not s.reset_event.is_set()
                and not s.model_error and s.receiver.state=='STREAMING'):
            return s.control_decision()
        return None

    def control(self):
        if self.manual and self.session and self.manual.host==self.session.receiver.host:
            return self.manual.control_snapshot()
        return dict(active=False,mode='MANUAL',reason='Control disconnected')

    def log_control(self,event,data):
        if self.session:self.session.record_control(event,data)

    def input_loop(self):
        while not self.stop.wait(.005):
            self.consume_input()
            if self.hard_ready and (not self.manual.thread.is_alive() or not self.manual.connected or self.manual.enforce_hard_wait()
                    or not self.manual.wanted or self.session.model_error
                    or self.session.receiver.state!='STREAMING' or self.terminal_dance()):
                self.inputs.hard_abort.set();self.manual.release()

    def terminal_dance(self):
        # The GUI normally handles one-shot completion. It is asleep in HARD.
        decision=self.session.decision.get()
        return bool(decision and not decision.get('continuous_dance',True)
            and decision.get('spacing_phase') in ('STOPPED','SEQUENCE_DONE')
            and isinstance(decision.get('prediction_time'),(int,float))
            and decision['prediction_time']>self.manual.dance_started)

    def consume_input(self):
        with self.input_lock:
            fence=self.inputs.fence.value
            if fence!=self.last_fence:
                self.last_fence=fence
                if self.manual:self.manual.release()
            message,self.input_version=self.inputs.latest.get(self.input_version)
            if not message:return
            now=time.monotonic();age=now-message['time']
            if not self.hard:
                self.timings.add('input_ipc_ms',age*1000)
                if self.last_pulse:self.timings.add('ui_pulse_gap_ms',(message['time']-self.last_pulse)*1000)
                self.last_pulse=message['time']
            m=self.manual
            if (m and message['identity']==self.identity and message['fence']==self.inputs.fence.value
                    and message.get('origin','gui')==self.input_origin and 0<=age<=.35):
                m.update(message['keys'],source_time=message['time'])
                if message.get('origin')=='console':self.console_pulse=message['time']

    def metadata(self,images=False):
        now=time.monotonic();s=self.session
        out=dict(pid=os.getpid(),time=now,session_id=self.session_id,
                 manual=manual_state(self.manual,self.identity),session=None,
                 real_time=self.real_time,preview_error=self.preview_error)
        if not s:return out
        recorder=s.recorder
        result=s.result.get();analysis=s.presentation_analysis() if images else None
        rs=None if not result else {k:v for k,v in result.items() if k!='frame'}
        data=dict(output=str(s.output),host=s.receiver.host,model_path=str(s.model_path),
                  settings=s.get_settings(),generation=s.generation,reset=s.reset_event.is_set(),
                  model_info=s.model_info,model_error=s.model_error,inference_count=s.inference_count,
                  inference_skips=s.inference_skips,decision=s.decision.get(),result=rs,
                  log_error=s.log.error,heading_status=s.heading.status,heading=s.heading.get(),
                  receiver=dict(state=s.receiver.state,error=s.receiver.error,count=s.receiver.count,
                    processed_count=s.receiver.processed_count,timing=s.receiver.timing_snapshot(now)),
                  recorder=None if not recorder else dict(status=recorder.status(),alive=recorder.thread.is_alive()))
        if images and not s.real_time:
            frame=s.receiver.frames.get()
            if frame:
                key=(self.session_id,frame.stream_epoch,frame.frame_id)
                if key!=self.last_live_key:
                    meta=self.live.put(frame)
                    if meta:self.live_meta=meta;self.last_live_key=key
                data['frame']=self.live_meta if key==self.last_live_key else None
                if data['frame'] is None and frame.image.nbytes>len(self.live.data):
                    self.preview_error='Preview exceeds 1920x1080 capacity; tracking keeps original resolution'
            if analysis:
                src=analysis['result']['frame'];key=(self.session_id,src.stream_epoch,src.frame_id)
                if key!=self.last_analysis_key:
                    meta=self.analyzed.put(src)
                    if meta:self.analysis_meta=meta;self.last_analysis_key=key
                meta=self.analysis_meta if key==self.last_analysis_key else None
                if meta:
                    data['analysis']=dict(analysis,result={k:v for k,v in analysis['result'].items() if k!='frame'},frame=meta)
        out['session']=data
        return out

    def publish_loop(self):
        last_log=0.
        while not self.stop.is_set():
            self.presentation_gate.wait()
            if self.stop.is_set():break
            started=time.monotonic()
            try:
                # This is a diagnostic worker, never capture/inference/control.
                with self.presentation_lock:
                    if self.hard:continue
                    value=self.metadata(images=not self.real_time)
                    sent=self.snapshots.put(value)
                self.timings.add('snapshot_publish_ms',(time.monotonic()-started)*1000)
                if not sent:self.timings.add('snapshot_skipped',1.)
                if activity.enabled and self.session and started-last_log>=1.:
                    last_log=started
                    self.session.log.write('architecture_health',dict(mode='process',runtime_pid=os.getpid(),
                        gui=self.ui_metrics,timing=self.timings.snapshot(),real_time=self.real_time,
                        preview_error=self.preview_error))
            except Exception as exc:
                self.preview_error=str(exc)
                if activity.enabled and self.session and started-last_log>=1.:
                    last_log=started;self.session.log.write('presentation_error',dict(error=str(exc)))
            self.stop.wait(max(.001,(.25 if self.real_time else 1/30)-(time.monotonic()-started)))

    def dispatch(self,op,args,kwargs,identity,stamp,fence):
        from follow_neo.runtime.session import FollowSession
        from follow_neo.transport.manual import ManualControl
        if op=='ping':return dict(pid=os.getpid(),parent_pid=os.getppid())
        if op=='hard_enter':
            if self.hard:raise RuntimeError('HARD already active')
            if not self.session or identity!=self.session_id or not self.manual or args[0]!=self.identity:
                raise RuntimeError('Connect video and keyboard first')
            with self.input_lock:
                self.input_origin='console';self.input_version=-1;self.console_pulse=None
            self.consume_input()
            try:
                m=self.manual;s=self.session
                if not s.target_policy.get('dance_allowed'):
                    raise RuntimeError('HARD requires the verified dual-class NEO model')
                if m.host!=s.receiver.host or s.receiver.state!='STREAMING':
                    raise RuntimeError('HARD requires live video from the control phone')
                if not m.connected or not m.control_snapshot()['active'] or m.motion_hold:
                    raise RuntimeError('Enable TCP control before HARD')
                if (fence!=self.inputs.fence.value or self.console_pulse is None
                        or not 0<=time.monotonic()-stamp<=.35):
                    raise RuntimeError('Expired or cancelled HARD request; enable control and try again')
                self.inputs.hard_abort.clear();m.manual_mode();m.detailed_logging=False
                self.hard=True;self.presentation_gate.clear()
                with self.presentation_lock:pass
                s.set_hard(True)
                self.consume_input()
                if (fence!=self.inputs.fence.value or not m.control_snapshot()['active']
                        or s.receiver.state!='STREAMING' or self.inputs.hard_abort.is_set()):
                    raise RuntimeError('HARD entry cancelled or control lost')
                m.set_hard_wait(True)
                m.start_dance();s.restart_control();self.hard_ready=True
                return None
            except Exception:
                self.exit_hard();raise
        if op=='hard_exit':
            if self.hard and identity!=self.session_id:raise RuntimeError('Video session replaced')
            return self.exit_hard()
        if op=='ui_metrics':self.ui_metrics=args[0];return None
        if op=='session_create':
            if self.session:raise RuntimeError('Disconnect current video session first')
            self.session=FollowSession(*args,architecture_mode='process',**kwargs);self.session_id+=1
            self.session.control_provider=self.control
            self.real_time=self.session.real_time
            self.session.log.write('architecture_start',dict(mode='process',runtime_pid=os.getpid(),
                gui_pid=os.getppid(),preview_hz=30,heartbeat_limit_ms=350))
            return self.metadata()
        if op=='manual_create':
            if self.manual and self.manual.thread.is_alive():raise RuntimeError('Wait for previous control to stop')
            self.identity+=1
            self.manual=ManualControl(*args,on_event=self.log_control,decision_provider=self.decision,**kwargs)
            return self.metadata()
        if op=='real_time':
            self.real_time=bool(args[0])
            if self.session:self.session.set_real_time(self.real_time)
            if self.manual:self.manual.detailed_logging=not self.real_time
            return None
        if op=='shutdown':self.stop.set();return None
        if op.startswith('session.'):
            # Display events can already be queued when stop retires a session.
            # Discard them; never attribute them to a replacement session.
            if op=='session.record_control' and (not self.session or identity!=self.session_id):return False
            if op=='session.stop' and not self.session and identity==self.session_id:return None
            if not self.session or identity!=self.session_id:raise RuntimeError('Video session replaced')
            method=op.split('.')[1]
            if method not in ('start','stop','configure','configure_search','reset','restart_control',
                              'start_recording','stop_recording','record_control'):raise ValueError(op)
            if method=='stop':
                if self.manual:
                    self.manual.stop()
                    if self.manual.thread.ident:self.manual.thread.join(timeout=4)
                    if self.manual.thread.is_alive():raise RuntimeError('Control shutdown still pending; retry Disconnect')
                self.session.stop();self.session=None
                return None
            return getattr(self.session,method)(*args,**kwargs)
        if op.startswith('manual.'):
            m=self.manual
            if m is None or identity!=self.identity:raise RuntimeError('Control connection replaced')
            method=op.split('.')[1]
            allowed=('start','stop','release','enable','start_dance','manual_mode','pause_control',
                     'request','set_axis_limits','set_send_hz','start_search_test','stop_search_test','detailed_logging')
            if method not in allowed:raise ValueError(op)
            # Delayed actions cannot turn into new flight authority. Release is
            # latched independently of this dispatcher and is never age-gated.
            if method in ('enable','start_dance','request','start_search_test'):
                self.consume_input()
                with self.input_lock:
                    if fence!=self.inputs.fence.value or not 0<=time.monotonic()-stamp<=.35:
                        raise RuntimeError('Expired operator request; press again')
                    result=self._manual_call(method,args,kwargs)
                    # A release can arrive during the local operation. Never
                    # hold a cross-process lock while calling flight code.
                    if fence!=self.inputs.fence.value:self.manual.release()
                    return result
            return self._manual_call(method,args,kwargs)
        raise ValueError('Unknown runtime operation: '+op)

    def exit_hard(self):
        self.hard_ready=False
        confirmed=True
        stop_reason=self.manual.hard_wait_error if self.manual else None
        try:
            if self.manual:
                m=self.manual;m.release();m.manual_mode()
                confirmed=m.release_acknowledged.is_set()
                if m.thread.is_alive():
                    confirmed=m.release_acknowledged.wait(3.)
                    if not confirmed:
                        m.stop();m.thread.join(timeout=4)
                        confirmed=m.release_acknowledged.is_set()
                m.detailed_logging=True
        finally:
            try:
                if self.manual:self.manual.set_hard_wait(False)
                if self.session:
                    self.session.set_hard(False);self.session.set_real_time(False)
            finally:
                activity.set_enabled(True)
                self.timings=TimingWindow();self.ui_metrics={}
                self.real_time=False;self.input_origin='gui';self.last_pulse=0.
                self.hard=False;self.presentation_gate.set()
        return dict(release_acknowledged=confirmed,hard_stop_reason=stop_reason)

    def _manual_call(self,method,args,kwargs):
        if method=='detailed_logging':self.manual.detailed_logging=bool(args[0]);return None
        if method=='start_search_test':
            if not self.session or self.manual.host!=self.session.receiver.host:
                raise ValueError('Connect video and keyboard to the same phone')
            return self.manual.start_search_test(self.session.get_settings(),self.session.heading.get,*args,**kwargs)
        return getattr(self.manual,method)(*args,**kwargs)

    def close(self):
        self.closed=True;self.stop.set()
        self.presentation_gate.set()
        if self.manual:
            self.manual.stop()
            if self.manual.thread.ident:self.manual.thread.join(timeout=4)
        if self.session:self.session.stop()


def worker_main(connection,inputs,snapshots,live,analyzed):
    runtime=Runtime(inputs,snapshots,live,analyzed)
    threads=[threading.Thread(target=fn,daemon=True) for fn in (runtime.input_loop,runtime.publish_loop)]
    for thread in threads:thread.start()
    try:
        while not runtime.stop.is_set():
            if not connection.poll(.05):continue
            request=connection.recv()
            try:
                result=runtime.dispatch(request['op'],request['args'],request['kwargs'],
                    request['identity'],request['time'],request['fence'])
                response=dict(id=request['id'],ok=True,result=result,manual=manual_state(runtime.manual,runtime.identity))
            except Exception as exc:
                response=dict(id=request['id'],ok=False,error=str(exc))
                if runtime.session:runtime.session.log.write('runtime_request_error',dict(operation=request['op'],error=str(exc)))
            connection.send(response)
    except (EOFError,BrokenPipeError,OSError):
        pass
    finally:
        runtime.close()
        for thread in threads:thread.join(timeout=1)
        connection.close()
