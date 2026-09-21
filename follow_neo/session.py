"""Independent capture, inference and control-preview scheduling. No HTTP server."""
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json
import queue
import threading
import time
import uuid

from .types import Settings, Box, Detection
from .video import VideoReceiver, LatestValue
from .detector import OnnxDroneDetector, DetectorSettings
from .controller import FollowController
from .recording import VideoRecorder
from . import __version__


class SessionLog:
    def __init__(self,path):
        self.path=Path(path); self.queue=queue.Queue(maxsize=512)
        self.dropped=0; self.error=None
        self.handle=self.path.open('w',encoding='utf-8')
        self.thread=threading.Thread(target=self._run,name='follow-log',daemon=True); self.thread.start()

    def write(self,event,data):
        row={'event':event,'wall_time':time.time(),'monotonic':time.monotonic(),**data}
        try: self.queue.put_nowait(row)
        except queue.Full: self.dropped+=1

    def _run(self):
        try:
            while True:
                row=self.queue.get()
                if row is None: break
                self.handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                self.handle.flush()
        except Exception as exc: self.error=str(exc)
        finally: self.handle.close()

    def close(self):
        if self.thread.is_alive():
            try: self.queue.put(None,timeout=2)
            except queue.Full: return
            self.thread.join(timeout=3)


class FollowSession:
    def __init__(self,root,host,codec='h264',settings=None,record_raw=False,compute_mode='Auto',device_id=-1):
        self.root=Path(root); self.settings=(settings or Settings()).validate()
        self.compute_mode=compute_mode; self.device_id=device_id; self.recorder=None
        self.recorder_lock=threading.Lock(); self.generation=0
        self.settings_lock=threading.Lock(); self.stop_event=threading.Event()
        self.reset_event=threading.Event(); self.result=LatestValue(); self.decision=LatestValue()
        self.analysis=LatestValue(); self.inference_count=0; self.inference_skips=0
        self.model_info=None; self.model_error=None; self.started_at=time.monotonic()
        self.flight_commands_sent=0
        self.output=self.root/'logs'/f'{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}'
        self.output.mkdir(parents=True)
        self.log=SessionLog(self.output/'events.jsonl')
        self.log.write('session_start',{'version':__version__,'host':host,'port':9999,
             'codec':codec,'compute_requested':compute_mode,'device_id':device_id,'settings':asdict(self.settings),
             'flight_control':'explicit_manual_or_dance_enable_required',
             'timestamp_origin':'local_decoder_monotonic_not_camera_exposure'})
        raw_path=self.output/f'video.{codec}' if record_raw else None
        self.receiver=VideoReceiver(host,codec=codec,raw_path=raw_path,on_event=self.log.write,on_frame=self._record_frame)
        self.inference_thread=threading.Thread(target=self._infer,name='neo-inference',daemon=True)
        self.control_thread=threading.Thread(target=self._control,name='follow-preview',daemon=True)

    def start(self):
        self.receiver.start(); self.inference_thread.start(); self.control_thread.start()

    def record_control(self,event,data):
        if event=='flight_command' and data.get('server_acknowledged'):
            self.flight_commands_sent+=1
        self.log.write(event,data)

    def get_settings(self):
        with self.settings_lock: return self.settings

    def configure(self,settings):
        settings.validate()
        with self.settings_lock: self.settings=settings; self.generation+=1
        self.reset_event.set(); self.log.write('settings',{'settings':asdict(settings)})

    def reset(self):
        with self.settings_lock: self.generation+=1
        self.reset_event.set()

    def start_recording(self,mode):
        if self.receiver.state!='STREAMING': raise RuntimeError('Connect and wait for live video first.')
        with self.recorder_lock:
            if self.recorder and self.recorder.thread.is_alive(): raise RuntimeError('Wait for the previous recording to finish.')
            if self.recorder: self.log.write('recording_end',self.recorder.status())
            self.recorder=VideoRecorder(self.output,mode)
        self.log.write('recording_start',self.recorder.status())

    def stop_recording(self):
        with self.recorder_lock:
            if self.recorder: self.recorder.stop(); self.log.write('recording_stop_requested',self.recorder.status())

    def _record_frame(self,frame):
        with self.recorder_lock: recorder=self.recorder
        if recorder: recorder.submit(frame,self.decision.get())

    def _infer(self):
        try:
            detector=OnnxDroneDetector(self.root/'models'/'best.onnx',cpu_threads=2,
                        compute_mode=self.compute_mode,device_id=self.device_id,diagnostics_dir=self.output)
            self.model_info={'input':detector.input.shape,'output':detector.output.shape,
                             'providers':detector.session.get_providers(),
                             'class_names':detector.class_names,'compute':detector.compute_info}
            self.log.write('model_ready',self.model_info)
            last_id=0; next_due=0.
            while not self.stop_event.is_set():
                with self.settings_lock: settings=self.settings; generation=self.generation
                frame=self.receiver.frames.get(); now=time.monotonic()
                if frame is None or frame.frame_id==last_id or now<next_due:
                    self.stop_event.wait(.005); continue
                self.inference_skips+=max(0,frame.frame_id-last_id-1); last_id=frame.frame_id
                if now-frame.decoded_at>settings.stale_seconds:
                    self.stop_event.wait(.01); continue
                detector.settings=DetectorSettings(confidence=settings.confidence,iou_threshold=settings.nms_iou)
                result=detector.detect(frame.image)
                result.update(frame=frame,completed_at=time.monotonic(),generation=generation)
                with self.settings_lock:
                    if generation==self.generation: self.result.set(result)
                self.inference_count+=1
                next_due=now+1/settings.inference_fps
        except Exception as exc:
            self.model_error=str(exc); self.log.write('inference_error',{'error':str(exc)})

    def _control(self):
        controller=FollowController(); last_id=0; last_state=None; last_log=0.
        try:
            while not self.stop_event.is_set():
                now=time.monotonic(); settings=self.get_settings()
                if self.reset_event.is_set():
                    controller.reset(); self.reset_event.clear()
                    previous=self.result.get(); last_id=previous['frame'].frame_id if previous else 0
                    self.analysis.set(None); self.decision.set(None)
                    self.log.write('tracking_reset',{})
                frame=self.receiver.frames.get(); result=self.result.get(); fresh_result=False
                if frame is None:
                    self.stop_event.wait(.02); continue
                h,w=frame.image.shape[:2]
                if result is not None and result.get('generation')==self.generation and result['frame'].frame_id>last_id:
                    src=result['frame']; last_id=src.frame_id
                    detections=[Detection(Box(*d['box']),d['confidence']) for d in result['detections']]
                    controller.observe(detections,src.decoded_at,now,src.frame_id,
                                       (src.image.shape[1],src.image.shape[0]),settings)
                    fresh_result=True
                decision=controller.tick(now,w,h,frame.decoded_at if self.receiver.state=='STREAMING' else None,settings)
                if self.model_error:
                    decision.update(state='ERROR',reason='inference_failed',stale=True,track_box=None,
                                    intent={'yaw':0.,'vertical':0.,'roll':0.,'forward':0.})
                self.decision.set(decision)
                if fresh_result:
                    chosen=controller.tracker.accepted
                    self.analysis.set({'result':result,'selected_box':None if chosen is None else asdict(chosen.box),'decision':decision})
                key=(decision['state'],decision['spacing_phase'],decision['stale'])
                if fresh_result or key!=last_state or now-last_log>=.5:
                    payload={'frame_id':frame.frame_id,'decision':decision,
                             'decoded_frames':self.receiver.count,'inferences':self.inference_count,
                             'skipped_frames':self.inference_skips}
                    if result:
                        payload.update(result_frame_id=result['frame'].frame_id,
                          decoded_at=result['frame'].decoded_at,inference_ms=result['inference_ms'],
                          preprocess_ms=result['preprocess_ms'],postprocess_ms=result['postprocess_ms'],detections=result['detections'])
                    self.log.write('observation',payload); last_state=key; last_log=now
                self.stop_event.wait(1/60)
        except Exception as exc:
            self.model_error=str(exc); self.log.write('controller_error',{'error':str(exc)})
            self.decision.set({'state':'ERROR','reason':str(exc),'stale':True,'track_box':None,
                              'intent':{'yaw':0.,'vertical':0.,'roll':0.,'forward':0.},'command_sent':False})

    def stop(self):
        self.stop_event.set(); self.receiver.stop()
        for thread in (self.inference_thread,self.control_thread):
            if thread.ident: thread.join()
        with self.recorder_lock: recorder=self.recorder
        if recorder:
            recorder.wait(); self.log.write('recording_end',recorder.status())
        self.log.write('session_end',{'decoded_frames':self.receiver.count,'inferences':self.inference_count,
                                     'log_dropped':self.log.dropped,'flight_commands_sent':self.flight_commands_sent})
        self.log.close()
