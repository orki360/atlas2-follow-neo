"""Independent capture, inference and control-preview scheduling. No HTTP server."""
from dataclasses import asdict,replace
from datetime import datetime
from pathlib import Path
import json
import queue
import threading
import time
import uuid
import traceback

from .types import Settings, Box, Detection
from .video import VideoReceiver, LatestValue
from .detector import OnnxDroneDetector, DetectorSettings
from .controller import FollowController
from .recording import VideoRecorder
from . import __version__, __revision__
from .botsort_tracker import low_threshold
from .cadence import FrameRateGate
from .telemetry import HeadingReceiver,heading_snapshot
from .models import resolve_model,DEFAULT_MODEL,NEO_CLASS_ID,NEO_CLASS_NAME,target_policy
from .nano_tracker import NanoWorker
from .nano_guard import strong_anchor, apply_visual, overlap, continuity_detection


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
    def __init__(self,root,host,codec='h264',settings=None,record_raw=False,compute_mode='Auto',device_id=-1,
                 model_name=DEFAULT_MODEL,recording_profile='Auto'):
        self.root=Path(root); self.settings=(settings or Settings()).validate()
        self.model_path,self.model_manifest=resolve_model(self.root,model_name)
        self.target_policy=target_policy(self.model_path.name)
        self.recording_profile=recording_profile
        self.compute_mode=compute_mode; self.device_id=device_id; self.recorder=None
        self.recorder_lock=threading.Lock(); self.generation=0
        self.settings_lock=threading.Lock(); self.stop_event=threading.Event()
        self.reset_event=threading.Event(); self.result=LatestValue(); self.decision=LatestValue()
        self.control_reset_event=threading.Event()
        self.control_wake=threading.Event()
        self.analysis=LatestValue(); self.inference_count=0; self.inference_skips=0
        self.model_info=None; self.model_error=None; self.started_at=time.monotonic()
        self.flight_commands_sent=0
        self.control_provider=None;self.search_test_log=None
        self.output=self.root/'logs'/f'{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}'
        self.output.mkdir(parents=True)
        self.log=SessionLog(self.output/'events.jsonl')
        # Bound OpenCV's process-wide native pool: 12-way DNN + detector + UI
        # oversubscribed this laptop. The offline Nano timing used one thread.
        import cv2
        cv2.setNumThreads(1)
        self.nano=NanoWorker(self.root,self.settings.nano_mode,self.generation,
                             wake=self.control_wake.set,emit=self.log.write)
        self.log.write('session_start',{'version':__version__,'host':host,'port':9999,
             'codec':codec,'compute_requested':compute_mode,'device_id':device_id,'settings':asdict(self.settings),
             'model':self.model_manifest,'recording_profile':recording_profile,
             'tracking_backend':'BoT-SORT','tracking_reid':False,
             'tracking_motion':'botsort_with_separate_nano','nano_mode':self.settings.nano_mode,
             'opencv_threads':cv2.getNumThreads(),'tracking_revision':__revision__,'control_revision':__revision__,
             'flight_control':'explicit_manual_or_dance_enable_required',
             'timestamp_origin':'local_decoder_monotonic_not_camera_exposure'})
        raw_path=self.output/f'video.{codec}' if record_raw else None
        self.receiver=VideoReceiver(host,codec=codec,raw_path=raw_path,on_event=self._video_event,
                                    on_frame=self._record_frame,processing_fps=self.settings.video_fps)
        self.heading=HeadingReceiver(host,on_event=self.log.write)
        self.inference_thread=threading.Thread(target=self._infer,name='neo-inference',daemon=True)
        self.control_thread=threading.Thread(target=self._control,name='follow-preview',daemon=True)

    def start(self):
        self.nano.start();self.receiver.start(); self.heading.start(); self.inference_thread.start(); self.control_thread.start()

    def _video_event(self,event,data):
        if event=='video_health':
            data=dict(data,context=dict(inferences=self.inference_count,log_dropped=self.log.dropped,
                log_queue_depth=self.log.queue.qsize(),inference_error=self.model_error,
                control=self.control_provider() if self.control_provider else None,
                heading_status=getattr(getattr(self,'heading',None),'status',None)))
        if event=='video_error':
            # Cancel in-flight YOLO/Nano and discard the old target before any
            # new stream can provide control evidence. Manual input is separate.
            with self.settings_lock:
                self.generation+=1
                self.result.set(None);self.analysis.set(None);self.decision.set(None)
                self.reset_event.set()
                self.nano.reset(self.settings.nano_mode,self.generation)
            self.control_wake.set()
            self.log.write('video_tracking_invalidated',dict(generation=self.generation,
                stream_epoch=data['stream_epoch'],reason='video_stream_interrupted'))
        self.log.write(event,data)

    def record_control(self,event,data):
        if event=='flight_command' and data.get('server_acknowledged'):
            self.flight_commands_sent+=1
        self.log.write(event,data)
        if event in ('control_error','control_released','control_enabled','control_movement_paused','control_movement_resumed'):
            if event in ('control_movement_paused','control_movement_resumed') and data.get('preserve_tracking'):
                self.log.write('control_context_preserved',dict(trigger=event,
                    pause_started=data.get('pause_started'),resume_measurement_after=data.get('resume_measurement_after'),
                    generation=self.generation,commands_require_live_authority=True))
                self.control_wake.set()
            else:self.restart_control()
        if event=='search_test_started':
            if self.search_test_log:self.search_test_log.close()
            folder=self.output/('search_test_'+data['run_id']);folder.mkdir()
            self.search_test_log=SessionLog(folder/'events.jsonl')
        if getattr(self,'search_test_log',None) and (event.startswith('search_test_') or event.startswith('control_') or event=='flight_command'):
            self.search_test_log.write(event,data)
            if event=='search_test_summary':
                self.search_test_log.close();self.search_test_log=None

    def with_control_status(self,decision):
        if not decision:return decision
        control=self.control_provider() if self.control_provider else dict(active=False,mode='MANUAL',reason='Control disconnected')
        return dict(decision,control_status=control)

    def control_decision(self):
        decision=self.decision.get()
        if (not decision or self.receiver.state!='STREAMING'
                or decision.get('generation')!=self.generation):return None
        now=time.monotonic()
        visual=self.nano.snapshot(now,source_time=decision.get('measurement_time'),frame_id=decision.get('measurement_id'))
        return apply_visual(decision,visual,now)

    def presentation_analysis(self):
        analysis=self.analysis.get()
        if not analysis or analysis['result'].get('generation')!=self.generation:return None
        return dict(analysis,nano=self.nano.at_frame(analysis['result']['frame']))

    def get_settings(self):
        with self.settings_lock: return self.settings

    def configure(self,settings):
        settings.validate()
        with self.settings_lock: self.settings=settings; self.generation+=1
        self.receiver.processing_fps=settings.video_fps
        self.nano.reset(settings.nano_mode,self.generation)
        self.restart_control()
        self.log.write('settings',{'settings':asdict(settings),'tracking_reset':False,'control_reset':True})

    def restart_control(self):
        self.decision.set(None)
        self.control_reset_event.set()
        if hasattr(self,'nano'):self.nano.reset(self.settings.nano_mode,self.generation)
        if hasattr(self,'control_wake'):self.control_wake.set()
        self.log.write('control_restart_requested',{'tracking_reset':False})

    def reset(self):
        with self.settings_lock: self.generation+=1
        self.reset_event.set()
        self.nano.reset(self.settings.nano_mode,self.generation);self.control_wake.set()

    def configure_search(self,angle,seconds,enabled,auto_duration=None):
        # Search settings do not change the detector, tracker or distance state.
        with self.settings_lock:
            self.settings=replace(self.settings,search_yaw_degrees=float(angle),
                edge_search_seconds=float(seconds),edge_search_enabled=bool(enabled),
                search_auto_duration=self.settings.search_auto_duration if auto_duration is None else bool(auto_duration)).validate()
            saved=asdict(self.settings)
        self.log.write('search_settings',{'settings':saved,'tracking_reset':False,
            'active_arc_policy':'existing arc keeps its original budget; zero/disabled cancels'})

    def start_recording(self,mode):
        if self.receiver.state!='STREAMING': raise RuntimeError('Connect and wait for live video first.')
        with self.recorder_lock:
            if self.recorder and self.recorder.thread.is_alive(): raise RuntimeError('Wait for the previous recording to finish.')
            if self.recorder: self.log.write('recording_end',self.recorder.status())
            self.recorder=VideoRecorder(self.output,mode,profile=self.recording_profile,on_event=self.log.write)
        self.log.write('recording_start',self.recorder.status())

    def stop_recording(self):
        with self.recorder_lock:
            if self.recorder: self.recorder.stop(); self.log.write('recording_stop_requested',self.recorder.status())

    def _record_frame(self,frame):
        self.nano.submit_frame(frame)
        with self.recorder_lock: recorder=self.recorder
        if recorder: recorder.submit(frame,self.with_control_status(self.control_decision()))

    def _infer(self):
        try:
            detector=OnnxDroneDetector(self.model_path,cpu_threads=2,
                        compute_mode=self.compute_mode,device_id=self.device_id,diagnostics_dir=self.output)
            self.model_info={'input':detector.input.shape,'output':detector.output.shape,
                             'outputs':[o.shape for o in detector.outputs],'model':self.model_manifest,
                             'providers':detector.session.get_providers(),
                             'class_names':detector.class_names,'compute':detector.compute_info}
            source_names=getattr(detector,'model_class_names',None)
            self.target_policy=target_policy(self.model_path.name,source_names if isinstance(source_names,dict) else None)
            self.model_info.update(model_class_names=source_names if isinstance(source_names,dict) else None,
                                   target_policy=self.target_policy)
            self.log.write('model_ready',self.model_info)
            last_id=0;frame_version=0;sampler=FrameRateGate()
            while not self.stop_event.is_set():
                frame,new_version=self.receiver.frames.wait_next(frame_version,.10)
                if self.stop_event.is_set():break
                if new_version==frame_version or frame is None:continue
                self.inference_skips+=max(0,new_version-frame_version-1)
                frame_version=new_version
                with self.settings_lock: settings=self.settings; generation=self.generation
                if frame.stream_epoch!=self.receiver.stream_epoch:continue
                now=time.monotonic()
                if frame.frame_id==last_id:continue
                last_id=frame.frame_id
                if settings.inference_fps<settings.video_fps and not sampler.admit(frame.decoded_at,settings.inference_fps):continue
                if now-frame.decoded_at>settings.stale_seconds:
                    continue
                detector.settings=replace(detector.settings,confidence=low_threshold(settings.confidence),
                                          iou_threshold=settings.nms_iou)
                inference_started=time.monotonic()
                result=detector.detect(frame.image)
                result.update(frame=frame,completed_at=time.monotonic(),generation=generation,
                              inference_started_at=inference_started)
                with self.settings_lock:
                    if generation==self.generation:
                        self.result.set(result);self.control_wake.set()
                self.inference_count+=1
        except Exception as exc:
            self.model_error=str(exc); self.log.write('inference_error',{'error':str(exc)})

    def _control(self):
        last_id=0; last_state=None; last_log=0.;result_version=0
        from .search_sectors import SearchObservationLog
        observation_log=SearchObservationLog()
        try:
            controller=FollowController(tracker_backend='botsort',target_class_id=NEO_CLASS_ID,
                                        target_class_name=NEO_CLASS_NAME)
            while not self.stop_event.is_set():
                # Wake as soon as inference completes, or tick for aging even
                # without a result. No unconditional sleep after fresh work.
                # Both YOLO and Nano wake control immediately. Clear BEFORE
                # reading latest values so a new event cannot be lost.
                self.control_wake.wait(1/60);self.control_wake.clear()
                if self.stop_event.is_set():break
                now=time.monotonic()
                with self.settings_lock:
                    settings=self.settings;generation=self.generation
                    if self.reset_event.is_set():
                        controller.reset(); self.reset_event.clear()
                        self.control_reset_event.clear()
                        previous=self.result.get(); last_id=previous['frame'].frame_id if previous else 0
                        self.analysis.set(None); self.decision.set(None)
                        self.log.write('tracking_reset',{'reason':'reset_requested','generation':generation})
                    elif self.control_reset_event.is_set():
                        controller.restart_control();self.control_reset_event.clear()
                        self.log.write('control_restart',{'tracking_reset':False})
                frame=self.receiver.frames.get(); result=self.result.get(); fresh_result=False
                if frame is None or frame.stream_epoch!=self.receiver.stream_epoch:
                    self.stop_event.wait(.02); continue
                h,w=frame.image.shape[:2]
                authority=self.control_provider() if self.control_provider else {}
                dance_active=bool(authority.get('active') and authority.get('mode')=='DANCE'
                                  and not authority.get('motion_hold') and not authority.get('keyboard_override'))
                # Keep perception/search clocks alive during a short ACK hold.
                # ManualControl remains neutral until the channel is recovered.
                dance_context=dance_active or authority.get('continuity_pause',False)
                tracker_ms=None;tracker_started=None
                if result is not None and result.get('generation')==generation and result['frame'].frame_id>last_id:
                    src=result['frame']; last_id=src.frame_id
                    detections=[Detection(Box(*d['box']),d['confidence'],d.get('class_id'),
                                          d.get('class_name'),d.get('model_class_name'),
                                          tuple(d.get('class_scores',()))) for d in result['detections']]
                    tracker_started=time.monotonic()
                    proof=continuity_detection(result,self.nano.evidence_at(src),
                        controller.tracker.logical_target_id,settings) if dance_context else None
                    controller.observe(detections,src.decoded_at,tracker_started,src.frame_id,
                                       (src.image.shape[1],src.image.shape[0]),settings,image=src.image,
                                       continuity=proof,allow_search_cues=dance_context)
                    if proof:self.log.write('yolo_visual_continuity',dict(proof,
                        accepted=not controller.rejected,association_reason=controller.tracker.association_reason))
                    anchor=strong_anchor(controller,result,settings)
                    if anchor is not None:self.nano.submit_anchor(anchor)
                    else:
                        visual=self.nano.at_frame(src) or {}
                        # Clear MINI evidence may veto, never initialize Nano.
                        box=visual.get('box')
                        if box and any(d.get('class_id')==1 and d['confidence']>=settings.confidence
                                       and overlap(box,d['box'])>.5 for d in result['detections']):
                            self.nano.veto('competing_mini_detection')
                    tracker_ms=(time.monotonic()-tracker_started)*1000
                    fresh_result=True
                authority=self.control_provider() if self.control_provider else {}
                heading,now=heading_snapshot(self.heading)
                tick_started=time.monotonic()
                visual=self.nano.snapshot(now,source_time=controller.tracker.measurement_time,
                                          frame_id=controller.tracker.measurement_id)
                decision=controller.tick(now,w,h,frame.decoded_at if self.receiver.state=='STREAMING' else None,
                                         settings,heading,control_available=dance_context,visual=visual,generation=generation)
                tick_ms=(time.monotonic()-tick_started)*1000
                decision['control_status']=authority
                decision['target_policy']=self.target_policy
                decision['generation']=generation
                visual=self.nano.snapshot(source_time=decision.get('measurement_time'),frame_id=decision.get('measurement_id'))
                decision=apply_visual(decision,visual,time.monotonic())
                if self.model_error:
                    decision.update(state='ERROR',reason='inference_failed',stale=True,track_box=None,
                                    intent={'yaw':0.,'vertical':0.,'roll':0.,'forward':0.})
                if generation!=self.generation:continue
                decision['generation']=generation
                self.decision.set(decision)
                observation_log.update(decision.get('edge_search') or {},now,self.log.write)
                if fresh_result:
                    report=controller.tracker.association_report()
                    evaluated=controller.tracker.last_result_id==result['frame'].frame_id
                    if not evaluated:
                        report=dict(accepted=False,stages=[],association_reason='result_rejected_before_association')
                    self.log.write('botsort_association',dict(report,generation=generation,evaluated=evaluated,
                        result_frame_id=result['frame'].frame_id,result_source_time=result['frame'].decoded_at))
                    chosen=controller.tracker.accepted
                    identity_event=decision.get('tracking',{}).get('identity_event')
                    if identity_event:self.log.write('target_reassociated',identity_event)
                    numerical_event=decision.get('tracking',{}).get('numerical_event')
                    if numerical_event:self.log.write('tracking_numerical_reset',numerical_event)
                    self.analysis.set({'result':result,'selected_box':None if chosen is None or controller.rejected else asdict(chosen.box),
                                       'selected_class_id':decision['selected_class_id'],'decision':decision})
                key=(decision['state'],decision['spacing_phase'],decision['stale'],decision['edge_search']['active'])
                if fresh_result or key!=last_state or now-last_log>=.5:
                    payload={'frame_id':frame.frame_id,'decision':decision,
                             'video_timing':self.receiver.timing_snapshot(now),
                             'decoded_frames':self.receiver.count,'inferences':self.inference_count,
                             'processed_frames':self.receiver.processed_count,
                             'sampled_out_frames':self.receiver.sampled_out,
                             'skipped_frames':self.inference_skips}
                    if result:
                        payload.update(result_frame_id=result['frame'].frame_id,
                          decoded_at=result['frame'].decoded_at,inference_ms=result['inference_ms'],
                          preprocess_ms=result['preprocess_ms'],postprocess_ms=result['postprocess_ms'],
                          detections=[{k:v for k,v in d.items() if k!='segments'} for d in result['detections']],
                          result_ready_at=result['completed_at'],
                          model_input_size=result.get('input_size'),input_transform=result.get('input_transform'),
                          pipeline_timing=dict(
                            inference_cycle_ms=result.get('total_ms'),
                            source_to_inference_start_ms=max(0.,result.get('inference_started_at',result['frame'].decoded_at)-result['frame'].decoded_at)*1000,
                            result_to_tracker_start_ms=None if tracker_started is None else max(0.,tracker_started-result['completed_at'])*1000,
                            tracker_ms=tracker_ms,control_tick_ms=tick_ms,
                            source_to_decision_ms=max(0.,now-result['frame'].decoded_at)*1000),
                          result_to_control_ms=max(0.,now-result['completed_at'])*1000)
                    self.log.write('observation',payload); last_state=key; last_log=now
        except Exception as exc:
            self.model_error=str(exc); self.log.write('controller_error',{'error':str(exc),'traceback':traceback.format_exc()})
            self.decision.set({'state':'ERROR','reason':str(exc),'stale':True,'track_box':None,
                              'intent':{'yaw':0.,'vertical':0.,'roll':0.,'forward':0.},'command_sent':False})

    def stop(self):
        self.stop_event.set(); self.receiver.stop(); self.heading.stop()
        self.control_wake.set();self.nano.stop()
        for thread in (self.inference_thread,self.control_thread):
            if thread.ident: thread.join()
        with self.recorder_lock: recorder=self.recorder
        if recorder:
            recorder.wait(); self.log.write('recording_end',recorder.status())
        self.log.write('session_end',{'decoded_frames':self.receiver.count,'inferences':self.inference_count,
                                     'processed_frames':self.receiver.processed_count,'sampled_out_frames':self.receiver.sampled_out,
                                     'log_dropped':self.log.dropped,'flight_commands_sent':self.flight_commands_sent})
        self.log.close()
        if self.search_test_log:self.search_test_log.close();self.search_test_log=None
