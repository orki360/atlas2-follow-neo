"""Spawn-safe offline GUI fixture. Never connects a camera or a drone."""
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import numpy as np


class FixtureSession:
    def __init__(self,root,host,codec,settings,*args,real_time=False,**kwargs):
        from follow_neo.capture.video import Frame,LatestValue
        self.output=Path(root)/'logs'/'fixture';self.output.mkdir(parents=True,exist_ok=True)
        self.settings=settings;self.real_time=real_time;self.generation=0
        self.model_path=Path('neo_minipro4_yolo26s_seg_960x544_v6.onnx')
        self.model_info=None;self.model_error=None;self.inference_count=0;self.inference_skips=0
        self.reset_event=threading.Event();self.recorder=None;self.result=LatestValue();self.decision=LatestValue()
        self.frames=LatestValue();self.frames.set(Frame(1,time.monotonic(),np.full((24,32,3),73,np.uint8)))
        self.receiver=SimpleNamespace(host=host,state='STREAMING',error=None,count=1,processed_count=1,frames=self.frames,
            timing_snapshot=lambda now=None:dict(decode_gaps_over_250ms=0,max_decode_gap_ms=0))
        self.heading=SimpleNamespace(status='offline heading',get=lambda:None)
        self.log=SimpleNamespace(error=None,write=lambda *args:None)
    def start(self):pass
    def stop(self):pass
    def get_settings(self):return self.settings
    def control_decision(self):return None
    def presentation_analysis(self):return None
    def record_control(self,*args):pass
    def restart_control(self):pass
    def reset(self):self.generation+=1
    def configure(self,settings):self.settings=settings;self.generation+=1
    def configure_search(self,*args):pass
    def set_real_time(self,value):self.real_time=value


def fixture_worker(*args):
    from unittest.mock import patch
    from follow_neo.runtime.worker import worker_main
    with patch('follow_neo.runtime.session.FollowSession',FixtureSession):worker_main(*args)


class HardFixtureSession(FixtureSession):
    """Real log barrier and synthetic live decisions for spawned HARD tests."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from follow_neo.runtime.session import SessionLog
        from follow_neo.perception.models import target_policy,DUAL_MODEL_SOURCE_NAMES
        self.log=SessionLog(self.output/'events.jsonl');self.search_test_log=None
        self.hard=False;self.model_info={}
        self.target_policy=target_policy(self.model_path.name,DUAL_MODEL_SOURCE_NAMES[self.model_path.name])
        self.receiver.finish_raw_recording=lambda:None

    def set_hard(self,enabled):
        real_set_hard(self,enabled)

    def stop(self):self.log.close()

    def control_decision(self):
        now=time.monotonic()
        value=dict(prediction_time=now,measurement_time=now,measurement_age_ms=0.,
            measurement_id=1,video_time=now,state='TRACK',confirmed=True,accepted=True,stale=False,
            spacing_phase='APPROACH',continuous_dance=True,selected_class_id=0,selected_class_name='neo',
            target_policy=self.target_policy,intent=dict(yaw=.1,vertical=0.,roll=.05,forward=.1))
        self.decision.set(value)
        return value


# Preserve the real barrier before replacing FollowSession inside the worker.
from follow_neo.runtime.session import FollowSession as _RealSession
real_set_hard=_RealSession.set_hard


def hard_fixture_worker(*args):
    from unittest.mock import patch
    from follow_neo.runtime.worker import worker_main
    with patch('follow_neo.runtime.session.FollowSession',HardFixtureSession):worker_main(*args)
