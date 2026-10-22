"""V13 class isolation contracts. Synthetic frames and socketpair only."""
import copy
import json
import tempfile
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

import numpy as np
from follow_neo.control.controller import FollowController
from follow_neo.control.dance import dance_command,ZERO
from follow_neo.perception.detector import OnnxDroneDetector,DetectorSettings
from follow_neo.ui.window import FollowLabWindow
from follow_neo.transport.manual import ManualControl
from follow_neo.perception.models import (DEFAULT_MODEL,PREVIOUS_MODEL,LEGACY_MODEL,
    DUAL_CLASS_NAMES,DUAL_SOURCE_NAMES,model_from_config,resolve_model,target_policy)
from follow_neo.presentation.overlay import render_bundle,NEO_COLOR,MINI_COLOR
from follow_neo.runtime.session import FollowSession
from follow_neo.shared.types import Box,Detection,Settings,settings_from_config
from follow_neo.capture.video import Frame
from follow_neo.perception.prediction import COAST_YAW,COAST_VERTICAL
from test_manual import Server,wait_for

ROOT=Path(__file__).resolve().parents[1]


def detection(class_id=0,confidence=.9,x=240):
    return Detection(Box(x,180,x+60,220),confidence,class_id,
                     DUAL_CLASS_NAMES[class_id],DUAL_SOURCE_NAMES[class_id])


def allowed_decision(now=10.):
    return dict(prediction_time=now,measurement_time=now-.02,measurement_age_ms=20,
        measurement_id=4,state='TRACK',spacing_phase='APPROACH',accepted=True,
        confirmed=True,stale=False,selected_class_id=0,selected_class_name='neo',
        target_policy=target_policy(DEFAULT_MODEL,DUAL_SOURCE_NAMES),
        intent=dict(yaw=.3,vertical=0.,roll=0.,forward=.2),
        intent_basis=dict(yaw=1.,vertical=1.,roll=1.,forward=1.))


class DecodeTests(unittest.TestCase):
    def setUp(self):
        self.d=OnnxDroneDetector.__new__(OnnxDroneDetector)
        self.d.settings=DetectorSettings();self.d.class_names=dict(DUAL_CLASS_NAMES)
        self.d.model_class_names=dict(DUAL_SOURCE_NAMES)
        self.transform=dict(scale=1.,pad_x=0.,pad_y=0.,source_w=640.,source_h=480.)
    def rows(self,*scores):
        raw=np.zeros((1,38,len(scores)),np.float32)
        for i,(neo,mini) in enumerate(scores):raw[0,:6,i]=[320,220,80,40,neo,mini]
        return raw
    def test_overlapping_classes_survive_but_same_class_duplicate_does_not(self):
        ds=self.d.postprocess(self.rows((.9,.1),(.1,.95),(.85,.2)),self.transform)
        self.assertEqual([d['class_id'] for d in ds],[1,0])
        self.assertEqual([d['class_name'] for d in ds],['MINI PRO4','neo'])
        self.assertEqual(ds[1]['model_class_name'],DUAL_SOURCE_NAMES[0])
        self.assertEqual(ds[0]['box'],ds[1]['box']);json.dumps(ds,allow_nan=False)
    def test_mini_only_and_transposed_export(self):
        raw=self.rows((.04,.91))
        for output in (raw,raw.transpose(0,2,1)):
            ds=self.d.postprocess(output,self.transform)
            self.assertEqual(len(ds),1);self.assertEqual(ds[0]['class_id'],1)
    def test_mask_coefficients_start_after_both_scores_and_follow_nms(self):
        raw=self.rows((.9,.1),(.1,.95),(.8,.1))
        raw[0,6:,0]=-10;raw[0,6:,1]=10;raw[0,6:,2]=10
        ds=self.d.postprocess(raw,self.transform,np.ones((1,32,160,160),np.float32))
        by_class={d['class_id']:d for d in ds}
        self.assertFalse(by_class[0]['segments']);self.assertTrue(by_class[1]['segments'])
        self.assertAlmostEqual(by_class[0]['confidence'],.9,places=5)
        low=self.rows((.01,.02));low[0,6:]=1000
        self.assertEqual(self.d.postprocess(low,self.transform),[])
    def test_many_mini_boxes_do_not_remove_neo_at_detection_cap(self):
        self.d.settings=DetectorSettings(max_detections=2)
        raw=self.rows((.8,.1),(.1,.99),(.1,.98),(.1,.97));raw[0,0,:]=[60,200,340,480]
        ds=self.d.postprocess(raw,self.transform)
        self.assertEqual(sum(d['class_id']==0 for d in ds),1)
        self.assertEqual(sum(d['class_id']==1 for d in ds),2)
    def test_nonfinite_or_invalid_class_scores_are_discarded(self):
        for score in (float('nan'),float('inf'),-1.,1.1):
            self.assertEqual(self.d.postprocess(self.rows((score,.9)),self.transform),[])
    def test_bad_class_mapping_fails_before_use(self):
        session=Mock();session.get_inputs.return_value=[SimpleNamespace(shape=[1,3,640,640])]
        session.get_outputs.return_value=[SimpleNamespace(shape=[1,38,8400]),SimpleNamespace(shape=[1,32,160,160])]
        for names in ({0:'minip4',1:DUAL_SOURCE_NAMES[0]}, {0:'neo'}, {1:'minip4'}):
            session.get_modelmeta.return_value=SimpleNamespace(custom_metadata_map={'names':repr(names)})
            with patch('follow_neo.perception.compute.create_session',return_value=(session,{})):
                with self.assertRaisesRegex(RuntimeError,'class 0'):
                    OnnxDroneDetector(ROOT/'models'/DEFAULT_MODEL,compute_mode='CPU')
    def test_previous_single_class_model_still_loads_for_display(self):
        d=OnnxDroneDetector(resolve_model(ROOT,PREVIOUS_MODEL)[0],compute_mode='CPU')
        self.assertTrue(d.segmentation);self.assertFalse(d.target_policy['dance_allowed'])
        json.dumps(d.detect(np.zeros((480,640,3),np.uint8)),allow_nan=False)


class TrackingIsolationTests(unittest.TestCase):
    def setUp(self):
        self.c=FollowController(tracker_backend='botsort',target_class_id=0,target_class_name='neo')
        self.s=Settings(confidence=.8,new_track_confidence=.8);self.t=10.;self.i=0
    def feed(self,ds):
        self.t+=.04;self.i+=1
        self.c.observe(ds,self.t,self.t,self.i,(640,480),self.s)
        return self.c.tick(self.t,640,480,self.t,self.s,
                           dict(yaw_deg=0.,time=self.t,source='offline_test'))
    def acquire(self):
        for _ in range(8):d=self.feed([detection(),detection(1,.99)])
        self.assertTrue(d['accepted']);self.assertTrue(d['confirmed']);return d
    def test_mini_only_never_acquires_and_remains_visible(self):
        for _ in range(20):d=self.feed([detection(1,.99)])
        self.assertIsNone(d['measurement_time']);self.assertFalse(d['confirmed'])
        self.assertEqual(len(d['raw_detections']),1)
        self.assertEqual(d['detection_summary']['target_eligible'],0)
        self.assertEqual(d['detection_summary']['class_counts'],{'MINI PRO4':1})
    def test_both_classes_use_neo_even_when_mini_has_higher_score_and_same_box(self):
        d=self.acquire()
        self.assertEqual(d['selected_class_id'],0);self.assertEqual(d['selected_class_name'],'neo')
        self.assertAlmostEqual(d['tracking_confidence'],.9)
        self.assertEqual(d['detection_summary']['target_eligible'],1)
        self.assertEqual(len(d['raw_detections']),2)
    def test_mini_cannot_continue_refresh_or_reacquire_a_lost_neo(self):
        initial=self.acquire();stamp=initial['measurement_time'];identity=self.c.tracker.logical_target_id
        for _ in range(40):
            d=self.feed([detection(1,.99)])
            self.assertEqual(d['measurement_time'],stamp);self.assertFalse(d['accepted'])
            self.assertEqual(d['detection_summary']['target_eligible'],0)
            self.assertFalse(d['candidate_pending'])
        self.assertEqual(self.c.tracker.logical_target_id,identity)
        self.assertTrue(self.c.missing_detection);self.assertGreater(d['measurement_age_ms'],1500)
        for _ in range(8):d=self.feed([detection(),detection(1,.99)])
        self.assertTrue(d['accepted']);self.assertGreater(d['measurement_time'],stamp)
    def test_unknown_or_wrongly_named_class_zero_cannot_acquire(self):
        wrong=Detection(Box(240,180,300,220),.99,0,'drone')
        unknown=Detection(Box(240,180,300,220),.99)
        for _ in range(10):d=self.feed([wrong,unknown])
        self.assertIsNone(d['measurement_time']);self.assertFalse(d['confirmed'])

    def command(self,d,started=9.):
        return dance_command(dict(d,target_policy=target_policy(DEFAULT_MODEL,DUAL_SOURCE_NAMES)),
                             self.t,started,require_target_policy=True)

    def test_actual_controller_retains_neo_identity_and_authorizes_bounded_coast(self):
        for _ in range(25):before=self.feed([detection()])
        self.assertEqual(before['state'],'TRACK');identity=before['target_identity']
        for ds in ([],[detection(1,.99)],[]):
            d=self.feed(ds);values,reason=self.command(d)
            self.assertEqual(d['state'],'COAST');self.assertTrue(d['prediction_recovery'])
            self.assertEqual(d['target_identity'],identity);self.assertEqual(d['selected_class_name'],'neo')
            self.assertEqual(d['target_class_source'],'retained_measurement')
            self.assertIsNone(d['selected_measurement_box']);self.assertFalse(d['accepted'])
            self.assertTrue(any(values));self.assertIn('Kalman recovery',reason)
            self.assertLessEqual(abs(values[0]),COAST_YAW);self.assertLessEqual(abs(values[1]),COAST_VERTICAL)
            self.assertEqual(values[2],0.)

    def test_actual_loss_search_can_turn_and_returning_neo_keeps_identity(self):
        before=self.acquire();identity=before['target_identity']['logical_target_id'];moving=[]
        for _ in range(40):
            d=self.feed([detection(1,.99)]);values,reason=self.command(d)
            self.assertEqual(d['selected_class_name'],'neo')
            self.assertEqual(d['measurement_time'],before['measurement_time'])
            if d['state']=='DIRECTIONAL_SEARCH' and any(values):
                moving.append(reason);self.assertEqual(values[1:],(0.,0.,0.))
        self.assertTrue(moving);self.assertTrue(all(r.startswith(('Search /','Scan /')) for r in moving))
        for _ in range(12):d=self.feed([detection(),detection(1,.99)])
        self.assertTrue(d['accepted']);self.assertTrue(d['confirmed'])
        self.assertEqual(d['target_identity']['logical_target_id'],identity)
        self.assertGreater(d['target_identity']['measurement_time'],before['measurement_time'])
        self.assertEqual(d['target_class_source'],'accepted_measurement')

    def test_restart_retains_identity_but_requires_new_measurement(self):
        before=self.acquire();started=self.t+.01;self.c.restart_control();self.t+=.02
        d=self.c.tick(self.t,640,480,self.t,self.s)
        self.assertEqual(d['target_identity'],before['target_identity'])
        self.assertFalse(d['accepted']);self.assertEqual(self.command(d,started)[0],ZERO)
        d=self.feed([detection(1,.99)]);self.assertEqual(self.command(d,started)[0],ZERO)
        d=self.feed([detection()]);self.assertTrue(d['accepted'])
        self.assertGreater(d['measurement_time'],started)
        self.assertTrue(any(self.command(d,started)[0]))

    def test_retained_identity_does_not_override_stale_video_heading_or_expired_coast(self):
        self.acquire()
        for _ in range(40):d=self.feed([])
        values,reason=dance_command(dict(d,target_policy=target_policy(DEFAULT_MODEL,DUAL_SOURCE_NAMES)),
                                    self.t+.30,9.,require_target_policy=True)
        self.assertEqual(values,ZERO);self.assertIn('fresh decision',reason)
        d=self.c.tick(self.t,640,480,self.t-2.,self.s)
        self.assertEqual(self.command(d)[0],ZERO)
        self.assertEqual(d['selected_class_name'],'neo')
        d=self.feed([]);d['heading']=None
        self.assertEqual(self.command(d)[0],ZERO)
        d.update(state='COAST',prediction_recovery=True,measurement_age_ms=651.)
        self.assertEqual(self.command(d)[0],ZERO)

    def test_explicit_reset_and_new_frame_size_clear_target_class(self):
        self.acquire();self.c.reset();d=self.feed([detection(1,.99)])
        self.assertIsNone(d['target_identity']);self.assertIsNone(d['selected_class_id'])
        self.assertEqual(self.command(d)[0],ZERO)
        self.acquire();self.c.tick(self.t,1280,720,self.t,self.s)
        self.assertIsNone(self.c.target_identity)

    def test_numerical_reset_discards_class_until_new_neo_is_confirmed(self):
        self.acquire();engine=self.c.tracker.engine
        with patch.object(engine,'update',side_effect=np.linalg.LinAlgError('offline fault')):
            d=self.feed([detection()])
        self.assertIsNone(d['target_identity']);self.assertFalse(d['confirmed'])
        self.assertEqual(self.command(d)[0],ZERO)
        for _ in range(8):d=self.feed([detection(1,.99)])
        self.assertIsNone(d['selected_class_id']);self.assertEqual(self.command(d)[0],ZERO)
        d=self.acquire();self.assertEqual(d['selected_class_name'],'neo')

    def test_class_provenance_cannot_authorize_a_different_tracker_identity(self):
        self.acquire();self.c.tracker.logical_target_id+=1
        d=self.c.tick(self.t,640,480,self.t,self.s)
        self.assertIsNone(d['target_identity']);self.assertIsNone(d['selected_class_id'])
        self.assertEqual(self.command(d)[0],ZERO)


class DanceGateTests(unittest.TestCase):
    def test_valid_neo_can_move_and_mini_or_missing_identity_cannot(self):
        d=allowed_decision()
        self.assertTrue(any(dance_command(d,10.,9.,require_target_policy=True)[0]))
        for change in ({'selected_class_id':1,'selected_class_name':'MINI PRO4'},
                       {'selected_class_id':None},{'selected_class_name':None}):
            self.assertEqual(dance_command({**d,**change},10.,9.,require_target_policy=True)[0],ZERO)
    def test_unknown_old_or_missing_model_policy_blocks_even_search_and_prediction(self):
        for state in ('TRACK','COAST','DIRECTIONAL_SEARCH'):
            for policy in (None,{},target_policy(PREVIOUS_MODEL),target_policy(LEGACY_MODEL),
                           {**target_policy(DEFAULT_MODEL,DUAL_SOURCE_NAMES),'target_class_id':1}):
                d={**allowed_decision(),'state':state,'target_policy':policy}
                values,reason=dance_command(d,10.,9.,require_target_policy=True)
                self.assertEqual(values,ZERO);self.assertIn('blocked',reason)
    def test_actual_worker_wire_and_logs_require_neo_policy(self):
        server=Server();events=[];state={'policy':False}
        def provider():
            d=allowed_decision(time.monotonic())
            if not state['policy']:d.pop('target_policy')
            return d
        control=ManualControl('127.0.0.1',server.connect,decision_provider=provider,
            require_target_policy=True,on_event=lambda e,d:events.append((e,d)))
        self.addCleanup(lambda:(control.stop(),control.thread.join(3)))
        control.start();control.update(set());control.enable();wait_for(lambda:control.enabled)
        control.start_dance()
        wait_for(lambda:any(e=='flight_command' and d['mode']=='DANCE' for e,d in events))
        self.assertFalse(any(any(float(v) for v in cmd.split()[1:]) for cmd in server.commands if cmd.startswith('rc ')))
        state['policy']=True;control.update(set())
        wait_for(lambda:any(e=='flight_command' and any(d['values']) for e,d in events))
        row=next(d for e,d in events if e=='flight_command' and any(d['values']))
        self.assertEqual(row['selected_class_name'],'neo');self.assertEqual(row['selected_class_id'],0)
        self.assertEqual(row['target_policy']['model_name'],DEFAULT_MODEL)
        self.assertEqual(row['measurement_id'],4);json.dumps(row,allow_nan=False)
        offset=len(events);state['policy']=False;control.update(set())
        wait_for(lambda:any(e=='flight_command' and d['gate_reason'].startswith('Dance blocked') for e,d in events[offset:]))
        row=next(d for e,d in events[offset:] if e=='flight_command' and d['gate_reason'].startswith('Dance blocked'))
        self.assertEqual(row['values'],[0.,0.,0.,0.])


class DisplayAndConfigTests(unittest.TestCase):
    def test_old_config_selects_new_model_and_preserves_user_settings(self):
        cfg=dict(schema_version=6,model_name=PREVIOUS_MODEL,settings=asdict(Settings(confidence=.77)),
                 axis_limits=dict(yaw=.2,vertical=.1,roll=.015,forward=.025))
        before=copy.deepcopy(cfg)
        self.assertEqual(model_from_config(cfg),DEFAULT_MODEL);self.assertEqual(cfg,before)
        self.assertEqual(settings_from_config(cfg).confidence,.77)
        cfg['schema_version']=7
        self.assertEqual(model_from_config(cfg),PREVIOUS_MODEL)
        self.assertEqual(model_from_config({}),DEFAULT_MODEL)
    def test_both_views_show_two_named_boxes_and_never_select_coincident_mini(self):
        frame=Frame(1,10.,np.zeros((480,640,3),np.uint8))
        ds=[dict(box=[240,180,300,220],confidence=.9,class_id=i,class_name=DUAL_CLASS_NAMES[i]) for i in (0,1)]
        d={**allowed_decision(), 'raw_detections':ds,'raw_detection_time':10.,
           'selected_measurement_box':ds[0]['box']}
        analysis=dict(result=dict(frame=frame,detections=ds,inference_ms=1.),decision=d,
            selected_box=dict(zip(('x1','y1','x2','y2'),ds[0]['box'])),selected_class_id=0)
        for mode in ('Analyzed frame','Live prediction'):
            with patch('follow_neo.presentation.overlay.draw_box') as draw:
                _,_,meta=render_bundle(frame,analysis,d,mode,now=10.05)
            calls=draw.call_args_list;self.assertEqual(len(calls),2)
            self.assertIn('neo',calls[0].args[3]);self.assertIn('SELECTED',calls[0].args[3])
            self.assertIn('MINI PRO4',calls[1].args[3]);self.assertNotIn('SELECTED',calls[1].args[3])
            self.assertEqual(calls[0].args[2],NEO_COLOR);self.assertEqual(calls[1].args[2],MINI_COLOR)
            self.assertEqual(len(meta['detections']),2);json.dumps(meta,allow_nan=False)
        self.assertFalse(frame.image.any())
    def test_old_model_dance_selection_is_rejected_with_reason_and_log(self):
        w=FollowLabWindow.__new__(FollowLabWindow)
        w.session=Mock(model_path=Path(PREVIOUS_MODEL));w.dance_selected=False
        w.notice=Mock();w.show_control_indicator=Mock();w.log_control=Mock()
        w.toggle_dance();self.assertFalse(w.dance_selected)
        self.assertIn(DEFAULT_MODEL,w.dance_rejection)
        self.assertEqual(w.log_control.call_args.args[0],'dance_rejected')
    def test_gui_transport_enforces_policy(self):
        w=FollowLabWindow.__new__(FollowLabWindow);w.manual=None;w.session=None
        w.ip=Mock(get=Mock(return_value='127.0.0.1'));w.log_control=Mock();w.dance_decision=Mock()
        w.current_axis_limits=Mock(return_value=dict(yaw=.5,vertical=.5,roll=.015,forward=.015))
        w.dance_selected=False;w.canvas=Mock();w.clear_keyboard_input=Mock()
        with patch('follow_neo.ui.window.ManualControl') as constructor:w.toggle_manual()
        self.assertIs(constructor.call_args.kwargs['require_target_policy'],True)


class SessionClassTests(unittest.TestCase):
    def test_live_worker_keeps_class_provenance_and_filter_counts_in_log(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('follow_neo.runtime.session.resolve_model',return_value=(Path(folder)/DEFAULT_MODEL,{'name':'test model'})):
                s=FollowSession(folder,'127.0.0.1',settings=Settings(confidence=.8,new_track_confidence=.8))
            s.target_policy=target_policy(DEFAULT_MODEL,DUAL_SOURCE_NAMES)
            s.receiver.state='STREAMING';s.control_thread.start()
            try:
                for i in range(1,7):
                    # Simulate video cadence, including Windows clock granularity.
                    # Back-to-back frames may otherwise have equal timestamps.
                    time.sleep(.04)
                    frame=Frame(i,time.monotonic()-.02,np.zeros((480,640,3),np.uint8))
                    ds=[dict(box=list(asdict(d.box).values()),confidence=d.confidence,
                             class_id=d.class_id,class_name=d.class_name,model_class_name=d.model_class_name)
                        for d in (detection(),detection(1,.99))]
                    s.receiver.frames.set(frame)
                    s.result.set(dict(frame=frame,generation=0,detections=ds,completed_at=frame.decoded_at,
                                      inference_ms=1.,preprocess_ms=1.,postprocess_ms=1.))
                    wait_for(lambda:s.analysis.get() is not None and s.analysis.get()['result']['frame'].frame_id==i)
                d=s.analysis.get()['decision']
                self.assertTrue(d['confirmed'],d['tracking']);self.assertTrue(d['accepted'],d['tracking'])
                self.assertEqual(d['selected_class_name'],'neo');self.assertEqual(len(d['raw_detections']),2)
            finally:s.stop()
            self.assertIsNone(s.model_error);self.assertIsNone(s.log.error)
            rows=[json.loads(line) for line in (s.output/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            observations=[row for row in rows if row['event']=='observation']
            self.assertTrue(observations)
            last=observations[-1];self.assertEqual(last['decision']['detection_summary']['display_only'],1)
            self.assertEqual(last['decision']['target_identity']['class_name'],'neo')
            self.assertIn('decode_gaps_over_250ms',last['video_timing'])
            self.assertEqual(last['detections'][0]['class_name'],'neo')
            self.assertEqual(last['detections'][0]['model_class_name'],DUAL_SOURCE_NAMES[0])


if __name__=='__main__':unittest.main()
