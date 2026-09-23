"""Prediction never masquerades as a new YOLO measurement or flight authority."""
import sys,time,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
from unittest.mock import Mock,patch
import numpy as np
from follow_neo.controller import FollowController
from follow_neo.tracker import BBoxTracker
from follow_neo.types import Settings,Detection,Box,Intent
from follow_neo.dance import dance_command,ZERO
from follow_neo.prediction import continuation_threshold
from follow_neo.overlay import render_bundle,box_at_frame
from follow_neo.video import Frame
from follow_neo.manual import ManualControl
from follow_neo.control_status import control_indicator
from test_manual import Server,wait_for


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.t=BBoxTracker();self.box=Box(280,180,360,240)

    def update(self,stamp,confidence,box=None,frame_id=None):
        return self.t.update([Detection(box or self.box,confidence)],stamp,stamp+.05,
                             frame_id or round(stamp*100),.3,.3,.25)

    def confirm(self):
        self.assertTrue(self.update(10,.8));self.assertTrue(self.update(10.05,.8))
        self.assertTrue(self.t.confirmed)

    def test_weak_detection_cannot_acquire_or_confirm(self):
        self.assertFalse(self.update(10,.27))
        self.assertFalse(self.t.initialized)
        self.assertTrue(self.update(10.05,.8))
        self.assertFalse(self.update(10.1,.27));self.assertFalse(self.t.confirmed)

    def test_matched_weak_detection_is_real_timestamped_measurement(self):
        self.confirm();self.assertTrue(self.update(10.1,.27))
        self.assertTrue(self.t.measurement_weak)
        self.assertEqual(self.t.measurement_time,10.1)
        self.assertEqual(self.t.last_strong_time,10.05)

    def test_weak_detection_cannot_sustain_track_forever(self):
        self.confirm()
        for stamp in (10.15,10.3,10.5,10.65):self.assertTrue(self.update(stamp,.27))
        self.assertFalse(self.update(10.71,.27))
        self.assertEqual(self.t.measurement_time,10.65)

    def test_weak_distractor_and_large_size_change_are_rejected(self):
        self.confirm()
        self.assertFalse(self.update(10.1,.27,Box(330,210,410,270)))
        self.assertFalse(self.update(10.2,.27,Box(260,150,390,280)))

    def test_source_time_association_uses_motion_aligned_box(self):
        # Update 6 keeps Post at measurement time, so a later control poll
        # cannot move the association gate away from the next source frame.
        for i in range(10):
            stamp=10+i*.03
            self.assertTrue(self.update(stamp,.8,Box(200+i*6,180,280+i*6,240)))
            self.t.snapshot(stamp+.15)
        aligned=Box(260,180,340,240)
        self.assertIsNotNone(self.t.select([Detection(aligned,.8)],.3,10.30,.3,.25))
        self.assertEqual(self.t.last_time,10.27)

    def test_old_velocity_is_not_reused_after_long_measurement_gap(self):
        self.confirm();self.update(10.1,.8,Box(290,180,370,240))
        self.assertIsNotNone(self.t.velocity)
        self.assertTrue(self.update(10.72,.8,Box(290,180,370,240)))
        self.assertIsNone(self.t.velocity)
        self.assertAlmostEqual(self.t.snapshot(10.77).vx,0.)

    def test_threshold_band_preserves_user_acquisition_setting(self):
        self.assertEqual(continuation_threshold(.3),.25)
        self.assertAlmostEqual(continuation_threshold(.75),.6)
        self.assertEqual(continuation_threshold(.1),.1)


class CoastTests(unittest.TestCase):
    def ready(self,box=Box(710,270,810,330)):
        c=FollowController();s=Settings(confidence=.3,new_track_confidence=.25,stop_width=.38)
        for i in range(6):
            stamp=10+i*.05
            c.observe([Detection(box,.8)],stamp,stamp+.02,i+1,(1280,720),s)
            d=c.tick(stamp+.02,1280,720,stamp+.02,s)
        self.assertEqual(d['state'],'TRACK')
        return c,s

    def missing(self,c,s,now,frame=100):
        c.observe([],now-.02,now,frame,(1280,720),s)
        return c.tick(now,1280,720,now,s)

    def test_orientation_continues_but_translation_stops_after_250ms(self):
        c,s=self.ready();d=self.missing(c,s,10.60)
        d['prediction_forward_allowed']=False  # no separate Update 6 extension
        self.assertTrue(d['prediction_recovery']);self.assertEqual(d['state'],'COAST')
        values,status=dance_command(d,10.60,9)
        self.assertGreater(values[0],0);self.assertGreater(values[1],0)
        self.assertEqual(values[2:],(0.,0.))
        self.assertLessEqual(abs(values[0]),.25);self.assertLessEqual(abs(values[1]),.20)
        self.assertEqual(d['measurement_time'],10.25);self.assertFalse(d['accepted'])
        self.assertTrue(status.startswith('Kalman recovery'))

    def test_frozen_coast_decision_expires_at_transport_time(self):
        c,s=self.ready();d=self.missing(c,s,10.80)
        self.assertNotEqual(dance_command(d,10.85,9)[0],ZERO)
        self.assertEqual(dance_command(d,10.901,9)[0],ZERO)

    def test_prediction_stops_after_horizon_despite_fresh_empty_results(self):
        c,s=self.ready()
        for i in range(15):d=self.missing(c,s,10.32+i*.05,100+i)
        self.assertFalse(d['prediction_recovery']);self.assertIsNone(d['track_box'])
        self.assertEqual(dance_command(d,d['prediction_time'],9)[0],ZERO)

    def test_competing_detection_prevents_coast(self):
        c,s=self.ready();now=10.6
        c.observe([Detection(Box(0,0,20,20),.99)],now-.02,now,100,(1280,720),s)
        d=c.tick(now,1280,720,now,s)
        self.assertFalse(d['prediction_recovery']);self.assertEqual(dance_command(d,now,9)[0],ZERO)

    def test_near_or_clipped_target_never_uses_coast(self):
        for box in (Box(600,270,950,370),Box(2,270,102,330)):
            c,s=self.ready(box);d=self.missing(c,s,10.6)
            self.assertFalse(d['prediction_recovery']);self.assertEqual(dance_command(d,10.6,9)[0],ZERO)

    def test_stale_video_and_terminal_state_override_coast(self):
        c,s=self.ready();d=self.missing(c,s,10.6)
        for change in ({'stale':True},{'spacing_phase':'STOPPED'},{'spacing_phase':'TRACK_PAUSE'},
                       {'spacing_close_guard_armed':True},{'confirmed':False}):
            bad={**d,**change};self.assertEqual(dance_command(bad,10.6,9)[0],ZERO)

    def test_transport_cannot_send_forged_translation_during_late_coast(self):
        c,s=self.ready();d=self.missing(c,s,10.6)
        d['intent']={a:1. for a in ('yaw','vertical','roll','forward')}
        d['brief_detection_gap']=True
        d['prediction_forward_allowed']=False
        values,_=dance_command(d,10.61,9)
        self.assertEqual(values[2:],(0.,0.));self.assertLessEqual(values[0],.25)

    def test_prediction_overlay_label_and_time_limit(self):
        c,s=self.ready();d=self.missing(c,s,10.6)
        frame=Frame(1,10.62,np.zeros((720,1280,3),np.uint8))
        with patch('follow_neo.overlay.draw_box') as draw:
            _,_,meta=render_bundle(frame,None,d,'Live prediction',now=10.62)
        self.assertIsNotNone(meta['display_box']);self.assertEqual(meta['track_support'],'prediction')
        self.assertIn('NO YOLO',draw.call_args.args[3]);self.assertTrue(draw.call_args.kwargs['dashed'])
        self.assertIsNone(box_at_frame(frame,d,10.901))

    def test_coast_status_does_not_claim_new_detection(self):
        control=Mock();control.mode='DANCE';control.thread.is_alive.return_value=True
        control.stop_event.is_set.return_value=False;control.wanted=control.enabled=True
        control.axis_limits={};control.motion_hold=False;control.dance_status='Kalman recovery / APPROACH'
        title,detail,_=control_indicator(control)
        self.assertIn('PREDICTION',title);self.assertIn('No fresh YOLO',detail)

    def test_actual_transport_respects_limits_and_clears_previous_roll(self):
        server=Server();events=[];state={'coast':False}
        def provider():
            now=time.monotonic()
            return {'prediction_time':now,'measurement_age_ms':400 if state['coast'] else 20,
                'state':'COAST' if state['coast'] else 'TRACK','confirmed':True,
                'accepted':not state['coast'],'spacing_phase':'APPROACH','stale':False,
                'prediction_recovery':state['coast'],'spacing_close_guard_armed':False,
                'intent':dict(yaw=1.,vertical=1.,roll=1.,forward=1.),
                'intent_basis':dict(yaw=1.,vertical=1.,roll=1.,forward=1.)}
        c=ManualControl('127.0.0.1',server.connect,
            on_event=lambda event,data:events.append((event,data)),decision_provider=provider)
        self.addCleanup(lambda:(c.stop(),c.thread.join(3)))
        c.start();c.update(set());c.enable();wait_for(lambda:c.enabled);c.start_dance()
        c.set_axis_limits(dict(yaw=.5,vertical=.5,roll=.015,forward=.015))
        wait_for(lambda:any(e=='flight_command' and d.get('values',[0]*4)[2]>0 for e,d in events))
        state['coast']=True;c.update(set())
        wait_for(lambda:any(e=='flight_command' and d.get('prediction_recovery') for e,d in events))
        d=next(d for e,d in events if e=='flight_command' and d.get('prediction_recovery'))
        self.assertEqual(d['values'][2:],[0.,0.])
        self.assertLessEqual(d['values'][0],.125);self.assertLessEqual(d['values'][1],.10)


if __name__=='__main__':unittest.main()
