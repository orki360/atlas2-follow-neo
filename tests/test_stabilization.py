"""Regression tests for the recorded yaw, association and keyboard failures."""
import socket
import threading
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from follow_neo.centering import DampedTrackingController, HeadingRate
from follow_neo.control_transport import ControlChannel
from follow_neo.control_status import acknowledged_command_text
from follow_neo.dance import DanceSmoother
from follow_neo.botsort_tracker import BoTSORTTracker
from follow_neo.recovery_search import RecoverySearch
from follow_neo.types import Box, Detection, Kinematics, Settings, settings_from_config


def heading(t,yaw=0):return dict(time=t,yaw_deg=yaw,source='test')
def detection(x=200,confidence=.95):return Detection(Box(x,180,x+60,220),confidence)


class CenteringTests(unittest.TestCase):
    def calc(self,c,ex,t,heading_value=None):
        k=Kinematics(True,320+ex*320,240,60,40,0,0)
        return c.compute(k,640,480,0,False,t,Settings(),heading_value,t,ex)[0]
    def test_small_center_jitter_does_not_turn(self):
        c=DampedTrackingController()
        for i,ex in enumerate([.02,-.04,.05,-.02,0]):
            self.assertEqual(self.calc(c,ex,10+i*.04).yaw,0)
    def test_yaw_has_no_minimum_floor_and_is_bounded(self):
        c=DampedTrackingController();large=self.calc(c,.3,10).yaw
        small=self.calc(c,.03,10.3).yaw
        self.assertGreater(small,0);self.assertLess(small,large/5)
        self.assertLessEqual(abs(self.calc(c,1.,10.6).yaw),.55*Settings().yaw_limit)
    def test_closing_target_brakes_before_crossing_center(self):
        c=DampedTrackingController()
        self.calc(c,.3,10)
        for i,ex in enumerate([.24,.18,.12,.06],1):out=self.calc(c,ex,10+i*.04)
        self.assertEqual(out.yaw,0);self.assertTrue(c.diagnostics['braking'])
    def test_duplicate_measurement_does_not_change_error_derivative(self):
        c=DampedTrackingController();self.calc(c,.3,10);self.calc(c,.25,10.04)
        before=c.error_rate;self.calc(c,.8,10.04)
        self.assertEqual(c.error_rate,before)
    def test_stale_heading_does_not_supply_rate_feedback(self):
        c=DampedTrackingController();self.calc(c,.3,10,heading(10,0))
        self.calc(c,.3,10.1,heading(10.1,3));self.assertIsNotNone(c.diagnostics['heading_rate_deg_s'])
        self.calc(c,.3,10.5,heading(10.1,3));self.assertIsNone(c.diagnostics['heading_rate_deg_s'])
    def test_heading_wrap_and_source_switch(self):
        r=HeadingRate();r.update(heading(10,179),10)
        self.assertAlmostEqual(r.update(heading(10.1,-179),10.1),20)
        self.assertIsNone(r.update(dict(time=10.2,yaw_deg=0,source='other'),10.2))
    def test_forward_is_cut_during_misalignment_and_returns_gradually(self):
        c=DampedTrackingController();self.calc(c,.7,10)
        self.assertEqual(c.forward_factor,0)
        self.calc(c,0,10.05);self.assertGreater(c.forward_factor,0);self.assertLessEqual(c.forward_factor,.051)
    def test_fast_body_turn_reduces_forward(self):
        c=DampedTrackingController();self.calc(c,.1,10,heading(10,0))
        self.calc(c,.1,10.1,heading(10.1,6));self.assertEqual(c.forward_factor,0)
    def test_smoother_allows_immediate_yaw_braking_and_reversal_via_zero(self):
        s=DanceSmoother()
        for i in range(5):s.update((.4,0,0,.2),10+i*.05,damped_yaw=True)
        self.assertAlmostEqual(s.update((.03,0,0,.2),10.25,damped_yaw=True)[0],.03)
        self.assertEqual(s.update((-.3,0,0,.2),10.3,damped_yaw=True)[0],0)
        self.assertLess(s.update((-.3,0,0,.2),10.35,damped_yaw=True)[0],0)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.tracker=BoTSORTTracker((640,480));self.t=10.;self.i=0
        self.feed([detection()]);self.feed([detection()])
    def feed(self,ds,dt=.04):
        self.i+=1;self.t+=dt
        return self.tracker.update(ds,self.t,self.t,self.i,.8,.8)
    def test_reference_recovers_when_prediction_drifts_offscreen(self):
        self.tracker.target.mean[0]=1100
        for i in range(3):accepted=self.feed([detection(250)])
        self.assertTrue(accepted);self.assertEqual(self.tracker.association_reason,'verified_strong_recovery')
    def test_candidate_survives_one_missing_frame(self):
        self.assertFalse(self.feed([detection(255,.2)]))
        self.feed([]);self.assertTrue(self.tracker.candidate_pending)
        for _ in range(3):accepted=self.feed([detection(255,.2)])
        self.assertTrue(accepted);self.assertTrue(self.tracker.measurement_weak)
    def test_long_gap_discards_old_candidate_evidence(self):
        self.feed([detection(255,.2)]);self.feed([],dt=.2)
        self.assertIsNone(self.tracker.pending)
        self.assertFalse(self.feed([detection(255,.2)]));self.assertEqual(self.tracker.pending['hits'],1)
    def test_broad_recovery_accepts_coherent_mixed_confidence(self):
        self.feed([],dt=1.)
        accepted=False
        for confidence in [.95,.95,.2,.95,.2,.95,.2,.95]:accepted=self.feed([detection(490,confidence)])
        self.assertTrue(accepted);self.assertEqual(self.tracker.association_reason,'verified_returning_target')
    def test_pause_budget_is_not_renewed_by_new_candidates(self):
        self.feed([detection(255,.2)])
        for _ in range(12):
            self.feed([],dt=.16);self.feed([detection(255,.2)])
        self.assertFalse(self.tracker.candidate_pending)
    def test_botsort_prediction_uses_actual_source_interval(self):
        x=self.tracker.target.mean[0];self.tracker.target.mean[4]=3.
        self.feed([],dt=.1)
        self.assertAlmostEqual(self.tracker.target.mean[0]-x,9.)
    def test_camera_motion_transforms_pending_candidate_once_per_frame(self):
        self.feed([detection(255,.2)])
        old=self.tracker.pending['point'].copy()
        def warp(image,detections):
            self.tracker.engine.gmc.last_warp=np.array([[1.,0.,25.],[0.,1.,0.]])
            return self.tracker.engine.gmc.last_warp
        self.tracker.engine.gmc.apply=warp
        self.feed([])
        np.testing.assert_allclose(self.tracker.pending['point'],old+[25,0])


class SearchTimingTests(unittest.TestCase):
    def start(self,auto=True):
        self.r=RecoverySearch();self.s=Settings(search_yaw_degrees=60,edge_search_seconds=2,search_auto_duration=auto)
        self.r.observe(True,False,True,Box(280,220,360,260),Kinematics(),640,480,10,1,self.s)
        self.r.update(10,False,False,'APPROACH',False,self.s,heading(10))
        return self.r.update(10.6,True,False,'APPROACH',False,self.s,heading(10.6,4))
    def test_auto_budget_and_loss_heading_anchor(self):
        d=self.start();self.assertAlmostEqual(d['duration_seconds'],4.75)
        self.assertEqual(d['requested_duration_seconds'],2);self.assertEqual(d['scan_offset_degrees'],4)
    def test_fixed_budget_is_preserved_and_auto_is_capped(self):
        self.assertEqual(self.start(False)['duration_seconds'],2)
        self.start();self.r.reset();self.s=replace(self.s,search_yaw_degrees=180)
        self.r.observe(True,False,True,Box(280,220,360,260),Kinematics(),640,480,10,1,self.s)
        d=self.r.update(10.6,True,False,'APPROACH',False,self.s,heading(10.6))
        self.assertEqual(d['duration_seconds'],10)
    def test_scan_brakes_before_boundary_when_heading_is_closing_fast(self):
        self.start()
        self.r.update(10.8,True,False,'APPROACH',False,self.s,heading(10.8,-23))
        d=self.r.update(10.9,True,False,'APPROACH',False,self.s,heading(10.9,-28))
        self.assertEqual(d['scan_yaw'],0);self.assertLess(d['scan_offset_degrees'],-27)
    def test_config_preserves_explicit_fixed_time_and_other_values(self):
        s=settings_from_config(dict(schema_version=6,settings=dict(search_auto_duration=False,confidence=.8)))
        self.assertFalse(s.search_auto_duration);self.assertEqual(s.confidence,.8)
    def test_boost_uses_departing_box_edge_before_center_reaches_outer_ten_percent(self):
        r=RecoverySearch();s=Settings(search_yaw_degrees=60)
        for i in range(10):
            t=10+i*.04;x=910+i*10
            r.observe(True,False,True,Box(x,320,x+200,390),Kinematics(),1280,720,t,i,s)
            r.update(t,False,False,'APPROACH',False,s,heading(t))
        self.assertLess((x+100)/1280,.9)
        d=r.update(t+.5,True,False,'APPROACH',False,s,heading(t+.5))
        self.assertEqual(d['phase'],'BOOST');self.assertEqual(d['direction'],1)


class NeutralizationTests(unittest.TestCase):
    def exchange(self,ack_zero):
        client,server=socket.socketpair();self.addCleanup(client.close);self.addCleanup(server.close)
        changed=threading.Event();zero_seen=threading.Event();release_ack=threading.Event()
        events=[];result=[];commands=[]
        def guard(keys):
            return 'movement keys changed while acknowledgement was pending' if keys and changed.is_set() else None
        channel=ControlChannel(client,guard,lambda e,d:events.append((e,d)))
        def run():
            try:result.append(channel.command('rc 0 0 0 -0.1',motion_keys={'down'}))
            except Exception as exc:result.append(exc)
        worker=threading.Thread(target=run);worker.start()
        stream=server.makefile('rb');self.addCleanup(stream.close)
        server.settimeout(1)
        commands.append(stream.readline().decode().strip());time.sleep(.046);changed.set()
        commands.append(stream.readline().decode().strip());zero_seen.set()
        server.sendall(b'success\r\n');time.sleep(.025)
        self.assertTrue(worker.is_alive(),'Old RC acknowledgement must not confirm neutral')
        if ack_zero:server.sendall(b'success\r\n')
        worker.join(.4);self.assertFalse(worker.is_alive())
        self.assertEqual(commands,['rc 0 0 0 -0.1','rc 0 0 0 0'])
        return channel,server,result,events
    def test_separate_neutral_ack_keeps_connection_usable(self):
        channel,server,result,events=self.exchange(True)
        self.assertEqual(result,[False]);self.assertFalse(channel.pending)
        self.assertTrue(any(e=='control_motion_neutralized' for e,d in events))
        def reply():
            self.assertEqual(server.recv(100),b'rc 0 0 0 0\r\n');server.sendall(b'success\r\n')
        peer=threading.Thread(target=reply);peer.start()
        self.assertTrue(channel.command('rc 0 0 0 0'));peer.join(1)
    def test_missing_neutral_ack_pauses_without_closing(self):
        channel,server,result,events=self.exchange(False)
        self.assertEqual(result,[False]);self.assertTrue(channel.paused)
        self.assertTrue(channel.pending);self.assertGreaterEqual(channel.sock.fileno(),0)
        self.assertFalse(any(e=='control_motion_neutralized' for e,d in events))
    def test_ack_display_distinguishes_authority_and_stale_ack(self):
        control=SimpleNamespace(enabled=True,wanted=True,last_acknowledged=dict(time=10,mode='DANCE',values=[.1,0,0,.2]))
        self.assertIn('RC ACK DANCE',acknowledged_command_text(control,10.1))
        self.assertIn('no recent',acknowledged_command_text(control,10.5))
        control.wanted=False;self.assertIn('no active',acknowledged_command_text(control,10.1))
