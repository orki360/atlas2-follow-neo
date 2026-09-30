"""Measured search, authority loss and the recorded heading timestamp race."""
import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
from follow_neo.types import Settings,Box,Kinematics,Intent
from follow_neo.telemetry import heading_snapshot,fresh_heading
from follow_neo.recovery_search import RecoverySearch,scan_command
from follow_neo.search_test import SearchTest
from follow_neo.policy import DeterministicTrackingPolicy
from follow_neo.manual import ManualControl
from follow_neo.session import FollowSession
from follow_neo.overlay import render_bundle
from follow_neo.video import Frame
from test_manual import Server,wait_for
from test_control_scan_fix import AsyncServer


def heading(t,yaw=0.):return dict(time=t,yaw_deg=yaw,source='AircraftAttitude')


class SearchContracts(unittest.TestCase):
    def start(self,scenario='RIGHT',angle=80):
        r=RecoverySearch();s=Settings(search_yaw_degrees=angle)
        r.start_test(10,s,heading(10),scenario)
        return r,s
    def tick(self,r,s,t=10.2,yaw=0,**kw):
        return r.update(t,True,False,'APPROACH',False,s,heading(t,yaw),**kw)
    def decision(self,d,t=10.2):
        return dict(edge_search=d,prediction_time=t,state='DIRECTIONAL_SEARCH',
                    accepted=False,stale=False,spacing_phase='APPROACH')
    def test_heading_snapshot_orders_comparison_after_concurrent_sample(self):
        calls=[]
        receiver=Mock();receiver.get.side_effect=lambda:(calls.append('sample') or heading(184624.390,85.1))
        sample,now=heading_snapshot(receiver,lambda:(calls.append('time') or 184624.406))
        self.assertEqual(calls,['sample','time'])
        self.assertFalse(fresh_heading(sample,184624.375))  # recorded old decision time
        self.assertTrue(fresh_heading(sample,now))
    def test_real_and_debug_planners_select_identical_ranges(self):
        for scenario,cx,bounds in [('LEFT',100,(-80,0)),('RIGHT',900,(0,80)),('CENTER',500,(-40,40))]:
            r=RecoverySearch();s=Settings(search_yaw_degrees=80)
            r.observe(True,False,True,Box(cx-30,100,cx+30,140),Kinematics(),1000,600,10,1,s)
            r.update(10,False,False,'APPROACH',False,s,heading(10))
            real=self.tick(r,s,10.5)
            debug,_=self.start(scenario)
            self.assertEqual((real['scan_min_degrees'],real['scan_max_degrees']),bounds)
            self.assertEqual((debug.low,debug.high),bounds)
    def test_full_scale_both_directions_then_return_to_anchor(self):
        for scenario,sign in [('LEFT',-1),('RIGHT',1)]:
            r,s=self.start(scenario);d=self.tick(r,s)
            self.assertEqual(scan_command(self.decision(d),10.2,9),sign)
            self.tick(r,s,10.3,sign*40)
            d=self.tick(r,s,10.4,sign*80)
            self.assertEqual(d['scan_target_degrees'],0)
            d=self.tick(r,s,10.6,sign*80)
            self.assertLess(sign*d['scan_yaw'],0)
            self.assertEqual(d['scan_min_degrees'],min(0,sign*80))
    def test_short_heading_gap_pauses_and_resumes_same_arc(self):
        r,s=self.start();end=r.until
        d=r.update(10.2,True,False,'APPROACH',False,s,None)
        self.assertTrue(d['paused']);self.assertFalse(d['active']);self.assertFalse(d['consumed'])
        d=self.tick(r,s,10.3,2)
        self.assertTrue(d['active']);self.assertEqual(d['until'],end)
    def test_candidate_pause_is_not_hover_and_cannot_extend_deadline(self):
        r,s=self.start();d=self.tick(r,s,candidate_pending=True)
        p=DeterministicTrackingPolicy();p.ever=True
        p.update(10.2,Kinematics(),640,480,9,1,Intent(),s,accepted=False,search=d)
        self.assertEqual(p.state,'SEARCH_PAUSED')
        end=r.until;d=self.tick(r,s,10.4)
        self.assertTrue(d['active']);self.assertEqual(d['until'],end)
    def test_reacquired_target_seeds_next_loss_without_consumed_flag(self):
        r,s=self.start()
        for i in range(3):
            t=10.2+i*.04
            r.observe(True,False,True,Box(470,100,530,140),Kinematics(),1000,600,t,i,s)
        self.assertEqual(r.phase,'IDLE');self.assertFalse(r.consumed)
        d=self.tick(r,s,10.32)
        p=DeterministicTrackingPolicy();p.ever=True
        p.update(10.32,Kinematics(),1000,600,10.28,2,Intent(),s,accepted=False,search=d)
        self.assertEqual(p.state,'COAST')
        d=self.tick(r,s,10.8)
        self.assertTrue(d['active']);self.assertEqual(d['span_mode'],'centered')
    def test_control_loss_cancels_and_does_not_resume_old_arc(self):
        r,s=self.start();d=self.tick(r,s,control_available=False)
        self.assertFalse(d['active']);self.assertEqual(d['reason'],'control_unavailable')
        self.assertFalse(self.tick(r,s,10.3)['active'])
    def test_scan_gate_rejects_paused_stale_and_outward_boundary(self):
        r,s=self.start();d=self.tick(r,s)
        for change in [dict(paused=True),dict(heading_time=9),dict(scan_offset_degrees=80),dict(scan_yaw=1.1),dict(scan_max_degrees=90)]:
            self.assertEqual(scan_command(self.decision({**d,**change}),10.2,9),0)
    def test_braking_reduces_command_before_endpoint(self):
        r,s=self.start();r.rate.update=Mock(return_value=60.)
        d=self.tick(r,s,yaw=60)
        self.assertEqual(d['scan_yaw'],0)
        r.rate.update=Mock(return_value=0.)
        d=self.tick(r,s,10.3,75)
        self.assertGreater(d['scan_yaw'],0);self.assertLess(d['scan_yaw'],1)


class DebugContracts(unittest.TestCase):
    def test_redundant_enable_does_not_invalidate_active_dance_arc(self):
        c=ManualControl('127.0.0.1');c.start_dance();c.enabled=c.wanted=True
        started=c.dance_started;c.enable();self.assertEqual(c.dance_started,started)
        c.release();c.enable();self.assertGreaterEqual(c.dance_started,started)
    def test_stop_while_test_command_ack_is_pending_neutralizes_immediately(self):
        peer=AsyncServer(True);events=[]
        c=ManualControl('127.0.0.1',peer.connect,on_event=lambda e,d:events.append((e,d)))
        self.addCleanup(lambda:(c.stop(),c.thread.join(3)))
        c.update(set());c.enable();c.start();wait_for(lambda:c.enabled)
        c.start_search_test(Settings(search_yaw_degrees=80),lambda:heading(time.monotonic()),'RIGHT')
        self.assertTrue(peer.motion.wait(.3));c.stop_search_test()
        self.assertTrue(peer.zero.wait(.20));wait_for(lambda:c.communication_paused)
        self.assertFalse(peer.disabled.is_set());self.assertTrue(c.thread.is_alive())
        c.stop();c.thread.join(1)
        self.assertFalse(c.wanted);self.assertFalse(c.enabled)
        self.assertTrue(any(e=='search_test_summary' and not d['tail_complete'] for e,d in events))
    def test_dance_scan_uses_full_scale_then_manual_override_obeys_slider(self):
        peer=Server();r=RecoverySearch();s=Settings(search_yaw_degrees=80)
        def decision():
            now=time.monotonic()
            if r.started is None:r.start_test(now,s,heading(now),'RIGHT')
            search=r.update(now,True,False,'APPROACH',False,s,heading(now))
            return dict(prediction_time=now,state='DIRECTIONAL_SEARCH',accepted=False,
                        stale=False,spacing_phase='APPROACH',edge_search=search)
        c=ManualControl('127.0.0.1',peer.connect,decision_provider=decision)
        self.addCleanup(lambda:(c.stop(),c.thread.join(3)))
        c.set_axis_limits(dict(yaw=.02,vertical=1.,roll=1.,forward=1.))
        c.start_dance();c.update(set());c.enable();c.start()
        wait_for(lambda:'rc 1.0000 0.0000 0.0000 0.0000' in peer.commands)
        c.update({'a'});wait_for(lambda:'rc -0.0200 0.0000 0.0000 0.0000' in peer.commands)

    def test_measured_wraparound_and_post_stop_drift_are_logged(self):
        events=[];test=SearchTest(Settings(search_yaw_degrees=80),heading(10,179),10,'RIGHT',lambda e,d:events.append((e,d)))
        test.update(heading(10.2,-176),10.2)
        self.assertEqual(test.offset,5)
        test.stop(10.3,'operator_stop')
        for i in range(11):test.update(heading(10.4+i*.2,-173),10.4+i*.2,False)
        summary=next(d for e,d in events if e=='search_test_summary')
        self.assertEqual(summary['measured_offset_degrees'],8)
        self.assertTrue(summary['tail_complete']);self.assertTrue(summary['measurement_complete'])
    def test_one_cycle_stops_on_measured_boundaries(self):
        test=SearchTest(Settings(search_yaw_degrees=80),heading(10),10,'RIGHT',lambda *a:None)
        for t,yaw in [(10.2,40),(10.4,80),(10.6,40),(10.8,0)]:values=test.update(heading(t,yaw),t)
        self.assertEqual(test.reason,'scan_complete');self.assertEqual(values,(0,0,0,0))
    def test_start_requires_fresh_heading_and_explicit_authority(self):
        control=ManualControl('127.0.0.1');control.update(set())
        with self.assertRaises(ValueError):control.start_search_test(Settings(),lambda:heading(time.monotonic()),'RIGHT')
        control.enabled=control.wanted=True
        with self.assertRaises(ValueError):control.start_search_test(Settings(),lambda:heading(time.monotonic()-1),'RIGHT')
        self.assertEqual(control.mode,'MANUAL')
    def test_real_control_worker_sends_yaw_only_full_scale_and_stop_zero(self):
        peer=Server();events=[]
        c=ManualControl('127.0.0.1',peer.connect,on_event=lambda e,d:events.append((e,d)))
        self.addCleanup(lambda:(c.stop(),c.thread.join(3)))
        c.set_axis_limits(dict(yaw=.01,vertical=1.,roll=1.,forward=1.))
        c.update(set());c.enable();c.start();wait_for(lambda:c.enabled)
        c.start_search_test(Settings(search_yaw_degrees=80),lambda:heading(time.monotonic()),'RIGHT')
        wait_for(lambda:'rc 1.0000 0.0000 0.0000 0.0000' in peer.commands)
        c.stop_search_test();index=len(peer.commands)
        wait_for(lambda:any(x=='rc 0.0000 0.0000 0.0000 0.0000' for x in peer.commands[index:]))
        self.assertEqual(c.mode,'MANUAL')
        commands=[d for e,d in events if e=='flight_command' and d['mode']=='SEARCH_TEST']
        self.assertTrue(commands)
        self.assertTrue(all(d['values'][1:]==[0,0,0] and d['search_yaw_override'] for d in commands))
    def test_dedicated_log_contains_heading_command_ack_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            session=FollowSession.__new__(FollowSession);session.log=Mock();session.output=Path(tmp)
            session.flight_commands_sent=0;session.search_test_log=None
            session.record_control('search_test_started',dict(run_id='test'))
            session.record_control('search_test_sample',dict(measured_offset_degrees=23.4))
            session.record_control('control_command_ack',dict(command='rc 1 0 0 0',reply='success'))
            session.record_control('search_test_summary',dict(run_id='test',tail_complete=True))
            rows=[json.loads(s) for s in (Path(tmp)/'search_test_test/events.jsonl').read_text().splitlines()]
            self.assertEqual([r['event'] for r in rows],['search_test_started','search_test_sample','control_command_ack','search_test_summary'])
            self.assertEqual(rows[1]['measured_offset_degrees'],23.4)
    def test_prediction_video_labels_disconnected_search(self):
        frame=Frame(1,10,np.zeros((180,640,3),np.uint8))
        d=dict(state='DIRECTIONAL_SEARCH',control_status=dict(active=False,mode='DANCE',time=10))
        with patch('follow_neo.overlay.cv2.putText',wraps=__import__('cv2').putText) as put:
            _,_,metadata=render_bundle(frame,None,d,'Live prediction',now=10)
        self.assertTrue(any('CONTROL OFF - SEARCH NOT EXECUTING' in call.args[1] for call in put.call_args_list))
        self.assertFalse(metadata['control_status']['active'])
