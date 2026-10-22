"""Recorded failure modes, using synthetic evidence and local fake transports."""
import copy
import json
import threading
import time
import unittest
from unittest.mock import Mock,patch
from follow_neo.shared.types import Settings,Box,Detection,Kinematics
from follow_neo.control.recovery_search import RecoverySearch,scan_command
from follow_neo.control.nano_guard import apply_visual,transport_visual_reason
from follow_neo.control.dance import dance_command,ZERO
from follow_neo.transport.manual import ManualControl
from follow_neo.runtime.session import FollowSession
from follow_neo.control.controller import FollowController
from follow_neo.capture.video import VideoReceiver
from follow_neo.capture.video_diagnostics import VideoDiagnostics
from test_nano_bridge import decision,visual
from test_manual import Server,wait_for


def heading(t,yaw=0.):return dict(time=t,yaw_deg=yaw,source='offline')


class SearchRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.s=Settings(search_yaw_degrees=80);self.r=RecoverySearch()
        self.r.observe(True,False,True,Box(270,200,330,240),Kinematics(),640,480,10.,1,self.s)
        self.neo=Detection(Box(430,200,510,240),.94,0,'neo')

    def feed(self,t,fid,ds=None):
        self.r.observe_candidate([self.neo] if ds is None else ds,t,fid,(640,480),self.s)
        return self.r.update(t+.12,True,False,'APPROACH',False,self.s,heading(t+.12),candidate_pending=True)

    def command(self,s,t,**changes):
        d=decision(t,state='DIRECTIONAL_SEARCH',accepted=False,edge_search=s,measurement_time=10.,
            candidate_pending=True,tracking=dict(acquisition_state='reacquiring'),**changes)
        out=apply_visual(d,visual(t,blocked=True,reason='semantic_expired'),t)
        return dance_command(out,t,9.)[0]

    def test_two_candidates_in_wait_state_turn_but_never_confirm_or_translate(self):
        a=self.feed(14.,2);self.assertFalse(a['active'])
        b=self.feed(14.1,3)
        self.assertTrue(b['candidate_only']);self.assertTrue(b['candidate_guidance'])
        self.assertFalse(b['yaw_full_scale']);self.assertEqual(b['source_time'],10.)
        cmd=self.command(b,14.22)
        self.assertGreater(cmd[0],0.);self.assertLessEqual(cmd[0],.2);self.assertEqual(cmd[1:],ZERO[1:])

    def test_hover_after_timeout_can_take_one_bounded_look(self):
        self.r.end('search_timeout')
        self.feed(16.,2);s=self.feed(16.1,3)
        self.assertTrue(s['candidate_guidance']);until=s['until']
        for i in range(4,14):last=self.feed(16.+i*.1,i)
        self.assertFalse(last['active']);self.assertEqual(self.r.until,until)
        self.assertTrue(self.r.cue_recovery_used)

    def test_duplicate_competing_mini_missing_and_old_identity_never_start_cue(self):
        self.feed(14.,2);self.assertFalse(self.feed(14.,2)['candidate_guidance'])
        for ds in ([],[self.neo,Detection(self.neo.box,.8,1,'MINI PRO4')],
                   [self.neo,Detection(Box(30,200,90,240),.9,0,'neo')]):
            self.setUp();self.feed(14.,2)
            self.assertFalse(self.feed(14.1,3,ds)['candidate_guidance'])
        self.setUp();self.feed(23.,2);self.assertFalse(self.feed(23.1,3)['active'])

    def test_cue_stops_on_missing_frame_or_stale_heading_and_cannot_be_replayed(self):
        self.feed(14.,2);s=self.feed(14.1,3)
        # The send gate catches expiry even if the planner did not tick again.
        d=decision(14.4,state='DIRECTIONAL_SEARCH',accepted=False,edge_search=s)
        self.assertEqual(scan_command(d,14.4,9.),0.)
        self.assertFalse(self.feed(14.2,4,[])['active'])
        self.setUp();self.feed(14.,2);self.feed(14.1,3)
        s=self.r.update(14.25,True,False,'APPROACH',False,self.s,None,candidate_pending=True)
        self.assertFalse(s['active'])

    def test_operator_stop_video_loss_close_and_no_prior_identity_forbid_cue(self):
        for reason in ('control_unavailable','video_or_result_unavailable','heading_discontinuity'):
            self.setUp();self.r.end(reason);self.feed(14.,2)
            self.assertFalse(self.feed(14.1,3)['active'])
        self.setUp();self.r.reset();self.feed(14.,2)
        self.assertFalse(self.feed(14.1,3)['active'])
        self.setUp();self.feed(14.,2)
        self.r.observe_candidate([self.neo],14.1,3,(640,480),self.s)
        s=self.r.update(14.22,True,False,'APPROACH',True,self.s,heading(14.22),candidate_pending=True)
        self.assertFalse(s['active'])

    def test_boost_handoff_keeps_original_budget_and_fresh_heading_gate(self):
        r=self.r;r.start_test(10.,self.s,heading(10.),'RIGHT',boost=True);until=r.until
        s=r.update(12.,True,False,'APPROACH',False,self.s,heading(12.,20))
        self.assertEqual(s['phase'],'SCAN');self.assertEqual(s['boost_exit_reason'],'boost_timeout')
        self.assertEqual(s['until'],until);self.assertFalse(s['consumed'])
        s=r.update(12.05,True,False,'APPROACH',False,self.s,heading(12.05,21))
        self.assertGreater(s['scan_yaw'],0);self.assertLessEqual(s['scan_yaw'],.051)
        s=r.update(until+.01,True,False,'APPROACH',False,self.s,heading(until+.01,22))
        self.assertFalse(s['active']);self.assertEqual(s['reason'],'search_timeout')

    def test_fast_crossing_uses_bounded_age_compensation_for_yaw_only(self):
        self.feed(14.,2,[Detection(Box(250,200,330,240),.94,0,'neo')])
        s=self.feed(14.1,3,[Detection(Box(310,200,390,240),.94,0,'neo')])
        motion=s['search_motion']
        self.assertGreater(motion['candidate_advance_px'],0.)
        self.assertLessEqual(motion['candidate_advance_px'],640*.2)
        self.assertGreater(motion['candidate_error_x'],s['candidate_cue']['error_x'])
        self.assertEqual(self.command(s,14.22)[1:],ZERO[1:])

    def test_controller_and_send_gate_turn_before_botsort_finishes_reacquisition(self):
        c=FollowController('botsort',0,'neo');s=Settings(search_yaw_degrees=80)
        for i in range(8):
            source=10.+i*.1
            c.observe([Detection(Box(100,300,300,360),.94,0,'neo')],source,source+.12,i,(1920,1080),s)
            c.tick(source+.12,1920,1080,source+.10,s,heading(source+.12))
        self.assertTrue(c.tracker.confirmed);original=c.tracker.measurement_time
        c.edge_search.end('search_timeout')
        c.observe([],14.,14.12,8,(1920,1080),s,allow_search_cues=True)
        c.tick(14.12,1920,1080,14.10,s,heading(14.12))
        for i,x in enumerate((600,800)):
            source=14.2+i*.1;now=source+.12
            c.observe([Detection(Box(x,300,x+240,360),.94,0,'neo')],source,now,9+i,(1920,1080),s,allow_search_cues=True)
            d=c.tick(now,1920,1080,now-.02,s,heading(now))
        self.assertFalse(d['accepted']);self.assertEqual(c.tracker.measurement_time,original)
        self.assertTrue(d['edge_search']['candidate_guidance']);self.assertEqual(d['state'],'DIRECTIONAL_SEARCH')
        out=apply_visual(d,visual(now,blocked=True,reason='waiting_strong_neo'),now)
        cmd,_=dance_command(out,now,9.)
        self.assertGreater(cmd[0],0.);self.assertEqual(cmd[1:],ZERO[1:])


class SoftNanoEvidenceTests(unittest.TestCase):
    def inputs(self):
        box=[110.,40.,140.,60.]
        d=decision(generation=1,raw_detection_time=9.88,selected_measurement_box=box,
            raw_detections=[dict(box=box,confidence=.94,class_id=0,class_name='neo')],
            detector_acquisition_threshold=.65)
        v=visual(checks=dict(geometry_suspect=False,background_like_motion=False,visible_fraction=1.,
            texture_loss_streak=0,box_flow_error_px=0,reference_std=30.,appearance_correlation=.05,
            control_suspect=True,appearance_unsupported=True))
        return d,v

    def test_soft_doubt_cannot_veto_matching_fresh_real_yolo(self):
        d,v=self.inputs();before=copy.deepcopy(d)
        out=apply_visual(d,v,10.)
        self.assertEqual(out['nano_authority'],'yolo_only')
        self.assertEqual(out['nano_soft_veto_overridden'],'visual_appearance_unsupported')
        self.assertGreater(dance_command(out,10.,9.)[0][3],0)
        self.assertEqual(d,before);self.assertEqual(out['measurement_time'],9.88)

    def test_hard_veto_is_not_hidden_by_soft_appearance_metrics(self):
        for reason in ('background','competing_mini_detection','target_leaving_frame',
                       'target_texture_lost','yolo_visual_disagreement','invalid_box'):
            d,v=self.inputs();v.update(blocked=True,reason=reason)
            out=apply_visual(d,v,10.)
            self.assertEqual(dance_command(out,10.,9.)[0],ZERO,reason)
            self.assertNotIn('nano_soft_veto_overridden',out)

    def test_competing_detection_geometry_texture_identity_generation_block(self):
        for kind in ('mini','two_neo','box','identity','generation','geometry','texture','background','flow','weak','raw_time'):
            d,v=self.inputs()
            if kind in ('mini','two_neo'):
                d['raw_detections'].append(dict(d['raw_detections'][0],class_id=1 if kind=='mini' else 0))
            elif kind=='box':v['box']=[0,0,20,20]
            elif kind=='identity':v['identity']=8
            elif kind=='generation':v['generation']=2
            elif kind=='geometry':v['checks']['geometry_suspect']=True
            elif kind=='texture':v['checks']['texture_loss_streak']=1
            elif kind=='background':v['checks']['background_like_motion']=True
            elif kind=='flow':v['checks']['box_flow_error_px']=15
            elif kind=='weak':d['raw_detections'][0]['confidence']=.70
            else:d['raw_detection_time']=9.5
            self.assertEqual(dance_command(apply_visual(d,v,10.),10.,9.)[0],ZERO,kind)

    def test_no_fresh_yolo_means_no_soft_override_or_deadline_extension(self):
        for changes in (dict(accepted=False,state='COAST'),dict(stale=True),dict(class_support={'history':True}),
                        dict(measurement_time=9.7,last_strong_measurement_time=9.7,raw_detection_time=9.7)):
            d,v=self.inputs();d.update(changes)
            self.assertEqual(dance_command(apply_visual(d,v,10.),10.,9.)[0],ZERO)
        d,v=self.inputs();out=apply_visual(d,v,10.)
        self.assertEqual(dance_command(apply_visual(out,v,10.14),10.14,9.)[0],ZERO)

    def test_new_hard_conflict_revokes_pending_independent_command(self):
        d,v=self.inputs();sent=apply_visual(d,v,10.)
        latest=apply_visual(d,dict(v,blocked=True,reason='competing_mini_detection'),10.01)
        self.assertEqual(transport_visual_reason(sent,latest,10.01),'visual_control_blocked')


class PauseContinuityTests(unittest.TestCase):
    def test_pause_preserves_only_short_known_dance_ack_context(self):
        events=[];c=ManualControl('offline.invalid',on_event=lambda e,d:events.append((e,d)))
        c.enabled=c.wanted=True;c.mode='DANCE';c.update(set())
        c.pause_control('Control acknowledgement delayed beyond 0.25s',automatic_recovery=True)
        self.assertTrue(events[0][1]['preserve_tracking']);self.assertFalse(c.control_snapshot()['active'])
        self.assertTrue(c.control_snapshot()['continuity_pause'])
        with patch('follow_neo.transport.manual.time.monotonic',return_value=c.pause_started+1.01):
            self.assertFalse(c.control_snapshot()['continuity_pause'])
        c.pause_control('window lost focus')
        self.assertFalse(c.control_snapshot()['continuity_pause'])

    def test_session_soft_pause_keeps_nano_and_full_stop_resets(self):
        s=FollowSession.__new__(FollowSession);s.log=Mock();s.control_wake=Mock();s.generation=3
        s.search_test_log=None;s.restart_control=Mock()
        for e in ('control_movement_paused','control_movement_resumed'):
            s.record_control(e,dict(preserve_tracking=True))
        s.restart_control.assert_not_called()
        s.record_control('control_released',dict(preserve_tracking=True))
        s.restart_control.assert_called_once()

    def test_resume_requires_new_video_and_real_measurement_even_if_nano_is_fresh(self):
        d=decision(10.,video_time=9.99)
        self.assertEqual(dance_command(d,10.,9.,resume_after=9.9)[0],ZERO)
        d.update(measurement_time=9.95,last_strong_measurement_time=9.95,measurement_age_ms=50.)
        self.assertGreater(dance_command(d,10.,9.,resume_after=9.9)[0][3],0)
        d['video_time']=9.89
        self.assertEqual(dance_command(d,10.,9.,resume_after=9.9)[0],ZERO)

    def test_preserved_search_needs_new_video_but_not_new_target(self):
        s=Settings();r=RecoverySearch();r.start_test(9.5,s,heading(9.5),'RIGHT')
        search=r.update(10.,True,False,'APPROACH',False,s,heading(10.))
        d=decision(10.,state='DIRECTIONAL_SEARCH',accepted=False,edge_search=search,video_time=9.99,
                   measurement_time=9.5)
        self.assertGreater(dance_command(d,10.,9.,resume_after=9.9)[0][0],0)
        d['video_time']=9.8
        self.assertEqual(dance_command(d,10.,9.,resume_after=9.9)[0],ZERO)

    def test_real_ack_pause_retains_dance_epoch_and_waits_for_acknowledged_neutral(self):
        armed=threading.Event();blocked=threading.Event();release=threading.Event();events=[]
        def reply(cmd):
            if armed.is_set() and cmd.startswith('rc ') and not blocked.is_set():
                blocked.set();release.wait(2)
        peer=Server(before_reply=reply)
        c=ManualControl('127.0.0.1',peer.connect,lambda e,d:events.append((e,d)))
        self.addCleanup(lambda:(release.set(),c.stop(),c.thread.join(3)))
        c.update(set());c.enable();c.start();wait_for(lambda:c.enabled)
        c.start_dance();started=c.dance_started;armed.set();self.assertTrue(blocked.wait(.3))
        deadline=time.monotonic()+.30
        while time.monotonic()<deadline:c.update(set());time.sleep(.01)
        self.assertTrue(c.communication_paused);self.assertFalse(c.control_snapshot()['active'])
        release.set()
        wait_for(lambda:any(e=='control_movement_resumed' for e,d in events))
        result=next(d for e,d in events if e=='control_movement_resumed')
        self.assertTrue(result['preserve_tracking']);self.assertEqual(c.dance_started,started)
        self.assertGreater(c.resume_measurement_after,started);self.assertEqual(peer.commands.count('enable'),1)


class VideoHealthTests(unittest.TestCase):
    def test_receive_gap_and_stage_cpu_are_bounded_serializable_and_reset(self):
        d=VideoDiagnostics();d.connected(10.);d.received(5,10.);d.received(7,10.6)
        with d.measure('parse'):pass
        d.count('receive_timeouts');d.decoded(True,10.7)
        s=d.snapshot(reset=True);json.dumps(s,allow_nan=False)
        self.assertEqual(s['counters']['bytes'],12);self.assertAlmostEqual(s['counters']['max_receive_gap_ms'],600.)
        self.assertEqual(s['last_keyframe_at'],10.7);self.assertEqual(s['stages']['parse']['count'],1)
        self.assertEqual(d.snapshot()['counters'],{});self.assertEqual(d.snapshot()['last_keyframe_at'],10.7)
        d.connected(11.);self.assertIsNone(d.snapshot()['last_keyframe_at'])

    def test_monitor_can_see_blocked_decode_without_touching_network(self):
        events=[];v=VideoReceiver('offline.invalid',on_event=lambda e,d:events.append((e,d)))
        with v.diagnostics.measure('decode'):
            v._health('periodic')
        data=events[-1][1]
        self.assertEqual(data['pipeline']['stage'],'decode');self.assertEqual(data['total_bytes'],0)
        self.assertIsNone(v.sock);json.dumps(data,allow_nan=False)

    def test_error_reports_last_stage_even_without_frames_or_inference(self):
        events=[];v=VideoReceiver('offline.invalid',on_event=lambda e,d:events.append((e,d)))
        v.diagnostics.received(100,time.monotonic())
        v._invalidate_stream(TimeoutError('No decoded frame'),retry_seconds=.5)
        health=next(d for e,d in events if e=='video_health')
        self.assertEqual(health['reason'],'stream_error');self.assertEqual(health['state'],'RECONNECTING')
        self.assertEqual(health['pipeline']['counters']['bytes'],100)


if __name__=='__main__':unittest.main()
