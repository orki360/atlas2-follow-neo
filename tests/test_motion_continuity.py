"""Offline regressions for the 20261018 stop/start flight; no aircraft IO."""
import copy
import math
import time
import unittest
from dataclasses import asdict
from unittest.mock import Mock, patch
from follow_neo.approach import LateralEvidence
from follow_neo.centering import DampedTrackingController
from follow_neo.control_transport import ControlChannel, VISUAL_CHANGE
from follow_neo.dance import DanceSmoother, dance_command
from follow_neo.motion_continuity import pending_motion_status
from follow_neo.nano_guard import apply_visual, guard_reason
from follow_neo.recovery_search import RecoverySearch, HEADING_PAUSE_SECONDS
from follow_neo.types import Settings, Kinematics, settings_from_config
from test_nano_bridge import decision, visual
from test_visual_continuity import GOOD
import test_nano_handoff as handoff_cases


def heading(t, yaw=0., source='test'):
    return dict(time=t, yaw_deg=yaw, source=source)


class PendingPermissionTests(unittest.TestCase):
    def case(self):
        d=decision(10.,intent=dict(yaw=.1,vertical=0.,roll=0.,forward=.1),
            motion_settings=dict(nano_visual_max_age_seconds=.07))
        old=visual(10.,visual_time=9.94,deadline=10.04,checks=dict(GOOD))
        sent=apply_visual(d,old,10.)
        new=visual(10.04,semantic_time=9.88,visual_time=10.02,deadline=10.12,checks=dict(GOOD))
        latest=apply_visual(dict(d,prediction_time=10.04),new,10.04)
        return sent,latest

    def status(self,sent,latest,now=10.04,values=(.02,0.,0.,.0135)):
        return pending_motion_status(sent,latest,now,9.,values,(.5,.1,.1,.09))

    def test_current_same_identity_support_avoids_zero_without_mutating_sources(self):
        sent,latest=self.case();before=copy.deepcopy((sent,latest))
        self.assertEqual(guard_reason(sent['nano'],10.04),'visual_expired')
        status=self.status(sent,latest)
        self.assertIsNone(status['reason']);self.assertTrue(status['renewed'])
        self.assertEqual((sent,latest),before)

    def test_expired_worker_visual_deadline_can_use_current_support(self):
        sent,latest=self.case()
        self.assertTrue(self.status(sent,latest,10.06)['renewed'])

    def test_custom_short_deadline_cannot_be_renewed(self):
        sent,latest=self.case();sent['nano'].update(control_max_age_seconds=.1,visual_time=10.,deadline=10.01)
        status=self.status(sent,latest)
        self.assertEqual(status['reason'],'visual_deadline');self.assertFalse(status['renewed'])

    def test_newer_measurement_cannot_keep_an_old_decision_alive(self):
        sent,latest=self.case();latest['nano'].update(visual_time=10.2,deadline=10.3,semantic_time=10.1)
        self.assertIsNotNone(self.status(sent,latest,10.21)['reason'])

    def test_background_mini_reset_and_mode_switch_always_revoke(self):
        for change in (dict(blocked=True,reason='appearance_or_background_drift'),
                       dict(blocked=True,reason='competing_mini_detection'),dict(epoch=3),
                       dict(identity=8),dict(generation=2),dict(mode='shadow'),
                       dict(checks=dict(GOOD,control_suspect=True))):
            with self.subTest(change=change):
                sent,latest=self.case();latest['nano'].update(change)
                status=self.status(sent,latest)
                self.assertIsNotNone(status['reason']);self.assertFalse(status['renewed'])

    def test_stop_lower_cap_or_reversal_cannot_renew_pending_command(self):
        for intent in (dict(yaw=.1,vertical=0.,roll=0.,forward=0.),
                       dict(yaw=-.1,vertical=0.,roll=0.,forward=.1),
                       dict(yaw=.1,vertical=0.,roll=0.,forward=.01)):
            sent,latest=self.case();latest['base_intent']=intent
            self.assertEqual(self.status(sent,latest)['reason'],'live_setpoint_incompatible')

    def test_proximity_clipping_and_spacing_hold_revoke_renewal(self):
        for change in (dict(spacing_close_guard_armed=True),dict(bbox_clipped=True),
                       dict(spacing_phase='VISUAL_HOLD')):
            sent,latest=self.case();latest.update(change)
            self.assertFalse(self.status(sent,latest)['renewed'])

    def test_forward_bridge_original_deadline_is_never_renewed(self):
        d,v=handoff_cases.HandoffTests().case();sent=apply_visual(d,v,10.)
        latest=copy.deepcopy(sent);latest['nano'].update(visual_time=10.08,deadline=10.18)
        self.assertEqual(self.status(sent,latest,10.085)['reason'],'visual_forward_expired')

    def test_semantic_expiry_and_stale_video_are_not_visual_refreshes(self):
        sent,latest=self.case();sent['nano']['semantic_time']=9.5
        self.assertEqual(self.status(sent,latest)['reason'],'semantic_expired')
        sent,latest=self.case();latest['stale']=True
        self.assertFalse(self.status(sent,latest)['renewed'])

    def test_fresh_measurement_keeps_damped_yaw_including_braking(self):
        for yaw in (0.,-.1,.30):
            d=decision(intent=dict(yaw=yaw,vertical=0.,roll=0.,forward=.1))
            out=apply_visual(d,visual(checks=dict(GOOD)),10.)
            self.assertEqual(out['nano_authority'],'verified_yaw')
            self.assertEqual(out['intent']['yaw'],yaw)

    def test_new_real_neo_can_take_over_a_still_compatible_setpoint(self):
        sent,latest=self.case()
        sent['measurement_time']=sent['last_strong_measurement_time']=9.8
        sent['nano']['semantic_time']=9.8
        latest['measurement_time']=latest['last_strong_measurement_time']=10.
        latest['measurement_id']=13
        latest['nano'].update(semantic_time=10.,visual_time=10.07,deadline=10.17)
        status=self.status(sent,latest,10.08)
        self.assertTrue(status['renewed']);self.assertEqual(status['support_source'],'new_real_neo')
        # Nano on its own cannot fabricate this fresh semantic handoff.
        latest['accepted']=False
        self.assertFalse(self.status(sent,latest,10.08)['renewed'])

    def test_repeated_refresh_does_not_reset_smoother_or_extend_source(self):
        sent,latest=self.case();s=DanceSmoother();forward=[]
        for i in range(4):
            t=10.+i*.04
            latest['prediction_time']=t
            latest['nano'].update(visual_time=t-.01,deadline=t+.09)
            status=self.status(sent,latest,t,values=(0.,0.,0.,.01))
            self.assertIsNone(status['reason'])
            forward.append(s.update((.1,0.,0.,.33),t,True)[3])
        self.assertTrue(all(a<=b for a,b in zip(forward,forward[1:])))
        self.assertGreater(forward[-1],forward[0])
        self.assertEqual(sent['measurement_time'],9.88)


class DelayedAckTests(unittest.TestCase):
    def exchange(self, veto=False, reply=True):
        sent,latest=PendingPermissionTests().case()
        now=[10.];events=[]
        class Peer:
            def __init__(self):self.wire=[];self.replied=False
            def setsockopt(self,*args):pass
            def settimeout(self,value):self.timeout=value
            def sendall(self,data):self.wire.append(data)
            def recv(self,n):
                import socket
                now[0]+=self.timeout
                if reply and now[0]>=10.12 and not self.replied:
                    self.replied=True
                    return b'success\r\n'*len(self.wire)
                raise socket.timeout()
        peer=Peer();statuses=[]
        def guard(keys):
            if keys is None:return None
            latest['prediction_time']=now[0]
            latest['nano'].update(visual_time=now[0]-.01,deadline=now[0]+.09)
            if veto and now[0]>=10.06:
                latest['nano'].update(blocked=True,reason='appearance_or_background_drift')
            s=PendingPermissionTests().status(sent,latest,now[0]);statuses.append(s)
            return VISUAL_CHANGE if s['reason'] else None
        with patch('follow_neo.control_transport.time.monotonic',side_effect=lambda:now[0]):
            c=ControlChannel(peer,guard,lambda e,d:events.append((e,d)))
            ok=c.command('rc 0.0200 0 0 0.0135',motion_keys=frozenset())
        return c,peer,ok,statuses,events

    def test_120ms_ack_with_new_nano_frames_keeps_one_command(self):
        c,peer,ok,statuses,events=self.exchange()
        self.assertTrue(ok);self.assertEqual(len(peer.wire),1)
        self.assertTrue(any(s['renewed'] for s in statuses));self.assertFalse(c.pending)

    def test_background_during_ack_sends_and_acknowledges_separate_neutral(self):
        c,peer,ok,_,events=self.exchange(veto=True)
        self.assertFalse(ok);self.assertEqual(peer.wire[-1],b'rc 0 0 0 0\r\n')
        self.assertTrue(c.neutral_acknowledged);self.assertFalse(c.pending)

    def test_refresh_never_extends_250ms_ack_deadline(self):
        c,peer,ok,_,events=self.exchange(reply=False)
        self.assertFalse(ok);self.assertTrue(c.paused)
        self.assertIn('0.25s',c.pause_reason)
        self.assertEqual(peer.wire[-1],b'rc 0 0 0 0\r\n')


class ManualHandoffTests(unittest.TestCase):
    def test_live_manual_path_logs_renewal_and_keeps_forward_between_10hz_results(self):
        from follow_neo.manual import ManualControl
        from test_manual import Server,wait_for
        events=[]
        def delay(cmd):
            if cmd.startswith('rc ') and any(float(v) for v in cmd.split()[1:]):time.sleep(.11)
        peer=Server(before_reply=delay);origin=time.monotonic()
        def provider():
            now=time.monotonic();frame=math.floor((now-origin)/.1)
            source=origin+frame*.1-.12
            d=decision(now,measurement_time=source,last_strong_measurement_time=source,
                measurement_age_ms=(now-source)*1000,measurement_id=frame,
                motion_settings=dict(nano_visual_max_age_seconds=.07),
                intent=dict(yaw=.1,vertical=0.,roll=0.,forward=.1))
            v=visual(now,semantic_time=source,visual_time=now-.04,deadline=now+.06,checks=dict(GOOD))
            return apply_visual(d,v,now)
        control=ManualControl('127.0.0.1',peer.connect,lambda e,d:events.append((e,d)),provider)
        self.addCleanup(lambda:(control.stop(),control.thread.join(3)))
        control.set_axis_limits(dict(yaw=.5,vertical=.1,roll=.1,forward=.09))
        control.update(set());control.enable();control.start()
        wait_for(lambda:control.enabled);control.start_dance()
        until=time.monotonic()+1.2
        while time.monotonic()<until:
            control.update(set());time.sleep(.01)
        moving=[d for e,d in events if e=='flight_command' and d['values'][3]>0]
        self.assertGreaterEqual(len(moving),4)
        self.assertTrue(any(e=='control_pending_motion' and d['renewed'] for e,d in events))
        self.assertFalse(any(e=='control_motion_neutralizing' for e,d in events))
        self.assertTrue(all(0<d['values'][3]<=.09 for d in moving))
        self.assertTrue(all(d['android_stick_units'][3]>0 for d in moving))


class LateralHistoryTests(unittest.TestCase):
    def obs(self,i,**changes):
        o=dict(time=10.+i*.1,frame_id=i,point=(300.+i*10,240.),width=960,
               camera_valid=True,warp=[[1.,0.,0.],[0.,1.,0.]],identity_key=(1,7))
        o.update(changes);return o

    def test_10hz_measurements_survive_intermediate_output_holds(self):
        e=LateralEvidence();last=0.
        for i in range(7):
            o=self.obs(i);last=e.update(o,o['time']+.12,True,'')
            if i==6:break
            self.assertEqual(e.update(o,o['time']+.16,False,'waiting_measurements'),0.)
        self.assertGreater(last,0.);self.assertEqual(e.diagnostics['samples'],4)
        self.assertTrue(e.diagnostics['evidence_confirmed'])

    def test_new_frame_can_build_history_while_turn_output_is_blocked(self):
        e=LateralEvidence()
        for i in range(6):
            o=self.obs(i)
            self.assertEqual(e.update(o,o['time']+.12,False,'align_yaw_first'),0.)
        self.assertTrue(e.diagnostics['history_retained'])
        self.assertGreater(e.update(o,o['time']+.13,True,''),0.)

    def test_stale_measurements_and_no_observation_never_move(self):
        e=LateralEvidence()
        for i in range(6):
            o=self.obs(i);e.update(o,o['time']+.12,True,'')
        self.assertEqual(e.update(o,o['time']+.16,True,''),0.)
        self.assertEqual(e.update(None,o['time']+.18,True,''),0.)
        e.update(None,o['time']+.36,True,'')
        self.assertEqual(len(e.steps),0);self.assertIsNone(e.previous)

    def test_identity_camera_failure_and_skipped_frames_break_evidence(self):
        for changes in (dict(identity_key=(1,8)),dict(identity_key=(2,7)),
                        dict(camera_valid=False),dict(frame_id=20)):
            e=LateralEvidence()
            for i in range(6):
                o=self.obs(i);e.update(o,o['time']+.12,True,'')
            o=self.obs(6,**changes)
            self.assertEqual(e.update(o,o['time']+.12,True,''),0.)
            self.assertEqual(len(e.steps),0)

    def test_camera_pan_does_not_become_target_motion(self):
        e=LateralEvidence()
        for i in range(10):
            o=self.obs(i,warp=[[1.,0.,10.],[0.,1.,0.]])
            self.assertEqual(e.update(o,o['time']+.12,True,''),0.)


class CoordinatedTurnTests(unittest.TestCase):
    def calc(self,ex=.60,rate=30.,verified=True,settings=None):
        c=DampedTrackingController();c.heading_rate.update=Mock(return_value=rate)
        k=Kinematics(True,100+100*ex,50,20,10)
        c.compute(k,200,100,.12,False,10.,settings or Settings(forward_yaw_demand_limit=.30),
            heading(10.),9.88,ex,continuity_verified=verified)
        return c

    def test_verified_turn_above_demand_threshold_retains_reduced_forward(self):
        c=self.calc()
        self.assertGreater(c.diagnostics['yaw_normalized'],.30)
        self.assertGreater(c.forward_factor,0.);self.assertLess(c.forward_factor,1.)
        self.assertEqual(c.diagnostics['yaw_demand_policy'],'slowdown')

    def test_unverified_or_missing_rate_and_explicit_legacy_mode_keep_ceiling(self):
        for kw in (dict(verified=False),dict(rate=None),dict(settings=Settings(
                coordinated_turns=False,forward_yaw_demand_limit=.30))):
            self.assertEqual(self.calc(**kw).forward_factor,0.)

    def test_window_and_actual_rate_remain_hard_limits(self):
        self.assertEqual(self.calc(ex=.9).forward_factor,0.)
        self.assertEqual(self.calc(rate=66.).forward_factor,0.)

    def test_config_boolean_roundtrips_without_overwriting_user_limits(self):
        for enabled in (True,False):
            s=Settings(coordinated_turns=enabled,forward_yaw_demand_limit=.30)
            self.assertEqual(settings_from_config(dict(schema_version=8,settings=asdict(s))),s)
        with self.assertRaises(ValueError):Settings(coordinated_turns='false').validate()


class HeadingPauseTests(unittest.TestCase):
    def case(self):
        s=Settings(search_yaw_degrees=90);r=RecoverySearch()
        r.start_test(10.,s,heading(10.),'RIGHT')
        return r,s

    def update(self,r,s,t,h):
        return r.update(t,True,False,'APPROACH',False,s,h)

    def test_transient_missing_heading_resumes_same_arc_and_deadline(self):
        r,s=self.case();until=r.until
        for t in (10.2,10.4,10.6):
            d=self.update(r,s,t,None)
            self.assertFalse(d['active']);self.assertFalse(d['consumed']);self.assertEqual(d['scan_yaw'],0.)
        d=self.update(r,s,10.65,heading(10.65,8.))
        self.assertTrue(d['active']);self.assertEqual(d['until'],until)
        self.assertEqual(d['scan_offset_degrees'],8.);self.assertEqual(d['heading_resumes'],1)

    def test_long_heading_gap_is_consumed_even_if_next_sample_is_fresh(self):
        r,s=self.case();self.update(r,s,10.2,None)
        d=self.update(r,s,10.2+HEADING_PAUSE_SECONDS+.01,heading(11.01))
        self.assertFalse(d['active']);self.assertTrue(d['consumed'])
        self.assertEqual(d['reason'],'heading_unavailable')

    def test_source_switch_discontinuity_and_original_timeout_cannot_resume(self):
        for h,reason in ((heading(10.6,0.,'other'),'heading_source_changed'),
                         (heading(10.6,110.),'heading_discontinuity')):
            r,s=self.case();self.update(r,s,10.2,None)
            d=self.update(r,s,10.6,h)
            self.assertEqual(d['reason'],reason);self.assertFalse(d['active'])
        r,s=self.case();self.update(r,s,r.until-.1,None)
        self.assertEqual(self.update(r,s,r.until+.01,heading(r.until+.01))['reason'],'search_timeout')


if __name__=='__main__':unittest.main()
