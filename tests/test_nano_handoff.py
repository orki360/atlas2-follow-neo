"""Regressions from guarded Dance 20261014_165341_4e2838, without flight IO."""
import copy
import unittest
import time
from unittest.mock import Mock
import numpy as np

from follow_neo.nano_guard import (apply_visual, ForwardAnchorHistory,
    appearance_unsupported, transport_visual_reason)
from follow_neo.nano_tracker import AppearanceCheck, NanoWorker
from follow_neo.controller import FollowController
from follow_neo.dance import dance_command, ZERO
from follow_neo.video import Frame
from test_nano_bridge import decision, visual
from test_visual_continuity import GOOD


def anchor(source, forward=.1, **changes):
    result=dict(time=source,identity=7,forward=forward,allowed=forward>0,box=[90,40,110,60])
    result.update(changes)
    return result


class HandoffTests(unittest.TestCase):
    def case(self, now=10., source=9.734, latest=9.828):
        old=anchor(source);new=anchor(latest)
        d=decision(now,measurement_time=latest,last_strong_measurement_time=latest,
                   measurement_age_ms=(now-latest)*1000,verified_forward_anchor=new,
                   verified_forward_history=[old,new])
        v=visual(now,semantic_time=source,box=[90,40,110,60],checks=dict(GOOD))
        return d,v

    def test_accepted_snapshot_crossing_250ms_hands_off_without_zero_pulse(self):
        # Recorded t=5.125: YOLO age 266 ms, visual age 47 ms, same source.
        d,v=self.case(latest=9.734)
        d.update(prediction_time=9.984,measurement_age_ms=250.)
        v.update(visual_time=9.953)
        before=copy.deepcopy(d)
        out=apply_visual(d,v,10.)
        values,reason=dance_command(out,10.,9.)
        self.assertEqual(out['nano_authority'],'bridge_forward')
        self.assertGreater(values[3],0);self.assertEqual(values[1:3],(0.,0.))
        self.assertEqual(out['nano_handoff']['bridge_reason'],'measurement_expired_at_send')
        self.assertTrue(reason.startswith('Nano bridge'))
        self.assertEqual(d,before)
        for key in ('measurement_time','last_strong_measurement_time','accepted','measurement_id'):
            self.assertEqual(out[key],d[key])

    def test_matching_fresh_measurement_keeps_normal_authority(self):
        d,v=self.case(source=9.828)
        out=apply_visual(d,v,10.)
        self.assertEqual(out['nano_authority'],'verified_yaw')
        self.assertGreater(dance_command(out,10.,9.)[0][3],0.)

    def test_delayed_nano_uses_older_compatible_forward_source(self):
        # Recorded t=4.172: newest YOLO 172 ms old, Nano source 266 ms old.
        d,v=self.case();out=apply_visual(d,v,10.)
        self.assertEqual(out['nano_authority'],'bridge_forward')
        self.assertEqual(out['nano_forward_source_time'],9.734)
        self.assertTrue(out['nano_handoff']['used_history'])
        self.assertGreater(dance_command(out,10.,9.)[0][3],0.)

    def test_new_measurement_cannot_extend_old_forward_deadline(self):
        d,v=self.case(now=10.085)
        out=apply_visual(d,v,10.085)
        self.assertEqual(out['nano_authority'],'bridge_yaw')
        self.assertEqual(dance_command(out,10.085,9.)[0][3],0.)

    def test_no_compatible_history_never_pairs_with_future_yolo(self):
        d,v=self.case();d['verified_forward_history']=[]
        out=apply_visual(d,v,10.)
        self.assertEqual(out['nano_authority'],'bridge_yaw')
        self.assertEqual(out['nano_handoff']['forward_reason'],'no_source_aligned_forward_anchor')

    def test_latest_stop_or_reduced_forward_has_priority_over_history(self):
        for speed in (0.,.02):
            d,v=self.case();d['verified_forward_anchor']=anchor(9.828,speed)
            out=apply_visual(d,v,10.)
            self.assertLessEqual(out['intent']['forward'],speed)
            if speed:self.assertGreater(out['intent']['forward'],0)

    def test_identity_weak_measurement_close_edge_and_class_conflicts_deny_forward(self):
        for changes in (
                dict(target_identity=dict(logical_target_id=8)),
                dict(last_strong_measurement_time=9.7),dict(spacing_close_guard_armed=True),
                dict(bbox_clipped=True),dict(spacing_phase='BRAKE'),dict(class_support={'expires_at':11.})):
            with self.subTest(changes=changes):
                d,v=self.case();d.update(changes)
                self.assertEqual(apply_visual(d,v,10.)['intent']['forward'],0.)

    def test_restart_requires_new_anchor_not_previous_dance_permission(self):
        d,v=self.case();out=apply_visual(d,v,10.)
        self.assertEqual(dance_command(out,10.,9.8)[0],ZERO)
        controller=FollowController()
        controller.forward_anchors.remember(anchor(9.9),10.)
        controller.last_forward_anchor=anchor(9.9)
        controller.restart_control()
        self.assertEqual(controller.forward_anchors.snapshot(),[])
        self.assertIsNone(controller.last_forward_anchor)

    def test_reapplication_does_not_double_decay_or_renew_source(self):
        d,v=self.case();d['prediction_time']=9.98
        first=apply_visual(d,v,10.)
        later=apply_visual(first,visual(10.02,semantic_time=9.734,
            box=[90,40,110,60],checks=dict(GOOD)),10.02)
        a=dance_command(first,10.02,9.)[0][3]
        b=dance_command(later,10.02,9.)[0][3]
        self.assertAlmostEqual(a,b)
        self.assertEqual(later['nano_forward_source_time'],9.734)
        self.assertLessEqual(b,dance_command(first,10.,9.)[0][3])

    def test_ack_wait_revokes_history_forward_on_new_stop(self):
        d,v=self.case();sent=apply_visual(d,v,10.)
        latest=dict(sent,verified_forward_anchor=anchor(9.85,0.))
        self.assertEqual(transport_visual_reason(sent,latest,10.01),'visual_forward_revoked')
        self.assertEqual(transport_visual_reason(sent,dict(sent,nano_authority='bridge_yaw'),10.01),
                         'visual_forward_revoked')
        self.assertEqual(transport_visual_reason(sent,sent,10.085),'visual_forward_expired')

    def test_semantic_expiry_never_becomes_new_authority(self):
        d,v=self.case(now=10.135)
        self.assertEqual(dance_command(apply_visual(d,v,10.135),10.135,9.)[0],ZERO)


class ForwardHistoryTests(unittest.TestCase):
    def test_same_source_ticks_replace_without_renewing_time(self):
        h=ForwardAnchorHistory()
        for i in range(20):h.remember(anchor(9.8,forward=.1-i*.001),10.+i*.001)
        self.assertEqual(len(h.snapshot()),1)
        self.assertEqual(h.snapshot()[0]['time'],9.8)
        h.prune(10.151);self.assertEqual(h.snapshot(),[])

    def test_zero_is_a_barrier_even_after_forward_resumes(self):
        h=ForwardAnchorHistory()
        h.remember(anchor(9.7),9.8);h.remember(anchor(9.8,0.),9.9)
        h.remember(anchor(9.9),10.)
        self.assertEqual([a['time'] for a in h.snapshot()],[9.8,9.9])
        d,v=HandoffTests().case(source=9.7,latest=9.9)
        d['verified_forward_history']=h.snapshot()
        self.assertEqual(apply_visual(d,v,10.)['intent']['forward'],0.)

    def test_identity_change_discards_previous_target_history(self):
        h=ForwardAnchorHistory();h.remember(anchor(9.8),9.9)
        h.remember(anchor(9.9,identity=8),10.)
        self.assertEqual([a['identity'] for a in h.snapshot()],[8])

    def test_history_is_bounded_and_snapshots_do_not_mutate(self):
        h=ForwardAnchorHistory()
        for i in range(30):h.remember(anchor(10.+i*.005),10.+i*.005)
        snapshot=h.snapshot();self.assertLessEqual(len(snapshot),8)
        h.remember(anchor(10.145,0.),10.15)
        self.assertTrue(snapshot[-1]['allowed'])


class BackgroundEvidenceTests(unittest.TestCase):
    def test_recorded_empty_wall_cues_revoke_nano_before_400ms(self):
        # Actual Nano source frames at 87.875 / 87.906 / 87.938 / 88.031 s.
        for corr,spread in ((.18635,4.37),(.11299,5.19),(.12510,3.90),(-.08948,5.)):
            checks=dict(GOOD,reference_std=25.447,patch_std=spread,
                        appearance_correlation=corr,flow_points=0)
            d,v=HandoffTests().case();v['checks']=checks
            out=apply_visual(d,v,10.)
            self.assertEqual(out['nano_block_reason'],'visual_appearance_unsupported')
            self.assertEqual(dance_command(out,10.,9.)[0],ZERO)

    def test_good_appearance_does_not_require_flow_on_hovering_drone(self):
        self.assertFalse(appearance_unsupported(dict(reference_std=25.,appearance_correlation=.5,flow_points=0)))

    def test_blur_can_be_supported_by_coherent_foreground_motion(self):
        checks=dict(reference_std=25.,appearance_correlation=.18,flow_points=8,
            background_points=20,foreground_flow_spread_px=1.,box_flow_error_px=2.,relative_motion_px=5.)
        self.assertFalse(appearance_unsupported(checks))
        for change in (dict(flow_points=0),dict(relative_motion_px=.5),dict(box_flow_error_px=15.)):
            self.assertTrue(appearance_unsupported(dict(checks,**change)))

    def test_uncertainty_pauses_control_without_destroying_visual_track(self):
        w=NanoWorker('.',mode='guarded',clock=lambda:10.)
        w.blocked=False;w.semantic_time=9.8
        w.observation=dict(visual_time=9.98,box=[90,40,110,60],
            checks=dict(reference_std=25.,appearance_correlation=.18,flow_points=0))
        result=w.snapshot()
        self.assertEqual(result['reason'],'visual_appearance_unsupported')
        self.assertTrue(result['blocked']);self.assertTrue(result['visual_active'])
        self.assertEqual(w.epoch,0);self.assertEqual(w.semantic_time,9.8)

    def test_soft_expiry_does_not_hide_an_independent_appearance_veto(self):
        d,v=HandoffTests().case(source=9.828)
        v.update(blocked=True,reason='visual_expired',checks=dict(GOOD,
            reference_std=25.,appearance_correlation=.18,flow_points=0))
        out=apply_visual(d,v,10.)
        self.assertEqual(out['nano_authority'],'blocked')
        self.assertEqual(out['nano_block_reason'],'visual_appearance_unsupported')

    def test_validator_exposes_uncertainty_in_worker_checks(self):
        rng=np.random.default_rng(4)
        image=rng.integers(0,256,(100,200,3),dtype=np.uint8)
        check=AppearanceCheck();check.anchor(Frame(1,9.8,image),dict(box=[90,40,110,60]))
        # Different textured pixels with no coherent flow are not visual proof.
        check.patch=Mock(return_value=rng.normal(128.,25.,(32,48)).astype(np.float32))
        check.flow=Mock(return_value=dict(flow_points=0))
        reason,metrics=check.update(Frame(2,9.84,image),[90,40,110,60])
        self.assertIsNone(reason)
        self.assertTrue(metrics['appearance_unsupported']);self.assertTrue(metrics['control_suspect'])


class HandoffTransportTests(unittest.TestCase):
    def test_cached_accepted_decision_hands_off_at_real_send_time(self):
        from follow_neo.manual import ManualControl
        from test_manual import Server, wait_for
        peer=Server();events=[]
        def provider():
            now=time.monotonic();source=now-.266;tick=now-.030
            d=decision(tick,measurement_time=source,last_strong_measurement_time=source,
                measurement_age_ms=236.,verified_forward_anchor=anchor(source),
                verified_forward_history=[anchor(source)])
            v=visual(now,semantic_time=source,visual_time=now-.050,
                     box=[90,40,110,60],checks=dict(GOOD))
            # Fresh at controller evaluation, stale by the transport's read.
            out=apply_visual(d,v,tick)
            assert out['nano_authority']=='verified_yaw'
            return out
        control=ManualControl('127.0.0.1',peer.connect,lambda e,d:events.append((e,d)),provider)
        self.addCleanup(lambda:(control.stop(),control.thread.join(2)))
        control.set_axis_limits(dict(yaw=.5,vertical=.1,roll=.1,forward=.065))
        control.update(set());control.enable();control.start()
        wait_for(lambda:control.enabled);control.start_dance()
        deadline=time.monotonic()+1.5
        while time.monotonic()<deadline:
            control.update(set());time.sleep(.01)
            if any(e=='flight_command' and d.get('nano_authority')=='bridge_forward'
                   and d['values'][3]>0 for e,d in events):break
        sent=[d for e,d in events if e=='flight_command' and d.get('nano_authority')=='bridge_forward'
              and d['values'][3]>0]
        self.assertTrue(sent)
        self.assertTrue(all(0<d['values'][3]<=.065*.35 for d in sent))
        self.assertTrue(all(d['nano_handoff']['bridge_reason']=='measurement_expired_at_send' for d in sent))
        self.assertTrue(all(d['values'][1:3]==[0.,0.] for d in sent))


if __name__=='__main__':unittest.main()
