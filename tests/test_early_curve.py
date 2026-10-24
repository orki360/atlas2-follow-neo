"""Offline early-turn and bounded continuation tests; no aircraft connection."""
import copy
import unittest
from unittest.mock import Mock,patch

from follow_neo.control.curve import EarlyCurve,CurveContinuation,verified_curve
from follow_neo.control.centering import DampedTrackingController
from follow_neo.control.controller import FollowController
from follow_neo.control.dance import dance_command,ZERO
from follow_neo.control.nano_guard import apply_visual,transport_visual_reason
from follow_neo.control.motion_continuity import pending_motion_status
from follow_neo.shared.types import Settings,Kinematics,Box,Detection
from follow_neo.diagnostics import activity
import test_tracking_recovery8 as previous
from test_tracking_recovery9 import heading,vector_case


def curve_case():
    d,v=previous.IndependentControlTests().inputs()
    v['checks'].update(control_suspect=False,appearance_correlation=.8,appearance_unsupported=False)
    d.update(video_time=9.99,heading=heading(10.),state='TRACK',track_box=d['selected_measurement_box'],
        centering_control=dict(early_curve=dict(active=True,direction=1)),
        motion_settings=dict(coordinated_turns=True,forward_window_width=.85,forward_window_height=.8))
    d['intent'].update(yaw=.14,forward=.09,roll=0.)
    m=CurveContinuation();m.remember(d,v,10.,True)
    d.update(curve_anchor=m.snapshot(),accepted=False,state='COAST',video_time=10.16,
             prediction_time=10.18,heading=heading(10.18),raw_detection_time=9.98,
             detection_summary=dict(target_eligible=0))
    v.update(visual_time=10.16,semantic_time=9.88,deadline=10.26)
    return d,v,m


class EarlyCurveTests(unittest.TestCase):
    def run_errors(self,errors,**changes):
        e=EarlyCurve();result=None
        for i,x in enumerate(errors):
            values=dict(source=10+i*.1,error=x,now=10+i*.1+.12,identity=(1,7),measured=True,allowed=True)
            values.update(changes);result=e.update(**values)
        return e,result

    def test_consistent_outward_motion_leads_both_directions(self):
        for sign in (-1,1):
            _,r=self.run_errors([sign*x for x in (.01,.03,.05)])
            self.assertTrue(r['active']);self.assertEqual(r['direction'],sign)
            self.assertAlmostEqual(r['lead'],sign*.024)
            self.assertLessEqual(abs(r['lead']),.10)

    def test_static_jitter_inward_and_reversal_do_not_get_lead(self):
        for errors in ((.15,.15,.15),(.15,.16,.15),(.20,.17,.14),(.02,.04,-.04)):
            _,r=self.run_errors(errors);self.assertFalse(r['active'],errors)
        e,r=self.run_errors((.1,.14,.18));self.assertTrue(r['active'])
        self.assertFalse(e.update(10.3,.12,10.42,(1,7),True,True)['active'])

    def test_repeated_frames_do_not_supply_votes_or_renew_expiry(self):
        e=EarlyCurve()
        for t in (10.1,10.12,10.15):self.assertFalse(e.update(10.,.1,t,(1,7),True,True)['active'])
        self.assertEqual(len(e.samples),1)
        self.assertFalse(e.update(10.,.2,10.26,(1,7),True,True)['active'])

    def test_discontinuity_unknown_identity_and_missing_proof_clear_lead(self):
        for args in ((10.3,.2,10.42,(2,7),True,True),
                     (10.6,.2,10.72,(1,7),True,True),
                     (10.3,.2,10.42,None,True,True),
                     (10.3,.2,10.42,(1,7),False,True),
                     (10.3,.2,10.42,(1,7),True,False)):
            e,_=self.run_errors((.1,.14,.18))
            self.assertFalse(e.update(*args)['active'])

    def test_controller_turns_before_old_deadzone_without_increasing_forward(self):
        outputs=[]
        for enabled in (False,True):
            c=DampedTrackingController();c.heading_rate.update=Mock(return_value=0.)
            s=Settings(coordinated_turns=enabled)
            for i,x in enumerate((.01,.03,.05)):
                t=10+i*.1
                cmd,_,_=c.compute(Kinematics(True,100+x*100,50,20,10),200,100,.12,False,t+.12,s,
                    heading(t+.12),t,x,dict(identity_key=(1,7)),current_visual=True,continuity_verified=True)
            outputs.append((cmd,c))
        self.assertEqual(outputs[0][0].yaw,0.)
        self.assertGreater(outputs[1][0].yaw,0.)
        self.assertLessEqual(outputs[1][1].forward_factor,outputs[0][1].forward_factor)

    def test_early_lead_never_bypasses_image_or_rate_forward_limits(self):
        for errors,rate in (((.80,.84,.88),0.),((.1,.14,.18),70.)):
            c=DampedTrackingController();c.heading_rate.update=Mock(return_value=rate)
            for i,x in enumerate(errors):
                t=10+i*.1
                cmd,_,_=c.compute(Kinematics(True,100+x*100,50,20,10),200,100,.12,False,t+.12,
                    Settings(),heading(t+.12),t,x,dict(identity_key=(1,7)),True,True)
            self.assertEqual(c.forward_factor,0.);self.assertLessEqual(abs(cmd.yaw),.55*Settings().yaw_limit)


class VerifiedCurveTests(unittest.TestCase):
    def test_verified_gap_combines_yaw_forward_without_side_or_altitude(self):
        d,v,m=curve_case();self.assertIsNotNone(m.snapshot())
        out=apply_visual(d,v,10.18);self.assertEqual(out['nano_authority'],'curve_bridge')
        cmd,reason=dance_command(out,10.18,9.)
        self.assertGreater(cmd[0],0.);self.assertGreater(cmd[3],0.)
        self.assertEqual(cmd[1:3],(0.,0.));self.assertIn('Verified curve',reason)

    def test_decay_at_send_never_renews_the_source_or_accelerates(self):
        d,v,m=curve_case();old=m.snapshot();forward=[];yaw=[]
        for now in (10.10,10.14,10.18,10.22,10.231):
            d.update(video_time=now-.01,prediction_time=now,heading=heading(now))
            v.update(visual_time=now-.01,deadline=now+.09)
            r=verified_curve(d,v,now);forward.append(r['forward']);yaw.append(r['yaw'])
            m.remember(d,v,now,True)
        self.assertEqual(forward,sorted(forward,reverse=True));self.assertEqual(forward[-1],0.)
        self.assertEqual(yaw,sorted(yaw,reverse=True));self.assertEqual(m.snapshot(),old)
        d,v,_=curve_case();out=apply_visual(d,v,10.18)
        a,_=dance_command(out,10.18,9.);b,_=dance_command(out,10.20,9.)
        self.assertLess(b[3],a[3]);self.assertEqual(dance_command(out,10.231,9.)[0],ZERO)

    def test_existing_350ms_source_ceiling_applies_even_with_new_nano_frames(self):
        d,v,_=curve_case();d.update(video_time=10.24,prediction_time=10.25,heading=heading(10.25))
        v.update(visual_time=10.24,deadline=10.34)
        self.assertFalse(verified_curve(d,v,10.25)['allowed'])

    def test_expiring_curve_cannot_restart_forward_from_older_longer_vector(self):
        d,v,_=curve_case();older,_,_=vector_case()
        d['recovery_anchor']=older['recovery_anchor']
        commands=[]
        for now in (10.18,10.22,10.24,10.27):
            d.update(prediction_time=now,video_time=now-.01,heading=heading(now))
            v.update(visual_time=now-.01,deadline=now+.08)
            out=apply_visual(d,v,now)
            commands.append(dance_command(out,now,9.)[0][3])
        self.assertGreater(commands[0],0.)
        self.assertEqual(commands,sorted(commands,reverse=True))
        self.assertEqual(commands[2:], [0.,0.])

    def test_all_motion_vetoes_remain_authoritative(self):
        changes=(dict(stale=True),dict(confirmed=False),dict(state='DIRECTIONAL_SEARCH'),
            dict(class_support={}),dict(candidate_pending=True),dict(bbox_clipped=True),
            dict(selected_class_id=1),dict(selected_class_name='MINI PRO4'),
            dict(spacing_phase='BRAKE'),dict(spacing_close_guard_armed=True),dict(video_time=10.),
            dict(heading=heading(10.18,21.)),dict(heading=heading(9.)),dict(generation=2),
            dict(measurement_time=9.98),dict(last_strong_measurement_time=9.8),
            dict(detection_summary=dict(target_eligible=1)),dict(motion_settings=dict(coordinated_turns=False)))
        for change in changes:
            d,v,_=curve_case();d.update(change)
            self.assertFalse(verified_curve(d,v,10.18)['allowed'],change)

    def test_nano_conflict_expiry_quality_and_reversal_revoke_curve(self):
        for change in (dict(blocked=True,reason='competing_mini_detection'),dict(epoch=3),dict(identity=8),
            dict(mode='shadow'),dict(visual_time=10.),dict(semantic_time=9.7),dict(deadline=10.17),
            dict(box=[65,40,95,60]),dict(box=[175,40,195,60]),
            dict(checks=dict(control_suspect=True)),dict(checks=dict(geometry_suspect=True))):
            d,v,_=curve_case();v.update(change)
            self.assertFalse(verified_curve(d,v,10.18)['allowed'],change)

    def test_new_real_stop_reversal_reset_and_disable_discard_anchor(self):
        for enabled,change in ((False,{}),(True,dict(spacing_close_guard_armed=True)),
            (True,dict(accepted=True,intent=dict(yaw=-.1,forward=.1))),
            (True,dict(accepted=True,intent=dict(yaw=.1,forward=0.))),
            (True,dict(accepted=True,centering_control=dict(early_curve=dict(active=False))))):
            d,v,m=curve_case();d.update(change);m.remember(d,v,10.18,enabled)
            self.assertIsNone(m.snapshot(),change)
        c=FollowController('botsort',0,'neo');_,_,c.curve_continuation=curve_case()
        c.restart_control();self.assertIsNone(c.curve_continuation.snapshot())

    def test_prior_dance_or_resume_cannot_reuse_anchor(self):
        d,v,_=curve_case();out=apply_visual(d,v,10.18)
        self.assertEqual(dance_command(out,10.18,9.9)[0],ZERO)
        self.assertEqual(dance_command(out,10.18,9.,resume_after=9.9)[0],ZERO)

    def test_pending_ack_rechecks_live_stop_and_original_expiry(self):
        d,v,_=curve_case();sent=apply_visual(d,v,10.18)
        self.assertIsNone(transport_visual_reason(sent,sent,10.18))
        for change in (dict(spacing_close_guard_armed=True),dict(stale=True),dict(curve_anchor=None)):
            self.assertIsNotNone(transport_visual_reason(sent,dict(sent,**change),10.18))
        self.assertIsNotNone(transport_visual_reason(sent,sent,10.231))

    def test_current_nano_can_support_pending_curve_without_renewing_real_deadline(self):
        d,v,_=curve_case();d['motion_settings']['nano_visual_max_age_seconds']=.04
        sent=apply_visual(d,v,10.18)
        latest=copy.deepcopy(d);latest.update(prediction_time=10.205,video_time=10.20,heading=heading(10.205))
        live=dict(v,visual_time=10.20,deadline=10.28)
        latest=apply_visual(latest,live,10.205)
        status=pending_motion_status(sent,latest,10.205,9.,(.001,0.,0.,.001),(.5,.1,.1,.1))
        self.assertIsNone(status['reason']);self.assertTrue(status['renewed'])
        self.assertEqual(status['authorized_measurement_time'],9.88)
        self.assertIsNotNone(pending_motion_status(sent,latest,10.231,9.,(.001,0.,0.,.001),(.5,.1,.1,.1))['reason'])

    def test_new_real_measurement_can_resume_compatible_curve_but_never_old_direction_after_reversal(self):
        d,v,_=curve_case();sent=apply_visual(d,v,10.18)
        latest=copy.deepcopy(d);latest.update(accepted=True,state='TRACK',measurement_time=10.08,
            last_strong_measurement_time=10.08,measurement_id=d['measurement_id']+1,
            prediction_time=10.20,video_time=10.19,heading=heading(10.20),measurement_age_ms=120.)
        live=dict(v,semantic_time=10.08,visual_time=10.19,deadline=10.28)
        latest=apply_visual(latest,live,10.20)
        status=pending_motion_status(sent,latest,10.20,9.,(.001,0.,0.,.001),(.5,.1,.1,.1))
        self.assertIsNone(status['reason']);self.assertEqual(status['support_source'],'new_real_neo')
        latest['base_intent']['yaw']=-.14
        status=pending_motion_status(sent,latest,10.20,9.,(.001,0.,0.,.001),(.5,.1,.1,.1))
        self.assertIsNotNone(status['reason']);self.assertFalse(status['renewed'])

    def controller_curve(self,hz=10,direction=1):
        c=FollowController('botsort',0,'neo');s=Settings()
        for i in range(18):
            source=10+i/hz;now=source+.12;x=950+direction*i*16
            box=[x,500,x+80,530]
            c.observe([Detection(Box(*box),.95,0,'neo')],source,now,i,(1920,1080),s)
            v=dict(mode='guarded',identity=c.tracker.logical_target_id,generation=1,epoch=1,
                blocked=False,reason='visual_supported',box=box,semantic_time=source,
                visual_time=now-.02,deadline=now+.08,frame_id=i*3,checks=dict(geometry_suspect=False,
                    control_suspect=False,texture_loss_streak=0,visible_fraction=1.,appearance_correlation=.8))
            d=c.tick(now,1920,1080,now-.01,s,heading(now),visual=v,generation=1)
        self.assertTrue(d['centering_control']['early_curve']['active'])
        self.assertIsNotNone(d['curve_anchor']);self.assertEqual(d['state'],'TRACK')
        c.observe([],source+.1,source+.22,18,(1920,1080),s)
        now=source+.28;v.update(visual_time=now-.02,deadline=now+.07)
        d=c.tick(now,1920,1080,now-.01,s,heading(now),visual=v,generation=1)
        out=apply_visual(d,v,now);self.assertEqual(out['nano_authority'],'curve_bridge')
        cmd,_=dance_command(out,now,9.)
        self.assertGreater(cmd[0]*direction,0.);self.assertGreater(cmd[3],0.)
        self.assertEqual(d['curve_anchor']['source_time'],source)
        c.restart_control();self.assertIsNone(c.curve_continuation.snapshot())
        return cmd

    def test_live_controller_at_10hz_keeps_a_short_verified_curve_on_missing_yolo(self):
        self.controller_curve()

    def test_live_curve_and_expiry_match_in_hard_at_10_and_12hz_both_directions(self):
        for hz in (10,12):
            for direction in (-1,1):
                with self.subTest(hz=hz,direction=direction):
                    normal=self.controller_curve(hz,direction)
                    with patch.object(activity,'enabled',False):
                        hard=self.controller_curve(hz,direction)
                    self.assertEqual(normal,hard)


if __name__=='__main__':unittest.main()
