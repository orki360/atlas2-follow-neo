"""v17 position-source integration and transport invariants; no aircraft IO."""
import copy
import unittest
from dataclasses import replace
from unittest.mock import patch

from follow_neo.control.controller import FollowController
from follow_neo.control.position import select_position, position_guard_reason
from follow_neo.control.nano_guard import apply_visual
from follow_neo.control.dance import dance_command, ZERO
from follow_neo.control.motion_continuity import pending_motion_status
from follow_neo.control.curve import EarlyCurve
from follow_neo.shared.types import Settings, Box, Detection, Kinematics, settings_from_config
from follow_neo.diagnostics import activity
from test_visual_continuity import GOOD
from test_tracking_recovery9 import heading


def visual(source, now, identity, x=1100., epoch=1):
    return dict(mode='guarded',blocked=False,reason='visual_supported',epoch=epoch,generation=1,
        identity=identity,semantic_time=source,visual_time=now-.02,deadline=now+.08,
        box=[x,480.,x+100.,550.],frame_id=round(now*60),score=.9,checks=dict(GOOD))


def established(enabled=True, **settings):
    c=FollowController('botsort',0,'neo');s=Settings(nano_primary_position=enabled,**settings)
    for i in range(12):
        source=10+i*.1;now=source+.12
        c.observe([Detection(Box(1080,480,1180,550),.95,0,'neo')],source,now,i,(1920,1080),s)
        v=visual(source,now,c.tracker.logical_target_id)
        d=c.tick(now,1920,1080,now-.01,s,heading(now),visual=v,generation=1)
    return c,s,d,v,now


class NanoPositionTests(unittest.TestCase):
    def test_camera_cadence_steers_between_10hz_detector_updates(self):
        for hz in (30,60):
            c=FollowController('botsort',0,'neo');s=Settings();positions=set();measurements=set()
            for i in range(hz*2):
                now=10.12+i/hz
                if i%(hz//10)==0:
                    source=now-.12
                    current_heading=heading(now)  # telemetry has its own 10 Hz source clock
                    x=1000.+150.*(source-10.)
                    c.observe([Detection(Box(x-50,480,x+50,550),.95,0,'neo')],
                              source,now,i,(1920,1080),s)
                x=1000.+150.*(now-.02-10.)
                v=visual(source,now,c.tracker.logical_target_id,x=x-50.)
                d=c.tick(now,1920,1080,now-.01,s,current_heading,visual=v,generation=1)
                out=apply_visual(d,v,now)
                if i>=hz:
                    self.assertTrue(out['position_applied'],(hz,i,d['control_position']))
                    self.assertEqual(d['measurement_time'],source)
                    self.assertAlmostEqual(d['error_x'],(x-960.)/960.)
                    self.assertAlmostEqual(d['raw_width_ratio'],100./1920)
                    positions.add(d['control_position']['source_time']);measurements.add(d['measurement_id'])
            self.assertEqual(len(positions),hz)
            self.assertEqual(len(measurements),10)
            self.assertTrue(d['centering_control']['early_curve']['active'])
            # Manual preview never switches the position source on its own.
            manual=c.tick(now,1920,1080,now-.01,s,heading(now),visual=v,generation=1,control_available=False)
            self.assertEqual(manual['control_position']['source'],'botsort')

    def test_live_controller_uses_new_nano_frames_between_same_yolo_measurement(self):
        c,s,a,v,t=established()
        self.assertEqual(a['state'],'TRACK')
        self.assertEqual(a['control_position']['source'],'nanotrack')
        v=visual(a['measurement_time'],t+.035,c.tracker.logical_target_id,x=1125.)
        b=c.tick(t+.035,1920,1080,t+.025,s,heading(t+.035),visual=v,generation=1)
        self.assertAlmostEqual(b['error_x'],(1175.-960.)/960.)
        self.assertNotEqual(a['error_x'],b['error_x'])
        self.assertNotEqual(a['intent']['yaw'],b['intent']['yaw'])
        for key in ('measurement_id','measurement_time','last_strong_measurement_time',
                    'raw_width_ratio','target_identity','confirmation_hits','reacquire_hits'):
            self.assertEqual(a[key],b[key],key)
        out=apply_visual(b,v,t+.035)
        self.assertTrue(out['position_applied'])
        self.assertTrue(any(dance_command(out,t+.035,9.)[0]))

    def test_nano_size_never_replaces_distance_or_detector_kinematics(self):
        c,s,a,v,t=established()
        changed=dict(v,box=[1085.,470.,1215.,560.])  # same centre, different size
        b=c.tick(t,1920,1080,t-.01,s,heading(t),visual=changed,generation=1)
        self.assertEqual(b['control_position']['source'],'nanotrack')
        for key in ('raw_width_ratio','projected_width_ratio','spacing_phase','spacing_close_guard_armed'):
            self.assertEqual(a[key],b[key],key)
        self.assertEqual(a['kinematics'],b['kinematics'])
        self.assertEqual(b['control_position']['size_source'],'yolo_botsort')

    def test_disabled_shadow_missing_and_invalid_visual_use_real_position(self):
        c,s,d,v,t=established()
        k=Kinematics(**d['kinematics'])
        for candidate in (None,dict(v,mode='shadow'),dict(v,blocked=True),dict(v,score=.3),
                          dict(v,generation=2),dict(v,identity=999),dict(v,visual_time=t-.11),
                          dict(v,checks=dict(GOOD,control_suspect=True)),dict(v,box=[0,0,10,10])):
            selected,info=select_position(d,k,candidate,t)
            self.assertEqual(info['source'],'botsort')
            self.assertEqual(selected,k)
        self.assertEqual(select_position(d,k,v,t,False)[1]['reason'],'disabled')

    def test_primary_cannot_acquire_identity_or_renew_a_missing_real_measurement(self):
        c,s,d,v,t=established();k=Kinematics(**d['kinematics'])
        for changes in (dict(accepted=False),dict(confirmed=False),dict(selected_class_id=1),
                        dict(class_support={}),dict(last_strong_measurement_time=t-1),
                        dict(stale=True),dict(target_identity=None)):
            self.assertEqual(select_position(dict(d,**changes),k,v,t)[1]['source'],'botsort',changes)
        expired=copy.deepcopy(d);expired['measurement_time']=t-.26
        self.assertEqual(select_position(expired,k,v,t)[1]['source'],'botsort')

    def test_current_visual_can_validate_the_original_source_without_teleporting(self):
        c,s,d,v,t=established();k=Kinematics(**d['kinematics'])
        # A different delayed source is not applied directly to the live box.
        stale_source=dict(v,visual_time=d['measurement_time']-.01)
        self.assertEqual(select_position(d,k,stale_source,t)[1]['source'],'botsort')
        self.assertEqual(select_position(d,k,dict(v,semantic_time=d['measurement_time']+.01),t)[1]['source'],'botsort')

    def test_source_switch_does_not_invent_image_velocity(self):
        c,s,d,v,t=established()
        fallback=c.tick(t+.01,1920,1080,t,s,heading(t+.01),visual=None,generation=1)
        self.assertEqual(fallback['control_position']['source'],'botsort')
        self.assertEqual(fallback['centering_control']['error_rate'],0.)
        live=visual(d['measurement_time'],t+.02,c.tracker.logical_target_id,x=1100.)
        resumed=c.tick(t+.02,1920,1080,t+.01,s,heading(t+.02),visual=live,generation=1)
        self.assertEqual(resumed['control_position']['reason'],'nano_reentry_confirming')
        live=visual(d['measurement_time'],t+.055,c.tracker.logical_target_id,x=1100.)
        resumed=c.tick(t+.055,1920,1080,t+.045,s,heading(t+.055),visual=live,generation=1)
        self.assertEqual(resumed['control_position']['source'],'nanotrack')
        self.assertEqual(resumed['centering_control']['error_rate'],0.)

    def test_old_nano_intent_is_not_relabelled_yolo_when_visual_expires(self):
        c,s,d,v,t=established();before=copy.deepcopy(d)
        out=apply_visual(d,v,t+.09)
        self.assertEqual(out['nano_authority'],'blocked')
        self.assertEqual(dance_command(out,t+.09,9.)[0],ZERO)
        self.assertEqual(d,before)
        old=apply_visual(d,v,t)
        self.assertEqual(dance_command(old,t+.09,9.)[0],ZERO)
        for missing in (None,dict(v,mode='off'),dict(v,mode='shadow')):
            out=apply_visual(d,missing,t)
            self.assertEqual(out['nano_authority'],'blocked')
            self.assertEqual(dance_command(out,t,9.)[0],ZERO)

    def test_visual_identity_veto_expiry_and_decision_deadline_at_send(self):
        c,s,d,v,t=established();out=apply_visual(d,v,t)
        for changes in (dict(epoch=2),dict(identity=99),dict(generation=2),
                        dict(blocked=True,reason='competing_mini_detection')):
            changed=copy.deepcopy(out);changed['nano'].update(changes)
            self.assertEqual(dance_command(changed,t,9.)[0],ZERO)
        self.assertIsNotNone(position_guard_reason(out,v,t+.09))
        self.assertEqual(dance_command(out,t+.21,9.)[0],ZERO)
        self.assertEqual(dance_command(out,t,t)[0],ZERO)

    def test_pending_same_real_measurement_can_use_compatible_new_position(self):
        c,s,d,v,t=established();sent=apply_visual(d,v,t)
        live=visual(d['measurement_time'],t+.09,c.tracker.logical_target_id,x=1102.)
        latest=c.tick(t+.09,1920,1080,t+.08,s,heading(t+.09),visual=live,generation=1)
        latest=apply_visual(latest,live,t+.09)
        before=copy.deepcopy((sent,latest))
        status=pending_motion_status(sent,latest,t+.09,9.,(.001,0.,0.,.001),(.5,.1,.1,.1))
        self.assertTrue(status['renewed'],status)
        self.assertEqual(status['authorized_measurement_time'],d['measurement_time'])
        self.assertEqual((sent,latest),before)
        # A live stop or reverse still defeats renewal under the old ACK budget.
        latest['base_intent']['yaw']=-.1
        self.assertFalse(pending_motion_status(sent,latest,t+.09,9.,(.001,0.,0.,.001),(.5,.1,.1,.1))['renewed'])

    def test_near_target_stop_and_original_loss_budget_still_win(self):
        c,s,d,v,t=established(edge_search_enabled=False)
        before=d['measurement_time']
        c.observe([],before+.1,t+.1,12,(1920,1080),s)
        live=visual(before,t+.25,c.tracker.logical_target_id)
        lost=c.tick(t+.25,1920,1080,t+.24,s,heading(t+.25),visual=live,generation=1)
        self.assertEqual(lost['control_position']['source'],'botsort')
        self.assertEqual(lost['measurement_time'],before)
        out=apply_visual(lost,live,t+.25)
        # The pre-existing recovery-vector path has its own 550 ms budget;
        # the new position path must neither erase nor extend that contract.
        self.assertEqual(out['recovery_motion']['source_time'],before)
        self.assertAlmostEqual(out['recovery_motion']['until'],before+.55)
        live=visual(before,t+.60,c.tracker.logical_target_id)
        expired=c.tick(t+.60,1920,1080,t+.59,s,heading(t+.60),visual=live,generation=1)
        self.assertEqual(dance_command(apply_visual(expired,live,t+.60),t+.60,9.)[0],ZERO)
        c,s,d,v,t=established(stop_width=.04)
        self.assertEqual(d['control_position']['source'],'nanotrack')
        self.assertLessEqual(dance_command(apply_visual(d,v,t),t,9.)[0][3],0.)

    def test_hard_and_normal_compute_identical_positions_without_diagnostics_io(self):
        _,_,normal,_,_=established()
        with patch.object(activity,'enabled',False):
            _,_,hard,_,_=established()
        # Separate acquisitions have distinct BoT identities by design.
        hard['control_position']['identity']=normal['control_position']['identity']
        for key in ('control_position','intent','raw_width_ratio','measurement_time','centering_control'):
            self.assertEqual(normal[key],hard[key],key)

    def test_early_curve_keeps_its_time_horizon_at_camera_and_detector_rates(self):
        for hz in (10,12,30,40,60):
            c=EarlyCurve();result=None
            for i in range(round(hz*.4)+1):
                t=10+i/hz
                result=c.update(t,.1+.25*(t-10),t+.04,(1,7),True,True)
            self.assertTrue(result['active'],hz)
            self.assertAlmostEqual(result['lead'],.03,places=6)
            # Even a small reversal in the latest high-rate sample must revoke
            # the lead, although the older 50 ms sample still suggests outward motion.
            self.assertFalse(c.update(t+1/hz,.1+.25*(t-10)-.001,t+1/hz+.04,(1,7),True,True)['active'])
            self.assertFalse(c.update(t+1/hz,-.1,t+1/hz+.04,(1,7),True,True)['active'])

    def test_config_upgrade_retains_every_existing_value_and_allows_ab_comparison(self):
        old=dict(schema_version=8,settings=dict(nano_mode='guarded',confidence=.6,stop_width=.42))
        s=settings_from_config(old)
        self.assertTrue(s.nano_primary_position);self.assertEqual(s.confidence,.6);self.assertEqual(s.stop_width,.42)
        self.assertFalse(replace(s,nano_primary_position=False).validate().nano_primary_position)
        with self.assertRaises(ValueError):replace(s,nano_primary_position='yes').validate()


if __name__=='__main__':unittest.main()
