"""Estimator transitions without renewing evidence or aircraft authority."""
import copy
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from follow_neo.control.position import PositionSelector
from follow_neo.control.position_motion import BearingMotion, YawHandoff, HANDOFF_SECONDS
from follow_neo.control.centering import DampedTrackingController
from follow_neo.control.nano_guard import apply_visual
from follow_neo.control.dance import dance_command, ZERO
from follow_neo.shared.types import Kinematics, Settings
from follow_neo.diagnostics import activity
from test_nano_position import established, visual
from test_tracking_recovery9 import heading


class PositionHandoffTests(unittest.TestCase):
    def test_reentry_needs_distinct_frames_and_keeps_detector_position_while_waiting(self):
        c,s,d,v,t=established();selector=PositionSelector();k=Kinematics(**d['kinematics'])
        original=copy.deepcopy((d,k,v))
        self.assertEqual(selector.select(d,k,v,t)[1]['source'],'nanotrack')
        self.assertEqual(selector.select(d,k,dict(v,blocked=True),t)[1]['source'],'botsort')
        for now in (t,t+.01,t+.02):
            got,info=selector.select(d,k,v,now)
            self.assertEqual(got,k);self.assertEqual(info['reason'],'nano_reentry_confirming')
        next_visual=visual(d['measurement_time'],t+.035,c.tracker.logical_target_id)
        got,info=selector.select(d,k,next_visual,t+.035)
        self.assertEqual(info['source'],'nanotrack')
        self.assertEqual(info['source_time'],next_visual['visual_time'])
        self.assertEqual(info['measurement_time'],d['measurement_time'])
        self.assertEqual((d,k,v),original)

    def test_failed_or_expired_candidate_cannot_complete_reentry(self):
        c,s,d,v,t=established();k=Kinematics(**d['kinematics'])
        for failure in (None,dict(v,blocked=True),dict(v,identity=999),dict(v,epoch=99),
                        dict(v,score=.1),dict(v,visual_time=t-.2),dict(v,mode='shadow')):
            q=PositionSelector();q.select(d,k,v,t);q.select(d,k,None,t)
            q.select(d,k,v,t);q.select(d,k,failure,t+.005)
            newer=visual(d['measurement_time'],t+.035,c.tracker.logical_target_id)
            self.assertEqual(q.select(d,k,newer,t+.035)[1]['reason'],'nano_reentry_confirming')
        expired=dict(d,measurement_time=t-.251,last_strong_measurement_time=t-.251)
        self.assertEqual(q.select(expired,k,newer,t+.035)[1]['reason'],'waiting_real_neo')

    def test_source_offset_never_becomes_velocity_and_detector_history_stays_warm(self):
        ctl=DampedTrackingController();ctl.heading_rate.update=Mock(return_value=0.)
        for i in range(15):
            now=10.+i*.04;is_nano=i not in (8,9)
            p=dict(source='nanotrack' if is_nano else 'botsort',generation=1,identity=7,epoch=1,
                   source_time=now-.02,center=[132.,50.],handoff_enabled=True,reason='visual_expired')
            ctl.compute(Kinematics(True,132. if is_nano else 120.,50.,20.,10.),200,100,.12,False,
                now,Settings(),heading(now),now-.12,.20,dict(identity_key=(1,7)),True,True,p)
            self.assertEqual(ctl.error_rate,0.)
            self.assertFalse(ctl.diagnostics['early_curve']['active'])
            self.assertEqual(ctl.motion['botsort'].last_measurement,(now-.12,.20))

    def test_outward_curve_survives_switch_to_independently_verified_detector_history(self):
        ctl=DampedTrackingController();ctl.heading_rate.update=Mock(return_value=0.)
        for i in range(25):
            now=10.+i*.025
            if i%4==0:source=now-.12
            bot_error=.1+.25*(source-9.88)
            nano_error=.1+.25*(now-.02-9.88)
            nano=i<24;error=nano_error if nano else bot_error
            p=dict(source='nanotrack' if nano else 'botsort',generation=1,identity=7,epoch=1,
                   source_time=now-.02,center=[100.+100.*nano_error,50.],handoff_enabled=True,
                   reason='visual_expired')
            cmd,_,_=ctl.compute(Kinematics(True,100.+100.*error,50.,20.,10.),200,100,now-source,False,
                now,Settings(),heading(now),source,bot_error,dict(identity_key=(1,7)),True,True,p)
        self.assertEqual(ctl.diagnostics['position_source'],'botsort')
        self.assertTrue(ctl.diagnostics['early_curve']['active'])
        self.assertGreater(ctl.error_rate,0.)
        self.assertGreater(cmd.yaw,0.)

    def test_history_cannot_cross_identity_epoch_or_long_gap(self):
        for change in ('identity','epoch','gap'):
            m=BearingMotion()
            for i in range(5):m.update(10.+i*.05,.1+i*.02,10.12+i*.05,(1,7,1),True,True,True)
            self.assertTrue(m.curve.active)
            key=(1,8,1) if change=='identity' else (1,7,2) if change=='epoch' else (1,7,1)
            source=10.25 if change!='gap' else 10.6
            rate,curve=m.update(source,.21,source+.12,key,True,True,True)
            self.assertEqual(rate,0.);self.assertFalse(curve['active'])

    def test_handoff_limits_only_a_bounded_same_direction_increase(self):
        h=YawHandoff();identity=(1,7);b=('botsort',identity);n=('nanotrack',1,7,1)
        self.assertEqual(h.update(.10,.2,10.,b,identity,True),(.1,False))
        got,limited=h.update(.20,.3,10.02,n,identity,True)
        self.assertTrue(limited);self.assertAlmostEqual(got,.14)
        deadline=h.until
        h.update(.22,.3,10.04,b,identity,True)
        self.assertEqual(h.until,deadline)  # switching again cannot extend it
        got,_=h.update(.22,.3,10.02+HANDOFF_SECONDS+.001,b,identity,True)
        self.assertEqual(got,.22)

    def test_zero_reduction_reverse_and_large_or_new_identity_jump_are_immediate(self):
        for requested,error,key,identity in ((0.,.2,('nanotrack',1,7,1),(1,7)),
                (.05,.2,('nanotrack',1,7,1),(1,7)),(-.2,-.2,('nanotrack',1,7,1),(1,7)),
                (.5,.9,('nanotrack',1,7,1),(1,7)),(.2,.3,('nanotrack',1,8,1),(1,8)),
                (.2,.3,('nanotrack',1,7,2),(1,7))):
            h=YawHandoff();h.update(.1,.2,10.,('nanotrack',1,7,1),(1,7),True)
            self.assertEqual(h.update(requested,error,10.02,key,identity,True),(requested,False))

    def test_handoff_cannot_relax_forward_gates_or_change_distance(self):
        c,s,a,v,t=established();before=copy.deepcopy(a)
        first=c.tick(t+.01,1920,1080,t,s,heading(t+.01),visual=None,generation=1)
        for key in ('raw_width_ratio','measurement_time','measurement_id','target_identity','kinematics'):
            if key!='kinematics':self.assertEqual(first[key],before[key])
        # A new safety stop always wins, regardless of estimator history.
        s=replace(s,stop_width=.04)
        stopped=c.tick(t+.02,1920,1080,t+.01,s,heading(t+.02),visual=v,generation=1)
        self.assertLessEqual(dance_command(apply_visual(stopped,v,t+.02),t+.02,9.)[0][3],0.)
        invalid=dict(v,blocked=True,reason='competing_mini_detection')
        blocked=c.tick(t+.03,1920,1080,t+.02,s,heading(t+.03),visual=invalid,generation=1)
        self.assertEqual(dance_command(apply_visual(blocked,invalid,t+.03),t+.03,9.)[0],ZERO)

    def test_forward_ceiling_uses_raw_yaw_even_when_handoff_reduces_sent_yaw(self):
        ctl=DampedTrackingController();ctl.heading_rate.update=Mock(return_value=0.)
        s=Settings(coordinated_turns=False,forward_yaw_demand_limit=.19)
        obs=dict(identity_key=(1,7))
        p=dict(source='botsort',reason='visual_expired',handoff_enabled=True)
        ctl.compute(Kinematics(True,120.,50.,20.,10.),200,100,.12,False,10.,s,
                    heading(10.),9.88,.2,obs,True,True,p)
        p=dict(source='nanotrack',generation=1,identity=7,epoch=1,source_time=10.,
               center=[135.,50.],handoff_enabled=True)
        out,_,_=ctl.compute(Kinematics(True,135.,50.,20.,10.),200,100,.12,False,10.02,s,
                           heading(10.02),9.90,.2,obs,True,True,p)
        self.assertTrue(ctl.diagnostics['position_handoff_limited'])
        self.assertLess(out.yaw,s.forward_yaw_demand_limit)
        self.assertGreater(ctl.diagnostics['yaw_requested'],s.forward_yaw_demand_limit)
        self.assertEqual(ctl.forward_factor,0.)

    def test_normal_same_source_motion_has_no_added_ramp(self):
        h=YawHandoff();key=('nanotrack',1,7,1);identity=(1,7)
        h.update(.1,.2,10.,key,identity,True)
        self.assertEqual(h.update(.3,.3,10.02,key,identity,True),(.3,False))

    def test_hard_and_normal_transitions_have_identical_control_results(self):
        outputs=[]
        for enabled in (True,False):
            with patch.object(activity,'enabled',enabled):
                c,s,d,v,t=established();items=[]
                for dt,available in ((.01,False),(.03,True),(.06,True),(.08,False)):
                    fresh=visual(d['measurement_time'],t+dt,c.tracker.logical_target_id) if available else None
                    out=c.tick(t+dt,1920,1080,t+dt-.01,s,heading(t+dt),visual=fresh,generation=1)
                    items.append((out['intent'],out['error_x'],out['centering_control']['error_rate'],out['control_position']['source']))
                outputs.append(items)
        self.assertEqual(*outputs)

    def test_release_restart_discards_reentry_and_motion_history(self):
        c,s,d,v,t=established();c.tick(t+.01,1920,1080,t,s,visual=None,generation=1)
        self.assertTrue(c.position_selector.was_nano)
        c.restart_control()
        self.assertFalse(c.position_selector.was_nano)
        self.assertIsNone(c.smart.motion['nanotrack'].last_measurement)
        d=c.tick(t+.02,1920,1080,t+.01,s,visual=v,generation=1)
        self.assertEqual(d['control_position']['source'],'botsort')


if __name__=='__main__':unittest.main()
