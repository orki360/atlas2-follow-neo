"""Loss-direction priority, BOOST continuity, and boundary-aware handoff."""
import unittest
from dataclasses import replace
from unittest.mock import Mock
from follow_neo.recovery_search import RecoverySearch
from follow_neo.edge_search import yaw_override
from follow_neo.dance import dance_command
from follow_neo.types import Box,Kinematics,Settings


def heading(t,yaw=0.):return dict(time=t,yaw_deg=yaw,source='AircraftAttitude')


class DirectionTests(unittest.TestCase):
    def begin(self,side=1,edge=False,stationary=False):
        self.r=RecoverySearch();self.s=Settings(search_yaw_degrees=90,edge_search_seconds=2)
        for i in range(10):
            t=10+i*.04
            cx=(1000+i*20 if edge else 700+i*8) if not stationary else 800
            if side<0:cx=1280-cx
            self.r.observe(True,False,True,Box(cx-40,320,cx+40,360),Kinematics(),1280,720,t,i,self.s)
            self.r.update(t,False,False,'APPROACH',False,self.s,heading(t))
        self.start=t+.5
        return self.step(self.start)
    def step(self,t,yaw=0,**kwargs):
        return self.r.update(t,True,False,'APPROACH',False,self.s,heading(t,yaw),**kwargs)
    def decision(self,d,t):
        return dict(edge_search=d,prediction_time=t,state='DIRECTIONAL_SEARCH',
                    accepted=False,stale=False,spacing_phase='APPROACH')
    def test_motion_estimate_sets_first_scan_on_both_sides_without_boost(self):
        for side in (-1,1):
            with self.subTest(side=side):
                d=self.begin(side);self.assertEqual(d['phase'],'SCAN')
                self.assertEqual(d['first_scan_direction'],side)
                self.assertEqual(d['search_direction_source'],'measured_motion')
                d=self.step(self.start+.2);self.assertEqual(d['scan_target_degrees'],side*45)
                self.assertGreater(side*d['scan_yaw'],0)
    def test_boost_then_scan_keeps_estimated_direction_when_room_remains(self):
        for side in (-1,1):
            with self.subTest(side=side):
                d=self.begin(side,edge=True);deadline=d['until']
                self.assertEqual(d['phase'],'BOOST')
                d=self.step(self.start+.36,side*3)
                self.assertEqual(d['phase'],'SCAN');self.assertEqual(d['scan_target_degrees'],side*90)
                d=self.step(self.start+.55,side*3)
                values,_=dance_command(self.decision(d,self.start+.55),self.start+.55,9.)
                self.assertGreater(values[0]*side,0);self.assertLessEqual(abs(values[0]),1.)
                self.assertEqual(values[1:],(0.,0.,0.));self.assertEqual(d['until'],deadline)
    def test_already_reached_boundary_reverses_without_recentering_range(self):
        for side in (-1,1):
            self.begin(side,edge=True)
            self.step(self.start+.1,side*45)
            d=self.step(self.start+.2,side*90)
            self.assertEqual(d['scan_target_degrees'],0.)
            self.assertEqual(d['first_scan_direction'],side)
            self.assertIn(side,d['boundaries_reached'])
            d=self.step(self.start+.4,side*90)
            self.assertLess(side*d['scan_yaw'],0)
    def test_stationary_side_is_fallback_when_motion_is_not_reliable(self):
        for side in (-1,1):
            d=self.begin(side,stationary=True)
            self.assertEqual(d['first_scan_direction'],side)
            self.assertEqual(d['search_direction_source'],'last_seen_side')
    def test_clear_motion_takes_precedence_over_current_image_side(self):
        r=RecoverySearch();s=Settings()
        for i in range(10):
            cx=300+i*12;t=10+i*.04
            r.observe(True,False,True,Box(cx-40,320,cx+40,360),Kinematics(),1280,720,t,i,s)
        self.assertLess(cx,640);self.assertEqual(r.estimated_direction,1)
        self.assertEqual(r.direction_source,'measured_motion')
    def test_approach_rate_brakes_boost_before_reaching_boundary(self):
        self.begin(-1,edge=True)
        self.step(self.start+.1,-8)
        self.r.rate.update=Mock(return_value=-50.)
        d=self.step(self.start+.2,-70)
        self.assertEqual(d['phase'],'SCAN');self.assertEqual(d['boost_exit_reason'],'boundary_braking')
        self.assertGreater(d['scan_offset_degrees'],-90)
        self.assertEqual(d['scan_target_degrees'],-90)
        self.assertEqual(yaw_override(self.decision(d,self.start+.2),self.start+.2,9),0)
    def test_send_gate_holds_boost_within_braking_margin(self):
        d=self.begin(1,edge=True);d.update(angle_progress_degrees=88,boost_brake_margin_degrees=3)
        self.assertEqual(yaw_override(self.decision(d,self.start),self.start,9),0)
    def test_recorded_left_loss_skips_boost_when_already_turning_toward_boundary(self):
        r=RecoverySearch();s=Settings(search_yaw_degrees=90,edge_search_seconds=2)
        r.eligible=True;r.source_time=10.;r.loss_heading=101.8;r.loss_heading_time=10.05
        r.loss_heading_source='AircraftAttitude';r.estimated_direction=-1;r.direction_source='measured_motion'
        r.edge.direction=-1;r.edge.evidence={'outward':True};r.box_edge=True
        r.rate.update=Mock(return_value=-42.037)
        d=r.update(10.469,True,False,'HOLD_REACQUIRE',False,s,heading(10.469,75.6))
        self.assertEqual(d['phase'],'SCAN');self.assertEqual(d['first_scan_direction'],-1)
        self.assertEqual(d['scan_target_degrees'],-45)
        self.assertEqual(d['boost_exit_reason'],'boundary_braking')
    def test_candidate_pause_preserves_direction_and_deadline(self):
        d=self.begin(1);end=d['until']
        d=self.step(self.start+.2,candidate_pending=True)
        self.assertFalse(d['active']);self.assertEqual(d['first_scan_direction'],1)
        d=self.step(self.start+.4)
        self.assertGreater(d['scan_yaw'],0);self.assertEqual(d['until'],end)
    def test_timeout_does_not_restart_scan_to_satisfy_preferred_direction(self):
        d=self.begin(1);d=self.step(d['until']+.01)
        self.assertFalse(d['active']);self.assertEqual(d['reason'],'search_timeout')
    def test_missing_heading_stops_directional_scan(self):
        self.begin(1)
        d=self.r.update(self.start+.2,True,False,'APPROACH',False,self.s,None)
        self.assertFalse(d['active']);self.assertEqual(d['reason'],'waiting_heading')
        d=self.r.update(self.start+.6,True,False,'APPROACH',False,self.s,None)
        self.assertEqual(d['reason'],'heading_unavailable')
    def test_retained_edge_direction_cannot_override_new_motion_estimate(self):
        r=RecoverySearch();s=Settings()
        r.source_time=10.;r.eligible=True;r.box_edge=True
        r.edge.direction=-1;r.edge.evidence={'outward':True}
        r.estimated_direction=1;r.direction_source='measured_motion'
        d=r.update(10.5,True,False,'APPROACH',False,s,heading(10.5))
        self.assertEqual(d['phase'],'SCAN');self.assertEqual(d['scan_target_degrees'],45)
