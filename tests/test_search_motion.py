"""Search-only rate stability, speed-envelope braking and BOOST gate contracts."""
import unittest
from unittest.mock import Mock
from follow_neo.search_motion import SearchHeadingRate,SearchYawProfile,BOOST_MAX_SECONDS
from follow_neo.search_test import SearchTest
from follow_neo.edge_search import yaw_override
from follow_neo.types import Settings,Box,Kinematics

def heading(t,yaw=0,source='AircraftAttitude'):
    return dict(time=t,yaw_deg=(yaw+180)%360-180,source=source)

class SearchMotionTests(unittest.TestCase):
    def test_cached_steps_keep_rate_stable_instead_of_zero_on_each_repeat(self):
        r=SearchHeadingRate();rates=[]
        for i in range(21):
            t=10+i*.05;yaw=(i//2)*6
            rate=r.update(heading(t,yaw),t)
            if i>=10:rates.append(rate)
        self.assertTrue(all(50<rate<70 for rate in rates),rates)

    def test_rate_wraparound_duplicates_stopping_and_source_change(self):
        r=SearchHeadingRate()
        for i in range(10):
            t=10+i*.05;rate=r.update(heading(t,179+i*3),t)
        self.assertAlmostEqual(rate,60)
        self.assertEqual(r.update(heading(t,179+9*3),t+.01),rate)
        for i in range(1,10):rate=r.update(heading(t+i*.05,206),t+i*.05)
        self.assertAlmostEqual(rate,0)
        self.assertIsNone(r.update(heading(11,206,'CompassHeading'),11))
        self.assertIsNone(r.update(heading(11,206),12))

    def test_rate_rejects_out_of_order_or_discontinuous_heading(self):
        r=SearchHeadingRate()
        r.update(heading(10),10);r.update(heading(10.1,5),10.1)
        self.assertIsNone(r.update(heading(10.05,6),10.1))
        self.assertIsNone(r.update(heading(10.2,150),10.2))

    def test_brake_hold_does_not_pulse_on_small_rate_changes(self):
        p=SearchYawProfile();p.reset(10,1.)
        self.assertEqual(p.update(10,90,10.1),0)
        self.assertEqual(p.update(10,45,10.2),0)
        self.assertEqual(p.diagnostics['control_reason'],'brake_hold')
        resumed=p.update(10,5,10.3)
        self.assertGreater(resumed,0);self.assertLessEqual(resumed,.101)

    def test_command_rise_is_time_bounded_and_braking_is_immediate(self):
        for sign in (-1,1):
            p=SearchYawProfile();p.reset(10)
            u=p.update(sign*100,0,10.05)
            self.assertLessEqual(abs(u),.051);self.assertGreater(u*sign,0)
            self.assertEqual(p.update(sign*1,sign*100,10.06),0)
            self.assertLessEqual(abs(p.update(sign*100,0,12)),.201)

    def test_boost_backup_stops_search_and_cannot_restart_without_new_target(self):
        test=SearchTest(Settings(search_yaw_degrees=80),heading(10),10,'RIGHT',lambda *a:None,boost=True)
        for i in range(1,20):
            t=10+i*.1
            self.assertEqual(test.update(heading(t),t)[0],1)
        self.assertEqual(test.update(heading(10+BOOST_MAX_SECONDS),10+BOOST_MAX_SECONDS),(0,0,0,0))
        self.assertEqual(test.reason,'boost_timeout');self.assertEqual(test.search.phase,'DONE')
        self.assertEqual(test.update(heading(12.1),12.1),(0,0,0,0))

    def test_send_gate_stops_old_boost_even_without_new_planner_update(self):
        test=SearchTest(Settings(search_yaw_degrees=80),heading(10),10,'LEFT',lambda *a:None,boost=True)
        test.update(heading(11.95),11.95)
        self.assertEqual(yaw_override(test.decision,11.96,10),-1)
        self.assertEqual(yaw_override(test.decision,12.01,10),0)

    def test_candidate_holds_boost_and_verified_target_exits_it(self):
        test=SearchTest(Settings(search_yaw_degrees=80),heading(10),10,'RIGHT',lambda *a:None,boost=True)
        r=test.search;s=test.settings
        held=r.update(10.5,True,False,'APPROACH',False,s,heading(10.5),candidate_pending=True)
        self.assertTrue(held['paused']);self.assertFalse(held['active'])
        for i in range(3):
            t=10.6+i*.04
            r.observe(True,False,True,Box(470,100,530,140),Kinematics(),1000,600,t,i,s)
        self.assertEqual(r.phase,'IDLE')

    def test_short_heading_gap_pauses_without_extending_boost_backup(self):
        test=SearchTest(Settings(search_yaw_degrees=80),heading(10),10,'RIGHT',lambda *a:None,boost=True)
        r=test.search;s=test.settings
        d=r.update(11.8,True,False,'APPROACH',False,s,None)
        self.assertFalse(d['active'])
        d=r.update(12.01,True,False,'APPROACH',False,s,heading(12.01))
        self.assertEqual(d['phase'],'DONE');self.assertEqual(d['reason'],'boost_timeout')

if __name__=='__main__':unittest.main()
