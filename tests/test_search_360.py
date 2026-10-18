"""Full-circle search contracts with scripted heading, without aircraft I/O."""
import unittest
from dataclasses import asdict
from follow_neo.types import Settings,settings_from_config,Box,Kinematics
from follow_neo.search_test import SearchTest
from follow_neo.recovery_search import RecoverySearch,scan_command
from follow_neo.edge_search import yaw_override


def heading(t,angle):
    return dict(time=t,yaw_deg=(angle+180)%360-180,source='AircraftAttitude')


class FullCircleSearchTests(unittest.TestCase):
    def test_settings_round_trip_and_reject_outside_range(self):
        s=Settings(search_yaw_degrees=360).validate()
        self.assertEqual(settings_from_config({'schema_version':6,'settings':asdict(s)}).search_yaw_degrees,360)
        for angle in (-1,360.01,float('nan'),float('inf')):
            with self.assertRaises(ValueError):Settings(search_yaw_degrees=angle).validate()

    def test_production_and_test_ranges_and_budgets_agree(self):
        for scenario,cx,bounds,budget in [('RIGHT',950,(0,360),31),('LEFT',50,(-360,0),31),('CENTER',500,(-180,180),29.35)]:
            s=Settings(search_yaw_degrees=360);r=RecoverySearch()
            r.observe(True,False,True,Box(cx-20,200,cx+20,240),Kinematics(),1000,600,10,1,s)
            r.update(10,False,False,'APPROACH',False,s,heading(10,170))
            d=r.update(10.5,True,False,'APPROACH',False,s,heading(10.5,170))
            t=SearchTest(s,heading(10,170),10,scenario,lambda *a:None)
            self.assertEqual((d['scan_min_degrees'],d['scan_max_degrees']),bounds)
            self.assertEqual((t.search.low,t.search.high),bounds)
            self.assertEqual(r.duration,budget);self.assertEqual(t.search.duration,budget)

    def test_finite_directional_and_centered_coverage_with_observation_stops_across_wraps(self):
        for scenario in ('LEFT','RIGHT','CENTER'):
            sign=1 if scenario=='RIGHT' else -1
            s=Settings(search_yaw_degrees=360);test=SearchTest(s,heading(10,170),10,scenario,lambda *a:None)
            now=10.;last=0.
            def step(offset):
                nonlocal now,last
                now+=.1;last=offset
                return test.update(heading(now,170+offset),now)
            # Actual heading traverses each sector, then settles at its end.
            for _ in range(4):step(0)
            targets=list(test.search.sectors.targets)
            for target in targets:
                direction=1 if target>last else -1
                while abs(target-last)>.01:step(last+direction*min(5,abs(target-last)))
                for _ in range(12):
                    step(last)
                    if test.stopped is not None:break
            self.assertEqual(test.reason,'scan_complete')
            self.assertTrue(test.measurement_complete)
            self.assertEqual(len(test.search.boundaries),2)
            self.assertAlmostEqual(test.offset,targets[-1])
            self.assertAlmostEqual(test.travel,360 if scenario!='CENTER' else 540)
            self.assertEqual(len(test.search.sectors.completed),len(targets))

    def test_large_arc_boost_hands_to_scan_without_extending_boost_timeout(self):
        for scenario,sign in [('LEFT',-1),('RIGHT',1)]:
            test=SearchTest(Settings(search_yaw_degrees=360),heading(10,170),10,scenario,lambda *a:None,boost=True)
            test.update(heading(10.1,170+sign*9),10.1)
            self.assertEqual(yaw_override(test.decision,10.1,10),sign)
            for i in range(2,20):
                t=10+i*.1;test.update(heading(t,170+sign*i*9),t)
                if test.search.phase=='SCAN':break
            self.assertEqual(test.search.phase,'SCAN')
            self.assertEqual(test.search.boost_exit_reason,'boundary_braking')
            self.assertEqual(test.decision['edge_search']['boost_angle_degrees'],60)
            self.assertLess(t,12)
            self.assertGreaterEqual(sign*scan_command(test.decision,t,10),0)
            self.assertEqual(test.search.target,sign*60)
        test=SearchTest(Settings(search_yaw_degrees=360),heading(10,0),10,'RIGHT',lambda *a:None,boost=True)
        test.update(heading(12,0),12)
        self.assertEqual(test.search.boost_exit_reason,'boost_timeout')
        self.assertEqual(test.search.phase,'SCAN');self.assertIsNone(test.stopped)
        self.assertEqual(test.search.until,41.)

    def test_long_scan_gate_accepts_and_stops_stale_outward_and_overbudget(self):
        test=SearchTest(Settings(search_yaw_degrees=360),heading(10,170),10,'RIGHT',lambda *a:None)
        test.update(heading(10.3,170),10.3);d=test.decision;s=d['edge_search']
        self.assertGreater(scan_command(d,10.3,10),0)
        for change in (dict(heading_time=9),dict(scan_offset_degrees=360),dict(paused=True),
                       dict(scan_max_degrees=361),dict(angle_degrees=361),dict(duration_seconds=36,until=46)):
            self.assertEqual(scan_command({**d,'edge_search':{**s,**change}},10.3,10),0)
        test.update(heading(41,170),41)
        self.assertEqual(test.reason,'search_timeout')

    def test_small_arc_budget_and_manual_duration_remain_unchanged(self):
        for angle,auto,expected in ((180,True,10),(360,False,7)):
            test=SearchTest(Settings(search_yaw_degrees=angle,edge_search_seconds=7,search_auto_duration=auto),heading(10,0),10,'RIGHT',lambda *a:None)
            self.assertEqual(test.search.duration,expected)


if __name__=='__main__':unittest.main()
