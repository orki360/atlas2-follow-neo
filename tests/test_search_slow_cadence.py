"""Regression for ROI-driven 10 FPS: three strong returns confirm without another state-machine count."""
import unittest
from follow_neo.botsort_tracker import BoTSORTTracker
from follow_neo.controller import FollowController
from follow_neo.types import Box,Detection,Settings


def detection(x):return Detection(Box(x,180,x+60,220),.9)
def heading(t):return dict(time=t,yaw_deg=0.,source='AircraftAttitude')


class SlowCadenceTests(unittest.TestCase):
    def test_far_return_confirms_at_ten_fps_with_three_strong_samples(self):
        tracker=BoTSORTTracker((640,480))
        for i in range(10):tracker.update([detection(100)],10+i*.1,10+i*.1,i,.8,.8)
        identity=tracker.logical_target_id
        for i in range(10,20):tracker.update([],10+i*.1,10+i*.1,i,.8,.8)
        for i in range(3):
            t=12+i*.1
            accepted=tracker.update([detection(450)],t,t,20+i,.8,.8)
            self.assertEqual(accepted,i==2)
        self.assertEqual(tracker.logical_target_id,identity)
        self.assertEqual(tracker.association_reason,'verified_returning_target')
        self.assertEqual(tracker.identity_event['observations'],3)

    def test_scan_returns_to_track_at_ten_fps(self):
        controller=FollowController(tracker_backend='botsort')
        settings=Settings(confidence=.8,new_track_confidence=.8,search_yaw_degrees=360)
        phases=[]
        for i in range(23):
            t=10+i*.1
            ds=[detection(100)] if i<10 else [] if i<20 else [detection(450)]
            controller.observe(ds,t,t,i,(640,480),settings)
            d=controller.tick(t,640,480,t,settings,heading(t))
            phases.append(d['state'])
        self.assertIn('DIRECTIONAL_SEARCH',phases)
        self.assertNotIn('TRACK',phases[20:22])
        self.assertEqual(d['state'],'TRACK')
        self.assertEqual(d['edge_search']['phase'],'IDLE')

    def test_three_fast_samples_share_confirmation_and_duplicates_do_not_count(self):
        for fps in (30,50):
            with self.subTest(fps=fps):
                c=FollowController(tracker_backend='botsort')
                settings=Settings(confidence=.8,new_track_confidence=.8,search_yaw_degrees=360)
                for i in range(20):
                    t=10+i*.1
                    c.observe([detection(100)] if i<10 else [],t,t,i,(640,480),settings)
                    d=c.tick(t,640,480,t,settings,heading(t))
                self.assertEqual(d['state'],'DIRECTIONAL_SEARCH')
                for j in range(3):
                    t=12+j/fps
                    c.observe([detection(450)],t,t+.05,20+j,(640,480),settings)
                    d=c.tick(t+.05,640,480,t,settings,heading(t))
                    if j<2:self.assertNotEqual(d['state'],'TRACK')
                    self.assertFalse(c.observe([detection(450)],t,t+.051,20+j,(640,480),settings))
                    duplicate=c.tick(t+.051,640,480,t,settings,heading(t))
                    self.assertEqual(duplicate['state'],d['state'])
                self.assertEqual(d['state'],'TRACK')
                self.assertEqual(d['edge_search']['phase'],'IDLE')
                self.assertEqual(c.tracker.identity_event['strong_observations'],3)
                expired=c.tick(t+.26,640,480,t+.26,settings,heading(t+.26))
                self.assertFalse(expired['accepted'])
                self.assertNotEqual(expired['state'],'TRACK')

    def test_long_gap_does_not_accumulate_unrelated_return_evidence(self):
        tracker=BoTSORTTracker((640,480))
        for i in range(10):tracker.update([detection(100)],10+i*.1,10+i*.1,i,.8,.8)
        for i in range(10,20):tracker.update([],10+i*.1,10+i*.1,i,.8,.8)
        for i in range(20):
            t=12+i*.21
            self.assertFalse(tracker.update([detection(450)],t,t,20+i,.8,.8))
        self.assertEqual(tracker.diagnostics()['candidate_hits'],1)

if __name__=='__main__':unittest.main()
