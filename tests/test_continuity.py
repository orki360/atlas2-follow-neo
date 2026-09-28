import unittest
from unittest.mock import patch,Mock
from dataclasses import replace
import numpy as np
from follow_neo.botsort_tracker import BoTSORTTracker
from follow_neo.controller import FollowController
from follow_neo.types import Box,Detection,Settings,Kinematics
from follow_neo.recovery_search import RecoverySearch
from follow_neo.dance import dance_command
from follow_neo.gui import FollowLabWindow
from follow_neo.overlay import render_bundle
from follow_neo.video import Frame

def detection(x=200,confidence=.95):return Detection(Box(x,180,x+60,220),confidence)
def heading(t):return dict(yaw_deg=0.,time=t,source='test')

class ContinuityTests(unittest.TestCase):
    def setUp(self):
        self.t=BoTSORTTracker((640,480));self.i=0
    def feed(self,ds=None,dt=1/30):
        self.i+=1;stamp=10+self.i*dt
        return self.t.update([detection()] if ds is None else ds,stamp,stamp+.01,self.i,.8,.8,image=None)
    def acquire(self):self.feed();self.feed()
    def test_only_botsort_kalman_exists(self):
        with patch('follow_neo.tracker.BBoxTracker.__init__',side_effect=AssertionError('second filter')):
            self.acquire();self.assertFalse(hasattr(self.t,'kf'))
            self.assertIs(self.t.target.kalman_filter,self.t.engine.kalman_filter)
    def test_velocity_units_and_repeated_snapshots_do_not_advance_state(self):
        self.acquire();self.t.target.mean[4]=3.;before=self.t.target.mean.copy()
        a=self.t.anchor();k=self.t.snapshot(a['time']+.1)
        self.assertAlmostEqual(k.vx,90.);self.assertAlmostEqual(k.cx,a['cx']+9)
        self.t.snapshot(a['time']+.2);np.testing.assert_equal(before,self.t.target.mean)
    def test_small_overlap_strong_sequence_preserves_logical_identity(self):
        self.acquire();identity=self.t.logical_target_id;old=self.t.target_id
        # IoU = 5 / 115, far below the old 0.5 continuation threshold.
        self.assertFalse(self.feed([detection(255)]));self.assertTrue(self.t.candidate_pending)
        self.assertFalse(self.feed([detection(255)]));self.assertTrue(self.feed([detection(255)]))
        self.assertEqual(self.t.logical_target_id,identity)
        self.assertEqual(self.t.target_id,old)
        self.assertNotEqual(self.t.identity_event['candidate_track_id'],old)
        self.assertEqual(self.t.identity_event['observations'],3)
    def test_weak_sequence_restores_lost_track_but_cannot_acquire(self):
        self.assertFalse(self.feed([detection(255,.2)]));self.acquire();identity=self.t.logical_target_id
        self.feed([]);stamp=self.t.measurement_time
        for _ in range(3):self.assertFalse(self.feed([detection(250,.2)]))
        self.assertEqual(self.t.measurement_time,stamp)
        # Floating source-time rounding can require one extra 30Hz sample.
        accepted=self.feed([detection(250,.2)])
        if not accepted:accepted=self.feed([detection(250,.2)])
        self.assertTrue(accepted);self.assertEqual(self.t.logical_target_id,identity)
        self.assertTrue(self.t.measurement_weak);self.assertFalse(self.t.quality()['stable'])
    def test_distant_background_never_replaces_target(self):
        self.acquire();identity=self.t.logical_target_id;stamp=self.t.measurement_time
        for _ in range(25):self.assertFalse(self.feed([detection(490,.99)]))
        self.assertEqual(self.t.logical_target_id,identity);self.assertEqual(self.t.measurement_time,stamp)
    def test_ambiguous_nearby_candidates_do_not_reacquire(self):
        self.acquire();self.feed([])
        for _ in range(5):self.assertFalse(self.feed([detection(150,.2),detection(250,.2)]))
        self.assertEqual(self.t.association_reason,'ambiguous_candidates')
    def test_return_after_long_gap_requires_long_strong_sequence(self):
        self.acquire();identity=self.t.logical_target_id
        for _ in range(100):self.feed([])
        for _ in range(7):self.assertFalse(self.feed([detection(490)]))
        accepted=False
        for _ in range(3):accepted=self.feed([detection(490)]) or accepted
        self.assertTrue(accepted);self.assertEqual(self.t.logical_target_id,identity)
    def test_long_gap_weak_background_cannot_acquire_elsewhere(self):
        self.acquire();stamp=self.t.measurement_time
        for _ in range(100):self.feed([])
        for _ in range(15):self.assertFalse(self.feed([detection(490,.2)]))
        self.assertEqual(self.t.measurement_time,stamp)
    def test_prediction_gmc_applies_to_same_control_estimate(self):
        self.acquire();x=self.t.snapshot(self.t.last_result_time).cx
        def warp(image,detections):
            self.t.engine.gmc.last_warp=np.array([[1.,0.,25.],[0.,1.,0.]])
            return self.t.engine.gmc.last_warp
        self.t.engine.gmc.apply=warp;self.feed([])
        self.assertAlmostEqual(self.t.snapshot(self.t.last_result_time).cx,x+25)
    def test_missing_data_does_not_refresh_measurement_or_identity(self):
        self.acquire();identity=self.t.logical_target_id;stamp=self.t.measurement_time
        for _ in range(35):self.feed([])
        self.assertEqual(self.t.logical_target_id,identity);self.assertEqual(self.t.measurement_time,stamp)
        self.assertTrue(self.feed());self.assertEqual(self.t.logical_target_id,identity)

class ControlContinuityTests(unittest.TestCase):
    def controller(self):
        c=FollowController(tracker_backend='botsort');s=Settings(confidence=.8,new_track_confidence=.8)
        for i in range(20):
            t=10+i/30;c.observe([detection()],t,t+.01,i,(640,480),s)
            c.tick(t+.01,640,480,t,s,heading(t))
        return c,s
    def test_control_restart_keeps_target_but_resets_spacing_and_search(self):
        c,s=self.controller();target=c.tracker.target;c.spacing.phase='BRAKE'
        c.restart_control();self.assertIs(c.tracker.target,target)
        self.assertEqual(c.spacing.phase,'APPROACH');self.assertEqual(c.policy.state,'WAIT_TARGET')
        d=c.tick(10.65,640,480,10.64,s,heading(10.64))
        self.assertFalse(d['accepted']);self.assertEqual(d['state'],'WAIT_TARGET')
    def test_short_gap_never_searches_and_expired_prediction_never_moves_forward(self):
        c,s=self.controller()
        for i in range(20,25):
            t=10+i/30;c.observe([],t,t+.01,i,(640,480),s)
            d=c.tick(t+.01,640,480,t,s,heading(t))
            self.assertNotEqual(d['state'],'DIRECTIONAL_SEARCH')
        for i in range(25,55):
            t=10+i/30;c.observe([],t,t+.01,i,(640,480),s)
            d=c.tick(t+.01,640,480,t,s,heading(t))
        values,_=dance_command(d,t+.01,9)
        self.assertEqual(values[1:],(0.,0.,0.));self.assertFalse(d['prediction_forward_allowed'])
    def test_visible_pending_candidate_pauses_all_motion(self):
        c,s=self.controller();t=10+20/30
        c.observe([detection(255)],t,t+.01,20,(640,480),s)
        d=c.tick(t+.01,640,480,t,s,heading(t))
        self.assertTrue(d['candidate_pending']);self.assertFalse(d['edge_search']['active'])
        values,reason=dance_command(d,t+.01,9)
        self.assertEqual(values,(0.,0.,0.,0.));self.assertIn('verifying',reason)
    def test_dance_requires_measurement_after_enable(self):
        c,s=self.controller();d=c.tick(10.65,640,480,10.64,s,heading(10.64))
        values,reason=dance_command(d,10.66,10.64)
        self.assertEqual(values,(0.,0.,0.,0.));self.assertIn('measurement after',reason)

class SearchPriorityTests(unittest.TestCase):
    def test_candidate_pause_does_not_extend_search_deadline(self):
        r=RecoverySearch();s=Settings(search_yaw_degrees=20)
        for i in range(5):
            t=10+i*.04;r.observe(True,False,True,Box(200,180,260,220),Kinematics(),640,480,t,i,s)
            r.update(t,False,False,'APPROACH',False,s,heading(t))
        self.assertFalse(r.update(10.30,True,False,'APPROACH',False,s,heading(10.30))['active'])
        d=r.update(10.7,True,False,'APPROACH',False,s,heading(10.7));end=d['until'];self.assertTrue(d['active'])
        d=r.update(10.8,True,False,'APPROACH',False,s,heading(10.8),candidate_pending=True)
        self.assertFalse(d['active']);self.assertEqual(d['until'],end)
        d=r.update(end+.01,True,False,'APPROACH',False,s,heading(end+.01),candidate_pending=True)
        self.assertEqual(d['phase'],'DONE')

class OverlayTests(unittest.TestCase):
    def test_live_view_keeps_unmatched_yolo_candidates_visible(self):
        from types import SimpleNamespace
        c=FollowController(tracker_backend='botsort');s=Settings(confidence=.8,new_track_confidence=.8)
        for i in range(3):c.observe([detection()],10+i*.04,10+i*.04+.01,i,(640,480),s)
        c.observe([detection(255)],10.12,10.13,3,(640,480),s)
        decision=c.tick(10.13,640,480,10.12,s,heading(10.12))
        frame=SimpleNamespace(image=np.zeros((480,640,3),np.uint8),frame_id=3,decoded_at=10.12)
        with patch('follow_neo.overlay.draw_box') as draw:
            _,_,meta=render_bundle(frame,None,decision,'Live prediction',now=10.13)
        labels=[call.args[3] for call in draw.call_args_list]
        self.assertTrue(any('CANDIDATE' in label for label in labels))
        self.assertTrue(any('YOLO UNMATCHED' in label for label in labels))
        self.assertFalse(any('NO YOLO' in label for label in labels))
        self.assertEqual(len(meta['detections']),1)

import test_update9 as contracts

class LiveStateContracts(unittest.TestCase):
    """Run existing spacing/coast contracts on the new live backend as well."""
    def setUp(self):
        replacement=patch.object(contracts,'FollowController',lambda:FollowController(tracker_backend='botsort'))
        replacement.start();self.addCleanup(replacement.stop)

    def test_one_frame_gap_without_target_displacement_returns_directly(self):
        f=contracts.Fixture()
        f.step(missing=True,cx=.5)
        d=f.step(cx=.5)
        self.assertEqual(d['state'],'TRACK');self.assertGreater(d['intent']['forward'],0)

for name in (
    'test_short_empty_result_keeps_forward_and_real_timestamp',
    'test_long_loss_never_ends_continuous_dance_and_reacquires',
    'test_close_loss_and_clipping_keep_dance_selected',
    'test_distance_hold_preserves_yaw_and_can_clear_close_flag',
    'test_stale_video_is_wait_not_search_or_terminal',
    'test_duplicate_measurements_cannot_complete_reacquire',
    'test_rejected_distractor_does_not_continue_translation'):
    setattr(LiveStateContracts,name,getattr(contracts.ContinuousTests,name))

if __name__=='__main__':unittest.main()
