"""BoT-SORT tests use real assignment/Kalman/GMC code, without aircraft I/O."""
import json
import unittest
import cv2
import numpy as np
from follow_neo.botsort_tracker import BoTSORTTracker, low_threshold
from follow_neo.camera_motion import CameraMotion
from follow_neo.controller import FollowController
from follow_neo.types import Box, Detection, Settings


def detection(x=200, score=.9, width=60):
    return Detection(Box(x, 170, x+width, 210), score)


class BoTSORTTests(unittest.TestCase):
    def setUp(self):
        self.tracker = BoTSORTTracker((640,480))
        self.image = np.zeros((480,640,3), np.uint8)

    def feed(self, i, ds=None, tracker=None, image=None):
        return (tracker or self.tracker).update(
            [detection()] if ds is None else ds, 10+i/30, 10+i/30+.01, i,
            image=self.image if image is None else image)

    def test_two_measurements_confirm_and_keep_id(self):
        self.assertTrue(self.feed(1)); self.assertFalse(self.tracker.confirmed)
        tid=self.tracker.target_id
        self.assertTrue(self.feed(2)); self.assertTrue(self.tracker.confirmed)
        self.assertEqual(tid,self.tracker.target_id)
        self.assertEqual(self.tracker.diagnostics()['backend'],'BoT-SORT')

    def test_weak_detection_continues_but_never_acquires(self):
        self.assertFalse(self.feed(1,[detection(score=.15)]))
        self.assertIsNone(self.tracker.target_id)
        self.feed(2); self.feed(3)
        self.assertTrue(self.feed(4,[detection(score=.15)]))
        self.assertTrue(self.tracker.measurement_weak)
        self.assertEqual(self.tracker.measurement_id,4)
        self.assertEqual(low_threshold(.25),.1)

    def test_threshold_boundary_is_not_dropped(self):
        self.feed(1); self.feed(2)
        self.assertTrue(self.feed(3,[detection(score=.25)]))
        self.assertTrue(self.feed(4,[detection(score=.1)]))

    def test_distractor_cannot_replace_locked_target(self):
        self.feed(1); self.feed(2); tid=self.tracker.target_id
        self.assertFalse(self.feed(3,[detection(490,.99)]))
        self.assertFalse(self.feed(4,[detection(490,.99)]))
        self.assertEqual(tid,self.tracker.target_id)
        self.assertEqual(self.tracker.measurement_id,2)
        self.assertTrue(self.feed(5,[detection(490,.99),detection()]))
        self.assertEqual(self.tracker.accepted.box,detection().box)

    def test_missing_and_duplicate_results_never_refresh_evidence(self):
        self.feed(1); self.feed(2); stamp=self.tracker.measurement_time
        self.assertFalse(self.feed(2)); self.assertFalse(self.feed(3,[]))
        self.assertEqual(self.tracker.measurement_time,stamp)
        self.assertIsNone(self.tracker.accepted)
        self.tracker.snapshot(10.4)
        self.assertEqual(self.tracker.measurement_time,stamp)

    def test_blank_camera_frames_are_valid(self):
        for i in range(1,8): self.assertTrue(self.feed(i))
        self.assertEqual(self.tracker.engine.gmc.status,'insufficient_features')

    def test_bad_coordinates_and_timestamp_are_rejected(self):
        self.assertFalse(self.feed(1,[detection(float('nan'))]))
        self.assertFalse(self.tracker.update([detection()],12,11,2))
        with self.assertRaises(ValueError):
            self.feed(2,image=np.zeros((200,200,3),np.uint8))

    def test_source_time_controls_prediction_interval(self):
        self.feed(1); self.feed(4)
        self.assertAlmostEqual(self.tracker.engine.kalman_filter._motion_mat[0,4],3.)

    def test_reset_releases_identity(self):
        self.feed(1); self.feed(2); self.tracker.reset()
        self.assertIsNone(self.tracker.target_id)
        self.assertTrue(self.feed(3,[detection(490)]))
        self.assertFalse(self.tracker.confirmed)

    def test_independent_trackers_do_not_reset_ids_of_existing_sessions(self):
        self.feed(1); other=BoTSORTTracker((640,480));self.feed(1,tracker=other)
        self.assertNotEqual(self.tracker.target_id,other.target_id)

    def test_camera_pan_keeps_small_target_id(self):
        rng=np.random.default_rng(12)
        image=rng.integers(0,256,(480,640,3),dtype=np.uint8)
        self.feed(1,[detection(200,width=20)],image=image)
        self.feed(2,[detection(200,width=20)],image=image)
        tid=self.tracker.target_id
        shifted=cv2.warpAffine(image,np.float32([[1,0,25],[0,1,0]]),(640,480))
        self.assertTrue(self.feed(3,[detection(225,width=20)],image=shifted))
        self.assertEqual(self.tracker.target_id,tid)
        self.assertEqual(self.tracker.engine.gmc.status,'applied')


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.c=FollowController(tracker_backend='botsort')
        self.s=Settings();self.image=np.zeros((480,640,3),np.uint8)

    def step(self,i,ds=None):
        t=10+i/30
        self.c.observe([detection()] if ds is None else ds,t,t+.01,i,(640,480),self.s,image=self.image)
        return self.c.tick(t+.01,640,480,t,self.s)

    def test_control_tracks_real_measurements_and_ages_predictions(self):
        for i in range(1,16):d=self.step(i)
        self.assertEqual(d['state'],'TRACK')
        self.assertEqual(d['tracking']['backend'],'BoT-SORT')
        stamp=d['measurement_time'];mid=d['measurement_id']
        d=self.step(16,[])
        self.assertFalse(d['accepted'])
        self.assertEqual((stamp,mid),(d['measurement_time'],d['measurement_id']))
        json.dumps(d,allow_nan=False)
        d=self.c.tick(12.,640,480,10.,self.s)
        self.assertTrue(d['stale']);self.assertIsNone(d['track_box'])
        self.assertTrue(all(v==0 for v in d['intent'].values()))

    def test_weak_evidence_does_not_grant_extended_forward(self):
        for i in range(1,16):self.step(i)
        for i in range(16,26):d=self.step(i,[detection(score=.15)])
        self.assertEqual(d['track_support'],'weak_yolo')
        self.assertFalse(d['prediction_quality']['stable'])
        d=self.step(26,[])
        self.assertFalse(d['prediction_forward_allowed'])

    def test_gap_does_not_reset_identity_or_adopt_distant_distractor(self):
        self.step(1);self.step(2);old=self.c.tracker.target_id
        for i in range(3,35):self.step(i,[])
        for i in range(35,42):d=self.step(i,[detection(450)])
        self.assertEqual(self.c.tracker.target_id,old)
        self.assertFalse(d['accepted'])
        self.assertTrue(d['confirmed'])


if __name__=='__main__':unittest.main()
