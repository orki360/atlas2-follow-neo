"""Mode transitions invalidate diagnostic samples, not perception results."""
import unittest
from unittest.mock import patch
import numpy as np

from follow_neo.diagnostics import activity
from follow_neo.perception.camera_motion import CameraMotion
from follow_neo.perception.botsort_motion import BoTSORTTracker
from follow_neo.shared.types import Box,Detection
from test_model_resolution import detector


class DiagnosticEpochTests(unittest.TestCase):
    def setUp(self):
        previous=activity.enabled;activity.set_enabled(True)
        self.addCleanup(activity.set_enabled,previous)

    def test_normal_sample_uses_elapsed_clock(self):
        self.assertAlmostEqual(activity.duration_ms(activity.token(),10.,10.025),25.)

    def test_hard_does_not_read_diagnostic_clock(self):
        activity.set_enabled(False)
        with patch.object(activity.time,'perf_counter',side_effect=AssertionError('diagnostic clock')):
            self.assertEqual(activity.clock(),0.)
            self.assertIsNone(activity.duration_ms(activity.token(),0.))

    def test_exit_hard_cannot_turn_zero_into_process_uptime(self):
        activity.set_enabled(False);token=activity.token();start=activity.clock()
        activity.set_enabled(True)
        self.assertIsNone(activity.duration_ms(token,start))

    def test_complete_off_on_cycle_invalidates_old_enabled_sample(self):
        token=activity.token();activity.set_enabled(False);activity.set_enabled(True)
        self.assertIsNone(activity.duration_ms(token,100.,100.1))
        self.assertAlmostEqual(activity.duration_ms(activity.token(),100.,100.1),100.)

    def test_idempotent_mode_request_does_not_drop_valid_sample(self):
        token=activity.token();activity.set_enabled(True)
        self.assertTrue(activity.valid(token))

    def test_detector_preserves_detections_but_discards_every_crossing_duration(self):
        d=detector();expected=[dict(box=[1,2,5,6],confidence=.9,class_id=0)]
        image=np.zeros((2,2,3),np.uint8)
        for initial,stage in ((False,'prepare'),(False,'run'),(True,'run'),(True,'postprocess')):
            with self.subTest(initial=initial,stage=stage):
                activity.set_enabled(initial)
                def transition():
                    if initial:activity.set_enabled(False)
                    activity.set_enabled(True)
                def prepare(frame):
                    if stage=='prepare':transition()
                    return np.zeros((1,3,1,1)),{}
                def run(*args):
                    if stage=='run':transition()
                    return [None,None]
                def post(*args):
                    if stage=='postprocess':transition()
                    return expected
                with patch.object(d,'_prepare',side_effect=prepare),patch.object(d.session,'run',side_effect=run),patch.object(d,'postprocess',side_effect=post):
                    result=d.detect(image)
                    self.assertEqual(result['detections'],expected)
                    for key in ('preprocess_ms','inference_ms','postprocess_ms','total_ms'):
                        self.assertIsNone(result[key],key)
                    clean=d.detect(image) if not initial else None
                    if clean:
                        self.assertGreaterEqual(clean['inference_ms'],0.)
                        self.assertLess(clean['total_ms'],1000.)

    def test_gmc_same_transform_no_mixed_timings_and_next_frame_recovers(self):
        import cv2
        rng=np.random.default_rng(7);frame=rng.integers(0,255,(270,480,3),np.uint8)
        moved=cv2.warpAffine(frame,np.float32([[1,0,3],[0,1,2]]),(480,270))
        reference=CameraMotion();reference.apply(frame);expected=reference.apply(moved)
        motion=CameraMotion();motion.apply(frame)
        original=cv2.goodFeaturesToTrack
        for initial in (False,True):
            motion=CameraMotion();motion.apply(frame);activity.set_enabled(initial)
            def features(*args,**kwargs):
                if initial:activity.set_enabled(False)
                activity.set_enabled(True)
                return original(*args,**kwargs)
            with patch.object(cv2,'goodFeaturesToTrack',side_effect=features):
                actual=motion.apply(moved)
            np.testing.assert_allclose(actual,expected,atol=1e-6)
            self.assertEqual(motion.timing,{})
            motion.apply(frame)
            self.assertTrue(motion.timing)
            self.assertTrue(all(0<=x<1000 for x in motion.timing.values()))

    def test_botsort_discards_nested_sample_on_mode_change_without_resetting_target(self):
        tracker=BoTSORTTracker();detection=Detection(Box(100,100,150,130),.95,0,'neo')
        for i in (1,2,3):tracker.update([detection],i*.1,i*.1+.12,i)
        identity=tracker.logical_target_id;original=tracker.engine.update
        def update(*args,**kwargs):
            activity.set_enabled(False);activity.set_enabled(True)
            return original(*args,**kwargs)
        with patch.object(tracker.engine,'update',side_effect=update):
            self.assertTrue(tracker.update([detection],.4,.52,4))
        self.assertEqual(tracker.logical_target_id,identity)
        self.assertIsNone(tracker.update_ms);self.assertIsNone(tracker.engine_ms)
        tracker.update([detection],.5,.62,5)
        self.assertGreaterEqual(tracker.update_ms,0.)


if __name__=='__main__':unittest.main()
