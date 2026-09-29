"""Initial acquisition must never remain locked to a discarded tentative ID."""
import unittest
import numpy as np
from follow_neo.botsort_tracker import BoTSORTTracker
from follow_neo.controller import FollowController
from follow_neo.control_status import target_status_text
from follow_neo.dance import dance_command
from follow_neo.types import Box,Detection,Settings


def detection(confidence=.9,x=200):return Detection(Box(x,180,x+60,220),confidence)


class SingleTargetTests(unittest.TestCase):
    def setUp(self):
        self.tracker=BoTSORTTracker((640,480));self.t=10.;self.i=0
        self.feed([])  # Reproduce acquisition after the engine's first frame.
    def feed(self,ds,dt=.04):
        self.t+=dt;self.i+=1
        return self.tracker.update(ds,self.t,self.t,self.i,.8,.8)
    def test_recorded_confidence_dip_confirms_new_id_without_reset(self):
        self.assertTrue(self.feed([detection(.884)]));first=self.tracker.target_id
        engine=self.tracker.engine
        for conf in (.774,.747):
            self.assertFalse(self.feed([detection(conf,202)]))
            self.assertFalse(self.tracker.confirmed);self.assertIsNone(self.tracker.target_id)
            self.assertEqual(self.tracker.diagnostics()['acquisition_strong_hits'],1)
        self.assertTrue(self.feed([detection(.895,205)]))
        self.assertTrue(self.tracker.confirmed);self.assertNotEqual(self.tracker.target_id,first)
        self.assertIs(self.tracker.engine,engine)
    def test_empty_frame_releases_id_but_preserves_brief_evidence(self):
        self.feed([detection()]);self.feed([])
        self.assertIsNone(self.tracker.target_id);self.assertEqual(self.tracker.hits,1)
        self.feed([detection(.95,203)]);self.assertTrue(self.tracker.confirmed)
    def test_evidence_expires_after_detector_gap(self):
        self.feed([detection()]);self.feed([],dt=.2)
        self.assertIsNone(self.tracker.acquisition)
        self.feed([detection()]);self.assertFalse(self.tracker.confirmed)
        self.feed([detection()]);self.assertTrue(self.tracker.confirmed)
    def test_weak_frames_cannot_keep_initial_evidence_forever(self):
        self.feed([detection()])
        for _ in range(20):self.feed([detection(.7)])
        self.assertFalse(self.tracker.confirmed)
        self.feed([detection()]);self.assertFalse(self.tracker.confirmed)
    def test_weak_only_never_acquires(self):
        for _ in range(25):self.assertFalse(self.feed([detection(.7)]))
        self.assertIsNone(self.tracker.target_id);self.assertFalse(self.tracker.confirmed)
    def test_two_strong_candidates_are_ambiguous(self):
        for _ in range(8):self.feed([detection(.95),detection(.96,450)])
        self.assertFalse(self.tracker.confirmed);self.assertIsNone(self.tracker.target_id)
        self.assertEqual(self.tracker.association_reason,'acquisition_ambiguous')
        self.assertIn('Ambiguous',target_status_text(dict(tracking=self.tracker.diagnostics())))
    def test_far_replacement_requires_its_own_evidence(self):
        self.feed([detection()]);self.feed([detection(.95,500)])
        self.assertFalse(self.tracker.confirmed);self.assertEqual(self.tracker.hits,1)
        self.feed([detection(.95,500)]);self.assertTrue(self.tracker.confirmed)
    def test_duplicate_frame_does_not_confirm(self):
        self.feed([detection()])
        self.assertFalse(self.tracker.update([detection()],self.t,self.t,self.i,.8,.8))
        self.assertFalse(self.tracker.confirmed);self.assertEqual(self.tracker.hits,1)
    def test_camera_motion_carries_initial_evidence(self):
        self.feed([detection()])
        def warp(image,detections):
            self.tracker.engine.gmc.last_warp=np.array([[1.,0.,180.],[0.,1.,0.]])
            return self.tracker.engine.gmc.last_warp
        self.tracker.engine.gmc.apply=warp
        self.feed([detection(.95,380)]);self.assertTrue(self.tracker.confirmed)
    def test_confirmed_identity_is_retained_after_gap(self):
        self.feed([detection()]);self.feed([detection()]);identity=self.tracker.logical_target_id
        for _ in range(25):self.feed([])
        self.assertTrue(self.tracker.confirmed);self.assertEqual(self.tracker.logical_target_id,identity)
    def test_unconfirmed_dance_holds_then_reaches_track_after_real_recovery(self):
        c=FollowController(tracker_backend='botsort');s=Settings(confidence=.8,new_track_confidence=.8)
        scores=[None,.884,.774,.747,.895,.9,.9,.9,.9,.9]
        for i,conf in enumerate(scores):
            t=10+i*.04;ds=[] if conf is None else [detection(conf)]
            c.observe(ds,t,t,i,(640,480),s);d=c.tick(t,640,480,t,s)
            values,reason=dance_command(d,t,9.)
            if i<4:self.assertEqual(values,(0.,0.,0.,0.))
        self.assertEqual(d['state'],'TRACK');self.assertTrue(any(values))
    def test_gui_reports_verifying_and_reacquiring(self):
        self.feed([detection()]);d=dict(tracking=self.tracker.diagnostics())
        self.assertIn('Verifying target',target_status_text(d))
        self.feed([],dt=.2);d=dict(tracking=self.tracker.diagnostics())
        self.assertEqual(target_status_text(d),'Reacquiring target')
