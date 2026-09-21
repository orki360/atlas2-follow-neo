import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from dataclasses import replace
import numpy as np
from follow_neo.types import Box, Detection, Settings, Kinematics, Intent
from follow_neo.tracker import BBoxTracker
from follow_neo.controller import FollowController
from follow_neo.policy import DeterministicTrackingPolicy, SmartTrackingController
from follow_neo.spacing import VisualSpacingController
from follow_neo.detector import OnnxDroneDetector


class TrackingTests(unittest.TestCase):
    def test_duplicate_result_does_not_confirm_or_refresh(self):
        t=BBoxTracker(); d=Detection(Box(100,100,200,160),.9)
        self.assertTrue(t.update([d],10,10.05,1))
        self.assertFalse(t.update([d],10,10.10,1))
        self.assertFalse(t.confirmed); self.assertEqual(t.measurement_time,10)
        t.update([d],10.1,10.15,2); self.assertTrue(t.confirmed)

    def test_far_distractor_is_rejected(self):
        t=BBoxTracker(); d=Detection(Box(100,100,200,160),.9)
        t.update([d],10,10.01,1); t.update([d],10.1,10.11,2)
        self.assertFalse(t.update([Detection(Box(900,500,990,570),.99)],10.2,10.21,3))

    def test_prediction_keeps_size_and_measurement_time(self):
        t=BBoxTracker()
        for i in range(4): t.update([Detection(Box(100+i*10,100,200+i*10,160),.9)],10+i*.1,10.02+i*.1,i+1)
        a=t.snapshot(10.32); b=t.snapshot(10.42)
        self.assertGreater(b.cx,a.cx)
        self.assertAlmostEqual(b.width,a.width,places=8)
        self.assertEqual(t.measurement_time,10.3)

    def test_controller_hard_stale_zeroes_intent_and_overlay(self):
        c=FollowController(); s=Settings(); d=[Detection(Box(800,270,920,340),.9)]
        for i in range(6):
            c.observe(d,10+i*.1,10.02+i*.1,i+1,(1280,720),s)
            out=c.tick(10.02+i*.1,1280,720,10+i*.1,s)
        self.assertEqual(out['state'],'TRACK')
        out=c.tick(12,1280,720,10.5,s)
        self.assertTrue(out['stale']); self.assertIsNone(out['track_box'])
        self.assertTrue(all(v==0 for v in out['intent'].values()))
        self.assertFalse(out['command_sent'])

    def test_edge_exit_then_timeout(self):
        p=DeterministicTrackingPolicy(); k=Kinematics(True,1240,350,100,70,180,0); s=Settings()
        for i in range(3): p.update(10+i*.1,k,1280,720,10+i*.1,i+1,Intent(.3,0,0,.2),s)
        p.update(10.7,k,1280,720,10.2,3,Intent(),s)
        self.assertEqual(p.state,'EDGE_RECOVERY')
        cmd=p.update(15,k,1280,720,10.2,3,Intent(),s)
        self.assertEqual(p.state,'ABORT_HOVER'); self.assertEqual(cmd,Intent())

    def test_spacing_confirm_requires_distinct_measurement(self):
        p=VisualSpacingController(); s=Settings()
        def update(now,mid,mt):
            return p.update(now,True,True,False,False,False,mid,mt,now-mt,.22,.22,0,0,Intent(forward=.2),s)
        cmd,_=update(10,1,10); self.assertEqual(p.phase,'CONFIRM_STOP'); self.assertEqual(cmd,Intent())
        update(10.1,1,10); self.assertEqual(p.phase,'CONFIRM_STOP')
        update(10.11,2,10.11); self.assertEqual(p.phase,'VISUAL_HOLD')

    def test_axis_reversal_brakes(self):
        c=SmartTrackingController()
        self.assertGreater(c.axis(.4,True,10,.7),0)
        self.assertEqual(c.axis(-.4,True,10.1,.7),0)
        self.assertEqual(c.axis(-.4,True,10.2,.7),0)
        self.assertLess(c.axis(-.4,True,10.3,.7),0)

    def test_model_letterbox_coordinates_and_nms(self):
        root=Path(__file__).resolve().parents[1]
        detector=OnnxDroneDetector(root/'models/best.onnx',cpu_threads=2)
        _,trans=detector._prepare(np.zeros((720,1280,3),np.uint8))
        raw=np.zeros((1,5,8400),np.float32)
        raw[0,:,0]=[320,320,100,50,.9]
        raw[0,:,1]=[322,320,100,50,.8]
        result=detector.postprocess(raw,trans)
        self.assertEqual(len(result),1)
        np.testing.assert_allclose(result[0]['box'],[540,310,740,410],atol=.01)


if __name__=='__main__': unittest.main(verbosity=2)
