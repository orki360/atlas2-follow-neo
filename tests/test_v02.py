import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import tempfile
import time
import unittest
from unittest.mock import patch
import av
import numpy as np
from follow_neo.types import Box,Detection,Settings
from follow_neo.tracker import BBoxTracker
from follow_neo.controller import FollowController
from follow_neo.overlay import render_bundle,box_at_frame
from follow_neo.video import Frame
from follow_neo.recording import VideoRecorder
from follow_neo.compute import create_session,provider_spec

ROOT=Path(__file__).resolve().parents[1]


def decision(t=10.,width=1920,height=1080):
    return {'state':'TRACK','spacing_phase':'APPROACH','confirmed':True,'stale':False,
            'measurement_time':t,'measurement_id':1,'prediction_time':t+.1,
            'frame_size':[width,height],
            'kinematics':{'cx':1000.,'cy':550.,'width':200.,'height':100.,'vx':300.,'vy':0.}}


class NewBehaviorTests(unittest.TestCase):
    def test_native_resolution_tracker_is_scale_equivalent(self):
        a=BBoxTracker((640,480));b=BBoxTracker((1920,1080));scale=[3,2.25,3,2.25]
        for i in range(8):
            coords=[100+8*i,120+3*i,180+8*i,160+3*i]
            a.update([Detection(Box(*coords),.85)],10+.08*i,10.10+.08*i,i+1)
            b.update([Detection(Box(*(x*s for x,s in zip(coords,scale))),.85)],10+.08*i,10.10+.08*i,i+1)
            ka=a.snapshot(10.12+.08*i);kb=b.snapshot(10.12+.08*i)
            np.testing.assert_allclose([kb.cx/3,kb.cy/2.25,kb.width/3,kb.height/2.25,kb.vx/3,kb.vy/2.25],
                                       [ka.cx,ka.cy,ka.width,ka.height,ka.vx,ka.vy],rtol=1e-8,atol=1e-8)

    def test_measurement_fade_250_300_and_box_expiry(self):
        s=Settings();c=FollowController();detections=[Detection(Box(1200,400,1300,450),.9)]
        for i in range(5):
            c.observe(detections,10+i*.08,10.01+i*.08,i+1,(1920,1080),s)
            c.tick(10.01+i*.08,1920,1080,10+i*.08,s)
        source=10.32
        for age,scale in [(.25,1.),(.275,.5),(.300,0.),(.301,0.)]:
            out=c.tick(source+age,1920,1080,source+age,s)
            self.assertAlmostEqual(out['measurement_scale'],scale,places=6)
            if age>.300: self.assertIsNone(out['track_box'])
        self.assertFalse(out['command_sent'])

    def test_live_box_at_display_frame_not_control_now(self):
        frame=Frame(1,10.05,np.zeros((1080,1920,3),np.uint8));d=decision()
        box=box_at_frame(frame,d,10.1)
        self.assertAlmostEqual((box['x1']+box['x2'])/2,985)
        self.assertIsNone(box_at_frame(frame,d,10.301))
        self.assertIsNone(box_at_frame(Frame(0,9.99,frame.image),d,10.1))

    def test_snapshot_metadata_stays_bound_to_analyzed_result(self):
        frame=Frame(1,10.,np.zeros((1080,1920,3),np.uint8));d=decision();future=decision(10.05)
        future['measurement_id']=2
        analysis={'result':{'frame':frame,'detections':[],'inference_ms':10.},'selected_box':None,'decision':d}
        _,chosen,meta=render_bundle(frame,analysis,future,'Analyzed frame',now=10.1)
        self.assertEqual(chosen.frame_id,1);self.assertEqual(meta['decision']['measurement_id'],1)
        d['measurement_id']=99;self.assertEqual(meta['decision']['measurement_id'],1)

    def test_gpu_unavailable_is_explicit_and_auto_falls_back(self):
        with patch('follow_neo.compute.platform.system',return_value='Linux'):
            with self.assertRaisesRegex(RuntimeError,'GPU initialization failed'):
                create_session(ROOT/'models/best.onnx','GPU')
            session,info=create_session(ROOT/'models/best.onnx','Auto')
            self.assertEqual(info['label'],'CPU');self.assertTrue(info['fallback_reason'])
        self.assertEqual(provider_spec('Windows',['DmlExecutionProvider'],2)[1]['device_id'],'2')
        self.assertEqual(provider_spec('Darwin',['CoreMLExecutionProvider'])[1]['MLComputeUnits'],'CPUAndGPU')

    def test_full_hd_clean_and_prediction_recordings_share_timestamps(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder=VideoRecorder(folder,'Both videos')
            source=np.full((1080,1920,3),70,np.uint8)
            for i,offset in enumerate([0,.04,.13,.24]):
                recorder.submit(Frame(i+1,10+offset,source),decision())
            recorder.wait()
            self.assertIsNone(recorder.error);self.assertEqual(recorder.written,4)
            self.assertTrue(np.all(source==70))
            streams=[]
            for kind in ['clean','prediction']:
                with av.open(str(recorder.directory/f'{kind}.mp4')) as container:
                    frames=list(container.decode(video=0));streams.append(frames)
                    self.assertEqual(len(frames),4)
                    self.assertEqual((frames[0].width,frames[0].height),(1920,1080))
            self.assertEqual([f.pts*f.time_base for f in streams[0]],[f.pts*f.time_base for f in streams[1]])
            np.testing.assert_allclose([float(f.pts*f.time_base) for f in streams[0]],[0,.04,.13,.24],atol=1/90000)
            clean=streams[0][0].to_ndarray(format='bgr24');marked=streams[1][0].to_ndarray(format='bgr24')
            self.assertLess(np.abs(clean.astype(float)-70).mean(),3.)
            self.assertGreater(np.count_nonzero(np.abs(marked.astype(int)-clean.astype(int))>20),1000)
            rows=[json.loads(x) for x in (recorder.directory/'frames.jsonl').read_text().splitlines()]
            self.assertEqual([r['frame_id'] for r in rows],[1,2,3,4])
            report=json.loads((recorder.directory/'recording.json').read_text())
            self.assertEqual(report['state'],'SAVED')

    def test_recording_resolution_change_finalizes_existing_video(self):
        with tempfile.TemporaryDirectory() as folder:
            r=VideoRecorder(folder,'Clean video')
            r.submit(Frame(1,10,np.zeros((1080,1920,3),np.uint8)),None)
            r.submit(Frame(2,10.1,np.zeros((720,1280,3),np.uint8)),None)
            r.wait();self.assertIn('resolution changed',r.error)
            with av.open(str(r.directory/'clean.mp4')) as c:self.assertEqual(len(list(c.decode(video=0))),1)


if __name__=='__main__': unittest.main(verbosity=2)
