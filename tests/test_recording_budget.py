"""Recording stays bounded and does not delay live frame publication."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import av
import cv2
import numpy as np
from follow_neo.recording import VideoRecorder
from follow_neo.session import FollowSession
from follow_neo.video import Frame, LatestValue
from follow_neo.video import VideoReceiver
from follow_neo.overlay import render_bundle
from follow_neo.models import V6_MODEL,DUAL_MODEL_SOURCE_NAMES,target_policy


def decision():
    """Current two-class display fixture, independent of removed Nano tests."""
    detections=[dict(box=[80,180,170,240],confidence=.9,class_id=0,class_name='neo'),
                dict(box=[330,180,440,240],confidence=.8,class_id=1,class_name='MINI PRO4')]
    return dict(raw_detections=detections,raw_detection_time=10.,frame_size=[640,480],
        selected_measurement_box=detections[0]['box'],selected_class_id=0,selected_class_name='neo',
        measurement_id=1,measurement_time=10.,measurement_age_ms=450.,prediction_time=10.45,
        confirmed=True,accepted=True,stale=False,state='TRACK',spacing_phase='APPROACH',
        target_policy=target_policy(V6_MODEL,DUAL_MODEL_SOURCE_NAMES[V6_MODEL]),
        intent=dict(yaw=.5,vertical=0.,roll=0.,forward=.5),
        intent_basis=dict(yaw=1.,vertical=1.,roll=1.,forward=1.))


class RecordingBudgetTests(unittest.TestCase):
    def test_sixty_hz_publication_keeps_original_frames_and_source_times(self):
        frames=[]
        receiver=VideoReceiver('offline.invalid',processing_fps=60,on_frame=frames.append)
        decoded=Mock(key_frame=False,to_ndarray=Mock(return_value=np.zeros((8,8,3),np.uint8)))
        with patch('follow_neo.video.time.monotonic') as clock:
            for i in range(120):
                stamp=10+i/60
                clock.return_value=stamp
                receiver._publish_decoded(decoded,stamp)
                self.assertTrue(receiver._publish_pending(stamp))
        self.assertEqual(receiver.count,120)
        self.assertEqual(receiver.processed_count,120)
        self.assertEqual([f.frame_id for f in frames],list(range(1,121)))
        self.assertEqual([f.decoded_at for f in frames],[10+i/60 for i in range(120)])

    def test_auto_samples_before_snapshot_preserving_paired_source_pts(self):
        source=np.zeros((1080,1920,3),np.uint8)
        d=decision();before=copy.deepcopy(d);supplier=Mock(return_value=d)
        with tempfile.TemporaryDirectory() as folder, patch('follow_neo.recording.probe_nvenc',return_value=(False,'test')):
            recorder=VideoRecorder(folder,'Both videos',profile='Auto')
            admitted=[]
            for i in range(8):
                if recorder.submit(Frame(i,10+i/30,source),decision_supplier=supplier):admitted.append(i)
            recorder.wait()
            self.assertEqual(admitted,[0,2,4,6]);self.assertEqual(supplier.call_count,4)
            self.assertEqual(recorder.status()['rate_skipped'],4)
            self.assertEqual(recorder.status()['target_fps'],15)
            self.assertEqual(recorder.status()['written'],4);self.assertIsNone(recorder.error)
            self.assertFalse(recorder.submit(Frame(10,11,source),decision_supplier=supplier))
            self.assertEqual(supplier.call_count,4)
            rows=[json.loads(line) for line in (recorder.directory/'frames.jsonl').read_text().splitlines()]
            self.assertEqual([r['frame_id'] for r in rows],admitted)
            pts=[]
            for kind in ('clean','prediction'):
                with av.open(str(recorder.directory/f'{kind}.mp4')) as video:
                    self.assertEqual((video.streams.video[0].width,video.streams.video[0].height),(1280,720))
                    pts.append([float(f.pts*f.time_base) for f in video.decode(video=0)])
            self.assertEqual(pts[0],pts[1])
            np.testing.assert_allclose(pts[0],[i/30 for i in admitted],atol=1/90000)
        self.assertEqual(d,before);self.assertFalse(source.any())

    def test_video_publication_uses_cached_decision_without_control_queries(self):
        session=FollowSession.__new__(FollowSession)
        session.control_provider=Mock(side_effect=AssertionError('control queried'))
        session.control_decision=Mock(side_effect=AssertionError('authority recomputed'))
        session.nano=Mock();session.recorder_lock=threading.Lock();session.decision=LatestValue();session.generation=3
        session.receiver=Mock(state='STREAMING');d=dict(decision(),generation=3)
        session.decision.set(d)
        class Sink:
            def submit(self,frame,*,decision_supplier):self.saved=decision_supplier()
        session.recorder=Sink()
        frame=Frame(1,10.5,np.zeros((480,640,3),np.uint8))
        session._record_frame(frame)
        self.assertIs(session.recorder.saved,d)
        session.nano.submit_frame.assert_called_once_with(frame);session.control_provider.assert_not_called()
        session.generation=4;self.assertIsNone(session.recording_decision())
        session.generation=3;session.receiver.state='WAITING_VIDEO'
        self.assertIsNone(session.recording_decision())

    def test_slow_encoder_discards_old_work_and_keeps_the_latest_frame(self):
        entered=threading.Event();release=threading.Event();encode=VideoRecorder._encode
        def delayed(output,source,pts):
            if pts==0:
                entered.set()
                if not release.wait(3):raise RuntimeError('test worker timeout')
            return encode(output,source,pts)
        with tempfile.TemporaryDirectory() as folder, patch('follow_neo.recording.probe_nvenc',return_value=(False,'test')), \
                patch.object(VideoRecorder,'_encode',staticmethod(delayed)):
            r=VideoRecorder(folder,'Clean video',profile='Auto')
            pixels=np.zeros((120,160,3),np.uint8)
            try:
                r.submit(Frame(0,10.,pixels));self.assertTrue(entered.wait(2))
                for i in range(1,6):self.assertTrue(r.submit(Frame(i,10+i*.1,pixels)))
                time.sleep(.18);release.set()
                deadline=time.perf_counter()+2
                while r.status()['written']<2 and time.perf_counter()<deadline:time.sleep(.01)
            finally:release.set();r.wait()
            self.assertIsNone(r.error);self.assertGreater(r.status()['backlog_skipped'],0)
            rows=[json.loads(line) for line in (r.directory/'frames.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['frame_id'],0);self.assertEqual(rows[-1]['frame_id'],5)
            self.assertEqual(rows[-1]['pts'],45000)

    def test_reused_resized_pixels_keep_clean_source_and_overlay_identical(self):
        rng=np.random.default_rng(3);source=rng.integers(0,256,(480,640,3),dtype=np.uint8)
        frame=Frame(1,10.45,source);original=source.copy();d=decision()
        base=cv2.resize(source,(320,240),interpolation=cv2.INTER_AREA);clean=base.copy()
        expected,_,meta=render_bundle(frame,None,d,'Live prediction',now=10.45,output_size=(320,240))
        actual,_,reused=render_bundle(frame,None,d,'Live prediction',now=10.45,output_size=(320,240),display_base=base)
        np.testing.assert_array_equal(actual,expected);self.assertEqual(reused,meta)
        np.testing.assert_array_equal(source,original);np.testing.assert_array_equal(base,clean)



if __name__=='__main__':unittest.main()
