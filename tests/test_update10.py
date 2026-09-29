"""Regression contracts for Update 10. No aircraft or network connections."""
import json,tempfile,threading,time,unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
from follow_neo.edge_search import EdgeYawSearch,yaw_override
from follow_neo.types import Box,Kinematics,Settings
from follow_neo.session import FollowSession
from follow_neo.gui import FollowLabWindow
from follow_neo.models import resolve_model,DEFAULT_MODEL
from follow_neo.detector import OnnxDroneDetector,DetectorSettings
from follow_neo.overlay import render_bundle,BLUE
from follow_neo.recording import VideoRecorder,recording_size
from follow_neo.video import Frame
from test_update9 import heading

ROOT=Path(__file__).resolve().parents[1]

class RecoveryTests(unittest.TestCase):
    def start(self,angle=90):
        self.search=EdgeYawSearch();self.s=Settings(search_yaw_degrees=angle)
        for i in range(12):
            t=10+i*.04;x=700+20*i
            self.search.observe(True,False,True,Box(x,320,x+80,360),Kinematics(),1280,720,t,i,self.s)
        self.now=10.60
        d=self.search.update(self.now,True,False,'APPROACH',False,self.s,heading(self.now,0))
        self.assertTrue(d['active']);return d
    def update(self,t,missing=True,yaw=0):
        return self.search.update(t,missing,False,'APPROACH',False,self.s,heading(t,yaw))
    def measure(self,t,fid=20,strong=True):
        self.search.observe(True,False,True,Box(950,320,1030,360),Kinematics(),1280,720,t,fid,self.s,strong=strong)
    def test_rejected_detection_cannot_end_search_or_renew_deadline(self):
        d=self.start();end=d['until']
        d=self.update(self.now+.05,missing=False)
        self.assertTrue(d['active']);self.assertEqual(d['until'],end)
        self.assertFalse(d['consumed'])
    def test_single_accepted_frame_pauses_then_resumes_same_budget(self):
        first=self.start();self.measure(10.65)
        paused=self.update(10.66,yaw=10)
        self.assertFalse(paused['active']);self.assertTrue(paused['verifying'])
        resumed=self.update(10.93,yaw=12)
        self.assertTrue(resumed['active']);self.assertEqual(resumed['until'],first['until'])
        self.assertEqual(resumed['started'],first['started']);self.assertEqual(resumed['source_time'],first['source_time'])
        self.assertAlmostEqual(resumed['angle_progress_degrees'],12)
    def test_three_distinct_measurements_finish_reacquire(self):
        self.start()
        for i in range(3):self.measure(10.65+i*.04,20+i);d=self.update(10.66+i*.04)
        self.assertFalse(d['active']);self.assertFalse(d['verifying']);self.assertEqual(d['reason'],'reacquired')
    def test_duplicate_and_weak_measurements_do_not_confirm(self):
        self.start();self.measure(10.65)
        for i in range(9):self.measure(10.65)
        self.assertEqual(self.search.verify_hits,1)
        self.measure(10.70,21,strong=False);self.measure(10.75,22,strong=False)
        self.assertEqual(self.search.verify_hits,1);self.assertTrue(self.update(10.76)['verifying'])
    def test_angle_and_deadline_apply_during_verification(self):
        self.start(angle=15);self.measure(10.65)
        self.assertEqual(self.update(10.67,yaw=16)['reason'],'search_angle_reached')
        first=self.start();self.measure(first['until']-.1)
        self.assertEqual(self.update(first['until']+.01)['reason'],'search_timeout')
    def test_live_settings_do_not_extend_started_budget(self):
        first=self.start();self.s=replace(self.s,search_yaw_degrees=180,edge_search_seconds=10)
        d=self.update(10.7)
        self.assertEqual(d['until'],first['until']);self.assertEqual(d['angle_degrees'],90)
        self.s=replace(self.s,search_yaw_degrees=0)
        self.assertFalse(self.update(10.8)['active'])
    def test_stationary_or_vertical_motion_does_not_invent_yaw(self):
        for horizontal in (False,True):
            search=EdgeYawSearch();s=Settings()
            for i in range(15):
                x=1000+(i%2)*2 if horizontal else 600;y=300+i*8
                search.observe(True,False,True,Box(x,y,x+80,y+40),Kinematics(),1280,720,10+i*.04,i,s)
            d=search.update(10.8,True,False,'APPROACH',False,s,heading(10.8))
            self.assertFalse(d['active'])

class ConfigTests(unittest.TestCase):
    def test_search_save_keeps_generation_result_and_reset_event(self):
        s=FollowSession.__new__(FollowSession);s.settings=Settings();s.settings_lock=threading.Lock()
        s.generation=12;s.reset_event=threading.Event();s.log=Mock();s.result=object();old=s.result
        s.configure_search(180,10,True)
        self.assertEqual(s.generation,12);self.assertFalse(s.reset_event.is_set());self.assertIs(s.result,old)
        self.assertEqual(s.settings.search_yaw_degrees,180);self.assertEqual(s.settings.edge_search_seconds,10)
        with self.assertRaises(ValueError):s.configure_search(181,10,True)
        self.assertEqual(s.settings.search_yaw_degrees,180)
    def test_gui_search_apply_does_not_apply_other_pending_settings(self):
        w=FollowLabWindow.__new__(FollowLabWindow);w.settings=Settings(confidence=.4)
        w.search_vars={k:Mock(get=Mock(return_value=v)) for k,v in [('search_yaw_degrees','180'),('edge_search_seconds','8')]}
        w.edge_search_on=Mock(get=Mock(return_value=True));w.session=Mock();w.save_config=Mock()
        w.mark_dirty=Mock();w.notice=Mock();w.canvas=Mock()
        w.search_auto=Mock(get=Mock(return_value=False))
        w.apply_search();w.session.configure_search.assert_called_once_with(180,8,True,False)
        self.assertFalse(w.settings.search_auto_duration)
        w.session.configure.assert_not_called();self.assertEqual(w.settings.confidence,.4)
    def test_search_focus_keeps_control_and_clears_movement_keys(self):
        from test_focus_transition import FocusTransitionTests
        fixture=FocusTransitionTests();fixture.setUp();w=fixture.window
        w.search_bar=object();w.manual.start_dance()
        fixture.focus(Mock(master=w.search_bar))
        self.assertTrue(w.manual.wanted);self.assertEqual(w.manual.mode,'DANCE');self.assertFalse(w.manual.keys)

class ModelTests(unittest.TestCase):
    def setUp(self):
        self.d=object.__new__(OnnxDroneDetector);self.d.settings=DetectorSettings();self.d.class_names={0:'NEO'}
        self.transform=dict(scale=1/3,pad_x=0,pad_y=140,source_w=1920,source_h=1080)
    def test_segmentation_coefficients_are_not_class_scores(self):
        raw=np.zeros((1,37,8400),np.float32);raw[0,:5,0]=[320,320,100,40,.9];raw[0,5:,0]=10
        raw[0,:5,1]=[320,320,100,40,.01];raw[0,5:,1]=1000
        proto=np.ones((1,32,160,160),np.float32)
        ds=self.d.postprocess(raw,self.transform,proto)
        self.assertEqual(len(ds),1);np.testing.assert_allclose(ds[0]['box'],[810,480,1110,600])
        self.assertTrue(ds[0]['segments']);json.dumps(ds,allow_nan=False)
        for polygon in ds[0]['segments']:
            for x,y in polygon:self.assertTrue(800<=x<=1120 and 470<=y<=610)
    def test_nms_preserves_corresponding_mask_coefficients(self):
        raw=np.zeros((1,37,8400),np.float32)
        raw[0,:5,0]=[320,320,100,40,.9];raw[0,5:,0]=-1
        raw[0,:5,1]=[320,320,100,40,.8];raw[0,5:,1]=1
        ds=self.d.postprocess(raw,self.transform,np.ones((1,32,160,160),np.float32))
        self.assertEqual(len(ds),1);self.assertFalse(ds[0]['segments'])
    def test_transposed_export_and_empty_output(self):
        self.assertEqual(self.d.postprocess(np.zeros((1,8400,37)),self.transform),[])
        with self.assertRaises(RuntimeError):self.d.postprocess(np.zeros((1,38,8400)),self.transform)
    def test_checksum_and_missing_model_refused(self):
        _,m=resolve_model(ROOT,DEFAULT_MODEL);self.assertEqual(m['task'],'segment')
        with tempfile.TemporaryDirectory() as path:
            root=Path(path);(root/'models').mkdir()
            (root/'models/models_update10.json').write_text(json.dumps({DEFAULT_MODEL:{'sha256':'bad'}}))
            with self.assertRaises(FileNotFoundError):resolve_model(root)
            (root/'models'/DEFAULT_MODEL).write_bytes(b'invalid')
            with self.assertRaisesRegex(RuntimeError,'checksum'):resolve_model(root)
    def test_both_real_models_load_with_existing_ort(self):
        for name in (DEFAULT_MODEL,'best.onnx'):
            d=OnnxDroneDetector(resolve_model(ROOT,name)[0],compute_mode='CPU')
            r=d.detect(np.zeros((720,1280,3),np.uint8));json.dumps(r,allow_nan=False)
            self.assertEqual(d.segmentation,name==DEFAULT_MODEL)

class RecordingTests(unittest.TestCase):
    def test_explicit_resolution_and_auto_policy(self):
        self.assertEqual(recording_size(1920,1080,'Auto',False),(1280,720))
        self.assertEqual(recording_size(1920,1080,'Auto',True),(1920,1080))
        self.assertEqual(recording_size(1920,1080,'1080p',False),(1920,1080))
    def test_hardware_unavailable_falls_back_with_visible_report(self):
        import av
        with tempfile.TemporaryDirectory() as path,patch('follow_neo.recording.probe_nvenc',return_value=(False,'test no GPU')):
            r=VideoRecorder(path,'Both videos',profile='Auto')
            for i in range(4):r.submit(Frame(i,10+i*.1,np.zeros((1080,1920,3),np.uint8)),None)
            r.wait();self.assertIsNone(r.error);self.assertEqual(r.encoder,'libx264');self.assertEqual(r.size,(1280,720))
            self.assertEqual(r.status()['fallback_reason'],'test no GPU')
            pts=[]
            for kind in ('clean','prediction'):
                with av.open(str(r.directory/f'{kind}.mp4')) as c:pts.append([float(f.time) for f in c.decode(video=0)])
            self.assertEqual(pts[0],pts[1]);np.testing.assert_allclose(pts[0],[0,.1,.2,.3],atol=1/90000)
    def test_blue_overlay_resizes_without_changing_source_or_metadata(self):
        frame=Frame(1,10,np.zeros((1080,1920,3),np.uint8))
        ds=[dict(box=[600,400,900,600],confidence=.9)]
        a={'result':dict(frame=frame,detections=ds,inference_ms=1),'selected_box':dict(zip(('x1','y1','x2','y2'),ds[0]['box'])),'decision':{}}
        image,_,meta=render_bundle(frame,a,None,'Analyzed frame',now=10.1,output_size=(1280,720))
        self.assertEqual(image.shape,(720,1280,3));self.assertTrue(np.all(image[267,450]==BLUE))
        self.assertFalse(frame.image.any());self.assertEqual(meta['selected_box']['x1'],600)

if __name__=='__main__':unittest.main()
