"""Production evidence path: distinct frames, shared ROI approval and early BOOST."""
import unittest
from dataclasses import replace
from copy import deepcopy
from follow_neo.controller import FollowController
from follow_neo.recovery_search import RecoverySearch
from follow_neo.types import Settings,Box,Kinematics,Detection,roi_confirmation_valid
from follow_neo.roi import RoiSettings
import test_roi


def heading(t):return dict(time=t,yaw_deg=0.,source='AircraftAttitude')


class EarlyBoostTests(unittest.TestCase):
    def ready(self,side=1,settings=None):
        s=settings or Settings(search_yaw_degrees=180)
        r=RecoverySearch()
        for i,x in enumerate((1460,1500)):
            a,b=(x,x+100) if side==1 else (1920-x-100,1920-x)
            t=10+i*.05
            r.observe(True,False,True,Box(a,400,b,450),Kinematics(),1920,1080,t,i,s)
            r.observe_result(t,i,True)
            r.update(t,False,False,'APPROACH',False,s,heading(t))
        return r,s
    def step(self,r,s,t,**kw):
        return r.update(t,True,False,'APPROACH',False,s,heading(t),**kw)
    def test_two_motion_samples_and_two_new_absences_allow_early_boost_both_sides(self):
        for side in (-1,1):
            r,s=self.ready(side)
            self.assertTrue(r.box_edge)
            r.observe_result(10.12,2,False);r.observe_result(10.16,3,False)
            self.assertEqual(self.step(r,s,10.18)['phase'],'IDLE')
            d=self.step(r,s,10.21,bridge_reliable=True)
            self.assertEqual(d['phase'],'BOOST');self.assertEqual(d['direction'],side)
            self.assertLess(d['started']-r.source_time,.45)
    def test_ticks_and_duplicate_frames_cannot_satisfy_two_absences(self):
        r,s=self.ready();r.observe_result(10.16,2,False)
        for t in (10.21,10.25,10.30):
            r.observe_result(10.16,2,False)
            self.assertNotEqual(self.step(r,s,t)['phase'],'BOOST')
        self.assertEqual(r.absent_frames,1)
        r.observe_result(10.31,3,False)
        self.assertEqual(self.step(r,s,10.32)['phase'],'BOOST')
    def test_visible_pending_neo_resets_missing_count_and_blocks_entry(self):
        r,s=self.ready();r.observe_result(10.12,2,False);r.observe_result(10.18,3,True)
        self.assertEqual(r.absent_frames,0)
        self.assertNotEqual(self.step(r,s,10.23)['phase'],'BOOST')
        r.observe_result(10.24,4,False)
        self.assertNotEqual(self.step(r,s,10.25)['phase'],'BOOST')
        r.observe_result(10.29,5,False)
        self.assertEqual(self.step(r,s,10.30)['phase'],'BOOST')
    def test_live_edge_delay_changes_and_central_scan_grace(self):
        r,s=self.ready(settings=Settings(search_yaw_degrees=180,boost_edge_zone=.08))
        r.observe_result(10.16,2,False);r.observe_result(10.20,3,False)
        self.assertEqual(self.step(r,s,10.22)['phase'],'IDLE')
        s=replace(s,boost_edge_zone=.20,boost_entry_delay=.25)
        self.assertEqual(self.step(r,s,10.23)['phase'],'IDLE')
        self.assertEqual(self.step(r,s,10.31)['phase'],'BOOST')
        r,s=self.ready(settings=Settings(search_yaw_degrees=180,boost_edge_zone=.08))
        r.observe_result(10.35,2,False);r.observe_result(10.45,3,False)
        self.assertEqual(self.step(r,s,10.49)['phase'],'IDLE')
        self.assertEqual(self.step(r,s,10.51)['phase'],'SCAN')
    def test_heading_control_and_pending_gates_remain(self):
        for kwargs in ({'candidate_pending':True},{'control_available':False}):
            r,s=self.ready();r.observe_result(10.12,2,False);r.observe_result(10.18,3,False)
            self.assertNotEqual(self.step(r,s,10.21,**kwargs)['phase'],'BOOST')
        r,s=self.ready();r.observe_result(10.12,2,False);r.observe_result(10.18,3,False)
        self.assertNotEqual(r.update(10.21,True,False,'APPROACH',False,s,None)['phase'],'BOOST')


class SharedRoiConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.fixture=test_roi.RoiTests();self.fixture.setUp()
        original=self.fixture.detector.detect.side_effect
        def detect(image):
            result=original(image);result['detections'][0]['confidence']=.6;return result
        self.fixture.detector.detect.side_effect=detect
        self.c=FollowController(tracker_backend='botsort')
        self.settings=Settings(confidence=.7,new_track_confidence=.6,search_yaw_degrees=360)
    def feed(self,fid,when):
        r=self.fixture.process(fid,when=when)
        ds=[Detection(Box(*d['box']),d['confidence'],d.get('confirmation')) for d in r['detections'] if d['tracking_eligible']]
        candidates=[Detection(Box(*d['box']),d['confidence']) for d in r['detections'] if d['model_class_name']=='NEO' and d['confidence']>=self.fixture.settings.confidence]
        self.c.observe(ds,when,when+.08,fid,(1920,1080),self.settings,visible_candidates=candidates)
        return r,self.c.tick(when+.08,1920,1080,when,self.settings,heading(when))
    def test_full_90_roi_60_enters_track_on_second_frame_without_hidden_70_gate(self):
        a,d=self.feed(1,10.);self.assertFalse(d['confirmed']);self.assertNotEqual(d['state'],'TRACK')
        b,d=self.feed(2,10.1);self.assertEqual(d['state'],'TRACK')
        self.assertAlmostEqual(self.c.tracker.accepted.confidence,.6)
        self.assertEqual(b['detections'][0]['confirmation']['hits'],2)
    def test_return_from_scan_reuses_same_two_frame_proof(self):
        self.feed(1,10.);self.feed(2,10.1)
        saved=self.fixture.candidates;self.fixture.candidates=[]
        for i in range(3,13):_,d=self.feed(i,10+i*.1)
        self.assertEqual(d['state'],'DIRECTIONAL_SEARCH')
        self.fixture.candidates=saved
        _,d=self.feed(13,11.3);self.assertNotEqual(d['state'],'TRACK')
        _,d=self.feed(14,11.4);self.assertEqual(d['state'],'TRACK')
        self.assertEqual(d['edge_search']['phase'],'IDLE')
    def test_first_full_frame_can_seed_when_roi_unresolved(self):
        original=self.fixture.detector.detect.side_effect
        self.fixture.detector.detect.side_effect=lambda image:dict(detections=[])
        _,d=self.feed(1,10.);self.assertNotEqual(d['state'],'TRACK')
        self.fixture.detector.detect.side_effect=original
        _,d=self.feed(2,10.1);self.assertEqual(d['state'],'TRACK')
    def test_duplicate_expired_wrong_class_and_low_full_confidence_do_not_confirm(self):
        self.feed(1,10.)
        _,d=self.feed(1,10.);self.assertNotEqual(d['state'],'TRACK')
        self.fixture.names=['MINI PRO4'];_,d=self.feed(2,10.1);self.assertNotEqual(d['state'],'TRACK')
        self.fixture.names=['NEO'];self.fixture.candidates[0]['confidence']=.49
        _,d=self.feed(3,10.2);self.assertNotEqual(d['state'],'TRACK')
        self.fixture.candidates[0]['confidence']=.9
        a,_=self.feed(4,10.3);b,_=self.feed(5,10.4)
        det=b['detections'][0];proof=det['confirmation']
        self.assertFalse(roi_confirmation_valid(Detection(Box(*det['box']),.6,proof),10.5,6))
        fake=deepcopy(proof);fake['first_frame_id']=fake['frame_id']
        self.assertFalse(roi_confirmation_valid(Detection(Box(*det['box']),.6,fake),10.4,5))
    def test_exact_fifty_percent_and_configured_confirmation_gap(self):
        original=self.fixture.detector.detect.side_effect
        def detect(image):
            result=original(image);result['detections'][0]['confidence']=.5;return result
        self.fixture.detector.detect.side_effect=detect
        self.fixture.candidates[0]['confidence']=.5
        self.fixture.settings=replace(self.fixture.settings,max_gap_seconds=1.2)
        self.feed(1,10.)
        _,d=self.feed(2,11.)
        self.assertEqual(d['state'],'TRACK')
        self.assertAlmostEqual(self.c.tracker.accepted.confidence,.5)
        self.assertEqual(self.c.tracker.hits,2)
    def test_pending_roi_neo_blocks_search_but_roi_mini_does_not(self):
        for fid,x in enumerate((1460,1500,1540),1):
            self.fixture.candidates[0]['box']=[x,450,x+80,490]
            self.feed(fid,10.+fid*.05)
        saved=self.fixture.candidates;self.fixture.candidates=[]
        self.feed(4,10.22)
        self.fixture.candidates=saved
        original=self.fixture.detector.detect.side_effect
        self.fixture.detector.detect.side_effect=lambda image:dict(detections=[])
        _,d=self.feed(5,10.30)
        self.assertEqual(d['edge_search']['absent_frames'],0)
        self.assertTrue(d['edge_search']['candidate_visible'])
        self.assertFalse(d['edge_search']['active'])
        self.fixture.detector.detect.side_effect=original;self.fixture.names=['MINI PRO4']
        _,d=self.feed(6,10.35)
        self.assertFalse(d['edge_search']['candidate_visible'])
        self.assertEqual(d['edge_search']['absent_frames'],1)
    def test_twenty_percent_iou_keeps_identity_where_thirty_would_reset(self):
        self.feed(1,10.)
        self.fixture.candidates[0]['box']=[848,450,928,490]  # 32 / 128 = 25% overlap.
        result,d=self.feed(2,10.1)
        self.assertEqual(result['detections'][0]['roi_hits'],2)
        self.assertEqual(d['state'],'TRACK')

if __name__=='__main__':unittest.main()
