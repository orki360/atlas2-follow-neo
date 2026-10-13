"""ROI gate, temporal identity and same-frame rendering contracts."""
import json
from pathlib import Path
import threading
import time
import unittest
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from follow_neo.roi import RoiSettings, RoiVerifier, crop_bounds
from follow_neo.overlay import render_roi, PURPLE, YELLOW, RED, ROI_NEO, ROI_MINI
from follow_neo.detector import DetectorSettings
from follow_neo.video import Frame
from follow_neo.session import FollowSession


def detection(box=(800,450,880,490), name='NEO', confidence=.9):
    return dict(box=list(box),confidence=confidence,class_id=0 if name=='NEO' else 1,
                model_class_name=name,tracking_eligible=name=='NEO')


class RoiTests(unittest.TestCase):
    def setUp(self):
        self.settings=RoiSettings()
        self.verifier=RoiVerifier()
        self.image=np.zeros((1080,1920,3),np.uint8)
        self.detector=Mock()
        self.detector.class_names={0:'NEO',1:'MINI PRO4'}
        self.detector.settings=DetectorSettings(input_width=960,input_height=544)
        self.original_settings=self.detector.settings
        self.names=['NEO']
        self.candidates=[detection()]
        self.calls=0
        def detect(image):
            i=self.calls%len(self.candidates);self.calls+=1
            d=self.candidates[i];bounds=crop_bounds(d['box'],self.image.shape,self.settings)
            x,y=bounds[:2]
            # Fake model consumes the actual view into the original frame.
            self.assertTrue(np.shares_memory(image,self.image))
            self.assertEqual(image.shape[:2],(self.settings.height,self.settings.width))
            result=detection([v-(x if j%2==0 else y) for j,v in enumerate(d['box'])],self.names[i%len(self.names)])
            result['segments']=[[[result['box'][0],result['box'][1]],[result['box'][2],result['box'][1]],[result['box'][2],result['box'][3]]]]
            return dict(detections=[result],inference_ms=2.,preprocess_ms=1.,postprocess_ms=1.)
        self.detector.detect.side_effect=detect

    def process(self, fid=1, when=None, generation=0, cancelled=lambda:False, **kwargs):
        frame=Frame(fid,10+fid*.1 if when is None else when,self.image)
        result=dict(detections=self.candidates,inference_ms=3.,preprocess_ms=1.,postprocess_ms=1.,total_ms=5.)
        return self.verifier.process(frame,result,self.detector,self.settings,generation=generation,cancelled=cancelled,**kwargs)

    def test_periodic_checks_only_after_two_real_confirmations(self):
        self.process(1,allow_cached=True,cache_confidence=.7)
        approved=self.process(2,allow_cached=True,cache_confidence=.7)
        self.assertEqual(approved['roi_inferences'],1)
        cached=self.process(3,allow_cached=True,cache_confidence=.7)
        self.assertEqual(cached['roi_inferences'],0)
        self.assertEqual(cached['roi_cached'],1)
        self.assertTrue(cached['detections'][0]['tracking_eligible'])
        self.assertEqual(cached['rois'][0]['hits'],2)
        self.assertEqual(cached['rois'][0]['detections'],[])
        self.assertEqual(cached['detections'][0]['roi_status'],'approved_cached')
        self.process(4,allow_cached=True,cache_confidence=.7)
        refreshed=self.process(5,allow_cached=True,cache_confidence=.7)
        self.assertEqual(refreshed['roi_inferences'],1)
        self.assertEqual(self.calls,3)

    def test_uncertain_tracking_or_low_full_confidence_forces_fresh_roi(self):
        self.process(1);self.process(2)
        self.assertEqual(self.process(3,allow_cached=False)['roi_inferences'],1)
        self.candidates[0]['confidence']=.6
        self.assertEqual(self.process(4,allow_cached=True,cache_confidence=.7)['roi_inferences'],1)
        self.settings=replace(self.settings,recheck_seconds=0.)
        self.process(5);self.process(6)
        self.assertEqual(self.process(7,allow_cached=True,cache_confidence=.5)['roi_inferences'],1)

    def test_cached_approval_is_revoked_by_wrong_class_refresh_or_missing_candidate(self):
        self.process(1);self.process(2)
        self.assertEqual(self.process(3,allow_cached=True,cache_confidence=.7)['roi_cached'],1)
        self.names=['MINI PRO4']
        result=self.process(5,allow_cached=True,cache_confidence=.7)
        self.assertFalse(result['detections'][0]['tracking_eligible'])
        self.names=['NEO']
        self.assertFalse(self.process(6,allow_cached=True,cache_confidence=.7)['detections'][0]['tracking_eligible'])
        self.assertTrue(self.process(7,allow_cached=True,cache_confidence=.7)['detections'][0]['tracking_eligible'])
        saved=self.candidates;self.candidates=[];self.process(8,allow_cached=True)
        self.candidates=saved
        self.assertFalse(self.process(9,allow_cached=True,cache_confidence=.7)['detections'][0]['tracking_eligible'])

    def test_cache_requires_unambiguous_temporal_identity(self):
        self.process(1);self.process(2)
        self.candidates=[detection(),detection((805,450,885,490))]
        result=self.process(3,allow_cached=True,cache_confidence=.7)
        self.assertEqual(result['roi_cached'],0)
        self.assertEqual(result['roi_inferences'],2)
        self.assertTrue(all(not d['tracking_eligible'] for d in result['detections']))

    def test_cache_requires_fresh_accepted_track(self):
        from follow_neo.roi import roi_cache_allowed
        d=dict(state='TRACK',confirmed=True,accepted=True,stale=False,prediction_time=10.,measurement_time=9.99)
        self.assertTrue(roi_cache_allowed(d,10.01))
        for changes in ({'state':'SCAN'},{'state':'REACQUIRE'},{'accepted':False},{'confirmed':False},
                        {'stale':True},{'prediction_time':9.7},{'measurement_time':9.7},
                        {'measurement_time':float('nan')}):
            self.assertFalse(roi_cache_allowed(dict(d,**changes),10.01))

    def test_two_distinct_frames_and_duplicate_is_not_confirmation(self):
        first=self.process()
        self.assertFalse(first['detections'][0]['tracking_eligible'])
        self.assertEqual(first['rois'][0]['hits'],1)
        self.assertIs(self.process(),first)
        self.assertEqual(self.calls,1)
        second=self.process(2)
        self.assertTrue(second['detections'][0]['tracking_eligible'])
        self.assertEqual(second['rois'][0]['id'],first['rois'][0]['id'])
        self.assertEqual(second['rois'][0]['hits'],2)
        self.assertEqual(second['detections'][0]['box'],self.candidates[0]['box'])
        self.assertEqual(second['rois'][0]['detections'][0]['box'],self.candidates[0]['box'])
        self.assertEqual(second['rois'][0]['detections'][0]['segments'][0][0],[800.,450.])
        json.dumps(second,allow_nan=False)

    def test_wrong_class_resets_and_is_display_only(self):
        self.process();self.names=['MINI PRO4']
        second=self.process(2)
        self.assertEqual(second['detections'][0]['model_class_name'],'MINI PRO4')
        self.assertFalse(second['detections'][0]['tracking_eligible'])
        self.assertEqual(second['rois'][0]['hits'],0)
        self.names=['NEO']
        self.assertFalse(self.process(3)['detections'][0]['tracking_eligible'])
        self.assertTrue(self.process(4)['detections'][0]['tracking_eligible'])

    def test_missing_or_ambiguous_local_detection_never_approves(self):
        self.process()
        for local in ([],[detection((440,250,520,290)),detection((440,250,520,290),'MINI PRO4')]):
            self.detector.detect.side_effect=lambda image:dict(detections=local)
            result=self.process(self.calls+2);self.calls+=1
            self.assertFalse(result['detections'][0]['tracking_eligible'])
            self.assertEqual(result['rois'][0]['hits'],0 if local else 1)
            self.assertNotIn('confirmation',result['detections'][0])

    def test_exact_entry_threshold_and_large_box_bypass(self):
        self.candidates=[detection((500,300,980,570))]
        self.assertEqual(self.process()['roi_inferences'],1)
        self.candidates=[detection((500,300,981,570))]
        result=self.process(2)
        self.assertEqual(result['roi_inferences'],0)
        self.assertTrue(result['detections'][0]['tracking_eligible'])
        self.assertEqual(result['detections'][0]['roi_status'],'skipped_size')
        self.candidates=[detection((500,300,980,571))]
        self.assertEqual(self.process(3)['roi_inferences'],0)

    def test_borders_preserve_exact_crop_and_small_source_is_blocked(self):
        self.assertEqual(crop_bounds([0,0,40,20],self.image.shape,self.settings),[0,0,960,540])
        self.assertEqual(crop_bounds([1880,1050,1920,1080],self.image.shape,self.settings),[960,540,1920,1080])
        self.image=np.zeros((400,600,3),np.uint8)
        self.candidates=[detection((100,100,150,130))]
        result=self.process()
        self.assertEqual(result['roi_inferences'],0)
        self.assertFalse(result['detections'][0]['tracking_eligible'])

    def test_each_neo_gets_serial_roi_and_own_count(self):
        self.candidates=[detection((400,300,480,340)),detection((1300,700,1380,740)),detection((200,100,280,140),'MINI PRO4')]
        # Detector side effect indexes only the NEO candidates that actually run.
        self.detector.detect.side_effect=None
        outputs=[]
        for d in self.candidates[:2]:
            x,y,_,_=crop_bounds(d['box'],self.image.shape,self.settings)
            outputs.append(dict(detections=[detection([v-(x if j%2==0 else y) for j,v in enumerate(d['box'])])]))
        thread_ids=[]
        def detect(image):
            thread_ids.append(threading.get_ident())
            return outputs[(len(thread_ids)-1)%2]
        self.detector.detect.side_effect=detect
        a=self.process();b=self.process(2)
        self.assertEqual(a['roi_inferences'],2)
        self.assertEqual([r['hits'] for r in b['rois']],[2,2])
        self.assertEqual(len(set(r['id'] for r in b['rois'])),2)
        self.assertEqual(set(thread_ids),{threading.get_ident()})
        self.assertFalse(b['detections'][2]['tracking_eligible'])

    def test_gap_generation_settings_and_missed_candidate_reset_count(self):
        self.process()
        self.assertFalse(self.process(2,when=12.)['detections'][0]['tracking_eligible'])
        self.assertFalse(self.process(3,when=12.1,generation=1)['detections'][0]['tracking_eligible'])
        self.settings=replace(self.settings,confidence=.6)
        self.assertFalse(self.process(4,when=12.2,generation=1)['detections'][0]['tracking_eligible'])
        saved=self.candidates;self.candidates=[];self.process(5,when=12.3,generation=1)
        self.candidates=saved
        self.assertFalse(self.process(6,when=12.4,generation=1)['detections'][0]['tracking_eligible'])

    def test_ambiguous_temporal_identity_does_not_share_confirmation(self):
        self.process()
        self.candidates=[detection((800,450,880,490)),detection((805,450,885,490))]
        result=self.process(2)
        self.assertEqual([r['hits'] for r in result['rois']],[1,1])
        self.assertTrue(all(not d['tracking_eligible'] for d in result['detections']))

    def test_stop_blocks_pending_candidates_and_restores_detector_settings(self):
        result=self.process(cancelled=lambda:True)
        self.assertEqual(result['roi_inferences'],0)
        self.assertFalse(result['detections'][0]['tracking_eligible'])
        self.assertIs(self.detector.settings,self.original_settings)

    def test_disabled_and_unsupported_model_keep_normal_detection(self):
        self.settings=replace(self.settings,enabled=False)
        self.assertTrue(self.process()['detections'][0]['tracking_eligible'])
        self.settings=replace(self.settings,enabled=True)
        self.detector.class_names={0:'drone'}
        self.assertEqual(self.process(2)['roi_inferences'],0)

    def test_settings_validation_and_config_roundtrip(self):
        self.assertEqual(RoiSettings.from_config({'roi':asdict(self.settings)}),self.settings)
        for fields in ({'max_box_width':961},{'width':20},{'confirmations':0},{'confirmations':1},{'confidence':float('nan')},{'match_iou':0.},{'height':540.5},{'recheck_seconds':-1},{'recheck_seconds':float('nan')}):
            with self.subTest(fields=fields),self.assertRaises(ValueError):replace(self.settings,**fields).validate()

    def test_render_uses_source_frame_and_requested_colors(self):
        result=self.process();result['frame']=Frame(7,10.,self.image)
        result['detections'].append(detection((100,100,150,130),'MINI PRO4'))
        with patch('follow_neo.overlay.draw_box') as draw:
            image=render_roi(result,(960,540))
        self.assertEqual(image.shape,(540,960,3))
        colors=[call.args[2] for call in draw.call_args_list]
        self.assertIn(ROI_NEO,colors);self.assertIn(RED,colors);self.assertIn(PURPLE,colors)
        rois=[call for call in draw.call_args_list if call.args[2]==PURPLE]
        self.assertTrue(rois[0].kwargs['dashed'])
        self.assertEqual(rois[0].kwargs['scale'],(.5,.5))
        self.assertFalse(self.image.any())

    def test_apply_settings_invalidates_old_generation(self):
        session=object.__new__(FollowSession)
        session.settings_lock=threading.Lock();session.generation=3
        session.result=Mock();session.analysis=Mock();session.decision=Mock();session.log=Mock()
        session.reset_event=threading.Event()
        session.configure_roi(self.settings)
        self.assertEqual(session.generation,4)
        self.assertTrue(session.reset_event.is_set())
        session.result.set.assert_called_once_with(None)


    def test_inference_worker_gates_before_publication(self):
        from follow_neo.types import Settings
        session=object.__new__(FollowSession)
        session.model_path=Path('model.onnx');session.model_manifest={'target_class_names':['NEO']}
        session.compute_mode='CPU';session.device_id=-1;session.output=None
        session.roi_settings=self.settings;session.settings=Settings()
        session.settings_lock=threading.Lock();session.stop_event=threading.Event()
        session.generation=session.inference_skips=session.inference_count=0
        session.model_error=None;session.log=Mock();session.result=Mock();session.receiver=Mock()
        start=time.monotonic()-.1
        frames=[Frame(i,start+i*.01,self.image) for i in (1,2)]
        def next_frame(*args):
            if frames:
                frame=frames.pop(0);return frame,frame.frame_id
            session.stop_event.set();return None,2
        session.receiver.frames.wait_next.side_effect=next_frame
        detector=self.detector
        detector.input.shape=[1,3,544,960];detector.output.shape=[1,38,10710]
        detector.outputs=[detector.output];detector.compute_info={}
        serial=[]
        def detect(image):
            serial.append(image.shape[:2])
            box=(800,450,880,490) if image.shape[1]==1920 else (440,250,520,290)
            return dict(detections=[detection(box)],inference_ms=1.,preprocess_ms=1.,postprocess_ms=1.)
        detector.detect.side_effect=detect
        with patch('follow_neo.session.OnnxDroneDetector',return_value=detector):session._infer()
        self.assertIsNone(session.model_error)
        outputs=[call.args[0] for call in session.result.set.call_args_list]
        self.assertEqual([o['detections'][0]['tracking_eligible'] for o in outputs],[False,True])
        self.assertEqual(serial,[(1080,1920),(540,960),(1080,1920),(540,960)])
        self.assertEqual([o['frame'].frame_id for o in outputs],[1,2])

    def test_model_error_restores_detector_settings(self):
        self.detector.detect.side_effect=RuntimeError('model failed')
        with self.assertRaisesRegex(RuntimeError,'model failed'):self.process()
        self.assertIs(self.detector.settings,self.original_settings)


    def test_analyzed_uses_exact_roi_boxes_labels_and_source_frame(self):
        from follow_neo.overlay import render_bundle
        import copy
        result=self.process();result['frame']=Frame(7,10.,self.image)
        # Local box differs from the acquisition box but refers to the same object.
        local=result['rois'][0]['detections'][0]
        local['box']=[802.,452.,878.,489.]
        mini=detection((1100,600,1180,640),'MINI PRO4')
        result['rois'][0]['detections'].append(mini)
        before=copy.deepcopy(result['detections'])
        analysis=dict(result=result,selected_box=None,decision={})
        newer=Frame(9,10.1,np.ones_like(self.image))
        with patch('follow_neo.overlay.draw_box') as draw:
            image,chosen,metadata=render_bundle(newer,analysis,None,'Analyzed frame',now=10.1,output_size=(960,540))
        self.assertIs(chosen,result['frame'])
        labels=[call.args[3] for call in draw.call_args_list]
        self.assertTrue(any('NEO FROM ROI' in label for label in labels))
        self.assertTrue(any('MINI PRO 4 FROM ROI' in label for label in labels))
        self.assertEqual(len(metadata['detections']),2)
        neo=next(d for d in metadata['detections'] if d['model_class_name']=='NEO')
        self.assertEqual(neo['box'],[802.,452.,878.,489.])
        self.assertEqual(result['detections'],before)
        self.assertEqual(image.shape,(540,960,3))

    def test_overlapping_rois_do_not_duplicate_same_detection(self):
        from follow_neo.overlay import roi_display_detections
        import copy
        result=self.process()
        duplicate=copy.deepcopy(result['rois'][0]);duplicate['id']=2
        result['rois'].append(duplicate)
        visible=roi_display_detections(result)
        self.assertEqual(len(visible),1)
        self.assertEqual(visible[0]['source'],'roi')


if __name__=='__main__':unittest.main()
