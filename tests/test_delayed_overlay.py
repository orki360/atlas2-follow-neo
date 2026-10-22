"""Slow detections remain visible without granting stale movement authority."""
import copy
import json
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np
from follow_neo.control.dance import dance_command, ZERO
from follow_neo.perception.models import DEFAULT_MODEL, DUAL_SOURCE_NAMES, target_policy
from follow_neo.presentation.overlay import render_bundle, BLUE, NANO_COLOR
from follow_neo.diagnostics.recording import VideoRecorder
from follow_neo.capture.video import Frame


def detections():
    return [dict(box=[80,180,170,240],confidence=.9,class_id=0,class_name='neo'),
            dict(box=[330,180,440,240],confidence=.8,class_id=1,class_name='MINI PRO4')]


def decision():
    return dict(raw_detections=detections(),raw_detection_time=10.,frame_size=[640,480],
        selected_measurement_box=detections()[0]['box'],selected_class_id=0,selected_class_name='neo',
        measurement_id=1,measurement_time=10.,measurement_age_ms=450.,prediction_time=10.45,
        confirmed=True,accepted=True,stale=False,state='TRACK',spacing_phase='APPROACH',
        target_policy=target_policy(DEFAULT_MODEL,DUAL_SOURCE_NAMES),
        intent=dict(yaw=.5,vertical=0.,roll=0.,forward=.5),
        intent_basis=dict(yaw=1.,vertical=1.,roll=1.,forward=1.))


class DelayedOverlayTests(unittest.TestCase):
    def test_analyzed_has_source_aligned_bot_and_historical_nano_boxes(self):
        frame=Frame(10,10.,np.zeros((480,640,3),np.uint8));d=decision()
        d['prediction_anchor']=dict(time=10.,cx=125.,cy=210.,width=90.,height=60.,vx=0.,vy=0.,model='botsort_cv')
        d['tracking']=dict(backend='BoT-SORT')
        # Current Nano is in the future and must never be drawn on frame 10.
        d['nano']=dict(mode='shadow',box=[200,200,300,250],visual_time=10.4,visual_active=True,blocked=False)
        historic=dict(mode='shadow',box=[80,180,170,240],visual_time=9.99,
                      historical=True,visual_active=True,blocked=True)
        a=dict(result=dict(frame=frame,detections=detections(),inference_ms=90.),decision=d,nano=historic)
        with patch('follow_neo.presentation.overlay.draw_box') as draw:
            _,_,meta=render_bundle(frame,a,d,'Analyzed frame',now=12.)
        colors=[call.args[2] for call in draw.call_args_list]
        self.assertIn(BLUE,colors);self.assertIn(NANO_COLOR,colors)
        self.assertEqual(meta['nano_box'],historic['box']);self.assertIsNotNone(meta['bot_box'])
        a.pop('nano')
        _,_,meta=render_bundle(frame,a,d,'Analyzed frame',now=12.)
        self.assertNotIn('nano_box',meta)

    def test_unverified_fresh_visual_box_is_visible_without_movement_authority(self):
        frame=Frame(10,10.4,np.zeros((480,640,3),np.uint8));d=decision()
        d['nano']=dict(mode='guarded',box=[80,180,170,240],visual_time=10.38,visual_active=True,
                       blocked=True,reason='semantic_expired')
        before=copy.deepcopy(d)
        with patch('follow_neo.presentation.overlay.draw_box') as draw:
            _,_,meta=render_bundle(frame,None,d,'Live prediction',now=10.4)
        self.assertEqual(meta['nano_box'],d['nano']['box'])
        self.assertIn('UNVERIFIED',draw.call_args_list[-1].args[3])
        self.assertEqual(d,before);self.assertEqual(dance_command(d,10.4,9.)[0],ZERO)

    def test_analyzed_frame_keeps_aligned_boxes_without_drawing_masks(self):
        source=Frame(1,10.,np.zeros((480,640,3),np.uint8))
        d=decision();ds=detections()
        ds[0]['segments']=[[[80,180],[170,180],[170,240],[80,240]]]
        analysis=dict(result=dict(frame=source,detections=ds,inference_ms=280.),
            selected_box=None,selected_class_id=None,decision=d)
        before=copy.deepcopy(d)
        for age in (.34,.65,2.):
            with patch('follow_neo.presentation.overlay.draw_box') as draw, patch('follow_neo.presentation.overlay.cv2.polylines') as contours:
                image,chosen,meta=render_bundle(Frame(99,10+age,source.image),analysis,
                    dict(measurement_id=99),'Analyzed frame',now=10+age)
            self.assertIs(chosen,source)
            self.assertEqual(len(draw.call_args_list),2)
            contours.assert_not_called()
            self.assertFalse(meta['masks_drawn'])
            self.assertEqual(meta['raw_detection_display'],'aligned')
            self.assertEqual(meta['detections'],ds)
            self.assertEqual(meta['decision']['measurement_id'],1)
            self.assertAlmostEqual(meta['raw_detection_age_ms'],age*1000)
            self.assertFalse(source.image.any())
            self.assertTrue(image.any())
        self.assertEqual(d,before)

    def test_live_delayed_measurements_have_class_age_and_no_current_selection(self):
        d=decision();before=copy.deepcopy(d)
        for age in (.344,.531,.672):
            frame=Frame(2,10+age,np.zeros((480,640,3),np.uint8))
            with patch('follow_neo.presentation.overlay.draw_box') as draw:
                _,_,meta=render_bundle(frame,None,d,'Live prediction',now=10+age)
            self.assertEqual(len(draw.call_args_list),2)
            for name,call in zip(('neo','MINI PRO4'),draw.call_args_list):
                self.assertIn(name,call.args[3]);self.assertIn('DELAYED',call.args[3])
                self.assertNotIn('SELECTED',call.args[3]);self.assertTrue(call.kwargs['dashed'])
            self.assertEqual(meta['raw_detection_display'],'delayed')
            self.assertEqual(meta['detections_drawn'],2)
            self.assertIsNone(meta['display_box'])
        self.assertEqual(d,before)

    def test_expired_future_or_different_resolution_results_are_hidden(self):
        for captured,now,size,expected in [(10.751,10.751,[640,480],'expired'),
            (10.2,10.751,[640,480],'expired'),(9.9,10.4,[640,480],'future_result'),
            (10.4,9.9,[640,480],'future_result'),(10.4,10.4,[1920,1080],'resolution_mismatch')]:
            d=decision();d.update(frame_size=size,confirmed=False)
            with patch('follow_neo.presentation.overlay.draw_box') as draw:
                _,_,meta=render_bundle(Frame(2,captured,np.zeros((480,640,3),np.uint8)),
                                      None,d,'Live prediction',now=now)
            draw.assert_not_called()
            self.assertEqual(meta['raw_detection_display'],expected)
            self.assertEqual(meta['detections'],[])

    def test_empty_result_clears_previous_detections_and_display_budget_is_separate(self):
        frame=Frame(2,10.4,np.zeros((480,640,3),np.uint8));d=decision()
        _,_,meta=render_bundle(frame,None,d,'Live prediction',now=10.4,stale_seconds=.3)
        self.assertEqual(meta['raw_detection_display'],'delayed')
        d['raw_detections']=[]
        _,_,meta=render_bundle(frame,None,d,'Live prediction',now=10.4)
        self.assertEqual(meta['raw_detection_display'],'empty');self.assertEqual(meta['detections'],[])

    def test_drawing_does_not_authorize_a_stale_measurement_for_dance(self):
        d=decision();before=copy.deepcopy(d)
        self.assertEqual(dance_command(d,10.45,9.)[0],ZERO)
        _,_,meta=render_bundle(Frame(2,10.45,np.zeros((480,640,3),np.uint8)),None,d,
                              'Live prediction',now=10.45)
        self.assertEqual(meta['detections_drawn'],2)
        self.assertEqual(d,before)
        self.assertEqual(dance_command(d,10.45,9.)[0],ZERO)

    def test_recording_contains_delayed_boxes_and_provenance_with_paired_timestamps(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder=VideoRecorder(folder,'Both videos',profile='720p')
            for i,stamp in enumerate((10.4,10.5)):
                recorder.submit(Frame(i,stamp,np.zeros((480,640,3),np.uint8)),decision())
            recorder.wait()
            self.assertIsNone(recorder.error)
            rows=[json.loads(line) for line in (recorder.directory/'frames.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),2)
            for row in rows:
                self.assertEqual(row['overlay']['detections_drawn'],2)
                self.assertEqual(row['overlay']['raw_detection_display'],'delayed')
                self.assertEqual(row['overlay']['raw_detection_time'],10.)
            timestamps=[]
            for kind in ('clean','prediction'):
                with av.open(str(recorder.directory/f'{kind}.mp4')) as container:
                    frames=list(container.decode(video=0))
                timestamps.append([f.pts*f.time_base for f in frames])
                image=frames[0].to_ndarray(format='bgr24')[170:245]
                green=(image[:,:,1].astype(int)>image[:,:,0].astype(int)+50)&(image[:,:,1].astype(int)>image[:,:,2].astype(int)+50)
                self.assertGreater(int(green.sum()),50) if kind=='prediction' else self.assertEqual(int(green.sum()),0)
            self.assertEqual(timestamps[0],timestamps[1])


if __name__=='__main__':unittest.main()
