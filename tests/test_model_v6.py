"""Rectangular two-class model regressions; no aircraft/network access."""
import json
import threading
import time
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from follow_neo.detector import DetectorSettings, OnnxDroneDetector
from follow_neo.models import DEFAULT_MODEL, MODEL_NAMES, resolve_model
from follow_neo.session import FollowSession
from follow_neo.types import Settings
from follow_neo.video import Frame

ROOT = Path(__file__).resolve().parents[1]


class ModelV6Tests(unittest.TestCase):
    def setUp(self):
        self.detector = object.__new__(OnnxDroneDetector)
        self.detector.settings = DetectorSettings(input_width=960, input_height=544)
        self.detector.class_names = {0: 'NEO', 1: 'MINI PRO4'}
        self.detector.target_class_ids = {0}
        self.detector.mask_channels = 32
        self.raw = np.zeros((1, 38, 10710), np.float32)
        _, self.transform = self.detector._prepare(np.zeros((1080, 1920, 3), np.uint8))

    def row(self, index, neo, mini, coefficient=1):
        self.raw[0, :6, index] = [480, 272, 100, 40, neo, mini]
        self.raw[0, 6:, index] = coefficient

    def test_rectangular_letterbox_and_rgb(self):
        frame = np.full((1080, 1920, 3), [10, 20, 30], np.uint8)
        tensor, transform = self.detector._prepare(frame)
        self.assertEqual(tensor.shape, (1, 3, 544, 960))
        self.assertEqual(transform['pad_y'], 2)
        np.testing.assert_allclose(tensor[0, :, 2, 0], np.array([30, 20, 10])/255, atol=1e-7)
        self.assertTrue(tensor.flags.c_contiguous)

    def test_both_classes_displayed_but_only_neo_eligible_and_masks_correct(self):
        self.row(0, .8, .1)
        self.row(1, .7, .99, -1)
        self.row(2, .01, .02, 1000)
        detections = self.detector.postprocess(self.raw, self.transform, np.ones((1,32,136,240), np.float32))
        self.assertEqual(len(detections), 2)
        self.assertEqual([d['tracking_eligible'] for d in detections], [True, False])
        self.assertEqual(detections[1]['model_class_name'], 'MINI PRO4')
        self.assertFalse(detections[1]['segments'])
        detection = detections[0]
        self.assertEqual(detection['model_class_name'], 'NEO')
        self.assertEqual(detection['class_id'], 0)
        self.assertAlmostEqual(detection['confidence'], .8)
        np.testing.assert_allclose(detection['box'], [860, 500, 1060, 580])
        self.assertTrue(detection['segments'])
        for polygon in detection['segments']:
            for x, y in polygon:
                self.assertTrue(860 <= x <= 1060 and 500 <= y <= 580)
        json.dumps(detections, allow_nan=False)

    def test_transposed_output_and_mini_only(self):
        self.row(0, .6, .9)
        detections = self.detector.postprocess(self.raw.transpose(0,2,1), self.transform)
        self.assertEqual(len(detections), 1)
        self.assertFalse(detections[0]['tracking_eligible'])
        self.row(0, .9, .1)
        self.assertEqual(len(self.detector.postprocess(self.raw.transpose(0,2,1), self.transform)), 1)

    def test_class_aware_nms_when_no_filter(self):
        self.detector.target_class_ids = None
        self.row(0, .9, .1)
        self.row(1, .1, .8)
        self.row(2, .7, .1)
        detections = self.detector.postprocess(self.raw, self.transform)
        self.assertEqual([d['class_id'] for d in detections], [0, 1])

    def test_nonfinite_and_degenerate_boxes_are_removed(self):
        self.row(0, .9, .1)
        self.row(1, .9, .1)
        self.raw[0, 6, 0] = np.nan
        self.raw[0, 2, 1] = -1
        self.assertEqual(self.detector.postprocess(self.raw, self.transform), [])

    def test_contract_rejects_wrong_channels_and_end_to_end(self):
        d = self.detector
        d.input = SimpleNamespace(shape=[1,3,544,960])
        d.output = SimpleNamespace(shape=[1,38,10710])
        d.outputs = [d.output, SimpleNamespace(shape=[1,32,136,240])]
        d.metadata = {'end2end': 'False'}
        d._validate_contract()
        self.assertTrue(d.segmentation)
        d.output.shape = [1,37,10710]
        with self.assertRaises(RuntimeError): d._validate_contract()
        d.output.shape = [1,38,10710]
        d.metadata['end2end'] = 'True'
        with self.assertRaises(RuntimeError): d._validate_contract()

    def test_real_default_model_uses_manifest_target(self):
        path, manifest = resolve_model(ROOT)
        self.assertEqual(path.name, DEFAULT_MODEL)
        self.assertEqual(manifest['target_class_names'], ['NEO'])
        d = OnnxDroneDetector(path, compute_mode='CPU', target_class_names=manifest['target_class_names'])
        self.assertEqual((d.settings.input_width, d.settings.input_height), (960, 544))
        self.assertEqual(d.target_class_ids, {0})
        self.assertEqual(d.detect(np.zeros((1080,1920,3), np.uint8))['detections'], [])

    def test_previous_segmentation_model_still_runs(self):
        self.assertEqual(len(MODEL_NAMES), 3)
        path, _ = resolve_model(ROOT, 'yolo26s_seg_best_v1.onnx')
        d = OnnxDroneDetector(path, compute_mode='CPU')
        self.assertEqual(d.settings.input_width, 640)
        self.assertTrue(d.segmentation)
        json.dumps(d.detect(np.zeros((720,1280,3), np.uint8)), allow_nan=False)

    def test_live_worker_preserves_rectangular_input_and_neo_filter(self):
        session = object.__new__(FollowSession)
        session.model_path = ROOT / 'models' / DEFAULT_MODEL
        session.model_manifest = {'target_class_names': ['NEO']}
        session.compute_mode, session.device_id, session.output = 'CPU', -1, None
        session.log, session.receiver, session.result = Mock(), Mock(), Mock()
        session.settings = Settings()
        session.settings_lock = threading.Lock()
        session.stop_event = threading.Event()
        session.generation = session.inference_skips = session.inference_count = 0
        session.model_error = None
        frame = Frame(1, time.monotonic(), np.zeros((1080,1920,3), np.uint8))
        session.receiver.frames.wait_next.return_value = (frame, 1)
        detector = Mock()
        detector.input.shape, detector.output.shape = [1,3,544,960], [1,38,10710]
        detector.outputs = [detector.output]
        detector.class_names = self.detector.class_names
        detector.compute_info = {}
        def detect(image):
            self.assertEqual((detector.settings.input_width, detector.settings.input_height), (960,544))
            session.stop_event.set()
            return {'detections': []}
        detector.detect.side_effect = detect
        with patch('follow_neo.session.OnnxDroneDetector', return_value=detector) as constructor:
            session._infer()
        self.assertIsNone(session.model_error)
        self.assertEqual(session.inference_count, 1)
        self.assertEqual(constructor.call_args.kwargs['target_class_names'], ['NEO'])
        session.result.set.assert_called_once()


    def test_display_only_boxes_cannot_displace_target_budget(self):
        from dataclasses import replace
        self.detector.settings = replace(self.detector.settings, max_detections=1)
        self.row(0, .8, .1)
        self.row(1, .1, .99)
        detections = self.detector.postprocess(self.raw, self.transform)
        self.assertEqual([d['model_class_name'] for d in detections], ['NEO', 'MINI PRO4'])

    def test_overlay_shows_mini_without_selecting_it_in_both_modes(self):
        from follow_neo.overlay import render_bundle
        frame = Frame(1, 10., np.zeros((1080,1920,3), np.uint8))
        mini = dict(box=[100,100,200,200], confidence=.95, model_class_name='MINI PRO4', tracking_eligible=False)
        # Coincident boxes must not cause a display-only class to be marked selected.
        selected = dict(zip(('x1','y1','x2','y2'),mini['box']))
        decision = dict(confirmed=False, raw_detections=[], raw_detection_time=10.,
                        selected_measurement_box=mini['box'], display_detections=[mini], display_detection_time=10.)
        analysis = dict(result=dict(frame=frame,detections=[mini],inference_ms=1.), selected_box=selected, decision=decision)
        for mode in ('Analyzed frame', 'Live prediction'):
            with self.subTest(mode=mode), patch('follow_neo.overlay.draw_box') as draw:
                _, _, metadata = render_bundle(frame, analysis, decision, mode, now=10.1)
                self.assertEqual(len(metadata['detections']), 1)
                label = draw.call_args.args[3]
                self.assertIn('MINI PRO4', label)
                self.assertIn('DISPLAY ONLY', label)
                self.assertNotIn('SELECTED', label)
                self.assertNotIn('MEASURED', label)
            with patch('follow_neo.overlay.draw_box') as draw:
                render_bundle(frame, analysis, decision, mode, now=10.4)
                draw.assert_not_called()

    def test_live_controller_receives_only_neo_but_publishes_both_classes(self):
        neo = dict(box=[100,100,150,130], confidence=.9, model_class_name='NEO', tracking_eligible=True)
        mini = dict(box=[500,100,600,150], confidence=.99, model_class_name='MINI PRO4', tracking_eligible=False)
        for detections, expected_count in (([mini],0),([mini,neo],1)):
            with self.subTest(expected_count=expected_count):
                session = object.__new__(FollowSession)
                session.model_manifest = {'target_class_names': ['NEO']}
                session.settings, session.settings_lock = Settings(), threading.Lock()
                session.generation = 0
                session.stop_event, session.reset_event, session.control_reset_event = (threading.Event() for _ in range(3))
                session.model_error, session.control_provider, session.heading = None, None, Mock()
                session.inference_count, session.inference_skips = 1, 0
                session.receiver, session.result, session.decision, session.analysis, session.log = (Mock() for _ in range(5))
                frame = Frame(1, 10., np.zeros((1080,1920,3),np.uint8))
                session.receiver.frames.get.return_value = frame
                session.receiver.state = 'STREAMING'
                result = dict(frame=frame,detections=detections,generation=0,inference_ms=1.,preprocess_ms=1.,postprocess_ms=1.,completed_at=10.)
                session.result.get.return_value = result
                session.result.wait_next.return_value = (result,1)
                controller = Mock()
                controller.tracker.accepted = None
                def tick(*args, **kwargs):
                    session.stop_event.set()
                    return dict(state='SEARCH',spacing_phase='SEARCH',stale=False,edge_search={'active':False})
                controller.tick.side_effect = tick
                with patch('follow_neo.session.FollowController',return_value=controller), patch('follow_neo.session.heading_snapshot',return_value=({},10.)):
                    session._control()
                self.assertIsNone(session.model_error)
                received = controller.observe.call_args.args[0]
                self.assertEqual(len(received),expected_count)
                if received:self.assertEqual(received[0].box.x1,100)
                published = session.decision.set.call_args.args[0]
                self.assertEqual(published['display_detections'],detections)
                self.assertEqual(session.analysis.set.call_args.args[0]['result']['detections'],detections)


if __name__ == '__main__':
    unittest.main()
