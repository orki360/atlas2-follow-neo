"""Fixed ONNX resolution, source coordinates and per-session model selection."""
import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from follow_neo.perception.detector import DetectorSettings, OnnxDroneDetector
from follow_neo.ui.window import FollowLabWindow
from follow_neo.perception.models import (DEFAULT_MODEL, MID_RES_MODEL, HIGH_RES_MODEL, DUAL_MODELS,
    DUAL_MODEL_SOURCE_NAMES, model_from_config, resolve_model, target_policy)
from follow_neo.runtime.session import FollowSession
from follow_neo.capture.video import Frame

ROOT = Path(__file__).resolve().parents[1]


def detector(name=HIGH_RES_MODEL, shape=(1,3,1088,1920), outputs=None,
             names=None, end2end='False'):
    session = Mock()
    session.get_inputs.return_value = [SimpleNamespace(shape=list(shape), name='images')]
    session.get_outputs.return_value = [SimpleNamespace(shape=list(s), name=f'output{i}')
        for i, s in enumerate(outputs or ((1,38,42840),(1,32,272,480)))]
    session.get_providers.return_value = ['TestProvider']
    session.get_modelmeta.return_value = SimpleNamespace(custom_metadata_map={
        'names': repr(names if names is not None else DUAL_MODEL_SOURCE_NAMES[name]),
        'end2end': end2end})
    with patch('follow_neo.perception.compute.create_session', return_value=(session, {})):
        return OnnxDroneDetector(ROOT/'models'/name, DetectorSettings(confidence=.31))


class ModelResolutionTests(unittest.TestCase):
    def test_registered_checksums_classes_and_saved_choice(self):
        for name in DUAL_MODELS:
            path, manifest = resolve_model(ROOT, name)
            self.assertEqual(path.name, name)
            self.assertEqual(manifest['tracking_class_id'], 0)
            self.assertTrue(target_policy(name, DUAL_MODEL_SOURCE_NAMES[name])['dance_allowed'])
            self.assertFalse(target_policy(name)['dance_allowed'])
            self.assertEqual(model_from_config({'schema_version':7, 'model_name':name}), name)
        self.assertEqual(model_from_config({}), DEFAULT_MODEL)

    def test_high_resolution_contract_and_aliases(self):
        d = detector()
        self.assertEqual((d.settings.input_width, d.settings.input_height), (1920,1088))
        self.assertEqual(d.settings.confidence, .31)
        self.assertEqual(d.class_names, {0:'neo', 1:'MINI PRO4'})
        self.assertEqual(d.model_class_names, {0:'NEO', 1:'MINI PRO4'})
        self.assertTrue(d.target_policy['dance_allowed'])

    def test_wrong_classes_shapes_and_end_to_end_fail_closed(self):
        cases = [dict(names={0:'MINI PRO4',1:'NEO'}), dict(names={0:'NEO'}),
            dict(names=DUAL_MODEL_SOURCE_NAMES[DEFAULT_MODEL]),
            dict(shape=(1,3,'height','width')), dict(shape=(1,3,1080,1920)),
            dict(outputs=((1,38,8400),(1,32,272,480))),
            dict(outputs=((1,38,42840),(1,32,160,160))), dict(end2end='True')]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(RuntimeError):
                detector(**case)

    def test_full_hd_pixels_are_preserved_with_four_rows_padding(self):
        d = detector()
        source = np.zeros((1080,1920,3), np.uint8)
        source[900:1010,1600:1770] = [17,121,243]
        tensor, transform = d._prepare(source)
        self.assertEqual(tensor.shape, (1,3,1088,1920))
        self.assertEqual((transform['scale'], transform['pad_x'], transform['pad_y']), (1,0,4))
        np.testing.assert_allclose(tensor[0,:,4:1084,:], source[:,:,::-1].transpose(2,0,1)/255., atol=1e-7)
        np.testing.assert_allclose(tensor[:,:,:4,:], 114/255., atol=1e-7)
        np.testing.assert_allclose(tensor[:,:,1084:,:], 114/255., atol=1e-7)

    def test_old_640_letterbox_is_unchanged(self):
        d = detector(DEFAULT_MODEL, (1,3,640,640), ((1,38,8400),(1,32,160,160)))
        tensor, transform = d._prepare(np.zeros((1080,1920,3), np.uint8))
        self.assertEqual(tensor.shape, (1,3,640,640))
        self.assertEqual((transform['scale'], transform['pad_x'], transform['pad_y']), (1/3,0,140))

    def test_960_contract_and_two_row_padding(self):
        import cv2
        d = detector(MID_RES_MODEL, (1,3,544,960), ((1,38,10710),(1,32,136,240)))
        self.assertEqual(d.class_names, {0:'neo', 1:'MINI PRO4'})
        self.assertTrue(d.target_policy['dance_allowed'])
        source = np.zeros((1080,1920,3), np.uint8)
        source[900:1010,1600:1770] = [17,121,243]
        tensor, transform = d._prepare(source)
        self.assertEqual(tensor.shape, (1,3,544,960))
        self.assertEqual((transform['scale'], transform['pad_x'], transform['pad_y']), (.5,0,2))
        expected = cv2.resize(source, (960,540), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_allclose(tensor[0,:,2:542,:], expected[:,:,::-1].transpose(2,0,1)/255., atol=1e-7)
        np.testing.assert_allclose(tensor[:,:,:2,:], 114/255., atol=1e-7)
        np.testing.assert_allclose(tensor[:,:,542:,:], 114/255., atol=1e-7)

    def test_960_boxes_and_masks_map_back_to_full_hd(self):
        d = detector(MID_RES_MODEL, (1,3,544,960), ((1,38,10710),(1,32,136,240)))
        _, transform = d._prepare(np.zeros((1080,1920,3), np.uint8))
        raw = np.ones((1,38,2), np.float32)
        raw[0,:6,0] = [850,502,40,20,.9,.1]
        raw[0,:6,1] = [150,52,40,20,.1,.95]
        proto = np.ones((1,32,136,240), np.float32)
        for output in (raw, raw.transpose(0,2,1)):
            results = d.postprocess(output, transform, proto)
            boxes = {r['class_id']:r['box'] for r in results}
            self.assertEqual(boxes, {0:[1660,980,1740,1020], 1:[260,80,340,120]})
            for result in results:
                self.assertTrue(result['segments'])
                x1,y1,x2,y2 = result['box']
                for polygon in result['segments']:
                    for x,y in polygon:
                        self.assertTrue(x1<=x<=x2 and y1<=y<=y2)
            json.dumps(results, allow_nan=False)

    def test_boxes_and_masks_return_to_full_hd_coordinates(self):
        d = detector()
        _, transform = d._prepare(np.zeros((1080,1920,3), np.uint8))
        raw = np.ones((1,38,2), np.float32)
        raw[0,:6,0] = [1700,1004,80,40,.9,.1]
        raw[0,:6,1] = [300,104,80,40,.1,.95]
        proto = np.ones((1,32,272,480), np.float32)
        for output in (raw, raw.transpose(0,2,1)):
            results = d.postprocess(output, transform, proto)
            boxes = {r['class_id']:r for r in results}
            self.assertEqual(boxes[0]['box'], [1660,980,1740,1020])
            self.assertEqual(boxes[1]['box'], [260,80,340,120])
            for result in results:
                self.assertTrue(result['segments'])
                x1,y1,x2,y2 = result['box']
                for polygon in result['segments']:
                    for x,y in polygon:
                        self.assertTrue(x1<=x<=x2 and y1<=y<=y2)
            json.dumps(results, allow_nan=False)

    def test_session_apply_retains_resolution_and_logs_verified_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('follow_neo.runtime.session.resolve_model', return_value=(ROOT/'models'/HIGH_RES_MODEL, {'name':HIGH_RES_MODEL})):
                s = FollowSession(folder, '127.0.0.1', model_name=HIGH_RES_MODEL)
            d = detector()
            seen = []
            def detect(image):
                seen.append((d.settings.input_width, d.settings.input_height, d.settings.iou_threshold))
                _, transform = d._prepare(image)
                if len(seen)==1:
                    s.configure(replace(s.get_settings(), confidence=.8, nms_iou=.6))
                else:
                    s.stop_event.set()
                return dict(detections=[], inference_ms=1., input_size=[1920,1088], input_transform=transform)
            d.detect = detect
            frame_ids=iter(range(1,21))
            def next_frame(*args):
                # The capture source advances independently of inference. With
                # a 60 FPS publication target, the 30 FPS inference gate may
                # skip a frame; tying its ID to detect() would repeat forever.
                time.sleep(1/30+.001)
                fid = next(frame_ids)
                return Frame(fid,time.monotonic(),np.zeros((1080,1920,3),np.uint8)), fid
            s.receiver.frames.wait_next = Mock(side_effect=next_frame)
            try:
                with patch('follow_neo.runtime.session.OnnxDroneDetector', return_value=d):
                    s._infer()
                self.assertIsNone(s.model_error)
                self.assertEqual(seen, [(1920,1088,.45),(1920,1088,.6)])
                self.assertEqual(s.result.get()['input_size'], [1920,1088])
                self.assertEqual(s.model_info['input'], [1,3,1088,1920])
                self.assertTrue(s.target_policy['dance_allowed'])
            finally:
                s.stop()
            self.assertIsNone(s.log.error)
            rows = [json.loads(line) for line in (s.output/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            ready = next(row for row in rows if row['event']=='model_ready')
            self.assertEqual(ready['model_class_names'], {'0':'NEO','1':'MINI PRO4'})

    def test_gui_permits_all_registered_dual_class_models(self):
        for name in DUAL_MODELS:
            w = FollowLabWindow.__new__(FollowLabWindow)
            w.session = Mock(model_path=Path(name))
            w.manual = Mock()
            w.dance_selected = False
            w.pressed = set()
            for attr in ('notice','root','canvas','log_control','show_control_indicator'):
                setattr(w,attr,Mock())
            w.toggle_dance()
            self.assertTrue(w.dance_selected)
            self.assertFalse(w.dance_rejection)
            w.manual.start_dance.assert_called_once()
