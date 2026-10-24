"""Offline geometry/identity gates for configurable camera compensation."""
from dataclasses import asdict, replace
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from follow_neo.shared.types import Settings, settings_from_config, Box, Detection
from follow_neo.shared.gmc import GMC_RESOLUTIONS
from follow_neo.perception.camera_motion import CameraMotion
from follow_neo.control.controller import FollowController
from follow_neo.diagnostics import activity


def texture():
    rng=np.random.default_rng(121)
    image=np.zeros((1080,1920,3),np.uint8)
    for x,y,value in zip(rng.integers(35,1880,1400),rng.integers(35,1040,1400),rng.integers(60,255,1400)):
        cv2.rectangle(image,(int(x),int(y)),(int(x+7),int(y+7)),(int(value),)*3,-1)
    return image


class GMCSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame=texture()
        cls.shift=np.array([[1.,0.,8.],[0.,1.,-4.]])
        cls.moved=cv2.warpAffine(cls.frame,cls.shift,(1920,1080))

    def setUp(self):
        self.old_activity=activity.enabled;activity.enabled=True
        self.old_threads=cv2.getNumThreads();cv2.setNumThreads(1)
        self.addCleanup(setattr,activity,'enabled',self.old_activity)
        self.addCleanup(cv2.setNumThreads,self.old_threads)

    def test_old_config_defaults_and_json_roundtrip(self):
        old=settings_from_config({'schema_version':8,'settings':{'confidence':.6}})
        self.assertEqual((old.gmc_resolution,old.gmc_max_corners),('480x270',400))
        for resolution in GMC_RESOLUTIONS:
            value=replace(old,gmc_resolution=resolution,gmc_max_corners=450).validate()
            loaded=settings_from_config(json.loads(json.dumps({'schema_version':8,'settings':asdict(value)})))
            self.assertEqual(loaded,value)

    def test_invalid_config_rejected_not_clamped(self):
        for points in (0,99,1001,400.5,400.,float('nan'),True,'400'):
            with self.subTest(points=points),self.assertRaises(ValueError):
                replace(Settings(),gmc_max_corners=points).validate()
        for resolution in ('0x0','423x544','960x544',None):
            with self.subTest(resolution=resolution),self.assertRaises(ValueError):
                replace(Settings(),gmc_resolution=resolution).validate()
        for points in (100,300,350,400,450,500,1000):
            replace(Settings(),gmc_max_corners=points).validate()

    def test_all_presets_return_source_pixel_translation(self):
        for resolution in GMC_RESOLUTIONS:
            with self.subTest(resolution=resolution):
                motion=CameraMotion(resolution,400)
                motion.apply(self.frame)
                warp=motion.apply(self.moved)
                self.assertEqual(motion.working_size,tuple(map(int,resolution.split('x'))))
                self.assertEqual(motion.status,'applied')
                np.testing.assert_allclose(warp,self.shift,atol=.45)
                self.assertLessEqual(len(motion.points),400)

    def test_mask_excludes_drone_in_original_coordinates(self):
        for resolution in GMC_RESOLUTIONS:
            with self.subTest(resolution=resolution):
                motion=CameraMotion(resolution,350)
                motion.apply(self.frame,np.array([[600,300,1200,800]]))
                w,h=motion.working_size
                original=motion.points.reshape(-1,2)/[w/1920,h/1080]
                inside=(original[:,0]>600)&(original[:,0]<1200)&(original[:,1]>300)&(original[:,1]<800)
                self.assertFalse(inside.any())

    def test_configuration_resets_pair_only_on_change(self):
        motion=CameraMotion('640x360',400);motion.apply(self.frame);previous=motion.previous
        motion.configure('640x360',400)
        self.assertIs(motion.previous,previous)
        motion.configure('480x270',300)
        self.assertIsNone(motion.previous);self.assertIsNone(motion.points)
        np.testing.assert_array_equal(motion.apply(self.frame),np.eye(2,3))
        self.assertEqual(motion.working_size,(480,270))
        self.assertLessEqual(len(motion.points),300)
        motion.configure('480x270',450)
        self.assertIsNone(motion.previous)

    def test_aspect_preserved_no_upscale_and_source_change_resets_pair(self):
        source=np.zeros((480,640,3),np.uint8)
        motion=CameraMotion('1920x1080',300);motion.apply(source)
        self.assertEqual(motion.working_size,(640,480))
        motion.configure('480x270',300);motion.apply(source)
        self.assertEqual(motion.working_size,(360,270))
        motion=CameraMotion();motion.apply(self.frame);motion.apply(self.moved)
        # Same working dimensions but a different source coordinate system.
        small=cv2.resize(self.frame,(1280,720))
        np.testing.assert_array_equal(motion.apply(small),np.eye(2,3))

    def test_controller_applies_gmc_changes_without_losing_identity(self):
        controller=FollowController(tracker_backend='botsort')
        frame=np.zeros((1080,1920,3),np.uint8)
        detection=Detection(Box(850,400,950,500),.95)
        settings=Settings()
        for i in (1,2,3):
            controller.observe([detection],i*.1,i*.1+.02,i,(1920,1080),settings,image=frame)
        tracker=controller.tracker;engine=tracker.engine;identity=tracker.logical_target_id
        self.assertTrue(tracker.confirmed)
        settings=replace(settings,gmc_resolution='960x540',gmc_max_corners=450)
        controller.observe([detection],.4,.42,4,(1920,1080),settings,image=frame)
        self.assertIs(tracker.engine,engine);self.assertEqual(tracker.logical_target_id,identity)
        self.assertTrue(tracker.confirmed)
        self.assertEqual((engine.gmc.resolution,engine.gmc.max_corners),('960x540',450))
        self.assertEqual(engine.gmc.working_size,(960,540))
        self.assertEqual(tracker.diagnostics()['camera_motion_config']['max_corners'],450)

    def test_hard_uses_same_geometry_without_timing_samples(self):
        normal=CameraMotion('480x270',300)
        normal.apply(self.frame);expected=normal.apply(self.moved)
        activity.enabled=False
        hard=CameraMotion('480x270',300)
        with patch('follow_neo.diagnostics.activity.time.perf_counter',side_effect=AssertionError('diagnostic clock')):
            hard.apply(self.frame);actual=hard.apply(self.moved)
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(hard.timing,{})


if __name__=='__main__':unittest.main()
