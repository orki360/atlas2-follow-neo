"""Map display contracts; no application, sockets, video or aircraft is opened."""
import math
from types import SimpleNamespace
import tkinter as tk
import unittest

from follow_neo.mapping_panel import (
    DEFAULT_PITCH, DEFAULT_YAW, FLIGHT_REQUIREMENTS, MappingPanel,
    prepare_cloud, project_points, render_cloud_image,
)


class ProjectionTests(unittest.TestCase):
    def test_z_changes_screen_position_in_three_dimensional_view(self):
        a,b = project_points([(0,0,0),(0,0,1)],center=(0,0,0),radius=2)
        self.assertNotEqual(a[1],b[1])
        self.assertNotEqual(a[2],b[2])

    def test_rotation_and_zoom_transform_same_world_coordinates(self):
        original = project_points([(1,0,0)],yaw=0,pitch=0,center=(0,0,0),radius=2)[0]
        rotated = project_points([(1,0,0)],yaw=90,pitch=45,center=(0,0,0),radius=2)[0]
        zoomed = project_points([(1,0,0)],yaw=0,pitch=0,zoom=2,center=(0,0,0),radius=2)[0]
        self.assertAlmostEqual(rotated[0],450)
        self.assertNotEqual(original[1],rotated[1])
        self.assertAlmostEqual(zoomed[0]-450,2*(original[0]-450))

    def test_invalid_points_are_omitted(self):
        values=[None,[1,2],['no',0,1],[math.nan,0,1],[0,math.inf,1],[1,2,3]]
        self.assertEqual(len(project_points(values)),1)
        cloud=prepare_cloud({'points':[{'position':v} for v in values],
                             'camera_position':[0,math.inf,2],'target_position':[1,2,3]})
        self.assertEqual(cloud['points'],[(1.,2.,3.)])
        self.assertIsNone(cloud['camera_position'])
        self.assertEqual(cloud['target_position'],(1.,2.,3.))

    def test_sampling_is_bounded_deterministic_and_does_not_mutate(self):
        points=[{'id':i,'position':[i,2*i,3*i]} for i in range(10001)]
        snapshot={'points':points}
        first=prepare_cloud(snapshot,3000)
        second=prepare_cloud(snapshot,3000)
        self.assertEqual(first,second)
        self.assertEqual(len(first['points']),3000)
        self.assertEqual(first['total_points'],10001)
        self.assertEqual(first['points'][0],(0,0,0))
        self.assertEqual(first['points'][-1],(10000,20000,30000))
        self.assertEqual(len(points),10001)

    def test_invalid_view_parameters_raise(self):
        for arguments in ({'zoom':0},{'zoom':math.nan},{'pitch':math.inf},{'radius':0},{'size':(0,10)}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                project_points([(1,2,3)],**arguments)
        with self.assertRaises(ValueError):prepare_cloud({},0)

    def test_render_is_headless_and_handles_empty_invalid_and_full_scenes(self):
        scenes=[{}, {'points':[{'position':[math.nan,0,1]}]},
                {'points':[{'position':[0,0,0]},{'position':[1,1,1]}],
                 'keyframes':[{'position':[0,0,1]},{'position':[0,1,1]}],
                 'camera_position':[0,1,1],'target_position':[1,0,1],
                 'path':[[0,1,1],[1,1,1],[1,0,1]],'source':'simulation','units':'meters'}]
        for scene in scenes:
            with self.subTest(scene=scene):
                image=render_cloud_image(scene,(640,360))
                self.assertEqual(image.size,(640,360))
                self.assertEqual(image.mode,'RGB')
                self.assertGreater(len(image.getcolors(640*360)),5)


class MappingPanelTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root=tk.Tk()
            self.root.withdraw()
        except tk.TclError as exc:
            self.skipTest(f'Tk unavailable: {exc}')
        self.calls=[]
        self.panel=MappingPanel(self.root,
            on_start_live=lambda:self.calls.append('live'),
            on_start_simulation=lambda:self.calls.append('simulation'),
            on_stop=lambda:self.calls.append('stop'),
            on_save=lambda:self.calls.append('save'),
            on_calibrate=lambda:self.calls.append('calibrate'),
            on_choose_calibration=lambda:self.calls.append('choose'))
        self.panel.pack(fill='both',expand=True)

    def tearDown(self):
        if hasattr(self,'root'):
            self.root.destroy()

    def test_each_button_routes_only_its_callback(self):
        for button in (self.panel.start_live_button,self.panel.start_simulation_button,
                       self.panel.stop_button,self.panel.save_button,
                       self.panel.calibrate_button,self.panel.choose_calibration_button):
            button.invoke()
        self.assertEqual(self.calls,['live','simulation','stop','save','calibrate','choose'])

    def test_busy_disables_starts_but_stop_remains_available(self):
        self.panel.set_busy(True)
        self.assertIn('disabled',self.panel.estimate_calibration_button.state())
        self.panel.start_live_button.invoke()
        self.panel.start_simulation_button.invoke()
        self.panel.stop_button.invoke()
        self.assertEqual(self.calls,['stop'])
        self.panel.set_busy(False)
        self.assertNotIn('disabled',self.panel.estimate_calibration_button.state())
        self.panel.start_live_button.invoke()
        self.assertEqual(self.calls,['stop','live'])

    def test_snapshot_is_copied_and_modes_remain_explicit(self):
        source={'points':[{'position':[1,2,3]}],'source':'simulation','units':'meters'}
        self.panel.set_snapshot(source)
        source['points'][0]['position'][0]=999
        self.assertEqual(self.panel._snapshot['points'][0]['position'],(1.,2.,3.))
        self.assertIn('SIMULATION',self.panel.mode_badge.cget('text'))
        self.assertIn('synthetic',self.panel.mode_description.get())
        self.panel.set_snapshot({'source':'live'})
        self.assertIn('Arbitrary',self.panel.mode_badge.cget('text'))
        self.assertEqual(self.panel.mode_description.get(),FLIGHT_REQUIREMENTS)

    def test_status_calibration_and_camera_staleness_are_visible(self):
        self.panel.set_status({'state':'LOST','message':'No current pose','landmarks':120,
                               'keyframes':6,'pose_valid':False})
        self.assertIn('LOST',self.panel.status_text.get())
        self.assertIn('Pose unavailable',self.panel.status_text.get())
        self.assertIn('120',self.panel.status_text.get())
        self.panel.set_calibration_path('D:/camera.json')
        self.assertIn('D:/camera.json',self.panel.calibration_path.get())

    def test_drag_zoom_reset_and_destroy_cancel_queued_drawing(self):
        self.panel._drag_start(SimpleNamespace(x=10,y=20))
        self.panel._drag(SimpleNamespace(x=30,y=40))
        self.assertNotEqual(self.panel.yaw,DEFAULT_YAW)
        self.panel._wheel(SimpleNamespace(delta=120))
        self.assertGreater(self.panel.zoom,1)
        self.panel.reset_view()
        self.assertEqual((self.panel.yaw,self.panel.pitch,self.panel.zoom),
                         (DEFAULT_YAW,DEFAULT_PITCH,1))
        self.root.update_idletasks()
        self.assertIsNotNone(self.panel._photo)
        self.panel._schedule_redraw()
        self.panel.destroy()
        self.root.update_idletasks()
        self.assertIsNone(self.panel._redraw_id)

    def test_map_capacity_is_visible_without_invalidating_tracking(self):
        for reason in ('keyframe_limit', 'landmark_limit', 'keyframe_limit,landmark_limit'):
            with self.subTest(reason=reason):
                self.panel.set_status({'state':'running', 'mode':'live', 'pose_valid':True,
                                       'mapping_capacity_reason':reason, 'message':'Live sparse mapping'})
                self.assertIn('Map capacity reached',self.panel.status_text.get())
                self.assertIn(reason,self.panel.message_text.get())
                self.assertIn('Pose valid',self.panel.status_text.get())
        self.panel.set_status({'state':'running', 'pose_valid':True, 'mapping_capacity_reason':''})
        self.assertNotIn('Map capacity reached',self.panel.status_text.get())

    def test_mapping_log_path_and_failure_are_visible_separately_from_map(self):
        path = 'D:/maps/run/mapping.jsonl'
        self.panel.set_status({'state':'running', 'log_path':path, 'log_closed':False})
        self.assertIn(path,self.panel.log_text.get())
        self.panel.set_status({'state':'running', 'log_path':path,
                               'log_error':'disk full', 'message':'Map still processing'})
        self.assertIn('ERROR',self.panel.log_text.get())
        self.assertIn('disk full',self.panel.log_text.get())
        self.assertIn('Map still processing',self.panel.message_text.get())
        self.panel.set_status({'state':'stopped', 'log_path':path, 'log_closed':True})
        self.assertIn('(closed)',self.panel.log_text.get())
        self.assertNotIn('ERROR',self.panel.log_text.get())


if __name__=='__main__':
    unittest.main()
