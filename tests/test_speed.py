import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import time
import unittest
from unittest.mock import Mock
from follow_neo.manual import ManualControl, shape_dance_axis
from follow_neo.gui import FollowLabWindow
from test_manual import Server, wait_for
from test_dance import decision


class SpeedTests(unittest.TestCase):
    def test_axis_limits_set_actual_manual_commands(self):
        server=Server(); c=ManualControl('127.0.0.1',server.connect)
        self.addCleanup(lambda:(c.stop(),c.thread.join(3)))
        limits={'yaw':.3,'vertical':.4,'roll':.5,'forward':.6}
        c.start(); c.set_axis_limits(limits); c.update({'d','w','right','up'}); c.enable()
        wait_for(lambda:'rc 0.3000 0.4000 0.5000 0.6000' in server.commands)
        c.set_axis_limits({axis:0 for axis in limits}); c.update({'d','w','right','up'})
        wait_for(lambda:'rc 0.0000 0.0000 0.0000 0.0000' in server.commands)

    def test_axis_limits_cap_actual_dance_commands(self):
        server=Server(); c=ManualControl('127.0.0.1',server.connect)
        self.addCleanup(lambda:(c.stop(),c.thread.join(3)))
        c.start(); c.update(set()); c.enable()
        wait_for(lambda:c.enabled); c.start_dance()
        c.set_axis_limits({'yaw':.05,'vertical':.01,'roll':.02,'forward':.03})
        time.sleep(.02); c.update(set(),decision(intent_basis={
            'yaw':.7,'vertical':.55,'roll':.45,'forward':.3}))
        wait_for(lambda:'rc 0.0071 -0.0003 0.0013 0.0040' in server.commands)
        c.set_axis_limits({axis:0 for axis in ('yaw','vertical','roll','forward')}); c.update(set(),decision())
        wait_for(lambda:server.commands[-1]=='rc 0.0000 0.0000 0.0000 0.0000')
        self.assertEqual(c.mode,'DANCE')

    def test_low_dance_caps_use_configured_authority_without_exceeding_it(self):
        self.assertEqual(shape_dance_axis(.8,.015),.8)
        self.assertEqual(shape_dance_axis(-.2,.015),-.2)
        self.assertEqual(shape_dance_axis(.049,.015),.049)
        self.assertAlmostEqual(shape_dance_axis(.4,.5),.4)

    def test_bounds_and_invalid_axis_limits(self):
        c=ManualControl('127.0.0.1')
        c.set_axis_limits({'yaw':3,'vertical':-3,'roll':.4,'forward':.5})
        self.assertEqual(c.axis_limits,{'yaw':1.,'vertical':0.,'roll':.4,'forward':.5})
        with self.assertRaises(ValueError):
            c.set_axis_limits({'yaw':float('nan'),'vertical':0,'roll':0,'forward':0})

    def test_gui_slider_updates_transport_and_label(self):
        w=FollowLabWindow.__new__(FollowLabWindow)
        w.manual=ManualControl('127.0.0.1'); w.session=None
        w.axis_limit_vars={axis:Mock() for axis in ('yaw','vertical','roll','forward')}
        w.axis_limit_labels={axis:Mock() for axis in w.axis_limit_vars}
        for value in w.axis_limit_vars.values(): value.get.return_value=20
        w.axis_limit_vars['yaw'].get.return_value=37
        w.change_axis_speed('yaw','37')
        self.assertEqual(w.manual.axis_limits,{'yaw':.37,'vertical':.2,'roll':.2,'forward':.2})
        w.axis_limit_labels['yaw'].set.assert_called_with('37.0%')
        w.axis_limit_vars['roll'].get.return_value=1.5
        w.change_axis_speed('roll','1.5')
        self.assertEqual(w.manual.axis_limits['roll'],.015)
        w.axis_limit_labels['roll'].set.assert_called_with('1.5%')

    def test_slider_focus_preserves_control_and_dance(self):
        w=FollowLabWindow.__new__(FollowLabWindow)
        w.root=Mock(); w.canvas=Mock(); w.speed_bar=object(); w.pressed={'w'}
        w.manual=ManualControl('127.0.0.1'); w.manual.wanted=True; w.manual.start_dance()
        w.root.focus_get.return_value=Mock(master=w.speed_bar)
        w.focus_out(Mock()); w.root.after_idle.call_args.args[0]()
        self.assertTrue(w.manual.wanted)
        self.assertEqual(w.manual.mode,'DANCE')
        self.assertEqual(w.pressed,set())


if __name__=='__main__': unittest.main()
