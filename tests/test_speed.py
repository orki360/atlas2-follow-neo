import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import time
import unittest
from unittest.mock import Mock
from follow_neo.manual import ManualControl
from follow_neo.gui import FollowLabWindow
from test_manual import Server, wait_for
from test_dance import decision


class SpeedTests(unittest.TestCase):
    def test_speed_scales_actual_manual_commands(self):
        server=Server(); c=ManualControl('127.0.0.1',server.connect)
        self.addCleanup(lambda:(c.stop(),c.thread.join(3)))
        c.start(); c.set_speed(.5); c.update({'d','w'}); c.enable()
        wait_for(lambda:'rc 0.0750 0.0075 0.0000 0.0000' in server.commands)
        c.set_speed(0); c.update({'d','w'})
        wait_for(lambda:'rc 0.0000 0.0000 0.0000 0.0000' in server.commands)
        c.set_speed(1); c.update({'d','w'})
        wait_for(lambda:'rc 0.1500 0.0150 0.0000 0.0000' in server.commands)

    def test_speed_scales_actual_dance_commands(self):
        server=Server(); c=ManualControl('127.0.0.1',server.connect)
        self.addCleanup(lambda:(c.stop(),c.thread.join(3)))
        c.start(); c.update(set()); c.enable()
        wait_for(lambda:c.enabled); c.start_dance(); c.set_speed(.5)
        time.sleep(.02); c.update(set(),decision())
        wait_for(lambda:'rc 0.0500 -0.0100 0.0150 0.0200' in server.commands)
        c.set_speed(0); c.update(set(),decision())
        wait_for(lambda:server.commands[-1]=='rc 0.0000 0.0000 0.0000 0.0000')
        self.assertEqual(c.mode,'DANCE')

    def test_bounds_and_invalid_speed(self):
        c=ManualControl('127.0.0.1')
        c.set_speed(3); self.assertEqual(c.speed,1)
        c.set_speed(-3); self.assertEqual(c.speed,0)
        with self.assertRaises(ValueError): c.set_speed(float('nan'))

    def test_gui_slider_updates_transport_and_label(self):
        w=FollowLabWindow.__new__(FollowLabWindow)
        w.manual=ManualControl('127.0.0.1'); w.speed_label=Mock(); w.session=None
        w.change_speed('37')
        self.assertEqual(w.manual.speed,.37)
        w.speed_label.set.assert_called_with('37%')

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
