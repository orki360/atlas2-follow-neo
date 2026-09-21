import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import Mock
from follow_neo.gui import FollowLabWindow
from follow_neo.manual import ManualControl


class FocusTransitionTests(unittest.TestCase):
    def setUp(self):
        self.window=FollowLabWindow.__new__(FollowLabWindow)
        w=self.window
        w.root=Mock(); w.canvas=Mock(); w.control_bar=object()
        w.manual=ManualControl('127.0.0.1')
        w.manual.enabled=w.manual.wanted=True
        w.pressed={'w'}; w.manual.update({'w'})
        w.session=Mock(model_error=None,model_info={'ready':True})
        w.session.receiver.state='STREAMING'; w.session.receiver.host='127.0.0.1'
        w.stopping=False; w.notice=Mock(); w.show_control_indicator=Mock()
        w.dance_selected=False

    def focus(self,widget):
        self.window.root.focus_get.return_value=widget
        self.window.focus_out(Mock())
        self.window.root.after_idle.call_args.args[0]()

    def test_mouse_down_then_mouse_up_enters_dance(self):
        w=self.window
        self.focus(Mock(master=w.control_bar))  # Mouse-down changes focus first.
        self.assertTrue(w.manual.wanted)
        self.assertEqual(w.manual.keys,set())
        w.toggle_dance()  # Mouse-up invokes the button.
        self.assertEqual(w.manual.mode,'DANCE')
        self.assertEqual(w.dance_rejection,'')
        w.session.reset.assert_called_once()

    def test_dance_survives_focus_to_stop_button(self):
        w=self.window; w.manual.start_dance()
        w.dance_selected=True
        self.focus(Mock(master=w.control_bar))
        self.assertEqual(w.manual.mode,'DANCE')
        w.toggle_dance()
        self.assertEqual(w.manual.mode,'MANUAL')
        self.assertTrue(w.manual.wanted)

    def test_other_window_releases_control(self):
        self.window.manual.start_dance()
        self.focus(None)
        self.assertFalse(self.window.manual.wanted)
        self.assertEqual(self.window.manual.mode,'DANCE')

    def test_settings_field_releases_control(self):
        self.focus(Mock(master=object()))
        self.assertFalse(self.window.manual.wanted)


if __name__=='__main__': unittest.main()
