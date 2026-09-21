import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import Mock
from follow_neo.control_status import control_indicator
from follow_neo.gui import FollowLabWindow


def control(**kwargs):
    obj=Mock(mode='MANUAL',wanted=True,enabled=True,status='connected',dance_status='Waiting for a new target')
    obj.thread.is_alive.return_value=True
    obj.stop_event.is_set.return_value=False
    for key,value in kwargs.items(): setattr(obj,key,value)
    return obj


class IndicatorTests(unittest.TestCase):
    def test_dance_waiting_tracking_stopped_are_distinct(self):
        obj=control(mode='DANCE')
        self.assertEqual(control_indicator(obj)[0],'NEO DANCE ON - WAITING')
        obj.dance_status='Tracking / VISUAL_HOLD'
        self.assertEqual(control_indicator(obj)[0],'NEO DANCE ON - TRACKING')
        obj.dance_status='Sequence stopped: restart Dance to reset'
        self.assertEqual(control_indicator(obj)[0],'NEO DANCE ON - STOPPED')

    def test_released_or_dead_control_shows_paused_selection(self):
        obj=control(mode='DANCE',wanted=False)
        self.assertEqual(control_indicator(obj)[0],'DANCE ON - CONTROL RELEASED')
        obj.wanted=True; obj.thread.is_alive.return_value=False
        self.assertEqual(control_indicator(obj)[0],'DANCE ON - CONTROL OFF')

    def test_rejection_persists_and_explains(self):
        self.assertEqual(control_indicator(None,'Enable first')[:2],('DANCE NOT STARTED','Enable first'))

    def test_manual_is_explicitly_not_dance(self):
        self.assertEqual(control_indicator(control())[0],'MANUAL CONTROL - DANCE OFF')

    def window(self):
        obj=FollowLabWindow.__new__(FollowLabWindow)
        obj.manual=control()
        obj.session=Mock(model_error=None,model_info={'ready':True})
        obj.session.receiver.state='STREAMING'
        obj.session.receiver.host=obj.manual.host='192.168.1.2'
        obj.stopping=False; obj.notice=Mock(); obj.root=Mock(); obj.canvas=Mock()
        obj.pressed=set(); obj.show_control_indicator=Mock()
        obj.dance_selected=False
        return obj

    def test_selection_available_without_control(self):
        obj=self.window(); obj.manual=None
        obj.toggle_dance()
        self.assertTrue(obj.dance_selected)
        self.assertTrue(any(call.args[0]=='dance_selection' for call in obj.session.record_control.call_args_list))
        obj.show_control_indicator.assert_called_once()

    def test_success_has_immediate_visual_and_audio_feedback(self):
        obj=self.window(); obj.manual.start_dance.return_value=True
        obj.toggle_dance()
        self.assertEqual(obj.dance_rejection,'')
        obj.show_control_indicator.assert_called_once()
        obj.root.bell.assert_called_once()

    def test_released_control_keeps_independent_toggle(self):
        obj=self.window(); obj.manual.wanted=False
        obj.toggle_dance()
        self.assertTrue(obj.dance_selected)
        obj.toggle_dance()
        self.assertFalse(obj.dance_selected)


if __name__=='__main__': unittest.main()
