import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import time
import unittest
from unittest.mock import Mock
from follow_neo.dance import dance_command, ZERO
from follow_neo.manual import ManualControl
from test_manual import Server, wait_for


def decision(now=None, **changes):
    data=dict(prediction_time=time.monotonic() if now is None else now,
              state='TRACK',confirmed=True,accepted=True,stale=False,
              measurement_age_ms=20.,spacing_phase='VISUAL_HOLD',measurement_id=12,
              intent=dict(yaw=.1,vertical=-.02,roll=.03,forward=.04))
    data.update(changes)
    return data


class DanceTests(unittest.TestCase):
    def test_confirmed_tracking_preserves_axis_mapping(self):
        values,reason=dance_command(decision(10),10.05,9)
        self.assertEqual(values,(.1,-.02,.03,.04))

    def test_missing_stale_lost_stopped_and_invalid_decisions_are_zero(self):
        cases=[None, decision(8),decision(11),decision(10,stale=True),
               decision(10,confirmed=False),decision(10,accepted=False),
               decision(10,measurement_age_ms=251),decision(10,measurement_age_ms=float('nan')),
               decision(10,measurement_age_ms=240),
               decision(10,spacing_phase='STOPPED'),decision(10,spacing_phase='SEQUENCE_DONE'),
               decision(10,intent={'yaw':float('nan')}),
               decision(10,intent=dict(yaw=2,vertical=0,roll=0,forward=0))]
        cases += [decision(10,state=state) for state in ('SEARCH','COAST','EDGE_RECOVERY','REACQUIRE','ABORT_HOVER','ERROR')]
        for item in cases:
            with self.subTest(item=item): self.assertEqual(dance_command(item,10.05,9)[0],ZERO)
        self.assertEqual(dance_command(decision(10),10.05,10)[0],ZERO)

    def test_dance_enable_send_override_and_log(self):
        server=Server(); events=[]
        control=ManualControl('127.0.0.1',server.connect,lambda event,data:events.append((event,data)))
        self.addCleanup(lambda:(control.stop(),control.thread.join(3)))
        self.assertTrue(control.start_dance())
        control.start(); control.update(set()); control.enable()
        wait_for(lambda:control.enabled)
        self.assertTrue(control.start_dance())
        time.sleep(.02)  # Windows monotonic clocks can share a 15ms tick.
        control.update(set(),decision())
        wait_for(lambda:'rc 0.1000 -0.0200 0.0300 0.0400' in server.commands)
        control.update({'w'},decision())
        wait_for(lambda:'rc 0.0000 0.0150 0.0000 0.0000' in server.commands)
        self.assertEqual(control.mode,'DANCE')
        count=server.commands.count('rc 0.1000 -0.0200 0.0300 0.0400')
        control.update(set(),decision())
        wait_for(lambda:server.commands.count('rc 0.1000 -0.0200 0.0300 0.0400')>count)
        self.assertTrue(any(event=='flight_command' and data['mode']=='DANCE' for event,data in events))
        self.assertNotIn('takeoff',server.commands)

    def test_frozen_decision_stops_even_with_ui_heartbeat(self):
        server=Server(); control=ManualControl('127.0.0.1',server.connect)
        self.addCleanup(lambda:(control.stop(),control.thread.join(3)))
        control.start(); control.update(set()); control.enable()
        wait_for(lambda:control.enabled); control.start_dance()
        time.sleep(.02)
        snapshot=decision(); control.update(set(),snapshot)
        wait_for(lambda:'rc 0.1000 -0.0200 0.0300 0.0400' in server.commands)
        for _ in range(8):
            time.sleep(.04); control.update(set(),snapshot)
        wait_for(lambda:server.commands[-1]=='rc 0.0000 0.0000 0.0000 0.0000')
        self.assertEqual(control.mode,'DANCE')

    def test_land_keeps_selection_and_pauses_movement(self):
        control=ManualControl('127.0.0.1')
        control.enabled=control.wanted=True
        control.start_dance()
        self.assertTrue(control.request('land'))
        self.assertEqual(control.mode,'DANCE')
        self.assertTrue(control.motion_hold)

    def test_release_keeps_dance_and_requires_enable(self):
        control=ManualControl('127.0.0.1')
        control.enabled=control.wanted=True
        control.start_dance(); control.release()
        self.assertFalse(control.wanted)
        self.assertEqual(control.mode,'DANCE')

    def test_session_records_acknowledged_commands(self):
        from follow_neo.session import FollowSession
        session=FollowSession.__new__(FollowSession)
        session.log=Mock(); session.flight_commands_sent=0
        session.record_control('flight_command',{'server_acknowledged':True,'mode':'DANCE'})
        session.record_control('control_error',{'error':'disconnected'})
        self.assertEqual(session.flight_commands_sent,1)
        self.assertEqual(session.log.write.call_count,2)

    def test_gui_refuses_different_phone(self):
        from follow_neo.gui import FollowLabWindow
        window=FollowLabWindow.__new__(FollowLabWindow)
        window.manual=Mock(mode='MANUAL',host='192.168.1.2',enabled=True)
        window.session=Mock(model_error=None,model_info={'ready':True})
        window.session.receiver.state='STREAMING'
        window.session.receiver.host='192.168.1.3'
        window.stopping=False; window.notice=Mock()
        self.assertIsNone(window.dance_decision())


if __name__=='__main__': unittest.main()
