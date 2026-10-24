"""Deterministic pacing plus offline TCP/GUI gates; never connects to a phone."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock

from follow_neo.diagnostics import activity
from follow_neo.transport.cadence import SendCadence, checked_send_hz, send_hz_from_config
from follow_neo.transport.control_transport import RC_ACK_TIMEOUT
from follow_neo.transport.manual import ManualControl
from test_manual import Server, wait_for
from test_control_scan_fix import AsyncServer


class FakeTime:
    def __init__(self, overshoot=0.):
        self.now=100.;self.overshoot=overshoot;self.stopped=False;self.waits=[]
    def clock(self):return self.now
    def is_set(self):return self.stopped
    def wait(self,seconds):
        self.waits.append(seconds);self.now+=seconds+self.overshoot
        return self.stopped


class CadenceTests(unittest.TestCase):
    def test_late_wakeups_and_work_do_not_accumulate_drift(self):
        for hz in (20,25,30):
            with self.subTest(hz=hz):
                clock=FakeTime(.008);pacer=SendCadence(clock.clock);starts=[]
                for _ in range(100):
                    self.assertTrue(pacer.wait(clock,hz));starts.append(clock.now)
                    clock.now+=.006  # decision work and ACK included, not added
                self.assertAlmostEqual(starts[-1]-starts[0],99/hz+.008,places=8)
                self.assertTrue(all(t>0 for t in clock.waits))

    def test_slow_ack_skips_slots_instead_of_bursting(self):
        clock=FakeTime();pacer=SendCadence(clock.clock)
        pacer.wait(clock,20);clock.now+=.261
        pacer.wait(clock,20)
        self.assertAlmostEqual(clock.now,100.3)
        pacer.wait(clock,20)
        self.assertAlmostEqual(clock.now,100.35)

    def test_long_os_stall_does_not_replay_missed_slots(self):
        clock=FakeTime();pacer=SendCadence(clock.clock)
        pacer.wait(clock,20);clock.overshoot=.222
        pacer.wait(clock,20);self.assertAlmostEqual(clock.now,100.272)
        clock.overshoot=0.
        pacer.wait(clock,20);self.assertAlmostEqual(clock.now,100.3)
        pacer.wait(clock,20);self.assertAlmostEqual(clock.now,100.35)

    def test_rate_change_rebases_from_last_start_without_immediate_extra_send(self):
        clock=FakeTime();pacer=SendCadence(clock.clock)
        for hz in (20,30,25,20):
            previous=pacer.last_started
            pacer.wait(clock,hz)
            if previous is not None:self.assertAlmostEqual(clock.now-previous,1/hz)
            clock.now+=.004

    def test_stop_interrupts_wait_and_reset_discards_pre_pause_schedule(self):
        clock=FakeTime();pacer=SendCadence(clock.clock)
        pacer.wait(clock,20);clock.stopped=True
        self.assertFalse(pacer.wait(clock,20))
        clock.stopped=False;clock.now+=10.;pacer.reset()
        before=len(clock.waits);self.assertTrue(pacer.wait(clock,25))
        self.assertEqual(len(clock.waits),before)

    def test_validation_and_old_configuration_default(self):
        self.assertEqual(send_hz_from_config({}),20)
        for hz in (20,25,30):
            self.assertEqual(checked_send_hz(str(hz)),hz)
            self.assertEqual(send_hz_from_config({'tcp_send_hz':hz}),hz)
        for bad in (0,19,20.5,31,True,None,'',float('nan'),float('inf')):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):checked_send_hz(bad)
                self.assertEqual(send_hz_from_config({'tcp_send_hz':bad}),20)


class SenderIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.previous=activity.enabled;activity.set_enabled(True)
        self.addCleanup(activity.set_enabled,self.previous)

    def control(self,server,hz,events):
        control=ManualControl('127.0.0.1',server.connect,send_hz=hz,
            on_event=lambda event,data:events.append((event,data)))
        pulse_stop=threading.Event()
        def pulse():
            while not pulse_stop.wait(.02):control.update({'up'})
        thread=threading.Thread(target=pulse,daemon=True);thread.start()
        def close():
            pulse_stop.set();thread.join(1);control.stop();control.thread.join(3)
        self.addCleanup(close)
        control.update({'up'});control.enable();control.start()
        return control

    def test_all_rates_send_serialized_commands_and_log_selected_target(self):
        for hz in (20,25,30):
            with self.subTest(hz=hz):
                events=[];server=Server();c=self.control(server,hz,events)
                def flights():return [d for e,d in events if e=='flight_command' and d['command']=='rc']
                wait_for(lambda:len(flights())>=6)
                self.assertTrue(all(d['tcp_send_hz']==hz for d in flights()))
                self.assertTrue(all(d['tcp_send_period_ms']==1000/hz for d in flights()))
                self.assertEqual(c.control_snapshot()['send_hz'],hz)
                self.assertFalse(c.communication_paused)
                attempts=[d for e,d in events if e=='control_command_attempt']
                self.assertTrue(all('send_attempt_perf_counter' in d for d in attempts))
                c.stop();c.thread.join(3)

    def test_rate_change_does_not_change_authority_or_accept_invalid_value(self):
        c=ManualControl('offline.invalid');c.wanted=c.enabled=True;c.mode='DANCE'
        original=(c.wanted,c.enabled,c.mode,c.dance_started,c.decision)
        c.set_send_hz(30)
        self.assertEqual((c.wanted,c.enabled,c.mode,c.dance_started,c.decision),original)
        with self.assertRaises(ValueError):c.set_send_hz(100)
        self.assertEqual(c.send_hz,30)

    def test_250ms_timeout_and_no_motion_pipeline_at_every_rate(self):
        self.assertEqual(RC_ACK_TIMEOUT,.25)
        for hz in (20,25,30):
            with self.subTest(hz=hz):
                events=[];server=AsyncServer(True);c=self.control(server,hz,events)
                self.assertTrue(server.motion.wait(1))
                wait_for(lambda:c.communication_paused)
                self.assertIn('0.25s',c.pause_reason)
                self.assertTrue(server.zero.wait(1))
                movements=[line for _,line in server.commands if line.startswith('rc ')
                           and any(map(float,line.split()[1:]))]
                self.assertEqual(len(movements),1)
                self.assertFalse(any(e=='flight_command' for e,_ in events))
                server.block=False
                for _ in list(server.withheld):server.server.sendall(b'success\r\n')
                c.stop();c.thread.join(3)

    def test_hard_uses_selected_rate_without_command_diagnostic_events(self):
        activity.set_enabled(False)
        events=[];server=Server();c=self.control(server,30,events)
        wait_for(lambda:len([cmd for cmd in server.commands if cmd.startswith('rc 0.')])>=6)
        self.assertEqual(c.send_hz,30)
        self.assertFalse(any(e=='flight_command' or
            e=='control_command_attempt' and d['command'].startswith('rc ') for e,d in events))
        self.assertFalse(any('send_attempt_perf_counter' in d for _,d in events))
        self.assertEqual(c.transport_status,{})
        c.stop();c.thread.join(3)


class SendRateGUITests(unittest.TestCase):
    def test_selection_requires_apply_and_survives_restart_without_resetting_tracking(self):
        import tkinter as tk
        from follow_neo.ui.window import FollowLabWindow
        with tempfile.TemporaryDirectory() as directory:
            root=tk.Tk();root.withdraw()
            try:
                window=FollowLabWindow(root,directory)
                settings=window.settings
                self.assertEqual(window.tcp_send_hz,20)
                self.assertEqual(tuple(window.send_rate_picker['values']),('20','25','30'))
                window.send_hz_var.set('25');window.update_send_rate_note();window.save_config()
                self.assertEqual(json.loads(window.config_path.read_text())['tcp_send_hz'],20)
                window.manual=Mock();window.manual.thread.is_alive.return_value=True
                window.session=Mock()
                window.apply_send_rate()
                window.manual.set_send_hz.assert_called_once_with(25)
                window.manual.start_dance.assert_not_called();window.manual.enable.assert_not_called()
                window.session.configure.assert_not_called();self.assertIs(window.settings,settings)
                self.assertEqual(json.loads(window.config_path.read_text())['tcp_send_hz'],25)
                window.manual=None;window.session=None
                root2=tk.Toplevel(root);root2.withdraw()
                restarted=FollowLabWindow(root2,directory)
                self.assertEqual(restarted.tcp_send_hz,25)
                self.assertIn('40.00 ms',restarted.send_rate_note.get())
            finally:root.destroy()


if __name__=='__main__':unittest.main()
