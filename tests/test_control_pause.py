"""Offline socket regressions: ACK faults never automatically close control."""
import socket
import threading
import time
import unittest
from follow_neo.control_transport import ControlChannel
from follow_neo.manual import ManualControl
from follow_neo.control_status import control_indicator
from test_control_scan_fix import AsyncServer
from test_manual import wait_for


class ChannelPauseTests(unittest.TestCase):
    def setUp(self):
        self.client,self.peer=socket.socketpair()
        self.addCleanup(self.client.close);self.addCleanup(self.peer.close)
        self.events=[]
        self.channel=ControlChannel(self.client,lambda keys:None,
                                    lambda e,d:self.events.append((e,d)))
        self.peer.settimeout(.2)

    def timeout(self):
        self.assertFalse(self.channel.command('rc 0 0 0 -0.1'))
        self.assertEqual(self.peer.recv(1024),b'rc 0 0 0 -0.1\r\nrc 0 0 0 0\r\n')
        self.assertTrue(self.channel.paused)
        self.assertGreaterEqual(self.client.fileno(),0)

    def test_missing_ack_keeps_connection_and_never_sends_disable(self):
        self.timeout()
        self.assertFalse(self.channel.command('rc 1 0 0 0'))
        self.assertFalse(self.channel.resume())
        self.channel.poll_pause()
        with self.assertRaises(socket.timeout):self.peer.recv(100)

    def test_original_late_ack_cannot_confirm_neutral_or_resume(self):
        self.timeout();self.peer.sendall(b'success\r\n');self.channel.poll_pause()
        self.assertFalse(self.channel.neutral_acknowledged)
        self.assertFalse(self.channel.resume())
        self.peer.sendall(b'success\r\n');self.channel.poll_pause()
        self.assertTrue(self.channel.neutral_acknowledged)
        self.assertTrue(self.channel.paused)  # ACK alone does not resume.
        self.assertTrue(self.channel.resume())
        acks=[d['command'] for e,d in self.events if e=='control_command_ack']
        self.assertEqual(acks,['rc 0 0 0 -0.1','rc 0 0 0 0'])

    def test_fragmented_and_coalesced_late_replies_preserve_fifo(self):
        self.timeout();self.peer.sendall(b'succ');self.channel.poll_pause()
        self.assertFalse(self.channel.neutral_acknowledged)
        self.peer.sendall(b'ess\r\nsuccess\r\n\r\n');self.channel.poll_pause()
        self.assertFalse(self.channel.pending);self.assertFalse(self.channel.ambiguous)
        self.assertTrue(self.channel.resume())

    def test_extra_reply_is_quarantined_not_assigned_to_new_neutral(self):
        self.peer.sendall(b'success\r\nsuccess\r\n')
        self.assertFalse(self.channel.command('rc 0 0 0 .1'))
        self.assertTrue(self.channel.ambiguous)
        self.assertFalse(self.channel.neutral_acknowledged)
        self.peer.sendall(b'success\r\n');self.channel.poll_pause()
        self.assertFalse(self.channel.neutral_acknowledged);self.assertFalse(self.channel.resume())
        self.assertTrue(any(e=='control_unmatched_reply' and d['raw_hex'] for e,d in self.events))
        self.assertNotIn(b'disable',self.peer.recv(1024))

    def test_blank_separators_are_not_an_extra_ack(self):
        self.peer.sendall(b'\r\nsuccess\r\n\r\n')
        self.assertTrue(self.channel.command('rc 0 0 0 0'))
        self.assertFalse(self.channel.paused)

    def test_partial_unmatched_reply_pauses_without_closing(self):
        self.peer.sendall(b'success\r\nsucc')
        self.assertFalse(self.channel.command('rc 0 0 0 0'))
        self.assertTrue(self.channel.ambiguous)
        self.assertGreaterEqual(self.client.fileno(),0)

    def test_rejection_followed_by_extra_success_cannot_confirm_new_neutral(self):
        self.peer.sendall(b'denied\r\nsuccess\r\n')
        self.assertFalse(self.channel.command('enable'))
        self.assertTrue(self.channel.paused);self.assertTrue(self.channel.ambiguous)
        self.assertFalse(self.channel.neutral_acknowledged)


class ManualPauseTests(unittest.TestCase):
    def start(self,block=True):
        self.peer=AsyncServer(block);self.events=[]
        self.control=ManualControl('127.0.0.1',self.peer.connect,
                                    on_event=lambda e,d:self.events.append((e,d)))
        self.addCleanup(lambda:(self.control.stop(),self.control.thread.join(3)))
        self.control.update(set());self.control.enable();self.control.start()
        wait_for(lambda:self.control.enabled)

    def test_delayed_ack_resumes_automatically_without_replaying_held_key(self):
        self.start();c=self.control;c.update({'down'})
        self.assertTrue(self.peer.motion.wait(.2))
        until=time.monotonic()+.35
        while time.monotonic()<until:c.update({'down'});time.sleep(.01)
        self.assertTrue(c.communication_paused);self.assertTrue(c.thread.is_alive())
        self.assertFalse(self.peer.disabled.is_set());self.assertTrue(c.connected)
        self.assertFalse(c.request('takeoff'));self.assertFalse(c.control_snapshot()['active'])
        self.assertIn('CONNECTED',control_indicator(c)[0])
        self.assertFalse(any(e=='control_motion_neutralized' for e,d in self.events))
        # The motion ACK alone cannot authorize recovery.
        self.peer.server.sendall(b'success\r\n');time.sleep(.06)
        self.assertTrue(c.communication_paused)
        self.peer.block=False
        self.peer.server.sendall(b'success\r\n')
        wait_for(lambda:any(e=='control_motion_neutralized' for e,d in self.events))
        wait_for(lambda:not c.communication_paused and c.enabled)
        until=time.monotonic()+.12
        while time.monotonic()<until:c.update({'down'});time.sleep(.01)
        count=sum(any(float(v) for v in cmd.split()[1:]) for _,cmd in self.peer.commands if cmd.startswith('rc '))
        self.assertEqual(count,1)
        self.assertFalse(c.keys)
        self.assertTrue(any(e=='control_movement_resumed' and d['automatic'] and not d['explicit_enable'] for e,d in self.events))
        c.update(set());c.update({'down'})
        wait_for(lambda:sum(cmd.startswith('rc ') and float(cmd.split()[-1])<0 for _,cmd in self.peer.commands)>1)

    def test_q_during_ack_wait_prevents_automatic_reenable(self):
        self.start();c=self.control;c.update({'down'});self.assertTrue(self.peer.motion.wait(.2))
        wait_for(lambda:c.communication_paused)
        c.release();self.assertTrue(self.peer.disabled.wait(.3))
        # Motion, pause neutral, explicit release neutral, disable.
        self.peer.block=False;self.peer.server.sendall(b'success\r\n'*4)
        until=time.monotonic()+.15
        while time.monotonic()<until:c.update(set());time.sleep(.01)
        self.assertFalse(c.wanted);self.assertFalse(c.enabled)
        self.assertTrue(c.connected)
        self.assertEqual(sum(cmd=='enable' for _,cmd in self.peer.commands),1)
        self.assertFalse(any(e=='control_movement_resumed' for e,d in self.events))

    def test_heartbeat_recovery_resumes_without_e_but_not_while_ui_stale(self):
        self.start(False);c=self.control
        wait_for(lambda:c.communication_paused)
        wait_for(lambda:any(e=='control_motion_neutralized' for e,d in self.events))
        time.sleep(.06);self.assertTrue(c.communication_paused)
        c.update(set());wait_for(lambda:not c.communication_paused and c.enabled)
        self.assertFalse(self.peer.disabled.is_set())

    def test_focus_loss_during_ack_pause_blocks_automatic_recovery(self):
        self.start();c=self.control;c.update({'down'});self.assertTrue(self.peer.motion.wait(.2))
        wait_for(lambda:c.communication_paused)
        c.pause_control('Window focus lost')
        self.peer.block=False;self.peer.server.sendall(b'success\r\n'*2)
        until=time.monotonic()+.15
        while time.monotonic()<until:c.update(set());time.sleep(.01)
        self.assertTrue(c.communication_paused)
        self.assertFalse(any(e=='control_movement_resumed' for e,d in self.events))

    def test_automatic_resume_preserves_existing_flight_action_hold(self):
        self.start(False);c=self.control;c.start_dance()
        c.motion_hold=True  # Existing takeoff/landing hold is independent of ACK recovery.
        wait_for(lambda:c.communication_paused)
        c.update(set());wait_for(lambda:not c.communication_paused and c.enabled)
        self.assertEqual(c.mode,'DANCE');self.assertTrue(c.motion_hold)
        self.assertFalse(any(cmd in ('takeoff','land') for _,cmd in self.peer.commands))

    def test_release_at_resume_boundary_still_sends_disable(self):
        self.start(False);c=self.control
        def on_event(event,data):
            self.events.append((event,data))
            if event=='control_movement_resumed':c.release()
        c.on_event=on_event
        wait_for(lambda:c.communication_paused)
        c.update(set())
        self.assertTrue(self.peer.disabled.wait(.5))
        self.assertFalse(c.wanted);self.assertTrue(c.connected)
        self.assertEqual(sum(cmd=='enable' for _,cmd in self.peer.commands),1)

    def test_dance_auto_resume_requires_post_recovery_measurements(self):
        from test_dance import decision
        self.start();c=self.control;c.start_dance()
        wait_for(lambda:time.monotonic()>c.dance_started)
        old=decision(measurement_time=time.monotonic())
        c.decision_provider=lambda:old
        self.assertTrue(self.peer.motion.wait(.25))
        wait_for(lambda:c.communication_paused)
        c.update(set());self.peer.block=False
        self.peer.server.sendall(b'success\r\n'*2)
        wait_for(lambda:not c.communication_paused and c.enabled)
        time.sleep(.08)
        def motions():
            return sum(cmd.startswith('rc ') and any(float(v) for v in cmd.split()[1:])
                       for _,cmd in self.peer.commands)
        self.assertEqual(motions(),1)
        self.assertEqual(c.mode,'DANCE')
        c.decision_provider=lambda:decision(measurement_time=time.monotonic())
        c.update(set());wait_for(lambda:motions()>1)

    def test_new_presses_during_pause_also_require_release(self):
        c=ManualControl('127.0.0.1')
        c.update({'down'});c.pause_control('test',automatic_recovery=True)
        c.update({'down','up'});self.assertFalse(c.keys)
        c.communication_paused=False
        c.update({'down','up'});self.assertFalse(c.keys)
        c.update({'down'});c.update({'down','up'})
        self.assertEqual(c.keys,{'up'})

    def test_heartbeat_loss_pauses_without_disable_and_q_remains_explicit(self):
        self.start(False);c=self.control
        wait_for(lambda:c.communication_paused)
        self.assertTrue(c.wanted);self.assertTrue(c.thread.is_alive())
        self.assertFalse(self.peer.disabled.is_set())
        c.release();self.assertTrue(self.peer.disabled.wait(.4))
        self.assertTrue(c.thread.is_alive());self.assertTrue(c.connected)

    def test_operator_disconnect_while_ack_missing_is_bounded(self):
        self.start();self.control.update({'down'});self.assertTrue(self.peer.motion.wait(.2))
        self.control.stop();self.control.thread.join(.6)
        self.assertFalse(self.control.thread.is_alive());self.assertTrue(self.peer.disabled.wait(.2))
        self.assertFalse(self.control.connected)

    def test_focus_pause_keeps_connection_and_drops_old_input(self):
        self.start(False);c=self.control;c.pause_control('Window focus lost')
        wait_for(lambda:any(e=='control_motion_neutralized' for e,d in self.events))
        self.assertFalse(c.keys);self.assertTrue(c.wanted);self.assertTrue(c.connected)
        self.assertFalse(self.peer.disabled.is_set())

    def test_local_processing_error_keeps_connection_until_operator_disconnect(self):
        self.start(False);c=self.control
        def broken_decision():raise ValueError('bad local decision')
        c.decision_provider=broken_decision
        wait_for(lambda:c.communication_paused)
        self.assertTrue(c.connected);self.assertTrue(c.thread.is_alive())
        self.assertIn('bad local decision',c.status)
        self.assertFalse(self.peer.disabled.is_set())


if __name__=='__main__':unittest.main()
