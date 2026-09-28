"""Reconnection, missing key-up, delayed acknowledgements and two-phase search."""
import socket
import threading
import time
import unittest
from unittest.mock import Mock,patch
from dataclasses import replace
from follow_neo.manual import ManualControl
from follow_neo.control_transport import ControlChannel
from follow_neo.gui import FollowLabWindow
from follow_neo.recovery_search import RecoverySearch,scan_command
from follow_neo.edge_search import yaw_override
from follow_neo.types import Box,Kinematics,Settings
from follow_neo.types import Detection
from follow_neo.controller import FollowController
from follow_neo.dance import dance_command
from test_manual import wait_for


class AsyncServer:
    """Read commands even when a previous reply is deliberately withheld."""
    def __init__(self,block_movement=False):
        self.client,self.server=socket.socketpair();self.commands=[];self.block=block_movement
        self.motion=threading.Event();self.zero=threading.Event();self.disabled=threading.Event()
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def connect(self,address,timeout):
        assert address==('127.0.0.1',9998)
        return self.client
    def run(self):
        try:
            with self.server.makefile('rb') as stream:
                for line in stream:
                    cmd=line.decode().strip();self.commands.append((time.monotonic(),cmd))
                    nonzero=cmd.startswith('rc ') and any(float(v) for v in cmd.split()[1:])
                    if nonzero:self.motion.set()
                    if cmd.startswith('rc ') and not nonzero and self.motion.is_set():self.zero.set()
                    if cmd=='disable':self.disabled.set()
                    if self.block and self.motion.is_set():continue
                    self.server.sendall(b'success\r\n')
        except OSError:pass
        finally:self.server.close()


class TransportTests(unittest.TestCase):
    def start(self):
        self.peer=AsyncServer(True);self.events=[]
        self.control=ManualControl('127.0.0.1',connector=self.peer.connect,
                                  on_event=lambda event,data:self.events.append((event,data)))
        self.addCleanup(lambda:(self.control.stop(),self.control.thread.join(3)))
        self.control.update(set());self.control.enable();self.control.start()
        wait_for(lambda:self.control.enabled)
        self.control.update({'down'})
        self.assertTrue(self.peer.motion.wait(1))
    def test_release_sends_zero_and_disable_without_waiting_for_old_reply(self):
        self.start();released=time.monotonic();self.control.update(set())
        self.assertTrue(self.peer.zero.wait(.20));self.assertTrue(self.peer.disabled.wait(.20))
        self.control.thread.join(1)
        self.assertFalse(self.control.enabled);self.assertFalse(self.control.wanted)
        self.assertLess(time.monotonic()-released,.4)
        self.assertIn('STOP UNCONFIRMED',self.control.status)
        self.assertTrue(any(e=='control_stop_unconfirmed' and not d['server_acknowledged'] for e,d in self.events))
        self.assertFalse(any(e=='flight_command' and any(d.get('values',[])) for e,d in self.events))
    def test_reverse_during_delayed_reply_stops_instead_of_latching_opposite(self):
        self.start();self.control.update({'up'})
        self.assertTrue(self.peer.disabled.wait(.3));self.control.thread.join(1)
        self.assertFalse(any(c.startswith('rc ') and float(c.split()[-1])>0 for _,c in self.peer.commands))
    def test_missing_ack_times_out_even_with_healthy_ui_heartbeat(self):
        self.start();deadline=time.monotonic()+.5
        while self.control.thread.is_alive() and time.monotonic()<deadline:
            self.control.update({'down'});time.sleep(.01)
        self.assertTrue(self.peer.disabled.wait(.15));self.assertFalse(self.control.enabled)
    def test_success_from_interrupted_exchange_is_not_claimed_as_stop_confirmation(self):
        client,server=socket.socketpair();events=[]
        self.addCleanup(client.close);self.addCleanup(server.close)
        channel=ControlChannel(client,lambda keys:None,lambda e,d:events.append((e,d)))
        server.sendall(b'success\r\n')
        channel.pending=True;channel.stop_unconfirmed('test late acknowledgement')
        self.assertFalse(events[-1][1]['server_acknowledged'])
        self.assertEqual(server.recv(100),b'rc 0 0 0 0\r\ndisable\r\n')


class KeyboardRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.w=FollowLabWindow.__new__(FollowLabWindow)
        self.w.root=Mock();self.w.canvas=Mock();self.w.manual=Mock()
        self.w.pressed={'down'};self.w.key_releases={'down':'old-timer'}
        self.w.session=None;self.w.notice=Mock();self.w.log_control=Mock()
    def test_enable_clears_old_keys_and_pending_callbacks(self):
        self.w.manual.stop_event.is_set.return_value=False
        self.w.enable_manual()
        self.assertFalse(self.w.pressed);self.assertFalse(self.w.key_releases)
        self.w.root.after_cancel.assert_called_with('old-timer')
        self.w.manual.update.assert_called_with(set());self.w.manual.enable.assert_called_once()
    def test_windows_release_is_immediate_even_if_idle_callbacks_never_run(self):
        event=Mock(keysym='Down',keycode=40)
        with patch('follow_neo.gui.sys.platform','win32'):self.w.key_release(event)
        self.assertFalse(self.w.pressed)
        self.w.manual.update.assert_called_with(set());self.w.root.after_idle.assert_not_called()
    def test_missing_release_event_reconciles_with_physical_keyboard(self):
        self.w.key_state_reader=lambda keys:set()
        self.w.reconcile_keyboard()
        self.assertFalse(self.w.pressed);self.w.manual.update.assert_called_with(set())
    def test_reconnect_starts_with_no_latched_keys(self):
        self.w.manual.thread.is_alive.return_value=False
        self.w.ip=Mock(get=Mock(return_value='127.0.0.1'));self.w.dance_selected=False
        self.w.current_axis_limits=Mock(return_value={'yaw':.1,'vertical':.1,'roll':.1,'forward':.1})
        with patch('follow_neo.gui.ManualControl') as create:
            self.w.toggle_manual()
            self.assertFalse(self.w.pressed);self.assertFalse(self.w.key_releases)
            create.return_value.enable.assert_not_called()


def heading(t,yaw=0):return dict(time=t,yaw_deg=yaw,source='AircraftAttitude')


class ScanTests(unittest.TestCase):
    def begin(self,outward=False,yaw=0):
        self.r=RecoverySearch();self.s=Settings(search_yaw_degrees=20)
        for i in range(12):
            t=9.64+i*.04;x=930+i*20 if outward else 600
            self.r.observe(True,False,True,Box(x,320,x+80,360),Kinematics(),1280,720,t,i,self.s)
            self.r.update(t,False,False,'APPROACH',False,self.s,heading(t,yaw))
        self.now=10.58
        return self.r.update(self.now,True,False,'APPROACH',False,self.s,heading(self.now,yaw))
    def step(self,t,yaw):return self.r.update(t,True,False,'APPROACH',False,self.s,heading(t,yaw))
    def decision(self,search,t):
        return dict(edge_search=search,prediction_time=t,state='DIRECTIONAL_SEARCH',
                    accepted=False,stale=False,spacing_phase='APPROACH')
    def test_stationary_target_loss_scans_both_directions_with_total_span(self):
        d=self.begin();self.assertEqual(d['phase'],'SCAN')
        self.assertEqual(d['angle_degrees'],20)
        d=self.step(10.8,0);self.assertLess(d['scan_yaw'],0)
        self.assertEqual(d['scan_target_degrees'],-10)
        d=self.step(10.9,-10);self.assertEqual(d['scan_yaw'],0)
        d=self.step(11.1,-10);self.assertGreater(d['scan_yaw'],0)
        self.assertEqual(d['scan_target_degrees'],10)
        d=self.step(11.5,10);self.assertEqual(d['scan_yaw'],0)
        d=self.step(11.7,10);self.assertLess(d['scan_yaw'],0)
    def test_boost_is_short_then_scan_has_no_full_yaw_override(self):
        d=self.begin(outward=True);end=d['until']
        self.assertEqual(d['phase'],'BOOST');self.assertEqual(yaw_override(self.decision(d,self.now),self.now,9),1)
        d=self.step(10.96,3);self.assertEqual(d['phase'],'SCAN');self.assertEqual(d['until'],end)
        self.assertEqual(yaw_override(self.decision(d,10.96),10.96,9),0)
        d=self.step(11.2,3);values,status=dance_command(self.decision(d,11.2),11.2,9)
        self.assertTrue(status.startswith('Scan /'));self.assertLess(abs(values[0]),1)
        self.assertEqual(values[1:],(0.,0.,0.))
    def test_boost_switches_at_half_angle_boundary(self):
        self.begin(outward=True);d=self.step(10.7,10)
        self.assertEqual(d['phase'],'SCAN');self.assertEqual(d['scan_yaw'],0)
    def test_zero_angle_and_lost_heading_stop_both_phases(self):
        self.begin();self.s=replace(self.s,search_yaw_degrees=0)
        self.assertFalse(self.step(10.8,0)['active'])
        self.begin(outward=True)
        d=self.r.update(10.8,True,False,'APPROACH',False,self.s,None)
        self.assertFalse(d['active']);self.assertEqual(d['reason'],'heading_unavailable')
    def test_reacquisition_pauses_immediately_and_three_measurements_finish(self):
        self.begin()
        for i in range(3):
            t=10.8+i*.04
            self.r.observe(True,False,True,Box(600,320,680,360),Kinematics(),1280,720,t,30+i,self.s)
            d=self.step(t,0);self.assertFalse(d['active'])
        self.assertEqual(d['reason'],'reacquired')
    def test_heading_wraparound_does_not_expand_scan(self):
        self.begin(yaw=179);d=self.step(10.8,-176)
        self.assertAlmostEqual(d['scan_offset_degrees'],5)
    def test_timeout_and_video_loss_do_not_restart_without_new_detection(self):
        d=self.begin();end=d['until']
        d=self.step(end+.01,0);self.assertFalse(d['active']);self.assertEqual(d['reason'],'search_timeout')
        self.assertFalse(self.step(end+.2,0)['active'])
        self.begin();d=self.r.update(10.8,True,True,'APPROACH',False,self.s,heading(10.8))
        self.assertFalse(d['active']);self.assertFalse(self.step(10.9,0)['active'])
    def test_send_gate_rejects_outward_scan_at_boundary_and_stale_decisions(self):
        self.begin();d=self.step(10.8,0);decision=self.decision(d,10.8)
        self.assertNotEqual(scan_command(decision,10.8,9),0)
        self.assertEqual(scan_command(decision,11.1,9),0)
        decision['edge_search']={**d,'scan_offset_degrees':-10,'scan_yaw':-.25}
        self.assertEqual(scan_command(decision,10.8,9),0)


class LivePipelineTests(unittest.TestCase):
    def test_botsort_controller_selects_scan_and_keeps_translation_zero(self):
        c=FollowController(tracker_backend='botsort');s=Settings(search_yaw_degrees=20)
        for i in range(20):
            t=10+i/30
            c.observe([Detection(Box(280,210,340,250),.9)],t,t,i,(640,480),s)
            d=c.tick(t,640,480,t,s,heading(t))
        self.assertEqual(d['state'],'TRACK')
        for i in range(20,46):
            t=10+i/30;c.observe([],t,t,i,(640,480),s)
            d=c.tick(t,640,480,t,s,heading(t))
        self.assertEqual(d['edge_search']['phase'],'SCAN')
        self.assertEqual(d['state'],'DIRECTIONAL_SEARCH')
        values,status=dance_command(d,t,9.)
        self.assertTrue(status.startswith('Scan /'));self.assertLess(values[0],0)
        self.assertEqual(values[1:],(0.,0.,0.))
        self.assertEqual(yaw_override(d,t,9.),0.)

    def test_scan_transport_respects_normal_yaw_cap(self):
        peer=AsyncServer();events=[];stamp=time.monotonic()
        def decision():
            now=time.monotonic()
            return dict(prediction_time=now,state='DIRECTIONAL_SEARCH',accepted=False,stale=False,
                spacing_phase='APPROACH',edge_search=dict(active=True,phase='SCAN',started=stamp+.001,
                    until=stamp+4.001,source_time=stamp-.1,heading_time=now,duration_seconds=4.,
                    angle_degrees=20.,scan_offset_degrees=0.,scan_target_degrees=-10.,scan_yaw=-.25))
        control=ManualControl('127.0.0.1',peer.connect,on_event=lambda e,d:events.append((e,d)),decision_provider=decision)
        self.addCleanup(lambda:(control.stop(),control.thread.join(3)))
        control.set_axis_limits(dict(yaw=.08,vertical=.9,roll=.9,forward=.9))
        control.start_dance();stamp=time.monotonic()+.01
        control.update(set());control.enable();control.start()
        wait_for(lambda:any(e=='flight_command' and d.get('gate_reason','').startswith('Scan /') for e,d in events))
        command=next(d for e,d in events if e=='flight_command' and d.get('gate_reason','').startswith('Scan /'))
        self.assertFalse(command['edge_yaw_override'])
        self.assertAlmostEqual(command['values'][0],-.02)
        self.assertEqual(command['values'][1:],[0.,0.,0.])


if __name__=='__main__':unittest.main()
