"""Session teardown and terminal TCP release; simulated peers only."""
import socket
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from follow_neo.transport.control_transport import ControlChannel, ZERO
from follow_neo.transport.manual import ManualControl
from follow_neo.runtime.client import RuntimeClient, RemoteSession
from follow_neo.shared.types import Settings
from process_fixture import fixture_worker
from test_control_resync import AndroidProtocolPeer
from test_manual import wait_for
from test_process_runtime import TCPServer


class TerminalPeer:
    def __init__(self,clock,marker=True,disable=b'success\r\n'):
        self.clock=clock;self.marker=marker;self.disable=disable
        self.wire=[];self.chunks=[];self.timeout=.02
    def setsockopt(self,*args):pass
    def settimeout(self,value):self.timeout=value
    def sendall(self,data):
        self.wire.append(data)
        if data.startswith(b'python_sync_'):
            if self.marker:self.chunks.append(b'Unknown command: '+data)
        else:self.chunks.append(self.disable if data==b'disable\r\n' else b'success\r\n')
    def recv(self,n):
        self.clock[0]+=self.timeout
        if self.chunks:return self.chunks.pop(0)
        raise socket.timeout()


class TerminalReleaseTests(unittest.TestCase):
    def run_case(self,prepare=lambda c:None,**kwargs):
        clock=[10.];peer=TerminalPeer(clock,**kwargs);events=[]
        with patch('follow_neo.transport.control_transport.time.monotonic',side_effect=lambda:clock[0]):
            channel=ControlChannel(peer,lambda keys:'operator stopped',lambda e,d:events.append((e,d)))
            prepare(channel)
            result=channel.finish_release()
        self.assertTrue(channel.paused)
        self.assertFalse(any(b'enable' in cmd for cmd in peer.wire))
        return result,channel,peer,events

    def test_clean_disconnect_confirms_zero_and_disable(self):
        ok,c,p,e=self.run_case()
        self.assertTrue(ok);self.assertTrue(c.disable_acknowledged)
        self.assertEqual(p.wire,[(ZERO+'\r\n').encode(),b'disable\r\n'])

    def test_focus_pause_drains_neutral_before_disable(self):
        ok,c,p,e=self.run_case(lambda c:c.pause('Window or settings focus lost'))
        self.assertTrue(ok);self.assertFalse(c.pending)
        self.assertEqual(p.wire.count(b'disable\r\n'),1)

    def test_release_in_progress_is_drained_without_duplicate_disable(self):
        ok,c,p,e=self.run_case(lambda c:c.release_control())
        self.assertTrue(ok);self.assertEqual(p.wire.count(b'disable\r\n'),1)

    def test_extra_success_requires_unique_marker_before_disable(self):
        ok,c,p,e=self.run_case(lambda c:c._unmatched(b'success\r\n'))
        self.assertTrue(ok);self.assertTrue(c.resync_confirmed)
        self.assertTrue(p.wire[-2].startswith(b'python_sync_'))
        self.assertEqual(p.wire[-1],b'disable\r\n')

    def test_generic_success_without_marker_cannot_confirm_shutdown(self):
        ok,c,p,e=self.run_case(lambda c:c._unmatched(b'success\r\n'),marker=False)
        self.assertFalse(ok);self.assertFalse(c.disable_acknowledged)
        self.assertFalse(e[-1][1]['server_acknowledged'])

    def test_rejected_disable_never_reports_confirmed(self):
        ok,c,p,e=self.run_case(disable=b'denied\r\n')
        self.assertFalse(ok);self.assertFalse(c.disable_acknowledged)

    def test_manual_focus_pause_with_extra_reply_can_close_without_reenabling(self):
        peer=AndroidProtocolPeer();events=[]
        c=ManualControl('127.0.0.1',peer.connect,lambda e,d:events.append((e,d)))
        self.addCleanup(lambda:(c.stop(),c.thread.join(4)))
        c.update(set());c.enable();c.start();wait_for(lambda:c.enabled)
        peer.inject=True;c.pause_control('Window or settings focus lost')
        wait_for(lambda:any(e=='control_unmatched_reply' for e,d in events))
        c.stop();c.thread.join(4)
        self.assertFalse(c.thread.is_alive());self.assertFalse(c.enabled)
        self.assertEqual(peer.commands.count('enable'),1)
        self.assertIn('disable',peer.commands)
        self.assertTrue(any(e=='control_released' and d.get('server_acknowledged') for e,d in events))
        self.assertFalse(any(e=='control_stop_unconfirmed' for e,d in events))


class SessionLifecycleTests(unittest.TestCase):
    def test_late_diagnostics_ignored_but_stale_actions_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            client=RuntimeClient(worker_target=fixture_worker)
            try:
                old=RemoteSession(client,Path(folder),'127.0.0.1','h264',Settings())
                old.stop();old.stop()
                self.assertFalse(old.record_control('control_indicator',{}))
                current=RemoteSession(client,Path(folder),'127.0.0.1','h264',Settings())
                self.assertFalse(client.call('session.record_control','control_indicator',{},identity=old.identity))
                with self.assertRaisesRegex(RuntimeError,'Video session replaced'):
                    client.call('session.reset',identity=old.identity)
                current.stop()
            finally:client.close()

    def test_realtime_refresh_survives_backend_session_retirement(self):
        import tkinter as tk
        from follow_neo.ui.window import FollowLabWindow
        with tempfile.TemporaryDirectory() as folder,TCPServer():
            client=RuntimeClient(worker_target=fixture_worker);root=tk.Tk();root.withdraw()
            try:
                window=FollowLabWindow(root,folder,runtime=client)
                window.ip.set('127.0.0.1');window.compute.set('CPU');window.connect()
                window.toggle_manual();window.toggle_real_time()
                old=window.session
                # Reproduce the interval before Tk's disconnect callback clears it.
                client.call('session.stop',identity=old.identity)
                window.last_control_indicator=None;window.runtime_ui_at=0
                window.refresh()
                self.assertIsNotNone(window.last_control_indicator)
                self.assertTrue(root.tk.call('after','info'))
                window.disconnect()
                end=time.monotonic()+3
                while window.stopping and time.monotonic()<end:root.update();time.sleep(.01)
                self.assertIsNone(window.session)
            finally:client.close();root.destroy()
