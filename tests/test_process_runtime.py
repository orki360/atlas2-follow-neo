"""Actual Windows spawn, shared frames and original TCP on loopback only."""
import multiprocessing as mp
import os
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock
import numpy as np

from follow_neo.ipc.mailbox import Mailbox,FrameSlot,InputState
from follow_neo.runtime.client import RuntimeClient,RemoteManual,RemoteSession
from follow_neo.runtime.worker import Runtime
from follow_neo.capture.video import Frame
from follow_neo.transport.manual import ManualControl
from follow_neo.shared.types import Settings
from process_fixture import fixture_worker


def wait(predicate,timeout=3):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if predicate():return True
        time.sleep(.01)
    raise AssertionError('Timed out waiting for offline condition')


class TCPServer:
    def __enter__(self):
        self.sock=socket.socket();self.sock.bind(('127.0.0.1',9998));self.sock.listen(1);self.sock.settimeout(.1)
        self.commands=[];self.stop=threading.Event();self.client=None
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start();return self
    def run(self):
        while not self.stop.is_set():
            try:self.client,_=self.sock.accept();break
            except socket.timeout:continue
            except OSError:return
        if not self.client:return
        try:
            self.client.settimeout(.1);buffer=b''
            while not self.stop.is_set():
                try:chunk=self.client.recv(4096)
                except socket.timeout:continue
                if not chunk:break
                buffer+=chunk
                while b'\n' in buffer:
                    row,buffer=buffer.split(b'\n',1);line=row.decode().strip()
                    self.commands.append((time.monotonic(),line));self.client.sendall(b'success\r\n')
        except OSError:pass
    def __exit__(self,*args):
        self.stop.set();self.sock.close()
        if self.client:self.client.close()
        self.thread.join(timeout=1)


class ProcessRuntimeTests(unittest.TestCase):
    def test_shared_frame_is_exact_and_overwrite_is_rejected(self):
        ctx=mp.get_context('spawn');slot=FrameSlot(ctx,256)
        pixels=np.arange(72,dtype=np.uint8).reshape(4,6,3)
        first=slot.put(Frame(5,123.,pixels,7));received=slot.get(first)
        self.assertTrue(np.array_equal(received.image,pixels));self.assertEqual(received.decoded_at,123.)
        slot.put(Frame(6,124.,pixels+1,7));self.assertIsNone(slot.get(first))
        slot.lock.acquire()
        try:self.assertIsNone(slot.put(Frame(7,125.,pixels)))
        finally:slot.lock.release()

    def test_mailbox_bounded_latest_only(self):
        box=Mailbox(mp.get_context('spawn'),256)
        for i in range(20):self.assertTrue(box.put(dict(value=i)))
        value,version=box.get();self.assertEqual(value['value'],19)
        self.assertEqual(box.get(version),(None,version))
        with self.assertRaises(ValueError):box.put('x'*1000)

    def test_input_poll_does_not_renew_heartbeat_and_release_fences_enable(self):
        ctx=mp.get_context('spawn');inputs=InputState(ctx)
        runtime=Runtime(inputs,Mailbox(ctx),FrameSlot(ctx,100),FrameSlot(ctx,100))
        runtime.manual=ManualControl('127.0.0.1');runtime.identity=1
        inputs.pulse({'w'},1);runtime.consume_input();stamp=runtime.manual.updated
        runtime.consume_input();self.assertEqual(runtime.manual.updated,stamp)
        old_fence=inputs.fence.value;inputs.release();runtime.consume_input()
        with self.assertRaisesRegex(RuntimeError,'Expired'):
            runtime.dispatch('manual.enable',(),{},1,time.monotonic(),old_fence)
        self.assertFalse(runtime.manual.wanted)
        inputs.pulse(set(),1);runtime.consume_input()
        runtime.dispatch('manual.enable',(),{},1,time.monotonic(),inputs.fence.value)
        runtime.consume_input();self.assertTrue(runtime.manual.wanted)

    def test_old_or_wrong_connection_input_is_not_accepted(self):
        ctx=mp.get_context('spawn');inputs=InputState(ctx)
        runtime=Runtime(inputs,Mailbox(ctx),FrameSlot(ctx,100),FrameSlot(ctx,100))
        runtime.manual=ManualControl('127.0.0.1');runtime.identity=2
        inputs.pulse({'w'},1);runtime.consume_input();self.assertEqual(runtime.manual.updated,0.)
        inputs.latest.put(dict(time=time.monotonic()-1,keys=['w'],identity=2,fence=0))
        runtime.consume_input();self.assertEqual(runtime.manual.updated,0.)
        runtime.manual.update({'w'},source_time=time.monotonic()-1)
        self.assertEqual(runtime.manual.updated,0.)

    def test_expired_flight_action_and_enable_are_rejected(self):
        ctx=mp.get_context('spawn');inputs=InputState(ctx)
        runtime=Runtime(inputs,Mailbox(ctx),FrameSlot(ctx,100),FrameSlot(ctx,100))
        runtime.manual=ManualControl('127.0.0.1');runtime.identity=1
        for op,args in [('enable',()),('request',('takeoff',)),('start_dance',())]:
            with self.assertRaisesRegex(RuntimeError,'Expired'):
                runtime.dispatch('manual.'+op,args,{},1,time.monotonic()-1,0)
        self.assertFalse(runtime.manual.wanted);self.assertIsNone(runtime.manual.action)

    def test_worker_uses_live_decision_not_gui_snapshot(self):
        ctx=mp.get_context('spawn');inputs=InputState(ctx)
        runtime=Runtime(inputs,Mailbox(ctx),FrameSlot(ctx,100),FrameSlot(ctx,100))
        runtime.manual=ManualControl('127.0.0.1');runtime.identity=1
        session=Mock(model_error=None);session.receiver.host='127.0.0.1';session.receiver.state='STREAMING'
        session.reset_event.is_set.return_value=False
        expected={'generation':7,'measurement_time':123.}
        session.control_decision.return_value=expected;runtime.session=session
        self.assertIs(runtime.decision(),expected)
        session.receiver.state='RECONNECTING';self.assertIsNone(runtime.decision())

    def test_preview_backpressure_does_not_prevent_control_input(self):
        ctx=mp.get_context('spawn');inputs=InputState(ctx)
        runtime=Runtime(inputs,Mailbox(ctx),FrameSlot(ctx,100),FrameSlot(ctx,100))
        runtime.manual=ManualControl('127.0.0.1');runtime.identity=1
        runtime.live.lock.acquire();runtime.snapshots.lock.acquire()
        try:
            inputs.pulse({'up'},1);runtime.consume_input()
            self.assertEqual(runtime.manual.keys,{'up'})
            self.assertFalse(runtime.snapshots.put({'test':1}))
        finally:runtime.live.lock.release();runtime.snapshots.lock.release()

    def test_comparison_preserves_settings_changes_and_weights_timing_counts(self):
        from analysis.compare_architecture_logs import summarize
        with tempfile.TemporaryDirectory() as folder:
            rows=[dict(event='session_start',monotonic=1,settings={'video_fps':60}),
                  dict(event='settings',monotonic=2,settings={'video_fps':30}),
                  dict(event='runtime_health',monotonic=3,architecture_mode='process',
                       scheduling={'control_tick_gap_ms':dict(count=2,mean=10,p95=12,maximum=12)}),
                  dict(event='runtime_health',monotonic=4,architecture_mode='process',
                       scheduling={'control_tick_gap_ms':dict(count=1,mean=40,p95=40,maximum=40)})]
            path=Path(folder)/'events.jsonl';path.write_text('\n'.join(map(json.dumps,rows)))
            report=summarize(path)
            self.assertEqual(report['timing']['control_tick_gap_ms']['mean'],20.)
            self.assertEqual(len(report['configuration_changes']),1)

    def test_spawn_owns_tcp_and_stalled_gui_neutralizes_motion(self):
        with TCPServer() as server:
            client=RuntimeClient()
            try:
                info=client.call('ping');self.assertNotEqual(info['pid'],os.getpid())
                manual=RemoteManual(client,'127.0.0.1',require_target_policy=True)
                manual.set_axis_limits(dict(yaw=.2,vertical=.2,roll=.2,forward=.2));manual.start()
                manual.update(set());manual.enable()
                deadline=time.monotonic()+.5
                while time.monotonic()<deadline:
                    # Fake GUI decision must never authorize Dance.
                    manual.update({'up'},dict(intent={'forward':1},confirmed=True));client.poll();time.sleep(.02)
                self.assertTrue(any(line.startswith('rc ') and float(line.split()[-1])>0 for _,line in server.commands))
                stopped_pulses=time.monotonic()
                time.sleep(.55);client.poll()
                self.assertTrue(any(t>stopped_pulses and line=='rc 0 0 0 0' for t,line in server.commands))
                self.assertTrue(manual.communication_paused)
                self.assertIn('heartbeat',manual.pause_reason.lower())
                manual.release();manual.stop()
                wait(lambda:any(line=='disable' for _,line in server.commands))
            finally:client.close()
            self.assertFalse(client.process.is_alive())

    def test_spawn_session_preview_realtime_and_reconnect(self):
        with tempfile.TemporaryDirectory() as folder:
            client=RuntimeClient(worker_target=fixture_worker)
            try:
                session=RemoteSession(client,Path(folder),'127.0.0.1','h264',Settings(),False,'CPU',-1,
                                      'neo_minipro4_yolo26s_seg_960x544_v6.onnx','Auto',real_time=False)
                session.start()
                def has_frame():client.poll();return session.frames.get() is not None
                wait(has_frame)
                frame=session.frames.get();self.assertEqual(frame.image.shape,(24,32,3))
                self.assertTrue((frame.image==73).all())
                stamp=frame.decoded_at;generation=session.generation
                session.set_real_time(True);time.sleep(.3);client.poll()
                self.assertIsNone(session.frames.get());self.assertIsNone(session.presentation_analysis())
                self.assertEqual(session.generation,generation)
                session.set_real_time(False);wait(has_frame)
                self.assertEqual(session.frames.get().decoded_at,stamp)
                old=session.identity;session.stop()
                session=RemoteSession(client,Path(folder),'127.0.0.1','h264',Settings(),real_time=True)
                self.assertGreater(session.identity,old);session.stop()
            finally:client.close()

    def test_gui_with_spawned_backend_can_render_and_toggle(self):
        import tkinter as tk
        from follow_neo.ui.window import FollowLabWindow
        with tempfile.TemporaryDirectory() as folder:
            client=RuntimeClient(worker_target=fixture_worker);root=tk.Tk();root.withdraw()
            try:
                window=FollowLabWindow(root,folder,runtime=client)
                window.ip.set('127.0.0.1');window.compute.set('CPU');window.connect()
                self.assertIsInstance(window.session,RemoteSession)
                time.sleep(.15);window.refresh()
                self.assertIsNotNone(window.last_frame)
                window.toggle_real_time();time.sleep(.3);window.refresh()
                self.assertTrue(window.real_time);self.assertIsNone(window.last_image)
                window.toggle_real_time();time.sleep(.15);window.refresh()
                self.assertFalse(window.real_time)
                window.session.stop();window.session=None
            finally:client.close();root.destroy()


if __name__=='__main__':unittest.main()
