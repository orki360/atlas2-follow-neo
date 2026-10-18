"""Offline interrupted-stream regressions. No phone or aircraft connections."""
from pathlib import Path
from unittest.mock import Mock,patch
import socket
import tempfile
import threading
import time
import unittest
import numpy as np

from follow_neo.video import Frame,VideoReceiver
from follow_neo.session import FollowSession
from follow_neo.detector import DetectorSettings
from follow_neo.manual import ManualControl
from test_manual import Server,wait_for
from test_control_scan_fix import AsyncServer
from test_dance import decision


def decoded():
    return Mock(to_ndarray=Mock(return_value=np.zeros((20,20,3),np.uint8)))


def publish_inline(receiver):
    """Drive the separate publisher deterministically in decoder/socket fixtures."""
    original=receiver._publish_decoded
    def offer(frame,stamp):
        original(frame,stamp);receiver._publish_pending(stamp)
    return patch.object(receiver,'_publish_decoded',side_effect=offer)


class VideoReconnectTests(unittest.TestCase):
    def test_real_h264_decoder_recovers_on_new_stream(self):
        import av
        from fractions import Fraction
        encoder=av.CodecContext.create('libx264','w')
        encoder.width=32;encoder.height=32;encoder.pix_fmt='yuv420p'
        encoder.time_base=Fraction(1,30)
        encoder.options={'preset':'ultrafast','tune':'zerolatency'}
        packets=[]
        for i in range(5):
            f=av.VideoFrame.from_ndarray(np.full((32,32,3),i*40,np.uint8),format='bgr24');f.pts=i
            packets.extend(encoder.encode(f))
        packets.extend(encoder.encode(None));payload=b''.join(bytes(p) for p in packets)
        frames=[];r=VideoReceiver('offline.invalid')
        def publish(f):
            frames.append(f)
            if f.stream_epoch==1:r.stop_event.set()
        r.on_frame=publish
        peers=[Mock(),Mock()]
        for p in peers:p.recv.side_effect=[payload,b'']
        with publish_inline(r), patch('follow_neo.video.socket.create_connection',side_effect=peers), \
             patch.object(r.stop_event,'wait',side_effect=lambda delay:r.stop_event.is_set()):r._run()
        self.assertEqual({f.stream_epoch for f in frames},{0,1})
        self.assertIsNone(r.error);self.assertEqual(r.reconnect_count,1)
        self.assertTrue(all(f.image.shape==(32,32,3) for f in frames))
        self.assertEqual([f.frame_id for f in frames],sorted({f.frame_id for f in frames}))

    def test_eof_reconnects_with_new_decoder_and_never_sends_application_bytes(self):
        events=[];frames=[]
        r=VideoReceiver('offline.invalid',on_event=lambda e,d:events.append((e,d)))
        def publish(f):
            frames.append(f)
            if len(frames)==2:r.stop_event.set()
        r.on_frame=publish
        peers=[Mock(),Mock()]
        peers[0].recv.side_effect=[b'first-stream',b'']
        peers[1].recv.return_value=b'new-keyframe'
        codecs=[Mock(parse=Mock(return_value=[object()]),decode=Mock(return_value=[decoded()])) for _ in peers]
        with publish_inline(r), patch('follow_neo.video.socket.create_connection',side_effect=peers) as connect, \
             patch('follow_neo.video.av.CodecContext',Mock(create=Mock(side_effect=codecs))), \
             patch.object(r.stop_event,'wait',side_effect=lambda delay:r.stop_event.is_set()):
            r._run()
        self.assertEqual([f.frame_id for f in frames],[1,2])
        self.assertEqual([f.stream_epoch for f in frames],[0,1])
        self.assertEqual(connect.call_count,2);self.assertEqual(r.reconnect_count,1)
        self.assertIsNone(r.error);self.assertEqual(r.state,'DISCONNECTED')
        self.assertEqual(sum(e=='video_recovered' for e,d in events),1)
        self.assertTrue(next(d for e,d in events if e=='video_error')['retrying'])
        for peer in peers:
            peer.sendall.assert_not_called();peer.send.assert_not_called();peer.close.assert_called()

    def test_backoff_is_capped_and_operator_stop_cancels_retry(self):
        r=VideoReceiver('offline.invalid');delays=[]
        def event(e,d):
            if e=='video_error' and r.consecutive_failures==5:r.stop_event.set()
        def wait(delay):
            delays.append(delay);return r.stop_event.is_set()
        r.on_event=event
        with patch('follow_neo.video.socket.create_connection',side_effect=ConnectionRefusedError('offline')) as connect, \
             patch.object(r.stop_event,'wait',side_effect=wait):
            r._run()
        self.assertEqual(delays,[.5,1.,2.,4.,4.]);self.assertEqual(connect.call_count,5)
        self.assertIsNone(r.frames.get());self.assertEqual(r.state,'DISCONNECTED')

    def test_stop_on_failure_never_opens_a_second_connection(self):
        r=VideoReceiver('offline.invalid')
        r.on_event=lambda e,d:r.stop_event.set() if e=='video_error' else None
        with patch('follow_neo.video.socket.create_connection',side_effect=TimeoutError('offline')) as connect:
            r._run()
        self.assertEqual(connect.call_count,1);self.assertFalse(r.thread)

    def test_each_connection_has_its_own_decode_deadline(self):
        # An old last_frame_at must not kill a new connection before its keyframe.
        r=VideoReceiver('offline.invalid');r.last_frame_at=1.
        clock=[100.];peer=Mock()
        def recv(_):
            clock[0]+=.5
            raise socket.timeout()
        peer.recv.side_effect=recv
        with patch('follow_neo.video.socket.create_connection',return_value=peer), \
             patch('follow_neo.video.av.CodecContext'), \
             patch('follow_neo.video.time.monotonic',side_effect=lambda:clock[0]):
            with self.assertRaisesRegex(TimeoutError,'No video bytes'):r._receive_stream(None)
        self.assertEqual(peer.recv.call_count,17)

    def test_undecodable_continuous_bytes_also_timeout(self):
        r=VideoReceiver('offline.invalid');clock=[100.];peer=Mock()
        def recv(_):clock[0]+=1.;return b'undecodable'
        peer.recv.side_effect=recv
        with patch('follow_neo.video.socket.create_connection',return_value=peer), \
             patch('follow_neo.video.av.CodecContext',Mock(create=Mock(return_value=Mock(parse=Mock(return_value=[]))))), \
             patch('follow_neo.video.time.monotonic',side_effect=lambda:clock[0]):
            with self.assertRaisesRegex(TimeoutError,'no decoded frames'):r._receive_stream(None)
        self.assertEqual(peer.recv.call_count,9)


class SessionStreamTests(unittest.TestCase):
    def make_session(self):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        with patch('follow_neo.session.resolve_model',return_value=(Path(folder.name)/'test.onnx',{})):
            s=FollowSession(folder.name,'offline.invalid')
        self.addCleanup(s.stop)
        return s

    def detector(self):
        d=Mock();d.input.shape=[1,3,640,640];d.output.shape=[1,6,100]
        d.outputs=[d.output];d.settings=DetectorSettings();d.class_names={0:'neo',1:'mini'}
        d.model_class_names={0:'neo',1:'mini'};d.compute_info={}
        d.session.get_providers.return_value=['OfflineTest']
        return d

    def test_gui_clears_old_image_while_reconnecting_and_recovers(self):
        import tkinter as tk
        from follow_neo.gui import FollowLabWindow
        s=self.make_session();root=tk.Tk();root.withdraw()
        self.addCleanup(root.destroy)
        window=FollowLabWindow(root,s.root);window.session=s
        window.last_image=np.zeros((20,20,3),np.uint8)
        s.receiver._invalidate_stream(ConnectionError('offline'),retry_seconds=.5)
        window.refresh()
        self.assertEqual(window.state.get(),'VIDEO RECONNECTING');self.assertIsNone(window.last_image)
        self.assertTrue(any('Waiting for live video' in window.canvas.itemcget(i,'text')
                            for i in window.canvas.find_all() if window.canvas.type(i)=='text'))
        s.receiver._publish_decoded(decoded(),time.monotonic())
        s.receiver._publish_pending()
        window.refresh()
        self.assertEqual(window.state.get(),'STREAMING');self.assertIsNotNone(window.last_image)
        self.assertIn('Video recovered',window.notice.get())

    def test_interruption_clears_target_nano_and_invalidates_cached_decisions(self):
        s=self.make_session();s.receiver.state='STREAMING'
        f=Frame(1,time.monotonic(),np.zeros((20,20,3),np.uint8))
        s.receiver.frames.set(f);s.result.set(dict(frame=f,generation=0))
        s.analysis.set(dict(result=s.result.get()));s.decision.set(dict(generation=0))
        with patch.object(s.nano,'reset') as reset:
            s.receiver._invalidate_stream(ConnectionError('offline'),retry_seconds=.5)
        reset.assert_called_once_with(s.settings.nano_mode,1)
        self.assertIsNone(s.receiver.frames.get());self.assertIsNone(s.result.get())
        self.assertIsNone(s.control_decision());self.assertIsNone(s.presentation_analysis())
        self.assertTrue(s.reset_event.is_set())
        # Simulate old control/render work finishing after invalidation.
        s.receiver.state='STREAMING';s.decision.set(dict(generation=0))
        s.analysis.set(dict(result=dict(frame=f,generation=0)))
        self.assertIsNone(s.control_decision());self.assertIsNone(s.presentation_analysis())

    def test_frame_fetched_before_reset_is_not_inferred_as_new_generation(self):
        s=self.make_session();d=self.detector();old=Frame(1,time.monotonic(),np.zeros((20,20,3),np.uint8))
        calls=[]
        def next_frame(*args):
            calls.append(1)
            if len(calls)==1:
                s.receiver._invalidate_stream(ConnectionError('offline'),retry_seconds=.5)
                return old,1
            return Frame(2,time.monotonic(),np.zeros((20,20,3),np.uint8),s.receiver.stream_epoch),2
        def detect(image):
            s.stop_event.set();return dict(detections=[],inference_ms=1.)
        s.receiver.frames.wait_next=next_frame;d.detect.side_effect=detect
        with patch('follow_neo.session.OnnxDroneDetector',return_value=d):s._infer()
        self.assertIsNone(s.model_error);self.assertEqual(d.detect.call_count,1)
        self.assertEqual(s.result.get()['frame'].frame_id,2);self.assertEqual(s.result.get()['generation'],1)

    def test_inflight_inference_is_discarded_after_video_loss(self):
        s=self.make_session();d=self.detector();entered=threading.Event();release=threading.Event()
        def detect(image):
            entered.set();release.wait(2)
            return dict(detections=[],inference_ms=1.)
        d.detect.side_effect=detect
        self.addCleanup(release.set)
        s.receiver.frames.set(Frame(1,time.monotonic(),np.zeros((20,20,3),np.uint8)))
        with patch('follow_neo.session.OnnxDroneDetector',return_value=d):
            s.inference_thread.start();self.assertTrue(entered.wait(2))
            s.receiver._invalidate_stream(ConnectionError('offline'),retry_seconds=.5)
            release.set();wait_for(lambda:s.inference_count==1)
            self.assertIsNone(s.result.get());self.assertIsNone(s.control_decision())
            s.stop_event.set();s.inference_thread.join(2)
        self.assertIsNone(s.model_error)


class ControlRecoveryTests(unittest.TestCase):
    def test_zero_ack_delay_during_takeoff_does_not_repeat_enable(self):
        armed=threading.Event();blocked=threading.Event();release=threading.Event();events=[]
        def before_reply(cmd):
            if armed.is_set() and cmd.startswith('rc ') and not blocked.is_set():
                blocked.set();release.wait(2)
        peer=Server(before_reply=before_reply)
        c=ManualControl('127.0.0.1',peer.connect,lambda e,d:events.append((e,d)))
        self.addCleanup(lambda:(release.set(),c.stop(),c.thread.join(3)))
        c.update(set());c.enable();c.start();wait_for(lambda:c.enabled)
        self.assertTrue(c.request('takeoff'));wait_for(lambda:'takeoff' in peer.commands)
        armed.set();self.assertTrue(blocked.wait(.3))
        until=time.monotonic()+.32
        while time.monotonic()<until:c.update(set());time.sleep(.01)
        self.assertTrue(c.communication_paused);release.set()
        wait_for(lambda:any(e=='control_movement_resumed' for e,d in events))
        self.assertEqual(peer.commands.count('enable'),1)
        self.assertTrue(c.enabled);self.assertTrue(c.motion_hold)
        self.assertTrue(next(d for e,d in events if e=='control_movement_resumed')['authority_retained'])

    def test_incomplete_initial_handshake_still_requires_enable(self):
        blocked=threading.Event();release=threading.Event();events=[]
        def before_reply(cmd):
            if cmd=='rc 0 0 0 0' and not blocked.is_set():
                blocked.set();release.wait(2)
        peer=Server(before_reply=before_reply)
        c=ManualControl('127.0.0.1',peer.connect,lambda e,d:events.append((e,d)))
        self.addCleanup(lambda:(release.set(),c.stop(),c.thread.join(3)))
        c.update(set());c.enable();c.start();self.assertTrue(blocked.wait(.3))
        until=time.monotonic()+.32
        while time.monotonic()<until:c.update(set());time.sleep(.01)
        self.assertTrue(c.communication_paused);self.assertFalse(c.enabled)
        release.set();wait_for(lambda:c.enabled)
        self.assertEqual(peer.commands.count('enable'),1)
        resumed=next(d for e,d in events if e=='control_movement_resumed')
        self.assertFalse(resumed['authority_retained']);self.assertTrue(resumed['handshake_required'])

    def test_video_invalidation_revokes_pending_dance_even_with_nano_off(self):
        peer=AsyncServer(True);latest=[None]
        c=ManualControl('127.0.0.1',peer.connect,decision_provider=lambda:latest[0])
        self.addCleanup(lambda:(c.stop(),c.thread.join(3)))
        c.update(set());c.enable();c.start();wait_for(lambda:c.enabled);c.start_dance()
        wait_for(lambda:time.monotonic()>c.dance_started)
        latest[0]=dict(decision(measurement_time=time.monotonic()),generation=0)
        self.assertTrue(peer.motion.wait(.25))
        latest[0]=None
        self.assertTrue(peer.zero.wait(.15))
        self.assertIn('Video hold',c.dance_status)
        self.assertFalse(c.communication_paused)  # Urgent neutral precedes ACK timeout.


if __name__=='__main__':unittest.main()
