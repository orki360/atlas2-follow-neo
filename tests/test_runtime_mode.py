"""Real-Time changes diagnostics, never tracking evidence or TCP authority."""
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from follow_neo.gui import FollowLabWindow
from follow_neo.manual import ManualControl
from follow_neo.recording import VideoRecorder
from follow_neo.runtime_mode import RuntimeStats, keep_realtime_event
from follow_neo.session import FollowSession, SessionLog
from follow_neo.types import Settings
from follow_neo.video import Frame, VideoReceiver
from follow_neo.video_diagnostics import VideoDiagnostics
from test_manual import Server as FakeServer, wait_for


class RuntimeModeTests(unittest.TestCase):
    def session(self, **kwargs):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        nano=Mock();nano.snapshot.return_value={};nano.at_frame.return_value={}
        nano.evidence_at.return_value=None
        with patch('follow_neo.session.resolve_model',return_value=(Path(folder.name)/'test.onnx',{})), \
             patch('follow_neo.session.NanoWorker',return_value=nano):
            session=FollowSession(folder.name,'offline.invalid',**kwargs)
        self.addCleanup(session.stop)
        return session

    def test_toggle_preserves_decision_result_identity_settings_and_deadlines(self):
        s=self.session(record_raw=True)
        s.receiver.state='STREAMING';s.receiver.stream_epoch=4
        frame=Frame(8,10.,np.zeros((8,8,3),np.uint8),4)
        result={'frame':frame,'generation':s.generation}
        decision=dict(state='TRACK',generation=s.generation,target_identity={'logical_target_id':7},
                      measurement_time=9.9,prediction_time=10.,intent={'forward':.2})
        s.receiver.frames.set(frame);s.result.set(result);s.decision.set(decision)
        s.analysis.set({'result':result});settings=s.settings;generation=s.generation;heading=s.heading
        recorder=Mock();recorder.thread.is_alive.return_value=False;s.recorder=recorder
        recorder.status.return_value={}
        for enabled in (True,False,True,False):
            s.set_real_time(enabled)
            self.assertIs(s.decision.get(),decision);self.assertIs(s.result.get(),result)
            self.assertIs(s.receiver.frames.get(),frame);self.assertIs(s.settings,settings)
            self.assertIs(s.heading,heading);self.assertEqual(s.generation,generation)
            self.assertEqual(s.receiver.stream_epoch,4)
            self.assertFalse(s.reset_event.is_set());self.assertFalse(s.control_reset_event.is_set())
        self.assertFalse(s.receiver.raw_recording_enabled)
        self.assertTrue(s.receiver.diagnostics.enabled)
        s.nano.reset.assert_not_called();s.nano.stop.assert_not_called()
        recorder.stop.assert_called_with(discard_pending=True)
        recorder.wait.assert_not_called()
        self.assertIsNone(s.analysis.get())

    def test_realtime_start_blocks_raw_and_mp4_but_still_feeds_nano(self):
        s=self.session(real_time=True,record_raw=True)
        self.assertIsNone(s.receiver.raw_path);self.assertFalse(s.receiver.raw_recording_enabled)
        self.assertFalse(s.receiver.diagnostics.enabled)
        s.receiver.state='STREAMING';s.recorder=Mock();s.recorder.thread.is_alive.return_value=False
        s.recorder.status.return_value={}
        frame=Frame(1,time.monotonic(),np.zeros((8,8,3),np.uint8))
        s._record_frame(frame);s.nano.submit_frame.assert_called_once_with(frame)
        s.recorder.submit.assert_not_called()
        with self.assertRaisesRegex(RuntimeError,'Real-Time'):s.start_recording('Both videos')
        s.set_real_time(False);s._record_frame(frame)
        s.recorder.submit.assert_called_once()

    def test_filters_details_but_keeps_faults_actions_search_and_identity_events(self):
        with tempfile.TemporaryDirectory() as folder:
            log=SessionLog(Path(folder)/'events.jsonl',real_time=True)
            for event in ('observation','botsort_association','nano_observation','heading'):
                log.write(event,{'unserializable_detail':object()})
            log.write('flight_command',{'command':'rc','values':[0,0,0,0]})
            kept=['control_error','control_movement_paused','control_focus_pause','video_error',
                  'target_reassociated','search_observation_transition','future_unknown_fault']
            for event in kept:log.write(event,{'reason':'test'})
            log.write('flight_command',{'command':'land'})
            log.close();self.assertIsNone(log.error);self.assertEqual(log.suppressed,5)
            events=[json.loads(row)['event'] for row in log.path.read_text().splitlines()]
            self.assertEqual(events,kept+['flight_command'])
        self.assertFalse(keep_realtime_event('control_command_ack',{'command':'rc .1 0 0 0'}))
        self.assertTrue(keep_realtime_event('control_command_ack',{'command':'enable'}))

    def test_motion_transitions_capture_subsecond_stops_and_ack_statistics(self):
        stats=RuntimeStats();d=dict(mode='DANCE',command='rc',values=[0,0,0,.1],ack_ms=12)
        self.assertTrue(stats.command(d,10.)['moving'])
        self.assertIsNone(stats.command(d,10.05))
        stopped=stats.command(dict(d,values=[0,0,0,0],gate_reason='stale',ack_ms=30),10.1)
        resumed=stats.command(d,10.2)
        self.assertFalse(stopped['moving']);self.assertAlmostEqual(resumed['previous_duration_seconds'],.1)
        metrics=stats.snapshot()
        self.assertEqual(metrics['commands_acknowledged'],4);self.assertEqual(metrics['neutral_commands'],1)
        self.assertEqual(metrics['ack_ms_max'],30);self.assertAlmostEqual(metrics['command_gap_ms_max'],100)
        self.assertEqual(stats.snapshot()['commands_acknowledged'],0)

    def test_runtime_rates_do_not_depend_on_gui_or_change_measurement_time(self):
        s=self.session(real_time=True);s.log.write=Mock()
        d=dict(state='TRACK',measurement_time=9.9);s.decision.set(d)
        with patch('follow_neo.session.time.monotonic',return_value=10.):s._runtime_health({})
        s.receiver.count=60;s.receiver.processed_count=42;s.inference_count=12
        with patch('follow_neo.session.time.monotonic',return_value=11.):s._runtime_health({})
        data=s.log.write.call_args.args[1]
        self.assertEqual((data['input_fps'],data['processed_fps'],data['inference_fps']),(60,42,12))
        self.assertAlmostEqual(data['measurement_age_ms'],1100)
        self.assertEqual(s.decision.get()['measurement_time'],9.9)

    def test_control_error_still_invalidates_authority_context(self):
        s=self.session(real_time=True);s.decision.set({'state':'TRACK'})
        s.record_control('control_error',{'error':'test'})
        self.assertIsNone(s.decision.get());self.assertTrue(s.control_reset_event.is_set())
        s.nano.reset.assert_called_once()

    def test_suppressed_profiling_still_runs_decoding_body(self):
        diagnostics=VideoDiagnostics();diagnostics.enabled=False;work=[]
        with diagnostics.measure('decode'):work.append('decoded')
        diagnostics.received(100,10.);diagnostics.decoded(True,10.)
        self.assertEqual(work,['decoded']);self.assertEqual(diagnostics.snapshot()['stages'],{})
        diagnostics.enabled=True
        with diagnostics.measure('decode'):work.append('decoded')
        self.assertEqual(diagnostics.snapshot()['stages']['decode']['count'],1)

    def test_stopping_raw_recording_keeps_receiving_and_decoding_same_stream(self):
        receiver=VideoReceiver('offline.invalid');sock=Mock();step=iter(range(3))
        class Raw(io.BytesIO):
            def close(self):
                self.saved=self.getvalue();super().close()
        recording=Raw();codec=Mock()
        codec.parse.side_effect=lambda data:[data]
        codec.decode.return_value=[Mock(key_frame=False)]
        def receive(size):
            i=next(step)
            if i==1:receiver.raw_recording_enabled=False
            if i==2:receiver.stop_event.set()
            return bytes([i+1])
        sock.recv.side_effect=receive
        with patch('follow_neo.video.socket.create_connection',return_value=sock), \
             patch('follow_neo.video.av.CodecContext',Mock(create=Mock(return_value=codec))):
            receiver._receive_stream(recording)
        self.assertEqual(receiver.count,3);self.assertEqual(receiver.bytes,3)
        self.assertEqual(codec.decode.call_count,3)
        self.assertTrue(recording.closed);self.assertEqual(recording.saved,b'\x01')
        self.assertEqual(receiver.stream_epoch,0)

    def test_gui_heartbeat_runs_each_refresh_even_when_text_is_throttled(self):
        w=FollowLabWindow.__new__(FollowLabWindow)
        w.real_time=True;w.runtime_ui_at=10.;w.pressed={'up'};w.root=Mock()
        w.reconcile_keyboard=Mock();w.manual=Mock();w.manual.thread.is_alive.return_value=True
        w.dance_decision=Mock(return_value={'state':'TRACK'});w.finish_terminal_dance=Mock(return_value=False)
        with patch('follow_neo.gui.time.monotonic',return_value=10.01), \
             patch('follow_neo.gui.render_bundle',side_effect=AssertionError('rendered')):
            for _ in range(3):w.refresh()
        self.assertEqual(w.manual.update.call_count,3);self.assertEqual(w.reconcile_keyboard.call_count,3)
        self.assertEqual(w.root.after.call_count,3)
        self.assertTrue(all(1<=c.args[0]<=34 for c in w.root.after.call_args_list))

    def test_realtime_button_focus_is_safe_but_external_focus_still_pauses(self):
        w=FollowLabWindow.__new__(FollowLabWindow)
        w.root=Mock();w.root.after_idle.side_effect=lambda callback:callback()
        w.canvas=object();w.realtime_button=object();w.manual=Mock(wanted=True)
        w.clear_keyboard_input=Mock();w.log_control=Mock();w.pressed={'up'};w.closing=False
        with patch('follow_neo.gui.safe_focus_widget',return_value=w.realtime_button):w.focus_out(None)
        w.manual.pause_control.assert_not_called();self.assertEqual(w.pressed,{'up'})
        with patch('follow_neo.gui.safe_focus_widget',return_value=None):w.focus_out(None)
        w.manual.pause_control.assert_called_once();w.clear_keyboard_input.assert_called_once_with('focus_lost')

    def test_original_tcp_keeps_movement_release_and_heartbeat_guard_in_compact_mode(self):
        server=FakeServer();events=[]
        control=ManualControl('127.0.0.1',connector=server.connect,on_event=lambda e,d:events.append((e,d)))
        self.addCleanup(lambda:(control.stop(),control.thread.join(3)))
        control.detailed_logging=False;control.start();control.update({'up'});control.enable()
        wait_for(lambda:any(e=='flight_command' and any(d.get('values',())) for e,d in events))
        command=next(d for e,d in events if e=='flight_command' and any(d.get('values',())))
        self.assertTrue(command['server_acknowledged']);self.assertNotIn('nano',command)
        wait_for(lambda:control.communication_paused)
        wait_for(lambda:control.transport_status.get('neutral_acknowledged'))
        self.assertTrue(control.thread.is_alive());self.assertIn('rc 0 0 0 0',server.commands)
        self.assertNotIn('disable',server.commands)
        control.release();wait_for(lambda:'disable' in server.commands)

    def test_tracker_decisions_match_with_realtime_off_on_and_midstream_switches(self):
        def replay(mode):
            s=self.session(settings=Settings(nano_mode='off',confidence=.4),real_time=mode=='on')
            s.receiver.state='STREAMING';s.log.write=Mock();clock=[10.];step=[0];trace=[]
            s.control_provider=lambda:dict(active=True,mode='DANCE',time=clock[0])
            original=s.decision.set
            def capture(value):
                original(value)
                if value:trace.append(copy.deepcopy({k:value.get(k) for k in
                    ('state','spacing_phase','stale','confirmed','measurement_time','measurement_id','intent','generation')}))
            s.decision.set=capture
            def feed(timeout):
                i=step[0];step[0]+=1
                if i==60:s.stop_event.set();return
                clock[0]=10+i*.08
                if mode=='switch' and i in (15,40):s.set_real_time(i==15)
                frame=Frame(i+1,clock[0]-.02,np.zeros((480,640,3),np.uint8))
                boxes=[] if 22<=i<32 else [dict(box=[260+i,180,320+i,220],confidence=.95,
                    class_id=0,class_name='neo',model_class_name='NEO',class_scores=[.95,.01])]
                s.receiver.frames.set(frame)
                s.result.set(dict(frame=frame,generation=0,detections=boxes,completed_at=clock[0]-.01,
                                  inference_ms=10,preprocess_ms=1,postprocess_ms=1))
            s.control_wake=Mock();s.control_wake.wait.side_effect=feed
            with patch('follow_neo.session.time.monotonic',side_effect=lambda:clock[0]), \
                 patch('follow_neo.session.heading_snapshot',side_effect=lambda receiver:({'time':clock[0],'yaw_deg':0,'source':'offline_test'},clock[0])):
                s._control()
            self.assertIsNone(s.model_error)
            self.assertEqual(len(trace),60)
            self.assertTrue(any(d['confirmed'] for d in trace))
            self.assertTrue(any(any((d['intent'] or {}).values()) for d in trace))
            return trace
        normal=replay('off')
        self.assertEqual(normal,replay('on'))
        self.assertEqual(normal,replay('switch'))

    def test_recording_stop_discards_pending_without_waiting_for_inflight_encoder(self):
        entered=threading.Event();release=threading.Event();encode=VideoRecorder._encode
        def delayed(output,source,pts):
            entered.set()
            if not release.wait(3):raise RuntimeError('test release timeout')
            return encode(output,source,pts)
        with tempfile.TemporaryDirectory() as folder, patch('follow_neo.recording.probe_nvenc',return_value=(False,'test')), \
             patch.object(VideoRecorder,'_encode',staticmethod(delayed)):
            recorder=VideoRecorder(folder,'Clean video',profile='Auto')
            pixels=np.zeros((120,160,3),np.uint8)
            try:
                recorder.submit(Frame(1,10.,pixels));self.assertTrue(entered.wait(2))
                for i in range(2,7):recorder.submit(Frame(i,10+i*.1,pixels))
                start=time.perf_counter();recorder.stop(discard_pending=True)
                self.assertLess(time.perf_counter()-start,.1)
                self.assertTrue(recorder.thread.is_alive());self.assertTrue(recorder.queue.empty())
                self.assertFalse(recorder.submit(Frame(9,11.,pixels)))
            finally:release.set();recorder.wait()
            self.assertIsNone(recorder.error);self.assertEqual(recorder.written,1)
            self.assertEqual(recorder.dropped,5)


if __name__=='__main__':unittest.main()
