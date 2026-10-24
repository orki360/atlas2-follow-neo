"""HARD has no diagnostic IO; original TCP authority and tracking still apply.

Offline only: socketpair/loopback, synthetic frames, no phone or aircraft.
"""
import multiprocessing as mp
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
import numpy as np

from follow_neo.diagnostics import activity
from follow_neo.diagnostics.timing import TimingWindow
from follow_neo.runtime.session import FollowSession, SessionLog
from follow_neo.runtime.worker import Runtime
from follow_neo.ipc.mailbox import InputState, Mailbox, FrameSlot
from follow_neo.transport.manual import ManualControl
from follow_neo.ui.hard_console import run_hard, WindowsConsole
from follow_neo.ui.window import FollowLabWindow
from follow_neo.control.controller import FollowController
from follow_neo.control.dance import dance_command
from follow_neo.shared.types import Settings, Detection, Box
from test_manual import Server, wait_for


class HardTests(unittest.TestCase):
    def setUp(self):
        activity.enabled=True
        self.addCleanup(setattr,activity,'enabled',True)

    def session(self):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        with patch('follow_neo.runtime.session.resolve_model',return_value=(Path(folder.name)/'test.onnx',{})), \
             patch('follow_neo.runtime.session.NanoWorker'):
            s=FollowSession(folder.name,'127.0.0.1')
        self.addCleanup(s.stop)
        s.model_info={};s.target_policy={'dance_allowed':True};s.receiver.state='STREAMING'
        return s

    def runtime(self):
        ctx=mp.get_context('spawn');inputs=InputState(ctx)
        r=Runtime(inputs,Mailbox(ctx),FrameSlot(ctx,1000),FrameSlot(ctx,1000))
        r.session=self.session();r.session_id=1;r.identity=1
        server=Server();r.manual=ManualControl('127.0.0.1',server.connect,on_event=r.log_control)
        self.addCleanup(r.close)
        r.manual.start();inputs.pulse(set(),1);r.consume_input();r.manual.enable()
        wait_for(lambda:r.manual.enabled)
        return r,server

    def enter(self,r):
        r.inputs.pulse(set(),r.identity,origin='console')
        return r.dispatch('hard_enter',(r.identity,),{},r.session_id,time.monotonic(),r.inputs.fence.value)

    def test_log_barrier_no_disk_writes_or_diagnostic_samples(self):
        s=self.session()
        s.log.write('before',{})
        wait_for(lambda:'before' in s.log.path.read_text())
        raw=s.output/'test.raw';s.receiver.raw_handle=raw.open('wb')
        s.receiver.raw_handle.write(b'buffered-before-hard')
        s.set_hard(True)
        before=s.log.path.read_bytes();raw_before=raw.read_bytes()
        with patch.object(s.log.handle,'write',side_effect=AssertionError('disk write')):
            for _ in range(100):
                s.log.write('hard',{'not_serializable':object()})
                s.record_control('flight_command',{'server_acknowledged':True,'command':'rc'})
                s.scheduling.add('sample',123)
                s._runtime_health({})
            time.sleep(.05)
        self.assertEqual(s.log.path.read_bytes(),before)
        self.assertEqual(raw.read_bytes(),raw_before);self.assertEqual(raw_before,b'buffered-before-hard')
        self.assertEqual(s.flight_commands_sent,0);self.assertEqual(s.scheduling.values,{})
        self.assertIsNone(s.runtime_sample);self.assertEqual(s.log.queue.qsize(),0)
        s.set_hard(False);s.log.write('after',{})
        wait_for(lambda:'after' in s.log.path.read_text())
        self.assertNotIn('hard',[json.loads(row)['event'] for row in s.log.path.read_text().splitlines()])

    def test_hard_preserves_evidence_and_safety_callbacks(self):
        s=self.session();value=dict(measurement_time=123.,generation=s.generation)
        s.result.set(value);s.decision.set(value);settings=s.settings;generation=s.generation
        s.set_hard(True)
        self.assertIs(s.result.get(),value);self.assertIs(s.decision.get(),value)
        self.assertIs(s.settings,settings);self.assertEqual(s.generation,generation)
        s.record_control('control_error',{'error':'offline test'})
        self.assertIsNone(s.decision.get());self.assertTrue(s.control_reset_event.is_set())
        s.nano.reset.assert_called_once()
        s.set_hard(False)

    def test_entry_rejects_unfinished_recording_and_unready_model(self):
        s=self.session();s.model_info=None
        with self.assertRaisesRegex(RuntimeError,'model'):s.set_hard(True)
        s.model_info={};s.recorder=Mock();s.recorder.thread.is_alive.return_value=True
        with self.assertRaisesRegex(RuntimeError,'Recording'):s.set_hard(True)
        s.recorder.thread.join.assert_called_with(timeout=3.)
        self.assertFalse(s.hard);self.assertTrue(activity.enabled)
        s.recorder=None

    def test_worker_hard_blocks_gui_origin_and_stale_console_input(self):
        r,server=self.runtime();self.enter(r)
        self.assertTrue(r.hard);self.assertEqual(r.manual.mode,'DANCE')
        stamp=r.manual.updated
        r.inputs.pulse({'up'},1);r.consume_input()
        self.assertEqual(r.manual.updated,stamp);self.assertFalse(r.manual.keys)
        r.inputs.latest.put(dict(time=time.monotonic()-1,keys=['up'],identity=1,
                                fence=r.inputs.fence.value,origin='console'))
        r.consume_input();self.assertEqual(r.manual.updated,stamp)
        result=r.exit_hard()
        self.assertTrue(result['release_acknowledged']);self.assertFalse(r.manual.wanted)
        self.assertEqual(r.manual.mode,'MANUAL');self.assertTrue(activity.enabled)
        self.assertEqual(server.commands[-1],'disable')
        self.assertEqual(server.commands[-2],'rc 0 0 0 0')
        self.assertEqual(server.commands.count('enable'),1)

    def test_release_during_entry_cannot_start_dance(self):
        r,_=self.runtime();original=r.session.set_hard
        def cancel(enabled):
            original(enabled)
            if enabled:r.inputs.release()
        with patch.object(r.session,'set_hard',side_effect=cancel), \
             patch.object(r.manual,'start_dance',wraps=r.manual.start_dance) as start:
            with self.assertRaisesRegex(RuntimeError,'cancelled'):self.enter(r)
        start.assert_not_called();self.assertFalse(r.manual.wanted)
        self.assertFalse(r.hard);self.assertTrue(r.presentation_gate.is_set())

    def test_missing_heartbeat_still_neutralizes_tcp_in_hard(self):
        r,server=self.runtime();self.enter(r)
        wait_for(lambda:r.manual.communication_paused)
        wait_for(lambda:'rc 0 0 0 0' in server.commands)
        self.assertIn('heartbeat',r.manual.pause_reason.lower())
        self.assertTrue(r.exit_hard()['release_acknowledged'])

    def test_no_publication_while_hard_and_publication_resumes(self):
        r,_=self.runtime();self.enter(r)
        thread=threading.Thread(target=r.publish_loop,daemon=True);thread.start()
        try:
            before=r.snapshots.version.value
            time.sleep(.1);self.assertEqual(r.snapshots.version.value,before)
            self.assertEqual(r.live.token.value,0);self.assertEqual(r.analyzed.token.value,0)
            r.exit_hard();wait_for(lambda:r.snapshots.version.value>before)
        finally:r.stop.set();r.presentation_gate.set();thread.join(2)

    def test_one_shot_completion_and_video_fault_abort_without_gui(self):
        for fault in ('terminal','video'):
            with self.subTest(fault=fault):
                r,_=self.runtime();self.enter(r)
                if fault=='terminal':
                    r.session.decision.set(dict(continuous_dance=False,spacing_phase='STOPPED',
                        prediction_time=r.manual.dance_started+.01))
                else:r.session.receiver.state='ERROR'
                thread=threading.Thread(target=r.input_loop,daemon=True);thread.start()
                try:
                    wait_for(r.inputs.hard_abort.is_set);self.assertFalse(r.manual.wanted)
                finally:r.stop.set();thread.join(2);r.exit_hard();r.close()

    def test_console_space_focus_loss_and_exceptions_release_and_restore(self):
        for trigger in ('space','focus','error'):
            with self.subTest(trigger=trigger):
                client=Mock(alive=True);client.inputs=InputState(mp.get_context('spawn'))
                client.call.return_value={'release_acknowledged':True}
                console=Mock();console.focused.return_value=True
                console.stop_pressed.return_value=False
                if trigger=='space':console.stop_pressed.side_effect=[False,False,True]
                if trigger=='focus':console.focused.side_effect=[True,True,False]
                if trigger=='error':console.stop_pressed.side_effect=[False,OSError('input failed')]
                before=client.inputs.fence.value
                reason,confirmed=run_hard(client,7,9,console)
                self.assertTrue(confirmed);self.assertGreater(client.inputs.fence.value,before)
                self.assertEqual([c.args[0] for c in client.call.call_args_list],['hard_enter','hard_exit'])
                console.restore.assert_called_once()

    def test_console_does_not_enable_control_and_reports_unconfirmed_release(self):
        client=Mock(alive=True);client.inputs=InputState(mp.get_context('spawn'))
        client.call.side_effect=RuntimeError('backend unavailable')
        console=Mock();console.focused.return_value=True;console.stop_pressed.return_value=False
        reason,confirmed=run_hard(client,7,9,console)
        self.assertFalse(confirmed);console.restore.assert_called_once()
        self.assertNotIn('manual.enable',[c.args[0] for c in client.call.call_args_list])

    def test_console_rejects_unrelated_window_instead_of_renewing_authority(self):
        console=WindowsConsole.__new__(WindowsConsole)
        console.kernel=Mock();console.user=Mock();console.kernel.GetConsoleWindow.return_value=123
        console.user.GetForegroundWindow.return_value=456
        with self.assertRaisesRegex(RuntimeError,'dedicated CMD'):console.activate()
        console.kernel.SetConsoleMode.assert_not_called()

    def test_console_waits_for_async_focus_and_rejects_hidden_terminal_handle(self):
        console=WindowsConsole.__new__(WindowsConsole)
        console.kernel=Mock();console.user=Mock();console.handle=99;console.mode=Mock(value=7)
        console.kernel.GetConsoleWindow.return_value=123
        console.user.IsWindowVisible.return_value=False
        with self.assertRaisesRegex(RuntimeError,'Restart'):console.activate()
        console.user.SetForegroundWindow.assert_not_called()
        console.user.IsWindowVisible.return_value=True
        console.user.GetForegroundWindow.side_effect=[456,456,123,123]
        console.activate()
        self.assertTrue(console.focused())
        console.kernel.SetConsoleMode.assert_called_once()

    def test_gui_hides_only_after_console_has_keyboard_focus(self):
        client=Mock(alive=True);client.inputs=InputState(mp.get_context('spawn'))
        client.call.return_value={'release_acknowledged':True}
        state={'hidden':False}
        console=Mock();console.focused.return_value=True
        console.activate.side_effect=lambda:self.assertFalse(state['hidden'])
        def stop():
            return state['hidden']
        console.stop_pressed.side_effect=stop
        def hide():state['hidden']=True
        run_hard(client,7,9,console,on_ready=hide)
        self.assertTrue(state['hidden'])
        self.assertEqual([c.args[0] for c in client.call.call_args_list],['hard_enter','hard_exit'])

    @unittest.skipUnless(__import__('os').name=='nt','Windows console launcher')
    def test_normal_and_compatibility_launchers_supply_real_console(self):
        # Exercise real CMD quoting, argument forwarding and recursion protection.
        # The substitute application never connects to a phone or starts a worker.
        import os,shutil,subprocess,sys
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='Follow NEO console ') as folder:
            target=Path(folder);scripts=target/'.venv'/'Scripts';scripts.mkdir(parents=True)
            shutil.copy2(sys.executable,scripts/'python.exe')
            shutil.copy2(Path(sys.prefix)/'pyvenv.cfg',target/'.venv'/'pyvenv.cfg')
            for name in ('RUN_FOLLOW_NEO.cmd','RUN_FOLLOW_NEO_HARD.cmd'):
                shutil.copy2(root/name,target/name)
            (target/'run_lab.py').write_text('''import ctypes,json,sys
from pathlib import Path
from ctypes import wintypes
k=ctypes.WinDLL('kernel32');u=ctypes.WinDLL('user32')
k.GetConsoleWindow.restype=wintypes.HWND
u.IsWindowVisible.argtypes=[wintypes.HWND]
hwnd=k.GetConsoleWindow()
report=dict(visible=bool(u.IsWindowVisible(hwnd)),args=sys.argv[1:])
sys.path.insert(0,PROJECT_ROOT)
import tkinter as tk
from follow_neo.ui.hard_console import WindowsConsole
r=tk.Tk();r.title('Follow NEO offline console test');r.update();r.focus_force();r.update()
c=None
try:
    c=WindowsConsole();c.activate()
    report['focused_before_hide']=c.focused()
    r.withdraw();r.update_idletasks()
    report['focused_after_hide']=c.focused()
except Exception as e:report['error']=str(e)
finally:
    if c:c.restore()
    r.destroy()
Path('probe.json').write_text(json.dumps(report))
'''.replace('PROJECT_ROOT',repr(str(root))),encoding='utf-8')
            env=dict(os.environ,WT_SESSION='offline-terminal-test')
            env.pop('FOLLOW_NEO_CONSOLE_HOSTED',None)
            for name,prefix in (('RUN_FOLLOW_NEO.cmd',[]),('RUN_FOLLOW_NEO_HARD.cmd',['--runtime','process'])):
                with self.subTest(launcher=name):
                    result=subprocess.run([os.environ['ComSpec'],'/d','/c',name,
                                           '--test-value','two words'],cwd=target,env=env,
                                          creationflags=subprocess.CREATE_NO_WINDOW,timeout=20,
                                          capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                    self.assertTrue((target/'probe.json').exists(),result.stdout+result.stderr+str({
                        p.name:p.read_text(errors='replace') for p in target.glob('*.txt')}))
                    report=json.loads((target/'probe.json').read_text())
                    self.assertTrue(report['visible'])
                    self.assertTrue(report.get('focused_before_hide'),report)
                    self.assertTrue(report.get('focused_after_hide'),report)
                    self.assertEqual(report['args'],prefix+['--test-value','two words'])
                    (target/'probe.json').unlink()

    def test_gui_refresh_is_completely_dormant(self):
        w=FollowLabWindow.__new__(FollowLabWindow);w.hard_active=True;w.root=Mock();w.runtime=Mock()
        w.reconcile_keyboard=Mock();w.ui_timing=Mock()
        w.refresh()
        w.root.after.assert_not_called();w.runtime.poll.assert_not_called()
        w.reconcile_keyboard.assert_not_called();w.ui_timing.add.assert_not_called()

    def test_gui_spawn_console_dance_space_and_return_end_to_end(self):
        import tkinter as tk
        from follow_neo.runtime.client import RuntimeClient,RemoteManual
        from process_fixture import hard_fixture_worker
        from test_process_runtime import TCPServer
        with tempfile.TemporaryDirectory() as folder,TCPServer() as server:
            client=RuntimeClient(worker_target=hard_fixture_worker);root=tk.Tk();root.withdraw()
            try:
                w=FollowLabWindow(root,folder,runtime=client)
                w.ip.set('127.0.0.1');w.compute.set('CPU');w.connect()
                manual=RemoteManual(client,'127.0.0.1',require_target_policy=True);w.manual=manual
                manual.start();manual.update(set());manual.enable()
                def active():
                    manual.update(set());client.poll();return manual.control_snapshot().get('active')
                wait_for(active)
                root.deiconify();root.update_idletasks()
                marker=[];root.after(0,lambda:marker.append('GUI timer'))
                case=self;started=time.monotonic()
                class Console:
                    entered=None
                    def activate(self):case.assertEqual(root.state(),'normal')
                    def focused(self):return True
                    def restore(self):pass
                    def stop_pressed(self):
                        if time.monotonic()-started>3:raise AssertionError('HARD failed to enter')
                        if manual.mode!='DANCE':return False
                        if self.entered is None:
                            self.entered=time.monotonic();self.version=client.snapshots.version.value
                            self.bytes=(w.session.output/'events.jsonl').read_bytes()
                        case.assertEqual(client.snapshots.version.value,self.version)
                        case.assertEqual((w.session.output/'events.jsonl').read_bytes(),self.bytes)
                        return time.monotonic()-self.entered>.16
                console=Console()
                with patch('follow_neo.ui.hard_console.WindowsConsole',return_value=console):w.start_hard()
                self.assertIsNotNone(console.entered);self.assertEqual(marker,[])
                self.assertFalse(w.hard_active);self.assertFalse(w.real_time)
                self.assertEqual(root.state(),'normal');self.assertIn('control released',w.notice.get())
                self.assertTrue(any(line.startswith('rc ') and any(float(v) for v in line.split()[1:])
                                    for _,line in server.commands))
                self.assertEqual([v for _,v in server.commands].count('enable'),1)
                self.assertEqual(server.commands[-1][1],'disable')
                root.update();self.assertEqual(marker,['GUI timer'])
                self.assertFalse(manual.wanted)
            finally:
                client.close()
                for pending in root.tk.call('after','info'):root.after_cancel(pending)
                root.destroy()

    def test_diagnostic_clocks_off_tracking_and_stale_guards_are_identical(self):
        def replay(enabled):
            activity.enabled=enabled
            c=FollowController(tracker_backend='botsort',target_class_id=0,target_class_name='neo')
            s=Settings();out=[]
            for i in range(12):
                stamp=10+i*.1
                d=[Detection(Box(600+i*3,300,660+i*3,340),.9,0,'neo')]
                c.observe(d,stamp,stamp+.02,i+1,(1280,720),s)
                v=c.tick(stamp+.02,1280,720,stamp,s)
                out.append((v['state'],v['intent'],v['measurement_time'],dance_command(v,stamp+.03,9)))
            v=c.tick(14,1280,720,11.1,s)
            out.append((v['state'],v['intent'],v['measurement_time'],dance_command(v,14,9)))
            return out
        normal=replay(True)
        with patch('follow_neo.diagnostics.activity.time.perf_counter',side_effect=AssertionError('profiling clock')):
            hard=replay(False)
        self.assertEqual(normal,hard)
        self.assertTrue(all(v==0 for v in hard[-1][1].values()))


if __name__=='__main__':unittest.main()
