import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import socket
import threading
import time
import unittest
from unittest.mock import Mock
from follow_neo.manual import ManualControl, movement
from follow_neo.gui import normalized_event_key


class Server:
    def __init__(self, reject=None, before_reply=None):
        self.client, self.server = socket.socketpair()
        self.commands = []
        self.reject = reject
        self.before_reply = before_reply
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def connect(self, address, timeout):
        assert address == ('127.0.0.1', 9998)
        return self.client

    def run(self):
        try:
            with self.server.makefile('rb') as stream:
                for line in stream:
                    cmd = line.decode().strip()
                    self.commands.append(cmd)
                    if self.before_reply is not None:
                        self.before_reply(cmd)
                    self.server.sendall(b'denied\r\n' if cmd == self.reject else b'success\r\n')
        except OSError:
            pass
        finally:
            self.server.close()


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('Timed out')
        time.sleep(.01)


class ManualTests(unittest.TestCase):
    def start_control(self, reject=None, before_reply=None, on_event=None):
        server = Server(reject, before_reply)
        control = ManualControl('127.0.0.1', server.connect, on_event=on_event)
        # This fixture tests the original explicit cap values. Update 9's
        # requested 50% yaw/vertical defaults are verified in test_update9.
        control.set_axis_limits(dict(yaw=.15,vertical=.015,roll=.015,forward=.015))
        self.addCleanup(lambda: (control.stop(), control.thread.join(3)))
        control.start()
        return server, control

    def test_axes_and_opposites(self):
        self.assertEqual(movement({'d','w','right','up'}), (1.,1.,1.,1.))
        self.assertEqual(movement({'a','s','left','down'}), (-1.,-1.,-1.,-1.))
        self.assertEqual(movement({'w','s','a','d','up','down','left','right'}), (0,0,0,0))

    def test_explicit_enable_release_and_actions(self):
        server, control = self.start_control()
        control.update({'w'})
        time.sleep(.08)
        self.assertEqual(server.commands, [])
        self.assertFalse(control.request('takeoff'))
        control.enable()
        wait_for(lambda: any(cmd == 'rc 0.0000 0.0150 0.0000 0.0000' for cmd in server.commands))
        self.assertTrue(control.request('takeoff'))
        wait_for(lambda: 'takeoff' in server.commands)
        control.update(set())
        wait_for(lambda: 'rc 0.0000 0.0000 0.0000 0.0000' in server.commands)
        control.release()
        wait_for(lambda: 'disable' in server.commands)
        i = server.commands.index('disable')
        self.assertEqual(server.commands[i-1], 'rc 0 0 0 0')
        self.assertEqual(server.commands.count('takeoff'), 1)

    def test_missing_ui_heartbeat_releases(self):
        received = threading.Event()
        allow_reply = threading.Event()
        completed = threading.Event()
        def before_reply(command):
            if command == 'disable':
                received.set()
                allow_reply.wait(2)
        def on_event(event, data):
            if event == 'control_released':
                completed.set()
        server, control = self.start_control(before_reply=before_reply, on_event=on_event)
        try:
            control.update({'up'}); control.enable()
            wait_for(lambda: control.enabled)
            self.assertTrue(received.wait(3), 'Missing heartbeat did not send disable')
            # Receipt by the server precedes acknowledgement and the local
            # enabled=False update. Hold the reply to exercise this ordering
            # deterministically, including on Windows.
            self.assertFalse(completed.is_set())
            self.assertFalse(control.wanted)
            self.assertEqual(control.keys, set())
            self.assertEqual(server.commands[-2:], ['rc 0 0 0 0', 'disable'])
            allow_reply.set()
            self.assertTrue(completed.wait(3), 'Acknowledged release did not complete')
            self.assertFalse(control.enabled)
        finally:
            allow_reply.set()

    def test_enable_failure_never_moves(self):
        server, control = self.start_control('enable')
        control.update({'up'}); control.enable()
        wait_for(lambda: not control.thread.is_alive())
        self.assertIn('denied', control.status)
        self.assertFalse(control.enabled)
        self.assertFalse(any('.0150' in cmd for cmd in server.commands))

    def test_close_releases(self):
        server, control = self.start_control()
        control.update({'a'}); control.enable()
        wait_for(lambda: control.enabled)
        control.stop(); control.thread.join(3)
        wait_for(lambda: 'disable' in server.commands)
        self.assertFalse(control.thread.is_alive())


class KeyboardTests(unittest.TestCase):
    def setUp(self):
        from follow_neo.gui import FollowLabWindow
        self.window = FollowLabWindow.__new__(FollowLabWindow)
        self.window.root = Mock()
        self.window.canvas = Mock()
        self.window.manual = Mock()
        self.window.pressed = set()
        self.window.key_releases = {}
        self.window.notice = Mock()
        self.window.session = None

    def event(self, key, widget='Canvas', keycode=None):
        event = Mock(keysym=key)
        event.keycode=keycode
        event.widget.winfo_class.return_value = widget
        return event

    def test_windows_physical_keys_work_under_hebrew_layout(self):
        hebrew={'w':(87,'צ'),'s':(83,'ד'),'a':(65,'ש'),'d':(68,'ג')}
        for expected,(keycode,keysym) in hebrew.items():
            with self.subTest(expected=expected):
                self.assertEqual(normalized_event_key(self.event(keysym,keycode=keycode),'win32'),expected)

    def test_hebrew_layout_wasd_reaches_manual_control(self):
        import follow_neo.gui as gui
        original=gui.sys.platform
        try:
            gui.sys.platform='win32'
            for keycode,keysym in ((87,'צ'),(83,'ד'),(65,'ש'),(68,'ג')):
                self.window.key_press(self.event(keysym,keycode=keycode))
            self.window.manual.update.assert_called_with({'w','s','a','d'})
        finally:
            gui.sys.platform=original

    def test_text_entry_does_not_fly(self):
        self.window.key_press(self.event('f', 'TEntry'))
        self.window.manual.request.assert_not_called()
        self.assertEqual(self.window.pressed, set())

    def test_takeoff_is_once_per_press(self):
        self.window.key_press(self.event('f'))
        self.window.key_press(self.event('f'))
        self.window.manual.request.assert_called_once_with('takeoff')

    def test_key_release_clears_motion(self):
        self.window.key_press(self.event('w'))
        self.window.key_release(self.event('w'))
        callback = self.window.root.after_idle.call_args.args[0]
        callback()
        self.window.manual.update.assert_called_with(set())

    def test_focus_leaving_canvas_releases(self):
        self.window.pressed.add('w')
        self.window.root.focus_get.return_value = None
        self.window.focus_out(Mock())
        self.window.root.after_idle.call_args.args[0]()
        self.window.manual.release.assert_called_once()
        self.assertEqual(self.window.pressed, set())

    def test_focus_entering_canvas_preserves_enable(self):
        self.window.root.focus_get.return_value = self.window.canvas
        self.window.focus_out(Mock())
        self.window.root.after_idle.call_args.args[0]()
        self.window.manual.release.assert_not_called()


if __name__ == '__main__':
    unittest.main()
