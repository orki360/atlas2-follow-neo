"""Foreground Windows console input. No logging, rendering or measurements."""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor


class WindowsConsole:
    def __init__(self):
        if os.name!='nt' or not sys.stdin.isatty():
            raise RuntimeError('HARD requires RUN_FOLLOW_NEO.cmd in an interactive Windows console')
        import ctypes
        from ctypes import wintypes
        self.ctypes=ctypes;self.user=ctypes.WinDLL('user32',use_last_error=True)
        self.kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        self.kernel.GetConsoleWindow.restype=wintypes.HWND
        self.user.GetForegroundWindow.restype=wintypes.HWND
        self.user.SetForegroundWindow.argtypes=[wintypes.HWND]
        self.user.ShowWindow.argtypes=[wintypes.HWND,wintypes.INT]
        self.user.IsWindowVisible.argtypes=[wintypes.HWND]
        self.kernel.GetStdHandle.restype=wintypes.HANDLE
        self.kernel.GetConsoleMode.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        self.kernel.SetConsoleMode.argtypes=[wintypes.HANDLE,wintypes.DWORD]
        self.handle=self.kernel.GetStdHandle(-10);self.mode=wintypes.DWORD()
        if not self.kernel.GetConsoleMode(self.handle,ctypes.byref(self.mode)):
            raise RuntimeError('Cannot read console input mode')

    def activate(self):
        hwnd=self.kernel.GetConsoleWindow()
        if not hwnd or not self.user.IsWindowVisible(hwnd):
            raise RuntimeError('Restart with RUN_FOLLOW_NEO.cmd to open the dedicated CMD window')
        self.user.ShowWindow(hwnd,9)  # Restore a minimized dedicated console.
        self.user.SetForegroundWindow(hwnd)
        # SetForegroundWindow may finish asynchronously across GUI/console threads.
        # This bounded handoff runs BEFORE HARD, never extending a control heartbeat.
        deadline=time.monotonic()+.15
        current=self.user.GetForegroundWindow()
        while current!=hwnd and time.monotonic()<deadline:
            time.sleep(.005)
            current=self.user.GetForegroundWindow()
        if not current:raise RuntimeError('Console must be in the foreground')
        if current!=hwnd:
            # A Terminal HWND can contain another tab/pane. It cannot prove
            # that THIS console owns the keyboard, so never renew authority.
            raise RuntimeError('Cannot focus dedicated CMD window; restart with RUN_FOLLOW_NEO.cmd')
        self.foreground=current
        # Disable selection freezes and line/echo input. Restore exactly on exit.
        if not self.kernel.SetConsoleMode(self.handle,(self.mode.value|0x80)&~(0x40|0x02|0x04)):
            raise RuntimeError('Cannot configure console input')

    def focused(self):return self.user.GetForegroundWindow()==self.foreground

    def stop_pressed(self):
        import msvcrt
        stop=False
        while msvcrt.kbhit():
            key=msvcrt.getwch()
            if key in (' ','\x1b','q','Q','\x03'):stop=True
        return stop or bool(self.user.GetAsyncKeyState(0x20)&0x8000)

    def restore(self):self.kernel.SetConsoleMode(self.handle,self.mode.value)


def run_hard(client,session_id,manual_id,console,*,on_ready=None):
    """Only this foreground input loop renews the heartbeat; RPC threads cannot."""
    reason='Space';entered=False
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='hard-lifecycle')
    try:
        console.activate()
        if not console.focused():raise RuntimeError('Console is not focused')
        if console.stop_pressed():raise RuntimeError('Cancelled before entry')
        if on_ready:on_ready()
        client.inputs.hard_abort.clear()
        client.inputs.pulse(set(),manual_id,origin='console')
        future=pool.submit(client.call,'hard_enter',manual_id,identity=session_id,timeout=20.)
        while True:
            if not console.focused():reason='Console focus lost';break
            if console.stop_pressed():reason='Space / release';break
            if client.inputs.hard_abort.is_set():reason='Control or video unavailable';break
            if not client.alive:raise RuntimeError(client.error or 'Runtime unavailable')
            client.inputs.pulse(set(),manual_id,origin='console')
            if future.done() and not entered:future.result();entered=True
            time.sleep(.02)
    except (Exception,KeyboardInterrupt) as exc:
        reason=str(exc) or 'Interrupted'
    finally:
        client.inputs.release()  # Immediate cancellation, even while entry is queued.
        try:
            result=pool.submit(client.call,'hard_exit',identity=session_id,timeout=12.).result(timeout=35.)
            confirmed=result['release_acknowledged']
        except (Exception,KeyboardInterrupt):
            confirmed=False
        finally:
            # A pre-HARD mailbox snapshot must not repaint enabled control after
            # the acknowledged release. The next fresh GUI snapshot will apply.
            client.minimum_snapshot_time=time.monotonic()
            pool.shutdown(wait=False,cancel_futures=True);console.restore()
    return reason,confirmed
