"""Gimbal integration uses fake transports only: no hardware or TCP clients."""
from collections import deque
import json
import threading
import time

import pytest

from follow_neo.gimbal_keyboard import GimbalKeyboard, pitch_step_command
from follow_neo.telemetry import HeadingReceiver


class FakeSocket:
    def __init__(self, chunks=(), send_error=None):
        self.chunks = deque(chunks)
        self.commands = []
        self.send_error = send_error
        self.closed = False
        self.timeouts = []

    def settimeout(self, value):
        self.timeouts.append(value)

    def sendall(self, value):
        self.commands.append(value)
        if self.send_error:
            raise self.send_error

    def recv(self, _size):
        if not self.chunks:
            raise TimeoutError('fake timeout')
        item = self.chunks.popleft()
        if isinstance(item, Exception):
            raise item
        return item

    def shutdown(self, _how):
        self.closed = True

    def close(self):
        self.closed = True


def connected(chunks=(), on_event=None):
    def forbidden_connector(*args, **kwargs):
        raise AssertionError('No second client may be opened')
    receiver = HeadingReceiver('fake-host', connector=forbidden_connector)
    sock = FakeSocket(chunks)
    # The telemetry worker owns this socket in production.
    receiver.sock = sock
    return receiver, sock, GimbalKeyboard(receiver, on_event=on_event)


def finish(receiver, sock, buffer=b''):
    return receiver._run_gimbal_step(sock, buffer)


def test_constructing_enabling_and_disabling_do_not_send_or_connect():
    receiver, sock, keyboard = connected()
    assert keyboard.set_enabled(True)
    keyboard.disable()
    keyboard.stop()
    assert sock.commands == []
    assert not sock.closed  # The shared heading connection belongs to receiver.
    assert not keyboard.set_enabled(True)


@pytest.mark.parametrize('direction,pitch', [(1, 3.0), (-1, -3.0)])
def test_sdk_relative_pitch_payload_has_no_yaw_or_roll_motion(direction, pitch):
    command = pitch_step_command(direction)
    assert command.startswith(b'action Gimbal RotateByAngle ')
    assert command.endswith(b'\r\n') and command.count(b'\n') == 1
    payload = json.loads(command.decode().split(' ', 3)[3])
    assert payload == dict(mode=0, pitch=pitch, roll=0.0, yaw=0.0,
                          pitchIgnored=False, rollIgnored=True, yawIgnored=True,
                          duration=0.3, jointReferenceUsed=False, timeout=0)


@pytest.mark.parametrize('direction', [0, 3, True, 1.0, 'up'])
def test_invalid_directions_are_not_serialized(direction):
    with pytest.raises(ValueError):
        pitch_step_command(direction)


def test_disabled_and_disconnected_steps_are_not_queued():
    receiver, sock, keyboard = connected()
    assert not keyboard.request_step(1)
    receiver.sock = None
    assert not keyboard.set_enabled(True)
    assert not keyboard.request_step(1)
    assert not receiver.gimbal_available()
    assert sock.commands == []


def test_one_slot_drops_backlogged_steps_instead_of_replaying_them():
    receiver, sock, keyboard = connected([b'Gimbal RotateByAngle success\r\n'])
    assert keyboard.set_enabled(True)
    assert keyboard.request_step(1)
    assert keyboard.read()['busy']
    assert not keyboard.request_step(-1)
    finish(receiver, sock)
    finish(receiver, sock)
    assert len(sock.commands) == 1
    assert json.loads(sock.commands[0].decode().split(' ', 3)[3])['pitch'] == 3
    assert not keyboard.read()['busy']


def test_focus_loss_disable_cancels_pending_step_without_closing_shared_socket():
    receiver, sock, keyboard = connected()
    keyboard.set_enabled(True)
    keyboard.request_step(1)
    keyboard.disable()
    finish(receiver, sock)
    assert sock.commands == []
    assert not sock.closed
    assert not keyboard.read()['enabled']
    assert not keyboard.read()['busy']


def test_short_lived_request_expires_before_send():
    receiver, sock, keyboard = connected()
    keyboard.set_enabled(True)
    keyboard.request_step(1)
    receiver._gimbal_request.expires_at = 0
    finish(receiver, sock)
    assert sock.commands == []
    assert 'expired' in keyboard.read()['status']


def test_fragmented_ack_skips_unrelated_keys_and_never_claims_measured_pitch():
    events = []
    receiver, sock, keyboard = connected([
        b'FlightController AircraftAttitude {yaw=10}\r\nGimbal Rota',
        b'teByAngle suc', b'cess\r\nextra\r\n'],
        on_event=lambda name, data: events.append((name, data)))
    receiver.sample = dict(yaw_deg=5)
    keyboard.set_enabled(True)
    keyboard.request_step(-1)
    assert finish(receiver, sock) == b'extra\r\n'
    result = keyboard.read()
    assert result['last_acknowledged']
    assert 'measured angle unavailable' in result['status']
    assert result['requested_pitch_deg'] == -3
    assert events[0][1]['measured_pitch_deg'] is None
    assert receiver.get() is None  # Never keep heading fresh through a gimbal wait.


@pytest.mark.parametrize('reply', [
    b'Gimbal RotateByAngle DJIError{code=NOT_SUPPORTED}\r\n',
    b'Gimbal RotateByAngle could not cast parameter\r\n',
    b'Gimbal RotateByAngle unsuccessful\r\n',
    b'Unknown key name: RotateByAngle\r\n',
])
def test_negative_ack_never_counts_as_success(reply):
    receiver, sock, keyboard = connected([reply])
    keyboard.set_enabled(True)
    keyboard.request_step(1)
    finish(receiver, sock)
    state = keyboard.read()
    assert not state['last_acknowledged']
    assert not state['completion_unknown']
    assert state['error'].startswith('Gimbal rejected:')


def test_timeout_never_retries_and_late_fragmented_ack_is_not_reused():
    receiver, sock, keyboard = connected([b'Gimbal RotateBy', TimeoutError('lost ACK')])
    keyboard.set_enabled(True)
    keyboard.request_step(1)
    partial = finish(receiver, sock)
    assert partial == b'Gimbal RotateBy'
    result = keyboard.read()
    assert result['completion_unknown']
    assert not result['enabled']
    assert not keyboard.request_step(-1)
    assert not keyboard.set_enabled(True)  # The unresolved ACK has no request ID.
    finish(receiver, sock)
    assert len(sock.commands) == 1
    sock.chunks.extend([b'Angle success\r\nFlightController CompassHeading 8\r\n'])
    line, partial = receiver._response(sock, partial, 'FlightController CompassHeading ',
                                       time.monotonic() + 1)
    assert line == 'FlightController CompassHeading 8'
    assert not keyboard.read()['last_acknowledged']  # Late ACK is drained, not displayed as measured motion.
    assert keyboard.set_enabled(True)  # Requires this NEW explicit enable.
    assert len(sock.commands) == 1
    sock.chunks.append(b'Gimbal RotateByAngle success\r\n')
    assert keyboard.request_step(-1)
    finish(receiver, sock, partial)
    assert len(sock.commands) == 2


def test_partial_send_failure_is_ambiguous_and_never_replayed_after_reconnect():
    receiver, sock, keyboard = connected()
    sock.send_error = OSError('write interrupted')
    keyboard.set_enabled(True)
    keyboard.request_step(1)
    with pytest.raises(OSError):
        finish(receiver, sock)
    assert keyboard.read()['completion_unknown']
    next_socket = FakeSocket()
    receiver.sock = next_socket
    finish(receiver, next_socket)
    assert next_socket.commands == []
    assert not keyboard.set_enabled(True)


def test_disconnect_discards_unsent_step_even_if_new_socket_arrives():
    receiver, sock, keyboard = connected()
    keyboard.set_enabled(True)
    keyboard.request_step(-1)
    receiver._abandon_gimbal('Query connection ended')
    receiver.sock = FakeSocket()
    finish(receiver, receiver.sock)
    assert receiver.sock.commands == []
    assert sock.commands == []
    assert not keyboard.read()['enabled']
    assert 'unsent' in keyboard.read()['error']


def test_nonblocking_stop_during_inflight_action_does_not_close_heading_socket():
    receiver, sock, keyboard = connected()
    receiving = threading.Event()
    release = threading.Event()

    def blocking_recv(_size):
        receiving.set()
        assert release.wait(2)
        return b'Gimbal RotateByAngle success\r\n'

    sock.recv = blocking_recv
    keyboard.set_enabled(True)
    keyboard.request_step(1)
    worker = threading.Thread(target=finish, args=(receiver, sock))
    worker.start()
    try:
        assert receiving.wait(1)
        started = time.monotonic()
        keyboard.stop()
        assert time.monotonic() - started < .1
        assert not sock.closed
        assert keyboard.read()['busy']
        assert not keyboard.request_step(-1)
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    assert not keyboard.read()['enabled']
    assert not keyboard.read()['busy']
    assert len(sock.commands) == 1


def test_disable_during_precommit_socket_setup_prevents_send_and_ambiguity():
    receiver, sock, keyboard = connected()
    setting_up = threading.Event()
    release = threading.Event()

    def blocked_timeout(value):
        setting_up.set()
        assert release.wait(2)

    sock.settimeout = blocked_timeout
    keyboard.set_enabled(True)
    keyboard.request_step(1)
    worker = threading.Thread(target=finish, args=(receiver, sock))
    worker.start()
    try:
        assert setting_up.wait(1)
        started = time.monotonic()
        keyboard.disable()
        assert time.monotonic() - started < .1
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    assert sock.commands == []
    assert not keyboard.read()['busy']
    assert not keyboard.read()['completion_unknown']
    assert receiver.gimbal_available()


def test_socket_setup_failure_is_unsent_not_unknown_completion():
    receiver, sock, keyboard = connected()

    def failed_timeout(value):
        raise OSError('socket no longer valid')

    sock.settimeout = failed_timeout
    keyboard.set_enabled(True)
    keyboard.request_step(1)
    with pytest.raises(OSError):
        finish(receiver, sock)
    assert sock.commands == []
    assert not keyboard.read()['completion_unknown']
    assert 'unsent step discarded' in keyboard.read()['error']
    assert not receiver._gimbal_unresolved


def test_callback_errors_do_not_kill_query_owner():
    def broken(*_args):
        raise RuntimeError('GUI/logger callback failed')
    receiver, sock, keyboard = connected([b'Gimbal RotateByAngle success\r\n'], broken)
    keyboard.set_enabled(True)
    keyboard.request_step(1)
    finish(receiver, sock)
    assert keyboard.read()['last_acknowledged']


def test_oversized_response_fails_without_retry():
    receiver, sock, keyboard = connected([b'x' * 17000])
    keyboard.set_enabled(True)
    keyboard.request_step(1)
    with pytest.raises(ValueError, match='too long'):
        finish(receiver, sock)
    assert keyboard.read()['completion_unknown']
    assert len(sock.commands) == 1


def test_real_heading_worker_serializes_action_on_its_only_fake_connection():
    class QuerySocket(FakeSocket):
        def __init__(self):
            super().__init__()
            self.condition = threading.Condition()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

        def sendall(self, value):
            with self.condition:
                super().sendall(value)
                if value.startswith(b'get FlightController '):
                    self.chunks.extend([b'FlightController AircraftAttitude {yaw=',
                                        b'12.5,pitch=0,roll=0}\r\n'])
                else:
                    self.chunks.extend([b'Gimbal RotateBy', b'Angle success\r\n'])
                self.condition.notify_all()

        def recv(self, _size):
            with self.condition:
                if not self.chunks and not self.closed:
                    self.condition.wait(1)
                return b'' if self.closed else self.chunks.popleft()

        def shutdown(self, _how):
            self.close()

        def close(self):
            with self.condition:
                self.closed = True
                self.condition.notify_all()

    sock = QuerySocket()
    connections = []

    def connect(address, timeout):
        connections.append(address)
        assert address == ('fake-host', 9997)
        return sock

    def wait_until(condition):
        deadline = time.monotonic() + 2
        while not condition() and time.monotonic() < deadline:
            time.sleep(.005)
        assert condition()

    receiver = HeadingReceiver('fake-host', connector=connect)
    keyboard = GimbalKeyboard(receiver)
    receiver.start()
    try:
        wait_until(lambda: receiver.get() is not None)
        assert all(c.startswith(b'get FlightController ') for c in sock.commands)
        assert keyboard.set_enabled(True)
        assert keyboard.request_step(1)
        wait_until(lambda: keyboard.read()['last_acknowledged'])
        wait_until(lambda: receiver.get() is not None)
        assert receiver.get()['yaw_deg'] == 12.5
        actions = [c for c in sock.commands if c.startswith(b'action ')]
        assert actions == [pitch_step_command(1)]
        assert connections == [('fake-host', 9997)]
    finally:
        keyboard.stop()
        receiver.stop()
    assert not receiver.thread.is_alive()
