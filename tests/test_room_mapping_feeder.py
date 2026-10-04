"""Real passive session ingress remains independent of controller map IO.

Frames originate from the existing fake receiver; only disk/map-lock boundaries
are gated. The actual MappingSession queue and worker run without hardware.
"""
import json
import threading
import time

import pytest

from follow_neo import room_mapping
from test_room_mapping_controller import (
    ExistingReceiver, calibration_path, controller, no_network, wait_until,
)


def assert_joined(instance):
    instance.thread.join(4.)
    assert not instance.busy
    assert instance._frame_feeder is None or not instance._frame_feeder.is_alive()
    assert not instance.mapping.status()['worker_alive']
    assert instance.read()[0]['log_closed']


def test_live_log_records_selected_mapping_policy(controller, calibration_path):
    receiver = ExistingReceiver()
    receiver.publish(1)
    controller.start_live(receiver, calibration_path)
    wait_until(lambda: controller.mapping.status()['processed_frames'] == 1)
    # Read the actual session's dataclass and public mode, not a test stub.
    config = controller.mapping._config
    assert config.adaptive_keyframes
    assert config.keyframe_redundancy_min_points == 100
    assert not controller.mapping.status()['map_update']['enabled']
    controller.stop()
    assert_joined(controller)
    files = list(controller.output_root.rglob('mapping.jsonl'))
    assert len(files) == 1
    events = [json.loads(line) for line in files[0].read_text(encoding='utf-8').splitlines()]
    start = next(event for event in events if event['event'] == 'session_start')
    assert start['data']['slam_config']['adaptive_keyframes'] is True
    assert start['data']['slam_config']['keyframe_redundancy_min_points'] == 100


@pytest.mark.parametrize('operation',['snapshot','save'])
def test_map_lock_blocked_display_or_save_does_not_block_new_frame_ingress(
        controller,calibration_path,monkeypatch,operation):
    session_class=room_mapping.mapping_import('mapping_session').MappingSession
    original=getattr(session_class,operation)
    entered,release=threading.Event(),threading.Event()
    def blocked(self,*args,**kwargs):
        if not entered.is_set():
            with self._map_lock:
                entered.set()
                if not release.wait(4.):
                    raise TimeoutError('Test map-lock gate was not released')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(session_class,operation,blocked)
    receiver=ExistingReceiver();first=receiver.publish(1)
    controller.start_live(receiver,calibration_path)
    try:
        if operation=='save':
            wait_until(lambda:controller.mapping.status()['submitted_frames']>=1)
            controller.request_save()
        assert entered.wait(2.)
        for frame_id in (2,3,4):
            receiver.publish(frame_id)
            wait_until(lambda:controller.mapping.status()['last_submitted_source_frame_id']==frame_id)
        # No archive lock release was needed for the real ingress counters to
        # advance. Exactly one feeder supplied these increasing source IDs.
        state=controller.mapping.status()
        assert state['submitted_frames']==4
        assert state['rejected_order_frames']==0
        assert state['pending_frame_id']==4
        assert controller._frame_feeder.is_alive() and not controller._frame_feeder.daemon
        assert first.image.sum()==0  # Original receiver arrays remain untouched.
        started=time.monotonic();controller.stop()
        assert time.monotonic()-started<.2
    finally:
        release.set()
    assert_joined(controller)
    assert controller.read()[0]['state']=='stopped'
    assert receiver.state=='STREAMING'


@pytest.mark.parametrize('failure',['receiver','wait','submit','base_exception'])
def test_feeder_failure_is_visible_even_while_snapshot_is_blocked(
        controller,calibration_path,monkeypatch,failure):
    session_class=room_mapping.mapping_import('mapping_session').MappingSession
    original_snapshot=session_class.snapshot
    original_submit=session_class.submit_frame
    entered,release=threading.Event(),threading.Event()
    def blocked(self):
        if not entered.is_set():
            entered.set()
            if not release.wait(4.):
                raise TimeoutError('Test snapshot gate was not released')
        return original_snapshot(self)
    def submit(self,image,frame_id,decoded_at):
        if frame_id==2 and failure in ('submit','base_exception'):
            if failure=='base_exception':
                raise SystemExit('fixture feeder exit')
            raise ValueError('fixture invalid decoded frame')
        return original_submit(self,image,frame_id,decoded_at)
    monkeypatch.setattr(session_class,'snapshot',blocked)
    monkeypatch.setattr(session_class,'submit_frame',submit)
    receiver=ExistingReceiver();receiver.publish(1)
    original_wait=receiver.frames.wait_next
    fail_wait=threading.Event()
    def wait(version,timeout):
        if fail_wait.is_set():
            raise RuntimeError('fixture latest-frame wait failed')
        return original_wait(version,timeout)
    monkeypatch.setattr(receiver.frames,'wait_next',wait)
    controller.start_live(receiver,calibration_path)
    try:
        assert entered.wait(2.)
        if failure=='receiver':
            receiver.error='fixture video disconnected';receiver.state='ERROR'
        elif failure=='wait':
            fail_wait.set()
        else:
            receiver.publish(2)
        wait_until(lambda:controller.read()[0]['state']=='error')
        status=controller.read()[0]
        assert 'fixture' in status['message'] and not status['pose_valid']
        assert controller.busy  # Snapshot is still blocked, but error is visible.
        assert not controller.stop_event.is_set()  # Failure is not a user stop.
    finally:
        release.set()
    assert_joined(controller)
    assert controller.read()[0]['state']=='error'
    assert (controller.output/'map.json').is_file()
    rows=[json.loads(line) for line in (controller.output/'mapping.jsonl').read_text().splitlines()]
    assert any(row['event']=='error' and 'fixture' in row['data']['message'] for row in rows)
    assert rows[-1]['event']=='session_end' and rows[-1]['data']['state']=='error'


def test_stop_joins_feeder_before_session_stop_and_drops_wait_result(
        controller,calibration_path,monkeypatch):
    session_class=room_mapping.mapping_import('mapping_session').MappingSession
    original_stop=session_class.stop
    receiver=ExistingReceiver();receiver.publish(1)
    original_wait=receiver.frames.wait_next
    waiting,release,session_stopped=threading.Event(),threading.Event(),threading.Event()
    def wait(version,timeout):
        if version>=1:
            waiting.set()
            if not release.wait(3.):
                raise TimeoutError('Test receiver wait gate was not released')
        return original_wait(version,timeout)
    def stop(self,*args,**kwargs):
        assert not controller._frame_feeder.is_alive()
        session_stopped.set()
        return original_stop(self,*args,**kwargs)
    monkeypatch.setattr(receiver.frames,'wait_next',wait)
    monkeypatch.setattr(session_class,'stop',stop)
    controller.start_live(receiver,calibration_path)
    try:
        assert waiting.wait(2.)
        controller.stop()
        receiver.publish(2)
        assert not session_stopped.wait(.15)
        assert controller.busy and controller._frame_feeder.is_alive()
    finally:
        release.set()
    assert_joined(controller)
    assert session_stopped.is_set()
    status=controller.mapping.status()
    assert status['submitted_frames']==1 and status['last_submitted_source_frame_id']==1


def test_feeder_start_failure_stops_session_and_preserves_error(
        controller,calibration_path,monkeypatch):
    original_start=threading.Thread.start
    def start(self):
        if self.name=='room-map-frame-feeder':
            raise RuntimeError('fixture cannot start frame feeder')
        return original_start(self)
    monkeypatch.setattr(threading.Thread,'start',start)
    receiver=ExistingReceiver();receiver.publish(1)
    controller.start_live(receiver,calibration_path)
    assert_joined(controller)
    status=controller.read()[0]
    assert status['state']=='error' and 'cannot start frame feeder' in status['message']
    assert controller._frame_feeder.ident is None


def test_feeder_error_between_last_check_and_snapshot_publish_cannot_be_overwritten(
        controller,calibration_path,monkeypatch):
    original_publish=controller._publish
    before_publish,release=threading.Event(),threading.Event()
    observed=[]
    def publish(view,status):
        if status.get('state')=='running' and not before_publish.is_set():
            # _live already checked the feeder; pause at the exact next call.
            before_publish.set()
            if not release.wait(4.):
                raise TimeoutError('Test publication barrier was not released')
            original_publish(view,status)
            with controller.lock:
                observed.append(dict(controller._status))
        else:
            original_publish(view,status)
    monkeypatch.setattr(controller,'_publish',publish)
    receiver=ExistingReceiver();receiver.publish(1)
    controller.start_live(receiver,calibration_path)
    try:
        assert before_publish.wait(2.)
        receiver.error='fixture failure after feeder check';receiver.state='ERROR'
        wait_until(lambda:controller.read()[0]['state']=='error')
        assert not controller.stop_event.is_set()
    finally:
        release.set()
    assert_joined(controller)
    assert len(observed)==1
    # Inspect the state immediately after the delayed running publication,
    # before a later check could hide the race by publishing the error again.
    assert observed[0]['state']=='error'
    assert observed[0]['message']=='fixture failure after feeder check'
    assert not observed[0]['pose_valid'] and observed[0]['pose'] is None
    assert controller.read()[0]['state']=='error'
