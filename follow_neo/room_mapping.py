"""Background mapping attached to an existing receiver; no flight transport.

Live output is a sparse monocular map. Autonomous survey is available only in
the explicitly labelled known-room simulation until metric free space and the
camera-to-aircraft transform can be supplied and validated.
"""
from __future__ import annotations

from datetime import datetime
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys
import threading
import time
import traceback
import uuid

from .mapping_guidance import build_mapping_guidance
from .mapping_log import MappingLog
from .mini4_calibration import calibration_info


def mapping_import(name):
    """Use the installed package or the user's adjacent SLAM project."""
    try:
        return importlib.import_module('rbd_slam.' + name)
    except ModuleNotFoundError as exc:
        if exc.name != 'rbd_slam':
            raise
    parent = Path(__file__).resolve().parents[2]
    for folder in (parent / 'RBD-SLAM-Python', parent / 'rbd-slam-python'):
        if (folder / 'rbd_slam' / '__init__.py').is_file():
            sys.path.insert(0, str(folder))
            return importlib.import_module('rbd_slam.' + name)
    raise RuntimeError('RBD-SLAM-Python is missing. Place it beside this application or install its package.')


def empty_view(source='live'):
    return dict(source=source, units='arbitrary_monocular', points=[], keyframes=[],
                camera_position=None, target_position=None, path=[])


class RoomMappingController:
    """A worker consumes the current video receiver without starting/stopping it."""
    def __init__(self, output_root):
        self.output_root = Path(output_root)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.save_event = threading.Event()
        self.thread = None
        self._frame_feeder = None
        self.mapping = None
        self.output = None
        self.log = None
        self.calibration_metadata = None
        self._last_log_at = 0.0
        self._last_log_key = None
        self._view = empty_view()
        self._version = 0
        self._status = dict(state='idle', mode='live', message='Choose calibration and connect existing video.',
                            landmarks=0, keyframes=0, pose_valid=False, flight_commands_sent=0)

    @property
    def busy(self):
        return bool(self.thread and self.thread.is_alive())

    def set_calibration_preview(self, info):
        """Label a selected file before the first run; never relabel a saved map."""
        with self.lock:
            if self._status['state'] == 'idle' and not self.busy:
                info = info or {}
                self._status.update(calibration_kind=info.get('kind'),
                                    calibration_label=info.get('label', ''),
                                    calibration_warnings=info.get('warnings', []))

    def _begin(self, mode, metadata=None):
        if self.busy:
            raise RuntimeError('Stop the current mapping session first.')
        output = self.output_root / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
        output.mkdir(parents=True, exist_ok=False)
        log = MappingLog(output / 'mapping.jsonl', {
            'mode': mode, 'session_id': output.name,
            'output': str(output),
            'units': 'meters_simulated' if mode == 'simulation' else 'arbitrary_monocular',
            'input_kind': 'known_room_simulation' if mode == 'simulation' else 'existing_receiver_decoded_frames',
            'timestamp_kind': ('simulation_worker_monotonic_time' if mode == 'simulation'
                               else 'local_monotonic_decode_time_not_exposure'),
            'status_sample_interval_seconds': 0.5,
            'sampling': 'Latest cached status about every 0.5 seconds and on observed state/advice changes; not every processed frame.',
            'flight_commands_sent': 0,
            **(metadata or {}),
        })
        # A failed log open/start must leave the previous run and its saved
        # paths intact, rather than claiming an empty new folder is saved.
        self.stop_event.clear()
        self.save_event.clear()
        self.mapping = None
        self._frame_feeder = None
        self.output = output
        self.log = log
        self.calibration_metadata = (metadata or {}).get('calibration')
        self._last_log_at = 0.0
        self._last_log_key = None
        with self.lock:
            self._view = empty_view(mode)
            if self.calibration_metadata:
                self._view['calibration'] = self.calibration_metadata
            self._version += 1
            self._status = dict(state='starting', mode=mode, message='Starting ' + mode,
                                landmarks=0, keyframes=0, pose_valid=False, flight_commands_sent=0,
                                calibration_selected=bool((metadata or {}).get('calibration')),
                                output=str(self.output))
            if self.calibration_metadata:
                self._status.update(
                    calibration_kind=self.calibration_metadata['calibration_kind'],
                    calibration_label=self.calibration_metadata['label'],
                    calibration_warnings=self.calibration_metadata['warnings'])

    def start_live(self, receiver, calibration_path):
        if receiver is None:
            raise ValueError('Connect the existing video first.')
        frame = receiver.frames.get()
        if (frame is None or receiver.state != 'STREAMING'
                or time.monotonic() - frame.decoded_at > 1.0):
            raise ValueError('Waiting for fresh decoded video from the existing receiver.')
        calibration = mapping_import('calibration').load_calibration(calibration_path)
        info = calibration_info(calibration_path)
        if frame.image.shape != (calibration.height, calibration.width, 3):
            raise ValueError('Calibration size does not match the original video frame.')
        session_class = mapping_import('mapping_session').MappingSession
        config = mapping_import('slam').SLAMConfig(retain_map=True, max_keyframes=400,
                                                   max_landmarks=20000, max_features=1800,
                                                   adaptive_keyframes=True,
                                                   keyframe_redundancy_min_points=100)
        session = session_class(calibration, config, stale_after=1.0)
        self._begin('live', {
            'calibration': {'path': str(Path(calibration_path).resolve()),
                            'width': calibration.width, 'height': calibration.height,
                            'K': calibration.K.tolist(), 'distortion': calibration.distortion.tolist(),
                            'calibration_kind': info['kind'], 'validated': info['validated'],
                            'label': info['label'], 'provenance': info['provenance'],
                            'warnings': info['warnings'],
                            **({'success': True} if info['kind'] == 'measured' else {})},
            'slam_config': asdict(config),
        })
        self.mapping = session
        self.thread = threading.Thread(target=self._live, args=(receiver,),
                                       name='room-map-video-tap', daemon=False)
        self._start_thread()

    def start_simulation(self):
        from .room_survey_sim import RoomSurveySimulation
        simulator = RoomSurveySimulation()
        self._begin('simulation')
        self.thread = threading.Thread(target=self._simulate, args=(simulator,),
                                       name='room-survey-simulation', daemon=False)
        self._start_thread()

    def _start_thread(self):
        try:
            self.thread.start()
        except Exception as exc:
            self._fail(exc)
            self._finish_log()
            raise

    def _publish(self, view, status):
        if view.get('source') == 'live' and self.calibration_metadata:
            view = dict(view, calibration=self.calibration_metadata)
        with self.lock:
            # A feeder can fail after the display worker's last error check.
            # Serialize publication against _fail so its terminal error cannot
            # be replaced by an older starting/running snapshot. _begin owns
            # resetting error state when a new, fully joined session starts.
            if (self._status.get('state') == 'error'
                    and status.get('state') in ('starting', 'running')):
                status = dict(status, state='error', pose_valid=False, pose=None,
                              message=self._status['message'])
            # A snapshot already in progress may finish after Stop was clicked.
            # Do not let its older running state override the stop request.
            if self.stop_event.is_set() and status.get('state') in ('starting', 'running'):
                status = dict(status, state='stopping', pose_valid=False,
                              message='Stopping mapping worker...')
            self._view = view
            self._version += 1
            self._status.update(status)

    def read(self):
        with self.lock:
            status = dict(self._status)
            view = self._view
            version = self._version
        # Cheap cached status, independent of expensive map snapshots.
        if status.get('mode') == 'live' and self.mapping:
            live = self.mapping.status()
            pose_valid = (status.get('state') == 'running' and not self.stop_event.is_set()
                          and live['pose_valid'])
            status.update(pose_valid=pose_valid, tracking_state=live['tracking_state'],
                          landmarks=live['landmarks'], keyframes=live['keyframes'],
                          frame_age_seconds=live['frame_age_seconds'],
                          replaced_pending_frames=live['replaced_pending_frames'],
                          mapping_capacity_reason=live['mapping_capacity_reason'])
            for field in ('num_matches', 'num_inliers', 'last_processing_seconds',
                          'last_processed_source_frame_id', 'stale', 'stale_after_seconds',
                          'diagnostics', 'diagnostics_fresh', 'last_processed_decoded_at',
                          'tracker_frame_id', 'relocalized', 'submitted_frames', 'processed_frames',
                          'rejected_order_frames', 'rejected_state_frames', 'discarded_on_stop',
                          'discarded_on_error', 'pending_frame_id', 'processing_frame_id'):
                status[field] = live.get(field)
            status['pose'] = live.get('pose') if pose_valid else None
            status['tracking_message'] = live.get('message', '')
            view = dict(view, camera_position=live['camera_position'] if pose_valid else None)
            if live['error']:
                status.update(state='error', message=live['error']['message'], pose_valid=False)
        if self.log:
            status.update(self.log.read())
        return status, view, version

    def _log_sample(self, *, force=False):
        """Sample on the mapper worker, never the GUI or video capture thread."""
        if not self.log:
            return
        status, view, _ = self.read()
        guidance = build_mapping_guidance(status, view)
        key = (status.get('state'), status.get('tracking_state'), status.get('pose_valid'),
               status.get('stale'), status.get('mapping_capacity_reason'), guidance['code'])
        now = time.monotonic()
        if force or key != self._last_log_key or now - self._last_log_at >= 0.5:
            self.log.write('status_sample', {
                'status': status, 'guidance': guidance,
                'camera_position': view.get('camera_position'),
                'units': view.get('units', 'arbitrary_monocular'),
            })
            self._last_log_at, self._last_log_key = now, key

    def _finish_log(self):
        if not self.log:
            return
        try:
            self._log_sample(force=True)
            status, _, _ = self.read()
            self.log.close(status)
        finally:
            # Also close if obtaining the final status unexpectedly fails.
            self.log.close()

    def stop(self):
        self.stop_event.set()
        with self.lock:
            if self.busy:
                self._status.update(state='stopping', pose_valid=False, message='Stopping mapping worker...')

    def request_save(self):
        if self.busy:
            self.save_event.set()
        elif self.output:
            # Every completed run is saved automatically; no stale export claim.
            with self.lock:
                if self._status.get('saved'):
                    self._status['message'] = 'Map already saved: ' + str(self.output)
                else:
                    self._status['message'] = 'No successful map export is available.'
        else:
            raise ValueError('Start a mapping session first.')

    def _save(self, view):
        if view['source'] == 'live':
            path = self.mapping.save(self.output / 'map.json')
            archived = mapping_import('map_io').load_map(path)
            exporter = mapping_import('map_export')
            exporter.export_ply(archived, self.output / 'cloud.ply')
            # SLAM may advance between the screen snapshot and save(). Derive
            # all persisted geometry from the same frozen archive revision.
            # An archived map has keyframes, not a current live camera pose.
            view = exporter.map_snapshot(archived)
            view.update(source='live', camera_position=None, target_position=None,
                        path=[keyframe['position'] for keyframe in view['keyframes']])
        else:
            lines = ['ply', 'format ascii 1.0', 'comment synthetic room survey, meters in simulation only',
                     f"element vertex {len(view['points'])}", 'property double x', 'property double y',
                     'property double z', 'end_header']
            lines += [' '.join(format(float(c), '.17g') for c in point['position']) for point in view['points']]
            self._atomic_text(self.output / 'cloud.ply', '\n'.join(lines) + '\n')
        if view['source'] == 'live' and self.calibration_metadata:
            view['calibration'] = self.calibration_metadata
            self._atomic_text(self.output / 'calibration.json',
                              json.dumps(self.calibration_metadata, indent=2, allow_nan=False))
        self._atomic_text(self.output / 'view.json', json.dumps(view, allow_nan=False))
        with self.lock:
            self._status.update(saved=True, message='Saved: ' + str(self.output))
        if self.log:
            files = ['cloud.ply', 'view.json']
            if view['source'] == 'live':
                files.insert(0, 'map.json')
                if self.calibration_metadata:
                    files.append('calibration.json')
            self.log.write('map_saved', {'files': [str(self.output / name) for name in files],
                                        'landmarks': (len(archived.landmarks) if view['source'] == 'live'
                                                      else len(view['points'])),
                                        'exported_points': len(view['points']),
                                        'keyframes': len(view['keyframes'])})
        self.save_event.clear()

    @staticmethod
    def _atomic_text(path, text):
        # Keep the unique temporary name short on Windows; calibration.json
        # plus a second full basename can exceed MAX_PATH in nested folders.
        temporary = path.with_name('.' + uuid.uuid4().hex + '.tmp')
        try:
            temporary.write_text(text, encoding='utf-8')
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _fail(self, error, *, record=True):
        with self.lock:
            self._status.update(state='error', message=str(error), pose_valid=False)
        if self.log and record:
            self.log.write('error', {'type': type(error).__name__, 'message': str(error),
                                    'traceback': ''.join(traceback.format_exception(error))})

    def _live(self, receiver):
        snapshot_at = 0.0
        error = None
        feeder_stop = threading.Event()
        feeder_failed = threading.Event()
        feeder_error = [None]

        def feed_frames():
            """One owner of the existing latest-frame tap; never snapshot/save.

            wait_next's bounded wait and the session's independent ingress let
            this worker stop promptly even while the map lock is occupied.
            No receiver lifecycle or transport operation belongs to this tap.
            """
            version = 0
            try:
                while not feeder_stop.is_set() and not self.stop_event.is_set():
                    if receiver.state in ('ERROR', 'DISCONNECTED'):
                        raise RuntimeError(receiver.error or 'Video disconnected; map retained.')
                    frame, next_version = receiver.frames.wait_next(version, .1)
                    # Stop can arrive during wait_next: never submit its late
                    # result after ownership has been asked to end.
                    if feeder_stop.is_set() or self.stop_event.is_set():
                        break
                    if frame is not None and next_version != version:
                        version = next_version
                        self.mapping.submit_frame(frame.image, frame.frame_id, frame.decoded_at)
            except BaseException as exc:
                # A failure must be visible even while the display worker is
                # blocked on a snapshot. Defer disk logging to that worker so
                # a slow log cannot block frame ingestion or error publication.
                if not isinstance(exc, Exception):
                    wrapped = RuntimeError('Frame feeder failed: ' + type(exc).__name__ + ': ' + str(exc))
                    wrapped.__cause__ = exc
                    exc = wrapped
                feeder_error[0] = exc
                feeder_failed.set()
                self._fail(exc, record=False)

        def check_feeder():
            if feeder_failed.is_set():
                raise feeder_error[0]

        try:
            self.mapping.start()
            self._frame_feeder = threading.Thread(target=feed_frames,
                                                  name='room-map-frame-feeder', daemon=False)
            self._frame_feeder.start()
            while not self.stop_event.is_set():
                check_feeder()
                live = self.mapping.status()
                if live['error']:
                    raise RuntimeError(live['error']['message'])
                now = time.monotonic()
                if now >= snapshot_at or self.save_event.is_set():
                    view = self.mapping.snapshot()
                    check_feeder()
                    view.update(source='live', camera_position=live['camera_position'], target_position=None,
                                path=[k['position'] for k in view['keyframes']])
                    self._publish(view, dict(state='running', mode='live',
                        message='Live sparse mapping; aircraft movement is not automated.',
                        pose_valid=live['pose_valid'], landmarks=live['landmarks'], keyframes=live['keyframes']))
                    snapshot_at = time.monotonic() + 1.0
                    if self.save_event.is_set():
                        self._save(view)
                        check_feeder()
                self._log_sample()
                self.stop_event.wait(.05)
        except Exception as exc:
            error = exc
            self._fail(exc)
        finally:
            try:
                feeder_stop.set()
                feeder = self._frame_feeder
                if feeder is not None and feeder.ident is not None:
                    # Retain ownership until joined, just as for numerical
                    # work below. A broken receiver that ignores its .1s wait
                    # must not leave an orphan thread or allow a new session.
                    while feeder.is_alive():
                        feeder.join(timeout=1.0)
                        if feeder.is_alive():
                            with self.lock:
                                if not feeder_failed.is_set() and error is None:
                                    self._status.update(state='stopping', pose_valid=False,
                                                        message='Waiting for video frame tap to finish...')
                if feeder_failed.is_set() and error is None:
                    error = feeder_error[0]
                    self._fail(error)
                if self.stop_event.is_set() and self.log:
                    self.log.write('stop_requested', {'source': 'controller_stop',
                                                      'observed_on_worker': True})
                # Keep ownership of numerical work until joined. GUI calls only
                # set flags/read cached status, so disk IO never blocks them.
                while True:
                    try:
                        self.mapping.stop(timeout=1.0)
                        break
                    except TimeoutError:
                        with self.lock:
                            self._status.update(state='stopping', pose_valid=False,
                                                message='Waiting for current SLAM calculation to finish...')
                view = self.mapping.snapshot()
                view.update(source='live', camera_position=None, target_position=None,
                            path=[k['position'] for k in view['keyframes']])
                self._publish(view, dict(state='stopped', pose_valid=False))
                self._save(view)
            except Exception as exc:
                error = error or exc
                self._fail(exc)
            finally:
                if error:
                    self._fail(error, record=False)
                self._finish_log()

    def _simulate(self, simulator):
        try:
            while not self.stop_event.is_set():
                view = simulator.step()
                self._publish(view, dict(state=view['state'], mode='simulation', message=view['message'],
                    landmarks=len(view['points']), keyframes=len(view['keyframes']),
                    coverage=view['coverage'], pose_valid=False))
                self._log_sample()
                if self.save_event.is_set():
                    self._save(view)
                if view['state'] != 'running':
                    break
                self.stop_event.wait(.1)
            if self.stop_event.is_set():
                self.log.write('stop_requested', {'source': 'controller_stop',
                                                  'observed_on_worker': True})
                view = simulator.stop()
                self._publish(view, dict(state='stopped', pose_valid=False, message='Simulation stopped.'))
            self._save(view)
        except Exception as exc:
            self._fail(exc)
        finally:
            self._finish_log()
