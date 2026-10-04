"""Flushed, append-only JSONL records for one local room-mapping session.

The mapping worker performs writes. UI readers obtain a small cached status
without waiting for disk I/O. This module never connects to the drone.
"""
from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading
import time


def _json_value(value):
    """Keep strict JSON values and replace unavailable numeric results with null."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError('Mapping log object keys must be strings')
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    # NumPy is optional here. Arrays and scalar values expose tolist()/item().
    if type(value).__module__.split('.')[0] == 'numpy':
        convert = getattr(value, 'tolist', None) or getattr(value, 'item', None)
        if convert is not None:
            return _json_value(convert())
    raise TypeError(f'Unsupported mapping log value: {type(value).__name__}')


class MappingLog:
    """Create a new UTF-8 log and flush each complete record before reporting it.

    Opening or writing the initial record raises on failure so the controller
    can make session-start failure visible. Later errors are sticky, stop new
    writes, and are exposed through read(); they never propagate to the mapper.
    close() is idempotent and always attempts to release the file handle.
    """

    def __init__(self, path, metadata):
        self.path = Path(path)
        self._write_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._started = time.monotonic()
        self._records = 0
        self._error = None
        self._closed = False
        self._status = {
            'log_path': str(self.path),
            'log_error': None,
            'log_records': 0,
            'log_closed': False,
        }
        # Exclusive creation protects previous sessions, even on name collision.
        self._stream = self.path.open('x', encoding='utf-8', newline='\n')
        if not self._write_record('session_start', metadata):
            try:
                self._stream.close()
            except Exception:
                pass
            self._closed = True
            self._publish_status()
            raise OSError(f'Could not start mapping log: {self._error}')

    def _publish_status(self):
        # No serialization, filesystem call, or write-lock acquisition while
        # holding this lock: the GUI's read() must not wait for the disk.
        with self._status_lock:
            self._status = {
                'log_path': str(self.path),
                'log_error': self._error,
                'log_records': self._records,
                'log_closed': self._closed,
            }

    def _fail(self, error):
        if self._error is None:
            self._error = f'{type(error).__name__}: {error}'
        self._publish_status()

    def _write_record(self, event, data):
        if self._closed or self._error is not None:
            return False
        try:
            if not isinstance(event, str) or not event:
                raise ValueError('Mapping log event must be a nonempty string')
            wall_time = time.time()
            monotonic_time = time.monotonic()
            record = {
                'schema_version': 1,
                'record_id': self._records + 1,
                'event': event,
                'timestamp_utc': datetime.fromtimestamp(
                    wall_time, timezone.utc
                ).isoformat(timespec='milliseconds'),
                'timestamp_epoch': wall_time,
                'timestamp_monotonic': monotonic_time,
                'elapsed_seconds': monotonic_time - self._started,
                'data': _json_value(data),
            }
            line = json.dumps(record, ensure_ascii=False, allow_nan=False)
            self._stream.write(line + '\n')
            self._stream.flush()
        except Exception as error:
            self._fail(error)
            return False
        self._records += 1
        self._publish_status()
        return True

    def write(self, event, data):
        """Write one event, returning False after closure or any logging error."""
        with self._write_lock:
            return self._write_record(event, data)

    def close(self, summary=None):
        """Write one final record and close; repeated calls have no effect."""
        with self._write_lock:
            if self._closed:
                return
            try:
                self._write_record('session_end', {} if summary is None else summary)
            finally:
                self._closed = True
                try:
                    self._stream.close()
                except Exception as error:
                    self._fail(error)
                self._publish_status()

    def read(self):
        """Return cached health without waiting for serialization or disk I/O."""
        with self._status_lock:
            return self._status.copy()
