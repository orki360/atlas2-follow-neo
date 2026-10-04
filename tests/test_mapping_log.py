"""Mapping diagnostics persist as strict JSON without blocking UI status reads."""
from datetime import datetime
import io
import json
from pathlib import Path
import threading

import pytest

from follow_neo.mapping_log import MappingLog


def records(path):
    def invalid_constant(value):
        raise AssertionError(f'Nonstandard JSON number: {value}')
    return [json.loads(line, parse_constant=invalid_constant)
            for line in path.read_text(encoding='utf-8').splitlines()]


def test_records_are_flushed_parseable_ordered_and_metadata_cannot_override(tmp_path):
    path = tmp_path / 'mapping.jsonl'
    log = MappingLog(path, {'camera': 'Mini 4 Pro', 'description': 'מיפוי חדר'})
    assert records(path)[0]['data']['description'] == 'מיפוי חדר'
    assert log.write('mapping_status', {'event': 'fake', 'record_id': -1, 'pose': [1, 2, 3]})
    # The file is readable before close(), including its latest record.
    assert len(records(path)) == 2
    log.close({'reason': 'operator_stop'})
    rows = records(path)
    assert [row['event'] for row in rows] == ['session_start', 'mapping_status', 'session_end']
    assert [row['record_id'] for row in rows] == [1, 2, 3]
    assert rows[1]['data']['record_id'] == -1
    assert rows[-1]['data'] == {'reason': 'operator_stop'}
    assert all(row['schema_version'] == 1 for row in rows)
    for row in rows:
        instant = datetime.fromisoformat(row['timestamp_utc'])
        assert instant.utcoffset().total_seconds() == 0
        assert abs(instant.timestamp() - row['timestamp_epoch']) < 0.0011
        assert row['elapsed_seconds'] >= 0
    assert rows[-1]['timestamp_monotonic'] >= rows[0]['timestamp_monotonic']
    assert rows[-1]['elapsed_seconds'] >= rows[0]['elapsed_seconds']
    assert log.read() == {
        'log_path': str(path), 'log_error': None, 'log_records': 3, 'log_closed': True,
    }


def test_close_is_idempotent_and_later_writes_do_not_change_log(tmp_path):
    path = tmp_path / 'mapping.jsonl'
    log = MappingLog(path, {})
    log.close()
    original = path.read_bytes()
    log.close({'duplicate': True})
    assert not log.write('after_close', {})
    assert path.read_bytes() == original
    assert [row['event'] for row in records(path)].count('session_end') == 1
    assert log.read()['log_error'] is None


def test_nonfinite_nested_values_and_paths_have_portable_json_encoding(tmp_path):
    path = tmp_path / 'mapping.jsonl'
    log = MappingLog(path, {})
    assert log.write('diagnostics', {
        'nested': [float('nan'), {'ratio': float('inf'), 'negative': -float('inf')}],
        'pose': (1.25, 2), 'saved': Path('maps') / 'cloud.ply', 'valid': True,
    })
    log.close()
    data = records(path)[1]['data']
    assert data['nested'] == [None, {'ratio': None, 'negative': None}]
    assert data['pose'] == [1.25, 2]
    assert data['saved'] == str(Path('maps') / 'cloud.ply')
    assert data['valid'] is True


def test_numpy_values_are_supported_without_mandatory_numpy_import(tmp_path):
    np = pytest.importorskip('numpy')
    path = tmp_path / 'mapping.jsonl'
    log = MappingLog(path, {})
    assert log.write('pose', {'points': np.array([[1, np.nan]]), 'count': np.int64(3),
                              'flag': np.bool_(True), 'ratio': np.float32(np.inf)})
    log.close()
    assert records(path)[1]['data'] == {
        'points': [[1.0, None]], 'count': 3, 'flag': True, 'ratio': None,
    }


def test_exclusive_creation_does_not_overwrite_existing_log(tmp_path):
    path = tmp_path / 'mapping.jsonl'
    path.write_text('previous session\n', encoding='utf-8')
    with pytest.raises(FileExistsError):
        MappingLog(path, {})
    assert path.read_text(encoding='utf-8') == 'previous session\n'


class FailedStream(io.StringIO):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.write_calls = 0
        self.close_attempted = False

    def write(self, text):
        self.write_calls += 1
        if self.operation == 'write':
            raise OSError('disk unavailable')
        return super().write(text)

    def flush(self):
        if self.operation == 'flush':
            raise OSError('flush unavailable')
        return super().flush()

    def close(self):
        self.close_attempted = True
        super().close()
        if self.operation == 'close':
            raise OSError('close unavailable')


@pytest.mark.parametrize('operation', ['write', 'flush'])
def test_io_failures_remain_visible_and_prevent_further_writes(tmp_path, operation):
    log = MappingLog(tmp_path / 'mapping.jsonl', {})
    log._stream.close()
    replacement = FailedStream(operation)
    log._stream = replacement
    assert not log.write('status', {})
    failed_status = log.read()
    assert 'unavailable' in failed_status['log_error']
    assert failed_status['log_records'] == 1  # Only the constructor's flushed row.
    assert not log.write('status', {})
    log.close()
    assert replacement.write_calls == 1
    assert replacement.close_attempted
    assert log.read()['log_closed']
    assert log.read()['log_error'] == failed_status['log_error']


def test_close_failure_is_visible_without_raising(tmp_path):
    log = MappingLog(tmp_path / 'mapping.jsonl', {})
    log._stream.close()
    replacement = FailedStream('close')
    log._stream = replacement
    log.close()
    assert log.read()['log_error'] == 'OSError: close unavailable'
    assert log.read()['log_closed']
    assert replacement.close_attempted


def test_initial_record_failure_raises_and_closes_handle(tmp_path, monkeypatch):
    replacement = FailedStream('write')
    monkeypatch.setattr(Path, 'open', lambda *args, **kwargs: replacement)
    with pytest.raises(OSError, match='Could not start mapping log'):
        MappingLog(tmp_path / 'mapping.jsonl', {})
    assert replacement.close_attempted


def test_serialization_failure_is_visible_and_does_not_append_partial_row(tmp_path):
    path = tmp_path / 'mapping.jsonl'
    log = MappingLog(path, {})
    assert not log.write('bad_data', {'unsupported': object()})
    assert 'Unsupported mapping log value' in log.read()['log_error']
    assert not log.write('later', {})
    log.close()
    assert [row['event'] for row in records(path)] == ['session_start']
    assert log.read()['log_closed']


def test_status_read_does_not_wait_for_blocked_disk_flush(tmp_path):
    log = MappingLog(tmp_path / 'mapping.jsonl', {})
    log._stream.close()
    entered = threading.Event()
    release = threading.Event()
    completed_read = threading.Event()

    class BlockedStream(io.StringIO):
        def flush(self):
            entered.set()
            if not release.wait(5):
                raise OSError('test timed out waiting for flush release')

    log._stream = BlockedStream()
    writer = threading.Thread(target=lambda: log.write('status', {}))
    reader = threading.Thread(target=lambda: (log.read(), completed_read.set()))
    writer.start()
    try:
        assert entered.wait(2)
        reader.start()
        assert completed_read.wait(1), 'GUI status read was blocked by filesystem I/O'
    finally:
        release.set()
        writer.join(5)
        if reader.ident is not None:
            reader.join(5)
        log.close()
    assert not writer.is_alive()


def test_concurrent_writers_produce_complete_sequential_records(tmp_path):
    path = tmp_path / 'mapping.jsonl'
    log = MappingLog(path, {})
    barrier = threading.Barrier(5)
    outcomes = []

    def write_batch(worker):
        barrier.wait(timeout=5)
        for index in range(10):
            outcomes.append(log.write('sample', {'worker': worker, 'index': index}))

    threads = [threading.Thread(target=write_batch, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    log.close()
    rows = records(path)
    assert outcomes == [True] * 40
    assert [row['record_id'] for row in rows] == list(range(1, 43))
    assert {(row['data']['worker'], row['data']['index']) for row in rows[1:-1]} == {
        (worker, index) for worker in range(4) for index in range(10)
    }
