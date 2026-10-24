"""Process-local switch for optional diagnostics, never for control clocks."""
import time

enabled = True
_epoch = 0


def set_enabled(value):
    """Invalidate samples spanning a mode change; never touch control clocks."""
    global enabled, _epoch
    if enabled != bool(value):
        _epoch += 1
        enabled = bool(value)


def token():
    return _epoch if enabled else None


def valid(sample):
    return enabled and sample is not None and sample == _epoch


def duration_ms(sample, start, end=None):
    end = clock() if end is None else end
    return max(0., (end-start)*1000) if valid(sample) else None


def clock():
    return time.perf_counter() if enabled else 0.
