"""Latest-command pacing only; evidence and TCP ACK clocks stay unchanged."""
import math
import time


SEND_HZ_OPTIONS = (20, 25, 30)
DEFAULT_SEND_HZ = 20


def checked_send_hz(value):
    try:
        number = float(value)
    except (ValueError, TypeError):
        raise ValueError('TCP send rate must be 20, 25 or 30 Hz') from None
    if isinstance(value, bool) or number not in SEND_HZ_OPTIONS:
        raise ValueError('TCP send rate must be 20, 25 or 30 Hz')
    return int(number)


def send_hz_from_config(config):
    try:
        return checked_send_hz(config.get('tcp_send_hz', DEFAULT_SEND_HZ))
    except ValueError:
        return DEFAULT_SEND_HZ


class SendCadence:
    """Absolute high-resolution deadlines, without a queue or catch-up sends.

    Wait BEFORE reading the current input/decision. Work and the serialized ACK
    consume the current slot's budget. A missed slot is skipped, not replayed.
    This uses QPC on Windows, but never converts evidence timestamps to QPC.
    No sampling, diagnostic counters, spin waits or OS timer changes.
    """
    def __init__(self, clock=None):
        self.clock = clock or time.perf_counter
        self.reset()

    def reset(self):
        self.hz = None
        self.deadline = None
        self.last_started = None

    def wait(self, stop_event, hz):
        """Return False if stopped; otherwise one fresh command may be computed."""
        now = self.clock()
        period = 1. / hz  # validated at the configuration boundary
        if self.deadline is None:
            due = now
        elif hz != self.hz:
            due = self.last_started + period
        else:
            due = self.deadline
        if due < now:
            due += math.ceil((now - due) / period) * period
        while now < due:
            if stop_event.wait(due - now):
                return False
            now = self.clock()
        if stop_event.is_set():
            return False
        self.hz = hz
        self.last_started = now
        self.deadline = due + period
        return True
