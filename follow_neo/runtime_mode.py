"""Presentation/logging policy only. Never grants or renews flight authority."""
import math
import threading


VERBOSE_EVENTS = frozenset({
    'observation', 'botsort_association', 'heading', 'nano_anchor',
    'nano_anchor_applied', 'nano_observation', 'nano_visual_rejected', 'nano_veto',
    'yolo_visual_continuity', 'control_indicator', 'control_pending_motion',
    'search_test_sample', 'video_health',
})


def keep_realtime_event(event, data):
    if event in VERBOSE_EVENTS:
        return False
    if event in ('flight_command', 'control_command_attempt', 'control_command_ack'):
        return str(data.get('command', '')).split(' ', 1)[0] != 'rc'
    # Unknown events, failures, authority changes and search transitions stay.
    return True


class RuntimeStats:
    """Bounded command aggregates and immediate transitions, independent of UI."""
    def __init__(self):
        self.lock = threading.Lock()
        self.last_command = None
        self.motion_key = self.tracking_key = None
        self.motion_since = self.tracking_since = None
        self._clear_window()

    def _clear_window(self):
        self.commands = self.moving = self.acks = self.gaps = 0
        self.ack_sum = self.ack_max = self.gap_sum = self.gap_max = 0.

    def command(self, data, now):
        moving = any(data.get('values') or ())
        reason = str(data.get('gate_reason', ''))[:200]
        key = (data.get('mode'), moving, reason if not moving else '')
        ack = data.get('ack_ms')
        if ack is None:
            ack = ((data.get('transport') or {}).get('last_ack') or {}).get('ack_ms')
        with self.lock:
            self.commands += 1
            self.moving += int(moving)
            if isinstance(ack, (int, float)) and math.isfinite(ack) and ack >= 0:
                self.acks += 1; self.ack_sum += ack; self.ack_max = max(self.ack_max, ack)
            if self.last_command is not None:
                gap = max(0., now-self.last_command)*1000
                self.gaps += 1; self.gap_sum += gap; self.gap_max = max(self.gap_max, gap)
            self.last_command = now
            if key == self.motion_key:
                return None
            out = dict(mode=key[0], moving=moving, reason=reason,
                       previous_duration_seconds=None if self.motion_since is None else now-self.motion_since,
                       values=list(data.get('values') or ()), server_acknowledged=True)
            self.motion_key = key; self.motion_since = now
            return out

    def tracking(self, decision, now):
        identity = (decision.get('target_identity') or {}).get('logical_target_id')
        key = (decision.get('state'), decision.get('spacing_phase'),
               bool(decision.get('confirmed')), bool(decision.get('stale')), identity,
               (decision.get('edge_search') or {}).get('phase'))
        with self.lock:
            if key == self.tracking_key:
                return None
            out = dict(state=key[0], spacing_phase=key[1], confirmed=key[2], stale=key[3],
                       target_id=identity, search_phase=key[5], reason=decision.get('reason'),
                       previous_duration_seconds=None if self.tracking_since is None else now-self.tracking_since)
            self.tracking_key = key; self.tracking_since = now
            return out

    def snapshot(self):
        with self.lock:
            out = dict(commands_acknowledged=self.commands, nonzero_commands=self.moving,
                       neutral_commands=self.commands-self.moving,
                       ack_ms_mean=self.ack_sum/self.acks if self.acks else None,
                       ack_ms_max=self.ack_max if self.acks else None,
                       command_gap_ms_mean=self.gap_sum/self.gaps if self.gaps else None,
                       command_gap_ms_max=self.gap_max if self.gaps else None)
            self._clear_window()
            return out
