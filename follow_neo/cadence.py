"""Timestamp-based sampling. No frame queue and no invented camera FPS."""
import math


class FrameRateGate:
    def __init__(self):self.next_due=None;self.fps=None

    def admit(self, stamp, fps):
        if not math.isfinite(stamp) or not math.isfinite(fps) or fps<=0:
            raise ValueError('Invalid frame time/rate')
        period=1./fps
        if fps!=self.fps:
            self.fps=fps;self.next_due=stamp
        if stamp+1e-7<self.next_due:return False
        # Keep the phase instead of adding a full wait after work. If the
        # source stalls, resume from now rather than bursting to catch up.
        self.next_due+=period
        if self.next_due<stamp-period:self.next_due=stamp+period
        return True


class LatestFrameSampler:
    """Capacity-one mailbox. Caller serializes offer/take with a condition.

    Keep an early frame until the publication deadline, replacing it whenever
    a newer frame arrives. Never duplicate a frame or change its source time.
    """
    MAX_AGE_SECONDS = .100

    def __init__(self):
        self.pending=None;self.next_due=None;self.fps=None
        self.offered=self.selected=self.replaced=self.expired=self.invalidated=0
        self.last_selected_age_ms=None

    def offer(self,value,stamp):
        if not math.isfinite(stamp):raise ValueError('Invalid source time')
        if self.pending is not None:self.replaced+=1
        self.pending=(value,stamp);self.offered+=1

    def clear(self):
        if self.pending is not None:self.invalidated+=1
        self.pending=None;self.next_due=None;self.fps=None

    def take(self,now,fps):
        if not math.isfinite(now) or not math.isfinite(fps) or not 1<=fps<=60:
            raise ValueError('Invalid publication time/rate')
        if fps!=self.fps:self.fps=fps;self.next_due=None
        if self.pending is None:return None
        value,source=self.pending
        if not 0<=now-source<=self.MAX_AGE_SECONDS:
            self.pending=None;self.expired+=1;return None
        if self.next_due is not None and now+1e-7<self.next_due:return None
        self.pending=None;self.selected+=1;self.last_selected_age_ms=(now-source)*1000
        period=1./fps
        # Preserve cadence through normal jitter, but never catch up a stall.
        due=now if self.next_due is None else self.next_due
        self.next_due=due+period if now-due<period else now+period
        return value

    def wait_seconds(self,now):
        if self.pending is None:return .1
        if self.next_due is None:return .001
        return max(.001,min(.1,self.next_due-now))

    def snapshot(self):
        return dict(policy='latest_frame_at_deadline',capacity=1,pending=int(self.pending is not None),
                    offered=self.offered,selected=self.selected,replaced=self.replaced,
                    expired=self.expired,invalidated=self.invalidated,
                    last_selected_age_ms=self.last_selected_age_ms,max_age_ms=self.MAX_AGE_SECONDS*1000)
