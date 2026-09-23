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
