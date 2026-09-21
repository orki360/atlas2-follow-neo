from __future__ import annotations
from dataclasses import dataclass, asdict
import math


def clamp(x, lo, hi):
    return max(lo, min(hi, float(x))) if math.isfinite(x) else 0.0


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self): return max(0.0, self.x2 - self.x1)
    @property
    def height(self): return max(0.0, self.y2 - self.y1)
    @property
    def cx(self): return (self.x1 + self.x2) * .5
    @property
    def cy(self): return (self.y1 + self.y2) * .5

    def clipped(self, width, height):
        return Box(clamp(self.x1, 0, width-1), clamp(self.y1, 0, height-1),
                   clamp(self.x2, 0, width-1), clamp(self.y2, 0, height-1))


@dataclass(frozen=True)
class Detection:
    box: Box
    confidence: float


@dataclass
class Kinematics:
    initialized: bool = False
    cx: float = 0.0
    cy: float = 0.0
    width: float = 0.0
    height: float = 0.0
    vx: float = 0.0
    vy: float = 0.0

    @property
    def box(self):
        return Box(self.cx-self.width/2, self.cy-self.height/2,
                   self.cx+self.width/2, self.cy+self.height/2)


@dataclass
class Intent:
    """A dimensionless simulation output, NEVER an authorized aircraft command."""
    yaw: float = 0.0
    vertical: float = 0.0
    roll: float = 0.0
    forward: float = 0.0

    def bounded(self): return Intent(*(clamp(x, -1, 1) for x in asdict(self).values()))


@dataclass(frozen=True)
class Settings:
    confidence: float = .25
    nms_iou: float = .45
    new_track_confidence: float = .45
    inference_fps: float = 60.0
    stale_seconds: float = .75
    reacquire_seconds: float = .75
    stop_width: float = .20
    yaw_limit: float = .70
    vertical_limit: float = .55
    forward_limit: float = .30
    search_yaw: float = .35
    search_timeout: float = 4.0
    follow_and_hold: bool = True

    def validate(self):
        bounds = {'confidence':(.05,.95), 'nms_iou':(.05,.95),
                  'new_track_confidence':(.05,.99), 'inference_fps':(1,60),
                  'stale_seconds':(.1,2), 'reacquire_seconds':(.65,5),
                  'stop_width':(.01,.50), 'yaw_limit':(.1,1),
                  'vertical_limit':(.1,1), 'forward_limit':(0,1),
                  'search_yaw':(0,1), 'search_timeout':(1,10)}
        for name,(lo,hi) in bounds.items():
            value = getattr(self,name)
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f'{name} must be between {lo} and {hi}')
        return self
