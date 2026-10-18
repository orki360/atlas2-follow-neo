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
    class_id: int | None = None
    class_name: str | None = None
    model_class_name: str | None = None
    class_scores: tuple[float, ...] = ()


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
    botsort_match_threshold: float = .80
    botsort_low_match_threshold: float = .50
    botsort_unconfirmed_match_threshold: float = .70
    inference_fps: float = 30.0
    video_fps: float = 30.0
    edge_search_enabled: bool = True
    edge_search_seconds: float = 4.0
    search_yaw_degrees: float = 90.0
    search_grace_seconds: float = .45
    search_auto_duration: bool = True
    stale_seconds: float = .75
    reacquire_seconds: float = .75
    stop_width: float = .20
    yaw_limit: float = .70
    vertical_limit: float = .55
    forward_limit: float = .30
    search_yaw: float = .35
    search_timeout: float = 4.0
    follow_and_hold: bool = True
    nano_mode: str = 'guarded'
    forward_window_width: float = .80
    forward_window_height: float = .70
    forward_soft_zone: float = .20
    forward_yaw_demand_limit: float = .50
    forward_yaw_rate_limit: float = 65.
    forward_rise_per_second: float = 2.
    forward_require_nano: bool = False
    coordinated_turns: bool = True
    nano_visual_max_age_seconds: float = .10
    search_translation_seconds: float = 1.0
    search_forward_fraction: float = .35
    search_side_fraction: float = .20
    search_look_seconds: float = .15
    search_settle_seconds: float = .08
    scan_observation_degrees: float = 60.
    scan_yaw_limit: float = 1.
    scan_observation_mode: str = 'stop'
    scan_observation_yaw: float = .08

    def validate(self):
        if not math.isfinite(self.scan_observation_degrees) or self.scan_observation_degrees<1.:
            raise ValueError('scan_observation_degrees must be finite and at least 1 degree')
        if self.scan_observation_mode not in ('stop', 'slowdown'):
            raise ValueError('scan_observation_mode must be stop or slowdown')
        if self.nano_mode not in ('off', 'shadow', 'guarded'):
            raise ValueError('nano_mode must be off, shadow or guarded')
        if not isinstance(self.edge_search_enabled,bool):
            raise ValueError('edge_search_enabled must be a boolean')
        if not isinstance(self.search_auto_duration,bool):
            raise ValueError('search_auto_duration must be a boolean')
        if not isinstance(self.forward_require_nano,bool):
            raise ValueError('forward_require_nano must be a boolean')
        if not isinstance(self.coordinated_turns,bool):
            raise ValueError('coordinated_turns must be a boolean')
        bounds = {'confidence':(.05,.95), 'nms_iou':(.05,.95),
                  'botsort_match_threshold':(.10,.90),
                  'botsort_low_match_threshold':(.10,.70),
                  'botsort_unconfirmed_match_threshold':(.10,.85),
                  'new_track_confidence':(.05,.99), 'inference_fps':(1,60), 'video_fps':(1,60),
                  'edge_search_seconds':(2,10), 'search_yaw_degrees':(0,360),
                  'search_grace_seconds':(.30,1.),
                  'stale_seconds':(.1,2), 'reacquire_seconds':(.65,5),
                  'stop_width':(.01,.50), 'yaw_limit':(.1,1),
                  'vertical_limit':(.1,1), 'forward_limit':(0,1),
                  'search_yaw':(0,1), 'search_timeout':(1,10),
                  'forward_window_width':(.20,.85), 'forward_window_height':(.20,.80),
                  'forward_soft_zone':(.10,.50), 'forward_yaw_demand_limit':(.10,.55),
                  'forward_yaw_rate_limit':(15.,65.), 'forward_rise_per_second':(.5,4.),
                  'nano_visual_max_age_seconds':(.04,.10),
                  'search_translation_seconds':(.55,1.20), 'search_forward_fraction':(0,.35),
                  'search_side_fraction':(0,.20), 'search_look_seconds':(.10,.50),
                  'search_settle_seconds':(.05,.20),
                  'scan_yaw_limit':(.10,1.), 'scan_observation_yaw':(0.,1.)}
        for name,(lo,hi) in bounds.items():
            value = getattr(self,name)
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f'{name} must be between {lo} and {hi}')
        return self


def settings_from_config(config):
    values=asdict(Settings())
    values.update({k:v for k,v in config.get('settings',{}).items() if k in values})
    if config.get('schema_version',1)<2:values['reacquire_seconds']=.75
    if config.get('schema_version',1)<4:
        # Requested 30/30 migration, once; other tracking/axis values survive.
        values.update(video_fps=30.,inference_fps=30.)
    if config.get('schema_version',1)<5:
        values.update(edge_search_seconds=4.,search_yaw_degrees=90.)
        # Continuous dance is now the default; existing axis limits are separate.
        values['follow_and_hold']=True
    return Settings(**values).validate()
