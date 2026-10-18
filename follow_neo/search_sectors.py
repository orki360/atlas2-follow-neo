"""Finite measured sweep with observation stops, independent of target identity."""
import math
from .search_motion import NOMINAL_YAW_RATE, braking_distance

SECTOR_DEGREES = 60.
SETTLE_RATE = 12.
SETTLE_SECONDS = .08
LOOK_SECONDS = .15


class SearchObservationLog:
    """Emit phase changes, including interrupted looks, without per-tick spam."""
    def __init__(self):self.key=None;self.since=None

    def update(self,search,now,emit):
        sectors=search.get('sectors') or {}
        key=((search.get('started'),search.get('route_revision'),search.get('phase'),
              sectors.get('index'),sectors.get('phase'),sectors.get('fallback'),search.get('reason'))
             if sectors.get('enabled') else None)
        if key==self.key:return
        if key is not None or self.key is not None:
            emit('search_observation_transition',dict(previous=self.key,current=key,
                previous_phase_seconds=None if self.since is None else max(0.,now-self.since),
                sectors=sectors,measured_offset_degrees=search.get('scan_offset_degrees'),
                measured_yaw_rate_deg_s=search.get('heading_rate_deg_s'),
                requested_yaw=search.get('scan_yaw'),scan_yaw_limit=search.get('scan_yaw_limit'),
                search_deadline=search.get('until'),reason=search.get('reason')))
        self.key=key;self.since=now


class SectorSweep:
    def __init__(self,offset,low,high,direction,*,require_frames=True,
                 settle_seconds=SETTLE_SECONDS,look_seconds=LOOK_SECONDS,
                 step_degrees=SECTOR_DEGREES,yaw_limit=1.,observation_mode='stop',observation_yaw=.08):
        if not math.isfinite(step_degrees) or step_degrees<1.:
            raise ValueError('Invalid observation interval')
        if observation_mode not in ('stop','slowdown'):raise ValueError('Invalid observation mode')
        if not math.isfinite(yaw_limit) or not .10<=yaw_limit<=1.:raise ValueError('Invalid scan yaw limit')
        if not math.isfinite(observation_yaw) or not 0.<=observation_yaw<=1.:raise ValueError('Invalid observation yaw')
        self.step_degrees=step_degrees;self.yaw_limit=yaw_limit
        self.observation_mode=observation_mode;self.observation_yaw=min(yaw_limit,observation_yaw)
        self.initial_offset=offset;self.fallback=None
        self.settle_seconds=settle_seconds;self.look_seconds=look_seconds
        self.targets=[];point=offset
        # A directional 360-degree sweep covers the circle once. A centered
        # arc visits one end and then the other, never repeats either leg.
        ends=[high if direction>0 else low]
        if low<0<high:ends.append(low if direction>0 else high)
        for end in ends:
            sign=1 if end>point else -1
            while abs(end-point)>step_degrees:
                point+=sign*step_degrees;self.targets.append(point)
            if not self.targets or abs(self.targets[-1]-end)>.01:self.targets.append(end)
            point=end
        self.low=low;self.high=high;self.require_frames=require_frames
        self.index=0;self.phase='move';self.settled_since=None;self.look_since=None
        self.sources=[];self.last_frame=None;self.completed=[];self.done=False

    @property
    def observation_target(self):return self.targets[min(self.index,len(self.targets)-1)]

    @property
    def pass_direction(self):
        before=self.targets[self.index-1] if self.index else self.initial_offset
        return 1 if self.observation_target>before else -1

    @property
    def slow_pass(self):
        # Arc boundaries and reversals always use a stationary observation.
        return (self.observation_mode=='slowdown' and not self.done
                and self.observation_yaw>0
                and self.low<self.observation_target<self.high)

    @property
    def observation_rate_limit(self):
        # This is a scan sampling threshold, never target/motion authorization.
        return max(SETTLE_RATE,NOMINAL_YAW_RATE*self.observation_yaw) if self.slow_pass else SETTLE_RATE

    @property
    def window_degrees(self):
        speed=NOMINAL_YAW_RATE*self.observation_yaw
        # Include braking distance and time to receive two post-look YOLO
        # sources. Stay inside the current sector and original outer arc.
        return min(self.step_degrees*.45,max(6.,braking_distance(speed)+
            speed*(self.settle_seconds+self.look_seconds+.25)))

    @property
    def target(self):
        return (max(self.low,min(self.high,self.observation_target+self.pass_direction*self.window_degrees))
                if self.slow_pass else self.observation_target)

    def observe(self,source,frame_id):
        if (self.phase!='look' or self.look_since is None or source<=self.look_since
                or self.last_frame is not None and frame_id<=self.last_frame):return
        self.last_frame=frame_id;self.sources.append(source);self.sources=self.sources[-3:]

    def suspend(self):
        self.settled_since=self.look_since=None;self.sources=[];self.last_frame=None
        if self.phase!='move':self.phase='settle'

    def update(self,offset,rate,now,profile):
        if self.done:return 0.
        if self.slow_pass:return self._slow_update(offset,rate,now,profile)
        self.fallback=(('zero_observation_yaw' if self.observation_yaw==0 else 'arc_boundary_stop')
                       if self.observation_mode=='slowdown' else None)
        distance=self.target-offset
        if self.phase=='move' and abs(distance)<=2.:
            self.phase='settle';profile.reset(now)
        if self.phase=='move':return profile.update(distance,rate,now,max_yaw=self.yaw_limit)
        profile.reset(now)
        # Excess overshoot needs another bounded approach, not a false claim
        # that the requested direction has been observed.
        if abs(distance)>12.:
            self.phase='move';self.suspend();return 0.
        if rate is None or not math.isfinite(rate) or abs(rate)>SETTLE_RATE:
            self.suspend();return 0.
        if self.settled_since is None:self.settled_since=now
        if now-self.settled_since<self.settle_seconds:return 0.
        if self.look_since is None:
            self.look_since=now;self.phase='look';self.sources=[];self.last_frame=None
        fresh=(not self.require_frames or len(self.sources)>=2
               and self.sources[-1]-self.sources[0]>=.06 and 0<=now-self.sources[-1]<=.25)
        if now-self.look_since>=self.look_seconds and fresh:
            self._complete(offset,now,profile)
        return 0.

    def _complete(self,offset,now,profile,*,keep_speed=False):
        self.completed.append(dict(target=self.observation_target,offset=offset,time=now,
            source_frames=len(self.sources),mode=self.observation_mode,fallback=self.fallback,
            look_elapsed_seconds=now-self.look_since))
        self.index+=1
        if self.index==len(self.targets):self.done=True;self.phase='done'
        else:
            self.phase='move';self.suspend()
            if not keep_speed:profile.reset(now)

    def _slow_update(self,offset,rate,now,profile):
        side=self.pass_direction;distance=self.observation_target-offset
        if self.phase=='move' and side*distance>2.:
            self.fallback=None
            return profile.update(distance,rate,now,max_yaw=self.yaw_limit,arrival_yaw=self.observation_yaw)
        if self.phase=='move':self.phase='settle'
        remaining=side*(self.target-offset)
        if remaining < -2.:
            self.suspend();profile.reset(now);self.fallback='observation_window_overshoot';return 0.
        # A bounded fly-through: absent results cannot move the observation
        # window or the outer arc. Hold here until fresh samples or search timeout.
        if remaining<=.5:
            self.fallback='observation_window_end';command=0.;profile.reset(now)
        else:
            self.fallback=None
            command=profile.update(self.target-offset,rate,now,max_yaw=self.observation_yaw)
        if rate is None or not math.isfinite(rate):
            self.suspend();profile.reset(now);self.fallback='waiting_rate';return 0.
        if abs(rate)>self.observation_rate_limit or side*rate < -2.:
            self.suspend();return command if side*rate>=-2. else 0.
        if self.settled_since is None:self.settled_since=now
        if now-self.settled_since<self.settle_seconds:return command
        if self.look_since is None:
            self.look_since=now;self.phase='look';self.sources=[];self.last_frame=None
        fresh=(not self.require_frames or len(self.sources)>=2
               and self.sources[-1]-self.sources[0]>=.06 and 0<=now-self.sources[-1]<=.25)
        if side*distance<=0 and now-self.look_since>=self.look_seconds and fresh:
            self._complete(offset,now,profile,keep_speed=True)
        return command

    def snapshot(self):
        return dict(enabled=True,phase=self.phase,index=self.index,count=len(self.targets),
                    target_degrees=self.target,observation_target_degrees=self.observation_target,
                    step_degrees=self.step_degrees,observation_mode=self.observation_mode,
                    yaw_limit=self.yaw_limit,observation_yaw=self.observation_yaw,
                    slow_pass=self.slow_pass,fallback=self.fallback,
                    observation_window_degrees=self.window_degrees if self.slow_pass else 0.,
                    settle_rate_deg_s=self.observation_rate_limit,settled_since=self.settled_since,
                    look_since=self.look_since,look_seconds=self.look_seconds,settle_seconds=self.settle_seconds,
                    fresh_source_frames=len(self.sources),require_frames=self.require_frames,
                    completed=list(self.completed),complete=self.done)
