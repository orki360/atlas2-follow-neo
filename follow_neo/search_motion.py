"""Search-only yaw estimation and braking; no changes to normal tracking control.

Heading times are local query times, not aircraft sensor timestamps. Fit a short
window so repeated cached angle values do not each become a zero-speed estimate.
The braking parameters below are conservative starting values for flight tests,
not a calibrated aircraft model or a guarantee of a physical stopping distance.
"""
from collections import deque
import math
from .telemetry import fresh_heading

BOOST_MAX_SECONDS = 2.0
MAX_BOOST_ANGLE = 180.0


def search_duration_limit(angle):
    """Large arcs need travel/return time; preserve the existing small-arc cap."""
    return 10. if angle <= 180. else 35.

RATE_WINDOW_SECONDS = .30
NOMINAL_YAW_RATE = 90.0
BRAKE_RESPONSE_SECONDS = .45
BRAKE_DECELERATION = 60.0
BRAKE_RESERVE_DEGREES = 2.0
YAW_RISE_PER_SECOND = 1.0

def motion_parameters():
    return dict(boost_max_seconds=BOOST_MAX_SECONDS,rate_window_seconds=RATE_WINDOW_SECONDS,
                nominal_yaw_rate_deg_s=NOMINAL_YAW_RATE,brake_response_seconds=BRAKE_RESPONSE_SECONDS,
                brake_deceleration_deg_s2=BRAKE_DECELERATION,brake_reserve_degrees=BRAKE_RESERVE_DEGREES,
                yaw_rise_per_second=YAW_RISE_PER_SECOND)


class SearchHeadingRate:
    def __init__(self):
        self.samples=deque();self.last=None;self.rate=None;self.position=0.

    def update(self,heading,now):
        if not fresh_heading(heading,now):
            self.__init__();return None
        if self.last and heading['source']==self.last['source'] and heading['time']==self.last['time']:
            return self.rate
        if self.last:
            dt=heading['time']-self.last['time']
            delta=(heading['yaw_deg']-self.last['yaw_deg']+180)%360-180
            if heading['source']!=self.last['source'] or not 0<dt<=.30 or abs(delta)>90:
                self.__init__()
            else:self.position+=delta
        self.last=dict(heading)
        self.samples.append((heading['time'],self.position))
        while len(self.samples)>2 and self.samples[1][0]<heading['time']-RATE_WINDOW_SECONDS:
            self.samples.popleft()
        if len(self.samples)<3 or self.samples[-1][0]-self.samples[0][0]<.10:
            self.rate=None;return None
        origin=self.samples[0][0]
        xs=[t-origin for t,_ in self.samples];ys=[p for _,p in self.samples]
        mx=sum(xs)/len(xs);my=sum(ys)/len(ys)
        denom=sum((x-mx)**2 for x in xs)
        rate=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/denom if denom>0 else 0.
        self.rate=rate if abs(rate)<=240 else None
        return self.rate


def braking_distance(speed):
    speed=max(0.,speed)
    return BRAKE_RESERVE_DEGREES+speed*BRAKE_RESPONSE_SECONDS+speed*speed/(2*BRAKE_DECELERATION)


class SearchYawProfile:
    def __init__(self):
        self.command=0.;self.time=None;self.holding=False;self.diagnostics={}

    def reset(self,now,command=0.):
        self.command=command;self.time=now;self.holding=False;self.diagnostics={}

    def update(self,distance,rate,now):
        side=1 if distance>0 else -1
        closing=max(0.,side*(rate or 0.))
        # Invert distance = response_delay * speed + speed^2 / (2 * decel).
        # Aim at the actual boundary; the planner stops within its tolerance.
        # Subtracting that tolerance here would asymptotically stop short of it.
        usable=abs(distance)
        ad=BRAKE_DECELERATION*BRAKE_RESPONSE_SECONDS
        desired=min(NOMINAL_YAW_RATE,max(0.,math.sqrt(ad*ad+2*BRAKE_DECELERATION*usable)-ad))
        raw=max(0.,min(desired/NOMINAL_YAW_RATE,(1.5*desired-.5*closing)/NOMINAL_YAW_RATE))
        if raw<=0:self.holding=True
        elif self.holding and closing<=max(5.,desired-5.):self.holding=False
        if self.holding:raw=0.
        previous=max(0.,side*self.command)
        dt=0. if self.time is None else max(0.,min(.20,now-self.time))
        magnitude=min(raw,previous+YAW_RISE_PER_SECOND*dt) if raw>previous else raw
        self.command=side*magnitude;self.time=now
        self.diagnostics=dict(desired_rate_deg_s=side*desired,closing_rate_deg_s=closing,
            braking_distance_degrees=braking_distance(closing),raw_yaw=side*raw,
            control_reason='brake_hold' if self.holding else 'accelerating' if magnitude<raw else 'braking' if raw<1 else 'turning')
        return self.command
