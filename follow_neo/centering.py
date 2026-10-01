"""Damped live centering; no aircraft authority and no additional tracking filter."""
import math
from .policy import SmartTrackingController
from .telemetry import fresh_heading
from .types import clamp
from . import __revision__
from .approach import LateralEvidence


class HeadingRate:
    def __init__(self):
        self.sample=None
        self.rate=None

    def update(self, heading, now):
        if not fresh_heading(heading,now):
            self.sample=None;self.rate=None
            return None
        old=self.sample
        if old and heading['source']==old['source'] and heading['time']==old['time']:
            return self.rate
        self.sample=dict(heading)
        if not old or heading['source']!=old['source']:
            self.rate=None;return None
        dt=heading['time']-old['time']
        delta=(heading['yaw_deg']-old['yaw_deg']+180)%360-180
        if not .02<=dt<=.30 or abs(delta/dt)>240:
            self.rate=None;return None
        value=delta/dt
        alpha=dt/(.10+dt)
        self.rate=value if self.rate is None else self.rate+alpha*(value-self.rate)
        return self.rate


class DampedTrackingController(SmartTrackingController):
    """Keep vertical/spacing contracts; replace ordinary live yaw centering only."""
    def __init__(self):
        super().__init__()
        self.heading_rate=HeadingRate()
        self.last_measurement=None;self.error_rate=0.;self.last_tick=None
        self.centered=True;self.forward_factor=1.;self.diagnostics={}
        self.lateral=LateralEvidence()

    def compute(self,k,w,h,age,occluded,now,settings,heading=None,measurement_time=None,measured_error=None,lateral_observation=None):
        out,ex,ey=super().compute(k,w,h,age,occluded,now,settings)
        rate=self.heading_rate.update(heading,now)
        dt=0. if self.last_tick is None else max(0.,min(.1,now-self.last_tick))
        self.last_tick=now
        valid=(not occluded and k.initialized and measurement_time is not None
               and measured_error is not None and math.isfinite(measured_error))
        if valid and (self.last_measurement is None or measurement_time>self.last_measurement[0]):
            old=self.last_measurement
            self.last_measurement=(measurement_time,measured_error)
            if old and .005<=measurement_time-old[0]<=.20 and abs(measured_error-old[1])<.35:
                span=measurement_time-old[0]
                derivative=clamp((measured_error-old[1])/span,-4.,4.)
                self.error_rate+=span/(.08+span)*(derivative-self.error_rate)
            else:self.error_rate=0.
        if self.last_measurement is None or now-self.last_measurement[0]>.25:
            self.error_rate=0.
        # Continuous demand to zero, with a small start/stop hysteresis.
        if abs(ex)<=(.06 if self.centered else .025):self.centered=True
        else:self.centered=False
        p=.65*ex;d=.09*self.error_rate
        # Body-rate feedback only damps a turn already directed toward the target.
        damping=.0012*rate if rate is not None and rate*ex>0 else 0.
        lead_error=ex+.14*self.error_rate
        demand=p+d-damping
        braking=ex*lead_error<=0 or demand*ex<=0
        if self.centered or braking or not k.initialized:demand=0.
        demand=clamp(demand,-.55,.55)
        out.yaw=demand*settings.yaw_limit
        lateral_allowed=(valid and age<=.15 and rate is not None and abs(rate)<=8.
                         and abs(ex)<=.25 and abs(demand)<=.08)
        lateral_reason=('waiting_measurements' if not valid or age>.15 else 'waiting_heading'
                        if rate is None else 'align_yaw_first')
        out.roll=self.lateral.update(lateral_observation,now,lateral_allowed,lateral_reason)
        self.roll=out.roll
        # Fast attenuation, gradual recovery. Spacing remains the final distance gate.
        target=min(clamp((.50-abs(ex))/.35,0,1),clamp((.30-abs(demand))/.20,0,1))
        if rate is not None:target=min(target,clamp((45.-abs(rate))/30.,0,1))
        if occluded or not k.initialized:target=0.
        self.forward_factor=min(target,self.forward_factor+dt)
        self.diagnostics=dict(revision=__revision__,error_x=ex,error_rate=self.error_rate,
            measured_error=measured_error,heading_rate_deg_s=rate,
            heading_age_ms=None if not fresh_heading(heading,now) else (now-heading['time'])*1000,
            proportional=p,derivative=d,rate_damping=damping,lead_error=lead_error,
            braking=braking,centered=self.centered,yaw_normalized=demand,
            forward_alignment_scale=self.forward_factor,lateral_control=dict(self.lateral.diagnostics))
        return out,ex,ey
