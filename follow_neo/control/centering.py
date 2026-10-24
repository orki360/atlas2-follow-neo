"""Damped live centering; no aircraft authority and no additional tracking filter."""
import math
from follow_neo.control.policy import SmartTrackingController
from follow_neo.transport.telemetry import fresh_heading
from follow_neo.shared.types import clamp
from .. import __revision__
from follow_neo.control.approach import LateralEvidence
from follow_neo.control.position_motion import BearingMotion, YawHandoff


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
        self.motion={'botsort':BearingMotion(),'nanotrack':BearingMotion()}
        self.curve=self.motion['botsort'].curve
        self.handoff=YawHandoff()
        self.position_key=None

    def compute(self,k,w,h,age,occluded,now,settings,heading=None,measurement_time=None,measured_error=None,lateral_observation=None,current_visual=False,continuity_verified=False,position=None):
        position=position or {}
        bot_time,bot_error=measurement_time,measured_error
        visual_position=(position or {}).get('source')=='nanotrack'
        identity_key=(lateral_observation or {}).get('identity_key')
        position_key=(('nanotrack',position['generation'],position['identity'],position['epoch'])
                      if visual_position else ('botsort',identity_key))
        self.position_key=position_key
        if visual_position:
            measurement_time=position['source_time']
            measured_error=(k.cx-w*.5)/(w*.5)
        out,ex,ey=super().compute(k,w,h,age,occluded,now,settings)
        rate=self.heading_rate.update(heading,now)
        dt=0. if self.last_tick is None else max(0.,min(.1,now-self.last_tick))
        self.last_tick=now
        valid=(not occluded and k.initialized and measurement_time is not None
               and measured_error is not None and math.isfinite(measured_error))
        # Continuous demand to zero, with a small start/stop hysteresis.
        if abs(ex)<=(.06 if self.centered else .025):self.centered=True
        else:self.centered=False
        combined=bool(valid and continuity_verified and age<=.25
                      and (current_visual or not settings.forward_require_nano))
        curve_allowed=combined and settings.coordinated_turns and rate is not None
        # Warm the detector history even while Nano supplies position. Samples
        # are differentiated ONLY against their own source and original clock.
        bot_valid=(not occluded and k.initialized and bot_time is not None
                   and bot_error is not None and math.isfinite(bot_error))
        bot_rate,bot_curve=self.motion['botsort'].update(bot_time,bot_error,now,identity_key,
            bot_valid,bot_valid and lateral_observation is not None,curve_allowed)
        candidate=position if visual_position else position.get('motion_candidate')
        if candidate:
            nano_error=(candidate['center'][0]-w*.5)/(w*.5)
            nano_key=tuple(candidate[key] for key in ('generation','identity','epoch'))
            nano_rate,nano_curve=self.motion['nanotrack'].update(candidate['source_time'],nano_error,
                now,nano_key,valid,valid and lateral_observation is not None,curve_allowed)
        elif position.get('reason') not in ('visual_expired','waiting_visual_update',
                                           'nano_position_unverified','missing_source_alignment'):
            # Rejected appearance/identity/geometry must rebuild Nano history.
            # A short missing update may retain old samples, never old authority.
            if self.motion['nanotrack'].last_measurement is not None:
                self.motion['nanotrack']=BearingMotion()
        source='nanotrack' if visual_position else 'botsort'
        self.last_measurement=self.motion[source].last_measurement;self.curve=self.motion[source].curve
        self.error_rate,curve=(nano_rate,nano_curve) if visual_position else (bot_rate,bot_curve)
        p=.65*(ex+curve['lead']);d=.09*self.error_rate
        # Body-rate feedback only damps a turn already directed toward the target.
        damping=.0012*rate if rate is not None and rate*ex>0 else 0.
        lead_error=ex+.14*self.error_rate
        demand=p+d-damping
        braking=ex*lead_error<=0 or demand*ex<=0
        if (self.centered and not curve['active']) or braking or not k.initialized:demand=0.
        demand=clamp(demand,-.55,.55)
        out.yaw=demand*settings.yaw_limit
        lateral_limit=.25 if current_visual else .15
        lateral_combined=combined and current_visual
        lateral_allowed=(valid and age<=lateral_limit and rate is not None and abs(rate)<=(35. if lateral_combined else 8.)
                         and abs(ex)<=(.70 if lateral_combined else .25) and abs(demand)<=(.40 if lateral_combined else .08))
        lateral_reason=('waiting_measurements' if not valid or age>lateral_limit else 'waiting_heading'
                        if rate is None else 'align_yaw_first')
        out.roll=self.lateral.update(lateral_observation,now,lateral_allowed,lateral_reason,current_visual=current_visual)
        self.roll=out.roll
        # Fast attenuation, gradual recovery. Spacing remains the final distance gate.
        # One visible envelope; fresh sustained real detections do not require
        # Nano to finish the same frame. Hard visual/identity vetoes still run
        # downstream in apply_visual and at transport time.
        fade=lambda value,limit:clamp((limit-abs(value))/(limit*settings.forward_soft_zone),0,1)
        factors=dict(horizontal=fade(ex,settings.forward_window_width),
                     vertical=fade(ey,settings.forward_window_height),
                     yaw_demand=fade(demand,settings.forward_yaw_demand_limit),
                     yaw_rate=fade(rate,settings.forward_yaw_rate_limit) if rate is not None else 1.)
        coordinated=bool(combined and settings.coordinated_turns and rate is not None)
        if coordinated:
            # A verified turn reduces approach speed continuously instead of
            # switching it off at a requested-yaw threshold. Image boundaries,
            # actual body-rate ceiling and downstream distance/loss gates stay
            # hard limits. No minimum forward command is introduced.
            excess=max(0.,abs(demand)/settings.forward_yaw_demand_limit-(1.-settings.forward_soft_zone))
            factors['yaw_demand']=1./(1.+2.*excess)
        if not combined:
            factors['evidence']=min(clamp((.50-abs(ex))/.35,0,1),
                clamp((.30-abs(demand))/.20,0,1),
                clamp((45.-abs(rate))/30.,0,1) if rate is not None else 1.)
        target=min(factors.values())
        # A brief detector gap must not restart the one-second acceleration
        # ramp. This only retains alignment history; the controller/Dance still
        # gate actual motion, prediction expiry, proximity and freshness.
        if not k.initialized or occluded and age>.25:target=0.
        elif occluded:target=min(target,self.forward_factor)
        rise=settings.forward_rise_per_second if combined else min(1.,settings.forward_rise_per_second)
        self.forward_factor=min(target,self.forward_factor+dt*rise)
        # Forward/side safety above used the unfiltered demand and live error.
        # Only an increase in ordinary yaw may be eased across a source switch.
        yaw,handoff_limited=self.handoff.update(demand,ex,now,position_key,identity_key,
            bool(position.get('handoff_enabled') and valid and identity_key is not None))
        out.yaw=yaw*settings.yaw_limit
        self.diagnostics=dict(revision=__revision__,error_x=ex,error_rate=self.error_rate,
            position_source='nanotrack' if visual_position else 'botsort',position_time=measurement_time,
            measured_error=measured_error,heading_rate_deg_s=rate,
            heading_age_ms=None if not fresh_heading(heading,now) else (now-heading['time'])*1000,
            proportional=p,derivative=d,rate_damping=damping,lead_error=lead_error,
            braking=braking,centered=self.centered,yaw_normalized=yaw,yaw_requested=demand,
            position_handoff_limited=handoff_limited,
            forward_alignment_scale=self.forward_factor,combined_motion=combined,
            coordinated_turn=coordinated,early_curve=curve,
            yaw_demand_policy='slowdown' if coordinated else 'ceiling',
            forward_factors=factors,forward_limiting_factor=min(factors,key=factors.get),
            forward_window=dict(width=settings.forward_window_width,height=settings.forward_window_height),
            lateral_control=dict(self.lateral.diagnostics))
        return out,ex,ey
