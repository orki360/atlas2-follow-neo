"""Visual spacing state machine port. All outputs remain simulated intents."""
from dataclasses import replace
import math
from .types import Intent, clamp


class VisualSpacingController:
    def __init__(self):
        self.phase='APPROACH'; self.reason='waiting_for_track'
        self.last_now=self.last_id=self.last_time=self.last_width=self.growth=None
        self.confirm_since=self.brake_since=self.last_forward=None
        self.close=False; self.pause=False; self.peak=0.; self.reverse=0.; self.hits=0
        self.trigger_width=0.; self.forward_scale=1.; self.growth_rate=None
        self.decelerating=False; self.deceleration_reported=False
        self.deceleration_started=False; self.brake_started=False
        self.stopped_now=False; self.completed_now=False
        self.resume_hits=0; self.resume_since=None; self.pause_since=None
        self.far_hits=0;self.far_since=None;self.close_cleared=False

    def clear_resume(self):
        self.resume_hits=0; self.resume_since=None

    def resume_confirmed(self,now,new):
        if new:
            if self.resume_since is None: self.resume_since=now
            self.resume_hits+=1
        return self.resume_hits>=3 and now-self.resume_since>=.10

    def begin_stop(self,now,width,reason,pause,settings):
        self.reason=reason; self.pause=pause
        self.trigger_width=width
        self.reverse=-min(.40,self.peak) if self.last_forward is not None and now-self.last_forward<=.50 else 0.
        if self.reverse<-.01:
            self.phase='BRAKE'; self.brake_since=now
        else:
            self.phase='STOPPED' if pause else 'VISUAL_HOLD' if settings.follow_and_hold else 'SEQUENCE_DONE'

    def update(self,now,track_active,valid,rejected,clipped,timed_out,mid,mtime,age,
               raw_width,filtered_width,ex,ey,command,settings):
        previous_phase=self.phase
        self.close_cleared=False
        self.deceleration_started=False; self.brake_started=False
        self.stopped_now=False; self.completed_now=False
        out=replace(command).bounded()
        if not math.isfinite(now) or self.last_now is not None and now<self.last_now:
            self.phase='STOPPED'; self.reason='invalid_control_time'
        else: self.last_now=now
        good=(valid and mid is not None and mtime is not None and mtime<=now
              and 0<=age<=.25 and 0<raw_width<=1.5 and 0<filtered_width<=1.5)
        new=good and (self.last_id is None or mid>self.last_id) and (self.last_time is None or mtime>self.last_time)
        width=max(raw_width,filtered_width) if good else self.last_width or 0.
        if new:
            if self.last_time is not None and self.last_width is not None:
                dt=mtime-self.last_time
                if .01<=dt<=.50:
                    slope=clamp((width-self.last_width)/dt,-2,2)
                    self.growth=slope if slope<=0 or self.growth is None else .35*slope+.65*max(0,self.growth)
                else: self.growth=None
            self.last_id=mid; self.last_time=mtime; self.last_width=width
        projected=width+max(0,self.growth or 0)*clamp((age if good else 0)+.15,0,.35)
        stop=settings.stop_width
        if new and (width>=stop*.60 or projected>=stop): self.close=True
        # Close is evidence of recent proximity, not an irreversible latch.
        # Clear only after several genuine, unclipped, distant measurements.
        distant=(settings.follow_and_hold and good and not rejected and not clipped
                 and track_active and width<stop*.50 and projected<stop*.60
                 and self.phase in ('APPROACH','TRACK_PAUSE'))
        if not distant:self.far_hits=0;self.far_since=None
        elif new:
            if self.far_since is None:self.far_since=mtime
            self.far_hits+=1
            if self.close and self.far_hits>=3 and mtime-self.far_since>=.10:
                self.close=False;self.close_cleared=True
        near_loss=self.close and (not good or rejected or clipped)
        if self.phase not in ('STOPPED','SEQUENCE_DONE'):
            if self.phase=='TRACK_PAUSE':
                if self.close_cleared:
                    self.phase='APPROACH';self.reason='distant_track_reconfirmed';self.clear_resume()
                elif timed_out or now-self.pause_since>=.65 or clipped:
                    self.phase='STOPPED'
                    self.reason='close_bbox_clipped' if clipped else 'close_track_lost'
                elif good and not rejected and track_active:
                    if self.resume_confirmed(now,new):
                        self.phase='VISUAL_HOLD' if width>=stop*.90 else 'APPROACH'
                        self.reason='close_track_reconfirmed'; self.clear_resume()
                else: self.clear_resume()
            elif self.phase=='BRAKE':
                if near_loss or timed_out:
                    self.pause=True; self.reason='close_bbox_clipped' if clipped else 'close_track_lost'
                if now-self.brake_since>=.40:
                    self.phase='STOPPED' if self.pause else 'VISUAL_HOLD' if settings.follow_and_hold else 'SEQUENCE_DONE'
            elif near_loss or timed_out:
                if settings.follow_and_hold and not timed_out and not clipped:
                    self.phase='TRACK_PAUSE'; self.pause_since=now
                    self.reason='close_track_pause'; self.clear_resume()
                else:
                    self.begin_stop(now,width,'search_timeout' if timed_out else 'close_bbox_clipped' if clipped else 'close_track_lost',True,settings)
            elif self.phase=='VISUAL_HOLD' and new and width>=min(.50,stop*1.50):
                self.begin_stop(now,width,'hold_spacing_exceeded',True,settings)
            elif self.phase=='CONFIRM_STOP':
                if new:
                    if projected>=stop:
                        self.hits+=1; self.begin_stop(now,width,'bbox_threshold' if width>=stop else 'predicted_threshold',False,settings)
                    elif settings.follow_and_hold:
                        # A derivative spike may cross the prediction threshold
                        # for one frame. Hold zero until several NEW clear frames
                        # confirm it; never turn a false alarm into a latched stop.
                        self.reason='threshold_rechecking'
                        if good and not rejected and track_active and self.resume_confirmed(now,new):
                            self.phase='VISUAL_HOLD' if width>=stop*.90 else 'APPROACH'
                            self.reason='threshold_cleared'; self.clear_resume(); self.hits=0
                            if width<stop*.50 and projected<stop*.60:
                                self.close=False  # confirmed distant after the false size spike
                    else: self.phase='STOPPED'; self.reason='threshold_not_confirmed'
                elif now-self.confirm_since>=.25:
                    self.begin_stop(now,width,'stop_confirmation_timeout',True,settings)
            elif self.phase=='APPROACH' and new and projected>=stop:
                self.phase='CONFIRM_STOP'; self.confirm_since=now; self.hits=1
                self.clear_resume()
                self.reason='first_threshold_forward_blocked' if width>=stop else 'predicted_threshold_forward_blocked'
            elif self.phase=='APPROACH' and settings.follow_and_hold and new and width>=stop*.90:
                self.phase='VISUAL_HOLD'; self.reason='visual_size_hold'
        self.forward_scale=1.
        if self.phase in ('STOPPED','SEQUENCE_DONE','CONFIRM_STOP','TRACK_PAUSE'):
            out=Intent(); self.forward_scale=0.
        elif self.phase=='BRAKE':
            out=Intent(forward=self.reverse); self.forward_scale=0.
        elif self.phase=='VISUAL_HOLD':
            self.reason='visual_size_hold'; out.roll=out.forward=0.; self.forward_scale=0.
            if not good or not track_active or rejected or clipped: out=Intent()
            else:
                error=(stop-width)/stop; aligned=abs(ex)<.35 and abs(ey)<.35
                if aligned and error>.10 and projected<stop*.90:
                    out.forward=min(.30,.60*(error-.10))*settings.forward_limit; self.reason='hold_approach'
                elif aligned and error<-.10:
                    out.forward=-min(.20,.50*(-error-.10))*settings.forward_limit; self.reason='hold_backoff'
        elif not good or not track_active or rejected or clipped:
            out.forward=0.
            if self.close: out=Intent()
            self.reason='waiting_for_confirmed_track'
        else:
            scale=clamp((stop-projected)/(stop*.40),0,1)
            self.forward_scale=scale
            out.forward=max(0,out.forward)*scale
            self.reason='predictive_deceleration' if scale<1 else 'approach'
        self.decelerating=self.phase=='APPROACH' and self.forward_scale<1.
        if self.decelerating and not self.deceleration_reported:
            self.deceleration_started=True; self.deceleration_reported=True
        if self.last_forward is not None and now-self.last_forward>.50: self.peak=0.
        if out.forward>.01: self.peak=max(self.peak,out.forward); self.last_forward=now
        self.brake_started=self.phase=='BRAKE' and previous_phase!=self.phase
        self.stopped_now=self.phase=='STOPPED' and previous_phase!=self.phase
        self.completed_now=self.phase=='SEQUENCE_DONE' and previous_phase!=self.phase
        self.growth_rate=self.growth
        return out.bounded(),projected
