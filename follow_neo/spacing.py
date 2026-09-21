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

    def begin_stop(self,now,reason,pause,settings):
        self.reason=reason; self.pause=pause
        self.reverse=-min(.40,self.peak) if self.last_forward is not None and now-self.last_forward<=.50 else 0.
        if self.reverse<-.01:
            self.phase='BRAKE'; self.brake_since=now
        else:
            self.phase='STOPPED' if pause else 'VISUAL_HOLD' if settings.follow_and_hold else 'SEQUENCE_DONE'

    def update(self,now,track_active,valid,rejected,clipped,timed_out,mid,mtime,age,
               raw_width,filtered_width,ex,ey,command,settings):
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
        near_loss=self.close and (not good or rejected or clipped)
        if self.phase not in ('STOPPED','SEQUENCE_DONE'):
            if self.phase=='BRAKE':
                if near_loss or timed_out:
                    self.pause=True; self.reason='close_bbox_clipped' if clipped else 'close_track_lost'
                if now-self.brake_since>=.40:
                    self.phase='STOPPED' if self.pause else 'VISUAL_HOLD' if settings.follow_and_hold else 'SEQUENCE_DONE'
            elif near_loss or timed_out:
                self.begin_stop(now,'search_timeout' if timed_out else 'close_bbox_clipped' if clipped else 'close_track_lost',True,settings)
            elif self.phase=='VISUAL_HOLD' and new and width>=min(.50,stop*1.50):
                self.begin_stop(now,'hold_spacing_exceeded',True,settings)
            elif self.phase=='CONFIRM_STOP':
                if new:
                    if projected>=stop:
                        self.hits+=1; self.begin_stop(now,'bbox_threshold' if width>=stop else 'predicted_threshold',False,settings)
                    elif settings.follow_and_hold and width>=stop*.90:
                        self.phase='VISUAL_HOLD'; self.reason='visual_size_hold'
                    else: self.phase='STOPPED'; self.reason='threshold_not_confirmed'
                elif now-self.confirm_since>=.25:
                    self.begin_stop(now,'stop_confirmation_timeout',True,settings)
            elif self.phase=='APPROACH' and new and projected>=stop:
                self.phase='CONFIRM_STOP'; self.confirm_since=now; self.hits=1
                self.reason='first_threshold_forward_blocked' if width>=stop else 'predicted_threshold_forward_blocked'
            elif self.phase=='APPROACH' and settings.follow_and_hold and new and width>=stop*.90:
                self.phase='VISUAL_HOLD'; self.reason='visual_size_hold'
        if self.phase in ('STOPPED','SEQUENCE_DONE','CONFIRM_STOP'): out=Intent()
        elif self.phase=='BRAKE': out=Intent(forward=self.reverse)
        elif self.phase=='VISUAL_HOLD':
            self.reason='visual_size_hold'; out.roll=out.forward=0.
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
            out.forward=max(0,out.forward)*scale
            self.reason='predictive_deceleration' if scale<1 else 'approach'
        if self.last_forward is not None and now-self.last_forward>.50: self.peak=0.
        if out.forward>.01: self.peak=max(self.peak,out.forward); self.last_forward=now
        return out.bounded(),projected
