"""Headless core. Owns no sockets, UI widgets or aircraft authority."""
from dataclasses import asdict
import math
from .types import Intent, Kinematics
from .tracker import BBoxTracker
from .policy import SmartTrackingController, DeterministicTrackingPolicy
from .spacing import VisualSpacingController


class FollowController:
    def __init__(self): self.reset()

    def reset(self):
        self.tracker=BBoxTracker(); self.smart=SmartTrackingController()
        self.policy=DeterministicTrackingPolicy(); self.spacing=VisualSpacingController()
        self.last_accepted=None; self.rejected=True; self.result_time=None
        self.frame_size=None; self.result_id=None

    def observe(self,detections,source_time,now,frame_id,size,settings):
        if self.frame_size is not None and self.frame_size!=size: self.reset()
        if self.frame_size is None: self.tracker=BBoxTracker(size)
        self.frame_size=size
        if self.result_id is not None and frame_id<=self.result_id: return False
        self.result_id=frame_id; self.result_time=source_time
        if not math.isfinite(source_time) or not 0<=now-source_time<=settings.stale_seconds:
            self.rejected=True; return False
        # Release the association gate after prolonged loss; policy remembers
        # loss age until a genuinely new measurement is accepted.
        t=self.tracker
        if t.measurement_time is not None and now-t.measurement_time>settings.reacquire_seconds:
            t.reset()
        accepted=t.update(detections,source_time,now,frame_id,settings.new_track_confidence)
        self.rejected=not accepted
        if accepted: self.last_accepted=t.accepted
        return accepted

    def tick(self,now,width,height,video_time,settings):
        if self.frame_size is not None and self.frame_size!=(width,height): self.reset()
        t=self.tracker; k=t.snapshot(now)
        mtime=t.measurement_time
        age=math.inf if mtime is None else max(0,now-mtime)
        stream_stale=video_time is None or now-video_time>settings.stale_seconds
        result_stale=self.result_time is None or now-self.result_time>settings.stale_seconds
        raw,ex,ey=self.smart.compute(k,width,height,age,self.rejected,now,settings)
        # Match the C++ 250-300ms measurement fade; recovery stays separate.
        measurement_scale=max(0.,min(1.,(.300-age)/.050))
        raw.yaw*=measurement_scale; raw.vertical*=measurement_scale
        if age>.300: raw=Intent()
        raw.forward*=settings.forward_limit
        # Unconfirmed candidates are not supplied as accepted policy observations.
        cmd=self.policy.update(now,k,width,height,mtime if t.confirmed else None,
                               t.measurement_id if t.confirmed else None,raw,settings,
                               abort=stream_stale or result_stale)
        box=k.box
        measured_box=None if self.last_accepted is None else self.last_accepted.box
        clipped=measured_box is not None and (measured_box.x1<=.02*width or measured_box.y1<=.02*height
                    or measured_box.x2>=.98*width or measured_box.y2>=.98*height)
        raw_width=0 if self.last_accepted is None else self.last_accepted.box.width/width
        cmd,projected=self.spacing.update(now,self.policy.state=='TRACK',t.confirmed,
                  self.rejected,clipped,self.policy.reason=='target_lost_timeout',
                  t.measurement_id,mtime,age,raw_width,k.width/width,ex,ey,cmd,settings)
        if stream_stale or result_stale: cmd=Intent()
        show_track=t.confirmed and age<=.300 and not stream_stale and not result_stale
        track_confidence=(self.last_accepted.confidence*math.exp(-.25*age/.300)
                          if show_track and self.last_accepted is not None else None)
        return {'state':self.policy.state,'reason':self.policy.reason,
                'spacing_phase':self.spacing.phase,'spacing_reason':self.spacing.reason,
                'intent':asdict(cmd),'authorized_intent':asdict(Intent()),
                'command_sent':False,'measurement_id':t.measurement_id,
                'measurement_time':mtime,'measurement_scale':measurement_scale,
                'frame_size':[width,height],
                'tracking_confidence':track_confidence,
                'measurement_age_ms':None if not math.isfinite(age) else age*1000,
                'confirmed':t.confirmed,'confirmation_hits':t.hits,
                'accepted':not self.rejected,'raw_width_ratio':raw_width,
                'projected_width_ratio':projected,'error_x':ex,'error_y':ey,
                'kinematics':asdict(k),'prediction_time':now,
                'track_box':asdict(box) if show_track else None,
                'stale':stream_stale or result_stale,'exit_edge':self.policy.edge}
