"""Headless core. Owns no sockets, UI widgets or aircraft authority."""
from dataclasses import asdict, replace
import math
from .types import Intent, Kinematics
from .tracker import BBoxTracker
from .policy import SmartTrackingController, DeterministicTrackingPolicy
from .spacing import VisualSpacingController
from .edge_search import EdgeYawSearch
from .prediction import (COAST_SECONDS, COAST_YAW, COAST_VERTICAL, STABLE_FORWARD_SECONDS,
                         fade, forward_fade, continuation_threshold)


class FollowController:
    def __init__(self): self.reset()

    def reset(self):
        self.tracker=BBoxTracker(); self.smart=SmartTrackingController()
        self.policy=DeterministicTrackingPolicy(); self.spacing=VisualSpacingController()
        self.last_accepted=None; self.rejected=True; self.result_time=None
        self.frame_size=None; self.result_id=None
        self.missing_detection=False; self.last_track_command=None
        self.last_center_command=None
        self.edge_search=EdgeYawSearch()

    def observe(self,detections,source_time,now,frame_id,size,settings):
        if self.frame_size is not None and self.frame_size!=size: self.reset()
        if self.frame_size is None: self.tracker=BBoxTracker(size)
        self.frame_size=size
        if self.result_id is not None and frame_id<=self.result_id: return False
        self.result_id=frame_id; self.result_time=source_time
        self.missing_detection=not detections
        if not math.isfinite(source_time) or not 0<=now-source_time<=settings.stale_seconds:
            self.rejected=True;self.edge_search.reset(); return False
        # Release the association gate after prolonged loss; policy remembers
        # loss age until a genuinely new measurement is accepted.
        t=self.tracker
        if t.measurement_time is not None and now-t.measurement_time>settings.reacquire_seconds:
            t.reset()
        accepted=t.update(detections,source_time,now,frame_id,
                          max(settings.new_track_confidence,settings.confidence),
                          settings.confidence,continuation_threshold(settings.confidence))
        self.rejected=not accepted
        if accepted: self.last_accepted=t.accepted
        self.edge_search.observe(accepted,not detections,t.confirmed,
                t.accepted.box if accepted else None,t.snapshot(source_time),*size,
                source_time,frame_id,settings)
        return accepted

    def tick(self,now,width,height,video_time,settings,heading=None):
        if self.frame_size is not None and self.frame_size!=(width,height):self.reset()
        t=self.tracker;k=t.snapshot(now);quality=t.quality();uncertainty=t.uncertainty(now)
        mtime=t.measurement_time
        age=math.inf if mtime is None else max(0.,now-mtime)
        stream_stale=video_time is None or not 0<=now-video_time<=settings.stale_seconds
        result_stale=self.result_time is None or not 0<=now-self.result_time<=settings.stale_seconds
        stale=stream_stale or result_stale
        real_fresh=not self.rejected and age<=.25
        raw,ex,ey=self.smart.compute(k,width,height,age,not real_fresh,now,settings)
        raw.forward*=settings.forward_limit
        current_center=replace(raw);measurement_scale=1. if real_fresh else 0.
        search=self.edge_search.update(now,self.missing_detection,stale,
                    self.spacing.phase,self.spacing.close,settings,heading)
        cmd=self.policy.update(now,k,width,height,mtime if t.confirmed else None,
                    t.measurement_id if t.confirmed else None,raw,settings,
                    abort=stale,accepted=real_fresh,search=search)
        box=k.box
        measured_box=None if self.last_accepted is None else self.last_accepted.box
        clipped=measured_box is not None and (measured_box.x1<=.02*width or measured_box.y1<=.02*height
                    or measured_box.x2>=.98*width or measured_box.y2>=.98*height)
        raw_width=0. if measured_box is None else measured_box.width/width
        cmd,projected=self.spacing.update(now,self.policy.state=='TRACK',t.confirmed,
                    not real_fresh,clipped,self.policy.reason=='search_timeout',
                    t.measurement_id,mtime,age,raw_width,k.width/width,ex,ey,cmd,settings)
        if real_fresh and t.confirmed and self.policy.state=='TRACK':
            self.last_center_command=replace(cmd)
        previous=self.last_center_command
        std=uncertainty.get('position_std_px')
        forecast_ok=std is not None and std[0]<width*.12 and std[1]<height*.12
        coast=(settings.follow_and_hold and self.missing_detection and not real_fresh
               and t.confirmed and age<=COAST_SECONDS and not stale
               and self.policy.state=='COAST' and previous is not None and forecast_ok)
        forward_coast=False;brief_gap=False
        if coast:
            def centering(old,current,limit):
                return math.copysign(min(abs(old),abs(current),limit)*fade(age),old) if old*current>0 else 0.
            cmd=Intent(centering(previous.yaw,current_center.yaw,COAST_YAW*settings.yaw_limit),
                       centering(previous.vertical,current_center.vertical,COAST_VERTICAL*settings.vertical_limit))
            forward_coast=(quality['stable'] and age<STABLE_FORWARD_SECONDS
                  and not self.spacing.close and not clipped and self.spacing.phase=='APPROACH'
                  and projected<settings.stop_width*.90 and previous.forward>0
                  and abs(ex)<.55 and abs(ey)<.55)
            if forward_coast:cmd.forward=previous.forward*forward_fade(age)
            brief_gap=forward_coast and age<=.25
        if search['active']:
            cmd=Intent(yaw=float(search['direction']));coast=forward_coast=brief_gap=False
        if stale:cmd=Intent();coast=forward_coast=brief_gap=False
        recovery_reason=('coast_active' if coast else 'video_or_result_unavailable' if stale
                         else 'measured_search_active' if search['active'] else 'yolo_available' if real_fresh
                         else 'competing_detection' if not self.missing_detection
                         else 'unconfirmed_target' if not t.confirmed else 'coast_expired' if age>COAST_SECONDS
                         else 'prediction_uncertainty' if not forecast_ok else 'no_prior_track_command')
        forward_reason=('forward_bridge_active' if forward_coast else recovery_reason if not coast
                        else quality['reason'] if not quality['stable'] else 'recent_proximity' if self.spacing.close
                        else 'bbox_clipped' if clipped else 'spacing_hold' if self.spacing.phase!='APPROACH'
                        else 'forward_horizon_expired' if age>=STABLE_FORWARD_SECONDS
                        else 'no_previous_forward_command' if previous.forward<=0 else 'alignment_or_size_limit')
        show_track=t.confirmed and (age<=.300 or coast) and not stream_stale and not result_stale
        track_confidence=(self.last_accepted.confidence*math.exp(-.25*age/.300)
                          if show_track and self.last_accepted is not None else None)
        return {'state':self.policy.state,'reason':self.policy.reason,
                'edge_search':search,
                'brief_detection_gap':brief_gap,
                'prediction_recovery':coast,
                'prediction_forward_allowed':forward_coast,
                'recovery_reason':recovery_reason,'forward_recovery_reason':forward_reason,
                'prediction_quality':quality,'prediction_uncertainty':uncertainty,
                'continuous_dance':settings.follow_and_hold,'heading':heading,
                'prediction_anchor':t.anchor(),
                'prediction_display_allowed':coast,
                'track_support':('prediction' if not real_fresh else 'weak_yolo' if t.measurement_weak else 'yolo') if show_track else 'none',
                'detector_continuation_threshold':continuation_threshold(settings.confidence),
                'detector_acquisition_threshold':max(settings.new_track_confidence,settings.confidence),
                'last_strong_measurement_time':t.last_strong_time,
                'bbox_clipped':clipped,
                'stop_width_ratio':settings.stop_width,
                'spacing_resume_hits':self.spacing.resume_hits,
                'spacing_phase':self.spacing.phase,'spacing_reason':self.spacing.reason,
                'state_age_ms':max(0.,now-self.policy.entered)*1000.,
                'time_since_last_detection_ms':None if not math.isfinite(self.policy.time_since_detection) else self.policy.time_since_detection*1000.,
                'measurement_fresh':self.policy.measurement_fresh,
                'new_measurement':self.policy.new_measurement,
                'reacquire_hits':self.policy.hits,
                'policy_transition':self.policy.transition,
                'current_edge_candidate':self.policy.current_edge,
                'spacing_forward_scale':self.spacing.forward_scale,
                'spacing_decelerating':self.spacing.decelerating,
                'spacing_deceleration_started':self.spacing.deceleration_started,
                'spacing_brake_started':self.spacing.brake_started,
                'spacing_stopped_now':self.spacing.stopped_now,
                'spacing_completed_now':self.spacing.completed_now,
                'spacing_close_guard_armed':self.spacing.close,
                'spacing_close_cleared':self.spacing.close_cleared,
                'spacing_threshold_hits':self.spacing.hits,
                'spacing_brake_trigger_width_ratio':self.spacing.trigger_width,
                'spacing_brake_command':self.spacing.reverse,
                'spacing_width_growth_per_second':self.spacing.growth_rate,
                # The policy controllers already apply these design limits.
                # Dance normalizes against them before the GUI's final axis
                # caps are applied, avoiding two independent attenuations.
                'intent_basis':{
                    'yaw':max(.001,settings.yaw_limit),
                    'vertical':max(.001,settings.vertical_limit),
                    'roll':.45,
                    'forward':max(.001,settings.forward_limit),
                },
                'intent':asdict(cmd),'authorized_intent':asdict(Intent()),
                'command_sent':False,'measurement_id':t.measurement_id,
                'measurement_time':mtime,'measurement_scale':measurement_scale,
                'frame_size':[width,height],
                'tracking_confidence':track_confidence,
                'measurement_age_ms':None if not math.isfinite(age) else age*1000,
                'confirmed':t.confirmed,'confirmation_hits':t.hits,
                'accepted':real_fresh,'raw_width_ratio':raw_width,
                'projected_width_ratio':projected,'error_x':ex,'error_y':ey,
                'kinematics':asdict(k),'prediction_time':now,
                'track_box':asdict(box) if show_track else None,
                'stale':stream_stale or result_stale,'exit_edge':self.policy.edge}
