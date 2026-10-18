"""Headless core. Owns no sockets, UI widgets or aircraft authority."""
from dataclasses import asdict, replace
import math
from .types import Intent, Kinematics
from .tracker import BBoxTracker
from .policy import SmartTrackingController, DeterministicTrackingPolicy
from .spacing import VisualSpacingController
from .edge_search import EdgeYawSearch
from .recovery_search import RecoverySearch
from .centering import DampedTrackingController
from .approach import forward_diagnostics
from .class_memory import NeoClassMemory
from .pursuit import RecoveryVector
from .nano_guard import ForwardAnchorHistory, RealNeoEvidence, MEASUREMENT_SECONDS, current_visual_support, sustained_real_neo
from .prediction import (COAST_SECONDS, COAST_YAW, COAST_VERTICAL, STABLE_FORWARD_SECONDS,
                         fade, forward_fade, continuation_threshold)


class FollowController:
    def __init__(self, tracker_backend='legacy', target_class_id=None, target_class_name=None):
        # Historical headless replays retain their original backend. Live sessions
        # explicitly use BoT-SORT; this also keeps comparisons reproducible.
        if tracker_backend not in ('legacy', 'botsort'):
            raise ValueError('Unknown tracking backend: '+tracker_backend)
        self.tracker_backend=tracker_backend
        self.target_class_id=target_class_id
        self.target_class_name=target_class_name
        self.reset()

    def _new_tracker(self, size=(640,480)):
        if self.tracker_backend=='botsort':
            from .botsort_tracker import BoTSORTTracker
            return BoTSORTTracker(size)
        return BBoxTracker(size)

    def reset(self):
        self.tracker=self._new_tracker(); self.smart=self._new_centering()
        self.policy=DeterministicTrackingPolicy(); self.spacing=VisualSpacingController()
        self.last_accepted=None; self.rejected=True; self.result_time=None
        self.result_received_at=None
        self.target_identity=None
        self.class_memory=NeoClassMemory();self.class_support=None
        self.frame_size=None; self.result_id=None
        self.missing_detection=False; self.last_track_command=None
        self.last_center_command=None
        self.last_forward_anchor=None
        self.forward_anchors=ForwardAnchorHistory()
        self.real_neo_evidence=RealNeoEvidence()
        self.recovery_vector=RecoveryVector()
        self.edge_search=RecoverySearch() if self.tracker_backend=='botsort' else EdgeYawSearch()
        self.raw_detections=[]
        self.detection_summary={}
        self.control_result_floor=None

    def restart_control(self):
        """Reset control history and spacing without erasing target identity."""
        self.smart=self._new_centering();self.policy=DeterministicTrackingPolicy()
        self.spacing=VisualSpacingController();self.last_center_command=self.last_track_command=None
        self.last_forward_anchor=None
        self.forward_anchors=ForwardAnchorHistory()
        self.real_neo_evidence=RealNeoEvidence()
        self.recovery_vector=RecoveryVector()
        self.edge_search.reset()
        self.control_result_floor=self.result_id

    def _new_centering(self):
        return DampedTrackingController() if self.tracker_backend=='botsort' else SmartTrackingController()

    def observe(self,detections,source_time,now,frame_id,size,settings,image=None,continuity=None,allow_search_cues=False):
        if self.frame_size is not None and self.frame_size!=size: self.reset()
        if self.frame_size is None: self.tracker=self._new_tracker(size)
        self.frame_size=size
        if self.result_id is not None and frame_id<=self.result_id: return False
        self.result_id=frame_id; self.result_time=source_time
        self.result_received_at=now
        self.raw_detections=[dict(box=list(vars(d.box).values()),confidence=d.confidence,
                                 class_id=d.class_id,class_name=d.class_name,
                                 model_class_name=d.model_class_name,class_scores=list(d.class_scores)) for d in detections]
        # Preserve every class for display/logging, but gate before association,
        # weak continuation, candidate verification and search reacquisition.
        eligible=[d for d in detections if self.target_class_id is None or
                  (d.class_id==self.target_class_id and
                   (self.target_class_name is None or d.class_name==self.target_class_name))]
        counts={}
        for d in detections:
            key=d.class_name or 'unknown';counts[key]=counts.get(key,0)+1
        self.detection_summary=dict(class_counts=counts,total=len(detections),
            target_eligible=len(eligible),display_only=len(detections)-len(eligible),
            target_class_id=self.target_class_id,target_class_name=self.target_class_name,
            filter_reason='neo_only' if self.target_class_name=='neo' else 'class_filter' if self.target_class_id is not None else 'unfiltered')
        raw_detections=detections
        support=None
        if self.tracker_backend=='botsort' and self.target_class_id==0 and self.target_class_name=='neo':
            resolved,support=self.class_memory.resolve(detections,self.tracker,source_time,
                                                       max(settings.confidence,settings.new_track_confidence))
            if resolved is not None:eligible=eligible+[resolved]
        self.detection_summary['identity_supported']=int(support is not None)
        detections=eligible
        self.missing_detection=not detections
        if not math.isfinite(source_time) or not 0<=now-source_time<=settings.stale_seconds:
            self.rejected=True;self.edge_search.reset();self.real_neo_evidence=RealNeoEvidence();return False
        # Release the association gate after prolonged loss; policy remembers
        # loss age until a genuinely new measurement is accepted.
        t=self.tracker
        if self.tracker_backend=='legacy' and t.measurement_time is not None and now-t.measurement_time>settings.reacquire_seconds:
            t.reset()
        kwargs={'image':image,'continuity':continuity if support is None else None,
                'match_thresholds':(settings.botsort_match_threshold,settings.botsort_low_match_threshold,
                                    settings.botsort_unconfirmed_match_threshold)} if self.tracker_backend=='botsort' else {}
        accepted=t.update(detections,source_time,now,frame_id,
                          max(settings.new_track_confidence,settings.confidence),
                          settings.confidence,continuation_threshold(settings.confidence),**kwargs)
        self.rejected=not accepted
        if self.tracker_backend=='botsort' and t.numerical_event:
            self.restart_control();self.last_accepted=None;self.target_identity=None
            self.class_memory=NeoClassMemory();self.class_support=None
        if accepted:
            self.last_accepted=t.accepted
            self.class_support=support
            if self.tracker_backend=='botsort' and support is None:
                self.class_memory.remember(t,raw_detections,source_time,frame_id,
                                           max(settings.confidence,settings.new_track_confidence))
            # Preserve explicit class provenance, including a bounded history
            # resolution recorded separately in class_support. A forecast alone
            # can never create this measurement or a new target identity.
            if self.target_class_id is not None and t.confirmed:
                self.target_identity=dict(class_id=self.target_class_id,class_name=self.target_class_name,
                    logical_target_id=getattr(t,'logical_target_id',None),
                    measurement_id=t.measurement_id,measurement_time=t.measurement_time)
        self.real_neo_evidence.observe(raw_detections,accepted and support is None,t.confirmed,
                t.accepted.box if accepted else None,getattr(t,'logical_target_id',None),
                source_time,frame_id,size,max(settings.confidence,settings.new_track_confidence))
        self.edge_search.observe(accepted and support is None,not detections,t.confirmed,
                t.accepted.box if accepted else None,t.snapshot(source_time),*size,
                source_time,frame_id,settings,
                strong=bool(accepted and t.accepted.confidence>=max(settings.confidence,settings.new_track_confidence)))
        if self.tracker_backend=='botsort' and allow_search_cues and not accepted:
            self.edge_search.observe_candidate(raw_detections,source_time,frame_id,size,settings)
        return accepted

    def tick(self,now,width,height,video_time,settings,heading=None,control_available=True,*,visual=None,generation=None):
        if visual:
            visual=dict(visual,control_max_age_seconds=settings.nano_visual_max_age_seconds)
        if self.frame_size is not None and self.frame_size!=(width,height):self.reset()
        t=self.tracker;k=t.snapshot(now);quality=t.quality();uncertainty=t.uncertainty(now)
        mtime=t.measurement_time
        identity=self.target_identity
        identity_valid=bool(identity is not None and t.confirmed
            and identity['logical_target_id']==getattr(t,'logical_target_id',None)
            and identity['measurement_id']==t.measurement_id and identity['measurement_time']==mtime)
        age=math.inf if mtime is None else max(0.,now-mtime)
        stream_stale=video_time is None or not 0<=now-video_time<=settings.stale_seconds
        result_stale=self.result_time is None or not 0<=now-self.result_time<=settings.stale_seconds
        stale=stream_stale or result_stale
        after_restart=self.control_result_floor is None or (self.result_id is not None and self.result_id>self.control_result_floor)
        real_fresh=not self.rejected and age<=MEASUREMENT_SECONDS and after_restart
        centering_kwargs={}
        if self.tracker_backend=='botsort':
            measured_error=None if self.last_accepted is None else (self.last_accepted.box.cx-width*.5)/(width*.5)
            centering_kwargs=dict(heading=heading,measurement_time=mtime,measured_error=measured_error)
            centering_kwargs['current_visual']=bool(real_fresh and identity_valid
                and self.class_support is None and not t.measurement_weak
                and self.last_accepted is not None
                and self.last_accepted.confidence>=max(.75,settings.confidence,settings.new_track_confidence)
                and current_visual_support(visual,now,identity['logical_target_id'],generation,
                                           list(vars(k.box).values())))
            centering_kwargs['continuity_verified']=sustained_real_neo(dict(
                real_neo_evidence=self.real_neo_evidence.snapshot(),target_identity=identity,
                measurement_id=t.measurement_id,measurement_time=mtime))
            if (real_fresh and identity_valid and self.class_support is None and not t.measurement_weak
                    and self.last_accepted is not None and t.engine is not None):
                gmc=t.engine.gmc;b=self.last_accepted.box
                centering_kwargs['lateral_observation']=dict(time=mtime,point=(b.cx,b.cy),width=width,frame_id=t.engine.frame_id,
                    identity_key=(generation,identity['logical_target_id']),
                    warp=gmc.last_warp.tolist(),camera_valid=gmc.status=='applied')
        raw,ex,ey=self.smart.compute(k,width,height,age,not real_fresh,now,settings,**centering_kwargs)
        raw.forward*=settings.forward_limit
        current_center=replace(raw);measurement_scale=1. if real_fresh else 0.
        std=uncertainty.get('position_std_px')
        forecast_ok=std is not None and std[0]<width*.12 and std[1]<height*.12
        # Inference latency has already elapsed at receipt. Give a fresh result
        # a bounded hold from receipt, while still enforcing source/video age
        # and the tracker's finite, non-renewable candidate pause budget.
        from .botsort_motion import CANDIDATE_GAP_SECONDS, CANDIDATE_PAUSE_SECONDS
        pending=bool(getattr(t,'candidate_pending',False) and not stale
            and self.result_received_at is not None and 0<=now-self.result_received_at<=CANDIDATE_GAP_SECONDS
            and t.pause_started is not None and now-t.pause_started<=CANDIDATE_PAUSE_SECONDS)
        search_kwargs=dict(candidate_pending=pending,
                           control_available=control_available,
                           bridge_reliable=forecast_ok and quality['stable'] and age<=COAST_SECONDS) if self.tracker_backend=='botsort' else {}
        search=self.edge_search.update(now,self.rejected if self.tracker_backend=='botsort' else self.missing_detection,stale,
                    self.spacing.phase,self.spacing.close,settings,heading,**search_kwargs)
        if search.get('candidate_guidance') or search.get('direction_memory'):pending=False
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
        if self.tracker_backend=='botsort' and cmd.forward>0:
            cmd.forward*=self.smart.forward_factor
        if real_fresh and t.confirmed and self.policy.state=='TRACK':
            self.last_center_command=replace(cmd)
            if self.class_support is None and not t.measurement_weak:
                self.last_forward_anchor=dict(time=mtime,identity=getattr(t,'logical_target_id',None),
                    forward=cmd.forward,box=list(vars(measured_box).values()),
                    allowed=bool(cmd.forward>0 and not clipped and not self.spacing.close
                                 and self.spacing.phase=='APPROACH'))
                self.forward_anchors.remember(self.last_forward_anchor,now)
        self.forward_anchors.prune(now)
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
            cmd=Intent(yaw=float(search.get('scan_yaw',0.) if search.get('phase')=='SCAN' else search['direction']));coast=forward_coast=brief_gap=False
        if pending and not real_fresh:
            cmd=Intent();coast=forward_coast=brief_gap=False
        if stale:cmd=Intent();coast=forward_coast=brief_gap=False
        if self.class_support is not None and not search['active']:
            # Ambiguous class support can maintain limited centering only. It
            # never grants translation, renews search, or outlives real NEO proof.
            cmd.forward=cmd.roll=0.;forward_coast=brief_gap=False
            cmd.yaw=max(-COAST_YAW*settings.yaw_limit,min(COAST_YAW*settings.yaw_limit,cmd.yaw))
            cmd.vertical=max(-COAST_VERTICAL*settings.vertical_limit,min(COAST_VERTICAL*settings.vertical_limit,cmd.vertical))
            if now>self.class_support['expires_at']:cmd=Intent();coast=False
        elif self.class_support is not None and search['source_time']>self.class_support['last_real_neo_time']:
            cmd=Intent()  # A history-resolved box cannot seed a new search.
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
        from .botsort_tracker import low_threshold
        decision={'state':self.policy.state,'reason':self.policy.reason,
                'real_neo_evidence':self.real_neo_evidence.snapshot(),
                'tracking':t.diagnostics() if self.tracker_backend=='botsort' else {'backend':'legacy'},
                'candidate_pending':pending,
                'verified_forward_anchor':self.last_forward_anchor,
                'verified_forward_history':self.forward_anchors.snapshot(),
                'result_received_at':self.result_received_at,
                'centering_control':getattr(self.smart,'diagnostics',None),
                'forward_control':forward_diagnostics(self.policy.state,self.spacing.phase,self.spacing.reason,
                    self.spacing.forward_scale,getattr(self.smart,'forward_factor',1.),real_fresh,clipped,cmd.forward),
                'raw_detections':self.raw_detections,'raw_detection_time':self.result_time,
                'detection_summary':self.detection_summary,
                'class_identity':dict(self.class_memory.diagnostics),
                'class_support':self.class_support,
                # An empty/rejected result clears t.accepted, not the identity
                # of the last real neo measurement. Dance still enforces all
                # freshness, coast, heading and search limits independently.
                'selected_class_id':identity['class_id'] if identity_valid else None,
                'selected_class_name':identity['class_name'] if identity_valid else None,
                'target_identity':dict(identity) if identity_valid else None,
                'target_class_source':('temporal_identity' if self.class_support else 'accepted_measurement' if real_fresh else 'retained_measurement') if identity_valid else 'none',
                'selected_measurement_box':None if t.accepted is None else list(vars(t.accepted.box).values()),
                'edge_search':search,
                'brief_detection_gap':brief_gap,
                'prediction_recovery':coast,
                'prediction_forward_allowed':forward_coast,
                'recovery_reason':recovery_reason,'forward_recovery_reason':forward_reason,
                'prediction_quality':quality,'prediction_uncertainty':uncertainty,
                'continuous_dance':settings.follow_and_hold,'heading':heading,
                'prediction_anchor':t.anchor(),
                'prediction_display_allowed':coast,
                'track_support':('class_memory' if self.class_support else 'prediction' if not real_fresh else 'weak_yolo' if t.measurement_weak else 'yolo') if show_track else 'none',
                'detector_continuation_threshold':low_threshold(settings.confidence) if self.tracker_backend=='botsort' else continuation_threshold(settings.confidence),
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
                'video_time':video_time,
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
        decision['generation']=generation
        decision['motion_settings']={name:getattr(settings,name) for name in (
            'forward_window_width','forward_window_height','forward_soft_zone',
            'forward_yaw_demand_limit','forward_yaw_rate_limit','forward_rise_per_second',
            'forward_require_nano','coordinated_turns','nano_visual_max_age_seconds','search_translation_seconds',
            'search_forward_fraction','search_side_fraction','search_look_seconds','search_settle_seconds',
            'scan_observation_degrees','scan_yaw_limit','scan_observation_mode','scan_observation_yaw')}
        self.recovery_vector.remember(decision,visual,now,control_available)
        decision['recovery_anchor']=self.recovery_vector.snapshot()
        return decision
