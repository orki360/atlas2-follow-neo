"""Separate a brief edge catch-up from a measured left/right loss scan.

The last measured image position selects a directional or centred span.
Both phases share one heading anchor and deadline. Search yaw may reach full
scale with rate-aware braking; no translation is requested.
"""
import math
from follow_neo.control.edge_search import EdgeYawSearch
from follow_neo.transport.telemetry import fresh_heading
from follow_neo.control.search_motion import (SearchHeadingRate,SearchYawProfile,braking_distance,
                            BOOST_MAX_SECONDS,NOMINAL_YAW_RATE,MAX_BOOST_ANGLE,search_duration_limit)
from follow_neo.control.search_sectors import SectorSweep,SECTOR_DEGREES

SEARCH_GRACE_SECONDS = .45
BOUNDARY_TOLERANCE = .5
REVERSAL_PAUSE = .15
VERIFY_GAP_SECONDS = .30
VERIFY_MAX_AGE_SECONDS = .45  # source interval plus processing latency
HEADING_PAUSE_SECONDS = .80  # hold zero, then resume the original bounded scan


class RecoverySearch:
    def __init__(self):self.reset()

    def reset(self):
        self.edge=EdgeYawSearch()
        self.started=self.until=self.source_time=self.measurement_id=None
        self.last_heading=self.heading_time=self.heading_source=None
        self.loss_heading=None;self.loss_heading_time=None;self.loss_heading_source=None
        self.phase='IDLE';self.reason='waiting_target';self.consumed=False
        self.eligible=False;self.offset=0.;self.angle=0.;self.duration=0.
        self.direction=0;self.target=0.;self.reversal_until=0.
        self.verifying=False;self.verify_hits=0;self.verify_since=None;self.verify_last=None
        self.loss_source=None
        self.rate=SearchHeadingRate();self.yaw_profile=SearchYawProfile()
        self.requested_duration=0.;self.boundaries=set()
        self.box_edge=False
        self.estimated_direction=0;self.direction_source='none'
        self.first_scan_direction=0;self.boost_exit_reason=None
        self.boost_brake_margin=braking_distance(NOMINAL_YAW_RATE*.6)
        self.centered=True;self.low=self.high=0.;self.span_mode='centered'
        self.heading_missing_since=None
        self.heading_resumes=0;self.last_heading_pause_seconds=None
        self.candidate_cue=None;self.cue_started=None;self.cue_cooldown_until=0.
        self.candidate_only=False;self.cue_recovery_used=False
        self.cue_scope='none'
        self.sectors=None;self.sector_require_frames=True
        self.direction_hint=None;self.route_revision=0
        self.look_seconds=.15;self.settle_seconds=.08
        self.sector_degrees=60.;self.scan_yaw_limit=1.
        self.observation_mode='stop';self.observation_yaw=.08

    def observe_candidate(self,detections,source_time,frame_id,size,settings):
        """Two raw NEO frames can guide yaw; never confirm identity or forward."""
        if not self.eligible:return
        if self.phase not in ('BOOST','SCAN') and (self.cue_recovery_used
                or self.source_time is None or not 0<source_time-self.source_time<=12.):return
        from follow_neo.control.nano_guard import overlap
        strong=[d for d in detections if d.class_id==0 and d.class_name=='neo'
                and d.confidence>=max(.75,settings.confidence,settings.new_track_confidence)]
        if len(strong)!=1:
            self.candidate_cue=None
            if detections:self.direction_hint=None
            return
        det=strong[0];box=list(vars(det.box).values());width,height=size
        if (det.box.width<=0 or det.box.height<=0 or not all(math.isfinite(v) for v in box)
                or det.box.x1<0 or det.box.y1<0 or det.box.x2>width or det.box.y2>height
                or any(d is not det and d.confidence>=min(.45,settings.confidence)
                       and overlap(box,list(vars(d.box).values()))>.3 for d in detections)):
            self.candidate_cue=None;self.direction_hint=None;return
        old=self.candidate_cue
        if old and (frame_id<=old['frame_id'] or source_time<=old['time']):return
        coherent=bool(old and 0<source_time-old['time']<=.30
            and abs(det.box.cx-old['cx'])<=max(width*.08,det.box.width*2,width*4.*(source_time-old['time']))
            and abs(det.box.cy-old['cy'])<=max(height*.08,det.box.height*2,height*1.5*(source_time-old['time']))
            and .5<=det.box.width/old['width']<=2 and .5<=det.box.height/old['height']<=2)
        self.candidate_cue=dict(frame_id=frame_id,time=source_time,hits=old['hits']+1 if coherent else 1,
            since=old['since'] if coherent else source_time,cx=det.box.cx,cy=det.box.cy,
            width=det.box.width,height=det.box.height,error_x=(det.box.cx-width*.5)/(width*.5),
            image_vx=(det.box.cx-old['cx'])/(source_time-old['time']) if coherent else 0.,frame_width=width)
        cue=self.candidate_cue
        if cue['hits']>=2 and source_time-cue['since']>=.06:
            self.direction_hint=dict(cue)

    def end(self,reason):
        self.phase='DONE';self.reason=reason;self.consumed=True;self.verifying=False

    def observe(self,accepted,missing,confirmed,box,k,width,height,source_time,measurement_id,settings,strong=True):
        if self.sectors:self.sectors.observe(source_time,measurement_id)
        if not accepted or box is None or (self.source_time is not None and source_time<=self.source_time):return
        if self.phase in ('BOOST','SCAN'):
            if not self.verifying or source_time-(self.verify_last or source_time)>VERIFY_GAP_SECONDS:
                self.verify_hits=0;self.verify_since=source_time
            self.verifying=True;self.verify_hits+=int(strong);self.verify_last=source_time
            self.source_time=source_time;self.measurement_id=measurement_id
            self.reason='reacquire_verifying'
            if confirmed and self.verify_hits>=3 and source_time-self.verify_since>=.06:
                # Seed the next loss from this measurement immediately. A DONE
                # flag must not shadow COAST if the very next frame is missing.
                self.reset()
            else:return
        if self.consumed:self.reset()
        self.source_time=source_time;self.measurement_id=measurement_id
        self.eligible=confirmed
        self.centered=.25<=box.cx/width<=.75
        self.edge.observe(accepted,missing,confirmed,box,k,width,height,source_time,measurement_id,settings,strong)
        self.box_edge=(box.x1<=.08*width if self.edge.direction<0 else box.x2>=.92*width if self.edge.direction>0 else False)
        evidence=self.edge.evidence
        vx=evidence.get('image_vx',0.);travel=evidence.get('net_travel',0.)
        recent=list(self.edge.history)[-3:]
        recent_steps=[b[1]-a[1] for a,b in zip(recent,recent[1:])]
        reversal=bool(len(recent)==3 and recent[-1][0]-recent[0][0]<=.30
            and abs(sum(recent_steps))>=.012 and all(x*sum(recent_steps)>0 for x in recent_steps))
        if reversal and sum(recent_steps)*vx<0:
            self.estimated_direction=1 if sum(recent_steps)>0 else -1;self.direction_source='recent_motion_reversal'
        elif (abs(vx)>=.035 and abs(travel)>=.012 and vx*travel>0
            and evidence.get('direction_agreement',0.)>=.65):
            # A filtered old trajectory must not beat the latest position
            # when the current Kalman velocity already points the other way.
            if k.initialized and k.vx/width*vx<0 and abs(k.vx/width)>.05 and abs(box.cx/width-.5)>.10:
                self.estimated_direction=1 if box.cx>width/2 else -1;self.direction_source='latest_side_after_reversal'
            else:
                self.estimated_direction=1 if vx>0 else -1;self.direction_source='measured_motion'
        elif abs(box.cx/width-.5)>.05:
            self.estimated_direction=1 if box.cx>width/2 else -1;self.direction_source='last_seen_side'
        else:
            self.estimated_direction=-1;self.direction_source='no_direction_left_fallback'

    def _scan(self,now):
        previous_phase=self.phase
        previous_direction=self.direction
        self.phase='SCAN';self.reason='sweeping'
        # Freeze the loss estimate for this search; BOOST must not silently
        # change the first scan direction. A reached boundary still reverses.
        side=self.first_scan_direction or -1
        self.target=self.high if side>0 else self.low
        if side*(self.offset-self.target)>=-BOUNDARY_TOLERANCE:
            self.boundaries.add(side);self.target=self.low if side>0 else self.high
        # A same-direction BOOST handoff needs no reversal pause. Still let
        # SCAN's rate-aware braking compute zero when the boundary is close.
        continues_boost=(previous_phase=='BOOST'
                         and previous_direction*(self.target-self.offset)>BOUNDARY_TOLERANCE)
        same_turn=self.rate.rate is not None and side*self.rate.rate>=-15.
        self.reversal_until=now if continues_boost or same_turn else now+REVERSAL_PAUSE
        self.yaw_profile.reset(now,float(previous_direction) if continues_boost else 0.)
        if self.angle>=120 and not self.candidate_only:
            if self.sectors is None:
                self.sectors=self._sectors(0.,side)
            self.target=self.sectors.target

    def _sectors(self,offset,side):
        return SectorSweep(offset,self.low,self.high,side,require_frames=self.sector_require_frames,
            look_seconds=self.look_seconds,settle_seconds=self.settle_seconds,
            step_degrees=self.sector_degrees,yaw_limit=self.scan_yaw_limit,
            observation_mode=self.observation_mode,observation_yaw=self.observation_yaw)

    def _retarget(self,side,now):
        """Change remaining route, retaining heading anchor, arc and deadline."""
        boundary=self.high if side>0 else self.low
        if side*(boundary-self.offset)<=BOUNDARY_TOLERANCE:return
        if side*(self.target-self.offset)>BOUNDARY_TOLERANCE:return
        if self.phase=='BOOST':self._scan(now)
        self.target=boundary
        if self.sectors:
            self.sectors=self._sectors(self.offset,side);self.target=self.sectors.target
        self.reversal_until=now;self.yaw_profile.reset(now);self.route_revision+=1

    def _set_span(self):
        self.span_mode='centered' if self.centered else 'directional'
        if self.centered:self.low,self.high=-self.angle/2,self.angle/2
        elif self.first_scan_direction>0:self.low,self.high=0.,self.angle
        else:self.low,self.high=-self.angle,0.

    def begin(self,now,settings,heading,anchor=None):
        """Shared scan planner used by target recovery and the explicit test."""
        self.sectors=None
        self.look_seconds=settings.search_look_seconds;self.settle_seconds=settings.search_settle_seconds
        self.sector_degrees=settings.scan_observation_degrees;self.scan_yaw_limit=settings.scan_yaw_limit
        self.observation_mode=settings.scan_observation_mode;self.observation_yaw=settings.scan_observation_yaw
        self.started=now;self.requested_duration=settings.edge_search_seconds
        self.angle=settings.search_yaw_degrees
        self.first_scan_direction=self.estimated_direction or -1
        self._set_span()
        # Include outward travel and return, reversal and braking. Duration is
        # a budget, never a claim that the measured arc has been completed.
        distance=self.angle*(1.5 if self.centered else 2.)
        planned=1.+distance/(24.*self.scan_yaw_limit)
        if self.angle>=120:
            distance=self.angle*(1.5 if self.centered else 1.)
            # Include measured settling and new detector samples at each view.
            planned=max(planned,1.+distance/(24.*self.scan_yaw_limit)+math.ceil(distance/self.sector_degrees)*.65)
        self.duration=min(search_duration_limit(self.angle),max(settings.edge_search_seconds,planned)) if settings.search_auto_duration else settings.edge_search_seconds
        self.until=now+self.duration;self.loss_source=self.source_time
        self.last_heading=heading['yaw_deg'];self.heading_time=heading['time'];self.heading_source=heading.get('source')
        self.offset=(self.last_heading-(self.last_heading if anchor is None else anchor)+180)%360-180

    def start_test(self,now,settings,heading,scenario,*,boost=False):
        if scenario not in ('LEFT','RIGHT','CENTER'):raise ValueError('Unknown search scenario')
        if boost and scenario=='CENTER':raise ValueError('BOOST test requires LEFT or RIGHT')
        if not fresh_heading(heading,now):raise ValueError('Fresh aircraft heading required')
        if not settings.edge_search_enabled or settings.search_yaw_degrees<=0:raise ValueError('Enable search and set a nonzero angle')
        self.reset();self.source_time=now;self.eligible=True
        # This explicit yaw diagnostic has no video input. It tests the same
        # route and physical settling, with timed looks instead of YOLO votes.
        self.sector_require_frames=False
        self.centered=scenario=='CENTER';self.estimated_direction=1 if scenario=='RIGHT' else -1
        self.direction_source='debug_scenario';self.begin(now,settings,heading)
        if boost:
            # Inject the edge-loss trigger only. Subsequent transitions and
            # command gates are the unchanged production BOOST / SCAN logic.
            self.direction=self.first_scan_direction
            self.phase='BOOST';self.reason='edge_catchup'
        else:self._scan(now)

    def _boost_boundary_close(self,yaw_rate):
        closing=0. if yaw_rate is None else max(0.,self.direction*yaw_rate)
        # Reserve a minimum approach distance during startup, then use measured
        # closing speed. Both phases use the same response/deceleration model.
        self.boost_brake_margin=braking_distance(max(NOMINAL_YAW_RATE*.6,closing))
        # A larger search expands SCAN, not the initial full-power catch-up.
        boundary=self.direction*min(SECTOR_DEGREES if self.angle>=120 else MAX_BOOST_ANGLE,
                                   self.high if self.direction>0 else -self.low)
        return self.direction*(boundary-self.offset)<=self.boost_brake_margin

    def update(self,now,missing,stale,spacing_phase,close,settings,heading=None,
               candidate_pending=False,bridge_reliable=False,control_available=True):
        active=self.phase in ('BOOST','SCAN')
        valid_heading=fresh_heading(heading,now)
        yaw_rate=self.rate.update(heading,now)
        cue=self.candidate_cue
        raw_ready=bool(cue and cue['hits']>=2 and cue['time']-cue['since']>=.06
            and 0<=now-cue['time']<=.25)
        # A short yaw-only look can precede association. It uses a prior real
        # target, never changes its measurement, and is single-use until another
        # accepted target observation. Do not revive a revoked search contract.
        idle_cue=bool(not active and not self.cue_recovery_used and self.eligible
            and self.phase in ('IDLE','DONE') and self.reason not in (
                'control_unavailable','video_or_result_unavailable','spacing_stopped',
                'search_disabled','heading_unavailable','heading_source_changed','heading_discontinuity')
            and self.source_time is not None and 0<=now-self.source_time<=12.
            and missing and raw_ready and valid_heading and control_available and not stale and not close
            and spacing_phase not in ('STOPPED','SEQUENCE_DONE')
            and settings.edge_search_enabled and settings.search_yaw_degrees>0)
        if idle_cue:
            # A cue during a recent loss overlays the normal recovery budget.
            # A short look AFTER a consumed/old search cannot renew that budget.
            recent_loss=(self.phase=='IDLE' and not self.consumed
                         and now-self.source_time<=settings.edge_search_seconds)
            if recent_loss:
                # The old motion estimate may precede a target reversal. This
                # is a bounded scan direction, never a new BOOST/identity proof.
                advance=max(-.20*cue['frame_width'],min(.20*cue['frame_width'],
                    cue['image_vx']*min(.15,max(0.,now-cue['time']))))
                error=cue['error_x']+advance/(cue['frame_width']*.5)
                if abs(error)>=.06:
                    self.estimated_direction=1 if error>0 else -1
                    self.direction_source='two_raw_neo_frames'
            self.begin(now,settings,heading)
            self.candidate_only=not recent_loss;self.cue_recovery_used=True;self.consumed=False
            self.cue_scope='recovery_overlay' if recent_loss else 'bounded_look'
            if recent_loss:self._scan(now)
            else:
                self.angle=min(30.,settings.search_yaw_degrees);self.low=-self.angle/2;self.high=self.angle/2
                self.duration=2.;self.until=now+2.;self.target=0.
            self.phase='SCAN';self.reason='raw_neo_candidate_guidance';self.offset=0.
            self.verifying=False;self.heading_missing_since=None;self.cue_started=now
            self.yaw_profile.reset(now);active=True
        if not control_available:
            if active:self.end('control_unavailable')
            else:self.reason='control_unavailable'
        elif stale or spacing_phase in ('STOPPED','SEQUENCE_DONE'):
            if active:self.end('video_or_result_unavailable' if stale else 'spacing_stopped')
            else:self.reason='video_or_result_unavailable' if stale else 'spacing_stopped'
        elif not settings.edge_search_enabled or settings.search_yaw_degrees<=0:
            if active:self.end('search_disabled')
            else:self.reason='search_disabled'
        elif active:
            if now>=self.until:self.end('search_timeout')
            elif self.heading_missing_since is not None and now-self.heading_missing_since>=HEADING_PAUSE_SECONDS:
                self.end('heading_unavailable')
            elif not valid_heading:
                if self.heading_missing_since is None:self.heading_missing_since=now
                self.reason='waiting_heading'
            elif heading.get('source')!=self.heading_source:self.end('heading_source_changed')
            else:
                if self.heading_missing_since is not None:
                    self.last_heading_pause_seconds=now-self.heading_missing_since
                    self.heading_resumes+=1
                self.heading_missing_since=None
                if heading['time']>self.heading_time:
                    delta=(heading['yaw_deg']-self.last_heading+180)%360-180
                    if abs(delta)>90:self.end('heading_discontinuity')
                    else:
                        self.offset+=delta;self.last_heading=heading['yaw_deg'];self.heading_time=heading['time']
                if self.phase in ('BOOST','SCAN'):
                    if self.verifying and now-self.verify_last>VERIFY_MAX_AGE_SECONDS:
                        self.verifying=False;self.verify_hits=0
                    if self.phase=='BOOST':
                        boundary_close=self._boost_boundary_close(yaw_rate)
                        if now-self.started>=BOOST_MAX_SECONDS:
                            self.boost_exit_reason='boost_timeout';self._scan(now)
                            # The fast catch-up is over. Start the slower scan
                            # from zero instead of carrying full BOOST demand.
                            self.yaw_profile.reset(now)
                        elif boundary_close:
                            self.boost_exit_reason='boundary_braking'
                            self._scan(now)
                    if self.phase=='SCAN' and not self.sectors and not self.verifying and now>=self.reversal_until:
                        toward_low=self.target==self.low
                        reached=(self.offset<=self.target+BOUNDARY_TOLERANCE if toward_low else self.offset>=self.target-BOUNDARY_TOLERANCE)
                        if reached:
                            self.boundaries.add(-1 if toward_low else 1)
                            if len(self.boundaries)==2:self.end('scan_complete')
                            else:
                                self.target=self.high if toward_low else self.low;self.reversal_until=now+REVERSAL_PAUSE
                                self.yaw_profile.reset(now)
                    if not self.verifying and self.phase in ('BOOST','SCAN'):
                        self.reason='edge_catchup' if self.phase=='BOOST' else 'sweeping'
        elif not self.consumed:
            if not missing and valid_heading:
                self.loss_heading=heading['yaw_deg'];self.loss_heading_time=heading['time'];self.loss_heading_source=heading.get('source')
            age=None if self.source_time is None else now-self.source_time
            if self.eligible and missing and age is not None and age>=settings.search_grace_seconds and not candidate_pending and not bridge_reliable:
                if age>settings.edge_search_seconds:self.end('search_timeout')
                elif not valid_heading:self.reason='waiting_heading'
                else:
                    anchor=self.loss_heading if (self.loss_heading is not None and self.loss_heading_source==heading.get('source')
                        and self.loss_heading_time is not None and abs(self.source_time-self.loss_heading_time)<=.25) else heading['yaw_deg']
                    self.begin(now,settings,heading,anchor)
                    evidence=self.edge.evidence
                    at_edge=self.box_edge and evidence.get('outward',False)
                    self.direction=self.edge.direction if at_edge and age<=.80 and self.edge.direction==self.first_scan_direction else 0
                    boundary_close=self._boost_boundary_close(yaw_rate) if self.direction else False
                    if self.direction and not boundary_close:
                        self.phase='BOOST';self.reason='edge_catchup'
                    else:
                        if boundary_close:self.boost_exit_reason='boundary_braking'
                        self._scan(now)
        cue=self.candidate_cue
        cue_ready=bool(raw_ready and now>=self.cue_cooldown_until
            and self.phase in ('BOOST','SCAN') and not self.verifying and valid_heading
            and control_available and not stale and not close)
        if self.cue_started is not None and now-self.cue_started>=.80:
            self.cue_started=None;self.cue_cooldown_until=now+.40
            self.candidate_cue=None;cue_ready=False
        # Accepted evidence temporarily suspends raw guidance while it is
        # verified. Suspension is not expiry and must not consume the search.
        if self.candidate_only and self.phase=='SCAN' and not cue_ready and not self.verifying:
            self.end('candidate_cue_expired')
        if cue_ready and self.cue_started is None:self.cue_started=now
        guided=cue_ready and self.cue_started is not None
        hint=self.direction_hint
        remembered=bool(not guided and not self.candidate_only and hint
            and 0<=now-hint['time']<=.65 and self.phase in ('BOOST','SCAN')
            and not self.verifying and valid_heading and control_available and not stale and not close)
        single=bool(cue and cue['hits']==1 and 0<=now-cue['time']<=.25 and not remembered)
        paused=self.phase in ('BOOST','SCAN') and (self.verifying or (candidate_pending or single) and not (guided or remembered) or not valid_heading)
        active=self.phase in ('BOOST','SCAN') and not paused
        if candidate_pending and self.phase in ('BOOST','SCAN'):self.reason='candidate_verifying'
        scan_yaw=0.
        motion=dict(control_reason='inactive',desired_rate_deg_s=None)
        if single and paused:self.reason='raw_neo_candidate_brake'
        if self.sectors and (paused or guided or remembered):self.sectors.suspend()
        if active and (guided or remembered):
            if self.phase=='BOOST':self._scan(now)
            self.reason='remembered_neo_direction' if remembered else 'raw_neo_candidate_guidance'
            self.direction_source='two_raw_neo_frames'
            # Compensate bounded processing age, not a future target trajectory.
            # Two-frame image velocity is only a yaw hint and cannot renew proof.
            steering=hint if remembered else cue
            width=steering['frame_width'];age=min(.15,max(0.,now-steering['time']))
            advance=max(-.20*width,min(.20*width,steering['image_vx']*age))
            error=max(-1.,min(1.,steering['error_x']+advance/(width*.5)))
            demand=max(-.20,min(.20,error*.45))
            if abs(error)<.06:demand=0.
            if demand:self._retarget(1 if demand>0 else -1,now)
            # Brake before reversing, retain the original measured arc/deadline.
            if yaw_rate is not None and demand*yaw_rate<0 and abs(yaw_rate)>15:demand=0.
            if self.offset>=self.high-BOUNDARY_TOLERANCE and demand>0 or self.offset<=self.low+BOUNDARY_TOLERANCE and demand<0:demand=0.
            scan_yaw=demand;self.yaw_profile.reset(now)
            motion=dict(control_reason=self.reason,desired_rate_deg_s=None,
                candidate_source_age_ms=(now-steering['time'])*1000,candidate_advance_px=advance,
                candidate_error_x=error,reversal_braking=bool(demand==0 and abs(error)>=.06))
        elif active and self.phase=='SCAN' and self.sectors:
            scan_yaw=self.sectors.update(self.offset,yaw_rate,now,self.yaw_profile)
            self.target=self.sectors.target
            for viewed in self.sectors.completed:
                if abs(viewed['target']-self.low)<.01:self.boundaries.add(-1)
                if abs(viewed['target']-self.high)<.01:self.boundaries.add(1)
            motion=dict(self.yaw_profile.diagnostics,control_reason='sector_'+self.sectors.phase)
            self.reason='sector_'+self.sectors.phase
            if self.sectors.done:
                # A retargeted partial route is not proof of full-circle coverage.
                if not self.route_revision:self.boundaries.update((-1,1))
                self.end('redirected_route_complete' if self.route_revision else 'scan_complete');active=False
        elif active and self.phase=='SCAN' and now>=self.reversal_until:
            distance=self.target-self.offset
            if abs(distance)>BOUNDARY_TOLERANCE:
                scan_yaw=self.yaw_profile.update(distance,yaw_rate,now,max_yaw=self.scan_yaw_limit)
                motion=dict(self.yaw_profile.diagnostics)
        elif active and self.phase=='BOOST':
            self.yaw_profile.reset(now,float(self.direction))
            motion=dict(control_reason='boost',desired_rate_deg_s=self.direction*NOMINAL_YAW_RATE)
        else:
            self.yaw_profile.reset(now)
            motion['control_reason']='reversal_pause' if active and self.phase=='SCAN' else self.reason
        scan_yaw=max(-self.scan_yaw_limit,min(self.scan_yaw_limit,scan_yaw))
        direction=(self.direction if self.phase=='BOOST' else (1 if scan_yaw>0 else -1 if scan_yaw<0 else 0)) if active else 0
        return dict(active=active,paused=paused,phase=self.phase,verifying=self.verifying,verify_hits=self.verify_hits,
            candidate_only=self.candidate_only,
            candidate_scope=self.cue_scope,
            candidate_guidance=guided,candidate_cue=dict(cue) if cue else None,
            direction_memory=remembered,direction_hint=dict(hint) if hint else None,route_revision=self.route_revision,
            candidate_guidance_until=None if self.cue_started is None else self.cue_started+.80,
            direction=direction,candidate_direction=self.edge.direction,started=self.started,until=self.until,
            first_scan_direction=self.first_scan_direction or self.estimated_direction,
            search_direction_source=self.direction_source,boost_exit_reason=self.boost_exit_reason,
            boost_max_seconds=BOOST_MAX_SECONDS,
            boost_until=self.started+BOOST_MAX_SECONDS if self.started is not None else None,
            boost_brake_margin_degrees=self.boost_brake_margin,
            source_time=self.loss_source if self.started is not None else self.source_time,measurement_id=self.measurement_id,
            reason=self.reason,consumed=self.consumed,evidence=dict(self.edge.evidence),
            angle_degrees=self.angle if self.started is not None else settings.search_yaw_degrees,
            boost_angle_degrees=min(SECTOR_DEGREES if self.angle>=120 else MAX_BOOST_ANGLE,
                                    self.high if self.direction>0 else -self.low),angle_progress_degrees=max(0.,self.direction*self.offset),
            span_mode=self.span_mode,scan_min_degrees=self.low,scan_max_degrees=self.high,
            yaw_full_scale=not self.candidate_only,
            scan_yaw_limit=self.scan_yaw_limit,observation_interval_degrees=self.sector_degrees,
            observation_mode=self.observation_mode,observation_yaw=self.observation_yaw,
            scan_offset_degrees=self.offset,scan_target_degrees=self.target,scan_yaw=scan_yaw,
            heading_time=self.heading_time,heading_source=self.heading_source,
            heading_pause_since=self.heading_missing_since,heading_pause_limit_seconds=HEADING_PAUSE_SECONDS,
            heading_resumes=self.heading_resumes,last_heading_pause_seconds=self.last_heading_pause_seconds,
            heading_rate_deg_s=yaw_rate,search_motion=motion,
            sectors=self.sectors.snapshot() if self.sectors else dict(enabled=False),
            requested_duration_seconds=self.requested_duration or settings.edge_search_seconds,
            coverage_complete=len(self.boundaries)==2,boundaries_reached=sorted(self.boundaries),
            auto_duration=settings.search_auto_duration,
            duration_seconds=self.duration if self.started is not None else settings.edge_search_seconds)


def scan_command(decision,now,started):
    search=decision.get('edge_search') or {}
    if search.get('phase')!='SCAN' or search.get('active') is not True:return 0.
    names=('started','until','source_time','heading_time','duration_seconds','angle_degrees',
           'scan_offset_degrees','scan_target_degrees','scan_yaw')
    if not all(isinstance(search.get(k),(int,float)) and math.isfinite(search[k]) for k in names):return 0.
    stamp=decision.get('prediction_time')
    if not isinstance(stamp,(int,float)) or not math.isfinite(stamp):return 0.
    if search.get('candidate_guidance') or search.get('direction_memory'):
        remembered=search.get('direction_memory')
        cue=search.get('direction_hint' if remembered else 'candidate_cue') or {}
        expires=cue.get('time',0)+.65 if remembered else search.get('candidate_guidance_until')
        values=[cue.get(k) for k in ('time','since','hits')]+[expires]
        if (not all(isinstance(v,(int,float)) and math.isfinite(v) for v in values)
                or not started<cue['since']<=cue['time']<=now or now-cue['time']>(.65 if remembered else .25)
                or cue['hits']<2 or cue['time']-cue['since']<.06 or now>=expires
                or abs(search.get('scan_yaw',0.))>.20):return 0.
    elif search.get('candidate_only'):return 0.
    low=search.get('scan_min_degrees',-search['angle_degrees']/2)
    high=search.get('scan_max_degrees',search['angle_degrees']/2)
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (low,high)):return 0.
    if (not 0<=now-stamp<=.20 or not started<=search['started']<=stamp<=now<search['until']
        or not search['source_time']<=search['started']
        or not 0<=now-search['heading_time']<=.25
        or not 2<=search['duration_seconds']<=search_duration_limit(search['angle_degrees']) or abs(search['until']-search['started']-search['duration_seconds'])>1e-6
        or not 0<search['angle_degrees']<=360 or not -360<=low<=0<=high<=360
        or abs(high-low-search['angle_degrees'])>1e-6 or not low<=search['scan_target_degrees']<=high
        or abs(search['scan_yaw'])>(1. if search.get('yaw_full_scale') is True else .25)
        or search.get('paused') or search.get('verifying') or decision.get('candidate_pending') and not (search.get('candidate_guidance') or search.get('direction_memory'))
        or decision.get('accepted') or decision.get('stale')
        or decision.get('state')!='DIRECTIONAL_SEARCH' or decision.get('spacing_phase') in ('STOPPED','SEQUENCE_DONE')):return 0.
    yaw=search['scan_yaw'];offset=search['scan_offset_degrees']
    if offset>=high-BOUNDARY_TOLERANCE and yaw>0 or offset<=low+BOUNDARY_TOLERANCE and yaw<0:return 0.
    limit=search.get('scan_yaw_limit',1.)
    if not isinstance(limit,(int,float)) or not math.isfinite(limit) or not .10<=limit<=1.:return 0.
    return max(-limit,min(limit,yaw))
