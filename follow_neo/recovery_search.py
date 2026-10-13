"""Separate a brief edge catch-up from a measured left/right loss scan.

The last measured image position selects a directional or centred span.
Both phases share one heading anchor and deadline. SCAN yaw uses a configurable cap (default 50%)
with rate-aware braking; BOOST retains its separate limit. No translation is requested.
"""
import math
from .edge_search import EdgeYawSearch
from .telemetry import fresh_heading
from .search_motion import (SearchHeadingRate,SearchYawProfile,braking_distance,
                            BOOST_MAX_SECONDS,NOMINAL_YAW_RATE,MAX_BOOST_ANGLE,SCAN_MAX_YAW,search_duration_limit)

SEARCH_GRACE_SECONDS = .45
BOUNDARY_TOLERANCE = .5
REVERSAL_PAUSE = .15


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
        self.box_edge=False;self.boost_blocked_by=[];self.last_box=None
        self.absent_frames=0;self.observation_id=None;self.observation_time=None;self.candidate_visible=False
        self.estimated_direction=0;self.direction_source='none'
        self.first_scan_direction=0;self.boost_exit_reason=None
        self.boost_brake_margin=braking_distance(NOMINAL_YAW_RATE*.6)
        self.centered=True;self.low=self.high=0.;self.span_mode='centered'
        self.heading_missing_since=None

    def observe_result(self,source_time,frame_id,candidate_visible):
        # Only distinct processed frames count; control ticks cannot fabricate loss.
        if self.observation_id is not None and (frame_id<=self.observation_id or source_time<=self.observation_time):return
        self.observation_id=frame_id;self.observation_time=source_time
        self.candidate_visible=bool(candidate_visible)
        self.absent_frames=0 if candidate_visible else min(2,self.absent_frames+1)

    def end(self,reason):
        self.phase='DONE';self.reason=reason;self.consumed=True;self.verifying=False

    def observe(self,accepted,missing,confirmed,box,k,width,height,source_time,measurement_id,settings,strong=True,verified_recovery=False):
        if not accepted or box is None or (self.source_time is not None and source_time<=self.source_time):return
        if self.phase in ('BOOST','SCAN'):
            if not self.verifying or source_time-(self.verify_last or source_time)>.20:
                self.verify_hits=0;self.verify_since=source_time
            self.verifying=True;self.verify_hits+=int(strong);self.verify_last=source_time
            self.source_time=source_time;self.measurement_id=measurement_id
            self.reason='reacquire_verifying'
            if confirmed and (verified_recovery or self.verify_hits>=2 and source_time>self.verify_since):
                # Seed the next loss from this measurement immediately. A DONE
                # flag must not shadow COAST if the very next frame is missing.
                self.reset()
            else:return
        if self.consumed:self.reset()
        self.source_time=source_time;self.measurement_id=measurement_id
        self.eligible=confirmed
        self.centered=.25<=box.cx/width<=.75
        self.edge.observe(accepted,missing,confirmed,box,k,width,height,source_time,measurement_id,settings,strong)
        self.last_box=(box.x1/width,box.x2/width)
        self.box_edge=(self.last_box[0]<=settings.boost_edge_zone if self.edge.direction<0 else
                       self.last_box[1]>=1-settings.boost_edge_zone if self.edge.direction>0 else False)
        evidence=self.edge.evidence
        vx=evidence.get('image_vx',0.);travel=evidence.get('net_travel',0.)
        if (abs(vx)>=.035 and abs(travel)>=.012 and vx*travel>0
            and evidence.get('direction_agreement',0.)>=.65):
            self.estimated_direction=1 if vx>0 else -1;self.direction_source='measured_motion'
        elif abs(box.cx/width-.5)>.05:
            self.estimated_direction=1 if box.cx>width/2 else -1;self.direction_source='last_seen_side'
        else:
            self.estimated_direction=-1;self.direction_source='no_direction_left_fallback'

    def _scan(self,now,limit=SCAN_MAX_YAW):
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
        self.reversal_until=now if continues_boost else now+REVERSAL_PAUSE
        self.yaw_profile.reset(now,previous_direction*limit if continues_boost else 0.)

    def _set_span(self):
        self.span_mode='centered' if self.centered else 'directional'
        if self.centered:self.low,self.high=-self.angle/2,self.angle/2
        elif self.first_scan_direction>0:self.low,self.high=0.,self.angle
        else:self.low,self.high=-self.angle,0.

    def begin(self,now,settings,heading,anchor=None):
        """Shared scan planner used by target recovery and the explicit test."""
        self.started=now;self.requested_duration=settings.edge_search_seconds
        self.angle=settings.search_yaw_degrees
        self.first_scan_direction=self.estimated_direction or -1
        self._set_span()
        # Include outward travel and return, reversal and braking. Duration is
        # a budget, never a claim that the measured arc has been completed.
        distance=self.angle*(1.5 if self.centered else 2.)
        planned=1.+distance/24.
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
        self.centered=scenario=='CENTER';self.estimated_direction=1 if scenario=='RIGHT' else -1
        self.direction_source='debug_scenario';self.begin(now,settings,heading)
        if boost:
            # Inject the edge-loss trigger only. Subsequent transitions and
            # command gates are the unchanged production BOOST / SCAN logic.
            self.direction=self.first_scan_direction
            self.phase='BOOST';self.reason='edge_catchup'
        else:self._scan(now,settings.scan_yaw_limit)

    def _boost_boundary_close(self,yaw_rate):
        closing=0. if yaw_rate is None else max(0.,self.direction*yaw_rate)
        # Reserve a minimum approach distance during startup, then use measured
        # closing speed. Both phases use the same response/deceleration model.
        self.boost_brake_margin=braking_distance(max(NOMINAL_YAW_RATE*.6,closing))
        # A larger search expands SCAN, not the initial full-power catch-up.
        boundary=self.direction*min(MAX_BOOST_ANGLE,self.high if self.direction>0 else -self.low)
        return self.direction*(boundary-self.offset)<=self.boost_brake_margin

    def update(self,now,missing,stale,spacing_phase,close,settings,heading=None,
               candidate_pending=False,bridge_reliable=False,control_available=True):
        active=self.phase in ('BOOST','SCAN')
        valid_heading=fresh_heading(heading,now)
        yaw_rate=self.rate.update(heading,now)
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
            elif not valid_heading:
                if self.heading_missing_since is None:self.heading_missing_since=now
                self.reason='waiting_heading'
                if now-self.heading_missing_since>=.35:self.end('heading_unavailable')
            elif heading.get('source')!=self.heading_source:self.end('heading_source_changed')
            else:
                self.heading_missing_since=None
                if heading['time']>self.heading_time:
                    delta=(heading['yaw_deg']-self.last_heading+180)%360-180
                    if abs(delta)>90:self.end('heading_discontinuity')
                    else:
                        self.offset+=delta;self.last_heading=heading['yaw_deg'];self.heading_time=heading['time']
                if self.phase in ('BOOST','SCAN'):
                    if self.verifying and now-self.verify_last>.26:
                        self.verifying=False;self.verify_hits=0
                    if self.phase=='BOOST':
                        boundary_close=self._boost_boundary_close(yaw_rate)
                        if now-self.started>=BOOST_MAX_SECONDS:
                            self.boost_exit_reason='boost_timeout';self.end('boost_timeout')
                        elif boundary_close:
                            self.boost_exit_reason='boundary_braking'
                            self._scan(now,settings.scan_yaw_limit)
                    if self.phase=='SCAN' and not self.verifying and now>=self.reversal_until:
                        toward_low=self.target==self.low
                        reached=(self.offset<=self.target+BOUNDARY_TOLERANCE if toward_low else self.offset>=self.target-BOUNDARY_TOLERANCE)
                        if reached:
                            self.boundaries.add(-1 if toward_low else 1)
                            self.target=self.high if toward_low else self.low;self.reversal_until=now+REVERSAL_PAUSE
                            self.yaw_profile.reset(now)
                    if not self.verifying and self.phase in ('BOOST','SCAN'):
                        self.reason='edge_catchup' if self.phase=='BOOST' else 'sweeping'
        elif not self.consumed:
            if not missing and valid_heading:
                self.loss_heading=heading['yaw_deg'];self.loss_heading_time=heading['time'];self.loss_heading_source=heading.get('source')
            age=None if self.source_time is None else now-self.source_time
            evidence=self.edge.evidence
            if self.last_box is not None:
                self.box_edge=(self.last_box[0]<=settings.boost_edge_zone if self.edge.direction<0 else
                               self.last_box[1]>=1-settings.boost_edge_zone if self.edge.direction>0 else False)
            direction_ok=bool(self.edge.direction and self.edge.direction==(self.estimated_direction or -1))
            observed_loss=(self.absent_frames>=2 and self.observation_time is not None
                           and 0<=now-self.observation_time<=.25)
            visible=(self.candidate_visible and self.observation_time is not None
                     and 0<=now-self.observation_time<=.25)
            boost_ready=(age is not None and settings.boost_entry_delay<=age<=.80 and observed_loss
                         and self.box_edge and evidence.get('outward',False) and direction_ok)
            normal_ready=age is not None and age>=settings.search_grace_seconds and not bridge_reliable
            if self.eligible and missing and not candidate_pending and not visible and (boost_ready or normal_ready):
                if age>settings.edge_search_seconds:self.end('search_timeout')
                elif not valid_heading:self.reason='waiting_heading'
                else:
                    anchor=self.loss_heading if (self.loss_heading is not None and self.loss_heading_source==heading.get('source')
                        and self.loss_heading_time is not None and abs(self.source_time-self.loss_heading_time)<=.25) else heading['yaw_deg']
                    self.begin(now,settings,heading,anchor)
                    self.direction=self.edge.direction if boost_ready else 0
                    boundary_close=self._boost_boundary_close(yaw_rate) if self.direction else False
                    self.boost_blocked_by=[name for name,ok in (
                        ('last_accepted_box_not_at_edge',self.box_edge),
                        ('no_supported_exit_direction',bool(self.edge.direction)),
                        ('not_outward',evidence.get('outward',False)),
                        ('entry_delay',age>=settings.boost_entry_delay),
                        ('two_missing_frames_required',observed_loss),
                        ('loss_older_than_800ms',age<=.80),
                        ('exit_direction_differs_from_scan',direction_ok),
                        ('boundary_braking',not boundary_close)) if not ok]
                    if self.direction and not boundary_close:
                        self.phase='BOOST';self.reason='edge_catchup'
                    else:
                        if boundary_close:self.boost_exit_reason='boundary_braking'
                        self._scan(now,settings.scan_yaw_limit)
        visible=(self.candidate_visible and self.observation_time is not None and 0<=now-self.observation_time<=.25)
        paused=self.phase in ('BOOST','SCAN') and (self.verifying or candidate_pending or visible or not valid_heading)
        active=self.phase in ('BOOST','SCAN') and not paused
        if (candidate_pending or visible) and self.phase in ('BOOST','SCAN'):self.reason='candidate_verifying'
        scan_yaw=0.
        motion=dict(control_reason='inactive',desired_rate_deg_s=None)
        if active and self.phase=='SCAN' and now>=self.reversal_until:
            distance=self.target-self.offset
            if abs(distance)>BOUNDARY_TOLERANCE:
                scan_yaw=self.yaw_profile.update(distance,yaw_rate,now,settings.scan_yaw_limit)
                motion=dict(self.yaw_profile.diagnostics)
        elif active and self.phase=='BOOST':
            self.yaw_profile.reset(now,float(self.direction))
            motion=dict(control_reason='boost',desired_rate_deg_s=self.direction*NOMINAL_YAW_RATE)
        else:
            self.yaw_profile.reset(now)
            motion['control_reason']='reversal_pause' if active and self.phase=='SCAN' else self.reason
        direction=(self.direction if self.phase=='BOOST' else (1 if scan_yaw>0 else -1 if scan_yaw<0 else 0)) if active else 0
        return dict(active=active,paused=paused,phase=self.phase,verifying=self.verifying,verify_hits=self.verify_hits,
            direction=direction,candidate_direction=self.edge.direction,started=self.started,until=self.until,
            first_scan_direction=self.first_scan_direction or self.estimated_direction,
            search_direction_source=self.direction_source,boost_exit_reason=self.boost_exit_reason,
            boost_max_seconds=BOOST_MAX_SECONDS,boost_blocked_by=list(self.boost_blocked_by),
            boost_edge_zone=settings.boost_edge_zone,boost_entry_delay=settings.boost_entry_delay,
            absent_frames=self.absent_frames,candidate_visible=visible,
            boost_until=self.started+BOOST_MAX_SECONDS if self.started is not None else None,
            boost_brake_margin_degrees=self.boost_brake_margin,
            source_time=self.loss_source if self.started is not None else self.source_time,measurement_id=self.measurement_id,
            reason=self.reason,consumed=self.consumed,evidence=dict(self.edge.evidence),
            angle_degrees=self.angle if self.started is not None else settings.search_yaw_degrees,
            boost_angle_degrees=min(MAX_BOOST_ANGLE,self.high if self.direction>0 else -self.low),angle_progress_degrees=max(0.,self.direction*self.offset),
            span_mode=self.span_mode,scan_min_degrees=self.low,scan_max_degrees=self.high,
            yaw_full_scale=True,  # Retained transport flag: SCAN bypasses the normal yaw slider.
            scan_yaw_limit=settings.scan_yaw_limit,
            scan_offset_degrees=self.offset,scan_target_degrees=self.target,scan_yaw=scan_yaw,
            heading_time=self.heading_time,heading_source=self.heading_source,
            heading_rate_deg_s=yaw_rate,search_motion=motion,
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
        or search.get('paused') or search.get('verifying') or decision.get('candidate_pending')
        or decision.get('accepted') or decision.get('stale')
        or decision.get('state')!='DIRECTIONAL_SEARCH' or decision.get('spacing_phase') in ('STOPPED','SEQUENCE_DONE')):return 0.
    limit=search.get('scan_yaw_limit',SCAN_MAX_YAW)
    if not isinstance(limit,(int,float)) or not math.isfinite(limit) or not 0<=limit<=1:return 0.
    yaw=search['scan_yaw'];offset=search['scan_offset_degrees']
    if offset>=high-BOUNDARY_TOLERANCE and yaw>0 or offset<=low+BOUNDARY_TOLERANCE and yaw<0:return 0.
    return max(-limit,min(limit,yaw))
