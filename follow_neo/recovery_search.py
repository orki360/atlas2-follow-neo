"""Separate a brief edge catch-up from a measured left/right loss scan.

The last measured image position selects a directional or centred span.
Both phases share one heading anchor and deadline. Search yaw may reach full
scale with rate-aware braking; no translation is requested.
"""
import math
from .edge_search import EdgeYawSearch
from .telemetry import fresh_heading
from .centering import HeadingRate

BOOST_SECONDS = .35
SEARCH_GRACE_SECONDS = .45
BOUNDARY_TOLERANCE = .5
REVERSAL_PAUSE = .15
# Allow for remaining BOOST time and delayed aircraft response. This is a
# conservative planning margin, not a calibrated physical stopping distance.
BOOST_BRAKE_LOOKAHEAD = .45
BOOST_BRAKE_BASE_DEGREES = 3.


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
        self.rate=HeadingRate();self.requested_duration=0.;self.boundaries=set()
        self.box_edge=False
        self.estimated_direction=0;self.direction_source='none'
        self.first_scan_direction=0;self.boost_exit_reason=None
        self.boost_brake_margin=BOOST_BRAKE_BASE_DEGREES
        self.centered=True;self.low=self.high=0.;self.span_mode='centered'
        self.heading_missing_since=None

    def end(self,reason):
        self.phase='DONE';self.reason=reason;self.consumed=True;self.verifying=False

    def observe(self,accepted,missing,confirmed,box,k,width,height,source_time,measurement_id,settings,strong=True):
        if not accepted or box is None or (self.source_time is not None and source_time<=self.source_time):return
        if self.phase in ('BOOST','SCAN'):
            if not self.verifying or source_time-(self.verify_last or source_time)>.20:
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
        if (abs(vx)>=.035 and abs(travel)>=.012 and vx*travel>0
            and evidence.get('direction_agreement',0.)>=.65):
            self.estimated_direction=1 if vx>0 else -1;self.direction_source='measured_motion'
        elif abs(box.cx/width-.5)>.05:
            self.estimated_direction=1 if box.cx>width/2 else -1;self.direction_source='last_seen_side'
        else:
            self.estimated_direction=-1;self.direction_source='no_direction_left_fallback'

    def _scan(self,now):
        self.phase='SCAN';self.reason='sweeping'
        # Freeze the loss estimate for this search; BOOST must not silently
        # change the first scan direction. A reached boundary still reverses.
        side=self.first_scan_direction or -1
        self.target=self.high if side>0 else self.low
        if side*(self.offset-self.target)>=-BOUNDARY_TOLERANCE:
            self.boundaries.add(side);self.target=self.low if side>0 else self.high
        self.reversal_until=now+REVERSAL_PAUSE

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
        self.duration=min(10.,max(settings.edge_search_seconds,planned)) if settings.search_auto_duration else settings.edge_search_seconds
        self.until=now+self.duration;self.loss_source=self.source_time
        self.last_heading=heading['yaw_deg'];self.heading_time=heading['time'];self.heading_source=heading.get('source')
        self.offset=(self.last_heading-(self.last_heading if anchor is None else anchor)+180)%360-180

    def start_test(self,now,settings,heading,scenario):
        if scenario not in ('LEFT','RIGHT','CENTER'):raise ValueError('Unknown search scenario')
        if not fresh_heading(heading,now):raise ValueError('Fresh aircraft heading required')
        if not settings.edge_search_enabled or settings.search_yaw_degrees<=0:raise ValueError('Enable search and set a nonzero angle')
        self.reset();self.source_time=now;self.eligible=True
        self.centered=scenario=='CENTER';self.estimated_direction=1 if scenario=='RIGHT' else -1
        self.direction_source='debug_scenario';self.begin(now,settings,heading);self._scan(now)

    def _boost_boundary_close(self,yaw_rate):
        closing=0. if yaw_rate is None else max(0.,self.direction*yaw_rate)
        self.boost_brake_margin=BOOST_BRAKE_BASE_DEGREES+closing*BOOST_BRAKE_LOOKAHEAD
        boundary=self.high if self.direction>0 else self.low
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
                        if now-self.started>=BOOST_SECONDS or boundary_close:
                            self.boost_exit_reason='boundary_braking' if boundary_close else 'boost_time_elapsed'
                            self._scan(now)
                    if self.phase=='SCAN' and not self.verifying and now>=self.reversal_until:
                        toward_low=self.target==self.low
                        reached=(self.offset<=self.target+BOUNDARY_TOLERANCE if toward_low else self.offset>=self.target-BOUNDARY_TOLERANCE)
                        if reached:
                            self.boundaries.add(-1 if toward_low else 1)
                            self.target=self.high if toward_low else self.low;self.reversal_until=now+REVERSAL_PAUSE
                    if not self.verifying:self.reason='edge_catchup' if self.phase=='BOOST' else 'sweeping'
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
        paused=self.phase in ('BOOST','SCAN') and (self.verifying or candidate_pending or not valid_heading)
        active=self.phase in ('BOOST','SCAN') and not paused
        if candidate_pending and self.phase in ('BOOST','SCAN'):self.reason='candidate_verifying'
        scan_yaw=0.
        if active and self.phase=='SCAN' and now>=self.reversal_until:
            distance=self.target-self.offset
            if abs(distance)>BOUNDARY_TOLERANCE:
                # Dampen approach to the boundary; no minimum turning floor.
                closing_rate=0. if yaw_rate is None else math.copysign(1.,distance)*yaw_rate
                remaining=max(0.,abs(distance)-max(0.,closing_rate)*.45)
                scan_yaw=math.copysign(min(1.,remaining*.025),distance)
        direction=(self.direction if self.phase=='BOOST' else (1 if scan_yaw>0 else -1 if scan_yaw<0 else 0)) if active else 0
        return dict(active=active,paused=paused,phase=self.phase,verifying=self.verifying,verify_hits=self.verify_hits,
            direction=direction,candidate_direction=self.edge.direction,started=self.started,until=self.until,
            first_scan_direction=self.first_scan_direction or self.estimated_direction,
            search_direction_source=self.direction_source,boost_exit_reason=self.boost_exit_reason,
            boost_brake_margin_degrees=self.boost_brake_margin,
            source_time=self.loss_source if self.started is not None else self.source_time,measurement_id=self.measurement_id,
            reason=self.reason,consumed=self.consumed,evidence=dict(self.edge.evidence),
            angle_degrees=self.angle if self.started is not None else settings.search_yaw_degrees,
            boost_angle_degrees=(self.high if self.direction>0 else -self.low),angle_progress_degrees=max(0.,self.direction*self.offset),
            span_mode=self.span_mode,scan_min_degrees=self.low,scan_max_degrees=self.high,
            yaw_full_scale=True,
            scan_offset_degrees=self.offset,scan_target_degrees=self.target,scan_yaw=scan_yaw,
            heading_time=self.heading_time,heading_source=self.heading_source,
            heading_rate_deg_s=yaw_rate,requested_duration_seconds=self.requested_duration or settings.edge_search_seconds,
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
        or not 2<=search['duration_seconds']<=10 or abs(search['until']-search['started']-search['duration_seconds'])>1e-6
        or not 0<search['angle_degrees']<=180 or not -180<=low<=0<=high<=180
        or abs(high-low-search['angle_degrees'])>1e-6 or not low<=search['scan_target_degrees']<=high
        or abs(search['scan_yaw'])>(1. if search.get('yaw_full_scale') is True else .25)
        or search.get('paused') or search.get('verifying') or decision.get('candidate_pending')
        or decision.get('accepted') or decision.get('stale')
        or decision.get('state')!='DIRECTIONAL_SEARCH' or decision.get('spacing_phase') in ('STOPPED','SEQUENCE_DONE')):return 0.
    yaw=search['scan_yaw'];offset=search['scan_offset_degrees']
    if offset>=high-BOUNDARY_TOLERANCE and yaw>0 or offset<=low+BOUNDARY_TOLERANCE and yaw<0:return 0.
    return yaw
