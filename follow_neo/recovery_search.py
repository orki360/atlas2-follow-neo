"""Separate a brief edge catch-up from a measured left/right loss scan.

The configured angle is the TOTAL scan span. Its centre is the last available
fresh heading at target loss. Both phases share one deadline, and only the
BOOST phase may bypass the normal yaw cap. No translation is requested.
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
                self.end('reacquired')
            return
        if self.consumed:self.reset()
        self.source_time=source_time;self.measurement_id=measurement_id
        self.eligible=confirmed
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
        self.target=side*self.angle/2
        if side*self.offset>=self.angle/2-BOUNDARY_TOLERANCE:
            self.boundaries.add(side);self.target=-self.target
        self.reversal_until=now+REVERSAL_PAUSE

    def _boost_boundary_close(self,yaw_rate):
        closing=0. if yaw_rate is None else max(0.,self.direction*yaw_rate)
        self.boost_brake_margin=BOOST_BRAKE_BASE_DEGREES+closing*BOOST_BRAKE_LOOKAHEAD
        return self.angle/2-self.direction*self.offset<=self.boost_brake_margin

    def update(self,now,missing,stale,spacing_phase,close,settings,heading=None,
               candidate_pending=False,bridge_reliable=False):
        active=self.phase in ('BOOST','SCAN')
        valid_heading=fresh_heading(heading,now)
        yaw_rate=self.rate.update(heading,now)
        if stale or spacing_phase in ('STOPPED','SEQUENCE_DONE'):
            if active:self.end('video_or_result_unavailable' if stale else 'spacing_stopped')
            else:self.reason='video_or_result_unavailable' if stale else 'spacing_stopped'
        elif not settings.edge_search_enabled or settings.search_yaw_degrees<=0:
            if active:self.end('search_disabled')
            else:self.reason='search_disabled'
        elif active:
            if now>=self.until:self.end('search_timeout')
            elif not valid_heading:self.end('heading_unavailable')
            elif heading.get('source')!=self.heading_source:self.end('heading_source_changed')
            else:
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
                        reached=(self.offset<=self.target+BOUNDARY_TOLERANCE if self.target<0 else self.offset>=self.target-BOUNDARY_TOLERANCE)
                        if reached:
                            self.boundaries.add(-1 if self.target<0 else 1)
                            self.target=-self.target;self.reversal_until=now+REVERSAL_PAUSE
                    if not self.verifying:self.reason='edge_catchup' if self.phase=='BOOST' else 'sweeping'
        elif not self.consumed:
            if not missing and valid_heading:
                self.loss_heading=heading['yaw_deg'];self.loss_heading_time=heading['time'];self.loss_heading_source=heading.get('source')
            age=None if self.source_time is None else now-self.source_time
            if self.eligible and missing and age is not None and age>=settings.search_grace_seconds and not candidate_pending and not bridge_reliable:
                if age>settings.edge_search_seconds:self.end('search_timeout')
                elif not valid_heading:self.reason='waiting_heading'
                else:
                    self.started=now;self.requested_duration=settings.edge_search_seconds
                    # Reserve time for 1.5 spans (centre -> first side -> opposite),
                    # including braking/reversal. The hard cap is always 10s.
                    planned=.6+1.5*settings.search_yaw_degrees/12.
                    self.duration=max(settings.edge_search_seconds,min(10.,planned)) if settings.search_auto_duration else settings.edge_search_seconds
                    self.until=now+self.duration
                    self.angle=settings.search_yaw_degrees;self.loss_source=self.source_time
                    self.last_heading=heading['yaw_deg'];self.heading_time=heading['time'];self.heading_source=heading.get('source')
                    anchor=self.loss_heading if (self.loss_heading is not None and self.loss_heading_source==self.heading_source
                        and self.loss_heading_time is not None and abs(self.source_time-self.loss_heading_time)<=.25) else self.last_heading
                    self.offset=(self.last_heading-anchor+180)%360-180
                    self.first_scan_direction=self.estimated_direction or -1
                    evidence=self.edge.evidence
                    at_edge=self.box_edge and evidence.get('outward',False)
                    self.direction=self.edge.direction if at_edge and age<=.80 and self.edge.direction==self.first_scan_direction else 0
                    boundary_close=self._boost_boundary_close(yaw_rate) if self.direction else False
                    if self.direction and abs(self.offset)<self.angle/2-BOUNDARY_TOLERANCE and not boundary_close:
                        self.phase='BOOST';self.reason='edge_catchup'
                    else:
                        if boundary_close:self.boost_exit_reason='boundary_braking'
                        self._scan(now)
        active=self.phase in ('BOOST','SCAN') and not self.verifying and not candidate_pending
        if candidate_pending:self.reason='candidate_verifying'
        scan_yaw=0.
        if active and self.phase=='SCAN' and now>=self.reversal_until:
            distance=self.target-self.offset
            if abs(distance)>BOUNDARY_TOLERANCE:
                # Dampen approach to the boundary; no minimum turning floor.
                closing_rate=0. if yaw_rate is None else math.copysign(1.,distance)*yaw_rate
                remaining=max(0.,abs(distance)-max(0.,closing_rate)*.18)
                scan_yaw=math.copysign(min(.25,remaining*.025),distance)
        direction=(self.direction if self.phase=='BOOST' else (1 if scan_yaw>0 else -1 if scan_yaw<0 else 0)) if active else 0
        return dict(active=active,phase=self.phase,verifying=self.verifying,verify_hits=self.verify_hits,
            direction=direction,candidate_direction=self.edge.direction,started=self.started,until=self.until,
            first_scan_direction=self.first_scan_direction or self.estimated_direction,
            search_direction_source=self.direction_source,boost_exit_reason=self.boost_exit_reason,
            boost_brake_margin_degrees=self.boost_brake_margin,
            source_time=self.loss_source if self.started is not None else self.source_time,measurement_id=self.measurement_id,
            reason=self.reason,consumed=self.consumed,evidence=dict(self.edge.evidence),
            angle_degrees=self.angle if self.started is not None else settings.search_yaw_degrees,
            boost_angle_degrees=self.angle/2,angle_progress_degrees=max(0.,self.direction*self.offset),
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
    if (not 0<=now-stamp<=.20 or not started<=search['started']<=stamp<=now<search['until']
        or not search['source_time']<=search['started']
        or not 0<=now-search['heading_time']<=.25
        or not 2<=search['duration_seconds']<=10 or abs(search['until']-search['started']-search['duration_seconds'])>1e-6
        or not 0<search['angle_degrees']<=180 or abs(search['scan_target_degrees'])>search['angle_degrees']/2
        or abs(search['scan_yaw'])>.25 or decision.get('accepted') or decision.get('stale')
        or decision.get('state')!='DIRECTIONAL_SEARCH' or decision.get('spacing_phase') in ('STOPPED','SEQUENCE_DONE')):return 0.
    yaw=search['scan_yaw'];offset=search['scan_offset_degrees'];half=search['angle_degrees']/2
    if offset>=half-BOUNDARY_TOLERANCE and yaw>0 or offset<=-half+BOUNDARY_TOLERANCE and yaw<0:return 0.
    return yaw
