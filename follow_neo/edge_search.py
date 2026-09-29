"""Measured search with bounded reacquisition checks and persistent loss budget."""
from collections import deque
import math
import statistics
from .telemetry import fresh_heading,HEADING_MAX_AGE


class EdgeYawSearch:
    def __init__(self):self.reset()

    def reset(self):
        self.history=deque(maxlen=20);self.direction=0;self.hits=0
        self.source_time=None;self.measurement_id=None;self.armed_time=None
        self.started=None;self.until=None;self.reason='insufficient_motion_history'
        self.consumed=False;self.last_heading=None;self.heading_source=None
        self.progress=0.;self.angle_limit=0.;self.heading_time=None;self.duration=0.
        self.verifying=False;self.verify_hits=0;self.verify_since=None;self.verify_last=None
        self.evidence={}

    def end(self,reason):
        self.started=None;self.until=None;self.consumed=True;self.reason=reason
        self.verifying=False

    def observe(self,accepted,missing,confirmed,box,k,width,height,source_time,measurement_id,settings,strong=True):
        if not accepted or box is None:return
        if self.source_time is not None and source_time<=self.source_time:return
        if self.started is not None:
            # One associated measurement stops full-speed yaw immediately, but
            # does not discard the loss direction, deadline or accumulated turn.
            if not self.verifying or source_time-(self.verify_last or source_time)>.20:
                self.verify_hits=0;self.verify_since=source_time
            self.verifying=True;self.verify_hits+=int(strong);self.verify_last=source_time
            self.reason='reacquire_verifying';self.source_time=source_time;self.measurement_id=measurement_id
            if confirmed and self.verify_hits>=3 and source_time-self.verify_since>=.06:
                self.end('reacquired');self.history.clear();self.direction=0
            return
        if self.consumed:self.reset()
        self.source_time=source_time;self.measurement_id=measurement_id
        self.history.append((source_time,box.cx/width,box.cy/height,box.width/width))
        while self.history and source_time-self.history[0][0]>.65:self.history.popleft()
        self.hits=len(self.history)
        if not confirmed or len(self.history)<3:
            self.reason='insufficient_motion_history';return
        first,last=self.history[0],self.history[-1];span=last[0]-first[0]
        if span<.08:self.reason='short_motion_history';return
        pairs=list(self.history)
        # Median of nonadjacent slopes resists frame-to-frame BBOX jitter.
        slopes=[(b[1]-a[1])/(b[0]-a[0]) for i,a in enumerate(pairs) for b in pairs[i+1:] if b[0]-a[0]>=.06]
        vx=statistics.median(slopes);travel=last[1]-first[1];side=1 if vx>0 else -1
        steps=[b[1]-a[1] for a,b in zip(pairs,pairs[1:])]
        agree=sum(v*side>=-.003 for v in steps)/len(steps)
        edge=box.x2/width if side>0 else 1-box.x1/width
        outward=last[1]>.54 if side>0 else last[1]<.46
        exit_seconds=max(0.,(1-edge)/max(abs(vx),.001))
        supported=(abs(vx)>=.035 and abs(travel)>=.012 and travel*side>0 and agree>=.65
                   and outward and (exit_seconds<=2.5 or edge>=.88))
        self.evidence=dict(image_vx=vx,kalman_vx=k.vx/width,span_seconds=span,
            direction_agreement=agree,estimated_exit_seconds=exit_seconds,
            center_x=last[1],net_travel=travel,outward=outward,
            blocked_by=[] if supported else [name for name,ok in (
                ('speed',abs(vx)>=.035),('travel',abs(travel)>=.012 and travel*side>0),
                ('consistency',agree>=.65),('side',outward),('exit_horizon',exit_seconds<=2.5 or edge>=.88)) if not ok])
        if supported:
            self.direction=side;self.armed_time=source_time;self.reason='outward_motion_history'
        elif self.direction and self.armed_time is not None and source_time-self.armed_time<=.45 and not (abs(travel)>=.025 and side!=self.direction):
            self.reason='retained_exit_direction'
        else:self.direction=0;self.reason='no_supported_horizontal_exit'

    def update(self,now,missing,stale,spacing_phase,close,settings,heading=None):
        if stale:
            if self.started is not None:self.end('video_or_result_unavailable')
            else:self.reason='video_or_result_unavailable'
        elif not settings.edge_search_enabled or settings.search_yaw_degrees<=0:
            if self.started is not None:self.end('search_disabled')
            else:self.reason='search_disabled'
        elif self.started is not None:
            if now>=self.until:self.end('search_timeout')
            elif not fresh_heading(heading,now):self.end('heading_unavailable')
            elif heading.get('source')!=self.heading_source:self.end('heading_source_changed')
            else:
                if heading['time']>self.heading_time:
                    delta=(heading['yaw_deg']-self.last_heading+180)%360-180
                    if abs(delta)>90:self.end('heading_discontinuity')
                    else:
                        self.progress+=self.direction*delta
                        self.last_heading=heading['yaw_deg'];self.heading_time=heading['time']
                        if self.progress>=self.angle_limit:self.end('search_angle_reached')
                if self.started is not None and self.verifying and now-self.verify_last>.26:
                    self.verifying=False;self.verify_hits=0;self.reason='search_resumed_after_unconfirmed_detection'
                elif self.started is not None and not self.verifying:
                    self.reason='directional_search' if missing else 'search_ignoring_rejected_detection'
        elif not self.consumed and self.direction and self.source_time is not None:
            age=now-self.source_time
            if age>settings.edge_search_seconds:self.end('search_timeout')
            elif age>=.12:
                if not fresh_heading(heading,now):self.reason='waiting_heading'
                else:
                    self.started=now;self.duration=settings.edge_search_seconds
                    self.until=now+self.duration;self.angle_limit=settings.search_yaw_degrees;self.progress=0.
                    self.last_heading=heading['yaw_deg'];self.heading_time=heading['time']
                    self.heading_source=heading.get('source');self.reason='directional_search'
                    self.loss_source_time=self.source_time
        active=self.started is not None and not self.verifying
        return dict(active=active,verifying=self.verifying,verify_hits=self.verify_hits,
            direction=self.direction if active else 0,candidate_direction=self.direction,
            started=self.started,until=self.until,
            source_time=getattr(self,'loss_source_time',self.source_time) if self.started is not None else self.source_time,
            measurement_id=self.measurement_id,reason=self.reason,consumed=self.consumed,
            evidence=dict(self.evidence),angle_degrees=self.angle_limit if self.started is not None or self.consumed else settings.search_yaw_degrees,
            angle_progress_degrees=max(0.,self.progress),heading_time=self.heading_time,heading_source=self.heading_source,
            duration_seconds=self.duration if self.started is not None else settings.edge_search_seconds)


def yaw_override(decision,now,dance_started=None):
    search=decision.get('edge_search') or {}
    if search.get('phase') not in (None,'BOOST'):return 0.
    if search.get('phase')=='BOOST':
        limit=search.get('boost_angle_degrees')
        if not isinstance(limit,(int,float)) or not math.isfinite(limit) or not 0<limit<=90:return 0.
        start=search.get('started')
        if not isinstance(start,(int,float)) or not 0<=now-start<.35:return 0.
        margin=search.get('boost_brake_margin_degrees',0.)
        progress=search.get('angle_progress_degrees')
        if (not isinstance(margin,(int,float)) or not math.isfinite(margin) or margin<0
            or not isinstance(progress,(int,float)) or not math.isfinite(progress)
            or limit-progress<=margin):return 0.
    if search.get('active') is not True:return 0.
    names=('started','until','source_time','heading_time','angle_degrees','angle_progress_degrees','duration_seconds')
    if not all(isinstance(search.get(k),(int,float)) and math.isfinite(search[k]) for k in names):return 0.
    start,end,source=search['started'],search['until'],search['source_time']
    stamp=decision.get('prediction_time');direction=search.get('direction')
    if (not isinstance(stamp,(int,float)) or not math.isfinite(stamp)
        or direction not in (-1,1) or not 0<=now-stamp<=.20
        or not source<=start<=stamp<=now<end or not 2<=search['duration_seconds']<=10
        or abs(end-start-search['duration_seconds'])>1e-6
        or not 0<=start-source<=search['duration_seconds']
        or not 0<search['angle_degrees']<=180
        or not 0<=search['angle_progress_degrees']<search.get('boost_angle_degrees',search['angle_degrees'])
        or not 0<=now-search['heading_time']<=HEADING_MAX_AGE
        or dance_started is not None and start<dance_started
        or decision.get('accepted') or decision.get('stale')
        or decision.get('state')!='DIRECTIONAL_SEARCH'
        or decision.get('spacing_phase') in ('STOPPED','SEQUENCE_DONE')):return 0.
    return float(direction)
